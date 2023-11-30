# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

@META_ARCH_REGISTRY.register()
class SED(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        in_features,
        fast_inference: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "conv":
                    params.requires_grad = True if "conv" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                elif clip_finetune == "mlp":
                    params.requires_grad = True if "mlp" in name or "position" in name else False
                elif clip_finetune == "full_res5":
                    if "stages.3" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        if clip_finetune == "fast_infer":
            for name, params in self.sem_seg_head.predictor.transformer.named_parameters():
                if "head1" in name or "head2" in name or "head0" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        finetune_backbone = backbone_multiplier > 0.
        for name, params in self.backbone.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

        self.sliding_window = sliding_window
        # self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.clip_resolution = (768, 768)
        self.sequential = False
        del self.backbone
        self.in_features = in_features
        self.fast_inference = fast_inference
        self.clip_finetune = clip_finetune

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "in_features": cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES,
            "fast_inference": cfg.TEST.FAST_INFERENCE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        self.sliding_window = False
        if not self.training:
            self.size_divisibility = -1
        if not self.training and self.sliding_window:
            if not self.sequential:
                with _ignore_torch_cuda_oom():
                    return self.inference_sliding_window(batched_inputs)
                self.sequential = True
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)

        images_resized = F.interpolate(images.tensor, size=(384, 384), mode='bilinear', align_corners=False,)
        # features = self.backbone(images_resized)
        clip_vis_dense = clip_features["clip_vis_dense"]
        fusion_features = {k: v.clone().detach() for k,v in clip_features.items() if k in self.in_features}
        outputs = self.sem_seg_head(clip_vis_dense, fusion_features)
        if self.training:
            print_flag = False
            for name, param in self.named_parameters():
                if param.grad == None and param.requires_grad:
                    print(name)
                    print_flag = True
            if print_flag:
                print("--------------------------------------------------------------------\n")
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            num_classes = outputs[0].shape[1]
            mask = targets != self.sem_seg_head.ignore_value
            losses = {}
            for i, output_ in enumerate(outputs):
                if self.clip_finetune == "fast_infer" and i==0:
                    continue
                output_ = F.interpolate(output_, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
                output_ = output_.permute(0,2,3,1)
                _targets = torch.zeros(output_.shape, device=self.device)
                _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
                _targets[mask] = _onehot
                loss = F.binary_cross_entropy_with_logits(output_, _targets)
                losses.update({f"loss_sem_seg_{i}" : loss})
            return losses
        else:
            if self.fast_inference:
                outputs = outputs[0]
            else:
                outputs = outputs[0].sigmoid()
            image_size = images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        if self.sequential:
            outputs = []
            for clip_feat, image in zip(clip_images, images):
                feature = self.backbone(image.unsqueeze(0))
                clip_feat = self.sem_seg_head.predictor.clip_model.encode_image(clip_feat.unsqueeze(0), dense=True)
                output = self.sem_seg_head(clip_feat, feature)
                outputs.append(output[0])
            outputs = torch.stack(outputs, dim=0)
        else:
            # features = self.backbone(images)
            features = {}
            clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
            outputs = self.sem_seg_head(clip_features["clip_vis_dense"], features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]