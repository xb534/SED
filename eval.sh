#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

#ADE20k-150
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-ade150 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
 DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#ADE20k-847
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-ade847 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade847.json" \
 DATASETS.TEST \(\"ade20k_full_sem_seg_freq_val_all\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal VOC
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-voc20 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20.json" \
 DATASETS.TEST \(\"voc_2012_test_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal VOC-b
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-voc20b \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20b.json" \
 DATASETS.TEST \(\"voc_2012_test_background_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal Context 59
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-pc59 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
 DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal Context 459
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-pc459 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc459.json" \
 DATASETS.TEST \(\"context_459_test_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/eval-ade150/log.txt | grep copypaste
cat $output/eval-ade847/log.txt | grep copypaste
cat $output/eval-voc20/log.txt | grep copypaste
cat $output/eval-voc20b/log.txt | grep copypaste
cat $output/eval-pc59/log.txt | grep copypaste
cat $output/eval-pc459/log.txt | grep copypaste