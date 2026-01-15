#!/bin/bash

REASONING_MODEL_PATH="/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/"
SEGMENTATION_MODEL_PATH="facebook/sam2.1-hiera-large"


OUTPUT_PATH="./reasonseg_eval_results"
TEST_DATA_PATH="/mnt/xiaoqian/dataset/Reasonseg/Ricky06662___reason_seg_val/default/0.0.0/f96ae9cafc5747620edff7c52812595582e6eb29/"
# TEST_DATA_PATH="/mnt/xiaoqian/dataset/Reasonseg/Ricky06662___reason_seg_test/default/0.0.0/51536947c13888c9790d9197c6fa30f5d57f3ab6/"
NUM_PARTS=1

# Create output directory
mkdir -p $OUTPUT_PATH

# Run 8 processes in parallel
for idx in 0; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 2 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH

