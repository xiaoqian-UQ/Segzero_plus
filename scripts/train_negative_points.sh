#!/bin/bash
# scripts/train_negative_points.sh

# 配置
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29500

# 路径
CONFIG="configs/negative_points_config.yaml"
OUTPUT_DIR="outputs/negative_points_exp1"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练
deepspeed --num_gpus=2 \
    src/train/grpo_seg_zero_negative.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --deepspeed configs/deepspeed_zero2.json \
    2>&1 | tee $OUTPUT_DIR/train.log