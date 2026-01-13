#!/bin/bash

# Seg-Zero++ 训练脚本
# 基于Seg-Zero-7B checkpoint，添加负点预测

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="segzero-plus"

# 基础配置
BASE_MODEL="pretrained_models/Seg-Zero-7B"
OUTPUT_DIR="outputs/segzero_plus_7b"
DATA_DIR="data/refcocog_9k_840"

# 训练参数
BATCH_SIZE=2
GRAD_ACCUM=8
NUM_SAMPLES=8  # GRPO采样数
LR=1e-6
KL_COEF=5e-3
NUM_STEPS=300

# 新增：负点奖励配置
USE_NEGATIVE_REWARD=true
NEGATIVE_REWARD_WEIGHT=1.0

python -m verl.trainer.main \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="segzero_plus_negative_points" \
    trainer.total_training_steps=$NUM_STEPS \
    \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.prompt_key="prompt" \
    data.image_key="image" \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.fsdp.torch_dtype=bf16 \
    \
    actor_rollout_ref.rollout.n=$NUM_SAMPLES \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    custom.use_negative_reward=$USE_NEGATIVE_REWARD \
    custom.negative_reward_weight=$NEGATIVE_REWARD_WEIGHT \
    custom.use_strict_format=true \
    custom.prompt_template="negative_point" \
    \
    trainer.save_freq=50 \
    trainer.save_path=$OUTPUT_DIR