#!/bin/bash
#===============================================================================
# Seg-Zero++ 训练脚本 - 针对 2×A6000 48GB 显存优化
# 
# 功能: 基于Seg-Zero-7B checkpoint，添加负点预测与对比奖励
# 硬件: 2× NVIDIA A6000 (48GB each, 96GB total)
# 预计训练时间: 约2-3天 (9K samples, 300 steps)
#===============================================================================

set -e  # 遇到错误立即退出

#-------------------------------------------------------------------------------
# 环境变量配置
#-------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# Weights & Biases 配置 (可选)
export WANDB_PROJECT="segzero-plus"
export WANDB_RUN_NAME="segzero_plus_neg_points_2xa6000"
# 如果不使用wandb，取消下面这行的注释
export WANDB_MODE=disabled




BASE_MODEL="/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/"

# SAM2模型路径
SAM_MODEL="/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt"

# 数据路径
DATA_DIR="/mnt/xiaoqian/dataset/refcocog/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/"
ALT_DATA_DIR="/mnt/xiaoqian/dataset/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/"
if [ ! -d "$DATA_DIR" ] && [ -d "$ALT_DATA_DIR" ]; then
    DATA_DIR="$ALT_DATA_DIR"
fi
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/val.parquet"

# 混淆区域数据 (如果预计算了)
CONFUSED_REGIONS_DIR="${DATA_DIR}/confused_regions"

# 输出路径
OUTPUT_DIR="outputs/segzero_plus_7b_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# 日志文件
LOG_FILE="${OUTPUT_DIR}/training.log"

#-------------------------------------------------------------------------------
# 显存优化参数 (针对2×A6000优化)
#-------------------------------------------------------------------------------
# Micro batch size per GPU (越小越省显存)
MICRO_BATCH=1

# 梯度累积步数 (有效batch size = MICRO_BATCH × GRAD_ACCUM × NUM_GPUS)
# 有效batch size = 1 × 16 × 2 = 32
GRAD_ACCUM=16

# GRPO采样数量 (原版用8-16，这里减少到4以节省显存)
NUM_SAMPLES=4

# GPU显存利用率 (0.85-0.95之间，太高可能OOM)
GPU_MEM_UTIL=0.88

# Tensor并行大小 (使用两张卡做tensor parallel)
TENSOR_PARALLEL=2

#-------------------------------------------------------------------------------
# 训练超参数
#-------------------------------------------------------------------------------
# 总训练步数
NUM_STEPS=300

# 学习率 (GRPO通常用较小的学习率)
LEARNING_RATE=1e-6

# KL散度系数 (控制与原始模型的偏离程度)
KL_COEF=5e-3

# 权重衰减
WEIGHT_DECAY=0.01

# 温度参数 (采样多样性)
TEMPERATURE=1.0

# 最大生成长度
MAX_NEW_TOKENS=512

# 图像分辨率
IMAGE_SIZE=840

#-------------------------------------------------------------------------------
# 负点奖励配置 (核心创新)
#-------------------------------------------------------------------------------
# 是否启用负点奖励
USE_NEGATIVE_REWARD=true

# 负点奖励权重
NEGATIVE_REWARD_WEIGHT=1.0

# 是否使用混淆区域 (需要预计算)
USE_CONFUSED_REGIONS=true

# 惩罚系数 (负点落在GT mask内)
NEGATIVE_ALPHA=1.0

# 奖励系数 (负点落在混淆区域)
NEGATIVE_BETA=0.5

# 最大负点数量
MAX_NEGATIVE_POINTS=2

#-------------------------------------------------------------------------------
# 格式奖励配置
#-------------------------------------------------------------------------------
# 是否使用严格格式检查
USE_STRICT_FORMAT=true

# Prompt模板类型: "original" 或 "negative_point"
PROMPT_TEMPLATE="negative_point"

#-------------------------------------------------------------------------------
# 检查点和日志配置
#-------------------------------------------------------------------------------
# 保存检查点频率 (每N步保存一次)
SAVE_FREQ=50

# 验证频率
VAL_FREQ=25

# 日志频率
LOG_FREQ=10

#-------------------------------------------------------------------------------
# 验证必要文件存在
#-------------------------------------------------------------------------------
echo "=========================================="
echo "Seg-Zero++ Training Script"
echo "=========================================="
echo ""
echo "Checking required files..."

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL"
    echo "Please download Seg-Zero-7B first:"
    echo "  cd pretrained_models && git clone https://huggingface.co/Ricky06662/Seg-Zero-7B"
    exit 1
fi

if [ ! -f "$SAM_MODEL" ]; then
    echo "WARNING: SAM2 model not found at $SAM_MODEL"
    echo "Reward computation may fail without SAM2 model"
fi

if [ ! -f "$TRAIN_DATA" ]; then
    if [ -f "$DATA_DIR/dataset_info.json" ] || compgen -G "$DATA_DIR"/*-train-*.arrow > /dev/null; then
        TRAIN_DATA="$DATA_DIR"
    else
        echo "ERROR: Training data not found at $TRAIN_DATA"
        echo "Please run: python training_scripts/download_dataset.py"
        exit 1
    fi
fi

if [ ! -f "$VAL_DATA" ]; then
    if [ -f "$DATA_DIR/dataset_info.json" ] || compgen -G "$DATA_DIR"/*-validation-*.arrow > /dev/null; then
        VAL_DATA="$DATA_DIR"
    else
        VAL_DATA=""
    fi
fi

echo "All required files found!"
echo ""

#-------------------------------------------------------------------------------
# 显示配置信息
#-------------------------------------------------------------------------------
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "Hardware:"
echo "  GPUs: 2× A6000 (48GB each)"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo ""
echo "Training:"
echo "  Base Model: $BASE_MODEL"
echo "  Total Steps: $NUM_STEPS"
echo "  Effective Batch Size: $((MICRO_BATCH * GRAD_ACCUM * 2))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  GRPO Samples: $NUM_SAMPLES"
echo ""
echo "Negative Point Reward:"
echo "  Enabled: $USE_NEGATIVE_REWARD"
echo "  Weight: $NEGATIVE_REWARD_WEIGHT"
echo "  Max Negative Points: $MAX_NEGATIVE_POINTS"
echo ""
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 等待用户确认
read -p "Press Enter to start training, or Ctrl+C to cancel..."
echo ""

#-------------------------------------------------------------------------------
# 开始训练
#-------------------------------------------------------------------------------
echo "Starting training at $(date)"
echo "Logs will be saved to $LOG_FILE"
echo ""

python -m verl.trainer.main \
    \
    `# ===== 项目配置 =====` \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.total_training_steps=${NUM_STEPS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.log_freq=${LOG_FREQ} \
    trainer.save_path="${OUTPUT_DIR}" \
    \
    `# ===== 数据配置 =====` \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.prompt_key="prompt" \
    data.image_key="image" \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_NEW_TOKENS} \
    \
    `# ===== Actor模型配置 =====` \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    `# ===== Actor优化器配置 =====` \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.strategy="adamw_bf16" \
    \
    `# ===== Actor PPO配置 =====` \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.grad_accum_steps=${GRAD_ACCUM} \
    \
    `# ===== KL散度配置 =====` \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
    \
    `# ===== FSDP配置 =====` \
    actor_rollout_ref.actor.fsdp.torch_dtype="bf16" \
    actor_rollout_ref.actor.fsdp.wrap_policy="qwen2_vl" \
    \
    `# ===== Rollout采样配置 =====` \
    actor_rollout_ref.rollout.n=${NUM_SAMPLES} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_new_tokens=${MAX_NEW_TOKENS} \
    actor_rollout_ref.rollout.tensor_parallel_size=${TENSOR_PARALLEL} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    \
    `# ===== Reference模型配置 =====` \
    actor_rollout_ref.ref.fsdp.torch_dtype="bf16" \
    \
    `# ===== GRPO算法配置 =====` \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.adv_estimator="grpo" \
    \
    `# ===== 自定义配置 (负点奖励) =====` \
    custom.use_negative_reward=${USE_NEGATIVE_REWARD} \
    custom.negative_reward_weight=${NEGATIVE_REWARD_WEIGHT} \
    custom.use_confused_regions=${USE_CONFUSED_REGIONS} \
    custom.confused_regions_dir="${CONFUSED_REGIONS_DIR}" \
    custom.negative_alpha=${NEGATIVE_ALPHA} \
    custom.negative_beta=${NEGATIVE_BETA} \
    custom.max_negative_points=${MAX_NEGATIVE_POINTS} \
    custom.use_strict_format=${USE_STRICT_FORMAT} \
    custom.prompt_template="${PROMPT_TEMPLATE}" \
    custom.sam_model_path="${SAM_MODEL}" \
    custom.image_size=${IMAGE_SIZE} \
    \
    2>&1 | tee ${LOG_FILE}

#-------------------------------------------------------------------------------
# 训练完成
#-------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Merge checkpoint to HuggingFace format:"
echo "   python training_scripts/model_merger.py --local_dir ${OUTPUT_DIR}/checkpoint-${NUM_STEPS}"
echo ""
echo "2. Evaluate on ReasonSeg:"
echo "   bash evaluation_scripts/eval_reasonseg_segzero_plus.sh ${OUTPUT_DIR}/checkpoint-${NUM_STEPS}"
echo ""
echo "3. Evaluate on RefCOCO:"
echo "   bash evaluation_scripts/eval_refcoco_segzero_plus.sh ${OUTPUT_DIR}/checkpoint-${NUM_STEPS}"
