set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export QWEN_DIR=/mnt/xiaoqian/model/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export REFCOCOG2K_DIR=/mnt/xiaoqian/model/hub/datasets--Ricky06662--refCOCOg_2k_840/snapshots/921ff4b780dc0805a63e41473a4cf44d9cd9e5cc/data
export SAM2_CKPT=/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt

MODEL_PATH=$QWEN_DIR
RUN_NAME=$(basename "$0" .sh)


RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/seg_zero_3b.yaml \
    data.train_files="${REFCOCOG2K_DIR}" \
    data.val_files=None \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.global_batch_size=2 \
    data.rollout_batch_size=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.75 \
    worker.rollout.n=4 \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.total_episodes=2000 \
    trainer.save_checkpoint_path=./workdir/${RUN_NAME}
