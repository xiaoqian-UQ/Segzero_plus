# src/train/grpo_seg_zero_negative.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any
import numpy as np

from src.utils.parser import SegZeroOutputParser
from src.utils.sam_utils import SAM2Wrapper
from src.train.reward_functions import NegativePointRewardCalculator

class GRPOTrainerWithNegativePoints:
    """支持负点预测的GRPO训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = config.get("device", "cuda")
        
        # 初始化模型
        self.model = self._init_model()
        self.tokenizer = self._init_tokenizer()
        
        # 初始化工具
        self.parser = SegZeroOutputParser(require_negative_points=True)
        self.sam_wrapper = SAM2Wrapper(
            model_cfg=config["sam_config"],
            checkpoint=config["sam_checkpoint"],
            device=self.device
        )
        self.reward_calculator = NegativePointRewardCalculator(
            sam_wrapper=self.sam_wrapper,
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            lambda_neg=config.get("lambda_neg", 0.3),
            lambda_format=config.get("lambda_format", 0.1)
        )
        
        # GRPO参数
        self.group_size = config.get("group_size", 8)
        self.clip_lower = config.get("clip_lower", -0.2)
        self.clip_upper = config.get("clip_upper", 0.28)
        self.temperature = config.get("temperature", 0.7)
        self.optimizer = None
        self.engine = None
        
    def _init_model(self):
        """初始化模型并添加LoRA"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA配置
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 64),
            lora_alpha=self.config.get("lora_alpha", 128),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _init_tokenizer(self):
        """初始化tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            trust_remote_code=True
        )
        return tokenizer
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 包含image, query, gt_mask的批次数据
            
        Returns:
            训练指标字典
        """
        images = batch["image"]
        queries = batch["query"]
        gt_masks = batch["gt_mask"]
        
        batch_size = len(images)
        all_rewards = []
        all_log_probs = []
        
        for i in range(batch_size):
            image = images[i]
            query = queries[i]
            gt_mask = gt_masks[i]
            
            # 构建prompt
            prompt = self._build_prompt(query)
            
            # 采样K个输出
            outputs, log_probs = self._sample_outputs(prompt, self.group_size)
            
            # 计算每个输出的奖励
            rewards = []
            for output in outputs:
                parsed = self.parser.parse(output)
                reward_output = self.reward_calculator.compute_reward(
                    image=image,
                    positive_points=parsed.positive_points,
                    negative_points=parsed.negative_points,
                    bbox=parsed.bbox,
                    gt_mask=gt_mask,
                    format_valid=parsed.is_valid
                )
                rewards.append(reward_output.total_reward)
            
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
        
        # GRPO更新
        loss = self._compute_grpo_loss(all_rewards, all_log_probs)
        
        # 反向传播
        if self.engine is not None:
            self.engine.backward(loss)
            self.engine.step()
        else:
            if self.optimizer is None:
                raise RuntimeError("Optimizer is not initialized.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_reward": np.mean([r for rewards in all_rewards for r in rewards]),
            "max_reward": np.max([r for rewards in all_rewards for r in rewards])
        }

    def set_engine(self, engine) -> None:
        """设置DeepSpeed引擎"""
        self.engine = engine
    
    def _build_prompt(self, query: str) -> str:
        """构建包含负点说明的prompt"""
        return (
            f"Please find '{query}' with bbox, points, and negative points."
            "Compare the difference between objects and find the most closely matched one."
            "Identify confusing background regions that should be excluded using negative points."
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "Output the one bbox, points of two largest inscribed circles inside the interested object, "
            "and negative points in confusing background regions, all in JSON format."
            "All coordinates should be normalized to [0, 1]."
            "i.e., <think> thinking process here </think>"
            '<answer>{"bbox": [x1, y1, x2, y2], "points": [[x1, y1], [x2, y2]], "negative_points": [[x1, y1]]}</answer>'
        )
    
    def _sample_outputs(self, prompt: str, k: int) -> tuple:
        """采样K个输出及其log概率"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = []
        log_probs = []
        
        model = self.engine.module if self.engine is not None else self.model
        for _ in range(k):
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=self.temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            output_text = self.tokenizer.decode(
                generated.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            outputs.append(output_text)
            
            # 计算log概率
            log_prob = self._compute_log_prob(generated)
            log_probs.append(log_prob)
        
        return outputs, log_probs
    
    def _compute_log_prob(self, generated) -> torch.Tensor:
        """计算生成序列的log概率"""
        scores = generated.scores
        sequences = generated.sequences
        
        log_prob = 0
        for i, score in enumerate(scores):
            token_id = sequences[0, i + 1]
            log_prob += torch.log_softmax(score, dim=-1)[0, token_id]
        
        return log_prob
    
    def _compute_grpo_loss(
        self,
        all_rewards: List[List[float]],
        all_log_probs: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        计算GRPO损失
        
        使用组内相对奖励进行优势估计
        """
        total_loss = 0
        count = 0
        
        for rewards, log_probs in zip(all_rewards, all_log_probs):
            # 计算组内平均奖励
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            
            # 计算优势（相对于组内平均）
            advantages = [(r - mean_reward) / std_reward for r in rewards]
            
            # 裁剪优势
            advantages = [
                np.clip(a, self.clip_lower, self.clip_upper)
                for a in advantages
            ]
            
            # 计算损失
            for log_prob, advantage in zip(log_probs, advantages):
                total_loss -= log_prob * advantage
                count += 1
        
        return total_loss / count


def main():
    """训练主函数"""
    import argparse
    import yaml
    import os
    import json
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    import torch.distributed as dist
    import deepspeed

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件")
    args = parser.parse_args()

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    use_distributed = args.local_rank != -1
    if use_distributed:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if use_distributed:
        config["device"] = f"cuda:{args.local_rank}"

    # 创建输出目录
    if not use_distributed or dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # 初始化trainer
    if not use_distributed or dist.get_rank() == 0:
        print("Initializing trainer...")
    trainer = GRPOTrainerWithNegativePoints(config)

    # 初始化优化器
    optimizer = AdamW(
        trainer.model.parameters(),
        lr=config.get("learning_rate", 1e-5)
    )
    trainer.optimizer = optimizer

    # 初始化学习率调度器
    num_training_steps = config.get("max_steps", 5000)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 100),
        num_training_steps=num_training_steps
    )

    # DeepSpeed初始化（可选）
    if args.deepspeed:
        if not os.path.exists(args.deepspeed):
            raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed}")
        with open(args.deepspeed, "r") as f:
            ds_config = json.load(f)
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=trainer.model,
            model_parameters=trainer.model.parameters(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config_params=ds_config
        )
        trainer.set_engine(model_engine)
        trainer.model = model_engine
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler

    # 加载数据
    if not use_distributed or dist.get_rank() == 0:
        print("Loading data...")
    from src.data.dataset import create_dataloader

    train_dataloader = create_dataloader(
        arrow_dir=config["train_data"]["arrow_dir"],
        mask_dir=config["train_data"]["mask_dir"],
        batch_size=config.get("batch_size", 2),
        image_size=config.get("image_size", 840),
        num_workers=4,
        shuffle=True,
        distributed=use_distributed,
        rank=dist.get_rank() if use_distributed else 0,
        world_size=dist.get_world_size() if use_distributed else 1
    )

    if not use_distributed or dist.get_rank() == 0:
        print(f"Training samples: {len(train_dataloader.dataset)}")

    # 训练循环
    if not use_distributed or dist.get_rank() == 0:
        print("Starting training...")
    global_step = 0
    trainer.model.train()

    # 创建进度条
    pbar = tqdm(total=num_training_steps, desc="Training") if not use_distributed or dist.get_rank() == 0 else None

    epoch = 0
    while global_step < num_training_steps:
        epoch += 1
        if not use_distributed or dist.get_rank() == 0:
            print(f"\n=== Epoch {epoch} ===")

        for batch_idx, batch in enumerate(train_dataloader):
            # 训练一步
            metrics = trainer.train_step(batch)

            # 更新学习率（DeepSpeed已接管时跳过）
            if trainer.engine is None:
                scheduler.step()

            global_step += 1
            if pbar:
                pbar.update(1)

            # 打印训练指标
            if global_step % 10 == 0 and pbar:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

            # 保存检查点
            if global_step % config.get("save_steps", 500) == 0 and (not use_distributed or dist.get_rank() == 0):
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                model_to_save = trainer.engine.module if trainer.engine is not None else trainer.model
                model_to_save.save_pretrained(save_path)
                print(f"\nSaved checkpoint to {save_path}")

            # 达到最大步数
            if global_step >= num_training_steps:
                break

    if pbar:
        pbar.close()

    # 保存最终模型
    if not use_distributed or dist.get_rank() == 0:
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        model_to_save = trainer.engine.module if trainer.engine is not None else trainer.model
        model_to_save.save_pretrained(final_save_path)
        print(f"\nTraining completed! Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
