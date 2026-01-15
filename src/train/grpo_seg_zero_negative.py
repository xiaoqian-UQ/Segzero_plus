# src/train/grpo_seg_zero_negative.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
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
        
        self.is_vl_model = False
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
        model_path = self.config["model_path"]
        model_type = None
        use_deepspeed = self.config.get("use_deepspeed", False)
        device_map = None if use_deepspeed else "auto"
        low_cpu_mem_usage = True if use_deepspeed else False
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None)
        except Exception:
            model_type = None

        if model_type == "qwen2_5_vl":
            self.is_vl_model = True
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
        elif model_type == "qwen2_vl":
            self.is_vl_model = True
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
        
        # 启用gradient checkpointing节省内存
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

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
        if self.is_vl_model:
            return AutoProcessor.from_pretrained(
                self.config["model_path"],
                trust_remote_code=True
            )
        return AutoTokenizer.from_pretrained(
            self.config["model_path"],
            trust_remote_code=True
        )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        单步训练（GRPO算法）

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
        all_sequences = []  # 保存生成的token序列
        all_inputs = []     # 保存输入（用于重新计算log_probs）

        for i in range(batch_size):
            image = images[i]
            query = queries[i]
            gt_mask = gt_masks[i]

            # 构建prompt并准备输入
            prompt = self._build_prompt(query)
            inputs = self._prepare_inputs(image, prompt)

            # 阶段1：采样K个输出（no_grad）
            outputs_text, sequences = self._sample_outputs(inputs, self.group_size)

            # 阶段2：计算每个输出的奖励
            rewards = []
            for output_text in outputs_text:
                parsed = self.parser.parse(output_text)
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
            all_sequences.append(sequences)
            all_inputs.append(inputs)

        # 阶段3：重新计算log_probs（带梯度）并计算GRPO loss
        loss = self._compute_grpo_loss_with_recompute(all_inputs, all_sequences, all_rewards)

        # 阶段4：反向传播和参数更新
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

    def _prepare_inputs(self, image: np.ndarray, prompt: str) -> Dict:
        """
        准备模型输入（支持VL模型）

        Args:
            image: numpy array (H, W, 3)
            prompt: 文本提示

        Returns:
            模型输入字典
        """
        from PIL import Image as PILImage

        if self.is_vl_model:
            # Qwen2.5-VL: 需要图像+文本
            pil_image = PILImage.fromarray(image.astype(np.uint8))

            # 构建VL消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 使用processor处理
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            )
        else:
            # 纯文本模型
            inputs = self.tokenizer(prompt, return_tensors="pt")

        # 移到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs

    def _sample_outputs(self, inputs: Dict, k: int) -> tuple:
        """
        采样K个输出（no_grad，只用于生成文本）

        Args:
            inputs: 模型输入（包含图像和文本）
            k: 采样数量

        Returns:
            (outputs_text, sequences): 生成的文本和token序列
        """
        outputs_text = []
        sequences = []

        model = self.engine.module if self.engine is not None else self.model

        for _ in range(k):
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=self.temperature,
                    do_sample=True,
                    return_dict_in_generate=True
                )

            # 提取生成的token序列（去除输入部分）
            # generated.sequences 是 [1, total_len]，取第一个样本，只保留生成的部分
            seq = generated.sequences[0, inputs["input_ids"].shape[1]:]  # [seq_len]
            sequences.append(seq)

            # 解码为文本
            if hasattr(self.tokenizer, "batch_decode"):
                output_text = self.tokenizer.batch_decode(
                    seq, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            else:
                output_text = self.tokenizer.decode(
                    seq[0], skip_special_tokens=True
                )
            outputs_text.append(output_text)

        return outputs_text, sequences

    def _compute_sequence_log_probs(
        self,
        inputs: Dict,
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        重新计算序列的log概率（带梯度）

        Args:
            inputs: 模型输入（包含图像和文本）
            sequence: 生成的token序列 (1, seq_len)

        Returns:
            log_prob: 序列的总log概率（标量tensor，带梯度）
        """
        model = self.engine.module if self.engine is not None else self.model

        # 构建完整的输入序列（输入 + 生成的序列）
        full_input_ids = torch.cat([inputs["input_ids"], sequence], dim=1)

        # 准备其他输入（如果有）
        forward_inputs = {"input_ids": full_input_ids}
        if "attention_mask" in inputs:
            # 扩展attention_mask
            seq_len = sequence.shape[1]
            extended_mask = torch.ones(
                (inputs["attention_mask"].shape[0], seq_len),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device
            )
            forward_inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], extended_mask], dim=1
            )

        # 如果是VL模型，传入图像特征
        if self.is_vl_model and "pixel_values" in inputs:
            forward_inputs["pixel_values"] = inputs["pixel_values"]
        if self.is_vl_model and "image_grid_thw" in inputs:
            forward_inputs["image_grid_thw"] = inputs["image_grid_thw"]

        # 前向传播
        outputs = model(**forward_inputs)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # 计算log_probs（只计算生成部分）
        # logits的最后seq_len个位置对应sequence的预测
        input_len = inputs["input_ids"].shape[1]
        gen_logits = logits[:, input_len-1:-1, :]  # 对应sequence的每个token

        # 修复1: 应用temperature，与采样时保持一致
        log_probs = torch.log_softmax(gen_logits / self.temperature, dim=-1)  # (1, seq_len, vocab_size)

        # 提取对应token的log_prob
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=sequence.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len)

        # 修复2: Mask掉EOS后的PAD token
        # 获取tokenizer的特殊token ID（兼容VL模型的processor）
        if self.is_vl_model and hasattr(self.tokenizer, 'tokenizer'):
            # VL模型：processor.tokenizer
            pad_token_id = self.tokenizer.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.tokenizer.eos_token_id
        else:
            # 普通模型
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id

        # 创建mask：在EOS之前的token为1，EOS及之后为0
        mask = torch.ones_like(sequence, dtype=torch.float)
        if eos_token_id is not None:
            # 找到第一个EOS的位置
            eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)
            if len(eos_positions[0]) > 0:
                # 有EOS token
                for batch_idx in range(sequence.shape[0]):
                    batch_eos = eos_positions[1][eos_positions[0] == batch_idx]
                    if len(batch_eos) > 0:
                        first_eos = batch_eos[0].item()
                        # 将EOS及之后的位置mask掉（包括EOS本身）
                        mask[batch_idx, first_eos:] = 0

        # 如果没有EOS但有PAD，也mask掉PAD
        if pad_token_id is not None:
            mask = mask * (sequence != pad_token_id).float()

        # 应用mask后求和
        total_log_prob = (token_log_probs * mask).sum()

        return total_log_prob

    def _compute_sequence_log_probs_batch(
        self,
        inputs: Dict,
        sequences: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        批量计算多个序列的log_probs（一次前向传播）

        Args:
            inputs: 模型输入 (batch_size=1)
            sequences: K个生成的序列

        Returns:
            log_probs: [K] 的tensor，每个序列的总log_prob
        """
        # 获取pad_token_id和eos_token_id
        if self.is_vl_model and hasattr(self.tokenizer, 'tokenizer'):
            pad_token_id = self.tokenizer.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id

        # 1. Pad所有序列到相同长度
        max_len = max(seq.shape[0] for seq in sequences)
        padded_sequences = []
        length_masks = []

        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                padded_seq = torch.cat([
                    seq,
                    torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device)
                ])
                mask = torch.cat([
                    torch.ones(seq.shape[0], device=seq.device),
                    torch.zeros(pad_len, device=seq.device)
                ])
            else:
                padded_seq = seq
                mask = torch.ones(seq.shape[0], device=seq.device)

            padded_sequences.append(padded_seq)
            length_masks.append(mask)

        # Stack成批次: [K, max_len]
        batch_sequences = torch.stack(padded_sequences)
        batch_masks = torch.stack(length_masks)

        # 2. 扩展inputs到批次大小K
        K = len(sequences)
        batch_inputs = {}

        for key in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']:
            if key in inputs:
                # 从 [1, ...] 扩展到 [K, ...]
                batch_inputs[key] = inputs[key].repeat(K, *([1] * (len(inputs[key].shape) - 1)))

        # 3. 拼接input_ids和generated sequences
        input_len = batch_inputs['input_ids'].shape[1]
        combined_ids = torch.cat([batch_inputs['input_ids'], batch_sequences], dim=1)

        # 更新attention_mask
        if 'attention_mask' in batch_inputs:
            combined_attention_mask = torch.cat([
                batch_inputs['attention_mask'],
                batch_masks
            ], dim=1)
            batch_inputs['attention_mask'] = combined_attention_mask

        batch_inputs['input_ids'] = combined_ids

        # 4. 前向传播（一次计算所有序列）
        with torch.amp.autocast('cuda'):
            outputs = self.model(**batch_inputs)
            logits = outputs.logits  # [K, input_len + max_len, vocab_size]

        # 5. 取生成部分的logits
        gen_logits = logits[:, input_len-1:-1, :]  # [K, max_len, vocab_size]

        # 6. 计算log_probs（应用temperature）
        log_probs = torch.log_softmax(gen_logits / self.temperature, dim=-1)

        # 7. Gather每个token的log_prob
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=batch_sequences.unsqueeze(-1)
        ).squeeze(-1)  # [K, max_len]

        # 8. 应用EOS mask
        final_masks = batch_masks.clone()
        if eos_token_id is not None:
            for k in range(K):
                eos_positions = (batch_sequences[k] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    final_masks[k, first_eos:] = 0

        # 9. 计算每个序列的总log_prob
        sequence_log_probs = (token_log_probs * final_masks).sum(dim=1)  # [K]

        return sequence_log_probs

    def _compute_grpo_loss_with_recompute(
        self,
        all_inputs: List[Dict],
        all_sequences: List[List[torch.Tensor]],
        all_rewards: List[List[float]]
    ) -> torch.Tensor:
        """
        重新计算log_probs（带梯度）并计算GRPO损失

        Args:
            all_inputs: 每个样本的输入
            all_sequences: 每个样本的K个生成序列
            all_rewards: 每个样本的K个奖励

        Returns:
            GRPO损失
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for inputs, sequences, rewards in zip(all_inputs, all_sequences, all_rewards):
            # 计算组内优势
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8

            advantages = [(r - mean_reward) / std_reward for r in rewards]
            advantages = [np.clip(a, self.clip_lower, self.clip_upper) for a in advantages]
            advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)

            # 批量计算所有序列的log_probs（一次前向传播，避免梯度重复）
            log_probs = self._compute_sequence_log_probs_batch(inputs, sequences)  # [K]

            # GRPO损失：-log_probs * advantages 的和
            loss_term = -(log_probs * advantages_tensor).sum()
            total_loss = total_loss + loss_term
            count += len(sequences)

        return total_loss / count if count > 0 else total_loss


def main():
    """训练主函数"""
    import argparse
    import yaml
    import os
    import json
    import sys
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    import torch.distributed as dist
    import deepspeed

    print("=" * 80, flush=True)
    print("GRPO Negative Points Training - START", flush=True)
    print("=" * 80, flush=True)

    # 解析命令行参数
    print("\n[1/10] Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件")
    args = parser.parse_args()
    print(f"   Config: {args.config}", flush=True)
    print(f"   Output dir: {args.output_dir}", flush=True)
    print(f"   Local rank: {args.local_rank}", flush=True)
    print(f"   DeepSpeed: {args.deepspeed}", flush=True)

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"   Local rank from env: {args.local_rank}", flush=True)

    use_distributed = args.local_rank != -1
    print(f"   Distributed: {use_distributed}", flush=True)

    if use_distributed:
        print(f"\n[2/10] Initializing distributed training...", flush=True)
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        print(f"   Rank: {dist.get_rank()}, World size: {dist.get_world_size()}", flush=True)
    else:
        print(f"\n[2/10] Single-process training", flush=True)

    # 加载配置
    print(f"\n[3/10] Loading config from {args.config}...", flush=True)
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config loaded successfully", flush=True)
        print(f"   Model: {config.get('model_path', 'N/A')}", flush=True)
        print(f"   Batch size: {config.get('batch_size', 'N/A')}", flush=True)
        print(f"   Max steps: {config.get('max_steps', 'N/A')}", flush=True)
    except Exception as e:
        print(f"   ✗ Failed to load config: {e}", flush=True)
        sys.exit(1)

    config["use_deepspeed"] = bool(args.deepspeed)

    if use_distributed:
        config["device"] = f"cuda:{args.local_rank}"

    # 创建输出目录
    print(f"\n[4/10] Creating output directory...", flush=True)
    if not use_distributed or dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"   ✓ Output dir: {args.output_dir}", flush=True)

    # 初始化trainer
    print(f"\n[5/10] Initializing trainer (this may take a few minutes)...", flush=True)
    try:
        trainer = GRPOTrainerWithNegativePoints(config)
        print(f"   ✓ Trainer initialized", flush=True)
    except Exception as e:
        print(f"   ✗ Trainer initialization failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 初始化优化器
    print(f"\n[6/10] Initializing optimizer and scheduler...", flush=True)
    try:
        optimizer = AdamW(
            trainer.model.parameters(),
            lr=config.get("learning_rate", 1e-5)
        )
        trainer.optimizer = optimizer
        print(f"   ✓ Optimizer: AdamW, LR={config.get('learning_rate', 1e-5)}", flush=True)

        # 初始化学习率调度器
        num_training_steps = config.get("max_steps", 5000)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )
        print(f"   ✓ Scheduler: warmup={config.get('warmup_steps', 100)}, total={num_training_steps}", flush=True)
    except Exception as e:
        print(f"   ✗ Optimizer/scheduler initialization failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # DeepSpeed初始化（可选）
    print(f"\n[7/10] DeepSpeed initialization...", flush=True)
    if args.deepspeed:
        print(f"   DeepSpeed config: {args.deepspeed}", flush=True)
        try:
            if not os.path.exists(args.deepspeed):
                raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed}")
            with open(args.deepspeed, "r") as f:
                ds_config = json.load(f)
            print(f"   Initializing DeepSpeed...", flush=True)
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
            print(f"   ✓ DeepSpeed initialized", flush=True)
        except Exception as e:
            print(f"   ✗ DeepSpeed initialization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"   Skipping DeepSpeed (not specified)", flush=True)

    # 加载数据
    print(f"\n[8/10] Loading data...", flush=True)
    try:
        from src.data.dataset import create_dataloader

        print(f"   Arrow dir: {config['train_data']['arrow_dir']}", flush=True)
        print(f"   Mask dir: {config['train_data']['mask_dir']}", flush=True)

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

        print(f"   ✓ Dataloader created, {len(train_dataloader.dataset)} samples", flush=True)
        print(f"   Batches per epoch: {len(train_dataloader)}", flush=True)
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 训练循环
    print(f"\n[9/10] Preparing training loop...", flush=True)
    print(f"   Max steps: {num_training_steps}", flush=True)
    print(f"   Batch size: {config.get('batch_size', 2)}", flush=True)
    print(f"   Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}", flush=True)

    global_step = 0
    trainer.model.train()
    print(f"   ✓ Model set to training mode", flush=True)

    # 创建进度条
    pbar = tqdm(total=num_training_steps, desc="Training") if not use_distributed or dist.get_rank() == 0 else None

    print(f"\n[10/10] Starting training loop...", flush=True)
    print("=" * 80, flush=True)

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
