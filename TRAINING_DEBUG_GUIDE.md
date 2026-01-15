# Training Script Debug Guide

## 问题：训练脚本立即退出

你遇到的问题是运行训练脚本时，程序立即退出而没有任何错误信息：

```bash
[2026-01-14 16:22:50,596] [INFO] [launch.py:367:main] Process 110473 exits successfully.
[2026-01-14 16:22:50,596] [INFO] [launch.py:367:main] Process 110472 exits successfully.
```

## 已完成的修复

### 1. 添加了 `__init__.py` 文件
所有Python包目录现在都有 `__init__.py` 文件：
- `src/__init__.py`
- `src/utils/__init__.py`
- `src/train/__init__.py`
- `src/data/__init__.py`

### 2. 添加了完整的 `main()` 函数
在 `src/train/grpo_seg_zero_negative.py` 中添加了：
- 命令行参数解析
- 配置文件加载
- 训练器初始化
- 数据加载器
- 训练主循环
- 检查点保存

### 3. 添加了详细的调试输出
现在训练脚本会在每个关键步骤打印详细信息：
- [1/10] 解析命令行参数
- [2/10] 初始化分布式训练
- [3/10] 加载配置文件
- [4/10] 创建输出目录
- [5/10] 初始化训练器
- [6/10] 初始化优化器和调度器
- [7/10] DeepSpeed初始化
- [8/10] 加载数据
- [9/10] 准备训练循环
- [10/10] 开始训练

每个步骤都有错误捕获和异常打印，帮助定位问题。

## 下一步诊断步骤

### 步骤1：检查日志输出

重新运行训练脚本：

```bash
bash scripts/train_negative_points.sh
```

现在你应该能看到详细的输出。查看程序在哪一步停止：
- 如果在 [1/10]，说明参数解析有问题
- 如果在 [5/10]，说明模型加载有问题
- 如果在 [8/10]，说明数据加载有问题

### 步骤2：检查输出日志文件

查看训练日志：

```bash
tail -100 outputs/negative_points_exp1/train.log
```

### 步骤3：运行简化测试

在项目根目录运行测试脚本（可能需要在服务器上运行）：

```bash
python3 test_training_init.py
```

这个脚本会测试所有导入是否正常。

### 步骤4：检查常见问题

#### 问题 A：数据路径不存在

检查配置文件中的路径是否存在：

```bash
# 检查 arrow 数据路径
ls -la /mnt/xiaoqian/dataset/refcocog/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/

# 检查 mask 路径
ls -la /mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks/ | head

# 检查模型路径
ls -la /mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/

# 检查 SAM2 路径
ls -la /mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt
```

如果路径不存在，修改 `configs/negative_points_config.yaml` 中的路径。

#### 问题 B：DeepSpeed 配置问题

检查 DeepSpeed 配置文件：

```bash
cat configs/deepspeed_zero2.json
```

确保配置文件格式正确（有效的JSON）。

#### 问题 C：CUDA 内存不足

如果模型加载时 CUDA 内存不足，编辑配置文件减小batch size：

```yaml
# configs/negative_points_config.yaml
batch_size: 1  # 从2改为1
```

#### 问题 D：Arrow 数据集加载失败

测试数据集是否能加载：

```python
from datasets import load_dataset
import glob

arrow_dir = "/mnt/xiaoqian/dataset/refcocog/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/"
arrow_files = glob.glob(f"{arrow_dir}/*.arrow")
print(f"Found {len(arrow_files)} arrow files")

dataset = load_dataset('arrow', data_files=arrow_files, split='train')
print(f"Loaded {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
```

### 步骤5：不使用 DeepSpeed 测试

如果问题出在 DeepSpeed，可以先测试单GPU训练：

编辑 `scripts/train_negative_points.sh`：

```bash
#!/bin/bash

# 配置
export CUDA_VISIBLE_DEVICES=0  # 只用一张卡

# 路径
CONFIG="configs/negative_points_config.yaml"
OUTPUT_DIR="outputs/negative_points_test"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 单卡训练（不用 DeepSpeed）
python src/train/grpo_seg_zero_negative.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/train.log
```

## 常见错误和解决方案

### 错误 1: `ModuleNotFoundError: No module named 'transformers'`

**解决方案**：安装依赖
```bash
pip install transformers torch peft datasets accelerate deepspeed
```

### 错误 2: `FileNotFoundError: [Errno 2] No such file or directory: 'configs/negative_points_config.yaml'`

**解决方案**：确保在项目根目录运行脚本
```bash
cd /path/to/Segzero_plus
bash scripts/train_negative_points.sh
```

### 错误 3: `CUDA out of memory`

**解决方案1**：减小batch size
```yaml
batch_size: 1
```

**解决方案2**：使用梯度累积
```yaml
batch_size: 1
gradient_accumulation_steps: 16  # 相当于batch_size=16
```

**解决方案3**：使用DeepSpeed ZeRO-3
```bash
--deepspeed configs/deepspeed_zero3.json
```

### 错误 4: 数据加载失败

**检查步骤**：
1. 确认arrow文件路径正确
2. 确认mask文件存在
3. 检查文件权限

```bash
# 测试数据加载
python -c "
from src.data.dataset import RefCOCOgDataset
dataset = RefCOCOgDataset(
    arrow_dir='refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e',
    mask_dir='/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks',
    image_size=840
)
print(f'Dataset size: {len(dataset)}')
batch = dataset[0]
print(f'Sample keys: {batch.keys()}')
"
```

## 成功启动的标志

如果训练成功启动，你应该看到类似的输出：

```
================================================================================
GRPO Negative Points Training - START
================================================================================

[1/10] Parsing arguments...
   Config: configs/negative_points_config.yaml
   Output dir: outputs/negative_points_exp1
   Local rank: 0
   DeepSpeed: configs/deepspeed_zero2.json
   Distributed: True

[2/10] Initializing distributed training...
   Rank: 0, World size: 2

[3/10] Loading config from configs/negative_points_config.yaml...
   ✓ Config loaded successfully
   Model: /mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/
   Batch size: 2
   Max steps: 5000

[4/10] Creating output directory...
   ✓ Output dir: outputs/negative_points_exp1

[5/10] Initializing trainer (this may take a few minutes)...
   Loading model... (this takes time)
   ✓ Trainer initialized

[6/10] Initializing optimizer and scheduler...
   ✓ Optimizer: AdamW, LR=1e-05
   ✓ Scheduler: warmup=100, total=5000

[7/10] DeepSpeed initialization...
   DeepSpeed config: configs/deepspeed_zero2.json
   Initializing DeepSpeed...
   ✓ DeepSpeed initialized

[8/10] Loading data...
   Arrow dir: refcocog/...
   Mask dir: /mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks
   Loaded 9000 samples
   ✓ Dataloader created, 9000 samples
   Batches per epoch: 4500

[9/10] Preparing training loop...
   Max steps: 5000
   ✓ Model set to training mode

[10/10] Starting training loop...
================================================================================

=== Epoch 1 ===
Training:   0%|          | 0/5000 [00:00<?, ?it/s]
```

然后训练会开始，进度条会更新。

## 获取更多帮助

如果以上步骤都无法解决问题，请提供以下信息：

1. 完整的训练日志输出
2. 系统信息（GPU型号、CUDA版本等）
3. Python包版本：
   ```bash
   pip list | grep -E "(torch|transformers|deepspeed|datasets)"
   ```
4. 数据集检查结果

## 总结

主要修复内容：
1. ✅ 添加了 `__init__.py` 文件
2. ✅ 完善了 `main()` 函数
3. ✅ 添加了详细的调试输出
4. ✅ 添加了错误捕获和异常处理
5. ✅ 创建了测试脚本

现在重新运行训练脚本应该能看到详细的输出，帮助你定位问题所在。
