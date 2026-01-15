# GRPO Implementation Fix Summary

## 🔴 原始问题

你发现了两个**致命缺陷**，会导致训练虽然能跑但无法学到东西：

### 问题 1：梯度丢失
**原因**：
- 在 `torch.no_grad()` 中生成输出
- 从 `generated.scores` 直接计算 log_probs
- 这些 log_probs **不带梯度**
- loss 无法反向传播到 LoRA 参数

**后果**：LoRA 参数不会更新，模型不会学习

### 问题 2：VL模型看不到图像
**原因**：
- `_sample_outputs` 只接收文本 prompt
- 没有传入图像数据
- Qwen2.5-VL 是视觉语言模型，**必须同时接收图像和文本**

**后果**：模型在盲打，无法根据图像内容预测负点

---

## ✅ 修复方案

### 正确的 GRPO 流程

```
阶段1：采样（no_grad）
  └─ 生成 K 个候选输出（只用于获取文本）

阶段2：奖励计算
  └─ 用 SAM2 评估每个候选的质量

阶段3：重新计算 log_probs（带梯度）
  └─ 前向传播计算带梯度的 log_probs

阶段4：参数更新
  └─ 用 GRPO loss 更新 LoRA
```

### 修复 1：重新计算带梯度的 log_probs

**修改前**：
```python
def _sample_outputs(self, prompt: str, k: int):
    for _ in range(k):
        with torch.no_grad():  # ❌ 没有梯度
            generated = model.generate(...)

        # ❌ 直接从 generated.scores 计算，没有梯度
        log_prob = self._compute_log_prob(generated)
        log_probs.append(log_prob)

    return outputs, log_probs  # ❌ log_probs 无梯度
```

**修改后**：
```python
def _sample_outputs(self, inputs: Dict, k: int):
    for _ in range(k):
        with torch.no_grad():  # ✅ 采样时不需要梯度
            generated = model.generate(...)

        # ✅ 只保存生成的 token 序列
        sequences.append(seq)

    return outputs_text, sequences  # ✅ 返回序列，不返回log_probs

def _compute_sequence_log_probs(self, inputs, sequence):
    """✅ 重新前向传播，计算带梯度的 log_probs"""

    # ✅ 前向传播（带梯度）
    outputs = model(**forward_inputs)
    logits = outputs.logits

    # ✅ 计算 log_probs（带梯度）
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=sequence.unsqueeze(-1))
    total_log_prob = token_log_probs.sum()  # ✅ 带梯度

    return total_log_prob

def _compute_grpo_loss_with_recompute(self, all_inputs, all_sequences, all_rewards):
    """✅ 重新计算 log_probs 并计算 GRPO loss"""
    for inputs, sequences, rewards in zip(...):
        for sequence, advantage in zip(sequences, advantages):
            # ✅ 重新计算带梯度的 log_prob
            log_prob = self._compute_sequence_log_probs(inputs, sequence)

            # ✅ GRPO loss（带梯度）
            loss_term = -log_prob * advantage
            total_loss = total_loss + loss_term

    return total_loss  # ✅ 可以反向传播
```

**关键变化**：
1. 采样时只保存 token 序列，不计算 log_probs
2. 新增 `_compute_sequence_log_probs`：重新前向传播计算带梯度的 log_probs
3. GRPO loss 基于带梯度的 log_probs，可以正确更新参数

### 修复 2：VL 模型正确输入图像

**修改前**：
```python
def _sample_outputs(self, prompt: str, k: int):
    # ❌ 只有文本
    inputs = self.tokenizer(text=prompt, return_tensors="pt")

    for _ in range(k):
        # ❌ 模型看不到图像
        generated = model.generate(**inputs, ...)
```

**修改后**：
```python
def _prepare_inputs(self, image: np.ndarray, prompt: str):
    """✅ 准备 VL 模型输入（图像 + 文本）"""
    from PIL import Image as PILImage

    if self.is_vl_model:
        pil_image = PILImage.fromarray(image)

        # ✅ 构建 VL 消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},  # ✅ 图像
                    {"type": "text", "text": prompt}        # ✅ 文本
                ]
            }
        ]

        # ✅ 使用 processor 处理
        text = self.tokenizer.apply_chat_template(messages, ...)
        inputs = self.tokenizer(
            text=[text],
            images=[pil_image],  # ✅ 传入图像
            return_tensors="pt"
        )

    return inputs  # ✅ 包含图像特征（pixel_values）

def _sample_outputs(self, inputs: Dict, k: int):
    # ✅ inputs 包含图像和文本
    for _ in range(k):
        generated = model.generate(**inputs, ...)  # ✅ 模型能看到图像

def _compute_sequence_log_probs(self, inputs, sequence):
    # ✅ 重新计算时也传入图像特征
    if "pixel_values" in inputs:
        forward_inputs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        forward_inputs["image_grid_thw"] = inputs["image_grid_thw"]

    outputs = model(**forward_inputs)  # ✅ 前向传播时包含图像
```

**关键变化**：
1. 新增 `_prepare_inputs`：使用 processor 正确处理图像和文本
2. 采样时传入完整的 inputs（包含图像特征）
3. 重新计算 log_probs 时也传入图像特征

---

## 📊 修复对比

### 训练流程对比

| 阶段 | 修改前 | 修改后 |
|------|--------|--------|
| 输入准备 | ❌ 只有文本 | ✅ 图像 + 文本 |
| 采样生成 | ❌ 在 no_grad 中计算 log_probs | ✅ 只生成序列 |
| 奖励计算 | ✅ 正确 | ✅ 正确 |
| Log_probs | ❌ 无梯度 | ✅ 重新计算，带梯度 |
| Loss 计算 | ❌ 无法反向传播 | ✅ 可以反向传播 |
| 参数更新 | ❌ 不更新 | ✅ 正确更新 LoRA |

### 代码结构对比

**修改前**：
```python
train_step():
    for each sample:
        outputs, log_probs = _sample_outputs(prompt, K)  # ❌ 无梯度
        rewards = compute_rewards(outputs)

    loss = compute_loss(log_probs, rewards)  # ❌ log_probs 无梯度
    loss.backward()  # ❌ 无效
```

**修改后**：
```python
train_step():
    for each sample:
        inputs = _prepare_inputs(image, prompt)  # ✅ 图像+文本
        outputs, sequences = _sample_outputs(inputs, K)  # ✅ 只生成
        rewards = compute_rewards(outputs)

    loss = _compute_grpo_loss_with_recompute(
        inputs, sequences, rewards
    )  # ✅ 重新计算带梯度的 log_probs

    loss.backward()  # ✅ 正确更新 LoRA
```

---

## 🧪 如何验证修复

运行测试脚本：

```bash
python test_grpo_gradient.py
```

测试会验证：

### Test 1: 梯度流动
- ✅ LoRA 参数是否 trainable
- ✅ 执行 training step 后参数是否更新
- ✅ 参数变化是否大于阈值

### Test 2: VL 输入
- ✅ 输入是否包含 `pixel_values`（图像特征）
- ✅ 输入是否包含 `input_ids`（文本）
- ✅ 模型是否能同时看到图像和文本

**期望输出**：
```
Test 1: Gradient Flow
✓ Trainer initialized
✓ Found 128 trainable LoRA parameters
✓ Parameter 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight' updated
✅ Gradient flow test PASSED

Test 2: Vision-Language Input
Is VL model: True
✓ 'input_ids' present, shape: torch.Size([1, 128])
✓ 'pixel_values' present, shape: torch.Size([1, 3, 448, 448])
✓ 'image_grid_thw' present, shape: torch.Size([1, 3])
✅ Vision-Language input test PASSED

🎉 All tests PASSED!
```

---

## 🚀 下一步

### 1. 运行验证测试（推荐）

在服务器上运行：
```bash
python test_grpo_gradient.py
```

确保两个测试都通过。

### 2. 开始真实训练

```bash
bash scripts/train_negative_points.sh
```

现在训练应该能：
- ✅ LoRA 参数会正确更新
- ✅ 模型能看到图像
- ✅ 能学习预测有效的负点

### 3. 监控训练指标

观察：
- `loss` 应该下降
- `mean_reward` 应该上升
- 负点应该逐渐落在混淆区域（而不是 GT 内）

---

## 📚 参考

### GRPO 算法核心思想

Group Relative Policy Optimization (GRPO)：
1. 对每个输入采样 K 个输出
2. 计算每个输出的奖励
3. 用**组内相对奖励**（相对于组内平均）计算优势
4. 优化 log P(output | input) × advantage

**关键**：重新计算 log_probs 时必须带梯度！

### Qwen2.5-VL 输入格式

- 必须使用 `AutoProcessor`（而不是 `AutoTokenizer`）
- 图像通过 `pixel_values` 传入
- 文本通过 chat template 格式化
- `apply_chat_template` + `processor(text=..., images=...)` 是标准用法

---

## 💡 总结

**原始实现的问题**：
- 🔴 梯度断了 → LoRA 不更新 → 模型不学习
- 🔴 没图像 → 模型盲打 → 无法预测有效负点

**修复后**：
- ✅ 梯度正确流动 → LoRA 正确更新
- ✅ 图像正确输入 → 模型能看到视觉信息
- ✅ GRPO 算法正确实现 → 模型能学习

现在可以真正开始训练了！🎉
