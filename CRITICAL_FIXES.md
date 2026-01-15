# Critical GRPO Fixes

## Fix 1: Temperature Consistency (High Priority)

### Problem
Sampling used temperature=0.7, but log_prob computation used temperature=1.0:

```python
# Sampling (line 276)
generated = model.generate(
    temperature=self.temperature,  # 0.7
    ...
)

# Log prob computation (line 351)
log_probs = torch.log_softmax(gen_logits, dim=-1)  # T=1.0
```

This causes **mismatch between sampling distribution and optimization objective**.

### Solution
Apply temperature when computing log_probs:

```python
# Now consistent with sampling
log_probs = torch.log_softmax(gen_logits / self.temperature, dim=-1)
```

### Why This Matters
- GRPO optimizes log P(sequence | input) under the policy distribution
- If policy uses T=0.7 but we optimize T=1.0, we're optimizing the wrong objective
- This can lead to unstable training and poor convergence

---

## Fix 2: EOS/PAD Token Masking (Medium Priority)

### Problem
model.generate() appends PAD tokens after EOS:

```
Sequence: [token1, token2, EOS, PAD, PAD, PAD]
```

Current code sums log_probs over entire sequence, including PADs:

```python
total_log_prob = token_log_probs.sum()  # Includes PAD tokens
```

### Solution
Create mask to stop at EOS:

```python
# Find first EOS position
eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)
# Mask EOS and everything after
mask[batch_idx, first_eos:] = 0

# Also mask PAD tokens
mask = mask * (sequence != pad_token_id).float()

# Apply mask
total_log_prob = (token_log_probs * mask).sum()
```

### Why This Matters
- PAD tokens are meaningless noise
- Including them in loss can:
  - Add random variance to gradients
  - Bias optimization toward sequences with more PADs
  - Confuse the model about when to stop generating

---

## Code Changes

File: src/train/grpo_seg_zero_negative.py

### Before (Incorrect)
```python
def _compute_sequence_log_probs(self, inputs, sequence):
    # ... forward pass ...
    gen_logits = logits[:, input_len-1:-1, :]

    # Wrong: T=1.0
    log_probs = torch.log_softmax(gen_logits, dim=-1)

    token_log_probs = torch.gather(...)

    # Wrong: includes PAD
    total_log_prob = token_log_probs.sum()

    return total_log_prob
```

### After (Correct)
```python
def _compute_sequence_log_probs(self, inputs, sequence):
    # ... forward pass ...
    gen_logits = logits[:, input_len-1:-1, :]

    # Fix 1: Apply temperature
    log_probs = torch.log_softmax(gen_logits / self.temperature, dim=-1)

    token_log_probs = torch.gather(...)

    # Fix 2: Create mask for EOS/PAD
    mask = torch.ones_like(sequence, dtype=torch.float)

    # Mask after first EOS
    eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)
    if len(eos_positions[0]) > 0:
        for batch_idx in range(sequence.shape[0]):
            batch_eos = eos_positions[1][eos_positions[0] == batch_idx]
            if len(batch_eos) > 0:
                first_eos = batch_eos[0].item()
                mask[batch_idx, first_eos:] = 0

    # Also mask PAD tokens
    if pad_token_id is not None:
        mask = mask * (sequence != pad_token_id).float()

    # Apply mask
    total_log_prob = (token_log_probs * mask).sum()

    return total_log_prob
```

---

## Verification

Run the test script:

```bash
python test_grpo_gradient.py
```

Expected output:
```
Test 3: Temperature Consistency
Temperature applied in log_prob calculation
PASSED

Test 4: EOS/PAD Masking
EOS/PAD masking implemented
Log probs will stop at EOS, not include PAD tokens
PASSED
```

---

## Impact

### Without These Fixes
- Training may converge slowly or not at all
- Model may learn incorrect distributions
- Gradients contaminated by PAD noise

### With These Fixes
- Correct GRPO objective optimization
- Cleaner gradients (no PAD noise)
- More stable and faster convergence
- Better final performance

---

## Technical Details

### Temperature in GRPO

GRPO objective:
```
max E[log π_θ(a|s) * A(s,a)]
```

where π_θ is the policy distribution.

If policy samples from π_θ(a|s) ∝ exp(logits / T), then log π_θ must also use temperature T.

Otherwise we optimize a different distribution than we sample from.

### EOS Masking in Sequence Models

Standard practice in sequence modeling:
- Loss is computed only up to EOS token
- PAD tokens are used for batching but ignored in loss
- Most implementations use attention_mask for this

Our GRPO implementation generates sequences independently, so we need explicit EOS detection.

---

## References

1. Group Relative Policy Optimization (GRPO)
   - Must match sampling and optimization distributions

2. Sequence-to-Sequence Models
   - Standard practice to mask PAD tokens in loss
   - See: Attention is All You Need (Transformer paper)

3. Temperature Sampling
   - Affects both sampling and probability computation
   - Must be consistent for correct gradient estimation
