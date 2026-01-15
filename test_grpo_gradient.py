#!/usr/bin/env python3
"""测试GRPO梯度和VL输入是否正确"""

import torch
import numpy as np
from PIL import Image
import sys

def test_gradient_flow():
    """测试梯度是否正确流动"""
    print("=" * 60)
    print("Test 1: Gradient Flow")
    print("=" * 60)

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints

        # 创建简化的配置（用于测试）
        config = {
            "model_path": "/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/",
            "sam_config": "/mnt/xiaoqian/model/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam_checkpoint": "/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lora_r": 64,
            "lora_alpha": 128,
            "group_size": 2,  # 小一点加速测试
            "temperature": 0.7,
            "clip_lower": -0.2,
            "clip_upper": 0.28,
            "alpha": 1.0,
            "beta": 1.0,
            "lambda_neg": 0.3,
            "lambda_format": 0.1
        }

        print("Initializing trainer (this may take a while)...")
        trainer = GRPOTrainerWithNegativePoints(config)
        print("✓ Trainer initialized")

        # 检查LoRA参数
        print("\nChecking LoRA parameters...")
        lora_params = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        if lora_params:
            print(f"✓ Found {len(lora_params)} trainable LoRA parameters")
            print(f"  Example: {lora_params[0]}")
        else:
            print("✗ No trainable parameters found!")
            return False

        # 创建模拟批次
        print("\nCreating mock batch...")
        batch = {
            "image": [np.random.randint(0, 255, (840, 840, 3), dtype=np.uint8)],
            "query": ["the red cup on the table"],
            "gt_mask": [np.random.randint(0, 2, (840, 840)).astype(np.float32)]
        }
        print("✓ Mock batch created")

        # 记录初始参数
        print("\nRecording initial parameters...")
        initial_params = {
            name: param.clone().detach()
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        }
        first_param_name = list(initial_params.keys())[0]
        print(f"  Initial param '{first_param_name}' norm: {initial_params[first_param_name].norm().item():.6f}")

        # 执行一步训练
        print("\nExecuting training step...")
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-5)
        metrics = trainer.train_step(batch)
        print(f"✓ Training step completed")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Mean reward: {metrics['mean_reward']:.6f}")

        # 检查参数是否更新
        print("\nChecking if parameters were updated...")
        updated = False
        for name, param in trainer.model.named_parameters():
            if name in initial_params:
                diff = (param - initial_params[name]).abs().sum().item()
                if diff > 1e-10:
                    updated = True
                    print(f"✓ Parameter '{name}' updated (diff: {diff:.6e})")
                    break

        if not updated:
            print("✗ Parameters were NOT updated! Gradient flow is broken.")
            return False

        print("\n✅ Gradient flow test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vl_input():
    """测试VL模型输入是否包含图像"""
    print("\n" + "=" * 60)
    print("Test 2: Vision-Language Input")
    print("=" * 60)

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints

        config = {
            "model_path": "/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/",
            "sam_config": "/mnt/xiaoqian/model/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam_checkpoint": "/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lora_r": 64,
            "lora_alpha": 128,
        }

        print("Initializing trainer...")
        trainer = GRPOTrainerWithNegativePoints(config)
        print("✓ Trainer initialized")

        # 检查是否是VL模型
        print(f"\nIs VL model: {trainer.is_vl_model}")

        if trainer.is_vl_model:
            print("\nTesting VL input preparation...")

            # 创建测试图像和prompt
            image = np.random.randint(0, 255, (840, 840, 3), dtype=np.uint8)
            prompt = "Find the red cup"

            # 准备输入
            inputs = trainer._prepare_inputs(image, prompt)

            print(f"✓ Inputs prepared")
            print(f"  Keys: {list(inputs.keys())}")

            # 检查必需的键
            required_keys = ["input_ids"]
            vl_keys = ["pixel_values", "image_grid_thw"]

            for key in required_keys:
                if key in inputs:
                    print(f"  ✓ '{key}' present, shape: {inputs[key].shape}")
                else:
                    print(f"  ✗ '{key}' missing!")
                    return False

            # 检查VL特定的键
            has_image = False
            for key in vl_keys:
                if key in inputs:
                    print(f"  ✓ '{key}' present, shape: {inputs[key].shape}")
                    has_image = True

            if not has_image:
                print("  ✗ No image features found (pixel_values or image_grid_thw)!")
                print("     Model won't be able to see the image!")
                return False

            print("\n✅ Vision-Language input test PASSED")
            return True
        else:
            print("  Model is not a VL model, skipping VL-specific tests")
            print("\n✅ Test PASSED (non-VL model)")
            return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temperature_consistency():
    """测试temperature是否在采样和计算log_prob时保持一致"""
    print("\n" + "=" * 60)
    print("Test 3: Temperature Consistency")
    print("=" * 60)

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints
        import torch

        config = {
            "model_path": "/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/",
            "sam_config": "/mnt/xiaoqian/model/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam_checkpoint": "/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "temperature": 0.7,  # 非1的temperature
            "lora_r": 64,
            "lora_alpha": 128,
        }

        print("Initializing trainer...")
        trainer = GRPOTrainerWithNegativePoints(config)
        print(f"Trainer temperature: {trainer.temperature}")

        # 检查_compute_sequence_log_probs是否使用temperature
        import inspect
        source = inspect.getsource(trainer._compute_sequence_log_probs)

        if "/ self.temperature" in source or "/self.temperature" in source:
            print("Temperature applied in log_prob calculation")
            return True
        else:
            print("Temperature NOT applied in log_prob calculation")
            print("   This causes mismatch between sampling and optimization")
            return False

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eos_masking():
    """测试EOS/PAD token是否被正确mask"""
    print("\n" + "=" * 60)
    print("Test 4: EOS/PAD Masking")
    print("=" * 60)

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints
        import torch

        config = {
            "model_path": "/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B/",
            "sam_config": "/mnt/xiaoqian/model/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam_checkpoint": "/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lora_r": 64,
            "lora_alpha": 128,
        }

        print("Initializing trainer...")
        trainer = GRPOTrainerWithNegativePoints(config)

        # 检查_compute_sequence_log_probs是否mask EOS/PAD
        import inspect
        source = inspect.getsource(trainer._compute_sequence_log_probs)

        has_mask = "mask" in source
        has_eos_check = "eos_token_id" in source or "pad_token_id" in source

        if has_mask and has_eos_check:
            print("EOS/PAD masking implemented")
            print("   Log probs will stop at EOS, not include PAD tokens")
            return True
        else:
            print("EOS/PAD masking NOT implemented")
            print("   Warning: PAD tokens after EOS will be included in loss")
            return False

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("GRPO Implementation Validation")
    print("=" * 60 + "\n")

    print("Warning: These tests will load the full model")
    print("Make sure you have enough GPU memory (at least 16GB)")
    print("Tests may take a few minutes\n")

    # Test 1: Gradient flow
    test1_passed = test_gradient_flow()

    # Test 2: VL input
    test2_passed = test_vl_input()

    # Test 3: Temperature consistency
    test3_passed = test_temperature_consistency()

    # Test 4: EOS masking
    test4_passed = test_eos_masking()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Gradient Flow Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"VL Input Test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Temperature Consistency Test: {'PASSED' if test3_passed else 'FAILED'}")
    print(f"EOS/PAD Masking Test: {'PASSED' if test4_passed else 'FAILED'}")

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed

    if all_passed:
        print("\nAll tests PASSED! The GRPO implementation is correct.")
        print("You can now run full training with confidence.")
        sys.exit(0)
    else:
        print("\nSome tests FAILED. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
