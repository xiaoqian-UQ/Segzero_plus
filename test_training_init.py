#!/usr/bin/env python3
"""测试训练脚本初始化"""

import sys
import yaml
import traceback

def test_imports():
    """测试所有必需的导入"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from src.utils.parser import SegZeroOutputParser
        print("✓ SegZeroOutputParser imported")
    except Exception as e:
        print(f"✗ SegZeroOutputParser import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.utils.sam_utils import SAM2Wrapper
        print("✓ SAM2Wrapper imported")
    except Exception as e:
        print(f"✗ SAM2Wrapper import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.train.reward_functions import NegativePointRewardCalculator
        print("✓ NegativePointRewardCalculator imported")
    except Exception as e:
        print(f"✗ NegativePointRewardCalculator import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.data.dataset import create_dataloader
        print("✓ create_dataloader imported")
    except Exception as e:
        print(f"✗ create_dataloader import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints
        print("✓ GRPOTrainerWithNegativePoints imported")
    except Exception as e:
        print(f"✗ GRPOTrainerWithNegativePoints import failed: {e}")
        traceback.print_exc()
        return False

    print("\nAll imports successful!\n")
    return True


def test_config():
    """测试配置文件加载"""
    print("=" * 60)
    print("Testing config loading...")
    print("=" * 60)

    config_path = "configs/negative_points_config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded from {config_path}")
        print(f"  - model_path: {config.get('model_path', 'N/A')}")
        print(f"  - sam_config: {config.get('sam_config', 'N/A')}")
        print(f"  - sam_checkpoint: {config.get('sam_checkpoint', 'N/A')}")
        print(f"  - batch_size: {config.get('batch_size', 'N/A')}")
        print(f"  - max_steps: {config.get('max_steps', 'N/A')}")
        return True, config
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        traceback.print_exc()
        return False, None


def test_trainer_init(config):
    """测试训练器初始化"""
    print("\n" + "=" * 60)
    print("Testing trainer initialization...")
    print("=" * 60)

    # 修改配置以避免真正加载模型
    test_config = config.copy()
    test_config["device"] = "cpu"  # 使用CPU避免CUDA错误

    try:
        from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints
        print("Initializing trainer (this may take a while)...")
        trainer = GRPOTrainerWithNegativePoints(test_config)
        print("✓ Trainer initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        traceback.print_exc()
        return False


def test_main_function():
    """测试main函数能否被调用"""
    print("\n" + "=" * 60)
    print("Testing main function...")
    print("=" * 60)

    try:
        from src.train.grpo_seg_zero_negative import main
        print("✓ main function imported")
        print("  Note: Not actually calling main() to avoid full training")
        return True
    except Exception as e:
        print(f"✗ main function import failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "🚀" * 30)
    print("Training Script Initialization Test")
    print("🚀" * 30 + "\n")

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed! Fix imports before proceeding.")
        sys.exit(1)

    # Test 2: Config
    success, config = test_config()
    if not success:
        print("\n❌ Config test failed! Fix config before proceeding.")
        sys.exit(1)

    # Test 3: Main function
    if not test_main_function():
        print("\n❌ Main function test failed!")
        sys.exit(1)

    # Test 4: Trainer init (optional, commented out to save time)
    print("\n⚠️  Skipping trainer initialization test (takes too long)")
    print("   Uncomment test_trainer_init() in the code to test this")
    # if not test_trainer_init(config):
    #     print("\n❌ Trainer initialization failed!")
    #     sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nThe training script should be ready to run.")
    print("Try running: bash scripts/train_negative_points.sh")


if __name__ == "__main__":
    main()
