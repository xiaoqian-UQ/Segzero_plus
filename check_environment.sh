#!/bin/bash
# check_environment.sh - 检查训练环境是否就绪

echo "================================"
echo "Environment Check Script"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. 检查Python版本
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [ $? -eq 0 ]; then
    check_pass "Python version: $PYTHON_VERSION"
else
    check_fail "Python not found"
    exit 1
fi
echo ""

# 2. 检查必需的Python包
echo "2. Checking Python packages..."
PACKAGES=("torch" "transformers" "peft" "datasets" "deepspeed" "yaml" "PIL" "numpy")

for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        check_pass "$pkg ($VERSION)"
    else
        check_fail "$pkg not installed"
    fi
done
echo ""

# 3. 检查CUDA
echo "3. Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    check_pass "nvidia-smi found"
    echo "   GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "   - $line"
    done
else
    check_warn "nvidia-smi not found (CPU-only mode?)"
fi
echo ""

# 4. 检查项目结构
echo "4. Checking project structure..."
REQUIRED_DIRS=("src" "src/train" "src/utils" "src/data" "configs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory: $dir"
    else
        check_fail "Directory missing: $dir"
    fi
done
echo ""

# 5. 检查必需的文件
echo "5. Checking required files..."
REQUIRED_FILES=(
    "src/__init__.py"
    "src/train/__init__.py"
    "src/utils/__init__.py"
    "src/data/__init__.py"
    "src/train/grpo_seg_zero_negative.py"
    "src/utils/parser.py"
    "src/utils/sam_utils.py"
    "src/utils/solution_parser.py"
    "src/train/reward_functions.py"
    "src/data/dataset.py"
    "configs/negative_points_config.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file"
    else
        check_fail "File missing: $file"
    fi
done
echo ""

# 6. 检查配置文件中的路径
echo "6. Checking paths in config file..."
CONFIG_FILE="configs/negative_points_config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    # 提取路径（简单的grep，可能不完美）
    MODEL_PATH=$(grep "model_path:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    SAM_CONFIG=$(grep "sam_config:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    SAM_CHECKPOINT=$(grep "sam_checkpoint:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    ARROW_DIR=$(grep "arrow_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    MASK_DIR=$(grep "mask_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

    echo "   Model path: $MODEL_PATH"
    if [ -d "$MODEL_PATH" ]; then
        check_pass "Model directory exists"
    else
        check_fail "Model directory not found: $MODEL_PATH"
    fi

    echo "   SAM config: $SAM_CONFIG"
    if [ -f "$SAM_CONFIG" ]; then
        check_pass "SAM config exists"
    else
        check_fail "SAM config not found: $SAM_CONFIG"
    fi

    echo "   SAM checkpoint: $SAM_CHECKPOINT"
    if [ -f "$SAM_CHECKPOINT" ]; then
        check_pass "SAM checkpoint exists"
    else
        check_fail "SAM checkpoint not found: $SAM_CHECKPOINT"
    fi

    echo "   Arrow data dir: $ARROW_DIR"
    if [ -d "$ARROW_DIR" ]; then
        ARROW_COUNT=$(find "$ARROW_DIR" -name "*.arrow" 2>/dev/null | wc -l)
        check_pass "Arrow directory exists ($ARROW_COUNT .arrow files)"
    else
        check_fail "Arrow directory not found: $ARROW_DIR"
    fi

    echo "   Mask dir: $MASK_DIR"
    if [ -d "$MASK_DIR" ]; then
        MASK_COUNT=$(find "$MASK_DIR" -name "*.png" 2>/dev/null | wc -l)
        check_pass "Mask directory exists ($MASK_COUNT .png files)"
    else
        check_fail "Mask directory not found: $MASK_DIR"
    fi
else
    check_fail "Config file not found: $CONFIG_FILE"
fi
echo ""

# 7. 检查DeepSpeed配置
echo "7. Checking DeepSpeed config..."
DS_CONFIG="configs/deepspeed_zero2.json"
if [ -f "$DS_CONFIG" ]; then
    check_pass "DeepSpeed config exists: $DS_CONFIG"
    # 检查JSON是否有效
    if python3 -c "import json; json.load(open('$DS_CONFIG'))" 2>/dev/null; then
        check_pass "DeepSpeed config is valid JSON"
    else
        check_fail "DeepSpeed config is invalid JSON"
    fi
else
    check_warn "DeepSpeed config not found: $DS_CONFIG (will use default)"
fi
echo ""

# 8. 测试导入
echo "8. Testing Python imports..."
python3 -c "
import sys
success = True
try:
    from src.utils.parser import SegZeroOutputParser
    print('✓ SegZeroOutputParser')
except Exception as e:
    print(f'✗ SegZeroOutputParser: {e}')
    success = False

try:
    from src.utils.sam_utils import SAM2Wrapper
    print('✓ SAM2Wrapper')
except Exception as e:
    print(f'✗ SAM2Wrapper: {e}')
    success = False

try:
    from src.train.reward_functions import NegativePointRewardCalculator
    print('✓ NegativePointRewardCalculator')
except Exception as e:
    print(f'✗ NegativePointRewardCalculator: {e}')
    success = False

try:
    from src.data.dataset import create_dataloader
    print('✓ create_dataloader')
except Exception as e:
    print(f'✗ create_dataloader: {e}')
    success = False

try:
    from src.train.grpo_seg_zero_negative import GRPOTrainerWithNegativePoints, main
    print('✓ GRPOTrainerWithNegativePoints and main')
except Exception as e:
    print(f'✗ GRPOTrainerWithNegativePoints: {e}')
    success = False

sys.exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo ""
    check_pass "All imports successful"
else
    echo ""
    check_fail "Some imports failed"
fi
echo ""

# 9. 总结
echo "================================"
echo "Environment Check Complete"
echo "================================"
echo ""
echo "If all checks passed, you can run:"
echo "  bash scripts/train_negative_points.sh"
echo ""
echo "If some checks failed, please:"
echo "  1. Install missing Python packages: pip install <package>"
echo "  2. Fix file paths in configs/negative_points_config.yaml"
echo "  3. Check the TRAINING_DEBUG_GUIDE.md for more help"
echo ""
