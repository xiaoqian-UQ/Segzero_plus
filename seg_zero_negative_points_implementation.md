# Seg-Zero++: 负点预测与对比奖励的详细实施方案

## 📋 项目概述

**目标**: 在Seg-Zero-7B基础上，通过引入负点预测机制和对比奖励函数，提升ReasonSeg和RefCOCO benchmark的性能。

**核心创新**: 训练MLLM同时预测正点和负点，利用SAM2的负点prompting能力来排除视觉上相似的背景区域。

**预期提升**: ReasonSeg gIoU从57.5提升到60+（目标超越RSVP的60.3）

---

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入 (Image + Query)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Qwen2.5-VL-7B (Reasoning Model)                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  <think>                                                 │    │
│  │  推理过程...识别目标对象...排除干扰对象...              │    │
│  │  </think>                                                │    │
│  │  <answer>                                                │    │
│  │  { "bbox": [x1,y1,x2,y2],                               │    │
│  │    "points_pos": [[px1,py1], [px2,py2]],     ← 正点     │    │
│  │    "points_neg": [[nx1,ny1], [nx2,ny2]] }    ← 负点(新) │    │
│  │  </answer>                                               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAM2.1-Large (Frozen)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Inputs:                                                 │    │
│  │  - bbox: [x1, y1, x2, y2]                               │    │
│  │  - point_coords: [[px1,py1], [px2,py2], [nx1,ny1], ...]│    │
│  │  - point_labels: [1, 1, 0, 0]  (1=正, 0=负)             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output: Segmentation Mask                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 第一步：环境搭建与代码准备

### 1.1 克隆并设置Seg-Zero环境

```bash
# 克隆原始Seg-Zero仓库
git clone https://github.com/dvlab-research/Seg-Zero.git
cd Seg-Zero

# 回退到单目标版本（如果需要）
git reset --hard 77f9ea5887ec7e6abf398ed3cb483c65631c82b7

# 创建conda环境
conda create -n segzero_plus python=3.12
conda activate segzero_plus

# 安装依赖
pip install torch==2.6.0 torchvision==0.21.0
pip install -e .
```

### 1.2 下载预训练模型

```bash
mkdir -p pretrained_models
cd pretrained_models

# 下载Seg-Zero-7B checkpoint
git lfs install
git clone https://huggingface.co/Ricky06662/Seg-Zero-7B

# 下载SAM2.1-Large
# 从 https://github.com/facebookresearch/sam2 获取
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 1.3 下载训练数据

```bash
# 下载RefCOCOg-9K训练数据
python training_scripts/download_dataset.py

# 数据将保存在 ./data/refcocog_9k_840/
```

---

## 📝 第二步：修改输出格式

### 2.1 新的用户Prompt模板

创建文件 `verl/utils/reward_score/prompt_templates.py`:

```python
# 原始Seg-Zero Prompt
ORIGINAL_PROMPT = """
Please find '{Question}' with bbox and points.
Compare the difference between objects and find the most closely matched one.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.
i.e., <think> thinking process here </think>
<answer> { 'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180] } </answer>
"""

# 新的带负点的Prompt
NEGATIVE_POINT_PROMPT = """
Please find '{Question}' with bbox, positive points, and negative points.
Compare the difference between objects and find the most closely matched one.
Identify confusing background regions that should be EXCLUDED from the segmentation.

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.

Output format in JSON:
- bbox: bounding box of the target object [x1, y1, x2, y2]
- points_pos: two positive points inside the target object [[x1,y1], [x2,y2]]
- points_neg: 1-3 negative points on confusing background regions [[x1,y1], ...] 
  (regions that look similar to target but should NOT be segmented)

Example:
<think> 
The query asks for "the person wearing red". 
There are two people in the image - one wearing red (target) and one wearing orange (similar, confusing).
I will place positive points on the person in red, and negative points on the orange clothing to help distinguish them.
</think>
<answer> {{ "bbox": [10,100,200,210], "points_pos": [[30,110], [35,180]], "points_neg": [[250,150]] }} </answer>
"""
```

### 2.2 修改输出解析器

创建文件 `verl/utils/reward_score/output_parser.py`:

```python
import re
import json
from typing import Dict, List, Tuple, Optional

def parse_seg_zero_output(response: str) -> Dict:
    """
    解析Seg-Zero模型输出，支持正点和负点格式
    
    Returns:
        {
            'think': str,           # 推理过程
            'bbox': [x1,y1,x2,y2],  # 边界框
            'points_pos': [[x,y], [x,y]],  # 正点列表
            'points_neg': [[x,y], ...],    # 负点列表（可选）
            'format_valid': bool    # 格式是否有效
        }
    """
    result = {
        'think': '',
        'bbox': None,
        'points_pos': [],
        'points_neg': [],
        'format_valid': False
    }
    
    # 提取think部分
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        result['think'] = think_match.group(1).strip()
    
    # 提取answer部分
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return result
    
    answer_text = answer_match.group(1).strip()
    
    try:
        # 尝试解析JSON
        # 处理单引号的情况
        answer_text = answer_text.replace("'", '"')
        data = json.loads(answer_text)
        
        # 解析bbox
        if 'bbox' in data:
            result['bbox'] = data['bbox']
        
        # 解析正点 - 支持新旧两种格式
        if 'points_pos' in data:
            result['points_pos'] = data['points_pos']
        elif 'points_1' in data and 'points_2' in data:
            # 兼容旧格式
            result['points_pos'] = [data['points_1'], data['points_2']]
        
        # 解析负点
        if 'points_neg' in data:
            result['points_neg'] = data['points_neg']
        
        # 验证格式
        result['format_valid'] = (
            result['bbox'] is not None and
            len(result['bbox']) == 4 and
            len(result['points_pos']) >= 1
        )
        
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试正则匹配
        bbox_match = re.search(r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', answer_text)
        if bbox_match:
            result['bbox'] = [int(x) for x in bbox_match.groups()]
        
        # 正则匹配正点
        pos_match = re.search(r'"points_pos"\s*:\s*\[\[(\d+),\s*(\d+)\]', answer_text)
        if pos_match:
            result['points_pos'].append([int(pos_match.group(1)), int(pos_match.group(2))])
        
        # 正则匹配负点
        neg_matches = re.findall(r'"points_neg"\s*:\s*\[((?:\[\d+,\s*\d+\],?\s*)+)\]', answer_text)
        if neg_matches:
            for match in neg_matches:
                coords = re.findall(r'\[(\d+),\s*(\d+)\]', match)
                for coord in coords:
                    result['points_neg'].append([int(coord[0]), int(coord[1])])
    
    return result


def prepare_sam_prompts(parsed_output: Dict, image_size: Tuple[int, int] = (840, 840)) -> Dict:
    """
    将解析的输出转换为SAM2的输入格式
    
    Args:
        parsed_output: parse_seg_zero_output的返回值
        image_size: 图像尺寸 (height, width)
    
    Returns:
        {
            'box': np.array([x1,y1,x2,y2]),
            'point_coords': np.array([[x,y], ...]),
            'point_labels': np.array([1,1,0,0,...])  # 1=正, 0=负
        }
    """
    import numpy as np
    
    result = {
        'box': None,
        'point_coords': None,
        'point_labels': None
    }
    
    if parsed_output['bbox']:
        result['box'] = np.array(parsed_output['bbox'], dtype=np.float32)
    
    # 合并正点和负点
    all_points = []
    all_labels = []
    
    for pt in parsed_output['points_pos']:
        all_points.append(pt)
        all_labels.append(1)  # 正点标签
    
    for pt in parsed_output['points_neg']:
        all_points.append(pt)
        all_labels.append(0)  # 负点标签
    
    if all_points:
        result['point_coords'] = np.array(all_points, dtype=np.float32)
        result['point_labels'] = np.array(all_labels, dtype=np.int32)
    
    return result
```

---

## 🎯 第三步：实现奖励函数

### 3.1 完整奖励函数实现

创建文件 `verl/utils/reward_score/segmentation_rewards.py`:

```python
"""
Seg-Zero++ 奖励函数模块
包含格式奖励、精度奖励和对比奖励
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

# ============================================
# 格式奖励
# ============================================

def compute_think_format_reward(response: str) -> float:
    """
    检查是否包含正确的<think></think>标签
    
    Returns:
        1.0 如果格式正确，0.0 否则
    """
    import re
    pattern = r'<think>.*?</think>'
    match = re.search(pattern, response, re.DOTALL)
    return 1.0 if match else 0.0


def compute_seg_format_reward_soft(parsed_output: Dict) -> float:
    """
    软格式奖励：检查是否包含必要的关键字
    
    Returns:
        1.0 如果格式基本正确，0.0 否则
    """
    has_bbox = parsed_output['bbox'] is not None and len(parsed_output['bbox']) == 4
    has_points = len(parsed_output['points_pos']) >= 1
    
    return 1.0 if (has_bbox and has_points) else 0.0


def compute_seg_format_reward_strict(parsed_output: Dict) -> float:
    """
    严格格式奖励：检查是否完全符合预定义格式
    包括正点和负点格式
    
    Returns:
        1.0 如果格式完全正确，0.0 否则
    """
    # 检查bbox
    if parsed_output['bbox'] is None or len(parsed_output['bbox']) != 4:
        return 0.0
    
    # 检查正点（至少2个）
    if len(parsed_output['points_pos']) < 2:
        return 0.0
    
    # 检查所有坐标值是否为有效数字
    try:
        bbox = parsed_output['bbox']
        if not all(isinstance(x, (int, float)) and 0 <= x <= 840 for x in bbox):
            return 0.0
        
        for pt in parsed_output['points_pos']:
            if not (len(pt) == 2 and all(isinstance(x, (int, float)) and 0 <= x <= 840 for x in pt)):
                return 0.0
        
        for pt in parsed_output['points_neg']:
            if not (len(pt) == 2 and all(isinstance(x, (int, float)) and 0 <= x <= 840 for x in pt)):
                return 0.0
                
    except (TypeError, ValueError):
        return 0.0
    
    return 1.0


# ============================================
# 精度奖励
# ============================================

def compute_bbox_iou_reward(
    pred_bbox: List[float],
    gt_bbox: List[float],
    threshold: float = 0.5
) -> float:
    """
    计算bbox IoU奖励（硬奖励）
    
    Args:
        pred_bbox: [x1, y1, x2, y2] 预测框
        gt_bbox: [x1, y1, x2, y2] 真实框
        threshold: IoU阈值
    
    Returns:
        1.0 如果IoU > threshold，0.0 否则
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    
    # 计算交集
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection
    
    iou = intersection / union if union > 0 else 0.0
    
    return 1.0 if iou > threshold else 0.0


def compute_bbox_l1_reward(
    pred_bbox: List[float],
    gt_bbox: List[float],
    threshold: float = 10.0
) -> float:
    """
    计算bbox L1距离奖励
    
    Returns:
        1.0 如果平均L1距离 < threshold，0.0 否则
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    
    l1_dist = sum(abs(p - g) for p, g in zip(pred_bbox, gt_bbox)) / 4.0
    
    return 1.0 if l1_dist < threshold else 0.0


def compute_point_l1_reward(
    pred_points: List[List[float]],
    gt_mask: np.ndarray,
    threshold: float = 100.0
) -> float:
    """
    计算正点L1距离奖励
    检查预测的点是否在GT mask内部
    
    Args:
        pred_points: 预测的正点列表 [[x1,y1], [x2,y2], ...]
        gt_mask: 真实mask (H, W) 二值数组
        threshold: 距离阈值（像素）
    
    Returns:
        1.0 如果所有点都在mask内或接近，0.0 否则
    """
    if not pred_points or gt_mask is None:
        return 0.0
    
    h, w = gt_mask.shape
    
    for pt in pred_points:
        x, y = int(pt[0]), int(pt[1])
        
        # 检查是否在图像范围内
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        # 检查是否在mask内
        if gt_mask[y, x] > 0:
            continue
        
        # 如果不在mask内，计算到mask的最小距离
        mask_coords = np.argwhere(gt_mask > 0)  # (N, 2) in (y, x) format
        if len(mask_coords) == 0:
            return 0.0
        
        distances = np.sqrt(
            (mask_coords[:, 1] - x) ** 2 + 
            (mask_coords[:, 0] - y) ** 2
        )
        min_dist = distances.min()
        
        if min_dist > threshold:
            return 0.0
    
    return 1.0


# ============================================
# 对比奖励（负点奖励）- 核心创新
# ============================================

def compute_negative_point_reward(
    pred_neg_points: List[List[float]],
    gt_mask: np.ndarray,
    pred_bbox: List[float],
    confused_regions: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.5
) -> float:
    """
    计算负点对比奖励
    
    设计原则：
    1. 负点不应该落在GT mask内部（惩罚）
    2. 负点应该落在"混淆区域"（奖励）
    3. 负点应该在bbox附近但不在mask内
    
    Args:
        pred_neg_points: 预测的负点列表
        gt_mask: 真实mask
        pred_bbox: 预测的bbox
        confused_regions: 可选，混淆区域mask（SAM多mask歧义区域）
        alpha: 惩罚系数（负点落在GT内）
        beta: 奖励系数（负点落在混淆区域）
    
    Returns:
        奖励分数 [0.0, 1.0]
    """
    if not pred_neg_points:
        # 没有预测负点，给予基础分
        return 0.5
    
    if gt_mask is None:
        return 0.0
    
    h, w = gt_mask.shape
    total_reward = 0.0
    valid_points = 0
    
    for pt in pred_neg_points:
        x, y = int(pt[0]), int(pt[1])
        
        # 检查边界
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        valid_points += 1
        point_reward = 0.0
        
        # 惩罚：负点在GT mask内部
        if gt_mask[y, x] > 0:
            point_reward -= alpha
        else:
            # 奖励：负点在mask外部
            point_reward += 0.3
        
        # 奖励：负点在混淆区域
        if confused_regions is not None and confused_regions[y, x] > 0:
            point_reward += beta
        
        # 奖励：负点在bbox附近（有效的排除区域）
        if pred_bbox is not None:
            bx1, by1, bx2, by2 = pred_bbox
            # 扩展bbox区域
            margin = 50  # 像素
            extended_bbox = [
                max(0, bx1 - margin),
                max(0, by1 - margin),
                min(w, bx2 + margin),
                min(h, by2 + margin)
            ]
            if (extended_bbox[0] <= x <= extended_bbox[2] and 
                extended_bbox[1] <= y <= extended_bbox[3]):
                point_reward += 0.2
        
        total_reward += point_reward
    
    if valid_points == 0:
        return 0.0
    
    # 归一化到[0, 1]
    avg_reward = total_reward / valid_points
    # 将[-alpha, 0.5+beta]映射到[0, 1]
    normalized = (avg_reward + alpha) / (alpha + 0.5 + beta)
    
    return max(0.0, min(1.0, normalized))


def identify_confused_regions(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam_predictor,
    num_samples: int = 5
) -> np.ndarray:
    """
    使用SAM识别混淆区域
    通过在不同位置采样点，找到SAM认为可能是目标的区域
    
    Args:
        image: 输入图像 (H, W, 3)
        gt_mask: 真实mask
        sam_predictor: SAM2 predictor实例
        num_samples: 采样次数
    
    Returns:
        confused_regions: 混淆区域mask (H, W)
    """
    h, w = gt_mask.shape
    confused_regions = np.zeros((h, w), dtype=np.float32)
    
    # 获取GT mask的边界框
    mask_coords = np.argwhere(gt_mask > 0)
    if len(mask_coords) == 0:
        return confused_regions
    
    y_min, x_min = mask_coords.min(axis=0)
    y_max, x_max = mask_coords.max(axis=0)
    
    # 在GT bbox周围采样点
    margin = 100
    sample_region = [
        max(0, x_min - margin),
        max(0, y_min - margin),
        min(w, x_max + margin),
        min(h, y_max + margin)
    ]
    
    for _ in range(num_samples):
        # 随机采样一个点（不在GT mask内）
        for _ in range(10):  # 最多尝试10次
            x = np.random.randint(sample_region[0], sample_region[2])
            y = np.random.randint(sample_region[1], sample_region[3])
            if gt_mask[y, x] == 0:
                break
        else:
            continue
        
        # 使用SAM预测
        sam_predictor.set_image(image)
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        # 将SAM预测的区域（非GT）加入混淆区域
        for mask, score in zip(masks, scores):
            if score > 0.5:  # 只考虑高置信度预测
                # 排除与GT重叠的部分
                non_gt_region = mask & (gt_mask == 0)
                confused_regions += non_gt_region.astype(np.float32)
    
    # 归一化
    if confused_regions.max() > 0:
        confused_regions = confused_regions / confused_regions.max()
    
    return confused_regions


# ============================================
# 总奖励计算
# ============================================

def compute_total_reward(
    response: str,
    parsed_output: Dict,
    gt_bbox: List[float],
    gt_mask: np.ndarray,
    confused_regions: Optional[np.ndarray] = None,
    use_strict_format: bool = True,
    use_negative_reward: bool = True,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    计算总奖励
    
    Args:
        response: 模型原始输出
        parsed_output: 解析后的输出
        gt_bbox: 真实bbox
        gt_mask: 真实mask
        confused_regions: 混淆区域（可选）
        use_strict_format: 是否使用严格格式检查
        use_negative_reward: 是否使用负点奖励
        weights: 各奖励项权重
    
    Returns:
        {
            'total': float,           # 总奖励
            'think_format': float,    # 思考格式奖励
            'seg_format': float,      # 分割格式奖励
            'bbox_iou': float,        # bbox IoU奖励
            'bbox_l1': float,         # bbox L1奖励
            'point_l1': float,        # 正点L1奖励
            'negative_point': float,  # 负点奖励
        }
    """
    default_weights = {
        'think_format': 1.0,
        'seg_format': 1.0,
        'bbox_iou': 1.0,
        'bbox_l1': 1.0,
        'point_l1': 1.0,
        'negative_point': 1.0  # 新增负点奖励权重
    }
    
    if weights:
        default_weights.update(weights)
    weights = default_weights
    
    rewards = {}
    
    # 格式奖励
    rewards['think_format'] = compute_think_format_reward(response)
    
    if use_strict_format:
        rewards['seg_format'] = compute_seg_format_reward_strict(parsed_output)
    else:
        rewards['seg_format'] = compute_seg_format_reward_soft(parsed_output)
    
    # 精度奖励
    rewards['bbox_iou'] = compute_bbox_iou_reward(parsed_output['bbox'], gt_bbox)
    rewards['bbox_l1'] = compute_bbox_l1_reward(parsed_output['bbox'], gt_bbox)
    rewards['point_l1'] = compute_point_l1_reward(parsed_output['points_pos'], gt_mask)
    
    # 负点奖励
    if use_negative_reward:
        rewards['negative_point'] = compute_negative_point_reward(
            parsed_output['points_neg'],
            gt_mask,
            parsed_output['bbox'],
            confused_regions
        )
    else:
        rewards['negative_point'] = 0.0
    
    # 计算加权总奖励
    rewards['total'] = sum(
        rewards[key] * weights[key] 
        for key in weights.keys()
    )
    
    return rewards
```

### 3.2 奖励函数集成到GRPO训练

修改 `verl/trainer/fsdp_sft_trainer.py` 或创建新的reward wrapper:

```python
# verl/utils/reward_score/reward_manager.py

from typing import Dict, List
import numpy as np
from .output_parser import parse_seg_zero_output, prepare_sam_prompts
from .segmentation_rewards import compute_total_reward, identify_confused_regions

class SegZeroRewardManager:
    """
    Seg-Zero++ 奖励管理器
    管理奖励计算、SAM推理和混淆区域识别
    """
    
    def __init__(
        self,
        sam_model_path: str = "pretrained_models/sam2.1_hiera_large.pt",
        use_negative_reward: bool = True,
        use_confused_regions: bool = True,
        device: str = "cuda"
    ):
        self.use_negative_reward = use_negative_reward
        self.use_confused_regions = use_confused_regions
        self.device = device
        
        # 加载SAM2模型（用于计算mask和混淆区域）
        self._load_sam_model(sam_model_path)
    
    def _load_sam_model(self, model_path: str):
        """加载SAM2模型"""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.sam_model = build_sam2(
            "sam2_hiera_l.yaml",
            model_path,
            device=self.device
        )
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
    
    def compute_reward_batch(
        self,
        responses: List[str],
        images: List[np.ndarray],
        gt_bboxes: List[List[float]],
        gt_masks: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """
        批量计算奖励
        
        Args:
            responses: 模型输出列表
            images: 图像列表 (N, H, W, 3)
            gt_bboxes: GT bbox列表
            gt_masks: GT mask列表
        
        Returns:
            奖励字典列表
        """
        all_rewards = []
        
        for response, image, gt_bbox, gt_mask in zip(
            responses, images, gt_bboxes, gt_masks
        ):
            # 解析输出
            parsed = parse_seg_zero_output(response)
            
            # 计算混淆区域（可选）
            confused_regions = None
            if self.use_confused_regions and self.use_negative_reward:
                confused_regions = identify_confused_regions(
                    image, gt_mask, self.sam_predictor
                )
            
            # 计算奖励
            rewards = compute_total_reward(
                response=response,
                parsed_output=parsed,
                gt_bbox=gt_bbox,
                gt_mask=gt_mask,
                confused_regions=confused_regions,
                use_negative_reward=self.use_negative_reward
            )
            
            all_rewards.append(rewards)
        
        return all_rewards
    
    def compute_segmentation_mask(
        self,
        image: np.ndarray,
        parsed_output: Dict
    ) -> np.ndarray:
        """
        使用SAM2计算分割mask
        
        Args:
            image: 输入图像
            parsed_output: 解析后的模型输出
        
        Returns:
            分割mask
        """
        sam_prompts = prepare_sam_prompts(parsed_output)
        
        self.sam_predictor.set_image(image)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=sam_prompts['point_coords'],
            point_labels=sam_prompts['point_labels'],
            box=sam_prompts['box'],
            multimask_output=False
        )
        
        return masks[0]  # 返回最佳mask
```

---

## ⚙️ 第四步：修改训练配置

### 4.1 创建新的训练脚本

创建文件 `training_scripts/run_segzero_plus_7b.sh`:

```bash
#!/bin/bash

# Seg-Zero++ 训练脚本
# 基于Seg-Zero-7B checkpoint，添加负点预测

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="segzero-plus"

# 基础配置
BASE_MODEL="pretrained_models/Seg-Zero-7B"
OUTPUT_DIR="outputs/segzero_plus_7b"
DATA_DIR="data/refcocog_9k_840"

# 训练参数
BATCH_SIZE=2
GRAD_ACCUM=8
NUM_SAMPLES=8  # GRPO采样数
LR=1e-6
KL_COEF=5e-3
NUM_STEPS=300

# 新增：负点奖励配置
USE_NEGATIVE_REWARD=true
NEGATIVE_REWARD_WEIGHT=1.0

python -m verl.trainer.main \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="segzero_plus_negative_points" \
    trainer.total_training_steps=$NUM_STEPS \
    \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.prompt_key="prompt" \
    data.image_key="image" \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.fsdp.torch_dtype=bf16 \
    \
    actor_rollout_ref.rollout.n=$NUM_SAMPLES \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    custom.use_negative_reward=$USE_NEGATIVE_REWARD \
    custom.negative_reward_weight=$NEGATIVE_REWARD_WEIGHT \
    custom.use_strict_format=true \
    custom.prompt_template="negative_point" \
    \
    trainer.save_freq=50 \
    trainer.save_path=$OUTPUT_DIR
```

### 4.2 显存优化配置（2×A6000）

```bash
# 针对2×A6000 48GB优化的配置
# training_scripts/run_segzero_plus_7b_2xa6000.sh

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# 显存优化参数
MICRO_BATCH=1
GRAD_ACCUM=16  # 有效batch size = 1 * 16 * 2 = 32
NUM_SAMPLES=4   # 减少GRPO采样数以节省显存
GPU_MEM_UTIL=0.90

python -m verl.trainer.main \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
    actor_rollout_ref.actor.gradient_checkpointing=true \
    actor_rollout_ref.rollout.n=$NUM_SAMPLES \
    actor_rollout_ref.rollout.tensor_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    # ... 其他参数同上
```

---

## 📊 第五步：数据预处理

### 5.1 生成混淆区域标注

创建文件 `prepare_dataset/generate_confused_regions.py`:

```python
"""
预计算混淆区域，加速训练
"""

import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyarrow.parquet as pq
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def generate_confused_regions_dataset(
    input_parquet: str,
    output_dir: str,
    sam_model_path: str,
    num_samples: int = 5
):
    """
    为数据集预计算混淆区域
    """
    # 加载SAM模型
    sam_model = build_sam2("sam2_hiera_l.yaml", sam_model_path, device="cuda")
    sam_predictor = SAM2ImagePredictor(sam_model)
    
    # 读取数据
    df = pq.read_table(input_parquet).to_pandas()
    
    os.makedirs(output_dir, exist_ok=True)
    
    confused_regions_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image = np.array(Image.open(row['image_path']))
        gt_mask = np.array(Image.open(row['mask_path']))
        
        # 计算混淆区域
        confused = identify_confused_regions_fast(
            image, gt_mask, sam_predictor, num_samples
        )
        
        # 保存混淆区域mask
        confused_path = os.path.join(output_dir, f"confused_{idx:06d}.npy")
        np.save(confused_path, confused)
        
        confused_regions_data.append({
            'image_id': row.get('image_id', idx),
            'confused_region_path': confused_path
        })
    
    # 保存索引
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(confused_regions_data, f)
    
    print(f"Generated confused regions for {len(df)} samples")


def identify_confused_regions_fast(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam_predictor,
    num_samples: int = 5
) -> np.ndarray:
    """快速版混淆区域识别"""
    h, w = gt_mask.shape[:2]
    confused = np.zeros((h, w), dtype=np.float32)
    
    # 获取mask边界
    mask_coords = np.argwhere(gt_mask > 0)
    if len(mask_coords) == 0:
        return confused
    
    y_min, x_min = mask_coords.min(axis=0)
    y_max, x_max = mask_coords.max(axis=0)
    
    # 采样区域
    margin = 80
    x_range = (max(0, x_min - margin), min(w, x_max + margin))
    y_range = (max(0, y_min - margin), min(h, y_max + margin))
    
    sam_predictor.set_image(image)
    
    for _ in range(num_samples):
        # 在mask外采样
        for _ in range(5):
            x = np.random.randint(x_range[0], x_range[1])
            y = np.random.randint(y_range[0], y_range[1])
            if gt_mask[y, x] == 0:
                break
        else:
            continue
        
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        for mask, score in zip(masks, scores):
            if score > 0.5:
                non_gt = mask & (gt_mask == 0)
                confused += non_gt.astype(np.float32)
    
    if confused.max() > 0:
        confused /= confused.max()
    
    return confused


if __name__ == "__main__":
    generate_confused_regions_dataset(
        input_parquet="data/refcocog_9k_840/train.parquet",
        output_dir="data/refcocog_9k_840/confused_regions",
        sam_model_path="pretrained_models/sam2.1_hiera_large.pt"
    )
```

---

## 🧪 第六步：评估脚本

### 6.1 创建评估脚本

创建文件 `evaluation_scripts/eval_segzero_plus.py`:

```python
"""
Seg-Zero++ 评估脚本
支持ReasonSeg和RefCOCO评估
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.reward_score.output_parser import parse_seg_zero_output, prepare_sam_prompts
from verl.utils.reward_score.prompt_templates import NEGATIVE_POINT_PROMPT


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def compute_giou(pred_masks: list, gt_masks: list) -> float:
    """计算gIoU (平均IoU)"""
    ious = [compute_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return np.mean(ious)


def compute_ciou(pred_masks: list, gt_masks: list) -> float:
    """计算cIoU (累积IoU)"""
    total_intersection = sum(
        np.logical_and(p, g).sum() for p, g in zip(pred_masks, gt_masks)
    )
    total_union = sum(
        np.logical_or(p, g).sum() for p, g in zip(pred_masks, gt_masks)
    )
    return total_intersection / total_union if total_union > 0 else 0.0


class SegZeroPlusEvaluator:
    def __init__(
        self,
        model_path: str,
        sam_model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        # 加载Qwen2.5-VL模型
        print(f"Loading model from {model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 加载SAM2模型
        print(f"Loading SAM2 from {sam_model_path}")
        self.sam_model = build_sam2("sam2_hiera_l.yaml", sam_model_path, device=device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
    
    def generate_response(self, image: Image.Image, query: str) -> str:
        """生成模型响应"""
        prompt = NEGATIVE_POINT_PROMPT.format(Question=query)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        response = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response
    
    def predict_mask(self, image: np.ndarray, response: str) -> np.ndarray:
        """根据模型响应生成分割mask"""
        parsed = parse_seg_zero_output(response)
        
        if not parsed['format_valid']:
            # 返回空mask
            return np.zeros(image.shape[:2], dtype=bool)
        
        sam_prompts = prepare_sam_prompts(parsed)
        
        self.sam_predictor.set_image(image)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=sam_prompts['point_coords'],
            point_labels=sam_prompts['point_labels'],
            box=sam_prompts['box'],
            multimask_output=False
        )
        
        return masks[0]
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_path: str = None,
        max_samples: int = None
    ) -> dict:
        """评估数据集"""
        # 加载数据
        with open(data_path) as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        pred_masks = []
        gt_masks = []
        results = []
        
        for item in tqdm(data, desc="Evaluating"):
            # 加载图像
            image = Image.open(item['image_path']).convert('RGB')
            image_np = np.array(image)
            
            # 加载GT mask
            gt_mask = np.array(Image.open(item['mask_path'])) > 0
            
            # 生成响应
            response = self.generate_response(image, item['query'])
            
            # 预测mask
            pred_mask = self.predict_mask(image_np, response)
            
            # 计算IoU
            iou = compute_iou(pred_mask, gt_mask)
            
            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)
            
            results.append({
                'image_id': item.get('image_id', ''),
                'query': item['query'],
                'iou': float(iou),
                'response': response
            })
        
        # 计算整体指标
        giou = compute_giou(pred_masks, gt_masks)
        ciou = compute_ciou(pred_masks, gt_masks)
        
        metrics = {
            'gIoU': float(giou),
            'cIoU': float(ciou),
            'num_samples': len(data)
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"gIoU: {giou:.4f}")
        print(f"cIoU: {ciou:.4f}")
        print(f"Samples: {len(data)}")
        
        # 保存结果
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sam_model_path', type=str, 
                        default='pretrained_models/sam2.1_hiera_large.pt')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    evaluator = SegZeroPlusEvaluator(
        model_path=args.model_path,
        sam_model_path=args.sam_model_path
    )
    
    evaluator.evaluate_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
```

### 6.2 评估脚本Shell wrapper

创建文件 `evaluation_scripts/eval_reasonseg_segzero_plus.sh`:

```bash
#!/bin/bash

MODEL_PATH="outputs/segzero_plus_7b/checkpoint-300"
SAM_PATH="pretrained_models/sam2.1_hiera_large.pt"
DATA_PATH="data/ReasonSeg/test.json"
OUTPUT_PATH="results/reasonseg_test_results.json"

python evaluation_scripts/eval_segzero_plus.py \
    --model_path $MODEL_PATH \
    --sam_model_path $SAM_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH
```

---

## 📈 第七步：消融实验设计

### 7.1 消融实验配置

```python
# experiments/ablation_configs.py

ABLATION_EXPERIMENTS = {
    # 基线：原始Seg-Zero
    "baseline": {
        "use_negative_reward": False,
        "use_confused_regions": False,
        "prompt_template": "original"
    },
    
    # 消融1：仅添加负点输出（无负点奖励）
    "neg_output_only": {
        "use_negative_reward": False,
        "use_confused_regions": False,
        "prompt_template": "negative_point"
    },
    
    # 消融2：负点+简单奖励（无混淆区域）
    "neg_simple_reward": {
        "use_negative_reward": True,
        "use_confused_regions": False,
        "prompt_template": "negative_point",
        "negative_reward_weight": 1.0
    },
    
    # 消融3：负点+完整对比奖励
    "neg_full_reward": {
        "use_negative_reward": True,
        "use_confused_regions": True,
        "prompt_template": "negative_point",
        "negative_reward_weight": 1.0
    },
    
    # 消融4：不同负点数量
    "neg_1_point": {"max_negative_points": 1},
    "neg_2_points": {"max_negative_points": 2},
    "neg_3_points": {"max_negative_points": 3},
    
    # 消融5：不同奖励权重
    "neg_weight_0.5": {"negative_reward_weight": 0.5},
    "neg_weight_1.0": {"negative_reward_weight": 1.0},
    "neg_weight_2.0": {"negative_reward_weight": 2.0},
}
```

### 7.2 预期结果表格模板

```markdown
| Method | RefCOCOg | ReasonSeg | 
|--------|----------|-----------|
| Baseline (Seg-Zero-7B) | 74.2 | 57.5 |
| + Neg Output Only | ~74.5 | ~58.0 |
| + Simple Neg Reward | ~75.0 | ~59.0 |
| + Full Contrastive Reward | **76.0+** | **60.5+** |
```

---

## ⏰ 第八步：时间线规划

```
Week 1-2: 环境搭建 & 代码修改
├── Day 1-2: 环境配置、模型下载
├── Day 3-5: 修改输出格式和解析器
├── Day 6-10: 实现奖励函数
└── Day 11-14: 集成到训练流程

Week 3-4: 初步实验
├── Day 15-18: 小规模验证实验（1K samples）
├── Day 19-21: 调试和修复bug
└── Day 22-28: 消融实验（负点数量、权重）

Week 5-6: 主实验
├── Day 29-35: 完整训练（9K samples, ~3天）
├── Day 36-38: 评估ReasonSeg
└── Day 39-42: 评估RefCOCO系列

Week 7-8: 分析与论文
├── Day 43-49: 结果分析、可视化
└── Day 50-56: 论文撰写
```

---

## 🔧 常见问题与解决方案

### Q1: 显存不足
```bash
# 解决方案：减少batch size和采样数
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.gradient_checkpointing=true
```

### Q2: 负点总是预测在mask内部
- 增加惩罚系数 alpha
- 检查混淆区域生成是否正常
- 考虑添加hard constraint

### Q3: 训练不稳定
- 降低学习率到5e-7
- 增加KL系数到1e-2
- 使用更保守的clip范围

### Q4: 格式奖励不收敛
- 检查prompt模板是否包含足够的示例
- 确保解析器能正确处理各种格式变体
- 考虑先用SFT预热几步

---

## 📚 参考资料

1. Seg-Zero Paper: https://arxiv.org/abs/2503.06520
2. SAM2 Documentation: https://github.com/facebookresearch/sam2
3. veRL Framework: https://github.com/volcengine/verl
4. GRPO Algorithm: https://arxiv.org/abs/2402.03300
