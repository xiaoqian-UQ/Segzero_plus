"""
Seg-Zero++ Reward Function Module
集成到 veRL 框架的奖励函数实现

使用方法:
1. 将此文件放在 verl/utils/reward_score/ 目录下
2. 在训练配置中指定 custom.reward_function="segzero_plus"
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class SegZeroPlusConfig:
    """负点奖励配置"""
    use_negative_reward: bool = True
    negative_reward_weight: float = 1.0
    use_confused_regions: bool = True
    negative_alpha: float = 1.0  # 惩罚系数
    negative_beta: float = 0.5   # 奖励系数
    max_negative_points: int = 2
    use_strict_format: bool = True
    image_size: int = 840


#===============================================================================
# 输出解析器
#===============================================================================

def parse_model_output(response: str) -> Dict[str, Any]:
    """
    解析Seg-Zero++模型输出
    
    支持的格式:
    <think>reasoning</think>
    <answer>{"bbox": [x1,y1,x2,y2], "points_pos": [[x,y],...], "points_neg": [[x,y],...]}</answer>
    
    Returns:
        {
            'think': str,
            'bbox': [x1,y1,x2,y2] or None,
            'points_pos': [[x,y], ...],
            'points_neg': [[x,y], ...],
            'format_valid': bool
        }
    """
    result = {
        'think': '',
        'bbox': None,
        'points_pos': [],
        'points_neg': [],
        'format_valid': False
    }
    
    # 提取think
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        result['think'] = think_match.group(1).strip()
    
    # 提取answer
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return result
    
    answer_text = answer_match.group(1).strip()
    
    try:
        # 处理单引号
        answer_text = answer_text.replace("'", '"')
        # 处理可能的格式问题
        answer_text = re.sub(r'(\w+):', r'"\1":', answer_text)
        
        data = json.loads(answer_text)
        
        # 解析bbox
        if 'bbox' in data and len(data['bbox']) == 4:
            result['bbox'] = [float(x) for x in data['bbox']]
        
        # 解析正点
        if 'points_pos' in data:
            for pt in data['points_pos']:
                if len(pt) == 2:
                    result['points_pos'].append([float(pt[0]), float(pt[1])])
        elif 'points_1' in data and 'points_2' in data:
            # 兼容旧格式
            result['points_pos'].append([float(data['points_1'][0]), float(data['points_1'][1])])
            result['points_pos'].append([float(data['points_2'][0]), float(data['points_2'][1])])
        
        # 解析负点
        if 'points_neg' in data:
            for pt in data['points_neg']:
                if len(pt) == 2:
                    result['points_neg'].append([float(pt[0]), float(pt[1])])
        
        # 验证格式
        result['format_valid'] = (
            result['bbox'] is not None and
            len(result['points_pos']) >= 1
        )
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # JSON解析失败，尝试正则
        bbox_match = re.search(
            r'"?bbox"?\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
            answer_text
        )
        if bbox_match:
            result['bbox'] = [float(x) for x in bbox_match.groups()]
    
    return result


#===============================================================================
# 格式奖励
#===============================================================================

def reward_think_format(response: str) -> float:
    """思考格式奖励"""
    pattern = r'<think>.*?</think>'
    return 1.0 if re.search(pattern, response, re.DOTALL) else 0.0


def reward_seg_format_soft(parsed: Dict) -> float:
    """软格式奖励"""
    has_bbox = parsed['bbox'] is not None and len(parsed['bbox']) == 4
    has_points = len(parsed['points_pos']) >= 1
    return 1.0 if (has_bbox and has_points) else 0.0


def reward_seg_format_strict(parsed: Dict, config: SegZeroPlusConfig) -> float:
    """严格格式奖励"""
    if parsed['bbox'] is None or len(parsed['bbox']) != 4:
        return 0.0
    
    if len(parsed['points_pos']) < 2:
        return 0.0
    
    # 验证坐标范围
    img_size = config.image_size
    try:
        for x in parsed['bbox']:
            if not (0 <= x <= img_size):
                return 0.0
        
        for pt in parsed['points_pos'] + parsed['points_neg']:
            if not (0 <= pt[0] <= img_size and 0 <= pt[1] <= img_size):
                return 0.0
    except (TypeError, IndexError):
        return 0.0
    
    return 1.0


#===============================================================================
# 精度奖励
#===============================================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def reward_bbox_iou(pred_bbox: List[float], gt_bbox: List[float], threshold: float = 0.5) -> float:
    """Bbox IoU奖励 (硬)"""
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    iou = compute_iou(pred_bbox, gt_bbox)
    return 1.0 if iou > threshold else 0.0


def reward_bbox_l1(pred_bbox: List[float], gt_bbox: List[float], threshold: float = 10.0) -> float:
    """Bbox L1奖励"""
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    l1 = sum(abs(p - g) for p, g in zip(pred_bbox, gt_bbox)) / 4.0
    return 1.0 if l1 < threshold else 0.0


def reward_points_l1(
    pred_points: List[List[float]], 
    gt_mask: np.ndarray,
    threshold: float = 100.0
) -> float:
    """正点L1奖励 - 检查点是否在mask内或附近"""
    if not pred_points or gt_mask is None:
        return 0.0
    
    h, w = gt_mask.shape
    
    for pt in pred_points:
        x, y = int(pt[0]), int(pt[1])
        
        # 边界检查
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        # 在mask内
        if gt_mask[y, x] > 0:
            continue
        
        # 计算到mask的最小距离
        mask_coords = np.argwhere(gt_mask > 0)
        if len(mask_coords) == 0:
            return 0.0
        
        distances = np.sqrt((mask_coords[:, 1] - x)**2 + (mask_coords[:, 0] - y)**2)
        if distances.min() > threshold:
            return 0.0
    
    return 1.0


#===============================================================================
# 负点对比奖励 (核心创新)
#===============================================================================

def reward_negative_points(
    pred_neg_points: List[List[float]],
    gt_mask: np.ndarray,
    pred_bbox: Optional[List[float]],
    confused_regions: Optional[np.ndarray],
    config: SegZeroPlusConfig
) -> float:
    """
    负点对比奖励
    
    原则:
    1. 负点不应落在GT mask内 (惩罚)
    2. 负点应落在混淆区域 (奖励)
    3. 负点应在目标附近但不在mask内 (奖励)
    """
    if not pred_neg_points:
        return 0.5  # 没有负点，给基础分
    
    if gt_mask is None:
        return 0.0
    
    h, w = gt_mask.shape
    alpha = config.negative_alpha
    beta = config.negative_beta
    
    total_reward = 0.0
    valid_count = 0
    
    for pt in pred_neg_points[:config.max_negative_points]:
        x, y = int(pt[0]), int(pt[1])
        
        # 边界检查
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        valid_count += 1
        point_reward = 0.0
        
        # 惩罚: 负点在GT mask内
        if gt_mask[y, x] > 0:
            point_reward -= alpha
        else:
            # 奖励: 负点在mask外
            point_reward += 0.3
        
        # 奖励: 负点在混淆区域
        if confused_regions is not None and confused_regions[y, x] > 0:
            point_reward += beta
        
        # 奖励: 负点在bbox附近
        if pred_bbox is not None:
            margin = 50
            extended = [
                max(0, pred_bbox[0] - margin),
                max(0, pred_bbox[1] - margin),
                min(w, pred_bbox[2] + margin),
                min(h, pred_bbox[3] + margin)
            ]
            if extended[0] <= x <= extended[2] and extended[1] <= y <= extended[3]:
                point_reward += 0.2
        
        total_reward += point_reward
    
    if valid_count == 0:
        return 0.0
    
    # 归一化到 [0, 1]
    avg = total_reward / valid_count
    normalized = (avg + alpha) / (alpha + 0.5 + beta)
    
    return max(0.0, min(1.0, normalized))


#===============================================================================
# 总奖励计算
#===============================================================================

def compute_segzero_plus_reward(
    response: str,
    gt_bbox: List[float],
    gt_mask: np.ndarray,
    confused_regions: Optional[np.ndarray] = None,
    config: Optional[SegZeroPlusConfig] = None
) -> Dict[str, float]:
    """
    计算Seg-Zero++总奖励
    
    Args:
        response: 模型输出文本
        gt_bbox: 真实bbox [x1,y1,x2,y2]
        gt_mask: 真实mask (H,W)
        confused_regions: 混淆区域mask (可选)
        config: 奖励配置
    
    Returns:
        {
            'total': float,
            'think_format': float,
            'seg_format': float,
            'bbox_iou': float,
            'bbox_l1': float,
            'point_l1': float,
            'negative_point': float
        }
    """
    if config is None:
        config = SegZeroPlusConfig()
    
    # 解析输出
    parsed = parse_model_output(response)
    
    rewards = {}
    
    # 格式奖励
    rewards['think_format'] = reward_think_format(response)
    
    if config.use_strict_format:
        rewards['seg_format'] = reward_seg_format_strict(parsed, config)
    else:
        rewards['seg_format'] = reward_seg_format_soft(parsed)
    
    # 精度奖励
    rewards['bbox_iou'] = reward_bbox_iou(parsed['bbox'], gt_bbox)
    rewards['bbox_l1'] = reward_bbox_l1(parsed['bbox'], gt_bbox)
    rewards['point_l1'] = reward_points_l1(parsed['points_pos'], gt_mask)
    
    # 负点奖励
    if config.use_negative_reward:
        rewards['negative_point'] = reward_negative_points(
            parsed['points_neg'],
            gt_mask,
            parsed['bbox'],
            confused_regions,
            config
        )
    else:
        rewards['negative_point'] = 0.0
    
    # 计算总奖励 (等权重)
    weights = {
        'think_format': 1.0,
        'seg_format': 1.0,
        'bbox_iou': 1.0,
        'bbox_l1': 1.0,
        'point_l1': 1.0,
        'negative_point': config.negative_reward_weight
    }
    
    rewards['total'] = sum(rewards[k] * weights[k] for k in weights)
    
    return rewards


#===============================================================================
# veRL集成接口
#===============================================================================

class SegZeroPlusRewardFunction:
    """
    veRL兼容的奖励函数类
    
    在veRL配置中使用:
    custom.reward_function="segzero_plus"
    """
    
    def __init__(self, config_dict: Dict = None):
        if config_dict:
            self.config = SegZeroPlusConfig(**config_dict)
        else:
            self.config = SegZeroPlusConfig()
        
        self.sam_predictor = None
        self.confused_regions_cache = {}
    
    def __call__(
        self,
        responses: List[str],
        batch_data: Dict[str, Any]
    ) -> List[float]:
        """
        计算一个batch的奖励
        
        Args:
            responses: 模型输出列表
            batch_data: 包含gt_bbox, gt_mask等的字典
        
        Returns:
            奖励分数列表
        """
        rewards = []
        
        gt_bboxes = batch_data.get('gt_bbox', [None] * len(responses))
        gt_masks = batch_data.get('gt_mask', [None] * len(responses))
        confused_regions = batch_data.get('confused_regions', [None] * len(responses))
        
        for resp, gt_box, gt_mask, confused in zip(
            responses, gt_bboxes, gt_masks, confused_regions
        ):
            reward_dict = compute_segzero_plus_reward(
                response=resp,
                gt_bbox=gt_box,
                gt_mask=gt_mask,
                confused_regions=confused,
                config=self.config
            )
            rewards.append(reward_dict['total'])
        
        return rewards


# 导出接口
def get_reward_function(config: Dict = None) -> SegZeroPlusRewardFunction:
    """获取奖励函数实例"""
    return SegZeroPlusRewardFunction(config)


#===============================================================================
# 测试代码
#===============================================================================

if __name__ == "__main__":
    # 测试解析
    test_response = """
<think>
The query asks for "the red car". There are two cars in the image.
One is red (target) and one is blue (background).
I will mark positive points on the red car and negative points on the blue car.
</think>
<answer>{"bbox": [100, 150, 300, 350], "points_pos": [[200, 250], [180, 280]], "points_neg": [[450, 260]]}</answer>
"""
    
    parsed = parse_model_output(test_response)
    print("Parsed output:")
    print(f"  bbox: {parsed['bbox']}")
    print(f"  points_pos: {parsed['points_pos']}")
    print(f"  points_neg: {parsed['points_neg']}")
    print(f"  format_valid: {parsed['format_valid']}")
    
    # 测试奖励计算
    gt_bbox = [100, 150, 300, 350]
    gt_mask = np.zeros((840, 840), dtype=np.uint8)
    gt_mask[150:350, 100:300] = 1
    
    rewards = compute_segzero_plus_reward(
        response=test_response,
        gt_bbox=gt_bbox,
        gt_mask=gt_mask
    )
    
    print("\nRewards:")
    for k, v in rewards.items():
        print(f"  {k}: {v:.4f}")