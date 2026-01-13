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