# src/eval/metrics.py

import numpy as np
from typing import Dict, List

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算IoU"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    return float(intersection / union) if union > 0 else 0.0

def compute_giou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算Generalized IoU (gIoU)"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    
    # 计算最小包围框
    pred_indices = np.where(pred)
    gt_indices = np.where(gt)
    
    if len(pred_indices[0]) == 0 or len(gt_indices[0]) == 0:
        return iou
    
    all_y = np.concatenate([pred_indices[0], gt_indices[0]])
    all_x = np.concatenate([pred_indices[1], gt_indices[1]])
    
    enclosing_box_area = (all_y.max() - all_y.min() + 1) * (all_x.max() - all_x.min() + 1)
    
    giou = iou - (enclosing_box_area - union) / enclosing_box_area
    
    return float(giou)

def compute_boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation: int = 5) -> float:
    """计算Boundary IoU（仅在边界区域计算）"""
    from scipy import ndimage
    
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # 计算边界
    pred_boundary = pred ^ ndimage.binary_erosion(pred, iterations=dilation)
    gt_boundary = gt ^ ndimage.binary_erosion(gt, iterations=dilation)
    
    # 在边界区域计算IoU
    boundary_region = pred_boundary | gt_boundary
    
    if boundary_region.sum() == 0:
        return 1.0
    
    intersection = (pred & gt & boundary_region).sum()
    union = ((pred | gt) & boundary_region).sum()
    
    return float(intersection / union) if union > 0 else 0.0

class SegmentationMetrics:
    """分割指标汇总"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.ious = []
        self.gious = []
        self.boundary_ious = []
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        self.ious.append(compute_iou(pred, gt))
        self.gious.append(compute_giou(pred, gt))
        self.boundary_ious.append(compute_boundary_iou(pred, gt))
    
    def compute(self) -> Dict[str, float]:
        return {
            "mIoU": np.mean(self.ious) * 100,
            "gIoU": np.mean(self.gious) * 100,
            "Boundary_IoU": np.mean(self.boundary_ious) * 100,
            "num_samples": len(self.ious)
        }