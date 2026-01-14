# src/train/reward_functions.py

import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class RewardOutput:
    """奖励计算结果"""
    total_reward: float
    mask_reward: float
    negative_point_reward: float
    format_reward: float
    details: Dict

class NegativePointRewardCalculator:
    """负点对比奖励计算器"""
    
    def __init__(
        self,
        sam_wrapper,
        alpha: float = 1.0,      # 惩罚系数：负点落在GT内
        beta: float = 1.0,       # 奖励系数：负点落在混淆区域
        lambda_neg: float = 0.3, # 负点奖励权重
        lambda_format: float = 0.1  # 格式奖励权重
    ):
        """
        Args:
            sam_wrapper: SAM2Wrapper实例
            alpha: 负点落在GT内的惩罚系数
            beta: 负点落在混淆区域的奖励系数
            lambda_neg: 负点奖励在总奖励中的权重
            lambda_format: 格式奖励在总奖励中的权重
        """
        self.sam_wrapper = sam_wrapper
        self.alpha = alpha
        self.beta = beta
        self.lambda_neg = lambda_neg
        self.lambda_format = lambda_format
    
    def compute_reward(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        gt_mask: np.ndarray,
        format_valid: bool
    ) -> RewardOutput:
        """
        计算总奖励
        
        Args:
            image: 输入图像 (H, W, 3)
            positive_points: 正点列表
            negative_points: 负点列表
            bbox: 边界框
            gt_mask: Ground Truth mask (H, W)
            format_valid: 输出格式是否有效
            
        Returns:
            RewardOutput对象
        """
        H, W = image.shape[:2]
        
        # 1. 计算格式奖励
        format_reward = 1.0 if format_valid else 0.0
        
        if not format_valid:
            # 格式无效，返回最低奖励
            return RewardOutput(
                total_reward=0.0,
                mask_reward=0.0,
                negative_point_reward=0.0,
                format_reward=0.0,
                details={"error": "invalid_format"}
            )
        
        # 2. 使用正负点生成最终mask
        final_mask, confidence = self.sam_wrapper.predict_mask(
            image, positive_points, negative_points, bbox
        )
        
        # 3. 计算mask IoU奖励
        mask_reward = self._compute_iou(final_mask, gt_mask)
        
        # 4. 计算负点对比奖励
        neg_reward, neg_details = self._compute_negative_point_reward(
            image, positive_points, negative_points, bbox, gt_mask
        )
        
        # 5. 计算总奖励
        total_reward = (
            mask_reward +
            self.lambda_neg * neg_reward +
            self.lambda_format * format_reward
        )
        
        return RewardOutput(
            total_reward=total_reward,
            mask_reward=mask_reward,
            negative_point_reward=neg_reward,
            format_reward=format_reward,
            details={
                "iou": mask_reward,
                "sam_confidence": confidence,
                **neg_details
            }
        )
    
    def _compute_negative_point_reward(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        gt_mask: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        计算负点对比奖励
        
        核心逻辑:
        1. 用正点生成基线mask
        2. 混淆区域 = 基线mask中的False Positive区域
        3. 奖励负点落在混淆区域，惩罚负点落在GT内
        """
        H, W = gt_mask.shape
        
        if not negative_points:
            return 0.0, {"neg_in_gt": 0, "neg_in_confused": 0, "confused_area": 0}
        
        # 1. 生成基线mask（只用正点，不用负点）
        baseline_mask = self.sam_wrapper.predict_baseline_mask(
            image, positive_points, bbox
        )
        
        # 2. 计算混淆区域（False Positive）
        confused_region = baseline_mask.astype(bool) & ~gt_mask.astype(bool)
        confused_area = confused_region.sum()
        
        # 3. 统计负点位置
        neg_in_gt = 0
        neg_in_confused = 0
        
        for nx, ny in negative_points:
            px, py = int(nx * W), int(ny * H)
            px = min(max(px, 0), W - 1)
            py = min(max(py, 0), H - 1)
            
            if gt_mask[py, px]:
                neg_in_gt += 1
            if confused_region[py, px]:
                neg_in_confused += 1
        
        # 4. 计算奖励
        num_neg = len(negative_points)
        
        # 归一化：负点落在GT内的比例（惩罚），负点落在混淆区域的比例（奖励）
        penalty = (neg_in_gt / num_neg) if num_neg > 0 else 0
        bonus = (neg_in_confused / num_neg) if num_neg > 0 else 0
        
        reward = -self.alpha * penalty + self.beta * bonus
        
        details = {
            "neg_in_gt": neg_in_gt,
            "neg_in_confused": neg_in_confused,
            "num_negative_points": num_neg,
            "confused_area_pixels": int(confused_area),
            "confused_area_ratio": float(confused_area / (H * W))
        }
        
        return reward, details
    
    def _compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """计算IoU"""
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)