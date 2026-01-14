# src/utils/reward_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def visualize_reward_components(
    image: np.ndarray,
    gt_mask: np.ndarray,
    baseline_mask: np.ndarray,
    final_mask: np.ndarray,
    positive_points: List[Tuple[float, float]],
    negative_points: List[Tuple[float, float]],
    save_path: str = None
):
    """
    可视化奖励计算的各个组件
    
    用于调试和验证奖励函数的正确性
    """
    H, W = image.shape[:2]
    
    # 计算混淆区域
    confused_region = baseline_mask.astype(bool) & ~gt_mask.astype(bool)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原图 + 点标注
    axes[0, 0].imshow(image)
    for x, y in positive_points:
        axes[0, 0].scatter(x * W, y * H, c='green', s=100, marker='o', label='Positive')
    for x, y in negative_points:
        axes[0, 0].scatter(x * W, y * H, c='red', s=100, marker='x', label='Negative')
    axes[0, 0].set_title('Input Image + Points')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth Mask
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    # 3. 基线Mask（只用正点）
    axes[0, 2].imshow(baseline_mask, cmap='gray')
    axes[0, 2].set_title('Baseline Mask (Pos Only)')
    axes[0, 2].axis('off')
    
    # 4. 混淆区域（False Positive）
    confused_vis = np.zeros((H, W, 3), dtype=np.uint8)
    confused_vis[confused_region] = [255, 0, 0]  # 红色
    confused_vis[gt_mask.astype(bool)] = [0, 255, 0]  # 绿色
    axes[1, 0].imshow(confused_vis)
    axes[1, 0].set_title('Confused Region (Red) vs GT (Green)')
    axes[1, 0].axis('off')
    
    # 5. 最终Mask（使用负点）
    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title('Final Mask (With Neg Points)')
    axes[1, 1].axis('off')
    
    # 6. IoU对比
    baseline_iou = compute_iou(baseline_mask, gt_mask)
    final_iou = compute_iou(final_mask, gt_mask)
    
    axes[1, 2].bar(['Baseline', 'Final'], [baseline_iou, final_iou], color=['orange', 'green'])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title(f'IoU Comparison\nBaseline: {baseline_iou:.3f} -> Final: {final_iou:.3f}')
    axes[1, 2].set_ylabel('IoU')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / union if union > 0 else 0.0