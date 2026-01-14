import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Optional, Dict
import cv2


def generate_negative_points_heuristic(
    mask: np.ndarray,
    bbox: List[int],
    num_points: int = 2,
    margin: int = 50,
    seed: Optional[int] = None
) -> List[List[int]]:
    """
    启发式方法生成负点真值
    
    策略: 在bbox扩展区域内、但mask外部选择点
    优先选择靠近mask边界的点（更容易混淆的区域）
    
    Args:
        mask: 二值mask, shape (H, W)
        bbox: [x1, y1, x2, y2]
        num_points: 生成的负点数量
        margin: bbox扩展的边界距离
        seed: 随机种子
        
    Returns:
        负点列表 [[x1,y1], [x2,y2], ...]
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    
    # 扩展 bbox 区域
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)
    
    # 创建候选区域: 扩展bbox内但mask外
    candidate_mask = np.zeros((h, w), dtype=bool)
    candidate_mask[ey1:ey2, ex1:ex2] = True
    candidate_mask[mask > 0] = False  # 排除正样本区域
    
    # 计算到mask边界的距离，优先选择靠近边界的点
    dist_to_mask = distance_transform_edt(mask == 0)
    
    # 只考虑候选区域内的点
    dist_to_mask[~candidate_mask] = np.inf
    
    # 获取候选点坐标
    candidates_y, candidates_x = np.where(candidate_mask)
    
    if len(candidates_y) == 0:
        return []
    
    # 按距离排序，选择距离mask边界近的点（更容易混淆）
    distances = dist_to_mask[candidates_y, candidates_x]
    
    # 过滤掉距离太近（在mask边界上）和太远的点
    valid_mask = (distances > 5) & (distances < margin)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # 如果没有合适的点，放宽限制
        valid_indices = np.arange(len(candidates_y))
    
    # 按距离排序
    sorted_indices = valid_indices[np.argsort(distances[valid_indices])]
    
    # 选择点，确保点之间有一定距离
    negative_points = []
    min_dist_between_points = 20
    
    for idx in sorted_indices:
        if len(negative_points) >= num_points:
            break
            
        x, y = candidates_x[idx], candidates_y[idx]
        
        # 检查与已选点的距离
        too_close = False
        for px, py in negative_points:
            if np.sqrt((x - px)**2 + (y - py)**2) < min_dist_between_points:
                too_close = True
                break
        
        if not too_close:
            negative_points.append([int(x), int(y)])
    
    # 如果点不够，随机采样补充
    if len(negative_points) < num_points and len(valid_indices) > 0:
        remaining = num_points - len(negative_points)
        random_indices = np.random.choice(
            valid_indices, 
            min(remaining * 3, len(valid_indices)), 
            replace=False
        )
        for idx in random_indices:
            if len(negative_points) >= num_points:
                break
            x, y = candidates_x[idx], candidates_y[idx]
            if [int(x), int(y)] not in negative_points:
                negative_points.append([int(x), int(y)])
    
    return negative_points


def generate_negative_points_edge_based(
    mask: np.ndarray,
    bbox: List[int],
    num_points: int = 2,
    edge_margin: int = 30,
    seed: Optional[int] = None
) -> List[List[int]]:
    """
    基于边缘的负点生成方法
    
    策略: 在mask边缘外侧选择点，这些位置SAM最容易错误包含
    
    Args:
        mask: 二值mask
        bbox: [x1, y1, x2, y2]
        num_points: 生成的负点数量
        edge_margin: 距离mask边缘的范围
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = mask.shape
    
    # 找到mask边缘
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    edge = dilated - eroded
    
    # 在边缘外侧创建候选区域
    outer_region = cv2.dilate(mask.astype(np.uint8), kernel, iterations=edge_margin//3)
    candidate_mask = (outer_region > 0) & (mask == 0)
    
    candidates_y, candidates_x = np.where(candidate_mask)
    
    if len(candidates_y) == 0:
        return generate_negative_points_heuristic(mask, bbox, num_points, seed=seed)
    
    # 随机采样
    indices = np.random.choice(len(candidates_y), min(num_points, len(candidates_y)), replace=False)
    
    negative_points = [[int(candidates_x[i]), int(candidates_y[i])] for i in indices]
    
    return negative_points


def generate_negative_points_sam_ambiguity(
    image: np.ndarray,
    bbox: List[int],
    positive_points: List[List[int]],
    gt_mask: np.ndarray,
    sam_predictor,
    num_points: int = 2,
    seed: Optional[int] = None
) -> List[List[int]]:
    """
    使用SAM2的多mask输出识别混淆区域
    
    这是更精准的方法，找到SAM可能混淆的区域
    
    Args:
        image: RGB图像
        bbox: [x1, y1, x2, y2]
        positive_points: 正点列表
        gt_mask: ground truth mask
        sam_predictor: SAM2 predictor实例
        num_points: 生成的负点数量
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 设置图像
    sam_predictor.set_image(image)
    
    # 准备输入
    input_box = np.array(bbox)
    input_points = np.array(positive_points)
    input_labels = np.array([1] * len(positive_points))
    
    # 请求多个mask
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=True
    )
    
    # 找到与真值不同的区域（混淆区域）
    best_mask_idx = np.argmax(scores)
    confused_regions = np.zeros_like(gt_mask, dtype=bool)
    
    for i, m in enumerate(masks):
        if i != best_mask_idx:
            # 其他候选mask覆盖但真值不覆盖的区域
            confused_regions |= (m > 0) & (gt_mask == 0)
    
    # 在混淆区域中采样负点
    candidates_y, candidates_x = np.where(confused_regions)
    
    if len(candidates_y) == 0:
        # 回退到启发式方法
        return generate_negative_points_heuristic(gt_mask, bbox, num_points, seed=seed)
    
    # 计算到真值mask的距离
    dist_to_mask = distance_transform_edt(gt_mask == 0)
    
    # 选择距离适中的点
    distances = dist_to_mask[candidates_y, candidates_x]
    valid_mask = (distances > 5) & (distances < 100)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        valid_indices = np.arange(len(candidates_y))
    
    # 按距离排序，选择最近的点（最容易混淆的）
    sorted_indices = valid_indices[np.argsort(distances[valid_indices])]
    
    negative_points = []
    for idx in sorted_indices[:num_points]:
        x, y = candidates_x[idx], candidates_y[idx]
        negative_points.append([int(x), int(y)])
    
    return negative_points


def compute_positive_points(mask: np.ndarray, num_points: int = 2) -> List[List[int]]:
    """
    计算正点: 使用距离变换找到mask内部的点
    
    策略: 找两个最大内接圆的圆心
    """
    h, w = mask.shape
    
    # 距离变换
    dist = distance_transform_edt(mask > 0)
    
    if dist.max() == 0:
        # mask为空，返回空列表
        return []
    
    # 第一个点: 全局最大
    p1_idx = np.unravel_index(dist.argmax(), dist.shape)
    points = [[int(p1_idx[1]), int(p1_idx[0])]]
    
    if num_points >= 2:
        # 第二个点: 抑制第一个点周围后的最大
        dist_copy = dist.copy()
        r = max(10, int(dist[p1_idx] * 0.5))
        y_grid, x_grid = np.ogrid[:h, :w]
        suppress_mask = (x_grid - p1_idx[1])**2 + (y_grid - p1_idx[0])**2 < r**2
        dist_copy[suppress_mask] = 0
        
        if dist_copy.max() > 0:
            p2_idx = np.unravel_index(dist_copy.argmax(), dist_copy.shape)
            points.append([int(p2_idx[1]), int(p2_idx[0])])
    
    return points


def compute_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """
    从mask计算bbox
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [0, 0, 0, 0]
    
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    
    return [int(x1), int(y1), int(x2), int(y2)]


def prepare_training_sample(
    image: np.ndarray,
    mask: np.ndarray,
    question: str,
    method: str = 'heuristic',
    sam_predictor = None,
    num_negative_points: int = 2,
    seed: Optional[int] = None
) -> Dict:
    """
    准备一个完整的训练样本，包含负点
    
    Args:
        image: RGB图像
        mask: 二值mask
        question: 问题文本
        method: 'heuristic', 'edge', 或 'sam'
        sam_predictor: SAM predictor (method='sam'时需要)
        num_negative_points: 负点数量
        seed: 随机种子
    
    Returns:
        包含所有标注的字典
    """
    # 计算bbox
    bbox = compute_bbox_from_mask(mask)
    
    # 计算正点
    positive_points = compute_positive_points(mask, num_points=2)
    
    if len(positive_points) < 2:
        # mask太小，使用bbox中心
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        positive_points = [[cx, cy], [cx, cy]]
    
    # 生成负点
    if method == 'heuristic':
        negative_points = generate_negative_points_heuristic(
            mask, bbox, num_negative_points, seed=seed
        )
    elif method == 'edge':
        negative_points = generate_negative_points_edge_based(
            mask, bbox, num_negative_points, seed=seed
        )
    elif method == 'sam' and sam_predictor is not None:
        negative_points = generate_negative_points_sam_ambiguity(
            image, bbox, positive_points, mask, sam_predictor, 
            num_negative_points, seed=seed
        )
    else:
        negative_points = generate_negative_points_heuristic(
            mask, bbox, num_negative_points, seed=seed
        )
    
    return {
        'question': question,
        'bbox': bbox,
        'points_1': positive_points[0] if len(positive_points) > 0 else [0, 0],
        'points_2': positive_points[1] if len(positive_points) > 1 else positive_points[0] if len(positive_points) > 0 else [0, 0],
        'negative_points': negative_points,
        'mask': mask
    }


# ============ 测试函数 ============

def test_negative_points_generation():
    """测试负点生成"""
    # 创建一个简单的测试mask
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 1, -1)
    
    bbox = compute_bbox_from_mask(mask)
    print(f"Bbox: {bbox}")
    
    # 测试启发式方法
    neg_points = generate_negative_points_heuristic(mask, bbox, num_points=2)
    print(f"Heuristic negative points: {neg_points}")
    
    # 测试边缘方法
    neg_points_edge = generate_negative_points_edge_based(mask, bbox, num_points=2)
    print(f"Edge-based negative points: {neg_points_edge}")
    
    # 验证负点不在mask内
    for p in neg_points:
        assert mask[p[1], p[0]] == 0, f"Point {p} is inside mask!"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_negative_points_generation()
