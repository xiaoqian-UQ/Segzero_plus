"""
可视化负点生成结果
用于验证负点生成的质量

使用方法:
    python visualize_negative_points.py --data_dir /path/to/refcocog --num_samples 10 --output_dir ./vis_output
    python visualize_negative_points.py --mode directory --images_dir ./images --masks_dir ./masks --method sam \
        --sam_config /path/to/sam2_config.yaml --sam_checkpoint /path/to/sam2.pt
    python visualize_negative_points.py --mode arrow --arrow_path /path/to/arrows \
        --masks_dir /path/to/gt_masks --num_samples 10 --output_dir ./vis_output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional
import json
import random
import io
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from negative_points_generator import (
    generate_negative_points_heuristic,
    generate_negative_points_edge_based,
    generate_negative_points_sam_ambiguity,
    compute_positive_points,
    compute_bbox_from_mask,
    prepare_training_sample
)

def load_sam2_predictor(
    config_path: str,
    checkpoint_path: str,
    device: str = "auto",
    config_dir: Optional[str] = None,
):
    """
    加载SAM2 predictor

    支持两种传法：
    1) --sam_config /abs/or/rel/path/to/sam2.1_hiera_l.yaml
       -> 自动从该目录 initialize hydra，然后用 config_name="sam2.1_hiera_l"
    2) --sam_config sam2.1_hiera_l (或 sam2.1_hiera_l.yaml)
       -> 默认从 pkg://sam2 内置配置找（前提是 sam2 包里确实有）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    # device auto
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # ---- 关键：处理 config_path / config_dir，让 Hydra 能找到你的本地 yaml ----
    # 情况 A：用户传的是一个存在的 yaml 文件路径
    if os.path.isfile(config_path) and config_path.lower().endswith((".yaml", ".yml")):
        abs_yaml = os.path.abspath(config_path)
        inferred_dir = os.path.dirname(abs_yaml)
        config_name = os.path.splitext(os.path.basename(abs_yaml))[0]  # 去掉 .yaml
        hydra_dir = os.path.abspath(config_dir) if config_dir else inferred_dir

        # 初始化 Hydra 的 config 搜索路径到你的目录
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        initialize_config_dir(
            config_dir=hydra_dir,
            version_base=None,
            job_name="sam2_local_cfg",
        )

        sam_model = build_sam2(config_name, ckpt_path=checkpoint_path, device=device, mode="eval")
        return SAM2ImagePredictor(sam_model)

    # 情况 B：用户传的是 config name（可能带 .yaml，也可能不带）
    config_name = os.path.splitext(config_path)[0]

    sam_model = build_sam2(config_name, ckpt_path=checkpoint_path, device=device, mode="eval")
    return SAM2ImagePredictor(sam_model)


def parse_solution(solution: str):
    """
    解析solution字段中的bbox与points
    """
    box_pattern = r'<box>\(([^,]+),([^)]+)\),\(([^,]+),([^)]+)\)</box>'
    points_pattern = r'<points>\(([^,]+),([^)]+)\),\(([^,]+),([^)]+)\)</points>'

    box_match = re.search(box_pattern, solution)
    points_match = re.search(points_pattern, solution)

    if not box_match or not points_match:
        return None, None

    bbox = [int(round(float(v))) for v in box_match.groups()]
    p1 = [int(round(float(points_match.group(1)))), int(round(float(points_match.group(2))))]
    p2 = [int(round(float(points_match.group(3)))), int(round(float(points_match.group(4))))]

    return bbox, [p1, p2]


def prepare_sample_with_points(
    image: np.ndarray,
    mask: np.ndarray,
    question: str,
    bbox: List[int],
    positive_points: List[List[int]],
    method: str = 'heuristic',
    sam_predictor=None,
    num_negative_points: int = 2,
    seed: Optional[int] = None
) -> Dict:
    """
    使用给定bbox与正点生成负点
    """
    if len(positive_points) < 2:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        positive_points = [[cx, cy], [cx, cy]]

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



def visualize_single_sample(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: List[int],
    positive_points: List[List[int]],
    negative_points: List[List[int]],
    question: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化单个样本的正负点
    
    Args:
        image: RGB图像
        mask: 二值mask
        bbox: [x1, y1, x2, y2]
        positive_points: 正点列表 [[x,y], ...]
        negative_points: 负点列表 [[x,y], ...]
        question: 问题文本
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原图 + bbox + points
    ax1 = axes[0]
    ax1.imshow(image)
    
    # 绘制bbox
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, 
        linewidth=2, edgecolor='yellow', facecolor='none'
    )
    ax1.add_patch(rect)
    
    # 绘制正点 (绿色圆点)
    for i, p in enumerate(positive_points):
        ax1.plot(p[0], p[1], 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax1.annotate(f'P{i+1}', (p[0]+5, p[1]-5), color='green', fontsize=12, fontweight='bold')
    
    # 绘制负点 (红色X)
    for i, p in enumerate(negative_points):
        ax1.plot(p[0], p[1], 'rx', markersize=15, markeredgewidth=3)
        ax1.annotate(f'N{i+1}', (p[0]+5, p[1]-5), color='red', fontsize=12, fontweight='bold')
    
    ax1.set_title(f'Image + Points\n"{question[:50]}..."' if len(question) > 50 else f'Image + Points\n"{question}"')
    ax1.axis('off')
    
    # 2. Mask叠加
    ax2 = axes[1]
    # 创建彩色叠加
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 255, 0]  # 绿色mask
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    ax2.imshow(overlay)
    
    # 绘制正点
    for p in positive_points:
        ax2.plot(p[0], p[1], 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # 绘制负点
    for p in negative_points:
        ax2.plot(p[0], p[1], 'rx', markersize=15, markeredgewidth=3)
    
    ax2.set_title('Mask Overlay (Green=GT)')
    ax2.axis('off')
    
    # 3. 混淆区域可视化
    ax3 = axes[2]
    
    # 创建混淆区域可视化
    h, w = mask.shape
    confusion_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 扩展bbox区域 (黄色)
    margin = 50
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)
    confusion_vis[ey1:ey2, ex1:ex2] = [255, 255, 0]  # 黄色: 混淆区域范围
    
    # GT mask (绿色)
    confusion_vis[mask > 0] = [0, 255, 0]
    
    # 候选负点区域 (蓝色): bbox扩展区域但mask外
    candidate_region = np.zeros((h, w), dtype=bool)
    candidate_region[ey1:ey2, ex1:ex2] = True
    candidate_region[mask > 0] = False
    confusion_vis[candidate_region] = [100, 100, 255]  # 蓝色: 候选负点区域
    
    ax3.imshow(confusion_vis)
    
    # 绘制bbox
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
    )
    ax3.add_patch(rect)
    
    # 绘制负点
    for p in negative_points:
        ax3.plot(p[0], p[1], 'rx', markersize=15, markeredgewidth=3)
    
    ax3.set_title('Confusion Region\n(Blue=Candidate, Green=GT, Yellow=Margin)')
    ax3.axis('off')
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='green', label='GT Mask'),
        patches.Patch(facecolor='blue', alpha=0.5, label='Candidate Neg Region'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive Points'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=10, markeredgewidth=2, label='Negative Points'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def load_refcocog_sample(data_dir: str, sample_idx: int = 0) -> Dict:
    """
    加载RefCOCOg数据集的一个样本
    
    需要根据你的数据集格式进行调整
    """
    # 这里需要根据实际数据格式修改
    # 假设数据结构:
    # data_dir/
    #   images/
    #   masks/
    #   annotations.json 或 instances.json
    
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # 尝试多种可能的标注文件
    ann_files = ['annotations.json', 'instances.json', 'refs.json']
    annotations = None
    
    for ann_file in ann_files:
        ann_path = os.path.join(data_dir, ann_file)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            break
    
    if annotations is None:
        raise FileNotFoundError(f"No annotation file found in {data_dir}")
    
    # 根据实际格式解析
    # 这里是一个通用的尝试
    if isinstance(annotations, list):
        sample = annotations[sample_idx % len(annotations)]
    elif isinstance(annotations, dict):
        if 'annotations' in annotations:
            sample = annotations['annotations'][sample_idx % len(annotations['annotations'])]
        elif 'images' in annotations:
            sample = annotations['images'][sample_idx % len(annotations['images'])]
        else:
            sample = list(annotations.values())[sample_idx % len(annotations)]
    
    return sample


def visualize_from_directory(
    images_dir: str,
    masks_dir: str,
    questions_file: Optional[str] = None,
    num_samples: int = 10,
    output_dir: str = './vis_output',
    method: str = 'heuristic',
    sam_predictor=None,
    seed: int = 42
):
    """
    从目录加载图像和mask进行可视化
    
    Args:
        images_dir: 图像目录
        masks_dir: mask目录
        questions_file: 问题文件 (json, 每行一个问题，或None使用默认问题)
        num_samples: 可视化的样本数量
        output_dir: 输出目录
        method: 负点生成方法
        seed: 随机种子
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像列表
    image_files = sorted([f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # 获取mask列表
    mask_files = sorted([f for f in os.listdir(masks_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # 加载问题
    questions = None
    if questions_file and os.path.exists(questions_file):
        with open(questions_file, 'r') as f:
            questions = json.load(f)
            if isinstance(questions, dict):
                questions = list(questions.values())
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 随机选择样本
    num_available = min(len(image_files), len(mask_files))
    if num_samples > num_available:
        print(f"Warning: Only {num_available} samples available")
        num_samples = num_available
    
    selected_indices = random.sample(range(num_available), num_samples)
    
    print(f"Visualizing {num_samples} samples...")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    for i, idx in enumerate(selected_indices):
        try:
            # 加载图像
            image_path = os.path.join(images_dir, image_files[idx])
            image = np.array(Image.open(image_path).convert('RGB'))
            
            # 加载mask
            mask_path = os.path.join(masks_dir, mask_files[idx])
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.uint8)  # 二值化
            
            # 获取问题
            if questions and idx < len(questions):
                question = questions[idx] if isinstance(questions[idx], str) else questions[idx].get('question', 'Find the object')
            else:
                question = f"Find the target object (sample {idx})"
            
            # 生成标注
            sample = prepare_training_sample(
                image=image,
                mask=mask,
                question=question,
                method=method,
                sam_predictor=sam_predictor,
                num_negative_points=2,
                seed=seed + i
            )
            
            # 可视化
            save_path = os.path.join(output_dir, f'sample_{i:03d}_{image_files[idx]}')
            
            visualize_single_sample(
                image=image,
                mask=mask,
                bbox=sample['bbox'],
                positive_points=[sample['points_1'], sample['points_2']],
                negative_points=sample['negative_points'],
                question=question,
                save_path=save_path,
                show=False
            )
            
            print(f"[{i+1}/{num_samples}] {image_files[idx]}")
            print(f"    Bbox: {sample['bbox']}")
            print(f"    Positive: {sample['points_1']}, {sample['points_2']}")
            print(f"    Negative: {sample['negative_points']}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print("-" * 50)
    print(f"Done! Results saved to: {output_dir}")


def sample_rows_from_arrow(
    arrow_paths: List[str],
    num_samples: int,
    seed: int = 42
) -> List[Dict]:
    """
    从arrow分片中抽样
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    random.seed(seed)
    samples = []
    seen = 0

    def extract_row(batch, row_idx: int) -> Dict:
        return {
            'id': batch.column(batch.schema.get_field_index('id'))[row_idx].as_py(),
            'problem': batch.column(batch.schema.get_field_index('problem'))[row_idx].as_py(),
            'solution': batch.column(batch.schema.get_field_index('solution'))[row_idx].as_py(),
            'image': batch.column(batch.schema.get_field_index('image'))[row_idx].as_py(),
        }

    for arrow_path in arrow_paths:
        with pa.memory_map(arrow_path, 'r') as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row_idx in range(batch.num_rows):
                    seen += 1
                    if len(samples) < num_samples:
                        samples.append(extract_row(batch, row_idx))
                    else:
                        j = random.randint(0, seen - 1)
                        if j < num_samples:
                            samples[j] = extract_row(batch, row_idx)

    return samples


def visualize_from_arrow(
    arrow_path: str,
    masks_dir: str,
    num_samples: int = 10,
    output_dir: str = './vis_output',
    method: str = 'heuristic',
    sam_predictor=None,
    seed: int = 42
):
    """
    从arrow文件读取图片与正点/bbox进行可视化
    """
    if os.path.isdir(arrow_path):
        arrow_paths = sorted([
            os.path.join(arrow_path, f)
            for f in os.listdir(arrow_path)
            if f.endswith('.arrow')
        ])
    else:
        arrow_paths = [arrow_path]

    if not arrow_paths:
        raise FileNotFoundError(f"No .arrow files found in {arrow_path}")

    os.makedirs(output_dir, exist_ok=True)
    samples = sample_rows_from_arrow(arrow_paths, num_samples, seed=seed)

    print(f"Visualizing {len(samples)} samples from arrow...")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    print("-" * 50)

    for i, row in enumerate(samples):
        try:
            sample_id = row['id']
            bbox, positive_points = parse_solution(row['solution'])
            if bbox is None or positive_points is None:
                print(f"Skip {sample_id}: cannot parse solution")
                continue

            image_bytes = row['image'].get('bytes') if isinstance(row['image'], dict) else None
            if not image_bytes:
                print(f"Skip {sample_id}: no image bytes")
                continue

            image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
            mask_path = os.path.join(masks_dir, f"{sample_id}.png")
            if not os.path.exists(mask_path):
                print(f"Skip {sample_id}: mask not found")
                continue
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.uint8)

            question = row.get('problem') or f"Find the target object ({sample_id})"

            sample = prepare_sample_with_points(
                image=image,
                mask=mask,
                question=question,
                bbox=bbox,
                positive_points=positive_points,
                method=method,
                sam_predictor=sam_predictor,
                num_negative_points=2,
                seed=seed + i
            )

            save_path = os.path.join(output_dir, f'sample_{i:03d}_{sample_id}.png')

            visualize_single_sample(
                image=image,
                mask=mask,
                bbox=sample['bbox'],
                positive_points=[sample['points_1'], sample['points_2']],
                negative_points=sample['negative_points'],
                question=question,
                save_path=save_path,
                show=False
            )

            print(f"[{i+1}/{len(samples)}] {sample_id}")
            print(f"    Bbox: {sample['bbox']}")
            print(f"    Positive: {sample['points_1']}, {sample['points_2']}")
            print(f"    Negative: {sample['negative_points']}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print("-" * 50)
    print(f"Done! Results saved to: {output_dir}")


def visualize_synthetic_demo(
    output_dir: str = './vis_output',
    method: str = 'heuristic',
    sam_predictor=None
):
    """
    使用合成数据进行演示，不需要真实数据集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic demo...")
    print("-" * 50)
    
    # 创建多个合成样本
    demo_configs = [
        {'shape': 'circle', 'size': 60, 'pos': (150, 150), 'question': 'Find the circular object'},
        {'shape': 'rectangle', 'size': 80, 'pos': (200, 100), 'question': 'Find the rectangular object'},
        {'shape': 'ellipse', 'size': 50, 'pos': (100, 200), 'question': 'Find the elliptical object'},
        {'shape': 'circle', 'size': 40, 'pos': (250, 250), 'question': 'Find the small circle'},
        {'shape': 'rectangle', 'size': 100, 'pos': (150, 150), 'question': 'Find the large rectangle'},
    ]
    
    for i, config in enumerate(demo_configs):
        # 创建合成图像 (带纹理的背景)
        h, w = 300, 300
        image = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        
        # 添加一些背景图案
        for _ in range(5):
            cx, cy = np.random.randint(50, 250, 2)
            cv2.circle(image, (cx, cy), np.random.randint(10, 30), 
                      (np.random.randint(50, 150),) * 3, -1)
        
        # 创建mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pos = config['pos']
        size = config['size']
        
        if config['shape'] == 'circle':
            cv2.circle(mask, pos, size, 1, -1)
            cv2.circle(image, pos, size, (200, 50, 50), -1)  # 红色目标
        elif config['shape'] == 'rectangle':
            x1, y1 = pos[0] - size//2, pos[1] - size//2
            x2, y2 = pos[0] + size//2, pos[1] + size//2
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (50, 200, 50), -1)  # 绿色目标
        elif config['shape'] == 'ellipse':
            cv2.ellipse(mask, pos, (size, size//2), 0, 0, 360, 1, -1)
            cv2.ellipse(image, pos, (size, size//2), 0, 0, 360, (50, 50, 200), -1)  # 蓝色目标
        
        # 生成标注
        sample = prepare_training_sample(
            image=image,
            mask=mask,
            question=config['question'],
            method=method,
            sam_predictor=sam_predictor,
            num_negative_points=2,
            seed=42 + i
        )
        
        # 可视化
        save_path = os.path.join(output_dir, f'synthetic_demo_{i:02d}.png')
        
        visualize_single_sample(
            image=image,
            mask=mask,
            bbox=sample['bbox'],
            positive_points=[sample['points_1'], sample['points_2']],
            negative_points=sample['negative_points'],
            question=config['question'],
            save_path=save_path,
            show=False
        )
        
        print(f"[{i+1}/{len(demo_configs)}] {config['shape'].capitalize()}")
        print(f"    Bbox: {sample['bbox']}")
        print(f"    Positive: {sample['points_1']}, {sample['points_2']}")
        print(f"    Negative: {sample['negative_points']}")
    
    print("-" * 50)
    print(f"Done! Demo results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize negative points generation')
    parser.add_argument('--mode', type=str, default='demo', 
                        choices=['demo', 'directory', 'refcocog', 'arrow'],
                        help='Visualization mode')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Directory containing masks')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for RefCOCOg format')
    parser.add_argument('--arrow_path', type=str, default=None,
                        help='Arrow file or directory for RefCOCOg 9k')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='JSON file containing questions')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./vis_output',
                        help='Output directory for visualizations')
    parser.add_argument('--method', type=str, default='heuristic',
                        choices=['heuristic', 'edge', 'sam'],
                        help='Negative points generation method')
    parser.add_argument('--sam_config', type=str,
                        default='../../model/checkpoint/sam2.1_hiera_l.yaml',
                        help='SAM2 config yaml (required for method=sam)')
    parser.add_argument("--sam_config_dir",
                        type=str,
                        default=None,
                        help="Directory that contains SAM2 hydra config yaml (e.g. ../../model/checkpoint)")
    parser.add_argument('--sam_checkpoint', type=str,
                        default='../../model/checkpoint/sam2.1_hiera_large.pt',
                        help='SAM2 checkpoint path (required for method=sam)')
    parser.add_argument('--sam_device', type=str, default='auto',
                        help='SAM2 device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    sam_predictor = None
    if args.method == 'sam':
        if not args.sam_config or not args.sam_checkpoint:
            print("Error: --sam_config and --sam_checkpoint are required for method 'sam'")
            return
        try:
            sam_predictor = load_sam2_predictor(
                args.sam_config,
                args.sam_checkpoint,
                device=args.sam_device,
                config_dir=args.sam_config_dir,
            )
            print("SAM2 predictor loaded.")
        except Exception as e:
            print(f"Error loading SAM2 predictor: {e}")
            return

    if args.mode == 'demo':
        # 使用合成数据演示
        visualize_synthetic_demo(
            output_dir=args.output_dir,
            method=args.method,
            sam_predictor=sam_predictor
        )
        
    elif args.mode == 'directory':
        # 从目录加载
        if not args.images_dir or not args.masks_dir:
            print("Error: --images_dir and --masks_dir are required for 'directory' mode")
            print("Example: python visualize_negative_points.py --mode directory --images_dir ./images --masks_dir ./masks")
            return
        
        visualize_from_directory(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            questions_file=args.questions_file,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            method=args.method,
            sam_predictor=sam_predictor,
            seed=args.seed
        )
        
    elif args.mode == 'refcocog':
        # RefCOCOg格式
        if not args.data_dir:
            print("Error: --data_dir is required for 'refcocog' mode")
            return
        
        # 假设RefCOCOg的标准目录结构
        images_dir = os.path.join(args.data_dir, 'images')
        masks_dir = os.path.join(args.data_dir, 'masks')
        
        if not os.path.exists(images_dir):
            # 尝试其他可能的路径
            images_dir = os.path.join(args.data_dir, 'train2014')
        
        visualize_from_directory(
            images_dir=images_dir,
            masks_dir=masks_dir,
            questions_file=args.questions_file,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            method=args.method,
            sam_predictor=sam_predictor,
            seed=args.seed
        )
        
    elif args.mode == 'arrow':
        if not args.arrow_path or not args.masks_dir:
            print("Error: --arrow_path and --masks_dir are required for 'arrow' mode")
            return

        visualize_from_arrow(
            arrow_path=args.arrow_path,
            masks_dir=args.masks_dir,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            method=args.method,
            sam_predictor=sam_predictor,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
