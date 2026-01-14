"""
可视化负点生成结果
用于验证负点生成的质量

使用方法:
    python visualize_negative_points.py --data_dir /path/to/refcocog --num_samples 10 --output_dir ./vis_output
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from negative_points_generator import (
    generate_negative_points_heuristic,
    generate_negative_points_edge_based,
    compute_positive_points,
    compute_bbox_from_mask,
    prepare_training_sample
)


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


def visualize_synthetic_demo(output_dir: str = './vis_output'):
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
            method='heuristic',
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
                        choices=['demo', 'directory', 'refcocog'],
                        help='Visualization mode')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Directory containing masks')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for RefCOCOg format')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='JSON file containing questions')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./vis_output',
                        help='Output directory for visualizations')
    parser.add_argument('--method', type=str, default='heuristic',
                        choices=['heuristic', 'edge'],
                        help='Negative points generation method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # 使用合成数据演示
        visualize_synthetic_demo(args.output_dir)
        
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
            seed=args.seed
        )


if __name__ == '__main__':
    main()