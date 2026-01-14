# src/data/dataset.py

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any, List
import glob

from src.utils.solution_parser import SolutionParser

class RefCOCOgDataset(Dataset):
    """RefCOCOg数据集加载器，用于Seg-Zero负点训练"""
    
    def __init__(
        self,
        arrow_dir: str,
        mask_dir: str,
        image_size: int = 840
    ):
        """
        Args:
            arrow_dir: Arrow文件目录路径
            mask_dir: GT mask目录路径 (例如 /mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks)
            image_size: 图像大小 (默认840)
        """
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.solution_parser = SolutionParser(image_size=image_size)
        
        # 加载HuggingFace数据集
        arrow_files = os.path.join(arrow_dir, "*.arrow")
        self.dataset = load_dataset('arrow', data_files=arrow_files, split='train')
        
        print(f"Loaded {len(self.dataset)} samples")
        print(f"Columns: {self.dataset.column_names}")
        
        # 验证mask目录
        if os.path.exists(mask_dir):
            mask_count = len([f for f in os.listdir(mask_dir) if f.endswith('.png')])
            print(f"Found {mask_count} mask files in {mask_dir}")
        else:
            print(f"Warning: Mask directory not found: {mask_dir}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # 获取ID (用于匹配mask文件)
        sample_id = item["id"]  # 例如 "refcocog_16521"
        
        # 加载图像
        image = item["image"]
        if hasattr(image, 'convert'):
            image = image.convert("RGB")
        image_np = np.array(image)
        
        # 确保图像尺寸正确
        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            from PIL import Image as PILImage
            image = PILImage.fromarray(image_np).resize((self.image_size, self.image_size))
            image_np = np.array(image)
        
        # 解析solution (获取bbox和points)
        solution_data = self.solution_parser.parse(item["solution"])
        
        if not solution_data.is_valid:
            print(f"Warning: Invalid solution for {sample_id}: {solution_data.error_message}")
        
        # 加载GT mask
        gt_mask = self._load_mask(sample_id)
        
        return {
            "image": image_np,
            "query": item["problem"],
            "gt_mask": gt_mask.astype(np.float32),
            "gt_bbox": solution_data.bbox,
            "gt_points": solution_data.points,
            "sample_id": sample_id,
            "image_id": idx,
            "original_size": (item["img_width"], item["img_height"])
        }
    
    def _load_mask(self, sample_id: str) -> np.ndarray:
        """
        加载GT mask
        
        Args:
            sample_id: 样本ID，例如 "refcocog_16521"
            
        Returns:
            二值mask数组
        """
        mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = mask > 127  # 转为二值
            
            # 确保尺寸正确
            if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
                mask = np.array(mask_pil) > 127
            
            return mask
        else:
            print(f"Warning: Mask not found for {sample_id}, using bbox mask")
            return self._create_bbox_mask(sample_id)
    
    def _create_bbox_mask(self, sample_id: str) -> np.ndarray:
        """当mask文件不存在时，用bbox创建粗略mask"""
        # 找到对应的item
        for i in range(len(self.dataset)):
            if self.dataset[i]["id"] == sample_id:
                solution_data = self.solution_parser.parse(self.dataset[i]["solution"])
                break
        else:
            return np.zeros((self.image_size, self.image_size), dtype=bool)
        
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        x1, y1, x2, y2 = solution_data.bbox_pixel
        mask[y1:y2, x1:x2] = True
        return mask


def collate_fn(batch: list) -> Dict[str, Any]:
    """自定义批次整理函数"""
    return {
        "image": [item["image"] for item in batch],
        "query": [item["query"] for item in batch],
        "gt_mask": [item["gt_mask"] for item in batch],
        "gt_bbox": [item["gt_bbox"] for item in batch],
        "gt_points": [item["gt_points"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "original_size": [item["original_size"] for item in batch]
    }


def create_dataloader(
    arrow_dir: str,
    mask_dir: str,
    batch_size: int = 2,
    image_size: int = 840,
    num_workers: int = 4,
    shuffle: bool = True
):
    """
    创建数据加载器
    
    使用示例:
```python
    dataloader = create_dataloader(
        arrow_dir="refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e",
        mask_dir="/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks",
        batch_size=2
    )
```
    """
    from torch.utils.data import DataLoader
    
    dataset = RefCOCOgDataset(
        arrow_dir=arrow_dir,
        mask_dir=mask_dir,
        image_size=image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader