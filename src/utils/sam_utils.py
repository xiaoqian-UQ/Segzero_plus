# src/utils/sam_utils.py

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Wrapper:
    """SAM2模型封装，支持正负点输入"""
    
    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        device: str = "cuda"
    ):

        self.device = device
        self.predictor = self._load_predictor(model_cfg, checkpoint, device)
        self.predictor.model.to(device)

    def _load_predictor(self, model_cfg: str, checkpoint: str, device: str):
        if os.path.isfile(model_cfg) and model_cfg.lower().endswith((".yaml", ".yml")):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

            from sam2.build_sam import build_sam2
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            abs_yaml = os.path.abspath(model_cfg)
            config_dir = os.path.dirname(abs_yaml)
            config_name = os.path.splitext(os.path.basename(abs_yaml))[0]

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            initialize_config_dir(
                config_dir=config_dir,
                version_base=None,
                job_name="sam2_local_cfg",
            )

            sam_model = build_sam2(config_name, ckpt_path=checkpoint, device=device, mode="eval")
            return SAM2ImagePredictor(sam_model)

        # fallback: treat model_cfg as Hugging Face model id
        return SAM2ImagePredictor.from_pretrained(model_cfg)
    
    def predict_mask(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        使用正负点和边界框生成分割mask
        
        Args:
            image: 输入图像 (H, W, 3)
            positive_points: 正点列表，归一化坐标
            negative_points: 负点列表，归一化坐标
            bbox: 边界框，归一化坐标 (x1, y1, x2, y2)
            multimask_output: 是否输出多个mask候选
            
        Returns:
            mask: 二值分割mask (H, W)
            confidence: SAM预测的IoU置信度
        """
        H, W = image.shape[:2]
        
        # 转换归一化坐标为像素坐标
        pos_points_pixel = [(int(x * W), int(y * H)) for x, y in positive_points]
        neg_points_pixel = [(int(x * W), int(y * H)) for x, y in negative_points]
        bbox_pixel = (
            int(bbox[0] * W),
            int(bbox[1] * H),
            int(bbox[2] * W),
            int(bbox[3] * H)
        )
        
        # 构建SAM输入
        all_points = pos_points_pixel + neg_points_pixel
        labels = [1] * len(pos_points_pixel) + [0] * len(neg_points_pixel)
        
        point_coords = np.array(all_points)
        point_labels = np.array(labels)
        box = np.array([bbox_pixel])
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 预测
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
        
        # 选择最高置信度的mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask, best_score
    
    def predict_baseline_mask(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        仅使用正点生成基线mask（用于计算混淆区域）
        
        Args:
            image: 输入图像
            positive_points: 正点列表
            bbox: 边界框
            
        Returns:
            baseline_mask: 不使用负点的基线mask
        """
        H, W = image.shape[:2]
        
        pos_points_pixel = [(int(x * W), int(y * H)) for x, y in positive_points]
        bbox_pixel = (
            int(bbox[0] * W),
            int(bbox[1] * H),
            int(bbox[2] * W),
            int(bbox[3] * H)
        )
        
        point_coords = np.array(pos_points_pixel)
        point_labels = np.array([1] * len(pos_points_pixel))
        box = np.array([bbox_pixel])
        
        self.predictor.set_image(image)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
        
        return masks[0]
