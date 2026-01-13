# verl/utils/reward_score/reward_manager.py

from typing import Dict, List
import numpy as np
from .output_parser import parse_seg_zero_output, prepare_sam_prompts
from .segmentation_rewards import compute_total_reward, identify_confused_regions

class SegZeroRewardManager:
    """
    Seg-Zero++ 奖励管理器
    管理奖励计算、SAM推理和混淆区域识别
    """
    
    def __init__(
        self,
        sam_model_path: str = "pretrained_models/sam2.1_hiera_large.pt",
        use_negative_reward: bool = True,
        use_confused_regions: bool = True,
        device: str = "cuda"
    ):
        self.use_negative_reward = use_negative_reward
        self.use_confused_regions = use_confused_regions
        self.device = device
        
        # 加载SAM2模型（用于计算mask和混淆区域）
        self._load_sam_model(sam_model_path)
    
    def _load_sam_model(self, model_path: str):
        """加载SAM2模型"""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.sam_model = build_sam2(
            "sam2_hiera_l.yaml",
            model_path,
            device=self.device
        )
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
    
    def compute_reward_batch(
        self,
        responses: List[str],
        images: List[np.ndarray],
        gt_bboxes: List[List[float]],
        gt_masks: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """
        批量计算奖励
        
        Args:
            responses: 模型输出列表
            images: 图像列表 (N, H, W, 3)
            gt_bboxes: GT bbox列表
            gt_masks: GT mask列表
        
        Returns:
            奖励字典列表
        """
        all_rewards = []
        
        for response, image, gt_bbox, gt_mask in zip(
            responses, images, gt_bboxes, gt_masks
        ):
            # 解析输出
            parsed = parse_seg_zero_output(response)
            
            # 计算混淆区域（可选）
            confused_regions = None
            if self.use_confused_regions and self.use_negative_reward:
                confused_regions = identify_confused_regions(
                    image, gt_mask, self.sam_predictor
                )
            
            # 计算奖励
            rewards = compute_total_reward(
                response=response,
                parsed_output=parsed,
                gt_bbox=gt_bbox,
                gt_mask=gt_mask,
                confused_regions=confused_regions,
                use_negative_reward=self.use_negative_reward
            )
            
            all_rewards.append(rewards)
        
        return all_rewards
    
    def compute_segmentation_mask(
        self,
        image: np.ndarray,
        parsed_output: Dict
    ) -> np.ndarray:
        """
        使用SAM2计算分割mask
        
        Args:
            image: 输入图像
            parsed_output: 解析后的模型输出
        
        Returns:
            分割mask
        """
        sam_prompts = prepare_sam_prompts(parsed_output)
        
        self.sam_predictor.set_image(image)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=sam_prompts['point_coords'],
            point_labels=sam_prompts['point_labels'],
            box=sam_prompts['box'],
            multimask_output=False
        )
        
        return masks[0]  # 返回最佳mask