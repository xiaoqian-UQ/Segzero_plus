"""
Seg-Zero++ 评估脚本
支持ReasonSeg和RefCOCO评估
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.reward_score.output_parser import parse_seg_zero_output, prepare_sam_prompts
from verl.utils.reward_score.prompt_templates import NEGATIVE_POINT_PROMPT


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def compute_giou(pred_masks: list, gt_masks: list) -> float:
    """计算gIoU (平均IoU)"""
    ious = [compute_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return np.mean(ious)


def compute_ciou(pred_masks: list, gt_masks: list) -> float:
    """计算cIoU (累积IoU)"""
    total_intersection = sum(
        np.logical_and(p, g).sum() for p, g in zip(pred_masks, gt_masks)
    )
    total_union = sum(
        np.logical_or(p, g).sum() for p, g in zip(pred_masks, gt_masks)
    )
    return total_intersection / total_union if total_union > 0 else 0.0


class SegZeroPlusEvaluator:
    def __init__(
        self,
        model_path: str,
        sam_model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        # 加载Qwen2.5-VL模型
        print(f"Loading model from {model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 加载SAM2模型
        print(f"Loading SAM2 from {sam_model_path}")
        self.sam_model = build_sam2("sam2_hiera_l.yaml", sam_model_path, device=device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
    
    def generate_response(self, image: Image.Image, query: str) -> str:
        """生成模型响应"""
        prompt = NEGATIVE_POINT_PROMPT.format(Question=query)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        response = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response
    
    def predict_mask(self, image: np.ndarray, response: str) -> np.ndarray:
        """根据模型响应生成分割mask"""
        parsed = parse_seg_zero_output(response)
        
        if not parsed['format_valid']:
            # 返回空mask
            return np.zeros(image.shape[:2], dtype=bool)
        
        sam_prompts = prepare_sam_prompts(parsed)
        
        self.sam_predictor.set_image(image)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=sam_prompts['point_coords'],
            point_labels=sam_prompts['point_labels'],
            box=sam_prompts['box'],
            multimask_output=False
        )
        
        return masks[0]
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_path: str = None,
        max_samples: int = None
    ) -> dict:
        """评估数据集"""
        # 加载数据
        with open(data_path) as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        pred_masks = []
        gt_masks = []
        results = []
        
        for item in tqdm(data, desc="Evaluating"):
            # 加载图像
            image = Image.open(item['image_path']).convert('RGB')
            image_np = np.array(image)
            
            # 加载GT mask
            gt_mask = np.array(Image.open(item['mask_path'])) > 0
            
            # 生成响应
            response = self.generate_response(image, item['query'])
            
            # 预测mask
            pred_mask = self.predict_mask(image_np, response)
            
            # 计算IoU
            iou = compute_iou(pred_mask, gt_mask)
            
            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)
            
            results.append({
                'image_id': item.get('image_id', ''),
                'query': item['query'],
                'iou': float(iou),
                'response': response
            })
        
        # 计算整体指标
        giou = compute_giou(pred_masks, gt_masks)
        ciou = compute_ciou(pred_masks, gt_masks)
        
        metrics = {
            'gIoU': float(giou),
            'cIoU': float(ciou),
            'num_samples': len(data)
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"gIoU: {giou:.4f}")
        print(f"cIoU: {ciou:.4f}")
        print(f"Samples: {len(data)}")
        
        # 保存结果
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sam_model_path', type=str, 
                        default='pretrained_models/sam2.1_hiera_large.pt')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    evaluator = SegZeroPlusEvaluator(
        model_path=args.model_path,
        sam_model_path=args.sam_model_path
    )
    
    evaluator.evaluate_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()