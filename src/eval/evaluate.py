# src/eval/evaluate.py

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import ReasonSegDataset, collate_fn
from src.utils.parser import SegZeroOutputParser
from src.utils.sam_utils import SAM2Wrapper
from src.eval.metrics import SegmentationMetrics

def evaluate(args):
    """运行评估"""
    
    # 加载模型
    model = load_model(args.model_path, args.checkpoint)
    tokenizer = load_tokenizer(args.model_path)
    
    # 初始化工具
    parser = SegZeroOutputParser(require_negative_points=args.use_negative_points)
    sam_wrapper = SAM2Wrapper(
        model_cfg=args.sam_config,
        checkpoint=args.sam_checkpoint
    )
    
    # 加载数据
    dataset = ReasonSegDataset(
        data_root=args.data_root,
        split=args.split
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 评估
    metrics = SegmentationMetrics()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        image = batch["image"][0]
        query = batch["query"][0]
        gt_mask = batch["gt_mask"][0]
        
        # 生成输出
        prompt = build_prompt(query, args.use_negative_points)
        output = generate(model, tokenizer, prompt, image)
        
        # 解析
        parsed = parser.parse(output)
        
        if not parsed.is_valid:
            # 格式错误，使用空mask
            pred_mask = np.zeros_like(gt_mask)
        else:
            # SAM推理
            if args.use_negative_points:
                pred_mask, _ = sam_wrapper.predict_mask(
                    image,
                    parsed.positive_points,
                    parsed.negative_points,
                    parsed.bbox
                )
            else:
                pred_mask, _ = sam_wrapper.predict_mask(
                    image,
                    parsed.positive_points,
                    [],  # 空负点
                    parsed.bbox
                )
        
        metrics.update(pred_mask, gt_mask)
    
    # 输出结果
    results = metrics.compute()
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")
    
    return results

def build_prompt(query: str, use_negative_points: bool) -> str:
    """构建prompt"""
    if use_negative_points:
        return (
            f"Please find '{query}' with bbox, points, and negative points."
            "Compare the difference between objects and find the most closely matched one."
            "Identify confusing background regions that should be excluded using negative points."
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "Output the one bbox, points of two largest inscribed circles inside the interested object, "
            "and negative points in confusing background regions, all in JSON format."
            "i.e., <think> thinking process here </think>"
            '<answer>{"bbox": [x1, y1, x2, y2], "points": [[x1, y1], [x2, y2]], "negative_points": [[x1, y1]]}</answer>'
        )
    else:
        return (
            f"Please find '{query}' with bbox and points."
            "Compare the difference between objects and find the most closely matched one."
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
            "i.e., <think> thinking process here </think>"
            '<answer>{"bbox": [x1, y1, x2, y2], "points": [[x1, y1], [x2, y2]]}</answer>'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam_config", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--use_negative_points", action="store_true")
    args = parser.parse_args()
    
    evaluate(args)