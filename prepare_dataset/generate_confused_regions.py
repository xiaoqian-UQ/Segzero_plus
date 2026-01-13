"""
预计算混淆区域，加速训练
"""

import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyarrow.parquet as pq
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def generate_confused_regions_dataset(
    input_parquet: str,
    output_dir: str,
    sam_model_path: str,
    num_samples: int = 5
):
    """
    为数据集预计算混淆区域
    """
    # 加载SAM模型
    sam_model = build_sam2("sam2_hiera_l.yaml", sam_model_path, device="cuda")
    sam_predictor = SAM2ImagePredictor(sam_model)
    
    # 读取数据
    df = pq.read_table(input_parquet).to_pandas()
    
    os.makedirs(output_dir, exist_ok=True)
    
    confused_regions_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image = np.array(Image.open(row['image_path']))
        gt_mask = np.array(Image.open(row['mask_path']))
        
        # 计算混淆区域
        confused = identify_confused_regions_fast(
            image, gt_mask, sam_predictor, num_samples
        )
        
        # 保存混淆区域mask
        confused_path = os.path.join(output_dir, f"confused_{idx:06d}.npy")
        np.save(confused_path, confused)
        
        confused_regions_data.append({
            'image_id': row.get('image_id', idx),
            'confused_region_path': confused_path
        })
    
    # 保存索引
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(confused_regions_data, f)
    
    print(f"Generated confused regions for {len(df)} samples")


def identify_confused_regions_fast(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam_predictor,
    num_samples: int = 5
) -> np.ndarray:
    """快速版混淆区域识别"""
    h, w = gt_mask.shape[:2]
    confused = np.zeros((h, w), dtype=np.float32)
    
    # 获取mask边界
    mask_coords = np.argwhere(gt_mask > 0)
    if len(mask_coords) == 0:
        return confused
    
    y_min, x_min = mask_coords.min(axis=0)
    y_max, x_max = mask_coords.max(axis=0)
    
    # 采样区域
    margin = 80
    x_range = (max(0, x_min - margin), min(w, x_max + margin))
    y_range = (max(0, y_min - margin), min(h, y_max + margin))
    
    sam_predictor.set_image(image)
    
    for _ in range(num_samples):
        # 在mask外采样
        for _ in range(5):
            x = np.random.randint(x_range[0], x_range[1])
            y = np.random.randint(y_range[0], y_range[1])
            if gt_mask[y, x] == 0:
                break
        else:
            continue
        
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        for mask, score in zip(masks, scores):
            if score > 0.5:
                non_gt = mask & (gt_mask == 0)
                confused += non_gt.astype(np.float32)
    
    if confused.max() > 0:
        confused /= confused.max()
    
    return confused


if __name__ == "__main__":
    generate_confused_regions_dataset(
        input_parquet="/mnt/xiaoqian/dataset/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/",
        output_dir="/mnt/xiaoqian/dataset/refcocog_9k/confused_region",
        sam_model_path="/mnt/xiaoqian/model/sam2/checkpoints/sam2.1_hiera_large.pt"
    )