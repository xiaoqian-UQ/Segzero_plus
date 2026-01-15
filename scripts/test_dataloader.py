# scripts/test_dataloader.py

from src.data.dataset import create_dataloader
import matplotlib.pyplot as plt
import numpy as np

# 配置路径 - 根据你的实际路径修改
ARROW_DIR = "/mnt/xiaoqian/dataset/refcocog/refcocog_9k/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e/"
MASK_DIR = "/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks"

# 创建dataloader
dataloader = create_dataloader(
    arrow_dir=ARROW_DIR,
    mask_dir=MASK_DIR,
    batch_size=1,
    shuffle=False
)

# 测试加载
batch = next(iter(dataloader))

print("=== Batch Info ===")
print(f"Image shape: {batch['image'][0].shape}")
print(f"Query: {batch['query'][0]}")
print(f"GT Mask shape: {batch['gt_mask'][0].shape}")
print(f"GT Mask sum: {batch['gt_mask'][0].sum()}")  # 检查mask是否有内容
print(f"GT Bbox: {batch['gt_bbox'][0]}")
print(f"GT Points: {batch['gt_points'][0]}")
print(f"Sample ID: {batch['sample_id'][0]}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

image = batch['image'][0]
mask = batch['gt_mask'][0]
bbox = batch['gt_bbox'][0]
points = batch['gt_points'][0]
H, W = 840, 840

# 原图
axes[0].imshow(image)
axes[0].set_title(f"Query: {batch['query'][0][:50]}...")
axes[0].axis('off')

# GT Mask
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("GT Mask")
axes[1].axis('off')

# 叠加显示
overlay = image.copy().astype(float)
mask_bool = mask > 0.5
overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
axes[2].imshow(overlay.astype(np.uint8))

# 绘制bbox
rect = plt.Rectangle(
    (bbox[0]*W, bbox[1]*H), 
    (bbox[2]-bbox[0])*W, 
    (bbox[3]-bbox[1])*H,
    fill=False, color='green', linewidth=2
)
axes[2].add_patch(rect)

# 绘制points
for px, py in points:
    axes[2].scatter(px*W, py*H, c='blue', s=100, marker='o')

axes[2].set_title(f"Overlay + BBox + Points\nID: {batch['sample_id'][0]}")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("dataloader_test.png", dpi=150)
print("\nSaved visualization to dataloader_test.png")