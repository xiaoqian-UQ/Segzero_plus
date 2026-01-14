"""新建
原始 Seg-Zero 输出格式:
<think> reasoning process </think>
<answer> { "bbox": [x1,y1,x2,y2], "points_1": [px1,py1], "points_2": [px2,py2] } </answer>

新的输出格式 (加入负点):
<think> reasoning process </think>
<answer> { 
    "bbox": [x1,y1,x2,y2], 
    "points_1": [px1,py1], 
    "points_2": [px2,py2],
    "negative_points": [[nx1,ny1], [nx2,ny2]]  # 新增
} </answer>
"""

import re
import json
from typing import Optional, Tuple, List, Dict

def parse_segmentation_output(response: str) -> Optional[Dict]:
    """
    解析模型输出，提取bbox、正点和负点
    """
    # 提取 <answer> 标签内容
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_pattern, response, re.DOTALL)
    
    if not match:
        return None
    
    answer_text = match.group(1).strip()
    
    try:
        # 尝试解析 JSON
        result = json.loads(answer_text)
        
        # 验证必需字段
        required_keys = ['bbox', 'points_1', 'points_2']
        for key in required_keys:
            if key not in result:
                return None
        
        # 负点是可选的，如果不存在则设为空列表
        if 'negative_points' not in result:
            result['negative_points'] = []
            
        return result
    except json.JSONDecodeError:
        return None


def validate_negative_points(neg_points: List[List[int]], 
                             bbox: List[int],
                             image_size: Tuple[int, int] = (840, 840)) -> bool:
    """
    验证负点格式是否正确
    - 负点应在图像范围内
    - 负点应在 bbox 外部（更合理的位置）
    """
    if not neg_points:
        return True  # 空列表是有效的
    
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    
    for point in neg_points:
        if len(point) != 2:
            return False
        px, py = point
        # 检查是否在图像范围内
        if not (0 <= px < img_w and 0 <= py < img_h):
            return False
    
    return True