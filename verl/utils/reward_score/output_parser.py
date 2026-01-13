import re
import json
from typing import Dict, List, Tuple, Optional

def parse_seg_zero_output(response: str) -> Dict:
    """
    解析Seg-Zero模型输出，支持正点和负点格式
    
    Returns:
        {
            'think': str,           # 推理过程
            'bbox': [x1,y1,x2,y2],  # 边界框
            'points_pos': [[x,y], [x,y]],  # 正点列表
            'points_neg': [[x,y], ...],    # 负点列表（可选）
            'format_valid': bool    # 格式是否有效
        }
    """
    result = {
        'think': '',
        'bbox': None,
        'points_pos': [],
        'points_neg': [],
        'format_valid': False
    }
    
    # 提取think部分
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        result['think'] = think_match.group(1).strip()
    
    # 提取answer部分
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return result
    
    answer_text = answer_match.group(1).strip()
    
    try:
        # 尝试解析JSON
        # 处理单引号的情况
        answer_text = answer_text.replace("'", '"')
        data = json.loads(answer_text)
        
        # 解析bbox
        if 'bbox' in data:
            result['bbox'] = data['bbox']
        
        # 解析正点 - 支持新旧两种格式
        if 'points_pos' in data:
            result['points_pos'] = data['points_pos']
        elif 'points_1' in data and 'points_2' in data:
            # 兼容旧格式
            result['points_pos'] = [data['points_1'], data['points_2']]
        
        # 解析负点
        if 'points_neg' in data:
            result['points_neg'] = data['points_neg']
        
        # 验证格式
        result['format_valid'] = (
            result['bbox'] is not None and
            len(result['bbox']) == 4 and
            len(result['points_pos']) >= 1
        )
        
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试正则匹配
        bbox_match = re.search(r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', answer_text)
        if bbox_match:
            result['bbox'] = [int(x) for x in bbox_match.groups()]
        
        # 正则匹配正点
        pos_match = re.search(r'"points_pos"\s*:\s*\[\[(\d+),\s*(\d+)\]', answer_text)
        if pos_match:
            result['points_pos'].append([int(pos_match.group(1)), int(pos_match.group(2))])
        
        # 正则匹配负点
        neg_matches = re.findall(r'"points_neg"\s*:\s*\[((?:\[\d+,\s*\d+\],?\s*)+)\]', answer_text)
        if neg_matches:
            for match in neg_matches:
                coords = re.findall(r'\[(\d+),\s*(\d+)\]', match)
                for coord in coords:
                    result['points_neg'].append([int(coord[0]), int(coord[1])])
    
    return result


def prepare_sam_prompts(parsed_output: Dict, image_size: Tuple[int, int] = (840, 840)) -> Dict:

    import numpy as np
    
    result = {
        'box': None,
        'point_coords': None,
        'point_labels': None
    }
    
    if parsed_output['bbox']:
        result['box'] = np.array(parsed_output['bbox'], dtype=np.float32)
    
    # 合并正点和负点
    all_points = []
    all_labels = []
    
    for pt in parsed_output['points_pos']:
        all_points.append(pt)
        all_labels.append(1)  # 正点标签
    
    for pt in parsed_output['points_neg']:
        all_points.append(pt)
        all_labels.append(0)  # 负点标签
    
    if all_points:
        result['point_coords'] = np.array(all_points, dtype=np.float32)
        result['point_labels'] = np.array(all_labels, dtype=np.int32)
    
    return result