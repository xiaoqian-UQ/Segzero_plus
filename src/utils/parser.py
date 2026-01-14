# src/utils/parser.py

import re
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class SegmentationOutput:
    """存储解析后的分割输出"""
    positive_points: List[Tuple[float, float]]  # [(x1,y1), (x2,y2)]
    negative_points: List[Tuple[float, float]]  # [(nx1,ny1), ...]
    bbox: Tuple[float, float, float, float]     # (x1, y1, x2, y2)
    thinking: Optional[str]                      # 思考过程
    is_valid: bool
    error_message: Optional[str] = None

class SegZeroOutputParser:
    """解析Seg-Zero模型输出（支持<think><answer>格式和JSON）"""
    
    # 正则表达式模式
    THINK_PATTERN = r'<think>(.*?)</think>'
    ANSWER_PATTERN = r'<answer>(.*?)</answer>'
    
    def __init__(self, require_negative_points: bool = True):
        """
        Args:
            require_negative_points: 是否要求必须包含负点
        """
        self.require_negative_points = require_negative_points
    
    def parse(self, output: str) -> SegmentationOutput:
        """
        解析模型输出字符串
        
        期望格式:
        <think>thinking process</think>
        <answer>{"bbox": [x1,y1,x2,y2], "points": [[p1x,p1y],[p2x,p2y]], "negative_points": [[n1x,n1y]]}</answer>
        
        Args:
            output: 模型生成的字符串
            
        Returns:
            SegmentationOutput对象
        """
        try:
            # 提取thinking部分（可选）
            think_match = re.search(self.THINK_PATTERN, output, re.DOTALL)
            thinking = think_match.group(1).strip() if think_match else None
            
            # 提取answer部分
            answer_match = re.search(self.ANSWER_PATTERN, output, re.DOTALL)
            if not answer_match:
                return self._create_invalid_output("Missing <answer> tag")
            
            answer_str = answer_match.group(1).strip()
            
            # 解析JSON
            try:
                answer_json = json.loads(answer_str)
            except json.JSONDecodeError as e:
                return self._create_invalid_output(f"Invalid JSON in answer: {str(e)}")
            
            # 提取bbox
            if "bbox" not in answer_json:
                return self._create_invalid_output("Missing 'bbox' in answer")
            bbox_list = answer_json["bbox"]
            if not isinstance(bbox_list, list) or len(bbox_list) != 4:
                return self._create_invalid_output("Invalid bbox format, expected [x1,y1,x2,y2]")
            bbox = tuple(float(x) for x in bbox_list)
            
            # 提取正点
            if "points" not in answer_json:
                return self._create_invalid_output("Missing 'points' in answer")
            points_list = answer_json["points"]
            if not isinstance(points_list, list) or len(points_list) < 1:
                return self._create_invalid_output("Invalid points format")
            positive_points = [(float(p[0]), float(p[1])) for p in points_list]
            
            # 提取负点
            negative_points = []
            if "negative_points" in answer_json:
                neg_list = answer_json["negative_points"]
                if isinstance(neg_list, list):
                    negative_points = [(float(p[0]), float(p[1])) for p in neg_list]
            
            if self.require_negative_points and not negative_points:
                return self._create_invalid_output("Missing 'negative_points' in answer")
            
            # 验证坐标范围
            if not self._validate_coordinates(positive_points, negative_points, bbox):
                return self._create_invalid_output("Coordinates out of range [0, 1]")
            
            return SegmentationOutput(
                positive_points=positive_points,
                negative_points=negative_points,
                bbox=bbox,
                thinking=thinking,
                is_valid=True
            )
            
        except Exception as e:
            return self._create_invalid_output(f"Parse error: {str(e)}")
    
    def _validate_coordinates(
        self,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """验证所有坐标在[0,1]范围内"""
        for x, y in positive_points + negative_points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return False
        
        for coord in bbox:
            if not (0 <= coord <= 1):
                return False
        
        return True
    
    def _create_invalid_output(self, error_message: str) -> SegmentationOutput:
        """创建无效输出对象"""
        return SegmentationOutput(
            positive_points=[],
            negative_points=[],
            bbox=(0, 0, 0, 0),
            thinking=None,
            is_valid=False,
            error_message=error_message
        )
    
    @staticmethod
    def format_answer(
        bbox: Tuple[float, float, float, float],
        points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]] = None
    ) -> str:
        """
        格式化输出为标准answer字符串
        
        用于构建训练数据或参考输出
        """
        answer_dict = {
            "bbox": list(bbox),
            "points": [list(p) for p in points]
        }
        if negative_points:
            answer_dict["negative_points"] = [list(p) for p in negative_points]
        
        return json.dumps(answer_dict)