# src/utils/solution_parser.py

import re
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class SolutionData:
    """解析后的solution数据"""
    bbox: Tuple[float, float, float, float]  # 归一化坐标 (x1, y1, x2, y2)
    points: List[Tuple[float, float]]         # 归一化坐标 [(x1,y1), (x2,y2)]
    bbox_pixel: Tuple[int, int, int, int]     # 像素坐标
    points_pixel: List[Tuple[int, int]]       # 像素坐标
    is_valid: bool
    error_message: Optional[str] = None

class SolutionParser:
    """
    解析Seg-Zero数据集中的solution字符串
    
    格式: <box>(x1,y1),(x2,y2)</box><points>(px1,py1),(px2,py2)</points>
    坐标为像素坐标，需要归一化
    """
    
    BOX_PATTERN = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    POINTS_PATTERN = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
    
    def __init__(self, image_size: int = 840):
        """
        Args:
            image_size: 图像尺寸，用于归一化坐标
        """
        self.image_size = image_size
    
    def parse(self, solution: str) -> SolutionData:
        """
        解析solution字符串
        
        Args:
            solution: 格式如 "<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>"
            
        Returns:
            SolutionData对象
        """
        try:
            # 解析bbox
            box_match = re.search(self.BOX_PATTERN, solution)
            if not box_match:
                return self._create_invalid("Missing or invalid <box> tag")
            
            x1, y1, x2, y2 = map(int, box_match.groups())
            bbox_pixel = (x1, y1, x2, y2)
            
            # 归一化bbox
            bbox = (
                x1 / self.image_size,
                y1 / self.image_size,
                x2 / self.image_size,
                y2 / self.image_size
            )
            
            # 解析points
            points_match = re.search(self.POINTS_PATTERN, solution)
            if not points_match:
                return self._create_invalid("Missing or invalid <points> tag")
            
            px1, py1, px2, py2 = map(int, points_match.groups())
            points_pixel = [(px1, py1), (px2, py2)]
            
            # 归一化points
            points = [
                (px1 / self.image_size, py1 / self.image_size),
                (px2 / self.image_size, py2 / self.image_size)
            ]
            
            return SolutionData(
                bbox=bbox,
                points=points,
                bbox_pixel=bbox_pixel,
                points_pixel=points_pixel,
                is_valid=True
            )
            
        except Exception as e:
            return self._create_invalid(f"Parse error: {str(e)}")
    
    def _create_invalid(self, error_message: str) -> SolutionData:
        return SolutionData(
            bbox=(0, 0, 0, 0),
            points=[],
            bbox_pixel=(0, 0, 0, 0),
            points_pixel=[],
            is_valid=False,
            error_message=error_message
        )
    
    @staticmethod
    def format_solution(
        bbox: Tuple[float, float, float, float],
        points: List[Tuple[float, float]],
        image_size: int = 840
    ) -> str:
        """
        将归一化坐标格式化为solution字符串
        
        用于验证或生成训练数据
        """
        x1, y1, x2, y2 = [int(c * image_size) for c in bbox]
        px1, py1 = int(points[0][0] * image_size), int(points[0][1] * image_size)
        px2, py2 = int(points[1][0] * image_size), int(points[1][1] * image_size)
        
        return f"<box>({x1},{y1}),({x2},{y2})</box><points>({px1},{py1}),({px2},{py2})</points>"