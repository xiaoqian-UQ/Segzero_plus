# Seg-Zero负点预测与对比奖励：完整项目实施方案

## 项目概述

本项目旨在扩展Seg-Zero模型，使其能够预测负点(negative points)以提升SAM2的分割精度。核心思想是：通过强化学习训练VLM预测最优负点位置，利用对比奖励机制引导模型学会将负点放置在SAM容易误判的区域(False Positive区域)。

### 核心创新点

1. 扩展Seg-Zero输出格式，增加负点预测能力
2. 设计对比奖励机制，基于SAM的False Positive区域计算负点奖励
3. 将负点预测与GRPO强化学习框架结合

### 预期效果

在ReasonSeg基准上提升2-3 gIoU，超越RSVP的60.3 gIoU基准线。

---

## 第一部分：代码结构分析与修改计划

### 1.1 需要修改的核心文件

```
Seg-Zero/
├── src/
│   ├── train/
│   │   ├── grpo_seg_zero.py          # [修改] GRPO训练主循环
│   │   └── reward_functions.py        # [新建] 奖励计算函数
│   ├── eval/
│   │   └── evaluate.py                # [修改] 评估脚本
│   ├── utils/
│   │   ├── parser.py                  # [修改] 输出解析器
│   │   └── sam_utils.py               # [修改] SAM调用封装
│   └── models/
│       └── seg_zero_model.py          # [修改] 模型输出处理
├── configs/
│   └── negative_points_config.yaml    # [新建] 负点训练配置
└── scripts/
    └── train_negative_points.sh       # [新建] 训练启动脚本
```

### 1.2 数据流概览

```
输入: [图像, 推理查询]
    ↓
VLM (Qwen2.5-VL + LoRA)
    ↓
输出: <think>推理过程</think><answer>{"bbox":[...],"points":[...],"negative_points":[...]}</answer>
    ↓
解析器提取JSON中的坐标
    ↓
SAM2推理 (positive points + negative points + bbox)
    ↓
生成最终mask
    ↓
计算奖励 (IoU奖励 + 负点奖励)
    ↓
GRPO更新VLM参数
```

---

## 第二部分：输出格式扩展

### 2.1 原始Seg-Zero输出格式

```xml
<points>x1,y1;x2,y2</points><bbox>x1,y1,x2,y2</bbox>
```

### 2.2 扩展后的输出格式

```xml
<points>x1,y1;x2,y2</points><negative_points>nx1,ny1;nx2,ny2</negative_points><bbox>x1,y1,x2,y2</bbox>
```

### 2.3 格式规范

原始Seg-Zero的Answer格式（JSON）:
```json
{"bbox": [x1, y1, x2, y2], "points": [[px1, py1], [px2, py2]]}
```

扩展后的Answer格式:
```json
{"bbox": [x1, y1, x2, y2], "points": [[px1, py1], [px2, py2]], "negative_points": [[nx1, ny1]]}
```

| 字段 | 数量 | 坐标格式 | 说明 |
|------|------|----------|------|
| bbox | 1个 | 归一化[0,1] | 边界框，格式为[x1, y1, x2, y2] |
| points | 2个 | 归一化[0,1] | 正点，两个最大内切圆圆心 |
| negative_points | 1-2个 | 归一化[0,1] | 负点，标识应排除的混淆区域 |

### 2.4 Prompt模板修改

原始prompt:
```python
QUESTION_TEMPLATE = \
    "Please find '{Question}' with bbox and points." \
    "Compare the difference between objects and find the most closely matched one." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

# Answer格式:
# {"bbox": [x1, y1, x2, y2], "points": [[px1, py1], [px2, py2]]}
```

修改后prompt:
```python
QUESTION_TEMPLATE_WITH_NEGATIVE = \
    "Please find '{Question}' with bbox, points, and negative points." \
    "Compare the difference between objects and find the most closely matched one." \
    "Identify confusing background regions that should be excluded using negative points." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the one bbox, points of two largest inscribed circles inside the interested object, " \
    "and negative points in confusing background regions, all in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

# Answer格式:
# {"bbox": [x1, y1, x2, y2], "points": [[px1, py1], [px2, py2]], "negative_points": [[nx1, ny1]]}
```

### 2.5 完整输出示例

```
<think>
The query asks for "the cup that is not being used". I can see two cups in the image. 
One cup is being held by a person's hand, and another cup is sitting on the table untouched.
The cup on the table is the one not being used. I will identify its bounding box and 
two points inside it. There is another similar cup nearby that could be confused, 
so I will mark a negative point on the cup being held to help distinguish them.
</think>
<answer>{"bbox": [0.45, 0.32, 0.58, 0.51], "points": [[0.51, 0.38], [0.52, 0.45]], "negative_points": [[0.23, 0.41]]}</answer>
```

---

## 第三部分：解析器实现

### 3.1 输出解析器代码

```python
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
```

### 3.2 解析器测试用例

```python
# tests/test_parser.py

import pytest
from src.utils.parser import SegZeroOutputParser

def test_valid_output_with_think():
    parser = SegZeroOutputParser()
    output = '''<think>
The query asks for the unused cup. I see two cups, one held by a person and one on the table.
The cup on the table is unused. The nearby held cup could be confused with it.
</think>
<answer>{"bbox": [0.45, 0.32, 0.58, 0.51], "points": [[0.51, 0.38], [0.52, 0.45]], "negative_points": [[0.23, 0.41]]}</answer>'''
    
    result = parser.parse(output)
    
    assert result.is_valid
    assert result.thinking is not None
    assert "unused cup" in result.thinking
    assert len(result.positive_points) == 2
    assert len(result.negative_points) == 1
    assert result.bbox == (0.45, 0.32, 0.58, 0.51)

def test_valid_output_without_think():
    parser = SegZeroOutputParser()
    output = '<answer>{"bbox": [0.1, 0.2, 0.8, 0.9], "points": [[0.5, 0.5], [0.6, 0.6]], "negative_points": [[0.2, 0.2]]}</answer>'
    
    result = parser.parse(output)
    
    assert result.is_valid
    assert result.thinking is None
    assert len(result.positive_points) == 2

def test_multiple_negative_points():
    parser = SegZeroOutputParser()
    output = '<answer>{"bbox": [0.1, 0.1, 0.9, 0.9], "points": [[0.5, 0.5], [0.6, 0.6]], "negative_points": [[0.2, 0.2], [0.3, 0.3]]}</answer>'
    
    result = parser.parse(output)
    
    assert result.is_valid
    assert len(result.negative_points) == 2

def test_missing_negative_points_required():
    parser = SegZeroOutputParser(require_negative_points=True)
    output = '<answer>{"bbox": [0.1, 0.1, 0.9, 0.9], "points": [[0.5, 0.5], [0.6, 0.6]]}</answer>'
    
    result = parser.parse(output)
    
    assert not result.is_valid
    assert "negative_points" in result.error_message.lower()

def test_missing_negative_points_optional():
    parser = SegZeroOutputParser(require_negative_points=False)
    output = '<answer>{"bbox": [0.1, 0.1, 0.9, 0.9], "points": [[0.5, 0.5], [0.6, 0.6]]}</answer>'
    
    result = parser.parse(output)
    
    assert result.is_valid
    assert len(result.negative_points) == 0

def test_invalid_json():
    parser = SegZeroOutputParser()
    output = '<answer>{"bbox": [0.1, 0.1, 0.9, 0.9], "points": [[0.5, 0.5]</answer>'
    
    result = parser.parse(output)
    
    assert not result.is_valid
    assert "json" in result.error_message.lower()

def test_missing_answer_tag():
    parser = SegZeroOutputParser()
    output = '<think>some thinking</think>{"bbox": [0.1, 0.1, 0.9, 0.9]}'
    
    result = parser.parse(output)
    
    assert not result.is_valid
    assert "answer" in result.error_message.lower()

def test_out_of_range_coordinates():
    parser = SegZeroOutputParser()
    output = '<answer>{"bbox": [0.1, 0.1, 1.5, 0.9], "points": [[0.5, 0.5], [0.6, 0.6]], "negative_points": [[0.2, 0.2]]}</answer>'
    
    result = parser.parse(output)
    
    assert not result.is_valid
    assert "out of range" in result.error_message.lower()

def test_format_answer():
    answer_str = SegZeroOutputParser.format_answer(
        bbox=(0.1, 0.2, 0.8, 0.9),
        points=[(0.5, 0.5), (0.6, 0.6)],
        negative_points=[(0.2, 0.3)]
    )
    
    import json
    answer = json.loads(answer_str)
    assert answer["bbox"] == [0.1, 0.2, 0.8, 0.9]
    assert len(answer["points"]) == 2
    assert len(answer["negative_points"]) == 1
```

---

## 第四部分：SAM2调用封装

### 4.1 SAM2工具类

```python
# src/utils/sam_utils.py

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
        """
        Args:
            model_cfg: SAM2配置文件路径
            checkpoint: SAM2权重文件路径
            device: 计算设备
        """
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_cfg=model_cfg,
            checkpoint=checkpoint
        )
        self.predictor.model.to(device)
    
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
```

---

## 第五部分：奖励函数设计

### 5.1 奖励函数架构

奖励函数由三个部分组成:

```
R_total = R_mask + lambda_neg * R_neg + lambda_format * R_format
```

| 奖励项 | 说明 | 权重 |
|--------|------|------|
| R_mask | 最终mask的IoU奖励 | 1.0 |
| R_neg | 负点位置的对比奖励 | 0.3 |
| R_format | 输出格式正确性奖励 | 0.1 |

### 5.2 混淆区域定义（选项C实现）

```
混淆区域 = 基线mask预测为正 BUT 实际GT为负的区域
       = baseline_mask AND NOT gt_mask
       = False Positive区域
```

### 5.3 奖励函数实现

```python
# src/train/reward_functions.py

import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class RewardOutput:
    """奖励计算结果"""
    total_reward: float
    mask_reward: float
    negative_point_reward: float
    format_reward: float
    details: Dict

class NegativePointRewardCalculator:
    """负点对比奖励计算器"""
    
    def __init__(
        self,
        sam_wrapper,
        alpha: float = 1.0,      # 惩罚系数：负点落在GT内
        beta: float = 1.0,       # 奖励系数：负点落在混淆区域
        lambda_neg: float = 0.3, # 负点奖励权重
        lambda_format: float = 0.1  # 格式奖励权重
    ):
        """
        Args:
            sam_wrapper: SAM2Wrapper实例
            alpha: 负点落在GT内的惩罚系数
            beta: 负点落在混淆区域的奖励系数
            lambda_neg: 负点奖励在总奖励中的权重
            lambda_format: 格式奖励在总奖励中的权重
        """
        self.sam_wrapper = sam_wrapper
        self.alpha = alpha
        self.beta = beta
        self.lambda_neg = lambda_neg
        self.lambda_format = lambda_format
    
    def compute_reward(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        gt_mask: np.ndarray,
        format_valid: bool
    ) -> RewardOutput:
        """
        计算总奖励
        
        Args:
            image: 输入图像 (H, W, 3)
            positive_points: 正点列表
            negative_points: 负点列表
            bbox: 边界框
            gt_mask: Ground Truth mask (H, W)
            format_valid: 输出格式是否有效
            
        Returns:
            RewardOutput对象
        """
        H, W = image.shape[:2]
        
        # 1. 计算格式奖励
        format_reward = 1.0 if format_valid else 0.0
        
        if not format_valid:
            # 格式无效，返回最低奖励
            return RewardOutput(
                total_reward=0.0,
                mask_reward=0.0,
                negative_point_reward=0.0,
                format_reward=0.0,
                details={"error": "invalid_format"}
            )
        
        # 2. 使用正负点生成最终mask
        final_mask, confidence = self.sam_wrapper.predict_mask(
            image, positive_points, negative_points, bbox
        )
        
        # 3. 计算mask IoU奖励
        mask_reward = self._compute_iou(final_mask, gt_mask)
        
        # 4. 计算负点对比奖励
        neg_reward, neg_details = self._compute_negative_point_reward(
            image, positive_points, negative_points, bbox, gt_mask
        )
        
        # 5. 计算总奖励
        total_reward = (
            mask_reward +
            self.lambda_neg * neg_reward +
            self.lambda_format * format_reward
        )
        
        return RewardOutput(
            total_reward=total_reward,
            mask_reward=mask_reward,
            negative_point_reward=neg_reward,
            format_reward=format_reward,
            details={
                "iou": mask_reward,
                "sam_confidence": confidence,
                **neg_details
            }
        )
    
    def _compute_negative_point_reward(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        gt_mask: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        计算负点对比奖励
        
        核心逻辑:
        1. 用正点生成基线mask
        2. 混淆区域 = 基线mask中的False Positive区域
        3. 奖励负点落在混淆区域，惩罚负点落在GT内
        """
        H, W = gt_mask.shape
        
        if not negative_points:
            return 0.0, {"neg_in_gt": 0, "neg_in_confused": 0, "confused_area": 0}
        
        # 1. 生成基线mask（只用正点，不用负点）
        baseline_mask = self.sam_wrapper.predict_baseline_mask(
            image, positive_points, bbox
        )
        
        # 2. 计算混淆区域（False Positive）
        confused_region = baseline_mask.astype(bool) & ~gt_mask.astype(bool)
        confused_area = confused_region.sum()
        
        # 3. 统计负点位置
        neg_in_gt = 0
        neg_in_confused = 0
        
        for nx, ny in negative_points:
            px, py = int(nx * W), int(ny * H)
            px = min(max(px, 0), W - 1)
            py = min(max(py, 0), H - 1)
            
            if gt_mask[py, px]:
                neg_in_gt += 1
            if confused_region[py, px]:
                neg_in_confused += 1
        
        # 4. 计算奖励
        num_neg = len(negative_points)
        
        # 归一化：负点落在GT内的比例（惩罚），负点落在混淆区域的比例（奖励）
        penalty = (neg_in_gt / num_neg) if num_neg > 0 else 0
        bonus = (neg_in_confused / num_neg) if num_neg > 0 else 0
        
        reward = -self.alpha * penalty + self.beta * bonus
        
        details = {
            "neg_in_gt": neg_in_gt,
            "neg_in_confused": neg_in_confused,
            "num_negative_points": num_neg,
            "confused_area_pixels": int(confused_area),
            "confused_area_ratio": float(confused_area / (H * W))
        }
        
        return reward, details
    
    def _compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """计算IoU"""
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
```

### 5.4 奖励函数可视化调试工具

```python
# src/utils/reward_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def visualize_reward_components(
    image: np.ndarray,
    gt_mask: np.ndarray,
    baseline_mask: np.ndarray,
    final_mask: np.ndarray,
    positive_points: List[Tuple[float, float]],
    negative_points: List[Tuple[float, float]],
    save_path: str = None
):
    """
    可视化奖励计算的各个组件
    
    用于调试和验证奖励函数的正确性
    """
    H, W = image.shape[:2]
    
    # 计算混淆区域
    confused_region = baseline_mask.astype(bool) & ~gt_mask.astype(bool)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原图 + 点标注
    axes[0, 0].imshow(image)
    for x, y in positive_points:
        axes[0, 0].scatter(x * W, y * H, c='green', s=100, marker='o', label='Positive')
    for x, y in negative_points:
        axes[0, 0].scatter(x * W, y * H, c='red', s=100, marker='x', label='Negative')
    axes[0, 0].set_title('Input Image + Points')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth Mask
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    # 3. 基线Mask（只用正点）
    axes[0, 2].imshow(baseline_mask, cmap='gray')
    axes[0, 2].set_title('Baseline Mask (Pos Only)')
    axes[0, 2].axis('off')
    
    # 4. 混淆区域（False Positive）
    confused_vis = np.zeros((H, W, 3), dtype=np.uint8)
    confused_vis[confused_region] = [255, 0, 0]  # 红色
    confused_vis[gt_mask.astype(bool)] = [0, 255, 0]  # 绿色
    axes[1, 0].imshow(confused_vis)
    axes[1, 0].set_title('Confused Region (Red) vs GT (Green)')
    axes[1, 0].axis('off')
    
    # 5. 最终Mask（使用负点）
    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title('Final Mask (With Neg Points)')
    axes[1, 1].axis('off')
    
    # 6. IoU对比
    baseline_iou = compute_iou(baseline_mask, gt_mask)
    final_iou = compute_iou(final_mask, gt_mask)
    
    axes[1, 2].bar(['Baseline', 'Final'], [baseline_iou, final_iou], color=['orange', 'green'])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title(f'IoU Comparison\nBaseline: {baseline_iou:.3f} -> Final: {final_iou:.3f}')
    axes[1, 2].set_ylabel('IoU')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / union if union > 0 else 0.0
```

---

## 第六部分：GRPO训练流程修改

### 6.1 训练流程概览

```
原始GRPO流程:
1. 采样K个输出 -> 2. 解析bbox+点 -> 3. SAM生成mask -> 4. 计算IoU奖励 -> 5. GRPO更新

修改后流程:
1. 采样K个输出 -> 2. 解析bbox+正点+负点 -> 3. 计算基线mask -> 4. 计算混淆区域 
   -> 5. SAM生成最终mask -> 6. 计算综合奖励 -> 7. GRPO更新
```

### 6.2 训练脚本修改

```python
# src/train/grpo_seg_zero_negative.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any
import numpy as np

from src.utils.parser import SegZeroOutputParser
from src.utils.sam_utils import SAM2Wrapper
from src.train.reward_functions import NegativePointRewardCalculator

class GRPOTrainerWithNegativePoints:
    """支持负点预测的GRPO训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = config.get("device", "cuda")
        
        # 初始化模型
        self.model = self._init_model()
        self.tokenizer = self._init_tokenizer()
        
        # 初始化工具
        self.parser = SegZeroOutputParser(require_negative_points=True)
        self.sam_wrapper = SAM2Wrapper(
            model_cfg=config["sam_config"],
            checkpoint=config["sam_checkpoint"],
            device=self.device
        )
        self.reward_calculator = NegativePointRewardCalculator(
            sam_wrapper=self.sam_wrapper,
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            lambda_neg=config.get("lambda_neg", 0.3),
            lambda_format=config.get("lambda_format", 0.1)
        )
        
        # GRPO参数
        self.group_size = config.get("group_size", 8)
        self.clip_lower = config.get("clip_lower", -0.2)
        self.clip_upper = config.get("clip_upper", 0.28)
        self.temperature = config.get("temperature", 0.7)
        
    def _init_model(self):
        """初始化模型并添加LoRA"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA配置
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 64),
            lora_alpha=self.config.get("lora_alpha", 128),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _init_tokenizer(self):
        """初始化tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            trust_remote_code=True
        )
        return tokenizer
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 包含image, query, gt_mask的批次数据
            
        Returns:
            训练指标字典
        """
        images = batch["image"]
        queries = batch["query"]
        gt_masks = batch["gt_mask"]
        
        batch_size = len(images)
        all_rewards = []
        all_log_probs = []
        
        for i in range(batch_size):
            image = images[i]
            query = queries[i]
            gt_mask = gt_masks[i]
            
            # 构建prompt
            prompt = self._build_prompt(query)
            
            # 采样K个输出
            outputs, log_probs = self._sample_outputs(prompt, self.group_size)
            
            # 计算每个输出的奖励
            rewards = []
            for output in outputs:
                parsed = self.parser.parse(output)
                reward_output = self.reward_calculator.compute_reward(
                    image=image,
                    positive_points=parsed.positive_points,
                    negative_points=parsed.negative_points,
                    bbox=parsed.bbox,
                    gt_mask=gt_mask,
                    format_valid=parsed.is_valid
                )
                rewards.append(reward_output.total_reward)
            
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
        
        # GRPO更新
        loss = self._compute_grpo_loss(all_rewards, all_log_probs)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_reward": np.mean([r for rewards in all_rewards for r in rewards]),
            "max_reward": np.max([r for rewards in all_rewards for r in rewards])
        }
    
    def _build_prompt(self, query: str) -> str:
        """构建包含负点说明的prompt"""
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
    
    def _sample_outputs(self, prompt: str, k: int) -> tuple:
        """采样K个输出及其log概率"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = []
        log_probs = []
        
        for _ in range(k):
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=self.temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            output_text = self.tokenizer.decode(
                generated.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            outputs.append(output_text)
            
            # 计算log概率
            log_prob = self._compute_log_prob(generated)
            log_probs.append(log_prob)
        
        return outputs, log_probs
    
    def _compute_log_prob(self, generated) -> torch.Tensor:
        """计算生成序列的log概率"""
        scores = generated.scores
        sequences = generated.sequences
        
        log_prob = 0
        for i, score in enumerate(scores):
            token_id = sequences[0, i + 1]
            log_prob += torch.log_softmax(score, dim=-1)[0, token_id]
        
        return log_prob
    
    def _compute_grpo_loss(
        self,
        all_rewards: List[List[float]],
        all_log_probs: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        计算GRPO损失
        
        使用组内相对奖励进行优势估计
        """
        total_loss = 0
        count = 0
        
        for rewards, log_probs in zip(all_rewards, all_log_probs):
            # 计算组内平均奖励
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            
            # 计算优势（相对于组内平均）
            advantages = [(r - mean_reward) / std_reward for r in rewards]
            
            # 裁剪优势
            advantages = [
                np.clip(a, self.clip_lower, self.clip_upper)
                for a in advantages
            ]
            
            # 计算损失
            for log_prob, advantage in zip(log_probs, advantages):
                total_loss -= log_prob * advantage
                count += 1
        
        return total_loss / count
```

### 6.3 配置文件

```yaml
# configs/negative_points_config.yaml

# 模型配置
model_path: "Qwen/Qwen2.5-VL-7B-Instruct"
sam_config: "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_checkpoint: "checkpoints/sam2.1_hiera_large.pt"

# LoRA配置
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05

# GRPO配置
group_size: 8
clip_lower: -0.2
clip_upper: 0.28
temperature: 0.7

# 奖励配置
alpha: 1.0       # 负点落在GT内的惩罚系数
beta: 1.0        # 负点落在混淆区域的奖励系数
lambda_neg: 0.3  # 负点奖励权重
lambda_format: 0.1  # 格式奖励权重

# 训练配置
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
warmup_steps: 100
max_steps: 5000
save_steps: 500
eval_steps: 200

# 数据配置
train_data:
  arrow_dir: "refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e"
  mask_dir: "/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks"
val_data: "data/ReasonSeg/val"

# 图像大小 (与mask匹配)
image_size: 840

# DeepSpeed配置
deepspeed:
  zero_stage: 2
  offload_optimizer: false
  offload_param: false

# 设备配置
device: "cuda"
num_gpus: 2
```

---

## 第七部分：数据加载与预处理

### 7.1 数据目录结构

```
# Arrow数据集 (包含id, problem, solution, image等)
refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/.../
├── dataset_info.json
├── ref_coc_og_9k_840-train-00000-of-00016.arrow
└── ... (16个分片)

# GT Mask和图像 (840x840)
/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/
├── gt_masks/      # GT分割mask, 命名格式: refcocog_{id}.png
├── images/        # 图像文件
└── overlays/      # 可视化叠加图
```

### 7.2 RefCOCOg数据集格式

Arrow文件中的数据结构:
```python
{
    "id": "refcocog_16521",                    # 样本ID，对应mask文件名
    "problem": "A black and white dog...",     # referring expression (query)
    "solution": "<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>",  # 像素坐标
    "image": <PIL.Image 840x840>,              # 已调整大小的图像
    "img_height": 426,                         # 原始图像高度
    "img_width": 640                           # 原始图像宽度
}
```

关键点:
- `id`字段对应mask文件名：`refcocog_16521` -> `gt_masks/refcocog_16521.png`
- `solution`中的坐标是**像素坐标**，需要归一化到[0,1]
- 图像已经是840x840，与mask尺寸一致
- 没有negative_points，这是模型需要学习预测的

### 7.3 Solution解析器

```python
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
```

### 7.4 数据集类

```python
# src/data/dataset.py

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any, List
import glob

from src.utils.solution_parser import SolutionParser

class RefCOCOgDataset(Dataset):
    """RefCOCOg数据集加载器，用于Seg-Zero负点训练"""
    
    def __init__(
        self,
        arrow_dir: str,
        mask_dir: str,
        image_size: int = 840
    ):
        """
        Args:
            arrow_dir: Arrow文件目录路径
            mask_dir: GT mask目录路径 (例如 /mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks)
            image_size: 图像大小 (默认840)
        """
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.solution_parser = SolutionParser(image_size=image_size)
        
        # 加载HuggingFace数据集
        arrow_files = os.path.join(arrow_dir, "*.arrow")
        self.dataset = load_dataset('arrow', data_files=arrow_files, split='train')
        
        print(f"Loaded {len(self.dataset)} samples")
        print(f"Columns: {self.dataset.column_names}")
        
        # 验证mask目录
        if os.path.exists(mask_dir):
            mask_count = len([f for f in os.listdir(mask_dir) if f.endswith('.png')])
            print(f"Found {mask_count} mask files in {mask_dir}")
        else:
            print(f"Warning: Mask directory not found: {mask_dir}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # 获取ID (用于匹配mask文件)
        sample_id = item["id"]  # 例如 "refcocog_16521"
        
        # 加载图像
        image = item["image"]
        if hasattr(image, 'convert'):
            image = image.convert("RGB")
        image_np = np.array(image)
        
        # 确保图像尺寸正确
        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            from PIL import Image as PILImage
            image = PILImage.fromarray(image_np).resize((self.image_size, self.image_size))
            image_np = np.array(image)
        
        # 解析solution (获取bbox和points)
        solution_data = self.solution_parser.parse(item["solution"])
        
        if not solution_data.is_valid:
            print(f"Warning: Invalid solution for {sample_id}: {solution_data.error_message}")
        
        # 加载GT mask
        gt_mask = self._load_mask(sample_id)
        
        return {
            "image": image_np,
            "query": item["problem"],
            "gt_mask": gt_mask.astype(np.float32),
            "gt_bbox": solution_data.bbox,
            "gt_points": solution_data.points,
            "sample_id": sample_id,
            "image_id": idx,
            "original_size": (item["img_width"], item["img_height"])
        }
    
    def _load_mask(self, sample_id: str) -> np.ndarray:
        """
        加载GT mask
        
        Args:
            sample_id: 样本ID，例如 "refcocog_16521"
            
        Returns:
            二值mask数组
        """
        mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = mask > 127  # 转为二值
            
            # 确保尺寸正确
            if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
                mask = np.array(mask_pil) > 127
            
            return mask
        else:
            print(f"Warning: Mask not found for {sample_id}, using bbox mask")
            return self._create_bbox_mask(sample_id)
    
    def _create_bbox_mask(self, sample_id: str) -> np.ndarray:
        """当mask文件不存在时，用bbox创建粗略mask"""
        # 找到对应的item
        for i in range(len(self.dataset)):
            if self.dataset[i]["id"] == sample_id:
                solution_data = self.solution_parser.parse(self.dataset[i]["solution"])
                break
        else:
            return np.zeros((self.image_size, self.image_size), dtype=bool)
        
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        x1, y1, x2, y2 = solution_data.bbox_pixel
        mask[y1:y2, x1:x2] = True
        return mask


def collate_fn(batch: list) -> Dict[str, Any]:
    """自定义批次整理函数"""
    return {
        "image": [item["image"] for item in batch],
        "query": [item["query"] for item in batch],
        "gt_mask": [item["gt_mask"] for item in batch],
        "gt_bbox": [item["gt_bbox"] for item in batch],
        "gt_points": [item["gt_points"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "original_size": [item["original_size"] for item in batch]
    }


def create_dataloader(
    arrow_dir: str,
    mask_dir: str,
    batch_size: int = 2,
    image_size: int = 840,
    num_workers: int = 4,
    shuffle: bool = True
):
    """
    创建数据加载器
    
    使用示例:
    ```python
    dataloader = create_dataloader(
        arrow_dir="refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e",
        mask_dir="/mnt/xiaoqian/dataset/refcocog/ref_coc_og_9k_840/gt_masks",
        batch_size=2
    )
    ```
    """
    from torch.utils.data import DataLoader
    
    dataset = RefCOCOgDataset(
        arrow_dir=arrow_dir,
        mask_dir=mask_dir,
        image_size=image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader
```

### 7.5 数据加载测试脚本

```python
# scripts/test_dataloader.py

from src.data.dataset import create_dataloader
import matplotlib.pyplot as plt
import numpy as np

# 配置路径 - 根据你的实际路径修改
ARROW_DIR = "refcocog/Ricky06662___ref_coc_og_9k_840/default/0.0.0/eb5ec70f57b92d0eacccbdc817e487da3292876e"
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
```

### 7.6 Solution格式对比

原始Seg-Zero训练时的answer格式与数据集solution格式不同：

| 格式 | 示例 | 用途 |
|------|------|------|
| 数据集solution | `<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>` | 像素坐标，用于GT |
| 模型answer (原始) | `{"bbox": [0.0, 0.54, 0.45, 0.8], "points": [[0.06, 0.7], [0.17, 0.72]]}` | 归一化坐标JSON |
| 模型answer (扩展) | `{"bbox": [...], "points": [...], "negative_points": [...]}` | 增加负点 |

在训练过程中：
1. 从数据集加载GT bbox/points（像素坐标），归一化后用于计算reward
2. 模型输出归一化坐标的JSON格式
3. 两者对比计算reward

---

## 第八部分：评估脚本

### 8.1 评估指标

```python
# src/eval/metrics.py

import numpy as np
from typing import Dict, List

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算IoU"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    return float(intersection / union) if union > 0 else 0.0

def compute_giou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算Generalized IoU (gIoU)"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    
    # 计算最小包围框
    pred_indices = np.where(pred)
    gt_indices = np.where(gt)
    
    if len(pred_indices[0]) == 0 or len(gt_indices[0]) == 0:
        return iou
    
    all_y = np.concatenate([pred_indices[0], gt_indices[0]])
    all_x = np.concatenate([pred_indices[1], gt_indices[1]])
    
    enclosing_box_area = (all_y.max() - all_y.min() + 1) * (all_x.max() - all_x.min() + 1)
    
    giou = iou - (enclosing_box_area - union) / enclosing_box_area
    
    return float(giou)

def compute_boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation: int = 5) -> float:
    """计算Boundary IoU（仅在边界区域计算）"""
    from scipy import ndimage
    
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # 计算边界
    pred_boundary = pred ^ ndimage.binary_erosion(pred, iterations=dilation)
    gt_boundary = gt ^ ndimage.binary_erosion(gt, iterations=dilation)
    
    # 在边界区域计算IoU
    boundary_region = pred_boundary | gt_boundary
    
    if boundary_region.sum() == 0:
        return 1.0
    
    intersection = (pred & gt & boundary_region).sum()
    union = ((pred | gt) & boundary_region).sum()
    
    return float(intersection / union) if union > 0 else 0.0

class SegmentationMetrics:
    """分割指标汇总"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.ious = []
        self.gious = []
        self.boundary_ious = []
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        self.ious.append(compute_iou(pred, gt))
        self.gious.append(compute_giou(pred, gt))
        self.boundary_ious.append(compute_boundary_iou(pred, gt))
    
    def compute(self) -> Dict[str, float]:
        return {
            "mIoU": np.mean(self.ious) * 100,
            "gIoU": np.mean(self.gious) * 100,
            "Boundary_IoU": np.mean(self.boundary_ious) * 100,
            "num_samples": len(self.ious)
        }
```

### 8.2 评估脚本

```python
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
```

---

## 第九部分：训练启动脚本

### 9.1 单机多卡训练脚本

```bash
#!/bin/bash
# scripts/train_negative_points.sh

# 配置
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29500

# 路径
CONFIG="configs/negative_points_config.yaml"
OUTPUT_DIR="outputs/negative_points_exp1"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练
deepspeed --num_gpus=2 \
    src/train/grpo_seg_zero_negative.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --deepspeed configs/deepspeed_zero2.json \
    2>&1 | tee $OUTPUT_DIR/train.log
```

### 9.2 DeepSpeed配置

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": false
}
```

---

## 第十部分：实验计划与消融实验

### 10.1 实验阶段

| 阶段 | 目标 | 时间估计 |
|------|------|----------|
| 阶段1 | 代码实现与调试 | 3-5天 |
| 阶段2 | 基线复现 | 1-2天 |
| 阶段3 | 负点训练 | 2-3天 |
| 阶段4 | 消融实验 | 3-5天 |
| 阶段5 | 最终评估与分析 | 2-3天 |

### 10.2 消融实验设计

| 实验ID | 配置 | 目的 |
|--------|------|------|
| A1 | 原始Seg-Zero | 基线 |
| A2 | +负点预测，无对比奖励 | 验证负点本身的作用 |
| A3 | +负点预测，+对比奖励 | 完整方案 |
| A4 | 变化alpha (0.5, 1.0, 2.0) | 惩罚系数敏感性 |
| A5 | 变化beta (0.5, 1.0, 2.0) | 奖励系数敏感性 |
| A6 | 变化lambda_neg (0.1, 0.3, 0.5) | 负点奖励权重敏感性 |
| A7 | 1个负点 vs 2个负点 | 负点数量影响 |

### 10.3 评估基准

| 数据集 | 指标 | Seg-Zero基线 | 目标 |
|--------|------|--------------|------|
| ReasonSeg val | gIoU | 57.5 | 60+ |
| ReasonSeg val | cIoU | 58.2 | 61+ |
| RefCOCO val | oIoU | - | 提升1-2% |
| RefCOCO+ val | oIoU | - | 提升1-2% |
| RefCOCOg val | oIoU | - | 提升1-2% |

---

## 第十一部分：实施检查清单

### 11.1 代码实现检查清单

- [ ] 输出解析器实现与测试
- [ ] SAM2封装类实现
- [ ] 奖励函数实现
  - [ ] IoU奖励
  - [ ] 负点对比奖励
  - [ ] 格式奖励
- [ ] GRPO训练循环修改
- [ ] 数据加载器
- [ ] 评估脚本
- [ ] 可视化调试工具

### 11.2 训练前检查清单

- [ ] 确认Seg-Zero-7B checkpoint可用
- [ ] 确认SAM2.1-Large checkpoint可用
- [ ] 确认ReasonSeg数据集完整
- [ ] 验证GPU内存足够（每卡40GB左右）
- [ ] 运行小规模测试（100步）验证流程

### 11.3 训练中监控指标

- [ ] 总奖励曲线
- [ ] Mask IoU奖励曲线
- [ ] 负点奖励曲线
- [ ] 负点落在GT内的比例（应下降）
- [ ] 负点落在混淆区域的比例（应上升）
- [ ] 格式正确率（应保持高位）

---

## 第十二部分：常见问题与解决方案

### 12.1 训练问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| CUDA OOM | batch size过大 | 减小batch_size，增大gradient_accumulation_steps |
| 奖励不收敛 | lambda_neg过大 | 减小lambda_neg到0.1-0.2 |
| 格式错误率高 | 温度过高 | 降低temperature到0.5-0.6 |
| 负点全落在GT内 | alpha过小 | 增大alpha到1.5-2.0 |

### 12.2 评估问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| gIoU下降 | 过拟合 | 增加数据增强，减少训练步数 |
| 推理速度慢 | SAM调用过多 | 使用batch推理 |

---

## 附录A：关键代码位置索引

```
项目根目录/
├── src/
│   ├── train/
│   │   ├── grpo_seg_zero_negative.py    # 主训练脚本
│   │   └── reward_functions.py           # 奖励计算
│   ├── eval/
│   │   ├── evaluate.py                   # 评估主脚本
│   │   └── metrics.py                    # 评估指标
│   ├── utils/
│   │   ├── parser.py                     # 输出解析
│   │   ├── sam_utils.py                  # SAM封装
│   │   └── reward_visualizer.py          # 可视化工具
│   └── data/
│       └── dataset.py                    # 数据加载
├── configs/
│   ├── negative_points_config.yaml       # 主配置
│   └── deepspeed_zero2.json              # DeepSpeed配置
├── scripts/
│   └── train_negative_points.sh          # 训练脚本
└── tests/
    └── test_parser.py                    # 单元测试
```

---

## 附录B：预期时间线

```
第1周: 代码实现
  - Day 1-2: 解析器、SAM封装
  - Day 3-4: 奖励函数、GRPO修改
  - Day 5: 集成测试、bug修复

第2周: 训练与调试
  - Day 1: 基线复现
  - Day 2-4: 负点训练主实验
  - Day 5: 初步结果分析

第3周: 消融与优化
  - Day 1-3: 消融实验
  - Day 4-5: 超参数调优

第4周: 最终评估与文档
  - Day 1-2: 完整评估
  - Day 3-5: 结果分析、可视化、文档
```
