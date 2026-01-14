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