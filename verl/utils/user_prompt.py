# 原始 prompt (参考论文 Figure 4)
ORIGINAL_USER_PROMPT = """
Please find '{Question}' with bbox and points.
Compare the difference between objects and find the most closely matched one.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.
i.e., <think> thinking process here </think>
<answer> { 'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180] } </answer>
"""

# 新的 prompt (加入负点说明)
NEGATIVE_POINTS_USER_PROMPT = """
Please find '{Question}' with bbox, positive points, and negative points.
Compare the difference between objects and find the most closely matched one.
Identify confusing background regions that might be mistaken for the target object.

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
Output:
1. One bbox of the target object
2. Center points of two largest inscribed circles inside the target (positive points)
3. 1-2 negative points in confusing background regions that should NOT be included

i.e., <think> thinking process here </think>
<answer> { 
    "bbox": [10,100,200,210], 
    "points_1": [30,110], 
    "points_2": [35,180],
    "negative_points": [[250,150], [300,200]]
} </answer>
"""