# 原始Seg-Zero Prompt
ORIGINAL_PROMPT = """
Please find '{Question}' with bbox and points.
Compare the difference between objects and find the most closely matched one.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.
i.e., <think> thinking process here </think>
<answer> { 'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180] } </answer>
"""

# 新的带负点的Prompt
NEGATIVE_POINT_PROMPT = """
Please find '{Question}' with bbox, positive points, and negative points.
Compare the difference between objects and find the most closely matched one.
Identify confusing background regions that should be EXCLUDED from the segmentation.

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.

Output format in JSON:
- bbox: bounding box of the target object [x1, y1, x2, y2]
- points_pos: two positive points inside the target object [[x1,y1], [x2,y2]]
- points_neg: 1-3 negative points on confusing background regions [[x1,y1], ...] 
  (regions that look similar to target but should NOT be segmented)

Example:
<think> 
The query asks for "the person wearing red". 
There are two people in the image - one wearing red (target) and one wearing orange (similar, confusing).
I will place positive points on the person in red, and negative points on the orange clothing to help distinguish them.
</think>
<answer> {{ "bbox": [10,100,200,210], "points_pos": [[30,110], [35,180]], "points_neg": [[250,150]] }} </answer>
"""