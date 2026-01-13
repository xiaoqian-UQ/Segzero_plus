# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score
from verl.utils.reward_score.segzero_plus_reward import SegZeroPlusConfig, SegZeroPlusRewardFunction


class CustomRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: str,
        custom_config: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.custom_config = custom_config
        reward_function = compute_score
        if custom_config is not None and getattr(custom_config, "reward_function", None):
            reward_function = custom_config.reward_function

        self.segzero_plus_reward_fn = None
        self.confused_regions_dir = None

        if reward_function == "segzero_plus":
            config_dict = None
            if custom_config is not None:
                config_dict = {}
                for key in SegZeroPlusConfig.__dataclass_fields__.keys():
                    if hasattr(custom_config, key):
                        config_dict[key] = getattr(custom_config, key)
            self.segzero_plus_reward_fn = SegZeroPlusRewardFunction(config_dict)
            if custom_config is not None and getattr(custom_config, "confused_regions_dir", None):
                self.confused_regions_dir = custom_config.confused_regions_dir
        elif reward_function == "math":
            self.compute_score = math_compute_score
        elif reward_function == "r1v":
            self.compute_score = r1v_compute_score
        elif reward_function == "seg":
            self.compute_score = seg_compute_score
        elif reward_function == "seg_strict":
            self.compute_score = seg_strict_compute_score
        else:
            raise NotImplementedError()

    def _normalize_mask(self, mask_value: Any) -> Optional[np.ndarray]:
        if mask_value is None:
            return None
        if isinstance(mask_value, str):
            if not os.path.exists(mask_value):
                return None
            mask_value = Image.open(mask_value)
        if isinstance(mask_value, Image.Image):
            mask = np.array(mask_value)
        elif isinstance(mask_value, torch.Tensor):
            mask = mask_value.detach().cpu().numpy()
        else:
            mask = np.asarray(mask_value)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def _extract_gt_mask(self, non_tensor: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("gt_mask", "mask"):
            if key in non_tensor and non_tensor[key] is not None:
                return self._normalize_mask(non_tensor[key])
        for key in ("mask_path", "mask_filepath"):
            if key in non_tensor and non_tensor[key]:
                return self._normalize_mask(non_tensor[key])
        return None

    def _extract_gt_bbox(self, non_tensor: Dict[str, Any], gt_mask: Optional[np.ndarray]) -> Optional[list]:
        for key in ("gt_bbox", "bbox", "box"):
            if key in non_tensor and non_tensor[key] is not None:
                bbox = non_tensor[key]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.detach().cpu().tolist()
                elif isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                elif isinstance(bbox, (list, tuple)):
                    bbox = list(bbox)
                else:
                    return None
                return [float(x) for x in bbox]

        if gt_mask is not None:
            coords = np.argwhere(gt_mask > 0)
            if coords.size:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                return [float(x_min), float(y_min), float(x_max), float(y_max)]

        solution = non_tensor.get("solution")
        if isinstance(solution, str):
            match = re.search(r"<box>\\((\\d+),(\\d+)\\),\\((\\d+),(\\d+)\\)</box>", solution)
            if match:
                return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]

        return None

    def _extract_confused_regions(self, non_tensor: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("confused_regions", "confused_region"):
            if key in non_tensor and non_tensor[key] is not None:
                return self._normalize_mask(non_tensor[key])
        for key in ("confused_region_path", "confused_regions_path"):
            if key in non_tensor and non_tensor[key]:
                path = non_tensor[key]
                if isinstance(path, str) and os.path.exists(path):
                    return np.load(path)
        if self.confused_regions_dir:
            idx = None
            for key in ("confused_region_id", "index", "idx", "sample_idx", "image_id"):
                if key in non_tensor and non_tensor[key] is not None:
                    value = non_tensor[key]
                    if isinstance(value, (int, np.integer)):
                        idx = int(value)
                        break
                    if isinstance(value, str) and value.isdigit():
                        idx = int(value)
                        break
            if idx is not None:
                candidate = os.path.join(self.confused_regions_dir, f"confused_{idx:06d}.npy")
                if os.path.exists(candidate):
                    return np.load(candidate)
        return None

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        if self.segzero_plus_reward_fn is not None:
            responses = []
            response_lengths = []
            prompt_strs = []
            gt_bboxes = []
            gt_masks = []
            confused_regions = []

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                prompt_strs.append(prompt_str)
                responses.append(response_str)
                response_lengths.append(valid_response_length)

                non_tensor = data_item.non_tensor_batch
                gt_mask = self._extract_gt_mask(non_tensor)
                gt_bbox = self._extract_gt_bbox(non_tensor, gt_mask)
                if self.segzero_plus_reward_fn.config.use_confused_regions:
                    confused = self._extract_confused_regions(non_tensor)
                else:
                    confused = None

                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)
                confused_regions.append(confused)

            scores = self.segzero_plus_reward_fn(
                responses,
                {"gt_bbox": gt_bboxes, "gt_mask": gt_masks, "confused_regions": confused_regions},
            )

            for i, score in enumerate(scores):
                if response_lengths[i] > 0:
                    reward_tensor[i, response_lengths[i] - 1] = float(score)
                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_strs[i])
                    print("[response]", responses[i])
                    print("[gt_bbox]", gt_bboxes[i])
                    print("[gt_mask]", None if gt_masks[i] is None else gt_masks[i].shape)
                    print("[score]", score)

            return reward_tensor

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ground_truth = data_item.non_tensor_batch["answer"]
            ground_truth = data_item.non_tensor_batch["solution"]
            # print(ground_truth,response_str)

            score = self.compute_score(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
