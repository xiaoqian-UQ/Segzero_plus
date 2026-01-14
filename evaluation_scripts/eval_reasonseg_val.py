import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reasoning_model_path",
        type=str,
        default="/mnt/xiaoqian/model/pretrained_models/Seg-Zero-7B",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/xiaoqian/dataset/ReasonSeg/val",
        help="ReasonSeg val directory containing *.json and images",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--resize_size", type=int, default=840)
    parser.add_argument(
        "--segmentation_model_path",
        type=str,
        default="facebook/sam2-hiera-large",
        help="SAM2 model id or local path (used if sam_config/ckpt not set)",
    )
    parser.add_argument("--sam_config", type=str, default=None)
    parser.add_argument("--sam_config_dir", type=str, default=None)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_device", type=str, default="auto")
    return parser.parse_args()


def get_mask_from_json(json_path, height, width):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except Exception:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]

    area_list = []
    valid_poly_list = []
    for item in inform:
        label_id = item["label"]
        points = item["points"]
        if "flag" == label_id.lower():
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        import cv2

        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(item)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32).tolist()
    sort_inform = [valid_poly_list[s_idx] for s_idx in sort_index]

    mask = np.zeros((height, width), dtype=np.uint8)
    for item in sort_inform:
        label_id = item["label"]
        points = item["points"]
        label_value = 255 if "ignore" in label_id.lower() else 1

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask.astype(bool)


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r"{[^}]+}"
    json_match = re.search(json_pattern, output_text)
    if json_match:
        data = json.loads(json_match.group(0))
        bbox_key = next((key for key in data.keys() if "bbox" in key.lower()), None)
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [
                round(int(content_bbox[0]) * x_factor),
                round(int(content_bbox[1]) * y_factor),
                round(int(content_bbox[2]) * x_factor),
                round(int(content_bbox[3]) * y_factor),
            ]
        points_keys = [key for key in data.keys() if "points" in key.lower()][:2]
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0]) * x_factor), round(int(point1[1]) * y_factor)]
            point2 = [round(int(point2[0]) * x_factor), round(int(point2[1]) * y_factor)]
            points = [point1, point2]

    think_pattern = r"<think>([^<]+)</think>"
    think_match = re.search(think_pattern, output_text)
    think_text = think_match.group(1) if think_match else ""

    return content_bbox, points, think_text


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union


def load_sam2_predictor(args):
    if args.sam_config and args.sam_checkpoint:
        if not os.path.exists(args.sam_checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam_checkpoint}")
        if args.sam_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.sam_device

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if os.path.isfile(args.sam_config) and args.sam_config.lower().endswith((".yaml", ".yml")):
            abs_yaml = os.path.abspath(args.sam_config)
            inferred_dir = os.path.dirname(abs_yaml)
            config_name = os.path.splitext(os.path.basename(abs_yaml))[0]
            hydra_dir = os.path.abspath(args.sam_config_dir) if args.sam_config_dir else inferred_dir

            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            initialize_config_dir(
                config_dir=hydra_dir,
                version_base=None,
                job_name="sam2_local_cfg",
            )

            sam_model = build_sam2(config_name, ckpt_path=args.sam_checkpoint, device=device, mode="eval")
            return SAM2ImagePredictor(sam_model)

        config_name = os.path.splitext(args.sam_config)[0]
        sam_model = build_sam2(config_name, ckpt_path=args.sam_checkpoint, device=device, mode="eval")
        return SAM2ImagePredictor(sam_model)

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)


def main():
    args = parse_args()

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()

    segmentation_model = load_sam2_predictor(args)

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    json_list = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    total_len = len(json_list)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    json_list = json_list[start_idx:end_idx]

    question_template = (
        "Please find '{Question}' with bbox and points."
        "Compare the difference between objects and find the most closely matched one."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )

    all_outputs = []

    for i in tqdm(range(0, len(json_list), args.batch_size)):
        batch_files = json_list[i:i + args.batch_size]
        batch_messages = []
        batch_meta = []

        for json_path in batch_files:
            with open(json_path, "r") as f:
                anno = json.loads(f.read())

            text = anno["text"][0]
            image_id = os.path.splitext(os.path.basename(json_path))[0]
            image_path = json_path.replace(".json", ".jpg")
            if not os.path.exists(image_path):
                image_path = json_path.replace(".json", ".png")

            image = PILImage.open(image_path).convert("RGB")
            width, height = image.size

            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image.resize((args.resize_size, args.resize_size), PILImage.BILINEAR),
                    },
                    {
                        "type": "text",
                        "text": question_template.format(
                            Question=text.lower().strip("."),
                            Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}",
                        ),
                    },
                ],
            }]
            batch_messages.append(message)
            batch_meta.append({
                "image_id": image_id,
                "ann_id": image_id,
                "image": image,
                "json_path": json_path,
                "img_height": height,
                "img_width": width,
            })

        text_inputs = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = reasoning_model.generate(
            **inputs, use_cache=True, max_new_tokens=1024, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                try:
                    bbox, points, think = extract_bbox_points_think(
                        batch_output_text[id_idx],
                        batch_meta[id_idx]["img_width"] / args.resize_size,
                        batch_meta[id_idx]["img_height"] / args.resize_size,
                    )
                    segmentation_model.set_image(batch_meta[id_idx]["image"])
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points,
                        point_labels=[1, 1],
                        box=bbox,
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    mask = masks[sorted_ind][0].astype(bool)
                    gt_mask = get_mask_from_json(
                        batch_meta[id_idx]["json_path"],
                        batch_meta[id_idx]["img_height"],
                        batch_meta[id_idx]["img_width"],
                    )
                    intersection, union = compute_iou(mask, gt_mask)
                except Exception:
                    think = ""
                    gt_mask = get_mask_from_json(
                        batch_meta[id_idx]["json_path"],
                        batch_meta[id_idx]["img_height"],
                        batch_meta[id_idx]["img_width"],
                    )
                    intersection = 0
                    union = gt_mask.sum()

                all_outputs.append({
                    "image_id": batch_meta[id_idx]["image_id"],
                    "ann_id": batch_meta[id_idx]["ann_id"],
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union),
                })

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
