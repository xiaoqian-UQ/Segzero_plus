import argparse
import glob
import json
import os
import math

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset, DatasetDict, concatenate_datasets
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class MaskHeadFixed(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3584,
        num_heads: int = 8,
        decoder_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_dim = decoder_dim

        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_size, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Linear(hidden_size, decoder_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(decoder_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 30->60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 60->120
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 120->240
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 240->480
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_hidden: torch.Tensor,
        target_size: tuple = (840, 840),
        image_grid_thw: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = text_hidden.size(0)

        if vision_tokens.dim() == 2:
            Nv_total = vision_tokens.size(0)
            Nv_per_image = Nv_total // B
            vision_tokens = vision_tokens.view(B, Nv_per_image, -1)

        Nv = vision_tokens.size(1)
        h_grid = w_grid = int(math.sqrt(Nv))
        actual_tokens = h_grid * w_grid

        if actual_tokens != Nv:
            if image_grid_thw is not None:
                t, h_orig, w_orig = image_grid_thw[0].tolist()
                h_grid, w_grid = int(h_orig) // 2, int(w_orig) // 2
                actual_tokens = h_grid * w_grid
            if actual_tokens > Nv:
                h_grid = w_grid = int(math.sqrt(Nv))
                actual_tokens = h_grid * w_grid

        v_proj = self.vision_proj(vision_tokens[:, :actual_tokens, :])
        t_proj = self.text_proj(text_hidden)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()

        v_attn, _ = self.cross_attn(
            query=v_proj,
            key=t_proj,
            value=t_proj,
            key_padding_mask=key_padding_mask,
        )
        v_fused = self.norm(v_attn + v_proj)
        v_map = v_fused.permute(0, 2, 1).view(B, self.decoder_dim, h_grid, w_grid)
        x = self.decoder(v_map)
        mask_logits = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return mask_logits


def load_hf_split(dataset_dir, split):
    try:
        ds = load_from_disk(dataset_dir)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            return ds
        return ds
    except Exception:
        pattern = os.path.join(dataset_dir, f"*-{split}-*.arrow")
        arrow_files = sorted(glob.glob(pattern))
        if not arrow_files:
            raise FileNotFoundError(f"No dataset found at {dataset_dir}")
        shards = [HFDataset.from_file(p) for p in arrow_files]
        return concatenate_datasets(shards)


def get_mask_from_json(json_path, height, width):
    try:
        with open(json_path, "r") as f:
            anno = json.load(f)
    except Exception:
        with open(json_path, "r", encoding="cp1252") as f:
            anno = json.load(f)

    inform = anno.get("shapes", [])
    area_list = []
    valid_poly_list = []
    for item in inform:
        label_id = item.get("label", "")
        points = item.get("points", [])
        if "flag" in label_id.lower():
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(item)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32).tolist()
    sort_inform = [valid_poly_list[s_idx] for s_idx in sort_index]

    mask = np.zeros((height, width), dtype=np.uint8)
    for item in sort_inform:
        label_id = item.get("label", "")
        points = item.get("points", [])
        label_value = 255 if "ignore" in label_id.lower() else 1
        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask.astype(bool)


class ReasonSegDataset(Dataset):
    def __init__(self, hf_ds, target_size=840):
        self.ds = hf_ds
        self.target_size = target_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            image = PILImage.fromarray(np.array(image)).convert("RGB")
        image = image.resize((self.target_size, self.target_size), resample=PILImage.BILINEAR)

        mask = np.array(item["mask"], dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (self.target_size, self.target_size):
            mask = cv2.resize(
                mask,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask > 0).astype(np.float32)

        return {
            "image": image,
            "text": item["text"],
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "image_id": item.get("image_id", str(idx)),
            "ann_id": item.get("ann_id", str(idx)),
        }


class ReasonSegLocalDataset(Dataset):
    def __init__(self, data_dir, target_size=840, indices=None):
        self.data_dir = data_dir
        self.target_size = target_size
        json_paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if indices is not None:
            json_paths = [json_paths[i] for i in indices]
        self.json_paths = json_paths

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        stem = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(self.data_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, f"{stem}.png")

        image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.target_size, self.target_size), resample=PILImage.BILINEAR)

        mask = get_mask_from_json(json_path, orig_h, orig_w)
        mask = mask.astype(np.uint8)
        if mask.shape != (self.target_size, self.target_size):
            mask = cv2.resize(
                mask,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask > 0).astype(np.float32)

        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except Exception:
            with open(json_path, "r", encoding="cp1252") as f:
                meta = json.load(f)
        text = meta.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""

        return {
            "image": image,
            "text": text,
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "image_id": stem,
            "ann_id": stem,
        }


def collate_fn(batch):
    return {
        "image": [b["image"] for b in batch],
        "text": [b["text"] for b in batch],
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "image_id": [b["image_id"] for b in batch],
        "ann_id": [b["ann_id"] for b in batch],
    }


def build_inputs(processor, images, texts):
    prompt_template = (
        "You are a segmentation expert. Please locate '{Q}' in the image. "
        "Output the bounding box and two reference points inside the object."
    )
    batch_messages = []
    for img, txt in zip(images, texts):
        batch_messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_template.format(Q=txt)},
                ],
            }
        ])

    text_inputs = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text_inputs,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def build_text_mask(inputs, processor):
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return None
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    text_mask = attention_mask.bool()
    tokenizer = processor.tokenizer

    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        text_mask &= input_ids != pad_id

    special_tokens = []
    if hasattr(processor, "image_token") and processor.image_token:
        special_tokens.append(processor.image_token)
    if hasattr(processor, "video_token") and processor.video_token:
        special_tokens.append(processor.video_token)
    special_tokens += [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
    ]

    all_special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None

    for tok in special_tokens:
        if not tok:
            continue
        if vocab is not None and tok not in vocab and tok not in all_special:
            continue
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None:
            continue
        if tokenizer.unk_token_id is not None and tok_id == tokenizer.unk_token_id and tok not in all_special:
            continue
        text_mask &= input_ids != tok_id

    return text_mask


def extract_vision_tokens_from_hidden(hidden_states, input_ids, image_token_id):
    if input_ids is None or image_token_id is None:
        return None
    if hidden_states is None or hidden_states.dim() != 3:
        return None

    mask = input_ids == image_token_id
    if mask.sum().item() == 0:
        return None

    counts = mask.sum(dim=1)
    if (counts != counts[0]).any():
        return hidden_states[mask].float()

    num_tokens = counts[0].item()
    return hidden_states[mask].view(hidden_states.size(0), num_tokens, hidden_states.size(2)).float()


def get_vision_tokens(outputs, inputs, processor):
    for attr in [
        "vision_hidden_states",
        "image_hidden_states",
        "image_embeds",
        "vision_embeds",
        "image_features",
    ]:
        val = getattr(outputs, attr, None)
        if val is None:
            continue
        if isinstance(val, (tuple, list)):
            if len(val) == 0:
                continue
            val = val[-1]
        if isinstance(val, torch.Tensor):
            return val.float()

    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1] if hidden_states else None

    tokenizer = processor.tokenizer
    image_token_id = None
    if hasattr(processor, "image_token") and processor.image_token:
        image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    return extract_vision_tokens_from_hidden(hidden_states, inputs.get("input_ids"), image_token_id)


def load_mask_head(mask_head, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        if "mask_head" in ckpt:
            state = ckpt["mask_head"]
        elif all(k.startswith("mask_head.") for k in ckpt.keys()):
            state = {k.replace("mask_head.", ""): v for k, v in ckpt.items()}
        else:
            state = ckpt
    else:
        state = ckpt
    mask_head.load_state_dict(state, strict=True)


def visualize_gt_pred_pair(image, gt_mask, pred_mask, save_path, target_size=840):
    img = image.resize((target_size, target_size))
    img_arr = np.array(img).astype(np.float32)

    gt = gt_mask.squeeze().cpu().numpy() > 0.5
    pred = pred_mask.squeeze().cpu().numpy() > 0.5

    def make_overlay(mask, color):
        color_mask = np.zeros_like(img_arr)
        color_mask[mask] = color
        return (img_arr * 0.6 + color_mask * 0.4).astype(np.uint8)

    gt_overlay = make_overlay(gt, (0, 200, 0))
    pred_overlay = make_overlay(pred, (200, 0, 0))
    pair = np.concatenate([gt_overlay, pred_overlay], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, required=True)
    parser.add_argument("--mask_head_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--target_size", type=int, default=840)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--attn_implementation", type=str, default="")
    parser.add_argument("--vis_dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    if os.path.isdir(args.test_data_path):
        split_dir = os.path.join(args.test_data_path, args.split)
        if os.path.isdir(split_dir) and glob.glob(os.path.join(split_dir, "*.json")):
            data_dir = split_dir
            total_len = len(glob.glob(os.path.join(data_dir, "*.json")))
            part_size = total_len // args.num_parts
            start_idx = args.idx * part_size
            end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
            indices = list(range(start_idx, end_idx))
            eval_ds = ReasonSegLocalDataset(data_dir, target_size=args.target_size, indices=indices)
        elif glob.glob(os.path.join(args.test_data_path, "*.json")):
            data_dir = args.test_data_path
            total_len = len(glob.glob(os.path.join(data_dir, "*.json")))
            part_size = total_len // args.num_parts
            start_idx = args.idx * part_size
            end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
            indices = list(range(start_idx, end_idx))
            eval_ds = ReasonSegLocalDataset(data_dir, target_size=args.target_size, indices=indices)
        else:
            dataset = load_hf_split(args.test_data_path, args.split)
            total_len = len(dataset)
            part_size = total_len // args.num_parts
            start_idx = args.idx * part_size
            end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
            dataset = dataset.select(range(start_idx, end_idx))
            eval_ds = ReasonSegDataset(dataset, target_size=args.target_size)
    else:
        dataset = load_dataset(args.test_data_path, split=args.split)
        total_len = len(dataset)
        part_size = total_len // args.num_parts
        start_idx = args.idx * part_size
        end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
        dataset = dataset.select(range(start_idx, end_idx))
        eval_ds = ReasonSegDataset(dataset, target_size=args.target_size)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    device = torch.device(args.device)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        **model_kwargs,
    )
    if not args.device_map:
        model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    mask_head = MaskHeadFixed(hidden_size=model.config.hidden_size).to(device).float()
    load_mask_head(mask_head, args.mask_head_ckpt)
    mask_head.eval()

    all_outputs = []
    total_intersection = 0
    total_union = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            texts = batch["text"]
            gt_mask = batch["mask"].to(device)

            inputs = build_inputs(processor, images, texts)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            text_mask = build_text_mask(inputs, processor)

            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            text_hidden = outputs.hidden_states[-1].float().to(device)
            vision_tokens = get_vision_tokens(outputs, inputs, processor)
            if vision_tokens is None:
                raise ValueError("Failed to extract vision tokens from model outputs.")
            vision_tokens = vision_tokens.to(device)

            mask_logits = mask_head(
                vision_tokens,
                text_hidden,
                target_size=(args.target_size, args.target_size),
                image_grid_thw=inputs.get("image_grid_thw"),
                text_mask=text_mask,
            )

            pred_mask = (torch.sigmoid(mask_logits) > 0.5)
            gt_bool = gt_mask > 0.5
            intersection = (pred_mask & gt_bool).flatten(1).sum(1)
            union = (pred_mask | gt_bool).flatten(1).sum(1)

            for i in range(pred_mask.size(0)):
                inter = int(intersection[i].item())
                uni = int(union[i].item())
                total_intersection += inter
                total_union += uni
                all_outputs.append(
                    {
                        "image_id": batch["image_id"][i],
                        "ann_id": batch["ann_id"][i],
                        "intersection": inter,
                        "union": uni,
                    }
                )
                if args.vis_dir:
                    image_id = str(batch["image_id"][i])
                    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in image_id)
                    save_path = os.path.join(args.vis_dir, f"part{args.idx}_{safe_id}.png")
                    visualize_gt_pred_pair(
                        images[i],
                        gt_mask[i].cpu(),
                        pred_mask[i].float().cpu(),
                        save_path,
                        target_size=args.target_size,
                    )

    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    if total_union > 0 and len(all_outputs) > 0:
        gIoU = np.mean([o["intersection"] / o["union"] if o["union"] > 0 else 0 for o in all_outputs])
        cIoU = total_intersection / total_union
        print(f"[Part {args.idx}] gIoU: {gIoU:.4f}, cIoU: {cIoU:.4f}")


if __name__ == "__main__":
    main()
