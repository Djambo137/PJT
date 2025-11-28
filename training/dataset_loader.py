"""Dataset utilities for multi-image Qwen2-VL fine-tuning.

Implements the two-pass encoding (prompt vs. full) to avoid image-token
mismatches and constructs label masks that ignore the prompt portion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MAX_LONG_SIDE = 768
MIN_LONG_SIDE = 640
TRUNCATION = False


@dataclass
class ProcessorBundle:
    processor: AutoProcessor


def load_processor(model_name: str = MODEL_NAME) -> ProcessorBundle:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return ProcessorBundle(processor=processor)


def resize_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    target = max(MIN_LONG_SIDE, min(MAX_LONG_SIDE, long_side))
    scale = target / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h))


def load_images(image_paths: Sequence[str]) -> List[Image.Image]:
    return [resize_image(Image.open(p)) for p in image_paths]


def build_example(record: dict, processor: AutoProcessor) -> dict:
    images = load_images(record["images"])
    gold_json = json.dumps(record["output"], ensure_ascii=False)
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": record.get("context_text", "")},
                {"type": "text", "text": record.get("instruction", "")},
            ],
        },
        {"role": "assistant", "content": gold_json},
    ]

    prompt_str = processor.apply_chat_template(
        messages[:-1], add_generation_prompt=False, tokenize=False
    )
    full_text = prompt_str + gold_json

    enc_full = processor(
        text=full_text,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=TRUNCATION,
    )
    enc_prompt = processor(
        text=prompt_str,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=TRUNCATION,
    )

    input_ids = enc_full["input_ids"][0]
    attention_mask = enc_full["attention_mask"][0]
    pixel_values = enc_full["pixel_values"]

    labels = input_ids.clone()
    labels[: len(enc_prompt["input_ids"][0])] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


class MultiviewJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: AutoProcessor) -> None:
        self.records = [
            json.loads(line)
            for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return build_example(self.records[idx], self.processor)


def load_dataset(jsonl_path: Path, model_name: str = MODEL_NAME) -> MultiviewJsonlDataset:
    bundle = load_processor(model_name)
    return MultiviewJsonlDataset(jsonl_path=jsonl_path, processor=bundle.processor)

