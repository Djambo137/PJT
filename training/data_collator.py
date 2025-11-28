"""Data collator that pads text and stacks multi-image pixel tensors."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


class MultimodalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = [f.pop("pixel_values") for f in features]
        # Pad text fields
        batch = self.tokenizer.pad(
            features, padding=True, return_tensors="pt", pad_to_multiple_of=None
        )

        max_nimg = max(pv.shape[0] for pv in pixel_values)
        # Pixel values arrive as [num_images, 3, H, W]; we pad to [B, max_nimg, 3, H, W].
        padded_pixels = []
        image_masks = []
        for pv in pixel_values:
            n = pv.shape[0]
            if n < max_nimg:
                pad = torch.zeros((max_nimg - n, *pv.shape[1:]), dtype=pv.dtype)
                pv = torch.cat([pv, pad], dim=0)
            padded_pixels.append(pv)
            mask = torch.cat(
                [
                    torch.ones(n, dtype=torch.long),
                    torch.zeros(max_nimg - n, dtype=torch.long),
                ]
            )
            image_masks.append(mask)

        batch["pixel_values"] = torch.stack(padded_pixels)
        batch["image_mask"] = torch.stack(image_masks)
        return batch

