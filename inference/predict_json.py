"""Run inference on multi-image fiches and return CAD JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from jsonschema import Draft202012Validator
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from metrics.schema import OUTPUT_SCHEMA
from training.dataset_loader import MAX_LONG_SIDE, MIN_LONG_SIDE

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_INSTRUCTION = (
    "Tu es ingénieur produit. Analyse ensemble les IMAGES (vues/diagrammes) ET le TEXTE fourni. "
    "Rends un JSON STRICT et SEUL selon le schéma :\n"
    "object, function, global_shape, components[], interfaces[], working_principle,\n"
    "dimensions{H,W,D,unit}, materials_guess[], manufacturing[], assembly_hypothesis,\n"
    "assumptions[], uncertainties[], risks_limits[]. Ne renvoie que le JSON valide."
)


def resize_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    target = max(MIN_LONG_SIDE, min(MAX_LONG_SIDE, long_side))
    scale = target / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.convert("RGB").resize((new_w, new_h))


def load_images(paths: List[str]) -> List[Image.Image]:
    return [resize_image(Image.open(p)) for p in paths]


def load_base_model(model_name: str, quantize: bool = True):
    quant_cfg = None
    if quantize:
        try:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            quant_cfg = None
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        try:
            return cls.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quant_cfg,
                trust_remote_code=True,
            )
        except Exception:
            continue
    raise RuntimeError("Failed to load base model")


def attach_adapter(model, adapter_path: Path):
    if adapter_path is None:
        return model
    return PeftModel.from_pretrained(model, adapter_path)


def build_prompt(processor, images: List[Image.Image], context: str, instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": context},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in generation")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--instruction", type=str, default="default")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample_pred.json"))
    args = parser.parse_args()

    instruction = (
        DEFAULT_INSTRUCTION
        if args.instruction == "default"
        else Path(args.instruction).read_text(encoding="utf-8")
    )

    images = load_images(args.images)
    model = load_base_model(args.model)
    model = attach_adapter(model, args.adapter)
    processor = AutoProcessor.from_pretrained(args.adapter or args.model, trust_remote_code=True)

    prompt = build_prompt(processor, images, args.context, instruction)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    try:
        parsed = extract_json(text)
    except Exception as exc:
        raise SystemExit(f"Failed to parse JSON: {exc}")

    validator = Draft202012Validator(OUTPUT_SCHEMA)
    errors = list(validator.iter_errors(parsed))
    if errors:
        for err in errors:
            print(f"Schema error: {err.message}")
        raise SystemExit("Generated JSON did not pass schema validation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
