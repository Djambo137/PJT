"""Training script for the vision stage of the fiche-idée project.

This module refactors the original Colab notebook into reusable Python
code.  It focuses on the data formatting issues that commonly arise when
working with custom JSON/JSONL datasets bundled with a ZIP archive of
images.  The code mirrors the behaviour of the original notebook while
being easier to test locally.

The two most common sources of runtime errors that we observed in the
notebook were:

* The dataset file was provided as a JSON array instead of the expected
  JSON Lines (JSONL) format.  The :func:`ensure_jsonl_dataset` helper
  normalises both formats into a temporary JSONL file that Hugging Face
  can ingest reliably.
* Image paths referenced in the dataset usually point to the extracted
  ZIP folder.  When the ZIP file is extracted multiple times the folder
  structure changes, which results in dangling paths.  The
  :func:`reconcile_image_paths` helper rewrites each example so that it
  always targets the canonical ``VISION_IMG_DIR`` directory.

The rest of the training pipeline is close to the original notebook but
packaged inside functions and a ``main`` entry point so it can be reused
in automated workflows.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_NAME = "vision_dataset.jsonl"
DEFAULT_IMAGE_ZIP_NAME = "antargaz_images_colab.zip"


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------
@dataclass
class VisionConfig:
    """Configuration container used by :func:`train_vision_model`."""

    dataset_path: Path
    image_zip: Path | None
    image_dir: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    num_epochs: int = 3
    learning_rate: float = 1e-4
    per_device_batch_size: int = 1
    grad_accumulation: int = 8
    warmup_steps: int = 20
    logging_steps: int = 5
    max_image_side: int = 560
    max_target_length: int = 1_200


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
def resolve_path_with_fallback(path: Path, *, description: str) -> Path:
    """Resolve ``path`` by searching common Colab/project locations."""

    path = Path(path).expanduser()
    if path.is_absolute() and path.exists():
        return path

    search_roots = [
        Path.cwd(),
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR / "data",
        SCRIPT_DIR.parent / "data",
    ]

    candidates: List[Path] = [path]
    for root in search_roots:
        candidates.append(root / path)
        if path.parent == Path("."):
            candidates.append(root / "data" / path.name)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"{description} not found: {path}")


def ensure_jsonl_dataset(dataset_path: Path) -> Path:
    """Ensure that ``dataset_path`` follows the JSONL format.

    The original notebook expected JSONL but many datasets were supplied
    as a single JSON array.  When such a file is detected we expand it
    into a temporary JSONL file located next to the original dataset and
    return its path.
    """

    dataset_path = dataset_path.expanduser().resolve()
    if dataset_path.suffix == ".jsonl":
        # The file already claims to be JSONL.  Some users still saved a
        # JSON array with the ``.jsonl`` suffix, therefore we validate the
        # first character to detect the situation.
        with dataset_path.open("r", encoding="utf8") as handle:
            first_char = handle.read(1)
            handle.seek(0)
            if first_char != "[":
                return dataset_path
            payload = json.load(handle)
    else:
        with dataset_path.open("r", encoding="utf8") as handle:
            payload = json.load(handle)

    if not isinstance(payload, Sequence):
        raise ValueError(
            "Dataset must be either a JSONL file or a JSON array of "
            "objects containing at least 'image', 'instruction' and 'output'."
        )

    jsonl_path = dataset_path.with_suffix(".jsonl")
    LOGGER.info("Writing normalised JSONL dataset to %s", jsonl_path)
    with jsonl_path.open("w", encoding="utf8") as handle:
        for example in payload:
            json.dump(example, handle, ensure_ascii=False)
            handle.write("\n")
    return jsonl_path


def extract_images(zip_path: Path | None, target_dir: Path) -> None:
    """Extract images from ``zip_path`` into ``target_dir`` if necessary."""

    target_dir.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        return

    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Image archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)

    # Normalise the folder so that every PNG ends up directly inside
    # ``target_dir`` regardless of nested structures in the ZIP.
    for png in target_dir.rglob("*.png"):
        destination = target_dir / png.name
        if png == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), destination)


def reconcile_image_paths(dataset: Dataset, image_dir: Path) -> Dataset:
    """Ensure that every dataset example points to an existing PNG file."""

    image_dir = image_dir.expanduser().resolve()

    def _normalise(example: Dict[str, Any]) -> Dict[str, Any]:
        path = example.get("image")
        if isinstance(path, str):
            filename = Path(path).name
            candidate = image_dir / filename
            if candidate.exists():
                example["image"] = str(candidate)
        return example

    dataset = dataset.map(_normalise, batched=False)

    def _has_valid_image(example: Dict[str, Any]) -> bool:
        path = example.get("image")
        ok = isinstance(path, str) and Path(path).exists()
        if not ok:
            LOGGER.warning("Filtered example with missing image: %s", path)
        return ok

    return dataset.filter(_has_valid_image)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def build_preprocess_fn(processor: AutoProcessor, *, max_image_side: int, max_target_length: int):
    """Create the preprocessing function used before feeding the trainer."""

    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        image = Image.open(example["image"]).convert("RGB")
        width, height = image.size
        scale = min(max_image_side / max(width, height), 1.0)
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)), Image.BICUBIC)

        target = example.get("output", "")
        if isinstance(target, str) and len(target) > max_target_length:
            target = target[:max_target_length]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example.get("instruction", "")},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]

        template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(text=template, images=[image], return_tensors="pt", padding=False, truncation=False)
        sample = {key: tensor.squeeze(0) for key, tensor in processed.items()}
        sample["labels"] = sample["input_ids"].clone()
        return sample

    return _preprocess


class QwenVLDataCollator:
    """Custom data collator that keeps the auxiliary image features."""

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], torch.Tensor):
                return value[0]
            try:
                return torch.as_tensor(value)
            except Exception:  # pragma: no cover - defensive branch
                return torch.stack([torch.as_tensor(v) for v in value])
        return torch.as_tensor(value)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = []
        for feature in features:
            tensor = self._to_tensor(feature.pop("pixel_values"))
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            pixel_values.append(tensor)
        batch_pixel_values = torch.stack(pixel_values)

        aux_keys = {key for feature in features for key in feature if key.startswith("image_")}
        aux_tensors = {}
        for key in aux_keys:
            values = [feature.pop(key) for feature in features]
            tensor = self._to_tensor(values)
            if key == "image_grid_thw":
                tensor = tensor.to(dtype=torch.long)
                if tensor.ndim == 3 and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
            aux_tensors[key] = tensor

        padded = self.processor.tokenizer.pad(
            {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
            },
            padding=True,
            return_tensors="pt",
        )
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100

        batch = dict(padded)
        batch["labels"] = labels
        batch["pixel_values"] = batch_pixel_values
        batch.update(aux_tensors)
        return batch


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def load_quantised_model(model_name: str) -> tuple[Any, BitsAndBytesConfig | None]:
    """Load the Qwen model with a best-effort 8-bit configuration."""

    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        LOGGER.warning("bitsandbytes unavailable, falling back to full precision: %s", exc)
        return load_model(model_name, quant_config=None), None

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    try:
        model = load_model(model_name, quant_config=quant_config)
        return model, quant_config
    except Exception as exc:
        LOGGER.warning("8-bit loading failed, retrying in FP16: %s", exc)
        model = load_model(model_name, quant_config=None)
        if torch.cuda.is_available():
            model = model.to(torch.float16)
        return model, None


def load_model(model_name: str, quant_config: BitsAndBytesConfig | None):
    """Load Qwen either as ImageTextToText or Vision2Seq."""

    try:
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    except Exception as exc:
        LOGGER.info("Falling back to AutoModelForVision2Seq: %s", exc)
        return AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_vision_model(config: VisionConfig) -> None:
    """Run the full fine-tuning loop for the vision model."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dataset_path = ensure_jsonl_dataset(config.dataset_path)
    LOGGER.info("Resolved dataset path: %s", dataset_path)
    if config.image_zip is not None:
        LOGGER.info("Using image archive: %s", config.image_zip)
    LOGGER.info("Images will be read from: %s", config.image_dir)

    extract_images(config.image_zip, config.image_dir)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    LOGGER.info("Loaded %d raw examples", len(dataset))
    dataset = reconcile_image_paths(dataset, config.image_dir)
    if len(dataset) == 0:
        raise RuntimeError("No valid examples were found after preprocessing.")
    LOGGER.info("Dataset contains %d examples after filtering", len(dataset))

    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    preprocess = build_preprocess_fn(
        processor,
        max_image_side=config.max_image_side,
        max_target_length=config.max_target_length,
    )
    dataset = dataset.map(
        preprocess,
        remove_columns=[name for name in dataset.column_names if name != "image"],
        batched=False,
    )

    model, _ = load_quantised_model(config.model_name)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_collator = QwenVLDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)


def parse_args(argv: Sequence[str] | None = None) -> VisionConfig:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL on fiche-idée images")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to vision_dataset.json or .jsonl (auto-discovers data/vision_dataset.jsonl)",
    )
    parser.add_argument(
        "--image-zip",
        type=Path,
        default=None,
        help="Optional ZIP archive containing PNG images (defaults to scripts/data/antargaz_images_colab.zip if present)",
    )
    parser.add_argument("--image-dir", type=Path, default=Path("dataset_fiches/vision"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fiche_idee_lora/vision"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=560)
    parser.add_argument("--max-target-length", type=int, default=1_200)
    args = parser.parse_args(argv)

    dataset_candidates: List[Path] = []
    if args.dataset is not None:
        dataset_candidates.append(args.dataset)
    else:
        dataset_candidates.extend(
            [
                Path(DEFAULT_DATASET_NAME),
                Path("data") / DEFAULT_DATASET_NAME,
                SCRIPT_DIR / "data" / DEFAULT_DATASET_NAME,
            ]
        )

    dataset_path: Path | None = None
    for candidate in dataset_candidates:
        try:
            dataset_path = resolve_path_with_fallback(candidate, description="dataset")
            break
        except FileNotFoundError:
            continue
    if dataset_path is None:
        raise FileNotFoundError(
            "Could not locate the dataset. Pass it explicitly or place it under data/vision_dataset.jsonl."
        )

    image_zip: Path | None
    if args.image_zip is not None:
        image_zip = resolve_path_with_fallback(args.image_zip, description="image archive")
    else:
        image_zip = None
        for candidate in (
            Path(DEFAULT_IMAGE_ZIP_NAME),
            Path("data") / DEFAULT_IMAGE_ZIP_NAME,
            SCRIPT_DIR / "data" / DEFAULT_IMAGE_ZIP_NAME,
        ):
            try:
                image_zip = resolve_path_with_fallback(candidate, description="image archive")
                break
            except FileNotFoundError:
                continue

    image_dir = args.image_dir.expanduser()
    if not image_dir.is_absolute():
        image_dir = (Path.cwd() / image_dir).resolve()

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    return VisionConfig(
        dataset_path=dataset_path,
        image_zip=image_zip,
        image_dir=image_dir,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        grad_accumulation=args.grad_accumulation,
        max_image_side=args.max_image_side,
        max_target_length=args.max_target_length,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    train_vision_model(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
