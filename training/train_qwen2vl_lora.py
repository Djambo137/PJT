"""Fine-tune Qwen2-VL-2B-Instruct with LoRA on multi-image fiches."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from training.data_collator import MultimodalDataCollator
from training.dataset_loader import (
    MODEL_NAME,
    MultiviewJsonlDataset,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.03


def load_model(model_name: str) -> tuple[torch.nn.Module, AutoProcessor]:
    quantization_config = None
    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        LOGGER.warning("bitsandbytes unavailable; falling back to FP16")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    except Exception:
        LOGGER.warning("AutoModelForImageTextToText failed; falling back to AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    return model, processor


def add_lora(model: torch.nn.Module) -> torch.nn.Module:
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if any(hasattr(module, "weight") and module.weight.dtype == torch.int8 for module in model.modules()):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/json_data/multiview_dataset.jsonl"))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen_lora_adapter"))
    args = parser.parse_args()

    model, processor = load_model(args.model)
    tokenizer = processor.tokenizer

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = MultiviewJsonlDataset(args.jsonl, processor)
    data_collator = MultimodalDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    model = add_lora(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
