"""Optional helper: PDF -> PNG, OCR (fra+eng), auto split pages into left/right crops.

This script is intentionally simple and documented for Colab use. It requires `pymupdf`
and `pytesseract`. Tesseract binaries are expected to be installed separately (see the
notebook for install commands).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def split_left_right(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size
    mid = w // 2
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))
    return left, right


def run_ocr(img: Image.Image, languages: str = "fra+eng") -> str:
    return pytesseract.image_to_string(img, lang=languages)


def save_crops(base_name: str, crops: List[Image.Image], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    suffixes = [f"_{chr(ord('a') + i)}" for i in range(len(crops))]
    for suf, crop in zip(suffixes, crops):
        out_path = out_dir / f"{base_name}{suf}.png"
        crop.save(out_path)
        paths.append(str(out_path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, help="Input PDF fiche")
    parser.add_argument("--out-dir", type=Path, default=Path("data/vision_crops"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    images = pdf_to_images(args.pdf, dpi=args.dpi)
    all_paths: List[str] = []
    for idx, img in enumerate(images):
        left, right = split_left_right(img)
        paths = save_crops(f"{args.pdf.stem}_{idx}", [left, right], args.out_dir)
        all_paths.extend(paths)
    print(f"Saved {len(all_paths)} crops to {args.out_dir}")

    # OCR stitched text for downstream context
    ocr_texts = [run_ocr(img) for img in images]
    print("\n--- OCR (fra+eng) ---\n")
    print("\n".join(ocr_texts))


if __name__ == "__main__":
    main()
