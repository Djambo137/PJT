"""Build a JSONL skeleton grouping multi-view images for each fiche.

Usage:
    python scripts/make_multiview_jsonl.py --images-dir data/vision_crops --output data/json_data/multiview_dataset.jsonl

The script scans image files, groups *_a/*_b style stems, and writes one JSON object per fiche
with empty context_text and output fields ready for annotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_INSTRUCTION = (
    "Tu es ingénieur produit. Analyse ensemble les IMAGES (vues/diagrammes) ET le TEXTE fourni. "
    "Rends un JSON STRICT et SEUL selon le schéma :\n"
    "object, function, global_shape, components[], interfaces[], working_principle,\n"
    "dimensions{H,W,D,unit}, materials_guess[], manufacturing[], assembly_hypothesis,\n"
    "assumptions[], uncertainties[], risks_limits[]. Ne renvoie que le JSON valide."
)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def group_images(images_dir: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        stem = path.stem
        # handle *_a, *_b, etc.
        if len(stem) > 2 and stem[-2:] in {f"_{chr(c)}" for c in range(ord('a'), ord('z') + 1)}:
            base = stem[:-2]
        else:
            base = stem
        groups.setdefault(base, []).append(str(path))
    return groups


def build_records(groups: Dict[str, List[str]]) -> List[dict]:
    records = []
    for base, paths in groups.items():
        records.append(
            {
                "images": paths,
                "context_text": "",
                "instruction": DEFAULT_INSTRUCTION,
                "output": {
                    "object": "",
                    "function": "",
                    "global_shape": "",
                    "components": [],
                    "interfaces": [],
                    "working_principle": "",
                    "dimensions": {"H": 0, "W": 0, "D": 0, "unit": "mm"},
                    "materials_guess": [],
                    "manufacturing": [],
                    "assembly_hypothesis": "",
                    "assumptions": [],
                    "uncertainties": [],
                    "risks_limits": [],
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("data/vision_crops"))
    parser.add_argument("--output", type=Path, default=Path("data/json_data/multiview_dataset.jsonl"))
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise SystemExit(f"Image directory not found: {args.images_dir}")

    groups = group_images(args.images_dir)
    if not groups:
        print("No images found. Place images in data/vision_crops/")

    records = build_records(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
