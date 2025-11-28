"""Validate multiview JSONL entries against the CAD schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator

from metrics.schema import CAD_SCHEMA


def validate_file(path: Path) -> int:
    validator = Draft202012Validator(CAD_SCHEMA)
    errors = 0
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Line {idx}: invalid JSON - {exc}")
            errors += 1
            continue
        for err in validator.iter_errors(obj):
            print(f"Line {idx}: {err.message}")
            errors += 1
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, default=Path("data/json_data/multiview_dataset.jsonl"))
    args = parser.parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"JSONL not found: {args.jsonl}")
    errors = validate_file(args.jsonl)
    if errors:
        raise SystemExit(f"Validation failed with {errors} error(s)")
    print("All lines valid!")


if __name__ == "__main__":
    main()
