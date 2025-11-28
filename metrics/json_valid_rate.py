"""Compute the rate of JSON-valid generations w.r.t the CAD schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from metrics.schema import OUTPUT_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preds", type=Path, help="File with one model generation per line")
    args = parser.parse_args()

    validator = Draft202012Validator(OUTPUT_SCHEMA)
    total = 0
    valid = 0
    for line in args.preds.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = obj.get("output", obj) if isinstance(obj, dict) else obj
        if isinstance(payload, dict) and not list(validator.iter_errors(payload)):
            valid += 1
    rate = 0.0 if total == 0 else valid / total
    print(f"valid: {valid}/{total} ({rate*100:.1f}%)")


if __name__ == "__main__":
    main()
