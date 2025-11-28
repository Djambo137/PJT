"""Micro-F1 on key slots: components, interfaces, materials_guess, manufacturing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

SLOTS = ["components", "interfaces", "materials_guess", "manufacturing"]


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_sets(records: List[Dict]) -> List[Dict[str, Set[str]]]:
    output_sets: List[Dict[str, Set[str]]] = []
    for rec in records:
        output = rec.get("output", rec)
        output_sets.append({slot: set(output.get(slot, [])) for slot in SLOTS})
    return output_sets


def f1_from_sets(ref: Set[str], pred: Set[str]) -> tuple[int, int, int]:
    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    return tp, fp, fn


def compute_micro_f1(refs: List[Dict[str, Set[str]]], preds: List[Dict[str, Set[str]]]) -> float:
    tp = fp = fn = 0
    for ref, pred in zip(refs, preds):
        for slot in SLOTS:
            t, p, n = f1_from_sets(ref[slot], pred[slot])
            tp += t
            fp += p
            fn += n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSONL with output field")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted JSONL (each line JSON)")
    args = parser.parse_args()

    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.pred)
    if len(gold_records) != len(pred_records):
        raise SystemExit("Gold and predictions must have same number of lines")

    gold_sets = collect_sets(gold_records)
    pred_sets = collect_sets(pred_records)
    score = compute_micro_f1(gold_sets, pred_sets)
    print(f"micro-F1: {score:.4f}")


if __name__ == "__main__":
    main()
