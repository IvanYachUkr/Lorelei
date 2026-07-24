#!/usr/bin/env python
"""Verify that the long-horizon cosine continuation replays Candidate B exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continued_training", type=Path, required=True)
    parser.add_argument("--reference_training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tensors_equal(left: Path, right: Path) -> tuple[bool, int, float]:
    with safe_open(str(left), framework="pt", device="cpu") as left_file:
        left_keys = list(left_file.keys())
        left_tensors = {key: left_file.get_tensor(key) for key in left_keys}
    with safe_open(str(right), framework="pt", device="cpu") as right_file:
        right_keys = list(right_file.keys())
        right_tensors = {key: right_file.get_tensor(key) for key in right_keys}
    if set(left_keys) != set(right_keys):
        return False, 0, float("inf")
    maximum = 0.0
    for key in left_keys:
        left_tensor = left_tensors[key]
        right_tensor = right_tensors[key]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            return False, len(left_keys), float("inf")
        maximum = max(
            maximum,
            float((left_tensor.float() - right_tensor.float()).abs().max()),
        )
    return maximum == 0.0, len(left_keys), maximum


def normalized_rows(path: Path, limit: int, removed: set[str]) -> list[dict]:
    rows = read_jsonl(path)[:limit]
    for row in rows:
        for key in removed:
            row.pop(key, None)
    return rows


def main() -> None:
    args = parse_args()
    checkpoints = {}
    for step in (175, 200):
        continued = args.continued_training / "checkpoints" / f"step_{step:06d}" / WEIGHTS_NAME
        reference = args.reference_training / "checkpoints" / f"step_{step:06d}" / WEIGHTS_NAME
        equal, count, maximum = tensors_equal(continued, reference)
        checkpoints[str(step)] = {
            "continued_sha256": sha256(continued),
            "max_abs_diff": maximum,
            "reference_sha256": sha256(reference),
            "tensor_count": count,
            "tensors_equal": equal,
        }

    metrics_equal = normalized_rows(
        args.continued_training / "training_metrics.jsonl", 200, {"elapsed_seconds"}
    ) == normalized_rows(
        args.reference_training / "training_metrics.jsonl", 200, {"elapsed_seconds"}
    )
    trace_equal = normalized_rows(
        args.continued_training / "training_trace.jsonl", 800, {"source_path"}
    ) == normalized_rows(
        args.reference_training / "training_trace.jsonl", 800, {"source_path"}
    )
    schedule_equal = normalized_rows(
        args.continued_training / "training_draw_schedule.jsonl", 2000, {"source_path"}
    ) == normalized_rows(
        args.reference_training / "training_draw_schedule.jsonl", 2000, {"source_path"}
    )
    result = {
        "checkpoints": checkpoints,
        "metrics_equal_through_step_200": metrics_equal,
        "schedule_equal": schedule_equal,
        "trace_equal_through_step_200": trace_equal,
    }
    result["verified"] = (
        all(row["tensors_equal"] for row in checkpoints.values())
        and metrics_equal
        and trace_equal
        and schedule_equal
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"]:
        raise RuntimeError("Long-horizon continuation verification failed")


if __name__ == "__main__":
    main()
