#!/usr/bin/env python
"""Compare a reproduced adapter with the submitted adapter tensor by tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--reference_training_dir", type=Path, required=True)
    parser.add_argument("--data_registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata() or {}
    return tensors, metadata


def main() -> None:
    args = parse_args()
    selected, selected_metadata = load_tensors(args.selected)
    reproduced, reproduced_metadata = load_tensors(args.reproduced)
    keys_equal = set(selected) == set(reproduced)
    unequal = []
    max_abs_diff = 0.0
    if keys_equal:
        for key in sorted(selected):
            left = selected[key]
            right = reproduced[key]
            if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                unequal.append(key)
                if left.shape == right.shape:
                    difference = (left.float() - right.float()).abs().max().item()
                    max_abs_diff = max(max_abs_diff, difference)

    metrics = args.training_dir / "training_metrics.jsonl"
    trace = args.training_dir / "training_trace.jsonl"
    schedule = args.training_dir / "training_draw_schedule.jsonl"
    sessions = args.training_dir / "sessions.jsonl"
    run_config = args.training_dir / "run_config.json"
    registry = json.loads(args.data_registry.read_text(encoding="utf-8"))
    session_rows = [
        json.loads(line)
        for line in sessions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reproduced_metrics = read_jsonl(metrics)
    reference_metrics = read_jsonl(
        args.reference_training_dir / "training_metrics.jsonl"
    )[:150]
    for row in reproduced_metrics + reference_metrics:
        row.pop("elapsed_seconds", None)
    metrics_equal = reproduced_metrics == reference_metrics

    reproduced_trace = read_jsonl(trace)
    reference_trace = read_jsonl(
        args.reference_training_dir / "training_trace.jsonl"
    )[:600]
    for row in reproduced_trace + reference_trace:
        row.pop("source_path", None)
    trace_equal = reproduced_trace == reference_trace

    reproduced_schedule = read_jsonl(schedule)
    reference_schedule = read_jsonl(
        args.reference_training_dir / "training_draw_schedule.jsonl"
    )
    for row in reproduced_schedule + reference_schedule:
        row.pop("source_path", None)
    schedule_equal = reproduced_schedule == reference_schedule

    result = {
        "data_manifest_sha256": registry["manifest_sha256"],
        "keys_equal": keys_equal,
        "max_abs_diff": max_abs_diff,
        "metrics_equal": metrics_equal,
        "metrics_rows": line_count(metrics),
        "reproduced_metadata": reproduced_metadata,
        "reproduced_sha256": sha256(args.reproduced),
        "run_config_sha256": sha256(run_config),
        "schedule_equal": schedule_equal,
        "schedule_rows": line_count(schedule),
        "selected_metadata": selected_metadata,
        "selected_sha256": sha256(args.selected),
        "session_steps": [row["step"] for row in session_rows],
        "tensor_count": len(selected),
        "tensors_equal": keys_equal and not unequal,
        "trace_equal": trace_equal,
        "trace_rows": line_count(trace),
        "unequal_tensors": unequal,
    }
    expected = {
        "metrics_rows": 150,
        "schedule_rows": 2000,
        "tensor_count": 353,
        "trace_rows": 600,
    }
    problems = [
        f"{name}={result[name]}, expected {value}"
        for name, value in expected.items()
        if result[name] != value
    ]
    if not result["tensors_equal"]:
        problems.append(f"unequal tensors: {len(unequal)}")
    for name in ("metrics_equal", "schedule_equal", "trace_equal"):
        if not result[name]:
            problems.append(f"{name}=false")
    if result["session_steps"][-1:] != [150]:
        problems.append(f"final session step: {result['session_steps'][-1:]}")
    result["verified"] = not problems
    result["problems"] = problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if problems:
        raise RuntimeError("Reproduction verification failed: " + "; ".join(problems))


if __name__ == "__main__":
    main()
