#!/usr/bin/env python
"""Verify and package the required assignment deliverables."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a submission ZIP after training, evaluation, and report generation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("assignment4_submission.zip"))
    parser.add_argument("--min_samples", type=int, default=3)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    required_files = [
        Path("lora_out/pytorch_lora_weights.safetensors"),
        Path("code/train_lora.py"),
        Path("code/eval_lora.py"),
        Path("requirements.txt"),
        Path("README.md"),
        Path("report.pdf"),
    ]
    for relative in required_files:
        require_file(root / relative)

    samples_dir = root / "samples"
    samples = sorted(samples_dir.glob("adapter_*.png"))
    if len(samples) < args.min_samples:
        raise FileNotFoundError(f"Expected at least {args.min_samples} adapter samples in {samples_dir}")

    zip_members = [root / relative for relative in required_files] + samples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for path in zip_members:
            archive.write(path, path.relative_to(root))

    print(f"Created {args.out} with {len(zip_members)} files")


if __name__ == "__main__":
    main()
