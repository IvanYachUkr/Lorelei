#!/usr/bin/env python
"""Build and verify the assignment submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfReader
from safetensors import safe_open


TOKEN_KEY = "__custom_token_embedding__"
TARGET_PROMPT = "a busy market, in <sks> style"
FORBIDDEN_HASHES = {
    "6d9c85bd4b7950e5f660dbab2ff0d11c7a7524b58efa714a029d24cb79d6970d",
    "2fd8f10d6406514d27b8f4079ae2c87228e8db28afa3e42ee92c9a8c7bf25",
    "fc17c486f24c47c93f726a2dd8673096b25ecd5acd5e51229c8bdfb5922f34b6",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("assignment4_submission.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    required = [
        "lora_out/pytorch_lora_weights.safetensors",
        "code/train_lora.py",
        "code/eval_lora.py",
        "code/token_utils.py",
        "code/verify_training_data.py",
        "code/verify_lora_weights.py",
        "code/verify_reproduced_model.py",
        "code/make_report.py",
        "code/package_submission.py",
        "code/auto_captions/florence_captions.jsonl",
        "docs/lora_training_algorithm.md",
        "training_data/auxiliary.jsonl",
        "training_data/registry.json",
        "requirements.txt",
        "README.md",
        "report.pdf",
        "samples/inference_manifest.json",
        "samples/comparisons/supplied_only_steps125_200.jpg",
        "samples/comparisons/self_market_steps125_200.jpg",
        "samples/comparisons/balanced_people_steps125_200.jpg",
        "samples/comparisons/refinement_steps45_60.jpg",
        "reproducibility/self_market_step150/README.md",
        "reproducibility/self_market_step150/verification.json",
        "reproducibility/self_market_step150/training/run_config.json",
        "reproducibility/self_market_step150/training/training_draw_schedule.jsonl",
        "reproducibility/self_market_step150/training/training_trace.jsonl",
        "reproducibility/self_market_step150/training/training_metrics.jsonl",
        "reproducibility/self_market_step150/training/sessions.jsonl",
        "experiments/clean_campaign/README.md",
        "experiments/clean_campaign/final_selection.md",
        "experiments/clean_campaign/self_market/visual_reviews.md",
        "experiments/clean_campaign/self_market/training/run_config.json",
        "experiments/clean_campaign/self_market/training/training_draw_schedule.jsonl",
        "experiments/clean_campaign/self_market/training/training_trace.jsonl",
        "experiments/clean_campaign/self_market/training/training_metrics.jsonl",
        "experiments/clean_campaign/self_market/training/sessions.jsonl",
    ]
    required.extend(f"samples/adapter_{index:02d}.png" for index in range(6))
    auxiliary_images = sorted((root / "training_data/images").glob("*.png"))
    if len(auxiliary_images) != 60:
        raise ValueError(f"Expected 60 auxiliary images, found {len(auxiliary_images)}")
    required.extend(path.relative_to(root).as_posix() for path in auxiliary_images)

    paths = [root / relative for relative in required]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if any("market_refs" in path.as_posix().lower() for path in paths):
        raise ValueError("Submission contains a forbidden PDF-reference path")
    for path in paths:
        if path.suffix.lower() in IMAGE_SUFFIXES and sha256(path) in FORBIDDEN_HASHES:
            raise ValueError(f"Submission contains a forbidden PDF image: {path}")

    weights = root / required[0]
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if TOKEN_KEY not in keys:
        raise ValueError("Adapter does not contain the custom token embedding")
    if not any(key.startswith("unet.") for key in keys):
        raise ValueError("Adapter does not contain UNet LoRA tensors")
    if not any(key.startswith("text_encoder.") for key in keys):
        raise ValueError("Adapter does not contain text-encoder LoRA tensors")

    manifest = json.loads((root / "samples/inference_manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt"] != TARGET_PROMPT:
        raise ValueError("Samples use the wrong prompt")
    if manifest["adapter_sha256"] != sha256(weights):
        raise ValueError("Sample manifest references a different adapter")
    adapter_images = [row for row in manifest["images"] if row["kind"] == "adapter"]
    if len(adapter_images) < 3:
        raise ValueError("At least three adapter samples are required")
    for row in adapter_images:
        image = root / "samples" / row["path"]
        if not image.is_file() or sha256(image) != row["sha256"]:
            raise ValueError(f"Sample hash mismatch: {image}")

    auxiliary_rows = [
        json.loads(line)
        for line in (root / "training_data/auxiliary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(auxiliary_rows) != 60:
        raise ValueError(f"Expected 60 auxiliary rows, found {len(auxiliary_rows)}")
    for row in auxiliary_rows:
        image = root / "training_data" / row["image"]
        if sha256(image) != row["sha256"]:
            raise ValueError(f"Auxiliary hash mismatch: {image}")
        if row["model"] != "runwayml/stable-diffusion-v1-5":
            raise ValueError(f"Unexpected auxiliary generator: {image}")
        if row["source_adapter_sha256"] is not None:
            raise ValueError(f"Adapter-generated auxiliary image: {image}")

    if len(PdfReader(root / "report.pdf").pages) > 2:
        raise ValueError("report.pdf exceeds two pages")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())

    with ZipFile(args.out, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate paths")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        lora_files = [name for name in names if name.startswith("lora_out/")]
        if lora_files != ["lora_out/pytorch_lora_weights.safetensors"]:
            raise ValueError(f"Unexpected lora_out contents: {lora_files}")
        if hashlib.sha256(archive.read(lora_files[0])).hexdigest() != sha256(weights):
            raise ValueError("Packaged adapter hash mismatch")

    print(f"Created {args.out} with {len(paths)} files; adapter SHA256 {sha256(weights)}")


if __name__ == "__main__":
    main()
