#!/usr/bin/env python
"""Verify that campaign manifests contain only explicitly allowed image sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
FORBIDDEN_HASHES = {
    "6d9c85bd4b7950e5f660dbab2ff0d11c7a7524b58efa714a029d24cb79d6970d",
    "2fd8f10d6406514d27b8f4079ae2c87228e8db28d28afa3e42ee92c9a8c7bf25",
    "fc17c486f24c47c93f726a2dd8673096b25ecd5acd5e51229c8bdfb5922f34b6",
}
EXPECTED_SHARES = {
    "self_market": {
        "supplied_full": 0.86,
        "base_self_market": 0.05,
        "base_self_market_character": 0.05,
        "base_self_people": 0.04,
    },
    "balanced_people": {
        "supplied_full": 0.82,
        "base_self_market": 0.04,
        "base_self_market_character": 0.04,
        "base_self_people": 0.02,
        "supplied_face_crop": 0.08,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self_market_manifest",
        type=Path,
        default=Path("training_data/auxiliary.jsonl"),
    )
    parser.add_argument(
        "--balanced_people_manifest",
        type=Path,
        default=Path("experiment_data/balanced_people/auxiliary.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiment_data/provenance_report.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def verify_image(path: Path, expected_hash: str | None = None) -> str:
    lowered = path.as_posix().lower()
    if not path.is_file() or "market_refs" in lowered or "pdf" in lowered:
        raise ValueError(f"Forbidden or missing image: {path}")
    digest = sha256(path)
    if digest in FORBIDDEN_HASHES:
        raise ValueError(f"PDF-derived image hash: {path}")
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"Hash mismatch: {path}")
    return digest


def verify_supplied() -> tuple[int, str]:
    data_dir = ROOT / "style_imgs/512"
    captions_path = ROOT / "code/auto_captions/florence_captions.jsonl"
    images = sorted(path.resolve() for path in data_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"})
    captions = {resolve(captions_path, row["image"]): row["caption"] for row in read_jsonl(captions_path)}
    if len(images) != 843 or set(images) != set(captions):
        raise ValueError("Supplied image/caption coverage is not exactly 843")
    digest = hashlib.sha256()
    for path in images:
        image_hash = verify_image(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(image_hash.encode("ascii"))
    return len(images), digest.hexdigest()


def verify_manifest(path: Path, expected_shares: dict[str, float]) -> dict:
    rows = read_jsonl(path)
    seen = set()
    counts = Counter()
    weighted = Counter({"supplied_full": 843.0})
    supplied_root = (ROOT / "style_imgs/512").resolve()
    generated_kinds = {
        "base_self_market",
        "base_self_people",
        "base_self_market_character",
    }
    for row in rows:
        image = resolve(path, row["image"])
        if image in seen:
            raise ValueError(f"Duplicate image in {path}: {image}")
        seen.add(image)
        verify_image(image, row["sha256"])
        kind = row["source_kind"]
        sample_weight = float(row["sampling_weight"])
        target_prob = float(row["target_prompt_prob"])
        if sample_weight <= 0 or not 0 <= target_prob <= 1 or not row["caption"].strip():
            raise ValueError(f"Invalid training row: {image}")
        counts[kind] += 1
        weighted[kind] += sample_weight

        if kind == "supplied_face_crop":
            source = resolve(path, row["source_image"])
            if source.parent != supplied_root:
                raise ValueError(f"Crop source is outside supplied data: {source}")
            verify_image(source, row["source_sha256"])
            if target_prob:
                raise ValueError(f"A supplied crop cannot receive the target prompt: {image}")
        elif kind in generated_kinds:
            required = (
                "model",
                "generation_prompt",
                "negative_prompt",
                "seed",
                "scheduler",
                "num_inference_steps",
                "guidance_scale",
                "source_adapter_sha256",
            )
            if any(key not in row for key in required):
                raise ValueError(f"Incomplete generation provenance: {image}")
            if row["model"] != BASE_MODEL or row["source_adapter_sha256"] is not None:
                raise ValueError(f"Image was not generated by the unmodified base model: {image}")
        else:
            raise ValueError(f"Unknown source kind: {kind}")

    total_weight = sum(weighted.values())
    shares = {kind: value / total_weight for kind, value in sorted(weighted.items())}
    if set(shares) != set(expected_shares):
        raise ValueError(f"Unexpected sources in {path}: {sorted(shares)}")
    for kind, target in expected_shares.items():
        if abs(shares[kind] - target) > 1e-9:
            raise ValueError(f"Unexpected {kind} share in {path}: {shares[kind]}")
    return {
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "weighted_shares": shares,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    supplied_count, supplied_hash = verify_supplied()
    manifests = {
        "self_market": verify_manifest(
            args.self_market_manifest.resolve(), EXPECTED_SHARES["self_market"]
        ),
        "balanced_people": verify_manifest(
            args.balanced_people_manifest.resolve(), EXPECTED_SHARES["balanced_people"]
        ),
    }
    report = {
        "base_model": BASE_MODEL,
        "forbidden_hash_matches": 0,
        "supplied_count": supplied_count,
        "supplied_set_sha256": supplied_hash,
        "manifests": manifests,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
