#!/usr/bin/env python
"""Copy a verified auxiliary manifest into a self-contained training directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORBIDDEN_HASHES = {
    "6d9c85bd4b7950e5f660dbab2ff0d11c7a7524b58efa714a029d24cb79d6970d",
    "2fd8f10d6406514d27b8f4079ae2c87228e8db28d28afa3e42ee92c9a8c7bf25",
    "fc17c486f24c47c93f726a2dd8673096b25ecd5acd5e51229c8bdfb5922f34b6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("training_data"))
    parser.add_argument("--supplied_data_dir", type=Path, default=Path("style_imgs/512"))
    parser.add_argument("--reuse_images_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.outdir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.outdir} exists; pass --overwrite")
        shutil.rmtree(args.outdir)

    images_dir = args.outdir / "images"
    images_dir.mkdir(parents=True)
    supplied_data_dir = args.supplied_data_dir.resolve()
    reusable = {}
    if args.reuse_images_dir is not None:
        reusable = {
            sha256(path): path.resolve()
            for path in sorted(args.reuse_images_dir.iterdir())
            if path.is_file()
        }
    rows = []
    copied = reused = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = (args.manifest.parent / row["image"]).resolve()
            lowered = source.as_posix().lower()
            digest = sha256(source)
            if "market_refs" in lowered or "pdf" in lowered or digest in FORBIDDEN_HASHES:
                raise ValueError(f"Forbidden source image: {source}")
            if digest != row["sha256"]:
                raise ValueError(f"Hash mismatch: {source}")
            kind = row["source_kind"]
            portable_row = dict(row)
            if kind.startswith("base_self_"):
                if row.get("model") != "runwayml/stable-diffusion-v1-5":
                    raise ValueError(f"Unexpected generator model: {source}")
                if row.get("source_adapter_sha256") is not None:
                    raise ValueError(f"Adapter-generated image is not allowed: {source}")
            elif kind == "supplied_face_crop":
                source_image = Path(row["source_image"])
                if not source_image.is_absolute():
                    source_image = (args.manifest.parent / source_image).resolve()
                else:
                    source_image = source_image.resolve()
                if source_image.parent != supplied_data_dir:
                    raise ValueError(f"Crop source is outside supplied data: {source_image}")
                if sha256(source_image) != row["source_sha256"]:
                    raise ValueError(f"Crop source hash mismatch: {source_image}")
                portable_row["source_image"] = Path(
                    os.path.relpath(source_image, args.outdir)
                ).as_posix()
            else:
                raise ValueError(f"Unknown source kind: {kind}")

            if digest in reusable:
                destination = reusable[digest]
                reused += 1
            else:
                name = f"{index:03d}_{kind}{source.suffix.lower()}"
                destination = images_dir / name
                shutil.copyfile(source, destination)
                if sha256(destination) != digest:
                    raise RuntimeError(f"Copied image hash mismatch: {destination}")
                copied += 1
            portable_row["image"] = Path(
                os.path.relpath(destination, args.outdir)
            ).as_posix()
            rows.append(portable_row)

    manifest_path = args.outdir / "auxiliary.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    registry = {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "copied_images": copied,
        "images": len(rows),
        "manifest_sha256": sha256(manifest_path),
        "reused_images": reused,
        "source_kinds": sorted({row["source_kind"] for row in rows}),
    }
    (args.outdir / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry, sort_keys=True))


if __name__ == "__main__":
    main()
