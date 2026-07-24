#!/usr/bin/env python
"""Render identical seeds for several LoRA models and build comparison sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model entry in LABEL=PATH form. Repeat for each adapter.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prompt", default="a busy market, in <sks> style")
    parser.add_argument("--num_images", type=int, default=15)
    parser.add_argument("--seed", type=int, default=66000)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--blind_seed", type=int, default=20260722)
    return parser.parse_args()


def parse_models(entries: list[str]) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for entry in entries:
        label, separator, path_text = entry.partition("=")
        label = label.strip()
        path = Path(path_text.strip())
        if not separator or not label or not path_text.strip():
            raise ValueError(f"Invalid --model value {entry!r}; expected LABEL=PATH")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        models.append((label, path))
    if len(models) < 2:
        raise ValueError("At least two --model entries are required")
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_models(args: argparse.Namespace, models: list[tuple[str, Path]]) -> list[dict]:
    manifest: list[dict] = []
    for label, weights in models:
        model_dir = args.outdir / label
        command = [
            sys.executable,
            "code/eval_lora.py",
            "--weights",
            str(weights),
            "--prompt",
            args.prompt,
            "--outdir",
            str(model_dir),
            "--num_images",
            str(args.num_images),
            "--seed",
            str(args.seed),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--guidance_scale",
            str(args.guidance_scale),
        ]
        print("$", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        images = []
        for index in range(args.num_images):
            path = model_dir / f"adapter_{index:02d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append({"index": index, "seed": args.seed + index, "image": str(path)})
        manifest.append(
            {
                "label": label,
                "weights": str(weights),
                "weights_sha256": sha256(weights),
                "images": images,
            }
        )
    return manifest


def make_model_sheet(args: argparse.Namespace, label: str) -> None:
    thumb = 256
    columns = 5
    rows = (args.num_images + columns - 1) // columns
    label_h = 30
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(16)
    for index in range(args.num_images):
        with Image.open(args.outdir / label / f"adapter_{index:02d}.png") as source:
            image = ImageOps.fit(source.convert("RGB"), (thumb, thumb), method=Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        x = column * thumb
        y = row * (thumb + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + thumb + 6), f"seed {args.seed + index}", fill="black", font=font)
    sheet.save(args.outdir / f"{label}_contact_sheet.jpg", quality=94)


def make_seed_sheets(args: argparse.Namespace, models: list[tuple[str, Path]]) -> None:
    thumb = 384
    seed_label_w = 115
    header_h = 46
    row_label_h = 28
    rows_per_sheet = 5
    font = load_font(18)
    header_font = load_font(20)
    for start in range(0, args.num_images, rows_per_sheet):
        stop = min(start + rows_per_sheet, args.num_images)
        row_count = stop - start
        width = seed_label_w + len(models) * thumb
        height = header_h + row_count * (thumb + row_label_h)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (label, _) in enumerate(models):
            draw.text((seed_label_w + column * thumb + 8, 12), label, fill="black", font=header_font)
        for row, index in enumerate(range(start, stop)):
            y = header_h + row * (thumb + row_label_h)
            draw.text((8, y + thumb // 2 - 10), str(args.seed + index), fill="black", font=font)
            for column, (label, _) in enumerate(models):
                path = args.outdir / label / f"adapter_{index:02d}.png"
                with Image.open(path) as source:
                    image = ImageOps.fit(source.convert("RGB"), (thumb, thumb), method=Image.Resampling.LANCZOS)
                sheet.paste(image, (seed_label_w + column * thumb, y))
            draw.text((seed_label_w + 4, y + thumb + 5), f"shared seed {args.seed + index}", fill="black", font=font)
        sheet.save(
            args.outdir / f"compare_seeds_{args.seed + start}_{args.seed + stop - 1}.jpg",
            quality=94,
        )


def make_blind_seed_sheets(
    args: argparse.Namespace,
    models: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    thumb = 384
    seed_label_w = 115
    header_h = 46
    row_label_h = 28
    rows_per_sheet = 5
    font = load_font(18)
    header_font = load_font(20)
    mappings: list[dict[str, object]] = []
    for start in range(0, args.num_images, rows_per_sheet):
        stop = min(start + rows_per_sheet, args.num_images)
        shuffled = list(models)
        random.Random(args.blind_seed + start).shuffle(shuffled)
        blind_labels = [chr(ord("A") + index) for index in range(len(shuffled))]
        width = seed_label_w + len(shuffled) * thumb
        height = header_h + (stop - start) * (thumb + row_label_h)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, blind_label in enumerate(blind_labels):
            draw.text((seed_label_w + column * thumb + 8, 12), blind_label, fill="black", font=header_font)
        for row, index in enumerate(range(start, stop)):
            y = header_h + row * (thumb + row_label_h)
            draw.text((8, y + thumb // 2 - 10), str(args.seed + index), fill="black", font=font)
            for column, (model_label, _) in enumerate(shuffled):
                path = args.outdir / model_label / f"adapter_{index:02d}.png"
                with Image.open(path) as source:
                    image = ImageOps.fit(source.convert("RGB"), (thumb, thumb), method=Image.Resampling.LANCZOS)
                sheet.paste(image, (seed_label_w + column * thumb, y))
            draw.text((seed_label_w + 4, y + thumb + 5), f"shared seed {args.seed + index}", fill="black", font=font)
        sheet_path = args.outdir / f"blind_compare_seeds_{args.seed + start}_{args.seed + stop - 1}.jpg"
        sheet.save(sheet_path, quality=94)
        mappings.append(
            {
                "sheet": sheet_path.name,
                "sheet_sha256": sha256(sheet_path),
                "columns": [
                    {"blind_label": blind_labels[index], "model_label": label, "column": index}
                    for index, (label, _path) in enumerate(shuffled)
                ],
            }
        )
    return mappings


def main() -> None:
    args = parse_args()
    if args.num_images < 3:
        raise ValueError("--num_images must be at least 3")
    models = parse_models(args.model)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = render_models(args, models)
    for label, _ in models:
        make_model_sheet(args, label)
    make_seed_sheets(args, models)
    blind_mappings = make_blind_seed_sheets(args, models)
    payload = {
        "prompt": args.prompt,
        "seed_start": args.seed,
        "num_images": args.num_images,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "blind_seed": args.blind_seed,
        "models": manifest,
    }
    (args.outdir / "comparison_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.outdir / "hidden_blind_mapping.json").write_text(
        json.dumps({"blind_seed": args.blind_seed, "sheets": blind_mappings}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(models) * args.num_images} images and comparison sheets to {args.outdir}")


if __name__ == "__main__":
    main()
