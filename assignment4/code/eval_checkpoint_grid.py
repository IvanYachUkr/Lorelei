#!/usr/bin/env python
"""Render fixed-seed target/control grids for visually selecting a checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from eval_lora import (
    add_or_restore_custom_token,
    choose_device,
    choose_dtype,
    load_pipeline,
    make_lora_only_file,
    read_metadata,
)


TARGET_PROMPT = "a busy market, in <sks> style"
CONTROL_PROMPTS = [
    "a friendly market vendor facing the viewer, clear expressive face, in <sks> style",
    "two shoppers talking face to face beside a vegetable stall, in <sks> style",
    "a family choosing food together at a busy market, in <sks> style",
    "a close portrait of a smiling market vendor, in <sks> style",
    "a quiet village street at sunrise, in <sks> style",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate periodic LoRA checkpoints with fixed seeds.")
    parser.add_argument("--checkpoints_dir", type=Path, required=True)
    parser.add_argument("--final_weights", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--steps", default="200,300,400,500,600,700,800,900")
    parser.add_argument("--num_target_images", type=int, default=6)
    parser.add_argument("--target_seed", type=int, default=9100)
    parser.add_argument("--control_seed", type=int, default=12000)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--instance_token", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    requested_steps = [int(value.strip()) for value in args.steps.split(",") if value.strip()]
    result = []
    for step in requested_steps:
        path = args.checkpoints_dir / f"step_{step:06d}" / "pytorch_lora_weights.safetensors"
        if path.exists():
            result.append((f"step_{step:06d}", path))
    best = args.checkpoints_dir / "best_objective" / "pytorch_lora_weights.safetensors"
    if best.exists():
        result.append(("best_objective", best))
    if args.final_weights is not None and args.final_weights.exists():
        result.append(("final", args.final_weights))
    if not result:
        raise FileNotFoundError(f"No requested checkpoints found in {args.checkpoints_dir}")
    return result


def generate(pipe, prompt: str, seed: int, args: argparse.Namespace, device: torch.device) -> Image.Image:
    generator = torch.Generator(device=device.type).manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=512,
        width=512,
        generator=generator,
    ).images[0]


def make_contact_sheet(items: list[tuple[Path, str]], output_path: Path, columns: int = 3) -> None:
    tile_size = 300
    label_height = 72
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, (path, label) in enumerate(items):
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        with Image.open(path) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (tile_size, tile_size), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        wrapped = "\n".join(textwrap.wrap(label, width=36)[:3])
        draw.multiline_text((x + 6, y + tile_size + 5), wrapped, fill="black", font=font, spacing=2)
        draw.rectangle((x, y, x + tile_size - 1, y + tile_size + label_height - 1), outline="#777777")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_target_comparison(
    rows: list[tuple[str, list[Path]]],
    output_path: Path,
) -> None:
    tile = 190
    label_width = 150
    columns = max(len(paths) for _, paths in rows)
    sheet = Image.new("RGB", (label_width + columns * tile, len(rows) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for row_index, (name, paths) in enumerate(rows):
        y = row_index * tile
        draw.multiline_text((8, y + 12), name.replace("_", "\n"), fill="black", font=font, spacing=3)
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                thumb = ImageOps.fit(image.convert("RGB"), (tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (label_width + column * tile, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.num_target_images < 3:
        raise ValueError("--num_target_images must be at least 3")
    checkpoints = checkpoint_paths(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    pipe = load_pipeline(args, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    baseline_path = args.outdir / "baseline_busy_market.png"
    generate(pipe, "a busy market", args.target_seed, args, device).save(baseline_path)
    records = [
        {
            "checkpoint": "base_model",
            "image": baseline_path.name,
            "prompt": "a busy market",
            "seed": args.target_seed,
            "sha256": sha256(baseline_path),
        }
    ]
    target_rows: list[tuple[str, list[Path]]] = []

    for name, weights in checkpoints:
        metadata = read_metadata(weights)
        checkpoint_hash = sha256(weights)
        add_or_restore_custom_token(pipe, weights, metadata, args.instance_token)
        with tempfile.TemporaryDirectory(prefix="lora_grid_") as tmp:
            lora_path = make_lora_only_file(weights, metadata, Path(tmp))
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

        checkpoint_dir = args.outdir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sheet_items: list[tuple[Path, str]] = []
        target_paths: list[Path] = []
        for index in range(args.num_target_images):
            path = checkpoint_dir / f"target_{index:02d}.png"
            generate(pipe, TARGET_PROMPT, args.target_seed + index, args, device).save(path)
            target_paths.append(path)
            sheet_items.append((path, f"target seed {args.target_seed + index}"))
            records.append(
                {
                    "checkpoint": name,
                    "checkpoint_sha256": checkpoint_hash,
                    "image": path.relative_to(args.outdir).as_posix(),
                    "prompt": TARGET_PROMPT,
                    "seed": args.target_seed + index,
                    "sha256": sha256(path),
                }
            )

        for index, prompt in enumerate(CONTROL_PROMPTS):
            path = checkpoint_dir / f"control_{index:02d}.png"
            generate(pipe, prompt, args.control_seed + index, args, device).save(path)
            sheet_items.append((path, prompt))
            records.append(
                {
                    "checkpoint": name,
                    "checkpoint_sha256": checkpoint_hash,
                    "image": path.relative_to(args.outdir).as_posix(),
                    "prompt": prompt,
                    "seed": args.control_seed + index,
                    "sha256": sha256(path),
                }
            )

        no_token_path = checkpoint_dir / "no_token_busy_market.png"
        generate(pipe, "a busy market", args.target_seed, args, device).save(no_token_path)
        sheet_items.append((no_token_path, "adapter active, no custom token: a busy market"))
        records.append(
            {
                "checkpoint": name,
                "checkpoint_sha256": checkpoint_hash,
                "image": no_token_path.relative_to(args.outdir).as_posix(),
                "prompt": "a busy market",
                "seed": args.target_seed,
                "sha256": sha256(no_token_path),
            }
        )
        make_contact_sheet(sheet_items, checkpoint_dir / "contact_sheet.jpg")
        target_rows.append((name, target_paths))
        print(f"Rendered {name}: {weights}")

        pipe.unload_lora_weights()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    make_target_comparison(target_rows, args.outdir / "target_checkpoint_comparison.jpg")
    manifest = {
        "command": sys.argv,
        "model_name": args.model_name,
        "scheduler": type(pipe.scheduler).__name__,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "width": 512,
        "height": 512,
        "device": str(device),
        "dtype": str(dtype),
        "records": records,
    }
    (args.outdir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Checkpoint review written to {args.outdir}")


if __name__ == "__main__":
    main()
