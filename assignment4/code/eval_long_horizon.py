#!/usr/bin/env python
"""Render fixed prompts across long-horizon LoRA checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
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


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
CASES = [
    ("market_84000", "a busy market, in <sks> style", 84000),
    ("market_84003", "a busy market, in <sks> style", 84003),
    ("vendor_face", "a close portrait of a smiling market vendor, in <sks> style", 92003),
    ("two_shoppers", "two shoppers talking beside a vegetable stall, in <sks> style", 92001),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=["cosine_500", "stable_500"], required=True)
    parser.add_argument("--min_total_step", type=int, default=150)
    parser.add_argument("--max_total_step", type=int, default=500)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


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


def checkpoints(trajectory: str, max_total_step: int) -> list[tuple[int, Path]]:
    rows = [(150, ROOT / "models" / "self_market_step150.safetensors")]
    training = ROOT / "long_horizon_out" / trajectory / "training" / "checkpoints"
    if trajectory == "cosine_500":
        candidates = range(175, max_total_step + 1, 25)
        for total_step in candidates:
            path = training / f"step_{total_step:06d}" / WEIGHTS_NAME
            if path.is_file():
                rows.append((total_step, path))
    else:
        candidates = range(25, max_total_step - 150 + 1, 25)
        for continuation_step in candidates:
            path = training / f"step_{continuation_step:06d}" / WEIGHTS_NAME
            if path.is_file():
                rows.append((150 + continuation_step, path))
    return rows


def make_sheets(records: list[dict], outdir: Path) -> None:
    font = load_font(20)
    label_width = 120
    tile = 256
    caption_height = 34
    rows_by_step = {}
    for row in records:
        rows_by_step.setdefault(row["total_step"], {})[row["case"]] = row
    steps = sorted(rows_by_step)
    for page_index, page_steps in enumerate(
        [steps[index:index + 5] for index in range(0, len(steps), 5)],
        1,
    ):
        width = label_width + tile * len(CASES)
        height = caption_height + tile * len(page_steps)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (case, _prompt, _seed) in enumerate(CASES):
            draw.text((label_width + column * tile + 8, 7), case, fill="black", font=font)
        for row_index, step in enumerate(page_steps):
            y = caption_height + row_index * tile
            draw.text((8, y + 10), f"step {step}", fill="black", font=font)
            for column, (case, _prompt, _seed) in enumerate(CASES):
                path = outdir / rows_by_step[step][case]["path"]
                with Image.open(path) as source:
                    image = ImageOps.fit(source.convert("RGB"), (tile, tile))
                sheet.paste(image, (label_width + column * tile, y))
        sheet.save(outdir / f"trajectory_sheet_{page_index:02d}.jpg", quality=94)


def main() -> None:
    args = parse_args()
    models = [
        row
        for row in checkpoints(args.trajectory, args.max_total_step)
        if row[0] >= args.min_total_step
    ]
    if len(models) < 2:
        raise FileNotFoundError("No long-horizon checkpoints found")
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    pipe_args = argparse.Namespace(
        model_name="runwayml/stable-diffusion-v1-5",
        revision=None,
        variant=None,
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    pipe = load_pipeline(pipe_args, device, dtype)

    records = []
    for total_step, weights in models:
        metadata = read_metadata(weights)
        add_or_restore_custom_token(pipe, weights, metadata, None)
        with tempfile.TemporaryDirectory(prefix="long_horizon_eval_") as temporary:
            lora_path = make_lora_only_file(weights, metadata, Path(temporary))
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        for case, prompt, seed in CASES:
            generator = torch.Generator(device=device.type).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=512,
                width=512,
                generator=generator,
            ).images[0]
            name = f"step_{total_step:03d}_{case}.png"
            path = args.outdir / name
            image.save(path)
            records.append(
                {
                    "adapter_sha256": sha256(weights),
                    "case": case,
                    "path": name,
                    "prompt": prompt,
                    "seed": seed,
                    "sha256": sha256(path),
                    "source_weights": str(weights),
                    "total_step": total_step,
                }
            )
        pipe.unload_lora_weights()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.outdir / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_sheets(records, args.outdir)
    print(f"Rendered {len(records)} images from {len(models)} checkpoints")


if __name__ == "__main__":
    main()
