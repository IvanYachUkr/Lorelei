#!/usr/bin/env python
"""Create the required two-page report.pdf after training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact assignment report PDF.")
    parser.add_argument("--out", type=Path, default=Path("report.pdf"))
    parser.add_argument("--samples_dir", type=Path, default=Path("samples"))
    parser.add_argument("--weights", type=Path, default=Path("lora_out/pytorch_lora_weights.safetensors"))
    parser.add_argument("--team", default="TODO: add team member names")
    parser.add_argument("--prompt", default="a busy market, in <sks> style")
    parser.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", default="style_imgs/512")
    parser.add_argument("--instance_token", default="<sks>")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning_rate", default="1e-4")
    parser.add_argument("--max_steps", type=int, default=800)
    return parser.parse_args()


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, max_chars: int, leading: float = 13) -> float:
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > max_chars and line:
            c.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = candidate
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_bullets(c: canvas.Canvas, items: list[str], x: float, y: float, max_chars: int) -> float:
    for item in items:
        c.drawString(x, y, "-")
        y = draw_wrapped(c, item, x + 12, y, max_chars=max_chars)
        y -= 3
    return y


def find_samples(samples_dir: Path) -> list[Path]:
    patterns = ["adapter_*.png", "*.png", "*.jpg", "*.jpeg"]
    samples: list[Path] = []
    for pattern in patterns:
        samples.extend(sorted(samples_dir.glob(pattern)))
        if len(samples) >= 3:
            break
    return samples[:3]


def draw_image_fit(c: canvas.Canvas, path: Path, x: float, y: float, max_w: float, max_h: float) -> None:
    with Image.open(path) as image:
        width, height = image.size
    scale = min(max_w / width, max_h / height)
    draw_w = width * scale
    draw_h = height * scale
    c.drawImage(str(path), x + (max_w - draw_w) / 2, y + (max_h - draw_h) / 2, draw_w, draw_h)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4
    margin = 2 * cm
    c = canvas.Canvas(str(args.out), pagesize=A4)

    c.setTitle("Deep Learning Project 4 Report")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - margin, "Deep Learning Project 4: LoRA Style-Tuning")

    y = page_h - margin - 26
    c.setFont("Helvetica", 10)
    y = draw_wrapped(c, f"Team members: {args.team}", margin, y, 95)
    y = draw_wrapped(c, f"Base model: {args.model_name}", margin, y - 6, 95)
    y = draw_wrapped(c, f"Training data: {args.data_dir}", margin, y - 6, 95)
    y = draw_wrapped(c, f"Evaluation prompt: {args.prompt}", margin, y - 6, 95)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y - 16, "Method")
    c.setFont("Helvetica", 10)
    y = draw_bullets(
        c,
        [
            f"Added the style token {args.instance_token} to the Stable Diffusion tokenizer and trained its embedding row.",
            "Applied LoRA adapters to both UNet attention projections and CLIP text encoder attention projections.",
            "Optimized the diffusion noise-prediction objective on the provided 512 x 512 style image crops.",
            "Stored the UNet LoRA, text encoder LoRA, and learned token embedding in one safetensors file.",
        ],
        margin,
        y - 34,
        max_chars=88,
    )

    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y - 12, "Hyperparameters")
    c.setFont("Helvetica", 10)
    hyperparams = [
        f"Token: {args.instance_token}",
        "Resolution: 512",
        f"LoRA rank: {args.rank}",
        f"Learning rate: {args.learning_rate}",
        f"Max steps: {args.max_steps}",
        f"Weights: {args.weights}",
    ]
    y = draw_bullets(c, hyperparams, margin, y - 30, max_chars=88)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y - 12, "Notes")
    c.setFont("Helvetica", 10)
    draw_bullets(
        c,
        [
            "The evaluation script renders at least three images and does not expose a style-strength slider.",
            "Replace this report's team names and add final observations after inspecting generated samples.",
        ],
        margin,
        y - 30,
        max_chars=88,
    )

    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - margin, "Generated Samples")
    c.setFont("Helvetica", 10)
    c.drawString(margin, page_h - margin - 18, f"Prompt: {args.prompt}")

    samples = find_samples(args.samples_dir)
    if samples:
        gap = 0.7 * cm
        box_w = (page_w - 2 * margin - 2 * gap) / 3
        box_h = box_w
        top = page_h - margin - 46
        for index, sample in enumerate(samples):
            x = margin + index * (box_w + gap)
            y_img = top - box_h
            c.rect(x, y_img, box_w, box_h)
            draw_image_fit(c, sample, x, y_img, box_w, box_h)
            c.drawString(x, y_img - 14, sample.name)
    else:
        c.drawString(margin, page_h - margin - 52, "No generated samples found. Run code/eval_lora.py first.")

    c.setFont("Helvetica", 9)
    c.drawString(margin, margin, "Report generated by code/make_report.py.")
    c.save()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
