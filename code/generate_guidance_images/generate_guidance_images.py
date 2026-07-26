from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = SCRIPT_DIR.parents[1]
RECIPE_PATH = ASSIGNMENT_DIR / "training_data" / "auxiliary.jsonl"
IMAGES_DIR = SCRIPT_DIR / "images"
OUT_PATH = SCRIPT_DIR / "auxiliary.jsonl"

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SIZE = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    recipes = [json.loads(line) for line in RECIPE_PATH.open(encoding="utf-8") if line.strip()]
    if args.limit:
        recipes = recipes[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for recipe in tqdm(recipes):
            generator = torch.Generator(device=device).manual_seed(recipe["seed"])
            image = pipe(
                prompt=recipe["generation_prompt"],
                negative_prompt=recipe["negative_prompt"],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=SIZE,
                width=SIZE,
                generator=generator,
            ).images[0]

            name = f"{recipe['id']}.png"
            image.save(IMAGES_DIR / name)

            row = dict(recipe)
            row["image"] = f"images/{name}"
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
