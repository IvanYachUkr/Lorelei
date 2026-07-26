from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = SCRIPT_DIR.parents[1]
RECIPE_PATH = ASSIGNMENT_DIR / "training_data" / "auxiliary.jsonl"
IMAGES_DIR = ASSIGNMENT_DIR / "training_data" / "images"

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SIZE = 512


def main():
    recipes = [json.loads(line) for line in RECIPE_PATH.open(encoding="utf-8") if line.strip()]

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

        image.save(IMAGES_DIR / Path(recipe["image"]).name)


if __name__ == "__main__":
    main()
