#!/usr/bin/env python
"""Render samples from the trained Stable Diffusion 1.5 LoRA adapter."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing Diffusers dependencies. Install them with: pip install -r requirements.txt"
    ) from exc


CUSTOM_TOKEN_EMBEDDING_KEY = "__custom_token_embedding__"
LORA_FILENAME = "pytorch_lora_weights.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with the trained style LoRA adapter.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to pytorch_lora_weights.safetensors.")
    parser.add_argument("--prompt", default="a busy market, in <sks> style", help="Prompt to render.")
    parser.add_argument("--outdir", type=Path, default=Path("samples"), help="Directory for generated PNGs.")
    parser.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5", help="Base SD 1.5 model id or path.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--variant", default=None, help="Optional model variant, such as fp16.")
    parser.add_argument("--instance_token", default=None, help="Override token if metadata is missing.")
    parser.add_argument("--num_images", type=int, default=3, help="Number of adapter samples to render.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--baseline", action="store_true", help="Also render baseline images before loading the adapter.")
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32" or device.type != "cuda":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float16


def read_metadata(weights: Path) -> dict[str, str]:
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def add_or_restore_custom_token(pipe, weights: Path, metadata: dict[str, str], instance_token_arg: str | None) -> None:
    tensors = load_file(str(weights), device="cpu")
    token_embedding = tensors.get(CUSTOM_TOKEN_EMBEDDING_KEY)
    instance_token = instance_token_arg or metadata.get("instance_token")
    if not instance_token:
        if token_embedding is not None:
            raise ValueError("Weights contain a custom token embedding but no instance token metadata.")
        return

    num_added = pipe.tokenizer.add_tokens([instance_token])
    token_id = pipe.tokenizer.convert_tokens_to_ids(instance_token)
    tokenized = pipe.tokenizer(instance_token, add_special_tokens=False).input_ids
    if len(tokenized) != 1:
        raise ValueError(f"{instance_token!r} must tokenize as one token, got {tokenized}")

    if num_added:
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    if token_embedding is None:
        print("Warning: no custom token embedding found in weights. The custom token will use its default embedding.")
        return

    embedding_weight = pipe.text_encoder.get_input_embeddings().weight
    if token_embedding.numel() != embedding_weight.shape[1]:
        raise ValueError(
            f"Custom token embedding has width {token_embedding.numel()}, expected {embedding_weight.shape[1]}"
        )

    with torch.no_grad():
        embedding_weight[token_id].copy_(token_embedding.to(device=embedding_weight.device, dtype=embedding_weight.dtype))


def make_lora_only_file(weights: Path, metadata: dict[str, str], tmpdir: Path) -> Path:
    tensors = load_file(str(weights), device="cpu")
    tensors = {key: value for key, value in tensors.items() if key != CUSTOM_TOKEN_EMBEDDING_KEY}
    filtered_path = tmpdir / LORA_FILENAME
    save_file(tensors, str(filtered_path), metadata=metadata)
    return filtered_path


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> StableDiffusionPipeline:
    load_kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
        "revision": args.revision,
    }
    if args.variant is not None:
        load_kwargs["variant"] = args.variant

    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, **load_kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def render_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    outdir: Path,
    prefix: str,
    num_images: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: torch.device,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for index in range(num_images):
        generator = torch.Generator(device=device.type).manual_seed(seed + index)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        image.save(outdir / f"{prefix}_{index:02d}.png")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.num_images < 3:
        raise ValueError("--num_images must be at least 3 to satisfy the assignment")

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    metadata = read_metadata(args.weights)

    pipe = load_pipeline(args, device=device, dtype=dtype)
    add_or_restore_custom_token(pipe, args.weights, metadata, args.instance_token)

    if args.baseline:
        render_images(
            pipe=pipe,
            prompt=args.prompt,
            outdir=args.outdir,
            prefix="baseline",
            num_images=args.num_images,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
        )

    with tempfile.TemporaryDirectory(prefix="lora_eval_") as tmp:
        lora_path = make_lora_only_file(args.weights, metadata, Path(tmp))
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    render_images(
        pipe=pipe,
        prompt=args.prompt,
        outdir=args.outdir,
        prefix="adapter",
        num_images=args.num_images,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=device,
    )

    print(f"Saved {args.num_images} adapter samples to {args.outdir}")


if __name__ == "__main__":
    main()
