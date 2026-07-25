#!/usr/bin/env python
"""Render samples from the trained Stable Diffusion 1.5 LoRA adapter."""

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
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


def parse_args():
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
    parser.add_argument("--num_inference_steps", type=int, default=150)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--baseline", action="store_true", help="Also render baseline images before loading the adapter.")
    return parser.parse_args()


def add_or_restore_custom_token(pipe, weights, metadata, instance_token_arg):
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



def make_lora_only_file(weights, metadata, tmpdir):
    tensors = load_file(str(weights), device="cpu")
    tensors = {key: value for key, value in tensors.items() if key != CUSTOM_TOKEN_EMBEDDING_KEY}
    filtered_path = tmpdir / LORA_FILENAME
    save_file(tensors, str(filtered_path), metadata=metadata)
    return filtered_path


def load_pipeline(args, device, dtype):
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
    pipe,
    prompt,
    outdir,
    prefix,
    num_images,
    seed,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    device,
):
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(num_images):
        image_seed = seed + index
        generator = torch.Generator(device=device.type).manual_seed(image_seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        image_path = outdir / f"{prefix}_{index:02d}.png"
        image.save(image_path)
        records.append(
            {
                "kind": prefix,
                "index": index,
                "seed": image_seed,
                "path": image_path.name,
            }
        )
    return records


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.num_images < 3:
        raise ValueError("--num_images must be at least 3 to satisfy the assignment")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "float32" or device.type != "cuda":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    with safe_open(str(args.weights), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    torch.use_deterministic_algorithms(True, warn_only=False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    pipe = load_pipeline(args, device=device, dtype=dtype)
    base_vocab_size = len(pipe.tokenizer)
    print(f"Base model '{args.model_name}' loaded, num of tokens: {base_vocab_size}")

    add_or_restore_custom_token(pipe, args.weights, metadata, args.instance_token)
    new_vocab_size = len(pipe.tokenizer)
    instance_token = args.instance_token or metadata.get("instance_token") or "<sks>"

    image_records = []

    if args.baseline:
        image_records.extend(render_images(
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
        ))

    with tempfile.TemporaryDirectory(prefix="lora_eval_") as tmp:
        lora_path = make_lora_only_file(args.weights, metadata, Path(tmp))
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    print(f"LoRA model '{args.weights}' loaded, num of tokens: {new_vocab_size} (base model tokens + {instance_token} token)")


    image_records.extend(render_images(
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
    ))

    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "base_model": args.model_name,
        "base_model_revision": args.revision,
        "base_model_variant": args.variant,
        "adapter_path": str(args.weights.resolve()),
        "adapter_metadata": metadata,
        "prompt": args.prompt,
        "instance_token": args.instance_token or metadata.get("instance_token"),
        "num_images": args.num_images,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "python": sys.version,
        "deterministic_algorithms": True,
        "images": image_records,
    }
    manifest_path = args.outdir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved {args.num_images} adapter samples and {manifest_path}")


if __name__ == "__main__":
    main()
