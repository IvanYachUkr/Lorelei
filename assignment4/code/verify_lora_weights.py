#!/usr/bin/env python
"""Inspect the final safetensors adapter for required assignment contents."""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open


CUSTOM_TOKEN_EMBEDDING_KEY = "__custom_token_embedding__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final LoRA safetensors contents.")
    parser.add_argument("--weights", type=Path, default=Path("lora_out/pytorch_lora_weights.safetensors"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    with safe_open(str(args.weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})

    has_unet = any(key.startswith("unet.") or ".unet." in key or key.startswith("lora_unet_") for key in keys)
    has_text_encoder = any(
        key.startswith("text_encoder.") or ".text_encoder." in key or key.startswith("lora_te_") for key in keys
    )
    has_custom_embedding = CUSTOM_TOKEN_EMBEDDING_KEY in keys

    if not has_unet:
        raise AssertionError("No UNet LoRA tensors found")
    if not has_text_encoder:
        raise AssertionError("No text encoder LoRA tensors found")
    if not has_custom_embedding:
        raise AssertionError(f"Missing custom token embedding tensor: {CUSTOM_TOKEN_EMBEDDING_KEY}")

    print(f"Tensor count: {len(keys)}")
    print(f"Metadata: {metadata}")
    print("LoRA weights verification passed.")


if __name__ == "__main__":
    main()
