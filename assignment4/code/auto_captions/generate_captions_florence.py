#!/usr/bin/env python
"""Generate short Florence-2 captions for style_imgs/512."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from transformers.configuration_utils import PretrainedConfig
except ImportError as exc:
    raise SystemExit("Missing Transformers. Install with: pip install -r requirements.txt") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = ASSIGNMENT_DIR / "style_imgs" / "512"
OUT_PATH = SCRIPT_DIR / "florence_captions.jsonl"

MODEL_NAME = "microsoft/Florence-2-large"
TASK = "<CAPTION>"
STYLE_TOKEN = "<sks>"
NUM_BEAMS = 3
MAX_NEW_TOKENS = 40
MAX_WORDS = 18
ATTN_IMPLEMENTATION = "eager"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
STYLE_OR_FRANCHISE_PHRASES = (
    "studio ghibli",
    "ghibli",
    "hayao miyazaki",
    "miyazaki",
    "disney pixar",
    "disney",
    "pixar",
    "dreamworks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption style_imgs/512 with Florence-2.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def collect_images(limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No images found in {DATA_DIR}")
    return paths


def batched(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def patch_florence_compat() -> None:
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None


def load_processor():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if getattr(tokenizer, "additional_special_tokens", None) is None:
        tokenizer.additional_special_tokens = []
    return AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, tokenizer=tokenizer)


def load_model(dtype: torch.dtype):
    kwargs = {"trust_remote_code": True, "attn_implementation": ATTN_IMPLEMENTATION}
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, **kwargs)

    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "model"):
        shared_weight = language_model.model.shared.weight
        language_model.model.encoder.embed_tokens.weight = shared_weight
        language_model.model.decoder.embed_tokens.weight = shared_weight
        language_model.lm_head.weight = shared_weight
    return model


def to_device(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        elif hasattr(value, "to"):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def strip_generated_tokens(text: str) -> str:
    text = re.sub(r"(?:<pad>|</s>|<s>)+", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_caption(processor, generated_text: str, image: Image.Image) -> str:
    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=TASK,
            image_size=(image.width, image.height),
        )
        if isinstance(parsed, dict) and isinstance(parsed.get(TASK), str):
            return parsed[TASK]
    except Exception:
        pass
    return generated_text


def clean_caption(text: str) -> str:
    text = strip_generated_tokens(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(an?|the)\s+scene\s+(showing|of|with)\s+", r"a ", text)
    text = re.sub(
        r"^(an?|the)\s+(painting|drawing|illustration|cartoon|anime image|image|picture|photo|photograph)\s+of\s+",
        r"a ",
        text,
    )
    text = re.sub(r"\b(painted|drawn|illustrated|cartoon|anime|animated)\s+", "", text)
    text = re.sub(r",?\s+in\s+(the\s+)?style\s+of\b.*$", "", text)
    for phrase in STYLE_OR_FRANCHISE_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    # Prefix substitutions above insert "a " in front of a subject that Florence
    # already articled (e.g. "a painting of a path" -> "a a path"); drop the dup.
    text = re.sub(r"^(an?|the)\s+(?=(an?|the)\b)", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]).rstrip(" ,.;:")
    return text or "a scene"


def prompt_for(caption: str) -> str:
    return f"{caption.strip().rstrip('.')}, in {STYLE_TOKEN} style"


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ASSIGNMENT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_paths = collect_images(args.limit)
    device = select_device()
    dtype = weight_dtype(device)
    patch_florence_compat()

    processor = load_processor()
    model = load_model(dtype).to(device)
    model.eval()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for batch_paths in tqdm(list(batched(image_paths, args.batch_size)), desc="Captioning"):
            images = [open_rgb(path) for path in batch_paths]
            inputs = processor(text=[TASK] * len(images), images=images, padding=True, return_tensors="pt")
            inputs = to_device(inputs, device, dtype)

            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    use_cache=False,
                    early_stopping=False,
                )

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
            for path, image, generated_text in zip(batch_paths, images, generated_texts):
                raw_caption = strip_generated_tokens(extract_caption(processor, generated_text, image))
                caption = clean_caption(raw_caption)
                row = {
                    "image": display_path(path),
                    "caption_raw": raw_caption,
                    "caption": caption,
                    "prompt": prompt_for(caption),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(image_paths)} captions to {OUT_PATH}")


if __name__ == "__main__":
    main()
