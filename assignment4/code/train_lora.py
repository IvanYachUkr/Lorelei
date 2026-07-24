#!/usr/bin/env python
"""Train the final dual-adapter Stable Diffusion 1.5 style LoRA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, CLIPTextModel

from token_utils import setup_custom_token


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
TOKEN_KEY = "__custom_token_embedding__"
GENERIC_PROMPT = "an animated movie scene, in {token} style"
TARGET_PROMPT = "a busy market, in {token} style"
SAFETENSOR_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def derive_seed(master_seed: int, namespace: str, index: int) -> int:
    value = f"{master_seed}:{namespace}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_canonical_safetensors(
    tensors: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, str],
) -> None:
    header = {"__metadata__": dict(sorted(metadata.items()))}
    chunks = []
    offset = 0
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        if value.dtype not in SAFETENSOR_DTYPES:
            raise TypeError(f"Unsupported Safetensors dtype: {value.dtype}")
        chunk = value.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": SAFETENSOR_DTYPES[value.dtype],
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(chunk)],
        }
        chunks.append(chunk)
        offset += len(chunk)
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)
    os.replace(temporary, path)


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_jsonl(path: Path, step: int, key: str) -> None:
    if not path.exists():
        return
    rows = [
        row for row in read_jsonl(path)
        if int(row[key]) <= step
    ]
    write_jsonl(path, rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    candidates = [path, manifest.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (manifest.parent / path).resolve()


def load_captions(path: Path) -> dict[Path, str]:
    captions: dict[Path, str] = {}
    for row in read_jsonl(path):
        image_path = resolve_manifest_path(path, str(row["image"]))
        caption = str(row["caption"]).strip().rstrip(".,")
        if not caption:
            raise ValueError(f"Empty caption for {image_path}")
        if image_path in captions:
            raise ValueError(f"Duplicate caption for {image_path}")
        captions[image_path] = caption
    return captions


@dataclass(frozen=True)
class Example:
    image_path: Path
    caption: str
    source_kind: str
    sampling_weight: float
    target_prompt_prob: float


def load_auxiliary_examples(path: Path | None) -> list[Example]:
    if path is None:
        return []
    examples = []
    seen = set()
    for row in read_jsonl(path):
        image_path = resolve_manifest_path(path, str(row["image"]))
        caption = str(row["caption"]).strip().rstrip(".,")
        source_kind = str(row["source_kind"]).strip()
        sampling_weight = float(row["sampling_weight"])
        target_prompt_prob = float(row.get("target_prompt_prob", 0.0))
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(image_path)
        if not caption or not source_kind or sampling_weight <= 0:
            raise ValueError(f"Invalid auxiliary row for {image_path}")
        if not 0 <= target_prompt_prob <= 1:
            raise ValueError(f"Invalid target_prompt_prob for {image_path}")
        if image_path in seen:
            raise ValueError(f"Duplicate auxiliary image: {image_path}")
        seen.add(image_path)
        examples.append(
            Example(
                image_path,
                caption,
                source_kind,
                sampling_weight,
                target_prompt_prob,
            )
        )
    return examples


class StyleDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        captions_jsonl: Path,
        auxiliary_jsonl: Path | None,
        instance_token: str,
        resolution: int,
        caption_dropout_prob: float,
        random_flip: bool,
    ) -> None:
        supplied = sorted(
            path.resolve() for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        supplied_captions = load_captions(captions_jsonl)
        missing_supplied = [path for path in supplied if path not in supplied_captions]
        if missing_supplied or len(supplied_captions) != len(supplied):
            raise ValueError(
                f"Caption coverage mismatch: {len(supplied)} images, "
                f"{len(supplied_captions)} captions, {len(missing_supplied)} missing"
            )
        self.examples = [
            Example(path, supplied_captions[path], "supplied_full", 1.0, 0.0)
            for path in supplied
        ]
        self.examples.extend(load_auxiliary_examples(auxiliary_jsonl))
        self.instance_token = instance_token
        self.resolution = resolution
        self.caption_dropout_prob = caption_dropout_prob
        self.random_flip = random_flip
        self.tokenizer = None

    def __len__(self) -> int:
        return len(self.examples)

    def sampling_weights(self) -> list[float]:
        return [example.sampling_weight for example in self.examples]

    def prompts(self, example: Example, rng: random.Random) -> tuple[str, str, str]:
        if rng.random() < example.target_prompt_prob:
            styled = TARGET_PROMPT.format(token=self.instance_token)
            plain = "a busy market"
            mode = "target"
        elif rng.random() < self.caption_dropout_prob:
            styled = GENERIC_PROMPT.format(token=self.instance_token)
            plain = "an animated movie scene"
            mode = "generic"
        else:
            styled = f"{example.caption}, in {self.instance_token} style"
            plain = example.caption
            mode = "caption"
        if styled.count(self.instance_token) != 1 or self.instance_token in plain:
            raise ValueError(f"Invalid styled/plain prompt pair: {styled!r}, {plain!r}")
        return styled, plain, mode

    def __getitem__(self, item: tuple[int, int, int]) -> dict[str, object]:
        dataset_index, draw_index, augmentation_seed = item
        example = self.examples[dataset_index]
        rng = random.Random(augmentation_seed)

        with Image.open(example.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        original_width, original_height = image.size
        image = TF.resize(
            image,
            self.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        resized_width, resized_height = image.size
        max_top = max(0, resized_height - self.resolution)
        max_left = max(0, resized_width - self.resolution)
        crop_top = rng.randint(0, max_top) if max_top else 0
        crop_left = rng.randint(0, max_left) if max_left else 0
        image = TF.crop(image, crop_top, crop_left, self.resolution, self.resolution)
        flipped = self.random_flip and rng.random() < 0.5
        if flipped:
            image = TF.hflip(image)

        styled_prompt, plain_prompt, prompt_mode = self.prompts(example, rng)
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been attached to the dataset")
        tokenize = lambda prompt: self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        trace = {
            "draw_index": draw_index,
            "dataset_index": dataset_index,
            "augmentation_seed": augmentation_seed,
            "source_kind": example.source_kind,
            "source_path": portable_path(example.image_path),
            "original_width": original_width,
            "original_height": original_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": self.resolution,
            "crop_height": self.resolution,
            "flipped": flipped,
            "prompt_mode": prompt_mode,
            "styled_prompt": styled_prompt,
            "plain_prompt": plain_prompt,
        }
        return {
            "pixel_values": TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3),
            "input_ids": tokenize(styled_prompt),
            "plain_input_ids": tokenize(plain_prompt),
            "draw_index": torch.tensor(draw_index, dtype=torch.long),
            "trace": json.dumps(trace, sort_keys=True),
        }


class DrawSampler(Sampler[tuple[int, int, int]]):
    def __init__(self, schedule: list[dict[str, object]], offset: int = 0) -> None:
        self.schedule = schedule[offset:]

    def __iter__(self):
        for row in self.schedule:
            yield (
                int(row["dataset_index"]),
                int(row["draw_index"]),
                int(row["augmentation_seed"]),
            )

    def __len__(self) -> int:
        return len(self.schedule)


def build_schedule(
    dataset: StyleDataset,
    total_draws: int,
    seed: int,
) -> list[dict[str, object]]:
    weights = torch.tensor(dataset.sampling_weights(), dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(derive_seed(seed, "draw_schedule", 0))
    indices = torch.multinomial(weights, total_draws, replacement=True, generator=generator).tolist()
    rows = []
    for draw_index, dataset_index in enumerate(indices):
        example = dataset.examples[dataset_index]
        rows.append(
            {
                "draw_index": draw_index,
                "dataset_index": dataset_index,
                "augmentation_seed": derive_seed(seed, "augmentation", draw_index),
                "source_kind": example.source_kind,
                "source_path": portable_path(example.image_path),
                "sampling_weight": float(weights[dataset_index]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--captions_jsonl", type=Path, default=Path("code/auto_captions/florence_captions.jsonl"))
    parser.add_argument("--auxiliary_jsonl", type=Path, default=None)
    parser.add_argument("--instance_token", default="<sks>")
    parser.add_argument("--token_initializer", default="ghibli style")
    parser.add_argument("--output_dir", type=Path, default=Path("lora_out"))
    parser.add_argument("--provenance_dir", type=Path, default=Path("training_run"))
    parser.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--text_encoder_rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--text_encoder_learning_rate", type=float, default=5e-6)
    parser.add_argument("--token_learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--caption_dropout_prob", type=float, default=0.10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--preservation_loss_weight", type=float, default=0.5)
    parser.add_argument("--token_anchor_loss_weight", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["fp16", "bf16", "no"], default="fp16")
    parser.add_argument("--checkpointing_steps", type=int, default=25)
    parser.add_argument("--stop_after_step", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--init_weights", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if not args.captions_jsonl.is_file():
        raise FileNotFoundError(args.captions_jsonl)
    if args.auxiliary_jsonl is not None and not args.auxiliary_jsonl.is_file():
        raise FileNotFoundError(args.auxiliary_jsonl)
    if args.init_weights is not None and not args.init_weights.is_file():
        raise FileNotFoundError(args.init_weights)
    positive = {
        "rank": args.rank,
        "text_encoder_rank": args.text_encoder_rank,
        "resolution": args.resolution,
        "max_steps": args.max_steps,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    for name in ("caption_dropout_prob", "lora_dropout"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be between 0 and 1")
    if args.output_dir.resolve() == args.provenance_dir.resolve():
        raise ValueError("Output and provenance directories must be different")
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= args.max_steps:
        raise ValueError("--stop_after_step must be between 1 and --max_steps")


def prepare_directories(args: argparse.Namespace) -> Path:
    state_path = args.provenance_dir / "latest_training_state.pt"
    if args.overwrite:
        for path in (args.output_dir, args.provenance_dir):
            if path.exists():
                shutil.rmtree(path)
    elif args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
    elif args.output_dir.exists() or args.provenance_dir.exists():
        raise FileExistsError("Output exists; pass --overwrite or --resume")
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    return state_path


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "torchvision", "diffusers", "peft", "transformers", "safetensors"):
        versions[package] = importlib.metadata.version(package)
    return versions


def run_config(args: argparse.Namespace, schedule: list[dict[str, object]], schedule_hash: str) -> dict:
    arguments = {
        key: portable_path(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in {"overwrite", "resume", "stop_after_step", "output_dir", "provenance_dir"}
    }
    counts: dict[str, int] = {}
    for row in schedule:
        kind = str(row["source_kind"])
        counts[kind] = counts.get(kind, 0) + 1
    contract = {
        "arguments": arguments,
        "input_hashes": {
            "data_dir": directory_sha256(args.data_dir),
            "captions_jsonl": file_sha256(args.captions_jsonl),
            "auxiliary_jsonl": file_sha256(args.auxiliary_jsonl) if args.auxiliary_jsonl else None,
            "init_weights": file_sha256(args.init_weights) if args.init_weights else None,
            "train_script": file_sha256(Path(__file__)),
        },
        "schedule_sha256": schedule_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": True,
        },
    }
    return {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "contract": contract,
        "contract_sha256": json_sha256(contract),
        "source_summary": {
            "total_draws": len(schedule),
            "draws_by_source_kind": counts,
        },
    }


def load_models(args: argparse.Namespace, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae",
        revision=args.revision,
        torch_dtype=dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name,
        subfolder="scheduler",
        revision=args.revision,
    )
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def add_adapters(unet, text_encoder, args: argparse.Namespace) -> None:
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    text_encoder.add_adapter(
        LoraConfig(
            r=args.text_encoder_rank,
            lora_alpha=args.text_encoder_rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
    )


def train_token_row(text_encoder, token_id: int, device: torch.device):
    embedding = text_encoder.get_input_embeddings()
    embedding.weight.requires_grad_(False)
    parameter = torch.nn.Parameter(embedding.weight[token_id].detach().float().to(device).clone())

    def inject(_module, inputs, output):
        mask = inputs[0].eq(token_id).unsqueeze(-1)
        base = embedding.weight[token_id].detach().to(output)
        learned = parameter.to(output)
        return output + mask.to(output) * (learned - base)

    return parameter, embedding.register_forward_hook(inject)


def set_adapters(unet, text_encoder, enabled: bool) -> None:
    method = "enable_adapters" if enabled else "disable_adapters"
    getattr(unet, method)()
    getattr(text_encoder, method)()


def compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas = noise_scheduler.alphas_cumprod.to(timesteps.device, dtype=torch.float32)
    return alphas[timesteps] / (1 - alphas[timesteps])


def diffusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noise_scheduler,
    timesteps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    error = F.mse_loss(prediction.float(), target.float(), reduction="none")
    per_example = error.mean(dim=tuple(range(1, error.ndim)))
    snr = compute_snr(noise_scheduler, timesteps)
    weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return (per_example * weights).mean()


def seeded_noise(reference: torch.Tensor, seeds: list[int]) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=reference.device).manual_seed(seed)
        values.append(
            torch.randn(
                reference[0].shape,
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
    return torch.stack(values)


def seeded_timesteps(seeds: list[int], count: int, device: torch.device) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        values.append(torch.randint(0, count, (1,), generator=generator, device=device))
    return torch.cat(values).long()


def save_adapter(
    directory: Path,
    unet,
    text_encoder,
    token: torch.Tensor,
    args: argparse.Namespace,
    metadata: dict[str, str],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    unet_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    text_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(directory),
        unet_lora_layers=unet_state,
        text_encoder_lora_layers=text_state,
        safe_serialization=True,
    )
    path = directory / WEIGHTS_NAME
    tensors = dict(sorted(load_file(str(path), device="cpu").items()))
    tensors[TOKEN_KEY] = token.detach().float().cpu()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        existing_metadata = dict(sorted((handle.metadata() or {}).items()))
    adapter_metadata = {
            "base_model": args.model_name,
            "instance_token": args.instance_token,
            "contains_custom_token_embedding": "true",
            "custom_token_embedding_key": TOKEN_KEY,
            "lora_rank": str(args.rank),
            "text_encoder_lora_rank": str(args.text_encoder_rank),
            **metadata,
    }
    if args.init_weights is not None:
        adapter_metadata["initial_weights_sha256"] = file_sha256(args.init_weights)
    existing_metadata.update(adapter_metadata)
    save_canonical_safetensors(tensors, path, existing_metadata)
    for extra in directory.iterdir():
        if extra.name != WEIGHTS_NAME:
            if extra.is_dir():
                shutil.rmtree(extra)
            else:
                extra.unlink()
    return path


def save_state(
    path: Path,
    unet,
    text_encoder,
    token: torch.Tensor,
    optimizer,
    scheduler,
    scaler,
    step: int,
    loss_ema: float | None,
    schedule_hash: str,
) -> None:
    state = {
        "unet": {key: value.cpu() for key, value in get_peft_model_state_dict(unet).items()},
        "text_encoder": {
            key: value.cpu() for key, value in get_peft_model_state_dict(text_encoder).items()
        },
        "token": token.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "loss_ema": loss_ema,
        "schedule_sha256": schedule_hash,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_state(
    path: Path,
    unet,
    text_encoder,
    token: torch.Tensor,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    schedule_hash: str,
) -> tuple[int, float | None]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schedule_sha256"] != schedule_hash:
        raise ValueError("Checkpoint belongs to a different draw schedule")
    set_peft_model_state_dict(unet, state["unet"])
    set_peft_model_state_dict(text_encoder, state["text_encoder"])
    token.data.copy_(state["token"].to(device))
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]), state["loss_ema"]


def load_adapter_initialization(
    path: Path,
    unet,
    text_encoder,
    token: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_metadata = {
        "base_model": args.model_name,
        "instance_token": args.instance_token,
        "lora_rank": str(args.rank),
        "text_encoder_lora_rank": str(args.text_encoder_rank),
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Initialization metadata mismatch: {mismatches}")

    tensors = load_file(str(path), device="cpu")
    allowed = {
        key
        for key in tensors
        if key == TOKEN_KEY or key.startswith("unet.") or key.startswith("text_encoder.")
    }
    if allowed != set(tensors):
        raise ValueError(f"Unexpected initialization tensors: {sorted(set(tensors) - allowed)}")
    if TOKEN_KEY not in tensors:
        raise ValueError(f"Initialization is missing {TOKEN_KEY}")

    unet_state = {
        key[len("unet."):]: value for key, value in tensors.items() if key.startswith("unet.")
    }
    text_state = {
        key[len("text_encoder."):]: value
        for key, value in tensors.items()
        if key.startswith("text_encoder.")
    }
    if not unet_state or not text_state:
        raise ValueError("Initialization must contain both UNet and text-encoder LoRA tensors")

    unet_state = convert_unet_state_dict_to_peft(unet_state)
    text_state = convert_state_dict_to_peft(text_state)
    set_peft_model_state_dict(unet, unet_state)
    set_peft_model_state_dict(text_encoder, text_state)
    token.data.copy_(tensors[TOKEN_KEY].to(device=token.device, dtype=token.dtype))

    loaded_unet = get_peft_model_state_dict(unet)
    loaded_text = get_peft_model_state_dict(text_encoder)
    for name, expected, actual in (
        ("UNet", unet_state, loaded_unet),
        ("text encoder", text_state, loaded_text),
    ):
        if set(expected) != set(actual):
            raise ValueError(f"{name} initialization key mismatch")
        if any(not torch.equal(expected[key], actual[key].cpu()) for key in expected):
            raise ValueError(f"{name} initialization tensor mismatch")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parse_args()
    validate_args(args)
    state_path = prepare_directories(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is required for full training; pass --allow_cpu for a smoke test")
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.mixed_precision == "fp16" and device.type == "cuda":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    dataset = StyleDataset(
        data_dir=args.data_dir,
        captions_jsonl=args.captions_jsonl,
        auxiliary_jsonl=args.auxiliary_jsonl,
        instance_token=args.instance_token,
        resolution=args.resolution,
        caption_dropout_prob=args.caption_dropout_prob,
        random_flip=args.random_flip,
    )
    total_draws = args.max_steps * args.gradient_accumulation_steps * args.train_batch_size
    schedule = build_schedule(dataset, total_draws, args.seed)
    schedule_path = args.provenance_dir / "training_draw_schedule.jsonl"
    schedule_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule)
    schedule_hash = hashlib.sha256(schedule_payload.encode("utf-8")).hexdigest()
    if args.resume:
        existing_schedule = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in read_jsonl(schedule_path)
        )
        if hashlib.sha256(existing_schedule.encode("utf-8")).hexdigest() != schedule_hash:
            raise ValueError("Existing schedule does not match this configuration")
    else:
        schedule_path.write_text(schedule_payload, encoding="utf-8")
        write_json(args.provenance_dir / "run_config.json", run_config(args, schedule, schedule_hash))

    print(f"Loading {args.model_name}")
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(args, dtype)
    dataset.tokenizer = tokenizer
    token_id = setup_custom_token(tokenizer, text_encoder, args.instance_token, args.token_initializer)
    for model in (vae, unet, text_encoder):
        model.requires_grad_(False)
        model.eval()
    add_adapters(unet, text_encoder, args)

    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    token_parameter, token_hook = train_token_row(text_encoder, token_id, device)
    for model in (unet, text_encoder):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if args.init_weights is not None and not args.resume:
        load_adapter_initialization(
            args.init_weights,
            unet,
            text_encoder,
            token_parameter,
            args,
        )
    token_anchor = token_parameter.detach().clone()

    unet_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    text_parameters = [parameter for parameter in text_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": unet_parameters, "lr": args.learning_rate},
            {"params": text_parameters, "lr": args.text_encoder_learning_rate},
            {"params": [token_parameter], "lr": args.token_learning_rate, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.mixed_precision == "fp16"
    )

    step = 0
    loss_ema = None
    if args.resume:
        step, loss_ema = load_state(
            state_path,
            unet,
            text_encoder,
            token_parameter,
            optimizer,
            scheduler,
            scaler,
            device,
            schedule_hash,
        )
        truncate_jsonl(args.provenance_dir / "training_trace.jsonl", step, "optimizer_step")
        truncate_jsonl(args.provenance_dir / "training_metrics.jsonl", step, "step")

    draw_offset = step * args.gradient_accumulation_steps * args.train_batch_size
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=DrawSampler(schedule, draw_offset),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    trace_path = args.provenance_dir / "training_trace.jsonl"
    metrics_path = args.provenance_dir / "training_metrics.jsonl"
    if not args.resume:
        trace_path.write_text("", encoding="utf-8")
        metrics_path.write_text("", encoding="utf-8")

    unet.train()
    text_encoder.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, initial=step, desc="Training LoRA")
    started = time.monotonic()
    pending_traces: list[dict[str, object]] = []
    style_total = preservation_total = anchor_total = objective_total = 0.0
    accumulation = 0
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "no"
    autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    for batch in loader:
        pixels = batch["pixel_values"].to(device=device, dtype=dtype)
        input_ids = batch["input_ids"].to(device)
        plain_ids = batch["plain_input_ids"].to(device)
        draw_indices = [int(value) for value in batch["draw_index"].tolist()]
        latent_seeds = [derive_seed(args.seed, "vae_latent", index) for index in draw_indices]
        noise_seeds = [derive_seed(args.seed, "diffusion_noise", index) for index in draw_indices]
        timestep_seeds = [derive_seed(args.seed, "timestep", index) for index in draw_indices]

        with torch.no_grad():
            latent_distribution = vae.encode(pixels).latent_dist
            latent_noise = seeded_noise(latent_distribution.mean, latent_seeds)
            latents = latent_distribution.mean + latent_distribution.std * latent_noise
            latents = latents * vae.config.scaling_factor
        noise = seeded_noise(latents, noise_seeds)
        timesteps = seeded_timesteps(
            timestep_seeds,
            noise_scheduler.config.num_train_timesteps,
            device,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        preservation_loss = torch.zeros((), device=device)
        if args.preservation_loss_weight:
            set_adapters(unet, text_encoder, False)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                teacher_hidden = text_encoder(plain_ids, return_dict=False)[0]
                teacher_prediction = unet(
                    noisy_latents, timesteps, teacher_hidden, return_dict=False
                )[0]
            set_adapters(unet, text_encoder, True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                student_hidden = text_encoder(plain_ids, return_dict=False)[0]
                student_prediction = unet(
                    noisy_latents, timesteps, student_hidden, return_dict=False
                )[0]
                preservation_loss = F.mse_loss(
                    student_prediction.float(), teacher_prediction.float()
                )
            scaler.scale(
                args.preservation_loss_weight
                * preservation_loss
                / args.gradient_accumulation_steps
            ).backward()
            del teacher_hidden, teacher_prediction, student_hidden, student_prediction

        with torch.amp.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            hidden = text_encoder(input_ids, return_dict=False)[0]
            prediction = unet(noisy_latents, timesteps, hidden, return_dict=False)[0]
            target = noise
            style_loss = diffusion_loss(
                prediction,
                target,
                noise_scheduler,
                timesteps,
                args.snr_gamma,
            )
            anchor_loss = (token_parameter - token_anchor).square().sum()
            trainable_loss = style_loss + args.token_anchor_loss_weight * anchor_loss
        scaler.scale(trainable_loss / args.gradient_accumulation_steps).backward()

        factor = 1 / args.gradient_accumulation_steps
        style_total += float(style_loss.detach()) * factor
        preservation_total += float(preservation_loss.detach()) * factor
        anchor_total += float(anchor_loss.detach()) * factor
        objective_total += (
            float(style_loss.detach())
            + args.preservation_loss_weight * float(preservation_loss.detach())
            + args.token_anchor_loss_weight * float(anchor_loss.detach())
        ) * factor

        for index, raw_trace in enumerate(batch["trace"]):
            trace = json.loads(raw_trace)
            trace.update(
                {
                    "optimizer_step": step + 1,
                    "accumulation_index": accumulation % args.gradient_accumulation_steps,
                    "vae_latent_seed": latent_seeds[index],
                    "diffusion_noise_seed": noise_seeds[index],
                    "timestep_seed": timestep_seeds[index],
                    "timestep": int(timesteps[index]),
                    "style_loss": float(style_loss.detach()),
                    "preservation_loss": float(preservation_loss.detach()),
                    "token_anchor_loss": float(anchor_loss.detach()),
                }
            )
            pending_traces.append(trace)

        accumulation += 1
        if accumulation % args.gradient_accumulation_steps:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            unet_parameters + text_parameters + [token_parameter], args.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        loss_ema = objective_total if loss_ema is None else 0.9 * loss_ema + 0.1 * objective_total

        append_jsonl(trace_path, pending_traces)
        append_jsonl(
            metrics_path,
            [
                {
                    "step": step,
                    "style_loss": style_total,
                    "preservation_loss": preservation_total,
                    "token_anchor_loss": anchor_total,
                    "objective": objective_total,
                    "objective_ema": loss_ema,
                    "unet_lr": optimizer.param_groups[0]["lr"],
                    "text_encoder_lr": optimizer.param_groups[1]["lr"],
                    "token_lr": optimizer.param_groups[2]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            ],
        )
        pending_traces = []
        progress.update(1)
        progress.set_postfix(objective=f"{loss_ema:.4f}")

        if args.checkpointing_steps and (
            step % args.checkpointing_steps == 0 or step == args.max_steps
        ):
            checkpoint = args.provenance_dir / "checkpoints" / f"step_{step:06d}"
            save_adapter(
                checkpoint,
                unet,
                text_encoder,
                token_parameter,
                args,
                {"checkpoint_kind": "periodic", "training_step": str(step)},
            )
            save_state(
                state_path,
                unet,
                text_encoder,
                token_parameter,
                optimizer,
                scheduler,
                scaler,
                step,
                loss_ema,
                schedule_hash,
            )

        style_total = preservation_total = anchor_total = objective_total = 0.0
        if step >= args.max_steps or (
            args.stop_after_step is not None and step >= args.stop_after_step
        ):
            break

    progress.close()
    expected_step = args.stop_after_step or args.max_steps
    if step != expected_step:
        raise RuntimeError(f"Training stopped at step {step}, expected {expected_step}")

    append_jsonl(
        args.provenance_dir / "sessions.jsonl",
        [
            {
                "completed": step == args.max_steps,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "stop_after_step": args.stop_after_step,
            }
        ],
    )

    metadata = {
        "training_step": str(step),
        "seed": str(args.seed),
        "unet_learning_rate": str(args.learning_rate),
        "text_encoder_learning_rate": str(args.text_encoder_learning_rate),
        "token_learning_rate": str(args.token_learning_rate),
        "snr_gamma": str(args.snr_gamma),
        "preservation_loss_weight": str(args.preservation_loss_weight),
        "token_anchor_loss_weight": str(args.token_anchor_loss_weight),
        "run_status": "complete" if step == args.max_steps else "paused",
        "training_draw_schedule_sha256": schedule_hash,
        "training_trace_sha256": file_sha256(trace_path),
    }
    weights_path = save_adapter(
        args.output_dir,
        unet,
        text_encoder,
        token_parameter,
        args,
        metadata,
    )
    token_hook.remove()
    print(f"Saved {weights_path}")


if __name__ == "__main__":
    main()
