#!/usr/bin/env python
"""Train a dual-adapter LoRA style model for Stable Diffusion 1.5.

The script follows the assignment contract:
- add a custom style token, normally "<sks>"
- train LoRA adapters for both the UNet and CLIP text encoder
- save a single file: output_dir/pytorch_lora_weights.safetensors
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
except ImportError as exc:  # pragma: no cover - exercised in a fresh env before install.
    raise SystemExit(
        "Missing Diffusers dependencies. Install them with: pip install -r requirements.txt"
    ) from exc

try:
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PEFT. Install it with: pip install -r requirements.txt") from exc

try:
    from transformers import AutoTokenizer, CLIPTextModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing Transformers. Install it with: pip install -r requirements.txt") from exc

from config import (
    CUSTOM_TOKEN_EMBEDDING_KEY,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPSILON,
    DEFAULT_ADAM_WEIGHT_DECAY,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_INSTANCE_TOKEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_WARMUP_STEPS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_STEPS,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_RANK,
    DEFAULT_REPEATS,
    DEFAULT_RESOLUTION,
    DEFAULT_TOKEN_INITIALIZER,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_SEED,
    LORA_FILENAME,
    SUPPORTED_EXTENSIONS,
)
from token_utils import setup_custom_token


class StyleImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        instance_token: str,
        resolution: int,
        prompt_template: str,
        repeats: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.image_paths = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No supported image files found in {data_dir}")

        crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        image_transforms = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
        ]
        if random_flip:
            image_transforms.append(transforms.RandomHorizontalFlip())
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.transforms = transforms.Compose(image_transforms)
        self.tokenizer = tokenizer
        self.prompt = prompt_template.format(instance_token=instance_token)
        self.repeats = max(1, repeats)

    def __len__(self) -> int:
        return len(self.image_paths) * self.repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[index % len(self.image_paths)]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixel_values = self.transforms(image)

        input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 1.5 LoRA adapters for a style token.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory of training images.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--variant", default=None, help="Optional model variant, such as fp16.")
    parser.add_argument("--instance_token", default=DEFAULT_INSTANCE_TOKEN, help="New tokenizer token used in prompts.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the final LoRA file.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Base SD 1.5 model id or path.")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha. Defaults to rank.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=DEFAULT_ADAM_BETA1)
    parser.add_argument("--adam_beta2", type=float, default=DEFAULT_ADAM_BETA2)
    parser.add_argument("--adam_weight_decay", type=float, default=DEFAULT_ADAM_WEIGHT_DECAY)
    parser.add_argument("--adam_epsilon", type=float, default=DEFAULT_ADAM_EPSILON)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--train_batch_size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--lr_scheduler", default=DEFAULT_LR_SCHEDULER, help="Diffusers scheduler name.")
    parser.add_argument("--lr_warmup_steps", type=int, default=DEFAULT_LR_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Use 0 on Windows unless you need multiprocessing.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Virtual repeats for small datasets.")
    parser.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Training prompt template. Must contain {instance_token}.",
    )

    parser.add_argument(
        "--token_initializer",
        default=DEFAULT_TOKEN_INITIALIZER,
        help="Phrase whose token embeddings initialize the new custom token.",
    )
    parser.add_argument("--center_crop", action="store_true", help="Use center crop instead of random crop.")
    parser.add_argument("--random_flip", action="store_true", help="Apply random horizontal flips.")
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default=DEFAULT_MIXED_PRECISION,
        help="Training precision. fp16 is recommended on CUDA GPUs.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xFormers attention if installed.")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU training. This is extremely slow.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if "{instance_token}" not in args.prompt_template:
        raise ValueError("--prompt_template must include {instance_token}")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    if args.output_dir.exists():
        existing_files = [path for path in args.output_dir.iterdir()]
        if existing_files and not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty. Re-run with --overwrite to replace it.")
        if args.overwrite:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise SystemExit("CUDA was not found. Training on CPU is impractical; pass --allow_cpu only for debugging.")


def get_weight_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def import_model_components(args: argparse.Namespace, torch_dtype: torch.dtype):
    common_kwargs = {
        "pretrained_model_name_or_path": args.model_name,
        "revision": args.revision,
    }
    if args.variant is not None:
        common_kwargs["variant"] = args.variant

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        **{k: v for k, v in common_kwargs.items() if k != "pretrained_model_name_or_path"},
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae",
        torch_dtype=torch_dtype,
        **{k: v for k, v in common_kwargs.items() if k != "pretrained_model_name_or_path"},
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name,
        subfolder="unet",
        torch_dtype=torch_dtype,
        **{k: v for k, v in common_kwargs.items() if k != "pretrained_model_name_or_path"},
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name,
        subfolder="scheduler",
        revision=args.revision,
    )
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def freeze_models(*models) -> None:
    for model in models:
        model.requires_grad_(False)
        model.eval()


def add_lora_adapters(unet, text_encoder, rank: int, alpha: int) -> None:
    unet_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    text_encoder_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet.add_adapter(unet_config)
    text_encoder.add_adapter(text_encoder_config)


def cast_trainable_parameters(models: Iterable[torch.nn.Module], dtype: torch.dtype) -> None:
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype)


def enable_token_row_training(text_encoder, token_id: int, device: torch.device) -> torch.nn.Parameter:
    embedding_param = text_encoder.get_input_embeddings().weight
    embedding_param.requires_grad_(True)

    row_mask = torch.zeros((embedding_param.shape[0], 1), device=device, dtype=embedding_param.dtype)
    row_mask[token_id] = 1

    def mask_embedding_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype)

    embedding_param.register_hook(mask_embedding_grad)
    return embedding_param


def trainable_parameters(unet, text_encoder, token_embedding_param: torch.nn.Parameter):
    lora_params = []
    for model in (unet, text_encoder):
        lora_params.extend(param for param in model.parameters() if param.requires_grad and param is not token_embedding_param)
    return lora_params


def save_single_lora_file(
    output_dir: Path,
    unet,
    text_encoder,
    token_embedding: torch.Tensor,
    instance_token: str,
    model_name: str,
    rank: int,
) -> Path:
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    text_encoder_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))

    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
        safe_serialization=True,
    )

    weights_path = output_dir / LORA_FILENAME
    tensors = load_file(str(weights_path), device="cpu")
    tensors[CUSTOM_TOKEN_EMBEDDING_KEY] = token_embedding.detach().float().cpu()

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    metadata.update(
        {
            "base_model": model_name,
            "instance_token": instance_token,
            "contains_custom_token_embedding": "true",
            "custom_token_embedding_key": CUSTOM_TOKEN_EMBEDDING_KEY,
            "lora_rank": str(rank),
        }
    )
    save_file(tensors, str(weights_path), metadata=metadata)

    for path in output_dir.iterdir():
        if path.name != LORA_FILENAME:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    return weights_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = get_device(args.allow_cpu)
    weight_dtype = get_weight_dtype(args.mixed_precision, device)
    alpha = args.lora_alpha if args.lora_alpha is not None else args.rank

    print(f"Loading base model: {args.model_name}")
    tokenizer, text_encoder, vae, unet, noise_scheduler = import_model_components(args, weight_dtype)

    token_id = setup_custom_token(tokenizer, text_encoder, args.instance_token, args.token_initializer)
    freeze_models(vae, unet, text_encoder)
    add_lora_adapters(unet, text_encoder, rank=args.rank, alpha=alpha)

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            raise RuntimeError("Could not enable xFormers attention. Is xformers installed?") from exc

    token_embedding_param = enable_token_row_training(text_encoder, token_id, device)
    cast_trainable_parameters([unet, text_encoder], torch.float32)

    dataset = StyleImageDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        instance_token=args.instance_token,
        resolution=args.resolution,
        prompt_template=args.prompt_template,
        repeats=args.repeats,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    lora_params = trainable_parameters(unet, text_encoder, token_embedding_param)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "weight_decay": args.adam_weight_decay},
            {"params": [token_embedding_param], "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_steps,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    progress_bar = tqdm(total=args.max_steps, desc="Training LoRA")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    unet.train()
    text_encoder.train()

    accumulation_counter = 0
    for _epoch in range(num_epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device=device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            autocast_enabled = device.type == "cuda" and args.mixed_precision != "no"
            autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            accumulation_counter += 1
            should_step = accumulation_counter % args.gradient_accumulation_steps == 0
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in optimizer.param_groups for param in group["params"]],
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    progress_bar.close()

    token_embedding = text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
    weights_path = save_single_lora_file(
        output_dir=args.output_dir,
        unet=unet,
        text_encoder=text_encoder,
        token_embedding=token_embedding,
        instance_token=args.instance_token,
        model_name=args.model_name,
        rank=args.rank,
    )

    del vae, unet, text_encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved final adapter: {weights_path}")


if __name__ == "__main__":
    main()
