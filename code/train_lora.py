#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel


WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
TOKEN_KEY = "__custom_token_embedding__"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1
MAX_GRAD_NORM = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--instance_token", default="<sks>")
    parser.add_argument("--output_dir", type=Path, default=Path("lora_out"))
    parser.add_argument("--captions_jsonl", type=Path)
    parser.add_argument("--auxiliary_jsonl", type=Path)
    parser.add_argument("--token_initializer", default="ghibli style")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--text_encoder_rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--text_encoder_learning_rate", type=float, default=2.5e-6)
    parser.add_argument("--token_learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--caption_dropout_prob", type=float, default=0.08)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--scheduler_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--preservation_loss_weight", type=float, default=0.65)
    parser.add_argument("--token_anchor_loss_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2202)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def derive_seed(seed, name, index):
    value = f"{seed}:{name}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image(manifest, value):
    return (manifest.parent / Path(value)).resolve()


def load_examples(args):
    images = sorted(
        path.resolve()
        for path in args.data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".png"}
    )
    if not images:
        raise ValueError(f"No images found in {args.data_dir}")

    captions = {}
    if args.captions_jsonl:
        for row in read_jsonl(args.captions_jsonl):
            image = resolve_image(args.captions_jsonl, row["image"])
            if image in captions:
                raise ValueError(f"Duplicate caption path: {row['image']}")
            captions[image] = row["caption"].strip().rstrip(".,")
        if set(captions) != set(images):
            raise ValueError("The captions file must cover every supplied image exactly once")

    examples = [
        {
            "image": image,
            "caption": captions.get(image, "an animated movie scene"),
            "weight": 1.0,
            "target_prompt_prob": 0.0,
        }
        for image in images
    ]

    if args.auxiliary_jsonl:
        for row in read_jsonl(args.auxiliary_jsonl):
            image = resolve_image(args.auxiliary_jsonl, row["image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            examples.append(
                {
                    "image": image,
                    "caption": row["caption"].strip().rstrip(".,"),
                    "weight": float(row["sampling_weight"]),
                    "target_prompt_prob": float(row.get("target_prompt_prob", 0.0)),
                }
            )
    return examples


class ImageDataset(Dataset):
    def __init__(self, examples, tokenizer, token, resolution, caption_dropout, random_flip):
        self.examples = examples
        self.tokenizer = tokenizer
        self.token = token
        self.resolution = resolution
        self.caption_dropout = caption_dropout
        self.random_flip = random_flip

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example_index, draw_index, augmentation_seed = item
        example = self.examples[example_index]
        rng = random.Random(augmentation_seed)

        with Image.open(example["image"]) as source:
            image = source.convert("RGB")
        image = TF.resize(image, self.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        max_top = max(0, image.height - self.resolution)
        max_left = max(0, image.width - self.resolution)
        top = rng.randint(0, max_top) if max_top else 0
        left = rng.randint(0, max_left) if max_left else 0
        image = TF.crop(image, top, left, self.resolution, self.resolution)
        if self.random_flip and rng.random() < 0.5:
            image = TF.hflip(image)

        if rng.random() < example["target_prompt_prob"]:
            styled_prompt = f"a busy market, in {self.token} style"
            plain_prompt = "a busy market"
        elif rng.random() < self.caption_dropout:
            styled_prompt = f"an animated movie scene, in {self.token} style"
            plain_prompt = "an animated movie scene"
        else:
            styled_prompt = f"{example['caption']}, in {self.token} style"
            plain_prompt = example["caption"]

        def tokenize(prompt):
            return self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        return {
            "pixel_values": TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3),
            "input_ids": tokenize(styled_prompt),
            "plain_input_ids": tokenize(plain_prompt),
            "draw_index": draw_index,
        }


def build_schedule(examples, total_draws, seed):
    weights = torch.tensor([example["weight"] for example in examples], dtype=torch.float64)
    generator = torch.Generator().manual_seed(derive_seed(seed, "draw_schedule", 0))
    indices = torch.multinomial(weights, total_draws, replacement=True, generator=generator).tolist()
    return [
        (example_index, draw_index, derive_seed(seed, "augmentation", draw_index))
        for draw_index, example_index in enumerate(indices)
    ]


def add_token(tokenizer, text_encoder, token, initializer):
    if tokenizer.add_tokens([token]) != 1:
        raise ValueError(f"{token} already exists in the tokenizer")
    token_id = tokenizer.convert_tokens_to_ids(token)
    if tokenizer(token, add_special_tokens=False).input_ids != [token_id]:
        raise ValueError(f"{token} must be one tokenizer token")
    text_encoder.resize_token_embeddings(len(tokenizer))
    initializer_ids = tokenizer(initializer, add_special_tokens=False).input_ids
    embedding = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        ids = torch.tensor(initializer_ids, device=embedding.device)
        embedding[token_id].copy_(embedding[ids].mean(dim=0))
    return token_id


def add_lora(unet, text_encoder, args):
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


def train_token_embedding(text_encoder, token_id, device):
    embedding = text_encoder.get_input_embeddings()
    embedding.weight.requires_grad_(False)
    token = torch.nn.Parameter(embedding.weight[token_id].detach().float().to(device).clone())

    def replace_token(_module, inputs, output):
        mask = inputs[0].eq(token_id).unsqueeze(-1)
        original = embedding.weight[token_id].detach().to(output)
        return output + mask.to(output) * (token.to(output) - original)

    return token, embedding.register_forward_hook(replace_token)


def set_lora(unet, text_encoder, enabled):
    method = "enable_adapters" if enabled else "disable_adapters"
    getattr(unet, method)()
    getattr(text_encoder, method)()


def seeded_noise(reference, seeds):
    return torch.stack(
        [
            torch.randn(
                reference[0].shape,
                generator=torch.Generator(device=reference.device).manual_seed(seed),
                device=reference.device,
                dtype=reference.dtype,
            )
            for seed in seeds
        ]
    )


def seeded_timesteps(seeds, count, device):
    return torch.cat(
        [
            torch.randint(
                0,
                count,
                (1,),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
            )
            for seed in seeds
        ]
    ).long()


def diffusion_loss(prediction, target, noise_scheduler, timesteps, gamma):
    error = F.mse_loss(prediction.float(), target.float(), reduction="none")
    error = error.mean(dim=tuple(range(1, error.ndim)))
    alphas = noise_scheduler.alphas_cumprod.to(timesteps.device, dtype=torch.float32)
    snr = alphas[timesteps] / (1 - alphas[timesteps])
    return (error * torch.minimum(snr, torch.full_like(snr, gamma)) / snr).mean()


def save_weights(output_dir, unet, text_encoder, token, args):
    unet_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    text_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    StableDiffusionPipeline.save_lora_weights(
        str(output_dir),
        unet_lora_layers=unet_state,
        text_encoder_lora_layers=text_state,
        safe_serialization=True,
    )
    path = output_dir / WEIGHTS_NAME
    tensors = load_file(str(path), device="cpu")
    tensors[TOKEN_KEY] = token.detach().float().cpu().contiguous()
    metadata = {
        "base_model": MODEL_NAME,
        "instance_token": args.instance_token,
        "contains_custom_token_embedding": "true",
        "custom_token_embedding_key": TOKEN_KEY,
        "lora_rank": str(args.rank),
        "text_encoder_lora_rank": str(args.text_encoder_rank),
        "training_step": str(args.max_steps),
    }
    save_file(dict(sorted(tensors.items())), str(path), metadata=metadata)
    return path


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if args.captions_jsonl and not args.captions_jsonl.is_file():
        raise FileNotFoundError(args.captions_jsonl)
    if args.auxiliary_jsonl and not args.auxiliary_jsonl.is_file():
        raise FileNotFoundError(args.auxiliary_jsonl)
    if args.max_steps > args.scheduler_steps:
        raise ValueError("--max_steps cannot exceed --scheduler_steps")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    device = torch.device("cuda")
    dtype = torch.float16
    examples = load_examples(args)
    total_draws = args.scheduler_steps * args.gradient_accumulation_steps * TRAIN_BATCH_SIZE
    schedule = build_schedule(examples, total_draws, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    token_id = add_token(tokenizer, text_encoder, args.instance_token, args.token_initializer)
    for model in (vae, unet, text_encoder):
        model.requires_grad_(False)
        model.eval()
    add_lora(unet, text_encoder, args)
    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    token, token_hook = train_token_embedding(text_encoder, token_id, device)
    for model in (unet, text_encoder):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    token_anchor = token.detach().clone()
    unet_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    text_parameters = [parameter for parameter in text_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": unet_parameters, "lr": args.learning_rate},
            {"params": text_parameters, "lr": args.text_encoder_learning_rate},
            {"params": [token], "lr": args.token_learning_rate, "weight_decay": 0.0},
        ],
        weight_decay=0.01,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.scheduler_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    dataset = ImageDataset(
        examples,
        tokenizer,
        args.instance_token,
        RESOLUTION,
        args.caption_dropout_prob,
        args.random_flip,
    )
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=schedule,
        num_workers=0,
        pin_memory=True,
    )

    unet.train()
    text_encoder.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="Training LoRA")
    step = 0
    accumulation = 0

    for batch in loader:
        pixels = batch["pixel_values"].to(device=device, dtype=dtype)
        input_ids = batch["input_ids"].to(device)
        plain_ids = batch["plain_input_ids"].to(device)
        draw_indices = [int(index) for index in batch["draw_index"].tolist()]
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
            timestep_seeds, noise_scheduler.config.num_train_timesteps, device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        set_lora(unet, text_encoder, False)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            teacher_hidden = text_encoder(plain_ids, return_dict=False)[0]
            teacher_prediction = unet(
                noisy_latents, timesteps, teacher_hidden, return_dict=False
            )[0]
        set_lora(unet, text_encoder, True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
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

        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            hidden = text_encoder(input_ids, return_dict=False)[0]
            prediction = unet(noisy_latents, timesteps, hidden, return_dict=False)[0]
            style_loss = diffusion_loss(
                prediction, noise, noise_scheduler, timesteps, args.snr_gamma
            )
            anchor_loss = (token - token_anchor).square().sum()
            loss = style_loss + args.token_anchor_loss_weight * anchor_loss
        scaler.scale(loss / args.gradient_accumulation_steps).backward()

        accumulation += 1
        if accumulation % args.gradient_accumulation_steps:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            unet_parameters + text_parameters + [token], MAX_GRAD_NORM
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update()
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if step == args.max_steps:
            break

    progress.close()
    token_hook.remove()
    path = save_weights(args.output_dir, unet, text_encoder, token, args)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
