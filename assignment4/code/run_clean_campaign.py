#!/usr/bin/env python
"""Run one reviewed session of a provenance-clean LoRA candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CANDIDATES = {
    "supplied_only": {
        "auxiliary_jsonl": None,
        "max_steps": 500,
        "seed": 1101,
        "learning_rate": "8e-5",
        "text_encoder_learning_rate": "2.5e-6",
        "token_learning_rate": "1.2e-5",
        "caption_dropout_prob": "0.10",
        "preservation_loss_weight": "0.60",
        "token_anchor_loss_weight": "0.05",
    },
    "self_market": {
        "auxiliary_jsonl": "training_data/auxiliary.jsonl",
        "max_steps": 500,
        "seed": 2202,
        "learning_rate": "7e-5",
        "text_encoder_learning_rate": "2.5e-6",
        "token_learning_rate": "1.0e-5",
        "caption_dropout_prob": "0.08",
        "preservation_loss_weight": "0.65",
        "token_anchor_loss_weight": "0.05",
    },
    "balanced_people": {
        "auxiliary_jsonl": "experiment_data/balanced_people/auxiliary.jsonl",
        "max_steps": 600,
        "seed": 3303,
        "learning_rate": "6e-5",
        "text_encoder_learning_rate": "2e-6",
        "token_learning_rate": "8e-6",
        "caption_dropout_prob": "0.08",
        "preservation_loss_weight": "0.70",
        "token_anchor_loss_weight": "0.06",
    },
    "market_people_refinement": {
        "auxiliary_jsonl": "experiment_data/balanced_people/auxiliary.jsonl",
        "init_weights": "quality_out_clean/self_market/training/checkpoints/step_000150/pytorch_lora_weights.safetensors",
        "max_steps": 60,
        "seed": 4404,
        "learning_rate": "1.5e-5",
        "text_encoder_learning_rate": "5e-7",
        "token_learning_rate": "1e-6",
        "caption_dropout_prob": "0.03",
        "preservation_loss_weight": "0.75",
        "token_anchor_loss_weight": "0.10",
        "checkpointing_steps": 15,
        "lr_warmup_steps": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--stop_after_step", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_review", action="store_true")
    return parser.parse_args()


def run(command: list[str], log_path: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def previous_step(provenance_dir: Path) -> int:
    sessions = provenance_dir / "sessions.jsonl"
    if not sessions.exists():
        return 0
    with sessions.open("r", encoding="utf-8") as handle:
        return max(json.loads(line)["step"] for line in handle if line.strip())


def main() -> None:
    args = parse_args()
    if args.resume == args.overwrite:
        raise ValueError("Pass exactly one of --resume or --overwrite")
    config = CANDIDATES[args.candidate]
    max_steps = int(config["max_steps"])
    checkpointing_steps = int(config.get("checkpointing_steps", 25))
    if not 1 <= args.stop_after_step <= max_steps:
        raise ValueError(f"stop_after_step must be between 1 and {max_steps}")

    root = Path("quality_out_clean") / args.candidate
    output_dir = root / "current"
    provenance_dir = root / "training"
    start_step = previous_step(provenance_dir) if args.resume else 0
    if args.stop_after_step <= start_step:
        raise ValueError(f"Candidate is already at step {start_step}")

    run([sys.executable, "code/verify_clean_assets.py"])
    command = [
        sys.executable,
        "code/train_lora.py",
        "--data_dir", "style_imgs/512",
        "--captions_jsonl", "code/auto_captions/florence_captions.jsonl",
        "--instance_token", "<sks>",
        "--token_initializer", "ghibli style",
        "--output_dir", str(output_dir),
        "--provenance_dir", str(provenance_dir),
        "--rank", "16",
        "--text_encoder_rank", "4",
        "--learning_rate", config["learning_rate"],
        "--text_encoder_learning_rate", config["text_encoder_learning_rate"],
        "--token_learning_rate", config["token_learning_rate"],
        "--lora_dropout", "0.05",
        "--caption_dropout_prob", config["caption_dropout_prob"],
        "--snr_gamma", "5.0",
        "--preservation_loss_weight", config["preservation_loss_weight"],
        "--token_anchor_loss_weight", config["token_anchor_loss_weight"],
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(config.get("lr_warmup_steps", 50)),
        "--max_steps", str(max_steps),
        "--stop_after_step", str(args.stop_after_step),
        "--checkpointing_steps", str(checkpointing_steps),
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--random_flip",
        "--allow_tf32",
        "--seed", str(config["seed"]),
    ]
    if config["auxiliary_jsonl"] is not None:
        command.extend(["--auxiliary_jsonl", config["auxiliary_jsonl"]])
    if config.get("init_weights") is not None:
        command.extend(["--init_weights", config["init_weights"]])
    command.append("--resume" if args.resume else "--overwrite")
    log_path = root / "logs" / f"steps_{start_step + 1:06d}_{args.stop_after_step:06d}.log"
    run(command, log_path)

    if not args.skip_review:
        review_steps = list(
            range(
                ((start_step // checkpointing_steps) + 1) * checkpointing_steps,
                args.stop_after_step + 1,
                checkpointing_steps,
            )
        )
        review_dir = root / "reviews" / f"steps_{start_step + 1:06d}_{args.stop_after_step:06d}"
        run(
            [
                sys.executable,
                "code/eval_checkpoint_grid.py",
                "--checkpoints_dir", str(provenance_dir / "checkpoints"),
                "--outdir", str(review_dir),
                "--steps", ",".join(str(step) for step in review_steps),
                "--num_target_images", "3",
                "--target_seed", "91000",
                "--control_seed", "92000",
                "--num_inference_steps", "30",
                "--guidance_scale", "7.5",
            ]
        )


if __name__ == "__main__":
    main()
