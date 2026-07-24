#!/usr/bin/env python
"""Run one reviewed session of the long-horizon LoRA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "long_horizon_out"
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"

TRAJECTORIES = {
    "cosine_500": {
        "max_steps": 500,
        "seed": 2202,
        "learning_rate": "7e-5",
        "text_encoder_learning_rate": "2.5e-6",
        "token_learning_rate": "1e-5",
        "caption_dropout_prob": "0.08",
        "preservation_loss_weight": "0.65",
        "token_anchor_loss_weight": "0.05",
        "lr_warmup_steps": 50,
    },
    "stable_500": {
        "max_steps": 350,
        "seed": 5505,
        "learning_rate": "2e-5",
        "text_encoder_learning_rate": "7.5e-7",
        "token_learning_rate": "2.5e-6",
        "caption_dropout_prob": "0.08",
        "preservation_loss_weight": "0.80",
        "token_anchor_loss_weight": "0.12",
        "lr_warmup_steps": 25,
        "init_weights": ROOT / "models" / "self_market_step150.safetensors",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=sorted(TRAJECTORIES), required=True)
    parser.add_argument("--stop_after_step", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_last_step(path: Path) -> int:
    if not path.is_file():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return int(rows[-1]["step"]) if rows else 0


def bootstrap_cosine(root: Path) -> None:
    source = ROOT / "reproducibility" / "self_market_step150"
    verification = json.loads((source / "verification.json").read_text(encoding="utf-8"))
    if not verification.get("verified") or not verification.get("tensors_equal"):
        raise ValueError("The step-150 reproduction is not verified")
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(source / "training", root / "training")
    shutil.copytree(source / "reproduced_model", root / "current")
    record = {
        "source_adapter_sha256": sha256(source / "reproduced_model" / WEIGHTS_NAME),
        "source_run_config_sha256": sha256(source / "training" / "run_config.json"),
        "source_state_sha256": sha256(source / "training" / "latest_training_state.pt"),
        "source_step": 150,
        "source_verification_sha256": sha256(source / "verification.json"),
    }
    (root / "bootstrap.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
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


def main() -> None:
    args = parse_args()
    if args.resume == args.overwrite:
        raise ValueError("Pass exactly one of --resume or --overwrite")
    config = TRAJECTORIES[args.trajectory]
    max_steps = int(config["max_steps"])
    if not 1 <= args.stop_after_step <= max_steps:
        raise ValueError(f"stop_after_step must be between 1 and {max_steps}")

    root = OUTPUT_ROOT / args.trajectory
    if args.trajectory == "cosine_500" and args.overwrite:
        bootstrap_cosine(root)
    start_step = read_last_step(root / "training" / "sessions.jsonl")
    if args.stop_after_step <= start_step:
        raise ValueError(f"Trajectory is already at step {start_step}")

    run(
        [
            sys.executable,
            "code/verify_training_data.py",
            "--data_dir", "style_imgs/512",
            "--captions_jsonl", "code/auto_captions/florence_captions.jsonl",
            "--auxiliary_jsonl", "training_data/auxiliary.jsonl",
        ],
        root / "logs" / f"verify_before_{start_step + 1:06d}.log",
    )

    command = [
        sys.executable,
        "code/train_lora.py",
        "--data_dir", "style_imgs/512",
        "--captions_jsonl", "code/auto_captions/florence_captions.jsonl",
        "--auxiliary_jsonl", "training_data/auxiliary.jsonl",
        "--instance_token", "<sks>",
        "--token_initializer", "ghibli style",
        "--output_dir", str(root / "current"),
        "--provenance_dir", str(root / "training"),
        "--rank", "16",
        "--text_encoder_rank", "4",
        "--learning_rate", str(config["learning_rate"]),
        "--text_encoder_learning_rate", str(config["text_encoder_learning_rate"]),
        "--token_learning_rate", str(config["token_learning_rate"]),
        "--lora_dropout", "0.05",
        "--caption_dropout_prob", str(config["caption_dropout_prob"]),
        "--snr_gamma", "5.0",
        "--preservation_loss_weight", str(config["preservation_loss_weight"]),
        "--token_anchor_loss_weight", str(config["token_anchor_loss_weight"]),
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(config["lr_warmup_steps"]),
        "--max_steps", str(max_steps),
        "--stop_after_step", str(args.stop_after_step),
        "--checkpointing_steps", "25",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--random_flip",
        "--allow_tf32",
        "--seed", str(config["seed"]),
    ]
    if "init_weights" in config:
        command.extend(["--init_weights", str(config["init_weights"])])
    if args.trajectory == "cosine_500":
        command.append("--resume")
    else:
        command.append("--resume" if args.resume else "--overwrite")
    run(
        command,
        root / "logs" / f"steps_{start_step + 1:06d}_{args.stop_after_step:06d}.log",
    )


if __name__ == "__main__":
    main()
