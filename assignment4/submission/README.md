# Stable Diffusion 1.5 Style LoRA

Team members: Ivan Iachnyk, Claudius Kühn, Robin Sternberg, Arham Shahzad,
Clemens Rosskopf.

The project adds `<sks>` as a style token and trains LoRA adapters for both the
Stable Diffusion 1.5 UNet and text encoder. The final checkpoint is
`lora_out/pytorch_lora_weights.safetensors`.

## Installation

Python 3.10 or newer and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The measured training environment used Python 3.12, PyTorch 2.11, CUDA 12.8,
and an RTX 4070 Laptop GPU with approximately 8 GB of VRAM. The selected
250-step run took about 55 minutes. Stable Diffusion 1.5 is downloaded on first
use.

## Training

Place the supplied `style_imgs` directory beside `code`, then run:

```bash
python code/train_lora.py \
  --data_dir style_imgs/512 \
  --instance_token "<sks>" \
  --output_dir lora_out \
  --rank 16 \
  --max_steps 250 \
  --overwrite
```

The trainer accepts optional `--captions_jsonl` and `--auxiliary_jsonl` inputs.
It saves one file in `lora_out` containing the UNet LoRA, text-encoder LoRA,
and learned token embedding.

## Evaluation

```bash
python code/eval_lora.py \
  --weights lora_out/pytorch_lora_weights.safetensors \
  --prompt "a busy market, in <sks> style" \
  --outdir samples \
  --num_images 3
```

The evaluation script restores `<sks>`, loads both LoRA branches, and renders
three 512x512 images. It does not expose an adapter or style-strength setting.
