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

The archive contains every training image, caption, and auxiliary manifest used
for the selected model. Run from the archive root:

```bash
python code/train_lora.py \
  --data_dir style_imgs/512 \
  --captions_jsonl code/auto_captions/florence_captions.jsonl \
  --auxiliary_jsonl training_data/auxiliary.jsonl \
  --instance_token "<sks>" \
  --token_initializer "ghibli style" \
  --output_dir lora_out \
  --rank 16 \
  --text_encoder_rank 4 \
  --learning_rate 7e-5 \
  --text_encoder_learning_rate 2.5e-6 \
  --token_learning_rate 1e-5 \
  --lora_dropout 0.05 \
  --caption_dropout_prob 0.08 \
  --snr_gamma 5.0 \
  --preservation_loss_weight 0.65 \
  --token_anchor_loss_weight 0.05 \
  --max_steps 250 \
  --scheduler_steps 500 \
  --lr_warmup_steps 50 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --random_flip \
  --allow_tf32 \
  --seed 2202 \
  --overwrite
```

The trainer saves one file in `lora_out` containing the UNet LoRA,
text-encoder LoRA, and learned token embedding. Different CUDA hardware may
produce numerically different weights while reproducing the same procedure.

## Evaluation

```bash
python code/eval_lora.py \
  --weights lora_out/pytorch_lora_weights.safetensors \
  --prompt "a busy market, in <sks> style" \
  --outdir samples \
  --num_images 3 \
  --seed 84000 \
  --num_inference_steps 40 \
  --guidance_scale 7.5
```

The evaluation script restores `<sks>`, loads both LoRA branches, and renders
three 512x512 images. It does not expose an adapter or style-strength setting.
