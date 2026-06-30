# Deep Learning Project 4 - Ghibli Market LoRA

Team members: replace this line with your group members' names.

This project trains a Stable Diffusion 1.5 LoRA style adapter using the provided `style_imgs/512` image folder. The training script adds the `<sks>` style token, LoRA-tunes both the UNet and the text encoder, and saves one adapter file at `lora_out/pytorch_lora_weights.safetensors`.

## Setup

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Stable Diffusion 1.5 model requires authentication in your environment, log in first:

```bash
hf auth login
```

## Train

Run from the `assignment4` directory:

```bash
python code/train_lora.py ^
  --data_dir style_imgs/512 ^
  --instance_token "<sks>" ^
  --output_dir lora_out ^
  --rank 8 ^
  --learning_rate 1e-4 ^
  --max_steps 800 ^
  --overwrite
```

Equivalent one-line command:

```bash
python code/train_lora.py --data_dir style_imgs/512 --instance_token "<sks>" --output_dir lora_out --rank 8 --learning_rate 1e-4 --max_steps 800 --overwrite
```

The required output is:

```text
lora_out/pytorch_lora_weights.safetensors
```

## Evaluate

```bash
python code/eval_lora.py ^
  --weights lora_out/pytorch_lora_weights.safetensors ^
  --prompt "a busy market, in <sks> style" ^
  --outdir samples
```

This writes at least three adapter images:

```text
samples/adapter_00.png
samples/adapter_01.png
samples/adapter_02.png
```

## Expected Hardware And Runtime

Recommended: one NVIDIA GPU with at least 12 GB VRAM. With `train_batch_size=1`, `gradient_accumulation_steps=4`, rank 8, fp16, and 800 steps, expect roughly 30-90 minutes depending on GPU and disk speed. CPU training is not recommended.

If memory is tight, try:

```bash
python code/train_lora.py --data_dir style_imgs/512 --instance_token "<sks>" --output_dir lora_out --rank 4 --max_steps 800 --gradient_checkpointing --overwrite
```

## Deliverable Checklist

- `lora_out/pytorch_lora_weights.safetensors`
- `code/train_lora.py`
- `code/eval_lora.py`
- `samples/` with at least three adapter images
- `requirements.txt`
- `README.md`
- `report.pdf`, maximum two pages

After evaluation, create the report:

```bash
python code/make_report.py --team "Name 1, Name 2, Name 3"
```

Then package the final archive:

```bash
python code/package_submission.py --out assignment4_submission.zip
```

## Notes

- The LoRA file also stores the learned `<sks>` token embedding row. This keeps the assignment's single-file adapter requirement while making evaluation reproducible in a fresh process.
- The evaluation script intentionally has no style-strength slider.
- Default prompt template during training is `an animated movie scene, in <sks> style`.
