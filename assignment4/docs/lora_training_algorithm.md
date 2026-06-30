# LoRA Style Training Algorithm

This document is the implementation contract for `code/train_lora.py` and `code/eval_lora.py`.

## Goal

Train a single-file LoRA adapter for Stable Diffusion 1.5 that makes prompts containing the style token `<sks>` produce images in the provided Ghibli-like style. The training must adapt both the UNet and the text encoder with LoRA and must save one file:

`lora_out/pytorch_lora_weights.safetensors`

## Inputs

- `data_dir`: directory of RGB images, expected to be `style_imgs/512`.
- `instance_token`: tokenizer token to add and train, expected to be `<sks>`.
- `model_name`: Stable Diffusion 1.5 checkpoint, default `runwayml/stable-diffusion-v1-5`.
- Training hyperparameters: resolution, LoRA rank, learning rate, batch size, gradient accumulation, max steps, random seed, mixed precision, scheduler, and prompt template.
- Evaluation inputs: trained safetensors file, prompt, output directory, sample count, random seed, inference steps, and guidance scale.

## Outputs

- Training writes exactly one adapter artifact in the selected output directory:
  - `pytorch_lora_weights.safetensors`
- Evaluation writes at least three generated PNG samples to the selected samples directory.

## Assumptions

- The dataset images are style references, not paired captions.
- The 512-pixel images are already suitable for training. The script still resizes and crops defensively.
- The tokenizer does not already contain `instance_token` as a single token.
- A CUDA GPU is available for practical training. CPU mode exists only for smoke testing or debugging.
- The Hugging Face account running the scripts can download the base Stable Diffusion 1.5 model.

## Training Steps

1. Validate arguments, output directory policy, and image files.
2. Load the base tokenizer, text encoder, VAE, UNet, and DDPM noise scheduler.
3. Add `instance_token` to the tokenizer and resize the text encoder input embeddings.
4. Initialize the new token embedding from the mean embedding of a configurable initializer phrase, default `style`.
5. Freeze all base model parameters.
6. Attach LoRA adapters to:
   - UNet attention projections: `to_q`, `to_k`, `to_v`, `to_out.0`
   - CLIP text encoder attention projections: `q_proj`, `k_proj`, `v_proj`, `out_proj`
7. Train only the LoRA parameters plus the new token embedding row. A gradient hook masks all other token embedding rows.
8. For each batch:
   - Load and normalize images to `[-1, 1]`.
   - Encode images to latent space with the frozen VAE.
   - Sample noise and diffusion timesteps.
   - Encode prompts containing `instance_token` with the LoRA-adapted text encoder.
   - Predict noise with the LoRA-adapted UNet.
   - Optimize MSE against the scheduler target.
9. At the end, collect UNet and text encoder LoRA state dicts in Diffusers format.
10. Save the LoRA weights to `pytorch_lora_weights.safetensors`.
11. Append the learned token embedding row and metadata to that same safetensors file so evaluation can restore the custom token without a second artifact.

## Evaluation Steps

1. Load the base Stable Diffusion 1.5 pipeline.
2. Read the safetensors metadata and custom token embedding.
3. Add `instance_token` to the tokenizer, resize the text encoder embeddings, and restore the learned token embedding row.
4. Load the LoRA adapter from the same safetensors file.
5. Generate `num_images` samples for the prompt, default `a busy market, in <sks> style`.
6. Save numbered PNG files in the output directory.

## Edge Cases

- Empty dataset: fail before loading large models.
- Existing non-empty output directory without `--overwrite`: fail to avoid mixing artifacts.
- Token already present: reuse it only if it tokenizes as a single token.
- Missing custom token embedding at evaluation: warn and continue after adding the token, but generated quality may be poor.
- Safetensors file contains the custom embedding key: evaluation filters that key into a temporary LoRA-only file before calling the Diffusers loader.
- No CUDA device: fail by default; allow with `--allow_cpu`.

## Expected Intermediate States

- After tokenizer setup, `instance_token` maps to exactly one token id.
- After adapter setup, trainable parameters are LoRA weights plus the text encoder input embedding matrix with gradients masked to the custom token row.
- During training, saved output directory remains empty until the final save.
- After final save, the output directory contains only `pytorch_lora_weights.safetensors`.
