# Selected Run Record

This directory records a full reproduction of the submitted step-150 adapter
from the self-contained `training_data` directory.

The registered trajectory has 500 maximum optimizer steps so the complete draw
schedule and cosine learning-rate curve are fixed. Training stops at step 150,
which was selected by checkpoint image review.

Run from `assignment4`:

```bat
python code\train_lora.py ^
  --data_dir style_imgs\512 ^
  --captions_jsonl code\auto_captions\florence_captions.jsonl ^
  --auxiliary_jsonl training_data\auxiliary.jsonl ^
  --instance_token "<sks>" ^
  --token_initializer "ghibli style" ^
  --output_dir reproducibility\self_market_step150\reproduced_model ^
  --provenance_dir reproducibility\self_market_step150\training ^
  --rank 16 ^
  --text_encoder_rank 4 ^
  --learning_rate 7e-5 ^
  --text_encoder_learning_rate 2.5e-6 ^
  --token_learning_rate 1e-5 ^
  --lora_dropout 0.05 ^
  --caption_dropout_prob 0.08 ^
  --snr_gamma 5.0 ^
  --preservation_loss_weight 0.65 ^
  --token_anchor_loss_weight 0.05 ^
  --lr_scheduler cosine ^
  --lr_warmup_steps 50 ^
  --max_steps 500 ^
  --stop_after_step 150 ^
  --checkpointing_steps 25 ^
  --gradient_accumulation_steps 4 ^
  --gradient_checkpointing ^
  --random_flip ^
  --allow_tf32 ^
  --seed 2202 ^
  --overwrite
```

`training` contains the original command and environment, the complete
2,000-draw schedule, the 600 realized microbatch traces through step 150, 150
optimizer metrics, and the session boundary. `verification.json` records the
selected and reproduced adapter hashes and the all-tensor comparison.

The same software and RTX 4070 Laptop GPU reproduce the submitted tensors
exactly. Other CUDA hardware may differ in floating-point details.
