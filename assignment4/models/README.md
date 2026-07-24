# Clean Comparison Models

The submitted model is `../lora_out/pytorch_lora_weights.safetensors`.

This directory contains eligible checkpoints retained to show the effect of the
training changes:

- `supplied_only_step175.safetensors`: 843 supplied images only. Strong style,
  flatter market structure, and simpler crowd faces.
- `balanced_people_step150.safetensors`: supplied images, original-SD market and
  people images, and an 8% share of face/person crops derived from supplied
  images. Close portraits improve, but small crowd faces do not improve
  consistently.
- `market_people_refinement_step60.safetensors`: short low-rate continuation of
  step 150 using the balanced data. It remains stable but loses a blind
  ten-seed comparison to the unrefined step-150 model.
- `self_market_step150.safetensors`: the reproducible self-market checkpoint
  used to initialize both long-horizon trajectories.
- `stable_cumulative_step225.safetensors`: lower-rate continuation with stronger
  preservation. It won 6 of 20 final blind comparisons.

All files contain UNet LoRA, text-encoder LoRA, and the learned `<sks>`
embedding. None was trained with assignment-PDF images or external datasets.
