# Long-Horizon Specification Audit

| Contract item | Result | Evidence |
| --- | --- | --- |
| Reuse only the selected data | Pass | Run config and schedule use 843 supplied images and 60 unmodified SD 1.5 images. |
| Resume the exact step-150 state | Pass | All 353 tensors, metrics, traces, and the full schedule match the reference at steps 175 and 200. |
| Continue the original cosine path to 500 | Pass | Metrics contain 500 steps; checkpoints and reviews cover cumulative steps 175-500 every 25 steps. |
| Train a stabilized continuation to cumulative 500 | Pass | The second trajectory ran 350 new steps from step 150 with the specified lower rates and preservation weights. |
| Save immutable checkpoints and resumable state | Pass | Both trajectories saved at 25-step boundaries. A step-432 OOM resumed from step 425 without changing the schedule. |
| Review pixels during training | Pass | Written reviews were recorded at the planned 250, 350, 450, and 500 cumulative boundaries and checkpoint grids covered every 25 steps. |
| Select by visual quality, not loss | Pass | Three finalists were compared blind on 20 fixed seeds spanning the exact prompt, close faces, and two people. |
| Preserve market reliability | Pass | Every finalist produced a populated market on all ten exact-prompt seeds. |
| Improve without a face/control regression | Pass | Cosine step 250 won 6/10 markets, 3/5 close faces, and 2/5 two-person scenes. |
| Keep one compliant final artifact | Pass | `lora_out/pytorch_lora_weights.safetensors` contains 353 UNet, text-encoder, and custom-token tensors. |

The original cosine trajectory improved through cumulative step 250 and then
plateaued. The stabilized trajectory peaked at cumulative step 225. The final
blind totals were 11 wins for cosine step 250, 6 for stable step 225, and 3 for
the old step-150 model, so cosine step 250 passed the replacement gate.
