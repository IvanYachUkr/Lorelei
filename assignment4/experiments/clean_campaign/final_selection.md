# Clean Campaign Final Selection

## Eligible Models

| Candidate | Selected checkpoint | Visual result |
| --- | ---: | --- |
| A: supplied only | 175 | Strong supplied style and readable markets; flatter scene geometry and simpler crowd faces. |
| B: self-market | 150 | Best overall balance of dense market structure, style, people separation, color, and control-prompt faces. |
| C: balanced people | 150 | Cleanest close portrait; occasional elongated or unstable small faces, with no consistent exact-prompt gain over B. |
| D: B-to-people refinement | 60 | Preserves B, improves one of ten audit seeds, but loses ground on four and is otherwise tied. |

## Evidence

- Development review: fixed exact-prompt seeds 91000-91002 and control seeds
  92000-92004 at 512 px, DPM-Solver, 30 inference steps, guidance 7.5.
- Wider review: exact assignment prompt on seeds 84000-84009, with model columns shuffled
  before visual inspection.
- All eligible finalists produced a recognizable populated market on all ten wider seeds.
- B150 most consistently retained foreground/background separation and detailed stalls.
- C150's supplied-derived crops improved the dedicated portrait, but not small crowd
  faces consistently.
- In the direct B150/D60 blind audit, B150 was preferred on four seeds, D60 on one, with
  five ties.

## Selection

Select Candidate B step 150:

`lora_out/pytorch_lora_weights.safetensors`

Training loss was used only as a stability diagnostic. The selected checkpoint is not a
run endpoint and was chosen by the recorded visual comparisons.
