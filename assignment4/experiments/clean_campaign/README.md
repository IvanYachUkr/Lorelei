# Clean Campaign Records

This directory contains the records used to compare the four permitted-data
training trajectories. Each candidate directory includes its immutable run
configuration, full draw schedule, realized microbatch trace, optimizer-step
metrics, session boundaries, and written checkpoint review.

The corresponding adapters are:

| Candidate | Checkpoint | File | SHA-256 |
| --- | ---: | --- | --- |
| Supplied only | 175 | `../../models/supplied_only_step175.safetensors` | `6566f4a54856c77c8d5be4ac192c1e70dc375c03cb174322b0d224c318400e01` |
| Self-market | 150 | `../../lora_out/pytorch_lora_weights.safetensors` | `606190249fe5df2d4c36bb48552cb9837bfcc91e8388ef85abe25688ad71c788` |
| Balanced people | 150 | `../../models/balanced_people_step150.safetensors` | `3fd08fe204936604e40f5d9cdb013c781898d2d5449c109f0a1dc3e696059b28` |
| Market-to-people refinement | 60 | `../../models/market_people_refinement_step60.safetensors` | `889fa40ebad4ff5a70f840526ef4f2dc79ad59e3ff4183ebbabe58e47a5499f5` |

`final_selection.md` records the ten-seed comparison and final decision.
`../../docs/clean_training_campaign.md` defines the shared protocol and each
candidate recipe. The portable auxiliary inputs are in `../../training_data`
and `../../experiment_data`.
