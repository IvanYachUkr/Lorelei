# Balanced-People Visual Reviews

## Steps 25-100

Reviewed `reviews/steps_000001_000100` at 512 px, 30 inference steps,
guidance 7.5, exact-prompt seeds 91000-91002, and control seeds 92000-92004.

- Step 25: target and medium-shot controls remain photographic. The portrait already
  responds to the token but is too flat to judge face retention.
- Step 50: outputs remain close to the base model. The portrait has stable eye and mouth
  placement, but the learned style is not established.
- Step 75: target seed 91002 and some people controls become illustrated. Seed 91000 and
  the vendor remain photographic. The family control begins to lose facial definition.
- Step 100: all target seeds are moving into the learned rendering, with seed 91000 now
  strongly illustrated. The portrait and two-shopper controls have clean simple faces,
  while the family control is less coherent than the self-market candidate.

Decision: resume to step 200. This candidate uses a lower learning rate and a 600-step
schedule, and its style transition occurs later than the other candidates. Stopping at
100 would not fairly test the supplied-derived crop strategy.

## Steps 125-200

Reviewed `reviews/steps_000101_000200` with the same prompts, seeds, and inference
settings.

- Step 125: all target seeds are illustrated and detailed, but seed 91001 contains a
  large malformed sign. The family and portrait controls have clean, readable faces.
- Step 150: strongest checkpoint in this candidate. Target markets are vivid, populated,
  and structurally clear. The portrait has the cleanest eyes and mouth among the three
  clean candidates so far, and the family retains distinct expressions.
- Step 175: target seed 91002 becomes less busy and the vendor control remains
  photographic. The face controls stay recognizable but do not improve over step 150.
- Step 200: the vendor finally becomes illustrated and target scenes remain correct, but
  overall contrast drops and crowd faces do not gain useful detail.

Decision: retain step 150 as the balanced-people leader. The supplied-derived crops help
medium and close faces at a low sampling share, but extending beyond 150 does not improve
the combined exact-prompt and control-image result.
