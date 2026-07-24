# Supplied-Only Visual Reviews

## Steps 25-100

Reviewed `reviews/steps_000001_000100` at 512 px, 30 inference steps,
guidance 7.5, exact-prompt seeds 91000-91002, and control seeds 92000-92004.

- Step 25: all target images are readable, populated markets but remain close to the
  photographic or generic base-model rendering. The custom style is not established.
- Step 50: seed 91002 and the two-shopper control begin changing to a flat illustrated
  rendering. Seeds 91000-91001 remain mostly base-like. Market composition is retained.
- Step 75: all target seeds are illustrated and contain stalls, produce, and crowds.
  The rendering is bright and flat with limited facial detail. The family and portrait
  controls have visible faces; the two-shopper control has simplified profiles.
- Step 100: target markets remain clear and populated. Style consistency improves only
  slightly over step 75 and is still more poster-like than the supplied film frames.
  No scene collapse or token-control failure is visible.

Decision: resume the same fixed schedule to step 200. The market prior is intact and style
is still developing, so stopping at 100 would underfit the supplied visual distribution.

## Steps 125-200

Reviewed `reviews/steps_000101_000200` with the same prompts, seeds, and inference
settings.

- Step 125: the three target images are dense, readable markets. The vendor control is
  still photographic, while the two-shopper, family, and portrait controls are clearly
  illustrated. Faces are simple but recognizable.
- Step 150: market layout remains stable. The vendor control starts to adopt the target
  rendering, but the exact-prompt images gain little over step 125. The family control
  retains clean expressions and distinct people.
- Step 175: strongest supplied-only checkpoint in this block. The exact prompt keeps
  stalls, produce, depth, and crowds; the family control has the cleanest medium-size
  faces; and the style remains coherent without flattening the scene further.
- Step 200: still sharp and compliant, but the exact-prompt outputs are more uniformly
  poster-like. The vendor is fully illustrated, while facial and scene quality do not
  improve consistently over step 175.

Decision: retain step 175 as the current supplied-only leader and pause this candidate.
The exact-prompt images plateaued after step 150, so the next GPU block should test
explicit base-model market and people preservation rather than add more supplied-only
steps immediately.
