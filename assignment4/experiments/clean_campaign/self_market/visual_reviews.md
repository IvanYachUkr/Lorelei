# Self-Market Visual Reviews

## Steps 25-100

Reviewed `reviews/steps_000001_000100` at 512 px, 30 inference steps,
guidance 7.5, exact-prompt seeds 91000-91002, and control seeds 92000-92004.

- Step 25: two target seeds and all medium-shot people controls remain photographic.
  Seed 91002 and the portrait control show that the custom token is beginning to alter
  rendering, but the style is not established.
- Step 50: seed 91002 becomes a clean illustrated market while seeds 91000-91001 remain
  base-like. Market structure and produce detail remain strong.
- Step 75: all exact-prompt seeds are illustrated and retain readable stalls, produce,
  crowds, and spatial depth. The two-shopper control is clean but flat; the family control
  has weaker facial definition.
- Step 100: all target seeds remain dense and legible. The family control improves to
  distinct, readable faces. The vendor control is still photographic, and the close
  portrait contains line and teeth artifacts.

Decision: resume to step 200. Explicit market preservation improves scene structure and
does not delay the target prompt beyond step 75, while the people controls show that the
style and face rendering are still developing.

## Steps 125-200

Reviewed `reviews/steps_000101_000200` with the same prompts, seeds, and inference
settings.

- Step 125: target markets are sharp and crowded, with strong produce and stall detail.
  The vendor control is partly stylized, the two shoppers are clean, and the family
  control unexpectedly shifts to grayscale.
- Step 150: strongest checkpoint in this candidate. All target seeds are vivid and
  structurally clear. The vendor is fully illustrated, and the family control has
  distinct people with readable expressions and consistent color.
- Step 175: target markets remain legible, but seed 91000 loses contrast. The family and
  portrait controls are stable; the two-shopper faces become slightly simpler.
- Step 200: no collapse, but seed 91000 is still washed out and the other target seeds do
  not gain enough detail to improve on step 150. Face controls are broadly unchanged.

Decision: retain step 150 as the self-market leader and pause this candidate. Later
training increases style strength but no longer improves the combined market, people,
and face criterion.
