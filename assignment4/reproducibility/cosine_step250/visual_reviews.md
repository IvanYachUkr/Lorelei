# Original Cosine Trajectory Visual Reviews

## Steps 150-250

Review sheet: `reviews/through_250/trajectory_sheet_01.jpg`

- Step 150 is the submitted baseline. Both target seeds are recognizable markets,
  but seed 84003 contains several merged or indistinct crowd figures.
- Step 175 increases separation between foreground people without changing the
  close-face rendering materially.
- Steps 200 and 225 progressively clarify the produce tables and crowd layers in
  seed 84003. Seed 84000 keeps its original composition while the central people
  become easier to distinguish.
- Step 250 is the strongest checkpoint in this sheet overall. It has the clearest
  target-seed crowd separation and stall organization. The close vendor face is
  effectively stable relative to step 150, and the two-shopper control remains
  coherent.
- No palette washout, scene collapse, or loss of the custom style is visible
  through step 250.

Decision: continue the exact trajectory to step 350. Step 250 is the provisional
long-horizon favorite, but it has not yet passed the ten-seed replacement gate.

## Steps 250-350

Review sheet: `reviews/steps_250_350/trajectory_sheet_01.jpg`

- The two market seeds remain coherent through step 350. Stalls, produce, and
  foreground/background crowd layers stay legible, with no visible collapse.
- Steps 275-350 mostly rearrange people and props rather than producing a
  consistent sharpness or anatomy improvement. Seed 84003 settles into a clear
  central figure, but the gain over step 250 is modest and seed-dependent.
- The close vendor face is effectively unchanged. Later checkpoints introduce a
  background person on the right without degrading the main face.
- The two-shopper control stays coherent, although steps 275-350 make the male
  shopper more frontal and reduce some of the profile interaction visible at
  step 250.
- The palette and style remain stable. Later states are slightly more regular and
  cartoon-flat, but not materially clearer.

Decision: continue to step 450 to inspect the low-learning-rate tail. Step 250
remains the provisional favorite because no later checkpoint has yet shown a
repeatable overall gain.

## Steps 350-450

Review sheet: `reviews/steps_350_450/trajectory_sheet_01.jpg`

- Both exact-prompt market seeds are visually stable across all five rows. Crowd
  count, stall layout, produce detail, and foreground figure placement change
  only minimally.
- The close vendor face does not become sharper or more anatomically detailed.
  Its expression and the background figure remain coherent, with only small line
  and color changes.
- The two-shopper control is effectively locked. Additional optimization neither
  improves the simple faces nor damages prompt adherence.
- No palette washout or collapse appears, but there is also no meaningful late
  refinement. The low-rate cosine tail is preserving the solution rather than
  improving it.

Decision: finish the contracted schedule at step 500, then compare the endpoint
with the earlier checkpoints. Step 250 remains the provisional visual winner of
this trajectory.

## Steps 450-500

Review sheet: `reviews/steps_450_500/trajectory_sheet_01.jpg`

- Steps 450, 475, and 500 are visually indistinguishable at practical review
  scale. They preserve the same people, stalls, produce layout, expressions, and
  palette in all four cases.
- Step 500 does not add facial detail, correct anatomy, sharpen the market, or
  improve prompt adherence relative to step 450.
- Mean absolute RGB differences from step 450 to step 500 are 3.1605 for market
  seed 84000, 4.8488 for market seed 84003, 1.7915 for the close face, and 2.6226
  for the two shoppers on a 0-255 scale. These small distributed changes do not
  produce a visible semantic improvement.

Decision: the original cosine trajectory has converged. Step 250 remains its
provisional visual winner; step 500 is retained as the reproducible endpoint but
does not qualify as a replacement candidate.
