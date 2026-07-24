# Market-to-People Refinement Visual Reviews

## Steps 15-30

Reviewed `reviews/steps_000001_000030` at 512 px, 30 inference steps,
guidance 7.5, exact-prompt seeds 91000-91002, and control seeds 92000-92004.

- Step 15: preserves the B150 market composition, color, and crowd density. The family
  and portrait faces remain clean, but no consistent improvement is visible yet.
- Step 30: nearly identical to step 15. The vendor, two-shopper, and family controls stay
  coherent; exact-prompt scenes retain detail and do not wash out.

Decision: resume to step 60. The refinement is controlled and non-destructive, but the
first half does not yet produce a meaningful visual gain over its B150 initialization.

## Steps 45-60

Reviewed `reviews/steps_000031_000060` with the same prompts, seeds, and inference
settings.

- Step 45: exact-prompt composition, crowd density, and palette remain effectively
  unchanged from step 30. Face controls are stable but not visibly better.
- Step 60: final output is still clean and non-destructive. The family, portrait, and two
  shoppers remain readable, but differences from the B150 initialization are cosmetic.

Decision: do not promote the refinement on the development grid. Retain B150 as the
leader unless a wider same-seed audit shows a repeatable advantage for D60.

## Ten-Seed Audit

Blindly reviewed `quality_out_clean/refinement_audit_10seeds` for seeds 84000-84009.
B150 was preferred on seeds 84000, 84006, 84007, and 84009; D60 was preferred on 84003;
the remaining five were effectively tied. D60 improves person separation on 84003 but
degrades the already malformed market sign on 84007 from `BUSY` toward `BURY`.

Final decision: the refinement is a useful non-destructive contrast experiment, but it
does not beat B150 consistently enough to become the final adapter.
