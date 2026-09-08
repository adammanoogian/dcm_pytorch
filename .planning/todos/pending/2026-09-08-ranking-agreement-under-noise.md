---
created: 2026-09-08T19:20
title: Ranking agreement falls below the 0.80 gate under CSD sampling noise
area: validation / SPM parity
priority: medium
files:
  - cluster/scripts/spm_xval_multiseed.py
  - validation/run_vl_validation.py
  - cluster/results/spm_xval_multiseed_55234748.json
---

## Problem

With genuine finite-sample CSD realisations (job 55234748, seeds 42-46), the
VL-vs-SPM cross-model ranking agreement is **1.0, 1.0, 0.667, 1.0, 1.0** --
seed 44 falls to 2/3 pairs, below the `>=0.80` gate.

Every previous run reported 1.0 on every seed. That was an artifact: the sweep
refitted the identical problem, so of course the ranking never moved. This is
the first time ranking has been tested against varying data at all.

## Why this matters more than the number suggests

Model ranking is the claim the project has leaned on hardest. The memory entry
and several phase closures use wording along the lines of "ranking agreement is
1.0, so ranking is the SPM-comparable claim even when absolute F is not". That
statement rested on a degenerate sweep and needs re-grounding.

Note the F gate itself replicates fine (all 5 seeds within 5%), so this is not
a regression -- it is the first honest measurement.

## What is not yet known

- Is seed 44 a **close call** (two models with nearly equal evidence, where
  either ordering is defensible) or a **real disagreement** on a well-separated
  pair? The per-pair delta-F magnitudes decide this and are in the JSON's
  `pairwise_results`, not yet inspected -- eduVPN dropped mid-analysis.
- Does agreement improve with more averages (`XVAL_CSD_AVERAGES` > 64), i.e. is
  this purely a noise-level effect that vanishes as the CSD estimate tightens?
- Is 5 seeds enough to characterise the rate at all? Almost certainly not.

## Next step

1. Pull `pairwise_results` for seed 44 and report the delta-F separation of the
   disagreeing pair alongside the disagreement.
2. Re-run over more seeds (20+) and at two or three `XVAL_CSD_AVERAGES` levels,
   and report agreement as a **rate with a separation threshold**, not a single
   number. A disagreement on a pair separated by < 3 nats is not the same
   finding as one on a well-separated pair.
3. Only then decide whether the `>=0.80` gate is the right shape.
