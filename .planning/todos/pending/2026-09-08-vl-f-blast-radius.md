---
created: 2026-09-08T17:30
title: Re-run every recorded spectral VL free energy after the Q-norm fix
area: inference / benchmarks / model-selection
priority: high
files:
  - src/pyro_dcm/inference/csd_precision.py
  - benchmarks/recovery_matrix_grid.py
  - src/pyro_dcm/model_selection/bmr.py
  - .planning/phases/30-recovery-matrix/30-RECOVERY-MATRIX-REPORT.md
---

## Problem

`d77ab53` fixed `compute_csd_precision` to use MATLAB's induced 1-norm (max
absolute column sum) rather than `torch.norm(Q, p=1)`'s entrywise sum. The
regulariser was 8.66x too large on the N=2 parity problem, so **Q changed, and
with it every spectral VL free energy**.

`compute_csd_precision` feeds both VL paths (`variational_laplace.py:564` and
`vl_forward_models.py:208`), so this is not confined to the parity harness.

## What is now stale

- **Phase 30 recovery matrix** F values (`benchmarks/results/recovery_matrix.json`,
  `30-RECOVERY-MATRIX-REPORT.md`). The pass / identifiability-limit
  classification is derived from metrics that include F.
- **Phase 23 / 31 BMR** results: every prune delta-F went through this Q.
- **Phase 32 parity numbers** wherever they are quoted outside the todo/STATE
  blocks already corrected.
- Any recorded `elbo_model_comparison` / model-evidence figure on the spectral
  path.

## What is probably fine, but is an expectation not a measurement

Parameter recovery (`Ep`, RMSE) and model **ranking**. Ranking agreement held at
1.0 through every regime of this bug -- `Y.Q` unset, the mar2csd units artifact,
and the over-regularised Q -- which is consistent with a smooth monotone
distortion of F that preserves ordering. **Verify rather than assume.**

## Possible interaction with the BMR overconfidence todo

`2026-06-09-vl-overconfidence-for-bmr` records posterior std/prior std ~0.001 at
high SNR, which suppressed absolute BMR pruning. An over-regularised Q changes
the effective observation precision, so it is worth checking whether that
overconfidence is partly the same bug rather than a separate calibration issue.
Do NOT assume it is -- measure before merging the two todos.

## Next step

1. Re-run the Phase 30 recovery-matrix sweep and diff F, RMSE and the
   pass/identifiability classification against the recorded matrix.
2. Re-run the Phase 23 BMR-vs-ELBO comparison.
3. Re-check `2026-06-09-vl-overconfidence-for-bmr` against the new Q.

All three are cluster jobs (see `cluster/sbatch/recovery_matrix_sweep.sbatch`,
`23_bmr_vs_elbo.sbatch`).
