---
created: 2026-09-08T17:30
title: Re-run every recorded spectral VL free energy after the Q-norm fix
area: inference / benchmarks / model-selection
priority: done
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

## CORRECTION 2026-09-08: the blast radius is SPECTRAL ONLY

The first draft of this todo said "every spectral VL free energy" and listed BMR
as affected. Checked instead of assumed: **only `SpectralDCMForward` calls
`compute_csd_precision`** (`vl_forward_models.py:208`). `TaskDCMForward`,
`LatentCircuitForward` and `ERPDCMForward` all build an identity precision
(`torch.eye(ny)`) and never touch `compute_csd_precision`.

### Affected

- **The 4 spectral cells** of the Phase 30 recovery matrix. The grid is
  `spectral 2N x 2SNR = 4` + `task 2N x 2SNR = 4` + `latent_circuit 1N x 2SNR = 2`
  (`recovery_matrix_grid.py:121` and its module docstring).
- Phase 32 SPM parity numbers (already re-measured: F gate now 0.0037 PASS).
- Any recorded spectral ELBO / model-evidence figure.

### NOT affected -- do not re-run

- **Phase 23 / 31 BMR.** `lc_vl_bmr_selection.py` uses `LatentCircuitForward`;
  `bmr_tempering_calibration.py` targets the **task**-N4 cell via
  `TaskDCMForward`. Both are identity-precision paths.
- The 4 task and 2 latent cells of the recovery matrix.
- **`2026-06-09-vl-overconfidence-for-bmr`** -- that was measured on the latent
  circuit, which uses identity precision. It is a **separate** bug, not a
  downstream symptom of this one. The earlier speculation that the two might be
  the same is now ruled out.

### Free correctness check on the re-run

Sweep job **55234625** re-runs all 10 cells. The 6 task/latent cells should come
back **numerically unchanged** against the baseline committed at `df1e0ce`; only
the 4 spectral cells should move. If a task or latent cell moves, something other
than the Q norm changed and needs investigating before the new matrix is trusted.

## Next step

1. Sweep **55234625** is running. When it lands, diff F, RMSE and the
   pass/identifiability classification against `df1e0ce`, and check the
   task/latent-unchanged prediction above.
2. Nothing else needs re-running. Items 2 and 3 of the original draft (BMR
   re-run, VL-overconfidence re-check) are withdrawn -- both are
   identity-precision paths.


---

## ✅ CLOSED 2026-09-08: re-run complete, prediction held exactly

Sweep job 55234625, all 10 cells present (0-9, full coverage verified against
the per-cell JSONs rather than assumed from exit codes).

**The task/latent-unchanged prediction held exactly.** Only the 4 spectral cells
moved; the 4 task and 2 latent cells are **numerically identical** to the
baseline at `df1e0ce`. That is an independent confirmation of the
spectral-only blast-radius correction: `compute_csd_precision` is called by
`SpectralDCMForward` alone, and the identity-precision forwards were untouched.

| cell | rmse_a_median | shrinkage_median |
|---|---|---|
| spectral N=2 snr=1 | +0.000341 | +0.025315 |
| spectral N=2 snr=3 | -0.001886 | +0.026423 |
| spectral N=4 snr=1 | +0.000117 | +0.001844 |
| spectral N=4 snr=3 | -0.000340 | +0.002984 |

`sign_recovery_masked`, `coverage_95` and `convergence_rate` are unchanged (all
1.0), and **no cell changed status**. Summary counts are identical: 10 cells,
6 pass, 4 identifiability-limit, 0 errored.

### What this settles

- **Parameter recovery is robust to the Q fix.** RMSE moves are tiny and
  bidirectional -- noise-level, not a systematic shift. The expectation
  recorded earlier ("recovery and ranking should be far less affected") is now
  measured, not assumed.
- **No re-classification.** The pass / identifiability-limit verdicts stand, so
  the Phase 30 conclusions do not need revisiting.
- **Posteriors got slightly wider** (shrinkage up ~12% relative at N=2, ~6-10%
  at N=4). Consistent with removing an over-regularised Q: the observation
  precision was distorted, and correcting it changed the posterior width. This
  is a *spectral* effect and does NOT touch
  `2026-06-09-vl-overconfidence-for-bmr`, which was measured on the latent
  circuit (identity precision) and whose cells came back bit-identical here.

### Gotcha for the next person

`recovery_matrix_aggregate.py` globs `cluster/results/recovery_matrix_*.json`
by default and will silently mix historical runs -- the first attempt harvested
**30 cells** across three separate sweeps. Scope it with
`RECOVERY_RESULTS_GLOB` (here: `'cluster/results/recovery_matrix_552346*_*.json'`).
Note also that array tasks each get their OWN job id, so per-cell files are
named `recovery_matrix_<task_jobid>_<cell>.json`, not `<arrayjobid>_<cell>` --
a `ls` on the array job id finds almost nothing and looks like a silent drop.
