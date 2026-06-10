---
phase: 29-vl-validation-infra-bmr-rank
plan: 02
subsystem: model-selection
tags: [bmr, variational-laplace, model-reduction, overconfidence, pytorch]

# Dependency graph
requires:
  - phase: 23-bmr
    provides: bayesian_model_reduction, make_reduced_prior_zero_connection, enumerate_reduced_models
  - phase: 28-variational-laplace-engine
    provides: VL posterior (mean + full covariance) that feeds BMR ranking
provides:
  - rank_connections() relative single-prune BMR ranking with separation-gap statistic
  - temper_vl_posterior() temperature-scaled covariance with Cholesky PD guard
  - re-export of both helpers from pyro_dcm.model_selection
  - vl-marked unit tests on a known ground-truth circuit
affects: [phase-30-recovery-sweep, phase-31-tempering-calibration, bmr-circuit-selection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Relative BMR ranking (K single-prune calls) instead of absolute delta-F thresholding"
    - "Separation-gap statistic on sorted prune costs to cut essential from non-essential edges"
    - "Cholesky PD guard that raises loudly (ValueError with shape + factor), never warn-and-continue"

key-files:
  created:
    - tests/test_bmr_rank_connections.py
  modified:
    - src/pyro_dcm/model_selection/bmr.py
    - src/pyro_dcm/model_selection/__init__.py

key-decisions:
  - "rank_connections is purely relative: absolute delta-F is never a pass/fail criterion (VL Laplace overconfidence, job 55772525)"
  - "separation_gap defined as the largest consecutive drop in essentiality on the sorted (ascending) prune costs"
  - "temper_vl_posterior is a primitive only; the calibrated factor is deferred to Phase 31"

patterns-established:
  - "Pattern: relative BMR ranking + separation gap is the only valid VL-BMR mode at high SNR"
  - "Pattern: PD guards raise ValueError with expected-vs-actual context (shape + tempering_factor)"

# Metrics
duration: 19min
completed: 2026-06-10
---

# Phase 29 Plan 02: Relative-Ranking BMR + PD-Safe Posterior Tempering Summary

**`rank_connections()` ranks edges by single-prune cost (K BMR calls, never absolute delta-F) with a separation-gap cut, and `temper_vl_posterior()` scales the VL covariance by a temperature behind a loud Cholesky PD guard.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-06-10T16:26:17Z
- **Completed:** 2026-06-10T16:45:59Z
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments

- `rank_connections()` added to `bmr.py`: runs exactly K single-connection BMR prune
  calls (reusing `make_reduced_prior_zero_connection` + `bayesian_model_reduction`),
  sorts by prune cost most-essential-first, and returns per-rank gaps plus a
  `separation_gap` / `separation_after_rank` cut. It is RELATIVE ranking only — the
  docstring documents (Notes) that absolute delta-F is never a pass/fail criterion
  because VL Laplace overconfidence drives every reduction deeply negative (cites
  cluster job 55772525 and REF-070).
- `temper_vl_posterior()` added to `bmr.py`: scales the posterior covariance by a
  positive temperature, symmetrizes, and asserts positive-definiteness via Cholesky,
  raising a loud `ValueError` (message includes matrix shape + tempering_factor) on
  failure. Calibration explicitly deferred to Phase 31; default factor `1.0` is identity.
- Both helpers re-exported from `pyro_dcm.model_selection` and covered by 5 fast,
  deterministic `vl`-marked unit tests on a known D=4 circuit (present edges {0,1} vs
  absent {2,3}), confirming present > absent ranking, a positive separation gap cutting
  after rank 2, the empty-indices guard, PD-preserving inflation, and the loud non-PD raise.

## Task Commits

1. **Task 1: Implement rank_connections()** - `6cbb349` (feat)
2. **Task 2: Implement temper_vl_posterior() with Cholesky PD guard** - `fb7aea7` (feat)
3. **Task 3: Re-export helpers + known-circuit unit tests** - `03fe8f4` (test)

## Files Created/Modified

- `src/pyro_dcm/model_selection/bmr.py` - Added `rank_connections()` (+ private
  `_single_prune_costs` helper) and `temper_vl_posterior()`; extended `__all__`.
- `src/pyro_dcm/model_selection/__init__.py` - Re-export both helpers (import + `__all__`).
- `tests/test_bmr_rank_connections.py` - 5 `vl`-marked unit tests on a known circuit.

## Decisions Made

- **rank_connections is purely relative.** Absolute delta-F is structurally broken by VL
  Laplace overconfidence (posterior std ~0.001–0.01x prior std at high SNR, job 55772525:
  a truly-absent edge scored delta_F = -115.9, indistinguishable by sign). Only relative
  ordering + the separation gap are reported. This avoids pitfall C1 by construction.
- **Separation gap = largest consecutive drop in essentiality** on the sorted (ascending)
  prune costs, since absolute delta-F values span orders of magnitude under overconfidence.
- **temper_vl_posterior is a primitive, not calibrated here.** It is just a temperature
  scale + symmetrize + Cholesky guard; the calibrated factor against Phase 30 coverage
  curves is Phase 31's job. Default `1.0` is backwards-compatible identity.

## Deviations from Plan

None — plan executed exactly as written. (A linter auto-sorted the test import block and
removed one redundant blank line; cosmetic only.)

## Issues Encountered

- **mypy reports `dict` type-arg and `torch.linalg.LinAlgError` attr-defined warnings on
  bmr.py.** Verified these are PRE-EXISTING: the original (pre-plan) file already emitted 5
  such warnings, and the new functions deliberately follow the module's established `-> dict`
  return-type convention used by `bmr_circuit_selection` / `enumerate_reduced_models`, and the
  same `except torch.linalg.LinAlgError` pattern already in `bayesian_model_reduction`. No NEW
  class of error was introduced — only additional instances of the repo's accepted style. Ruff
  (the linter/formatter the plan checks) is fully clean on all touched files.

## Next Phase Readiness

- VLINFRA-03 (`rank_connections`) and VLINFRA-04 (`temper_vl_posterior`) both delivered and
  unit-tested. Ready for Phase 30 (recovery sweep) to consume `rank_connections` as the BMR
  scoring mode, and for Phase 31 to calibrate `temper_vl_posterior`'s factor against Phase 30
  coverage curves.
- No blockers introduced. (Pre-existing INFRA item: Mutagen `models/` ignore must be fixed
  before any Phase 30 M3 run touching `src/pyro_dcm/models/` — unchanged by this plan.)

---
*Phase: 29-vl-validation-infra-bmr-rank*
*Completed: 2026-06-10*
