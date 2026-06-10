---
phase: 30-recovery-matrix-sweep
plan: 01
subsystem: testing
tags: [recovery-metrics, variational-laplace, r-squared, sign-recovery, eig-clamp, snr, vlrec]

# Dependency graph
requires:
  - phase: 29-vl-validation-infra-bmr-rank
    provides: "run_spectral_vl / run_task_vl / run_latent_circuit_vl runner result dicts; vl pytest marker"
  - phase: 20-latent-circuit-forward-model
    provides: "compute_trajectory_r_squared(pooled=False) per-region R-squared"
  - phase: 25-hybrid-vae-dcm
    provides: "masked_sign_recovery (|true|>threshold) guarding the sign(0) artifact"
provides:
  - "assemble_cell_metrics(): one VL runner result -> hardened per-cell metric block"
  - "compute_shrinkage_ratio(): std_post/std_prior with scalar per-model prior"
  - "exclude_near_boundary_A() + resample_A_until_accepted(): reject max-Re-eig in [-0.05,0]"
  - "snr_for_model(): per-model SNR injection (task/latent SNR kwarg vs spectral noise log-amplitude)"
affects: [30-02-sweep-driver, 30-03-harvest-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-cell metric assembler consumes runner result dicts; never re-derives fits"
    - "Driver supplies per-seed r2_per_region_list / shrinkage_list / coverage_list; assembler median-aggregates (None+note when absent, never fabricates)"
    - "Named near-boundary band constants (NEAR_BOUNDARY_LO/HI) for the eig_clamp injective regime"

key-files:
  created:
    - benchmarks/recovery_matrix_metrics.py
    - tests/test_recovery_matrix_metrics.py
  modified: []

key-decisions:
  - "Per-region R-squared consumed from a driver-supplied r2_per_region_list (pooled=False upstream); assembler never re-pools (guards R1)"
  - "Masked sign recovery reuses pyro_dcm.models.hybrid_vae_dcm.masked_sign_recovery on reshaped (N,N) A matrices (guards R2/sign(0))"
  - "Near-boundary exclusion band [-0.05,0] is inclusive-rejected, exposed as NEAR_BOUNDARY_LO/HI constants"
  - "Spectral SNR maps to noise_log_amplitude=-log(snr); task/latent map to {'SNR': snr} -- the one place SNR semantics diverge"
  - "type: ignore[import-untyped] on the masked_sign_recovery import (pyro_dcm ships no py.typed); pyproject mypy config left untouched"

# Metrics
duration: 11min
completed: 2026-06-10
---

# Phase 30 Plan 01: Hardened Per-Cell Recovery-Metric Assembler Summary

**Per-cell VL recovery-metric assembler with per-region (non-pooled) R-squared, masked sign recovery, 95% coverage, RMSE, std_post/std_prior shrinkage, plus the [-0.05,0] near-boundary-A exclusion band and per-model SNR injection -- all laptop-tested, ruff/mypy clean.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-10T20:41:43Z
- **Completed:** 2026-06-10T20:53:19Z
- **Tasks:** 3
- **Files modified:** 2 (both new)

## Accomplishments

- `assemble_cell_metrics()` turns ONE `run_spectral_vl` / `run_task_vl` / `run_latent_circuit_vl` result dict into a flat, JSON-serializable per-cell block: `rmse_a` (median/IQR via `compute_summary_stats`), `coverage_95` (median of `coverage_list`), `sign_recovery_masked` (median of per-seed `masked_sign_recovery` on reshaped A), `r2_per_region` (median of driver-supplied per-region list, NEVER pooled), `shrinkage` (median of driver `shrinkage_list`), and pass-through `n_success`/`n_failed`/`convergence_rate`. Absent metrics are `None` + an explanatory `*_note`, never fabricated. No tensors leak across the JSON boundary.
- `compute_shrinkage_ratio(std_post, std_prior)` -- identifiability shrinkage accepting a scalar `std_prior` so it works across models with different A prior variances (1/64 BOLD, `LC_A_PRIOR_VARIANCE`) and for B; raises with expected-vs-actual on non-positive prior.
- `exclude_near_boundary_A()` rejects ground-truth A whose max real eigenvalue sits in `[-0.05, 0]` (eig_clamp non-injectivity, pitfall N2), with companion `resample_A_until_accepted()` driving a seeded closure and raising `RuntimeError` (tries count) if exhausted. Band edges exposed as `NEAR_BOUNDARY_LO` / `NEAR_BOUNDARY_HI`.
- `snr_for_model()` maps an SNR level onto each forward model's own knob: `{'SNR': snr}` for task/latent (direct simulator kwarg), `{'noise_log_amplitude': -log(snr)}` for spectral (observation-noise mechanism; higher SNR -> more-negative log-amplitude).
- 7 `@pytest.mark.vl` unit tests (3.2s laptop) concretely prove the hardening: a pooled-R2 or unmasked-sign implementation would fail `test_per_region_r2_not_pooled` / `test_masked_sign_recovery_ignores_structural_zeros`.

## Task Commits

1. **Tasks 1+2: metric assembler + design guards** - `6ae82cd` (feat)
2. **Task 3: unit tests** - `9fb55f4` (test)

Note: Tasks 1 and 2 share `benchmarks/recovery_matrix_metrics.py` and are interdependent (Task 1's verify imports Task 2's `exclude_near_boundary_A`/`snr_for_model`), so they landed in a single `feat` commit rather than two. See Deviations.

## Files Created/Modified

- `benchmarks/recovery_matrix_metrics.py` - assembler (`assemble_cell_metrics`), shrinkage helper (`compute_shrinkage_ratio`), near-boundary guard (`exclude_near_boundary_A`, `resample_A_until_accepted`), SNR mapping (`snr_for_model`), band constants.
- `tests/test_recovery_matrix_metrics.py` - 7 vl-marked unit tests.

## Decisions Made

- **Per-region R-squared is consumed, not computed.** The runners emit no per-seed trajectories, so `assemble_cell_metrics` reads a driver-supplied `r2_per_region_list` (the 30-02 driver calls `compute_trajectory_r_squared(pooled=False)`) and median-aggregates it; it never re-pools. Spectral/task have no trajectory -> `r2_per_region=None` + note.
- **Masked sign recovery reshapes the runner's flat A lists** (`a_true_list`/`a_inferred_list`, row-major `N*N`) back to `(N, N)` and calls the existing `masked_sign_recovery(pred, true, magnitude_threshold=sign_threshold)` per seed, then medians (dropping nan).
- **Spectral SNR uses `noise_log_amplitude = -log(snr)`** as a scalar the 30-02 driver expands into the `noise_params` `b`/`c` observation-noise tensors -- the single place SNR semantics diverge across the three models, keeping the matrix SNR axis comparable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Tasks 1 and 2 committed together (single shared file)**
- **Found during:** Task 1 verification
- **Issue:** Task 1's `<verify>` block imports `exclude_near_boundary_A` and `snr_for_model` (Task 2 functions) from the same module, so the file cannot pass Task 1's own verification until Task 2 is present. Per-task atomic commits on a single interdependent file would leave an intermediate non-importable state.
- **Fix:** Implemented both tasks' functions, then committed the complete module once as `feat(30-01)` (`6ae82cd`); tests committed separately (`9fb55f4`).
- **Files modified:** benchmarks/recovery_matrix_metrics.py
- **Verification:** ruff + mypy clean; all import/functional done-checks pass.
- **Committed in:** 6ae82cd

**2. [Rule 3 - Blocking] type: ignore[import-untyped] on the masked_sign_recovery import**
- **Found during:** Task 1 mypy verification
- **Issue:** `pyro_dcm` ships no `py.typed` marker, so under `strict = true` mypy raises `import-untyped` on `from pyro_dcm.models.hybrid_vae_dcm import masked_sign_recovery` (the same condition affects every existing `pyro_dcm`-importing benchmark module, e.g. the 29-04 runners).
- **Fix:** Added a scoped `# type: ignore[import-untyped]` on that one import line, keeping the change inside the plan's declared file. Deliberately did NOT touch `pyproject.toml` mypy config (broader than plan scope).
- **Files modified:** benchmarks/recovery_matrix_metrics.py
- **Verification:** `mypy benchmarks/recovery_matrix_metrics.py` -> Success, no issues.
- **Committed in:** 6ae82cd

**3. [Rule 2 - Missing Critical] Added a 7th test for the resample-exhausted path**
- **Found during:** Task 3
- **Issue:** Plan specified 6 tests but `resample_A_until_accepted`'s `RuntimeError` (exhausted `max_tries`) path was untested -- a fail-loud guard with an expected-vs-actual message should be covered.
- **Fix:** Added `test_resample_raises_when_none_accepted`.
- **Files modified:** tests/test_recovery_matrix_metrics.py
- **Verification:** 7/7 vl tests pass.
- **Committed in:** 9fb55f4

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing-critical test). **Impact on plan:** None on scope -- all within the two declared files; no architectural change.

## Issues Encountered

- The single-value `compute_summary_stats` call (1-seed quick smoke) emits a benign `std(): degrees of freedom <= 0` warning and yields `std=nan`. `nan` is JSON-serializable (`json.dumps` succeeds) and the sweep driver uses multiple seeds, so this is cosmetic only -- not a defect in the assembler.

## Next Phase Readiness

- The hardened metric layer, near-boundary guard, and SNR mapping are tested laptop library functions ready for the 30-02 sweep driver to call per cell.
- **30-02 driver responsibilities surfaced here:** populate `r2_per_region_list` via `compute_trajectory_r_squared(pooled=False)` for latent_circuit; populate `shrinkage_list` (per-seed mean `std_post/std_prior` via `compute_shrinkage_ratio`); wire `resample_A_until_accepted` with fresh per-try seeds; build the spectral `noise_params` from `snr_for_model`'s `noise_log_amplitude`.
- **Carried-forward INFRA blocker (unchanged):** the Mutagen `models/` ignore must be fixed before any M3 latent-circuit run (spectral/task sweeps unaffected).

---
*Phase: 30-recovery-matrix-sweep*
*Completed: 2026-06-10*
