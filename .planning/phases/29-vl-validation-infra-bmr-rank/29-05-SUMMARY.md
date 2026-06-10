---
phase: 29-vl-validation-infra-bmr-rank
plan: 05
subsystem: testing
tags: [variational-laplace, determinism, regression-tests, reproducibility, multi-restart, vl]

# Dependency graph
requires:
  - phase: 29-04
    provides: "three method=vl runners (spectral_vl, task_vl, latent_circuit_vl) + their ground-truth setup"
  - phase: 28-variational-laplace-engine
    provides: "run_variational_laplace_generic + ForwardModel protocol (3 forward models)"
provides:
  - "VL fixed-seed determinism regression tests across spectral, task, latent-circuit forward models"
  - "VL multi-restart reproducibility test (fixed restart-seed schedule -> deterministic winner)"
  - "documented determinism contract + known non-determinism sources (methods reference note)"
affects: [phase-30-recovery-sweep, vl-engine-edits, numerical-robustness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Determinism = run same fixed-seed fit twice, assert posterior means equal (bitwise preferred, atol 1e-8 fallback for BLAS-order jitter)"
    - "Multi-restart reproducibility = fixed restart-seed schedule -> same winning restart index + posterior across invocations"
    - "Tiny-settings regression: N=2 / short duration / low max_iter proves determinism without needing recovery accuracy"

key-files:
  created:
    - tests/test_vl_determinism.py
    - docs/03_methods_reference/vl_determinism_notes.md
  modified: []

key-decisions:
  - "torch.use_deterministic_algorithms(True) intentionally NOT forced (raises on VL linalg ops); reproducibility via fixed seeds + identical inputs"
  - "Multi-restart kept as a test-local helper, NOT added to the engine (out of scope for 29-05)"
  - "Per-forward-model max_iter tuned for laptop speed: spectral=24, task=6, latent=3; whole suite ~2m42s"

patterns-established:
  - "Determinism contract anchored at within-machine atol 1e-8; cross-machine comparison must use looser tol (~1e-6)"

# Metrics
duration: ~70 min
completed: 2026-06-10
---

# Phase 29 Plan 05: VL Determinism Regression Tests Summary

**Fixed-seed + multi-restart VL determinism suite across spectral/task/latent-circuit forward models (within-machine atol 1e-8, ~2m42s laptop) plus a methods-reference note documenting the determinism contract and its BLAS/float/ODE non-determinism sources.**

## Performance

- **Duration:** ~70 min (dominated by repeated laptop timing iterations to land under the 3-min budget)
- **Completed:** 2026-06-10
- **Tasks:** 3
- **Files created:** 2

## Accomplishments

- **Fixed-seed determinism** proven for all three forward models: running the same fit twice with the same seed yields equal posterior means within `atol=1e-8` (bitwise equality asserted first, allclose fallback for sub-1e-8 BLAS-order jitter).
- **Seed-sensitivity guard** confirms the seed genuinely drives the fit (different seeds -> different means), so the determinism assertions are not passing trivially.
- **Multi-restart reproducibility** (pitfall N4): a fixed restart-seed schedule `[10, 11, 12]` selects the same winning restart index and the same posterior mean across repeated invocations, proving the restart PATH is deterministic even though each restart explores a different basin.
- **Determinism contract documented** in `docs/03_methods_reference/vl_determinism_notes.md`: the within-machine contract, the four known non-determinism sources (BLAS reduction order, float64 accumulation, rk4 ODE step accumulation, finite-difference Jacobian step), the multi-restart caveat, and the cross-machine caveat for Phase 30.
- **Laptop-fast:** whole `-m vl` suite of 5 tests runs in ~2m42s, under the 3-minute budget, with no cluster job (per project routing policy; full sweep is Phase 30).

## Task Commits

1. **Task 1 + Task 2: fixed-seed determinism + multi-restart reproducibility tests** - `ed71f9c` (test) — both tasks write to the single artifact `tests/test_vl_determinism.py` and were committed as one atomic test-file unit.
2. **Task 3: document non-determinism sources** - `b0bfd6e` (docs)

**Plan metadata:** (docs commit, see git log)

## Files Created/Modified

- `tests/test_vl_determinism.py` - 5 `@pytest.mark.vl` tests: per-model fixed-seed determinism (`_fit_spectral`/`_fit_task`/`_fit_latent` helpers reusing the 29-04 runner ground-truth setup), seed sensitivity, and `_multistart_spectral` reproducibility.
- `docs/03_methods_reference/vl_determinism_notes.md` - determinism contract + non-determinism sources (BLAS, float64, ODE, FD step), multi-restart (N4) and cross-machine caveats; cites VLROBUST-01, pitfalls N4/N5.

## Decisions Made

- **`torch.use_deterministic_algorithms(True)` intentionally not forced.** It can raise on the VL engine's linalg ops (`solve`, `slogdet`, `cholesky`, `matrix_exp`). Reproducibility is achieved via fixed seeds + identical inputs and documented as such (in both the test module docstring and the methods note).
- **Multi-restart stays a test-local helper.** The plan scopes multi-restart wrapping out of the engine; `_multistart_spectral` re-seeds and re-fits from the default prior start, selecting the highest final free energy. This reproduces the restart-selection PATH deterministically without engine changes.
- **Per-model `max_iter` tuned to the laptop budget.** Spectral fits are cheap (~3-7s each) so `max_iter=24`; task and latent fits are dominated by per-parameter ODE Jacobians and a dense `(T*N, T*N)` precision, so they use `max_iter=6` (task, 20s duration) and `max_iter=3` (latent, 10s duration, N=4). Determinism holds regardless of `max_iter`, so the low caps cost nothing scientifically.

## Deviations from Plan

None - plan executed exactly as written. Tasks 1 and 2 both target the same file (`tests/test_vl_determinism.py`); they were committed as a single atomic test-file commit (`ed71f9c`) rather than two, since the multistart helper and the fixed-seed helpers share imports and cannot be cleanly separated on disk. All three tasks' deliverables, verifications, and done-criteria were met.

## Issues Encountered

- **Initial timing overshoot.** The first full run landed at ~3m40s (task fit at max_iter=16 / 40s duration was ~140s alone). Resolved by iteratively trimming task to `max_iter=6` / 20s and latent to `max_iter=3` / 10s, bringing the suite to ~2m42s with margin. Determinism is independent of these settings, so trimming did not weaken the tests.
- **mypy `no-any-return` on `pack_params(...).to(...)`.** The `pyro_dcm` package has no `py.typed` marker, so `pack_params` is `Any` to mypy. Bound each packed result to an explicitly-typed `torch.Tensor` local before returning, eliminating the three `no-any-return` errors. The remaining mypy notes on the file are the project-wide `import-untyped` class (present in every existing module), not new.

## Next Phase Readiness

- VLROBUST-01 satisfied: determinism regression suite green across all three forward models + non-determinism sources documented. Phase 30's recovery sweep now has a trustable within-machine reproducibility anchor.
- **Carry-forward caveat for Phase 30:** comparisons must be **within-machine** (laptop vs M3 BLAS builds can differ below `atol ~1e-6`); the `atol=1e-8` contract is a within-machine target only. This is documented in the methods note.
- **Unchanged prerequisite from 29-04:** the Mutagen `models/` unanchored-ignore fix is still required before any M3 latent-circuit VL run (latent forward model imports `pyro_dcm.models.*`).

---
*Phase: 29-vl-validation-infra-bmr-rank*
*Completed: 2026-06-10*
