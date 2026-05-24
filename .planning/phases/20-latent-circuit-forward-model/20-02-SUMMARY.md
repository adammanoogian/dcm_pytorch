---
phase: 20
plan: 02
subsystem: inference
tags: [svi, multi-start, guide-factory, elbo, optimization]
dependency_graph:
  requires: []
  provides: [multi-start-svi, guide-factory-pattern]
  affects: [20-03, 20-04, 20-05]
tech_stack:
  added: []
  patterns: [multi-start-optimization, nan-resilient-loop, param-store-checkpoint-restore]
key_files:
  created:
    - tests/test_multi_start_svi.py
  modified:
    - src/pyro_dcm/models/guides.py
decisions:
  - id: D20-02-01
    description: "n_restarts=1 uses exact original code path (no structural changes)"
    rationale: "Backward compatibility paramount -- no risk of breaking existing callers"
  - id: D20-02-02
    description: "guide_factory required for n_restarts>1 (not optional)"
    rationale: "Safe re-initialization requires fresh guide; reusing same guide accumulates state"
  - id: D20-02-03
    description: "NaN restarts get final_loss=inf and are skipped"
    rationale: "Resilience: one bad init should not abort entire multi-start run"
  - id: D20-02-04
    description: "Param store restored to best restart after completion"
    rationale: "Callers can immediately use extract_posterior_params without extra steps"
metrics:
  duration: "20m"
  completed: "2026-05-24"
---

# Phase 20 Plan 02: Multi-Start SVI Summary

Multi-start SVI via n_restarts + guide_factory parameters on run_svi(), enabling 10+ independent SVI optimizations with best-ELBO selection for latent circuit DCM fitting.

## What Was Done

### Task 1: Add n_restarts and guide_factory to run_svi

Extended `run_svi()` in `src/pyro_dcm/models/guides.py` with two new parameters:

- `n_restarts: int = 1` -- number of independent SVI runs (default preserves backward compat)
- `guide_factory: Callable[[], Any] | None = None` -- callable returning fresh guide per restart

Implementation details:
- **Single-restart path** (`n_restarts <= 1`): Delegates to `_run_single_svi()` helper, returns exact same dict structure as before (`{"losses", "final_loss", "num_steps"}`). No `all_restarts`, `n_restarts`, or `best_restart_idx` keys.
- **Multi-restart path** (`n_restarts > 1`): Validates `guide_factory`, runs N independent SVI loops, catches NaN (assigns `inf`), selects best by minimum `final_loss`, restores param store to best state.
- Extracted `_build_elbo()` and `_run_single_svi()` helpers to avoid code duplication.
- Added logging via `pyro_dcm.svi` logger (INFO level per-restart and best-selection messages).

### Task 2: Multi-start SVI test suite

Created `tests/test_multi_start_svi.py` with 9 tests across 6 test classes:

1. `TestSingleRestartBackwardCompat` (2 tests): Exact key-set verification
2. `TestMultiRestartReturnsBest` (1 test): 5-restart best-selection logic
3. `TestMultiRestartRequiresGuideFactory` (2 tests): ValueError enforcement
4. `TestMultiRestartFreshInit` (1 test): Independent loss trajectories
5. `TestMultiRestartNanResilience` (2 tests): NaN -> inf + all-NaN RuntimeError
6. `TestMultiRestartParamStoreRestored` (1 test): Best state in param store

## Verification Results

- `python -m pytest tests/test_multi_start_svi.py -x -v`: 9/9 passed (9.84s)
- `python -m pytest tests/test_svi_integration.py tests/test_guide_factory.py -x -q`: 41/41 passed (166.75s) -- zero regressions

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| D20-02-01 | n_restarts=1 uses exact original code path | Backward compat paramount |
| D20-02-02 | guide_factory required for n_restarts>1 | Safe re-initialization needs fresh guide |
| D20-02-03 | NaN restarts get inf and are skipped | One bad init shouldn't abort entire multi-start |
| D20-02-04 | Param store restored to best restart | Callers can immediately extract posteriors |

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| 39890c7 | feat(20-02): add multi-start SVI to run_svi |
| 5bcc809 | test(20-02): add multi-start SVI test suite |

## Next Phase Readiness

Multi-start SVI (MODEL-05) is now available for all downstream plans:
- Plan 20-03 (latent circuit DCM model) can use `n_restarts >= 10`
- Plan 20-04 (synthetic validation) will exercise multi-start on realistic models
- Plan 20-05 (prior recalibration) will rely on multi-start for fair comparison
