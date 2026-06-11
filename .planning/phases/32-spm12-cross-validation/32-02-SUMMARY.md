---
phase: 32-spm12-cross-validation
plan: 02
subsystem: testing
tags: [validation, free-energy, spm12, bmr, model-comparison, vlspm-02]

# Dependency graph
requires:
  - phase: 29-vl-validation-infrastructure
    provides: rank_connections / relative-evidence BMR semantics (absolute delta-F never gated)
  - phase: 32-spm12-cross-validation (Plan 32-01)
    provides: same-CSD injection so VL F and SPM F are evaluated on the IDENTICAL CSD
provides:
  - "compare_free_energies() — strict-5% relative-tolerance comparator for the single matched-problem free energy (VL free_energy[-1] vs SPM DCM.F) with a HARD within_tolerance pass/fail gate"
  - "S3-boundary test pinning compare_model_ranking (relative ranking) as the cross-model path"
affects: [32-03 run-vl-validation, vlspm-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-problem-only absolute-F comparator kept structurally separate from the cross-model relative-ranking path (compare_model_ranking) to preserve the S3 boundary"

key-files:
  created:
    - tests/test_compare_free_energies.py
  modified:
    - validation/compare_results.py

key-decisions:
  - "Strict 5% relative-F tolerance is the HARD default pass/fail gate (binding user decision), overriding the research's softer report-descriptively fallback."
  - "compare_free_energies is single-problem-only; its docstring forbids S3 cross-model absolute-F use and ties the 5% target to same-CSD injection (Plan 32-01)."
  - "No new mypy override added; the new function returns bare dict to match every existing sibling comparator (module's established pattern)."

patterns-established:
  - "Comparator separation: absolute-F (single matched problem) vs relative-ranking (cross-model) are distinct functions, never conflated."

# Metrics
duration: ~9min
completed: 2026-06-11
---

# Phase 32 Plan 02: Strict-5% Matched Free-Energy Comparator Summary

**`compare_free_energies()` encodes the binding strict-5% VL-vs-SPM matched-F gate as a single-problem-only `within_tolerance` pass/fail comparator, structurally separated from the S3-forbidden cross-model absolute-F path (which stays `compare_model_ranking`).**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-06-11T17:07:00Z
- **Completed:** 2026-06-11T17:16:18Z
- **Tasks:** 2
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments
- Added `compare_free_energies(vl_free_energy, spm_F, rel_tolerance=0.05)` to `validation/compare_results.py`, returning `{vl_free_energy, spm_F, relative_error, within_tolerance, rel_tolerance}` with the 5% default as a HARD gate (`rel_err < rel_tolerance`).
- Docstring binds three contract points verbatim in intent: VLSPM-02 same-matched-problem-only + 5% HARD gate (user decision); NEVER for S3 cross-model absolute-F (cross-model is `compare_model_ranking`); 5% target only meaningful on the IDENTICAL CSD (same-CSD injection, Plan 32-01).
- Added `tests/test_compare_free_energies.py` (5 `@pytest.mark.vl` tests, 1.48s laptop): within-tol pass, outside-tol fail, custom tolerance, zero-`spm_F` div-by-zero guard, and a cross-model-ranking-is-separate-path test pinning the S3 boundary.
- Existing functions untouched; the `within_tolerance` return key is the contract consumed by Plan 32-03's `run_vl_validation.py`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add compare_free_energies() to compare_results.py** - `4e1ed26` (feat)
2. **Task 2: Unit tests for compare_free_energies + S3-boundary guard** - `1a2a096` (test)

**Plan metadata:** see final `docs(32-02)` commit.

## Files Created/Modified
- `validation/compare_results.py` - Appended `compare_free_energies()` (relative-tolerance single matched-F comparator with `within_tolerance` gate); no existing function modified.
- `tests/test_compare_free_energies.py` - 5 `@pytest.mark.vl` unit tests covering the gate, custom tolerance, zero-F denominator safety, and the S3 cross-model boundary.

## Decisions Made
- **[32-02-D1] Strict 5% relative-F is the HARD default gate (binding user decision).** `compare_free_energies` defaults `rel_tolerance=0.05` and returns `within_tolerance = bool(rel_err < rel_tolerance)`; this overrides the research's softer "report descriptively" fallback. The comparator is single-problem-only and its docstring forbids S3 cross-model absolute-F use — keeping the gate defensible by construction. Cross-model agreement remains `compare_model_ranking` (relative ranking), pinned by `test_cross_model_ranking_is_separate_path`.
- **[32-02-D2] No new mypy override; bare `dict` return matches the module's established sibling pattern.** Every existing comparator (`compare_posterior_means`, `compare_model_ranking`, `compute_free_param_comparison`) returns annotated `-> dict:`; the new function follows suit. mypy baseline 15 → 16 errors, the single delta being the same `[type-arg]` on bare `dict` the whole file already emits (no new error category; pre-existing scipy-stub + bare-generic noise, consistent with 30-01-D4 scoping). Test file introduces zero mypy errors of its own.

## Deviations from Plan

None - plan executed exactly as written. Signature, return dict (including the `within_tolerance` key contract for Plan 32-03), all five tests, and the S3 docstring/test boundary match the plan verbatim.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. (No MATLAB needed; all tests are laptop-fast `@pytest.mark.vl`.)

## Next Phase Readiness
- `compare_free_energies` is ready for Plan 32-03 (`run_vl_validation.py`): the `within_tolerance` bool at `rel_tolerance=0.05` is the strict matched-F gate contract — signature and return dict must not be changed by downstream consumers.
- Plan 32-01 (parallel wave) supplies the same-CSD injection that makes the 5% target meaningful; this comparator assumes both F values come from the IDENTICAL CSD.
- No blockers. ruff clean on both files; mypy delta is pre-existing bare-generic noise only.

---
*Phase: 32-spm12-cross-validation*
*Completed: 2026-06-11*
