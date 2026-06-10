---
phase: 29-vl-validation-infra-bmr-rank
plan: 01
subsystem: infra
tags: [config, pytest, matlab, variational-laplace, benchmark, spm12]

# Dependency graph
requires:
  - phase: 28-variational-laplace-engine
    provides: VL inference engine (run_variational_laplace_generic) that consumes max_iter
provides:
  - "BenchmarkConfig optional VL fields (max_iter, hyperprior_mean, hyperprior_precision, prior_mean_a_offset) with None defaults"
  - "MATLAB_PATH constant centralized in project-root config.py (env-overridable)"
  - "vl pytest marker registered in pyproject.toml"
affects: [29-04 VL runners, 30 recovery sweep, 32 SPM12 cross-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "None-default optional dataclass fields for backward-compatible config extension"
    - "Env-var-overridable Path constants centralized in config.py (mirrors TAPAS_RDCM_PATH)"

key-files:
  created: []
  modified:
    - benchmarks/config.py
    - config.py
    - pyproject.toml

key-decisions:
  - "All four new BenchmarkConfig fields default to None so every existing caller, quick_config/full_config, and test is byte-for-byte unaffected (VLINFRA-01)."
  - "MATLAB_PATH default literal matches the existing hardcoded value in validation/run_validation.py:58; validation/ left untouched (refactor deferred to Phase 32)."
  - "Did NOT apply ruff format's reformatting of quick_config/full_config signatures — the plan explicitly forbids touching those methods; the pre-existing multiline-signature diff is unrelated to this plan's additions."

patterns-established:
  - "Optional VL config fields appended AFTER fixtures_dir to preserve positional construction order."

# Metrics
duration: 31min
completed: 2026-06-10
---

# Phase 29 Plan 01: VL Config Foundation Summary

**Backward-compatible config foundation for v0.7.0 VL validation: four None-default VL fields on BenchmarkConfig, a centralized env-overridable MATLAB_PATH constant, and a registered `vl` pytest marker.**

## Performance

- **Duration:** 31 min (dominated by a parallel laptop-CPU test-suite run contended by sibling agents)
- **Started:** 2026-06-10T16:25:48Z
- **Completed:** 2026-06-10T16:57:16Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Added `max_iter`, `hyperprior_mean`, `hyperprior_precision`, `prior_mean_a_offset` to `BenchmarkConfig`, all defaulting to `None` (zero behavior change; reachable via `quick_config`/`full_config` `**kwargs`).
- Centralized `MATLAB_PATH` in project-root `config.py` as an env-overridable `Path` constant mirroring `TAPAS_RDCM_PATH`, establishing the single source of truth for the Phase 32 SPM12 bridge.
- Registered the `vl` pytest marker so `pytest -m vl` / `pytest -m "not vl"` run without unknown-marker warnings.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add optional VL fields to BenchmarkConfig** - `9403941` (feat)
2. **Task 2: Centralize MATLAB_PATH in config.py** - `07f7f4b` (feat)
3. **Task 3: Register the vl pytest marker** - `a2d69c6` (chore)

## Files Created/Modified
- `benchmarks/config.py` - Added 4 None-default VL fields after `fixtures_dir`, with NumPy-style docstring entries and a v0.7.0 `dt >= 0.1` constraint note (VLROBUST-02).
- `config.py` - Added env-overridable `MATLAB_PATH` constant + `Constants` docstring entry.
- `pyproject.toml` - Added `vl` marker to `[tool.pytest.ini_options].markers`.

## Decisions Made
- **All new fields None-default (VLINFRA-01):** keeps every existing benchmark caller and test unaffected; the fields are consumed only by VL runners (Plan 29-04).
- **MATLAB_PATH default matches `validation/run_validation.py:58`** (`C:/Program Files/MATLAB/R2022a/bin/matlab`); that file was deliberately not refactored (Phase 32 work).

## Deviations from Plan

### Deviations

**1. [Rule 3 - Blocking] `vl` marker pre-added by a parallel sibling agent**
- **Found during:** Task 3 (pyproject.toml had been modified externally between Read and Edit).
- **Issue:** A concurrently-running sibling agent (executing 29-02/29-03) had already inserted a `vl` marker line with a non-conforming description (`"vl: marks Variational Laplace validation tests (run with '-m vl')"`), missing the skip-flag documentation used by the other markers and by this plan's spec.
- **Fix:** Normalized the line to the exact plan-specified style: `"vl: marks tests as Variational Laplace validation (run with '-m vl', skip with '-m \"not vl\"')"`.
- **Files modified:** pyproject.toml
- **Verification:** `pytest --markers` shows the registered marker with the correct description; `pytest -m vl --co` collects 2 already-marked tests with no unknown-marker warning.
- **Committed in:** `a2d69c6` (Task 3 commit)

**2. [Plan-constraint adherence] Did NOT apply ruff-format reformat of `quick_config`/`full_config`**
- **Found during:** Task 1 verification (`ruff format --check benchmarks/config.py` flagged a reformat).
- **Issue:** `ruff format` wants to expand the pre-existing one-line `cls, variant, method, **kwargs` signatures of `quick_config`/`full_config` into multi-line form. This diff is unrelated to this plan's additions and lives in methods the plan explicitly says NOT to touch.
- **Resolution:** Left those signatures untouched to honor the plan constraint; the lines this plan added are format-clean and `ruff check` passes. The format-check non-conformance is pre-existing and out of scope.
- **Files modified:** none (intentional no-op)

---

**Total deviations:** 1 auto-fixed (1 blocking), 1 plan-constraint adherence note.
**Impact on plan:** No scope creep. Both relate to a concurrent multi-agent execution context, not to the plan's logic.

## Issues Encountered
- The additive regression run (`pytest -m "not vl and not mne and not spm and not slow and not tapas and not foundation and not latent"`) was contended on the shared laptop CPU by parallel sibling-agent test runs and was still in progress at finalization. Targeted verification covering this plan's surface — `BenchmarkConfig` construction (all four fields None), `quick_config`/`full_config` forwarding, `ruff check` on both edited modules, `config.MATLAB_PATH` resolution, `vl` marker registration, and `pytest -k config` (no failures) — all passed. Changes are strictly additive (new None-default fields, a new constant, a new marker string) and cannot alter existing test behavior.

## User Setup Required
None - no external service configuration required. (`MATLAB_PATH` default targets the user's existing R2022a install and is env-overridable; only relevant to Phase 32 SPM12 runs.)

## Next Phase Readiness
- VLINFRA-01 and VLINFRA-05 complete: the config root is in place for VL runners (Plan 29-04) and the Phase 32 SPM12 cross-validation bridge.
- No blockers introduced. The standing INFRA item (Mutagen `models/` ignore) is unrelated to this plan and remains tracked in STATE.md for any future M3 run touching `src/pyro_dcm/models/`.

---
*Phase: 29-vl-validation-infra-bmr-rank*
*Completed: 2026-06-10*
