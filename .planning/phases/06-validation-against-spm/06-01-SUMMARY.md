---
phase: 06-validation-against-spm
plan: 01
subsystem: validation
tags: [mat-export, spm12, tapas, comparison, round-trip]

dependency_graph:
  requires: ["01", "02", "03", "04", "05"]
  provides: ["validation-export", "comparison-utilities", "matlab-batch-scripts"]
  affects: ["06-02", "06-03"]

tech_stack:
  added: []
  patterns: ["python-to-mat-export", "nested-struct-dict", "hybrid-error-metric"]

key_files:
  created:
    - validation/__init__.py
    - validation/export_to_mat.py
    - validation/compare_results.py
    - validation/matlab_scripts/run_spm_task_dcm.m
    - validation/matlab_scripts/run_spm_spectral_dcm.m
    - validation/matlab_scripts/run_tapas_rdcm.m
    - tests/test_validation_export.py
  modified: []

decisions:
  - id: safe-division-hybrid-metric
    decision: "Use np.where with safe_ref=1.0 for zero-valued refs"
    rationale: "Avoids RuntimeWarning divide-by-zero in positions where absolute error is used anyway"

metrics:
  duration: "~9 minutes"
  completed: "2026-03-28"
---

# Phase 6 Plan 1: Validation Infrastructure Summary

**One-liner:** Python-to-MATLAB .mat export for all 3 DCM variants, MATLAB batch scripts for SPM12/tapas headless estimation, and hybrid comparison utilities with 14 round-trip tests.

## What Was Done

### Task 1: DCM Export Functions and MATLAB Batch Scripts
Built the complete `validation/` directory structure with export functions and MATLAB batch scripts.

**Export functions** (`validation/export_to_mat.py`):
- `export_task_dcm_for_spm`: Builds full DCM struct matching `spm_dcm_estimate` requirements -- Y.y (BOLD), U.u (microtime stimulus), connectivity masks (a/b/c/d), dimensions, timing, and options. All scalars wrapped as `np.array([[value]])` per MATLAB convention.
- `export_spectral_dcm_for_spm`: Same struct but with `options.induced=1`, `options.analysis='CSD'`, `options.order=8`. Uses constant input stimulus. Exports BOLD (not CSD) since SPM computes CSD from BOLD via MAR model internally.
- `export_rdcm_for_tapas`: Same DCM struct format for tapas compatibility with minimal options.
- `upsample_stimulus`: Converts TR-resolution stimulus to microtime resolution (TR/16) using nearest-neighbor interpolation, with 32-row zero-padding at start (SPM discards first 32 microtime samples).

**MATLAB batch scripts** (`validation/matlab_scripts/`):
- `run_spm_task_dcm.m`: Loads DCM .mat, adds `Y.Q` precision components if missing, calls `spm_dcm_estimate`, saves `Ep_A`, `Ep_C`, `Cp`, `F`, predicted BOLD, residuals.
- `run_spm_spectral_dcm.m`: Forces CSD analysis mode, calls `spm_dcm_fmri_csd`, saves A posterior + spectral-specific outputs (transit, decay, Hc, Hz).
- `run_tapas_rdcm.m`: Runs both rigid (methods=1) and sparse (methods=2) rDCM estimation, with path availability checks and clear error messages.

All MATLAB scripts support environment variable overrides for input/output paths and include try/catch error handling with fprintf status messages.

### Task 2: Comparison Utilities and Round-trip Tests
**Comparison utilities** (`validation/compare_results.py`):
- `load_spm_results`: Handles scipy.io.loadmat nested struct access (`results[field][0, 0]`) for SPM12 output.
- `load_tapas_results`: Extracts rigid/sparse sub-structs with Ep, logF, Ip fields.
- `compare_posterior_means`: Hybrid metric -- relative error for `|ref| > 0.01`, absolute error for near-zero parameters. Returns max/mean errors, within_tolerance boolean, element error matrix.
- `compare_model_ranking`: Pairwise ranking agreement between SPM free energy (higher=better) and Pyro ELBO (higher=better). Returns agreement rate, detailed pairwise results.
- `compute_free_param_comparison`: Compare A matrices in free parameter space (SPM Ep.A stores free params, not parameterized).

**Tests** (`tests/test_validation_export.py`): 14 tests across 6 test classes:
- 3 round-trip tests (task, spectral, rDCM): export to .mat, reload, verify all nested struct fields, shapes, dtypes, and 2D scalar wrapping.
- 2 upsampling tests: shape/padding verification, block pattern preservation.
- 5 comparison metric tests: large values, near-zero values, failure detection, shape consistency, mixed metric switching.
- 4 ranking comparison tests: perfect agreement, one disagreement (2/3 rate), minimal 2-scenario case, pairwise detail structure.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Safe division in hybrid error metric**
- **Found during:** Task 2 test execution
- **Issue:** `np.where(mask, abs_diff / abs_ref, abs_diff)` evaluates both branches, producing RuntimeWarning when `abs_ref` contains zeros even though those positions use absolute error.
- **Fix:** Replaced with `safe_ref = np.where(large_mask, abs_ref, 1.0)` before division.
- **Files modified:** `validation/compare_results.py`
- **Commit:** 4389321

## Verification

1. `validation/` directory exists with `export_to_mat.py`, `compare_results.py`, `matlab_scripts/`
2. `python -m pytest tests/test_validation_export.py -v` -- 14/14 tests pass (0 warnings with -W error::RuntimeWarning)
3. Export functions produce .mat files loadable by `scipy.io.loadmat` with correct nested struct structure
4. MATLAB scripts have valid syntax and correct SPM12 API calls (spm_dcm_estimate, spm_dcm_fmri_csd, tapas_rdcm_estimate)
5. Comparison utilities correctly implement hybrid error metric (relative for |x|>0.01, absolute for |x|<=0.01)
6. No MATLAB dependency required for test suite

## Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | DCM export functions and MATLAB batch scripts | 4bf4808 | validation/export_to_mat.py, validation/matlab_scripts/*.m |
| 2 | Comparison utilities and round-trip tests | 4389321 | validation/compare_results.py, tests/test_validation_export.py |

## Next Phase Readiness

Plan 06-01 provides the complete infrastructure for Plans 06-02 and 06-03:
- Export functions ready for generating validation .mat files from synthetic data
- MATLAB scripts ready for headless SPM12/tapas execution via subprocess
- Comparison utilities ready for element-wise posterior comparison
- All Python-side tests pass; MATLAB execution depends on SPM12 availability (verified installed at `C:/Users/aman0087/Documents/Github/spm12/`)

**Blockers for 06-02/06-03:** None. tapas may need to be cloned from GitHub if rDCM validation is included in a later plan.
