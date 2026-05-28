---
phase: 24-foundation-model-use-cases
plan: 04
subsystem: cross-modal-comparison
tags: [cross-modal, pearson, kappa, ci-overlap, fmri, meeg, effective-connectivity]
dependency-graph:
  requires: ["24-02", "24-03"]
  provides:
    - Cross-modal A-matrix comparison metrics (Pearson r, sign kappa, CI overlap)
    - Comparison pipeline script with 3-panel figure generation
    - Complete foundation package with all extractor exports
  affects: []
tech-stack:
  added: []
  patterns:
    - "Frobenius normalization for scale-independent cross-modal comparison"
    - "Metric functions in library module, CLI in scripts/ for testability"
key-files:
  created:
    - src/pyro_dcm/foundation/comparison.py
    - scripts/24_compare_crossmodal.py
    - tests/test_crossmodal_comparison.py
  modified:
    - src/pyro_dcm/foundation/__init__.py
decisions: []
metrics:
  duration: "~10 minutes"
  completed: "2026-05-28"
---

# Phase 24 Plan 04: Cross-Modal Comparison Pipeline Summary

Frobenius-normalized A-matrix comparison with Pearson r, sign-pattern Cohen's kappa, and 95% CI overlap fraction; 3-panel figure (fMRI heatmap, M/EEG heatmap, element scatter); 11 unit tests on synthetic ground truth.

## What Was Built

### Comparison Metrics Module (`src/pyro_dcm/foundation/comparison.py`)

Four metric functions for comparing DCM A matrices across modalities:

- `normalize_a_matrix(a)`: Divides by Frobenius norm for scale-independent comparison. Addresses Pitfall 4 from RESEARCH.md (fMRI at 1 Hz vs M/EEG at 200 Hz).
- `compute_pearson_correlation(a1, a2)`: Flattens and computes Pearson r via `scipy.stats.pearsonr`.
- `compute_sign_kappa(a1, a2, threshold=0.0)`: Binarizes to {-1, 0, +1} sign patterns, computes Cohen's kappa via `sklearn.metrics.cohen_kappa_score`.
- `compute_credible_interval_overlap(a1_mean, a1_std, a2_mean, a2_std, z=1.96)`: Checks 95% CI overlap for each (i, j) element, returns overlap fraction.

All functions include shape validation with informative error messages.

### Comparison Script (`scripts/24_compare_crossmodal.py`)

Argparse CLI pipeline:
- `--fmri-results`: path to `tribe_dcm_results.npz` (from 24-02 pipeline)
- `--meeg-results`: path to `meeg_dcm_results.npz` (from 24-03 pipeline)
- `--output-dir`: comparison output directory
- `--roi-mapping`: optional JSON for ROI name alignment between modalities

Pipeline flow:
1. Load both `.npz` files, extract `A_mean` and `A_std`
2. Optionally reorder M/EEG ROIs to match fMRI ordering via mapping
3. Normalize A matrices by Frobenius norm
4. Compute Pearson r, sign kappa, CI overlap
5. Generate 3-panel figure: (a) fMRI heatmap, (b) M/EEG heatmap, (c) scatter with r annotation
6. Save metrics to `crossmodal_metrics.json`, figures as PNG (300 dpi) and PDF

### Foundation Package Exports (`src/pyro_dcm/foundation/__init__.py`)

Updated `__all__` to include:
- `TRIBEExtractor` (was missing from exports)
- `normalize_a_matrix`, `compute_pearson_correlation`, `compute_sign_kappa`, `compute_credible_interval_overlap`

### Unit Tests (`tests/test_crossmodal_comparison.py`)

11 tests organized by metric class:
- `TestNormalizeAMatrix`: unit norm (1.0), sign preservation, zero-matrix ValueError
- `TestPearsonCorrelation`: identical (r=1.0), uncorrelated (|r|<0.5), shape mismatch
- `TestSignKappa`: perfect agreement (kappa=1.0), opposite signs (kappa<0)
- `TestCredibleIntervalOverlap`: identical (1.0), separated (0.0), shape mismatch

## Decisions Made

None -- plan executed as written. The plan's preferred approach of placing metrics in `comparison.py` (library) with script importing from package was adopted for testability.

## Deviations from Plan

### Minor Scope Additions

**1. [Rule 2 - Missing Critical] Added comparison.py as library module**

- **Found during:** Task 1
- **Issue:** Plan suggested this as the preferred approach for testability
- **Fix:** Created `src/pyro_dcm/foundation/comparison.py` with all four metric functions; script and tests both import from it
- **Files created:** `src/pyro_dcm/foundation/comparison.py`

**2. [Rule 2 - Missing Critical] Added 3 extra tests for error handling**

- **Found during:** Task 2
- **Issue:** Plan specified 8 tests; added 3 more for ValueError edge cases (zero matrix, shape mismatches)
- **Fix:** Added `test_normalize_zero_matrix_raises`, `test_pearson_shape_mismatch_raises`, `test_ci_overlap_shape_mismatch_raises`
- **Files modified:** `tests/test_crossmodal_comparison.py`

## Verification Results

| Check | Result |
|-------|--------|
| `pytest tests/test_crossmodal_comparison.py -v` | 11/11 passed |
| `python scripts/24_compare_crossmodal.py --help` | Parses correctly |
| `python -c "from pyro_dcm.foundation import BaseExtractor, TRIBEExtractor, LaBraMExtractor, BrainOmniExtractor"` | All imports OK |
| `ruff check` on all source files | All checks passed |
| Full foundation suite (37 tests) | 37/37 passed |

## Commits

| Hash | Description |
|------|-------------|
| `66eedb4` | feat(24-04): add cross-modal A-matrix comparison script and metrics module |
| `94887d5` | test(24-04): add unit tests for cross-modal A-matrix comparison metrics |

## Next Phase Readiness

- Phase 24 is now complete: all four plans (24-01 base extractor, 24-02 TRIBE pipeline, 24-03 M/EEG pipeline, 24-04 cross-modal comparison) are done
- Cross-modal comparison requires actual Cam-CAN data and cluster runs to produce real `tribe_dcm_results.npz` and `meeg_dcm_results.npz`
- No blockers for downstream phases
