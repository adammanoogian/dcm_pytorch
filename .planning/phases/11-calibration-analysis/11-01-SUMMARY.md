---
phase: 11-calibration-analysis
plan: 01
subsystem: benchmarks
tags: [coverage, calibration, metrics, sweep, multi-level-ci]
dependency_graph:
  requires: [09-01, 09-02, 09-03, 10-01, 10-02, 10-03]
  provides: [multi-level-coverage-metrics, calibration-sweep-orchestrator, per-parameter-breakdown]
  affects: [11-02, 11-03]
tech_stack:
  added: []
  patterns: [tiered-benchmark-config, empirical-quantile-ci, z-score-multi-level-ci]
key_files:
  created:
    - benchmarks/calibration_sweep.py
  modified:
    - benchmarks/metrics.py
    - benchmarks/runners/spectral_svi.py
    - benchmarks/runners/task_svi.py
    - benchmarks/runners/rdcm_vb.py
decisions:
  - id: empirical-quantile-ci
    choice: "torch.quantile for SVI runners, z-scores for rDCM analytic posterior"
    rationale: "Empirical quantiles accurate for non-Gaussian posteriors (IAF, flows); rDCM has known Gaussian posterior"
  - id: str-keys-for-coverage-multi
    choice: "String keys for coverage_multi dicts in JSON output"
    rationale: "JSON does not support float keys; convert at serialization boundary"
metrics:
  duration: ~35 minutes
  completed: 2026-04-12
---

# Phase 11 Plan 01: Multi-Level Coverage and Calibration Sweep Summary

**One-liner:** Empirical quantile-based multi-level coverage at 4 CI levels (0.50-0.95) with diagonal/off-diagonal A breakdown, plus tiered calibration_sweep.py orchestrator with resume support.

## What Was Done

### Task 1: Multi-level coverage metrics functions
Added three new functions to `benchmarks/metrics.py`:
- `compute_coverage_multi_level`: Computes coverage at 4 CI levels (0.50, 0.75, 0.90, 0.95) using empirical quantiles via `torch.quantile` (not z-scores), which is accurate for non-Gaussian posteriors from AutoIAF and flow-based guides
- `compute_coverage_by_param_type`: Splits A matrix into diagonal and off-diagonal elements, computes multi-level coverage for each subset
- `compute_summary_stats`: Returns median, q25, q75, mean, std -- compliant with STATE.md risk P12 (avoid mean-only reporting)

### Task 2: Runner multi-level coverage extensions
Extended all 4 runners (spectral SVI, task SVI, rDCM rigid, rDCM sparse):
- SVI runners: Compute `A_param_samples` via `parameterize_A` on each posterior sample, then empirical quantile coverage at 4 levels with diagonal/off-diagonal breakdown
- rDCM runners: Use z-score CIs at 4 levels (z=0.6745, 1.1503, 1.6449, 1.9600) from analytic Gaussian posterior
- All runners: Added `coverage_multi`, `coverage_diag_multi`, `coverage_offdiag_multi` dicts to results, plus `compute_summary_stats` for RMSE/coverage/correlation/time
- Backward compatible: Existing `mean_rmse`, `mean_coverage` etc. keys preserved

### Task 3: Calibration sweep orchestrator
Created `benchmarks/calibration_sweep.py` with:
- 3 tiers: T1 (6 guides x spectral x N=3), T2 (mean-field x 3 ELBOs x N=3,5), T3 (all variants x all sizes + rDCM)
- `--tier 1|2|3|all`, `--quick`, `--resume`, `--fixtures-dir`, `--output-dir`, `--seed` CLI
- RUNNER_REGISTRY dispatch with guide-ELBO validation and auto_mvn N>7 skip (STATE.md P6)
- Intermediate JSON save after each config for crash resume safety
- Result keys: `{variant}_{guide}_{elbo}_{N}` convention
- Summary table with median RMSE, cov@90, correlation, wall time

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Empirical quantiles for SVI, z-scores for rDCM | SVI posteriors may be non-Gaussian (IAF/flows); rDCM is analytic Gaussian |
| String keys for coverage_multi dicts | JSON does not support float keys; convert at serialization boundary |
| Tiered sweep with deduplication | Tiers overlap intentionally; `_expand_tier("all")` deduplicates via set |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added `_extract_A_std_rigid` and `_extract_A_std_sparse` helpers**
- Found during: Task 2 (rDCM runner extension)
- Issue: Multi-level z-score CIs need A_mu and A_std matrices. Existing `_extract_A_ci` only returned A_lo/A_hi at fixed 1.96z, not raw std
- Fix: Added two helper functions to extract standard deviation matrices from VB posterior for use at arbitrary z-scores
- Files modified: `benchmarks/runners/rdcm_vb.py`

## Verification Results

1. `compute_coverage_multi_level` returns dict with 4 float keys, float values in [0,1] -- PASS
2. `compute_coverage_by_param_type` returns dict with "all", "diagonal", "off_diagonal" keys -- PASS
3. `compute_summary_stats` returns median, q25, q75, mean, std -- PASS
4. All runner imports succeed -- PASS
5. `calibration_sweep.py --help` shows all 6 flags -- PASS
6. Tier expansion: T1=6, T2=12, T3=34, all=42 configs (deduplicated) -- PASS
7. Skip logic: rejects rDCM+SVI guide, SVI+vb, auto_mvn N>7 -- PASS
8. Existing test suite (72 tests in relevant modules) passes -- PASS

## Next Phase Readiness

Plan 11-02 (figures and calibration curves) can now proceed. All runners produce multi-level coverage data in structured JSON format. `calibration_sweep.py` can generate the full dataset needed for analysis.
