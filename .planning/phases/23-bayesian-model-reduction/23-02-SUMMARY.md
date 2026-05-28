---
phase: 23
plan: 02
subsystem: model-selection
tags: [bmr, circuit-selection, bayesian-model-reduction, pruning]
dependency-graph:
  requires: [23-01]
  provides: [bmr-circuit-selection, enumerate-reduced-models]
  affects: [23-03]
tech-stack:
  added: []
  patterns: [exhaustive-subset-enumeration, ranked-model-comparison]
key-files:
  created:
    - tests/test_bmr_circuit_selection.py
  modified:
    - src/pyro_dcm/model_selection/bmr.py
    - src/pyro_dcm/model_selection/__init__.py
decisions: []
metrics:
  duration: ~3 min
  completed: 2026-05-27
---

# Phase 23 Plan 02: BMR Circuit-Size Selection Summary

**One-liner:** Exhaustive BMR circuit search over 2^k prunable parameters with ranked delta-F scoring and sparse ground truth recovery.

## What Was Built

Two new functions added to `pyro_dcm.model_selection.bmr`:

1. **`enumerate_reduced_models`** -- generates all 2^k - 1 non-trivial subsets of prunable parameter indices, creating reduced priors for each via `make_reduced_prior_zero_connection`. Guards against k > 20 (ValueError) and warns at k > 15.

2. **`bmr_circuit_selection`** -- main entry point for circuit topology selection. Calls `enumerate_reduced_models`, scores each candidate against the full model using `bayesian_model_reduction`, includes the full model as baseline (delta_log_evidence = 0), and returns results sorted by evidence descending.

Both functions re-exported from `pyro_dcm.model_selection.__init__`.

## Test Coverage

7 tests in `tests/test_bmr_circuit_selection.py`:

| Test | What It Verifies |
|------|-----------------|
| test_enumerate_count | k=3 produces exactly 7 candidates |
| test_enumerate_labels | Labels and pruned_indices tuples correct for k=2 |
| test_enumerate_too_many_raises | k=21 raises ValueError |
| test_circuit_selection_identifies_sparse_truth | 5D posterior [0.5, 0, 0.3, 0, 0.8] -> best prunes {1,3} |
| test_circuit_selection_full_model_included | Full model entry has delta=0 and label="full_model" |
| test_circuit_selection_result_structure | All required keys present, results sorted descending |
| test_circuit_selection_single_prunable | k=1 gives exactly 2 candidates |

All 15 BMR tests pass (8 from 23-01 + 7 new).

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| ac2f851 | feat(23-02): implement BMR circuit-size selection with enumeration and ranking |

## Next Phase Readiness

Phase 23-03 (if it exists) can proceed. The circuit selection API is complete and tested.
