# Phase 10 Plan 03: Predictive-Based Extraction and Runner Plumbing Summary

Predictive-based extract_posterior_params with per-site mean/std/samples for all 6 guide types, sample-based quantiles in runners

## What Was Done

### Task 1: Redesign extract_posterior_params with Predictive-based sampling
- Replaced `guide.median()` + param store approach with `pyro.infer.Predictive` sampling
- New signature adds optional `model` and `num_samples` parameters (backward-compatible)
- Returns per-site dicts with `mean`, `std`, and `samples` keys
- Preserves backward-compatible `median` key at top level
- Handles complex-valued sites (e.g. `predicted_csd`) without casting warnings
- Works identically for all 6 guide types including AutoDelta (std=0)
- **Commit:** `23f2e37`

### Task 2: Update runners with guide_type and elbo_type parameterization
- **task_svi.py:** `create_guide` with `guide_type=config.guide_type, n_regions=N`, `run_svi` with `elbo_type=config.elbo_type, guide_type=config.guide_type`, post-Laplace guide extraction via `svi_result.get("guide", guide)`, replaced `guide.quantiles()` with `torch.quantile` on posterior samples
- **spectral_svi.py:** Same pattern as task_svi
- **task_amortized.py:** Per-subject SVI comparison uses `create_guide` with config params, `extract_posterior_params` instead of `svi_guide.median()`
- **spectral_amortized.py:** Same pattern as task_amortized
- **Commit:** `94b7f5e`

### Task 3: Add extraction tests for all 6 guide types
- 13 tests in `tests/test_posterior_extraction.py`
- Parametrized extraction test for all 6 guide types (6 tests)
- AutoDelta std=0 verification
- Backward-compatible median key test
- Sample-based quantiles bracket posterior mean
- num_samples parameter controls sample count (3 parameterized tests)
- AutoLaplace post-guide produces non-zero std
- **Commit:** `782147f`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Complex-site handling via `tensor.mean(dim=0)` for complex types | Avoids casting warning while preserving correct complex mean |
| `model` defaults to `guide.model` | All AutoGuide subclasses store the model; preserves backward compatibility |
| 1000 default num_samples | Balances accuracy vs speed for standard extraction |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Complex tensor casting warning in spectral DCM extraction**
- **Found during:** Task 1 verification
- **Issue:** `tensor.float().mean(dim=0)` on complex-valued `predicted_csd` site produces UserWarning about discarding imaginary part
- **Fix:** Added conditional branch: use `tensor.mean(dim=0)` for complex tensors, `tensor.float().mean(dim=0)` for real tensors
- **Files modified:** `src/pyro_dcm/models/guides.py`
- **Commit:** `23f2e37`

## Verification Results

| Check | Result |
|-------|--------|
| `pytest tests/test_posterior_extraction.py -v` | 13/13 passed |
| `pytest tests/test_guide_factory.py tests/test_elbo_variants.py -x -q` | 50/50 passed |
| `pytest tests/test_svi_integration.py -x -q` | 9/9 passed |
| All 4 runner files import without error | Confirmed |

## Key Files

### Created
- `tests/test_posterior_extraction.py` -- 13 tests for Predictive-based extraction

### Modified
- `src/pyro_dcm/models/guides.py` -- Predictive-based `extract_posterior_params`
- `benchmarks/runners/task_svi.py` -- guide_type/elbo_type plumbing, sample-based quantiles
- `benchmarks/runners/spectral_svi.py` -- guide_type/elbo_type plumbing, sample-based quantiles
- `benchmarks/runners/task_amortized.py` -- guide_type/elbo_type plumbing, extract_posterior_params
- `benchmarks/runners/spectral_amortized.py` -- guide_type/elbo_type plumbing, extract_posterior_params

## Duration

~11 minutes
