# Phase 20 Plan 04: Benchmark Runner & Metrics Summary

Latent circuit DCM benchmark metrics and multi-start SVI recovery runner with fixture generation.

## Metadata

- **Phase:** 20
- **Plan:** 04
- **Subsystem:** benchmarks
- **Tags:** metrics, recovery, multi-start-svi, latent-circuit
- **Completed:** 2026-05-24
- **Duration:** ~6 minutes

## Dependency Graph

- **Requires:** 20-01 (forward model), 20-02 (multi-start SVI), 20-03 (Pyro model)
- **Provides:** Recovery runner, acceptance gate metrics, fixture generation for latent circuit DCM
- **Affects:** 20-05 (prior recalibration sweep consumes runner output)

## Tech Stack

- **Added:** None (no new dependencies)
- **Patterns:** Multi-start SVI recovery runner pattern (extends Phase 16 bilinear benchmark); median-based acceptance gates; trajectory R-squared metric

## Key Files

### Created

- `benchmarks/latent_circuit_metrics.py` -- Trajectory R-squared, ELBO model selection, acceptance gates, multi-level coverage
- `benchmarks/runners/latent_circuit_recovery.py` -- End-to-end recovery runner with multi-start SVI

### Modified

- `benchmarks/runners/__init__.py` -- Registered `(latent_circuit, svi)` in RUNNER_REGISTRY
- `benchmarks/generate_fixtures.py` -- Added `generate_latent_circuit_fixtures()` and registered in CLI

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| D20-04-01 | Median-based gate evaluation (not mean) | Robust to outlier seeds with NaN or extreme values |
| D20-04-02 | Train/test split 80/20 temporal | Preserves temporal structure; test set = future prediction ability |
| D20-04-03 | Provisional thresholds are soft (calibrated in 20-05) | Cannot set final thresholds without cluster sweep data |
| D20-04-04 | Prior override via module monkey-patching (restored in finally) | Simple mechanism for prior_var sweep in 20-05 without modifying model signature |

## Verification Results

All import checks pass:
```
python -c "from benchmarks.latent_circuit_metrics import compute_trajectory_r_squared, compute_elbo_model_selection, compute_latent_circuit_acceptance_gates; print('OK')"
python -c "from benchmarks.runners import run_latent_circuit_recovery; print('OK')"
python -c "from benchmarks.generate_fixtures import generate_latent_circuit_fixtures; print('OK')"
```

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| 68a7cc3 | feat(20-04): add latent circuit metrics module |
| 092b6fe | feat(20-04): add latent circuit recovery runner and fixture generation |

## Next Phase Readiness

Plan 20-05 (prior recalibration sweep) can proceed. The runner exposes `lc_a_prior_var` and `lc_b_prior_var` kwargs for the joint prior_var x init_scale grid search. Full recovery runs must execute on M3 cluster (projected >30 min per seed x 10 restarts).
