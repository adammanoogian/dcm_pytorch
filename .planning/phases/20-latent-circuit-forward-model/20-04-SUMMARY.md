---
phase: 20-latent-circuit-forward-model
plan: 04
subsystem: benchmarks
tags: [pyro, svi, latent-circuit, recovery-benchmark, trajectory-r2, acceptance-gates, fixture-generation]

requires:
  - phase: 20-01
    provides: simulate_latent_circuit, make_stable_latent_circuit_A, CoupledDCMSystem(hemodynamic=False)
  - phase: 20-02
    provides: run_svi(n_restarts, guide_factory) multi-start SVI
  - phase: 20-03
    provides: latent_circuit_dcm_model, LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE, direct_observation

provides:
  - benchmarks/latent_circuit_metrics.py: trajectory R-squared, ELBO model selection, acceptance gate aggregation
  - benchmarks/runners/latent_circuit_recovery.py: SYNTH-01/02 recovery runner with multi-start SVI, 80/20 split, per-seed metrics
  - benchmarks/generate_fixtures.py extended with generate_latent_circuit_fixtures (.npz + manifest.json)
  - benchmarks/runners/__init__.py updated with run_latent_circuit_recovery and registry entry

affects:
  - 20-05 (prior recalibration sweep uses lc_a_prior_var/lc_b_prior_var kwargs and _patch_lc_priors)
  - cluster sbatch scripts (full SYNTH-01/02 acceptance runs go to M3, not laptop)

tech-stack:
  added: []
  patterns:
    - "_patch_lc_priors context manager for monkey-patching module constants during calibration sweep"
    - "importlib.import_module to bypass pyro_dcm.models.__init__ re-export naming conflict"
    - "_duration_override kwarg pattern for smoke test without cluster (API verification only)"
    - "80/20 train/test split: T_train = int(T * 0.8), held-out test for trajectory R-squared"

key-files:
  created:
    - benchmarks/latent_circuit_metrics.py
    - benchmarks/runners/latent_circuit_recovery.py
  modified:
    - benchmarks/generate_fixtures.py
    - benchmarks/runners/__init__.py

key-decisions:
  - "20-04-D1: importlib.import_module required to access LC_*_PRIOR_VARIANCE constants because pyro_dcm.models.__init__ re-exports latent_circuit_dcm_model as a function, shadowing the submodule name under import-as syntax."
  - "20-04-D2: _duration_override kwarg added to runner for API smoke testing; full 100s simulations take ~6s and each SVI step ~16s on laptop, making any non-trivial n_svi_steps smoke test exceed the 3-min routing threshold."
  - "20-04-D3: All acceptance thresholds (LC_A_RMSE_THRESHOLD=0.15, LC_B_RMSE_THRESHOLD=0.20, LC_SIGN_RECOVERY_THRESHOLD=0.80, LC_CI_COVERAGE_THRESHOLD=0.85, LC_TRAJECTORY_R2_THRESHOLD=0.95) are provisional and subject to Plan 20-05 recalibration."
  - "20-04-D4: Ground-truth B stability verified at generation time (max Re(eig(A + sum(B_j))) < -0.05 with 0.5x scale-down loop). B magnitudes [0.4, 0.3, 0.2] are stable for the chosen A with self_inhibition=0.5."
  - "20-04-D5: compute_latent_circuit_acceptance_gates uses MEDIAN aggregation across seeds (not mean) for robustness at small N (3 quick, 10+ full), matching the bilinear benchmark precedent."

patterns-established:
  - "Per-seed result dict shape: {a_rmse, b_rmse, sign_recovery, ci_coverage_95, trajectory_r_squared, shrinkage_A, shrinkage_B, final_elbo, elapsed_s, seed}"
  - "Runner returns aggregate medians alongside per-seed results and ground_truth tensors"
  - "Acceptance gates return {gates: {name: {observed_median, threshold, pass}}, per_seed, all_pass, n_seeds, thresholds_used}"

duration: 18min
completed: 2026-05-24
---

# Phase 20 Plan 04: Latent Circuit Recovery Runner and Metrics Summary

**Benchmark runner + metrics for SYNTH-01/02 latent-circuit parameter recovery: trajectory R-squared on held-out data, acceptance gates with provisional thresholds, multi-start SVI loop with 80/20 train/test split, and .npz fixture generation.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-05-24T21:25:52Z
- **Completed:** 2026-05-24T21:43:56Z
- **Tasks:** 2
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- Created `latent_circuit_metrics.py` with trajectory R-squared (per-region, mean across N), ELBO-based model selection, multi-level CI coverage, and a 5-gate acceptance aggregator using median across seeds.
- Created `latent_circuit_recovery.py` runner with N=4 directed chain ground truth, simulate_latent_circuit per seed, 80/20 train/test split, multi-start SVI via `run_svi(n_restarts, guide_factory)`, `extract_posterior_params` from best restart, per-seed metrics for all SYNTH-01/02 gates, seed-pool NaN rejection, and `_patch_lc_priors` context manager for Plan 20-05 calibration sweep.
- Extended `generate_fixtures.py` with `generate_latent_circuit_fixtures` following the .npz + manifest.json pattern, with the same A/B/C topology as the runner.
- Registered `("latent_circuit", "svi")` in RUNNER_REGISTRY for uniform benchmarking CLI access.

## Task Commits

1. **Task 1: Create latent_circuit_metrics.py** - `0419288` (feat)
2. **Task 2: Create runner + fixture generation** - `5aba415` (feat)

**Plan metadata:** (committed below in docs commit)

## Files Created/Modified

- `benchmarks/latent_circuit_metrics.py` - Trajectory R-squared, ELBO model selection, multi-level CI coverage, compute_latent_circuit_acceptance_gates with 5 provisional gates
- `benchmarks/runners/latent_circuit_recovery.py` - Main recovery runner: simulate, split 80/20, fit with multi-start SVI, extract posterior, compute all metrics, fixture-style seed-pool
- `benchmarks/generate_fixtures.py` - Added generate_latent_circuit_fixtures + latent_circuit entry in _GENERATORS
- `benchmarks/runners/__init__.py` - Added run_latent_circuit_recovery import and registry entry

## Decisions Made

- **[20-04-D1] importlib.import_module required for module constant monkey-patching.** The `pyro_dcm.models.__init__` re-exports `latent_circuit_dcm_model` as a function using the same name as the submodule; `import pyro_dcm.models.latent_circuit_dcm_model as lc` resolves to the function, not the module. Fixed using `importlib.import_module("pyro_dcm.models.latent_circuit_dcm_model")`.
- **[20-04-D2] Smoke test uses _duration_override=2.0 not full 100s.** At dt=0.01, 100s simulation = 10000 time points; one SVI step on this ODE takes ~16s on laptop. Any non-trivial n_svi_steps test exceeds the 3-min cluster routing threshold. Full acceptance runs go to M3 via sbatch.
- **[20-04-D3] Provisional acceptance thresholds with calibration-pending docstrings.** A-RMSE 0.15, B-RMSE 0.20, sign recovery 0.80, CI coverage 0.85, trajectory R2 0.95. Subject to Plan 20-05 recalibration.
- **[20-04-D4] B stability guard at ground-truth construction.** `while max Re(eig(A+sum(B_j)*scale)) >= -0.05: scale *= 0.5`. B magnitudes [0.4, 0.3, 0.2] are stable for the N=4 chain with self_inhibition=0.5.
- **[20-04-D5] Median aggregation for acceptance gates.** Matches bilinear benchmark precedent; robust at N=3 (quick) and N=10+ (full).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] importlib.import_module to bypass __init__ re-export naming conflict**
- **Found during:** Task 2 (runner implementation)
- **Issue:** `import pyro_dcm.models.latent_circuit_dcm_model as lc_model_module` resolved to the function (not module), so `lc_model_module.LC_A_PRIOR_VARIANCE` raised `AttributeError: 'function' object has no attribute 'LC_A_PRIOR_VARIANCE'`. This is because `pyro_dcm.models.__init__` does `from pyro_dcm.models.latent_circuit_dcm_model import latent_circuit_dcm_model` which shadows the submodule name.
- **Fix:** Used `importlib.import_module("pyro_dcm.models.latent_circuit_dcm_model")` to get the actual module object. Also imported the constants directly at module level for default fallback.
- **Files modified:** benchmarks/runners/latent_circuit_recovery.py
- **Verification:** `_lc_model_submodule.LC_A_PRIOR_VARIANCE` accessible; monkey-patching round-trips correctly.
- **Committed in:** 5aba415

**2. [Rule 2 - Missing Critical] Added _duration_override for smoke test routing compliance**
- **Found during:** Task 2 verification (runner smoke test)
- **Issue:** Full 100s simulation produces 10000 timesteps; 1 SVI step takes ~16s → 50 steps = ~14min, violating 3-min cluster routing threshold in CLAUDE.md. Plan's "50 steps in <3 min" expectation was not achievable locally.
- **Fix:** Added `_duration_override: float | None = None` kwarg to runner to allow 2s (200-timestep) test runs for API verification without cluster. Documented that full acceptance runs must go to M3.
- **Files modified:** benchmarks/runners/latent_circuit_recovery.py
- **Verification:** 2s duration smoke test completes in ~23s, API structure verified.
- **Committed in:** 5aba415

---

**Total deviations:** 2 auto-fixed (1 bug fix, 1 missing critical for compute routing compliance)
**Impact on plan:** Both auto-fixes necessary for correct operation and policy compliance. No scope creep.

## Issues Encountered

- Windows PermissionError on tempfile.TemporaryDirectory cleanup when np.load files are still open — not a code issue, used a persistent temp path for verification instead.

## Next Phase Readiness

- Plan 20-05 can immediately use `run_latent_circuit_recovery(config, lc_a_prior_var=X, lc_b_prior_var=Y)` for prior variance sweep.
- Full SYNTH-01/02 acceptance run requires M3 cluster sbatch job (100s simulation, 1000+ SVI steps, 10 restarts, 10+ seeds).
- All 5 acceptance gates provisional — Plan 20-05 recalibration will tighten or loosen based on recovery distributions.

---
*Phase: 20-latent-circuit-forward-model*
*Completed: 2026-05-24*
