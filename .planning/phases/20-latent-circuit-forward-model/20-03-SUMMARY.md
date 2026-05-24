---
phase: 20-latent-circuit-forward-model
plan: 03
subsystem: models
tags: [pyro, dcm, latent-circuit, svi, autoguide, prior-recalibration, observation-model]

# Dependency graph
requires:
  - phase: 20-01
    provides: CoupledDCMSystem(hemodynamic=False) and simulate_latent_circuit
  - phase: 20-02
    provides: multi-start run_svi with n_restarts and guide_factory

provides:
  - direct_observation(x, C_obs, noise_prec) function in forward_models
  - latent_circuit_dcm_model Pyro generative model (linear + bilinear)
  - LC_A_PRIOR_VARIANCE=1/16 and LC_B_PRIOR_VARIANCE=1.0 module-level constants
  - 11-test suite covering trace, guide auto-discovery, SVI smoke, and prior separation

affects:
  - phase 20-04 (prior recalibration calibrates LC_A_PRIOR_VARIANCE on synthetic RNNs)
  - phase 20-05 (latent circuit demo script uses latent_circuit_dcm_model)
  - phase 21 (alignment uses latent_circuit_dcm_model for fitting)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "latent_circuit_dcm_model forks task_dcm_model pattern without shared base (no hemodynamics, no downsampling, identity C_obs)"
    - "pyro.deterministic() nodes appear as type='sample' in Pyro trace in this version -- check by name not type"
    - "AutoIAFNormal hidden_dim must exceed latent_dim (N*N + N*M + 1 = 21 for N=4, M=1); use 32"
    - "direct_observation is a pure deterministic function; Pyro likelihood in calling model"

key-files:
  created:
    - src/pyro_dcm/forward_models/latent_observation.py
    - src/pyro_dcm/models/latent_circuit_dcm_model.py
    - tests/test_latent_circuit_model.py
  modified:
    - src/pyro_dcm/forward_models/__init__.py
    - src/pyro_dcm/models/__init__.py

key-decisions:
  - "LC_A_PRIOR_VARIANCE=1/16 (wider than BOLD 1/64) to match RNN hidden-state timescales (pitfall LC4)"
  - "C_obs fixed at identity for v0.6.0 -- rotation ambiguity (pitfall LC5) deferred to v0.7.0+"
  - "Initial state torch.zeros(N) not make_initial_state(5N) for hemodynamic=False mode [20-01-D3]"
  - "AutoIAFNormal hidden_dim must exceed latent_dim; [32] chosen for N=4 test fixture"
  - "pyro.deterministic() sites classified as type='sample' in this Pyro version; check by name"

patterns-established:
  - "LC model: no TR, no downsampling, dt=0.01, CoupledDCMSystem(hemodynamic=False)"
  - "Same site naming as task_dcm_model (A_free, C, B_free_j, noise_prec, obs) for guide auto-discovery"
  - "NaN-safe guard: zero-fill predicted_trajectories before likelihood, same as task_dcm_model"

# Metrics
duration: 19min
completed: 2026-05-24
---

# Phase 20 Plan 03: Latent Circuit Model Summary

**Pyro latent-circuit DCM with direct neural-state observation: direct_observation function, latent_circuit_dcm_model (linear+bilinear), LC-specific prior constants 1/16 and 1.0, and 11-test suite covering traces, guide auto-discovery, and SVI smoke tests**

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-24T21:03:56Z
- **Completed:** 2026-05-24T21:22:16Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Created `direct_observation(x, C_obs, noise_prec)` -- pure deterministic observation function with identity C_obs (v0.6.0 scope, pitfall LC5)
- Created `latent_circuit_dcm_model` forked from task_dcm_model: no hemodynamics, no downsampling, dt=0.01, CoupledDCMSystem(hemodynamic=False), LC-specific priors
- Created 11-test suite: all 9 fast tests pass immediately, both slow SVI tests pass in 94s; AutoNormal, AutoLowRankMVN, AutoIAFNormal all auto-discover sites without factory changes

## Task Commits

1. **Task 1: direct_observation function** - `e4eeefe` (feat)
2. **Task 2: latent_circuit_dcm_model** - `c9b0bd2` (feat)
3. **Task 3: 11-test suite** - `8779580` (test)

## Files Created/Modified

- `src/pyro_dcm/forward_models/latent_observation.py` - direct_observation(x, C_obs, noise_prec) pure deterministic function
- `src/pyro_dcm/forward_models/__init__.py` - re-export direct_observation
- `src/pyro_dcm/models/latent_circuit_dcm_model.py` - latent_circuit_dcm_model Pyro model + LC_A/B_PRIOR_VARIANCE constants
- `src/pyro_dcm/models/__init__.py` - re-export model and constants
- `tests/test_latent_circuit_model.py` - 11-test suite (9 fast, 2 slow)

## Decisions Made

- **pyro.deterministic() type='sample' in this Pyro version.** Tests cannot distinguish deterministic from stochastic sites by `v["type"] == "deterministic"` -- in this Pyro install, `pyro.deterministic()` registers with `type="sample"`. Tests check by site name instead. A comment was added to each affected test.
- **AutoIAFNormal hidden_dim=32** (not [20]). For N=4, M=1, latent dim = N*N + N*M + 1 = 21. AutoRegressiveNN requires hidden_dim >= latent_dim; [20] fails. [32] chosen as smallest power-of-2 above 21.
- **LC_A_PRIOR_VARIANCE=1/16 confirmed.** Calibrated for RNN hidden-state timescales. This is a separate constant from task_dcm_model's A prior (1/64), addressing pitfall LC4.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test assertions for pyro.deterministic() site type**

- **Found during:** Task 3 (test suite)
- **Issue:** Tests used `v["type"] == "deterministic"` to find pyro.deterministic() sites, but in this Pyro version those nodes have `type="sample"`. This caused 3 test failures on the first run.
- **Fix:** Changed all deterministic-site tests to use name-based lookup (`k in all_sites` where `all_sites` includes all type="sample" nodes). Added comment explaining the Pyro version behavior.
- **Files modified:** tests/test_latent_circuit_model.py
- **Verification:** All 9 fast tests pass after fix.
- **Committed in:** 8779580

**2. [Rule 1 - Bug] Fixed AutoIAFNormal hidden_dim < latent_dim**

- **Found during:** Task 3 (test 5 for AutoIAFNormal)
- **Issue:** `hidden_dim=[20]` in test was below latent_dim=21 (N=4, M=1), causing `ValueError: Hidden dimension must not be less than input dimension` in AutoRegressiveNN.
- **Fix:** Changed to `hidden_dim=[32]` with a comment explaining the constraint.
- **Files modified:** tests/test_latent_circuit_model.py
- **Verification:** test_guide_auto_discovery_auto_iaf_linear passes.
- **Committed in:** 8779580

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug during Task 3 test implementation)
**Impact on plan:** Both required for correct test execution; no scope creep.

## Issues Encountered

None beyond the auto-fixed test bugs above.

## Next Phase Readiness

- `latent_circuit_dcm_model` is ready for Phase 20 Plan 04 (prior recalibration on synthetic RNNs)
- All guide auto-discovery contracts verified; no factory changes needed for Phase 20 Plan 05 demo scripts
- Phase 20 Plan 04 will calibrate LC_A_PRIOR_VARIANCE on synthetic RNNs and document whether 1/16 is appropriate or needs adjustment

---
*Phase: 20-latent-circuit-forward-model*
*Completed: 2026-05-24*
