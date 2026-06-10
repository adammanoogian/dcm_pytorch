---
phase: 29-vl-validation-infra-bmr-rank
plan: 04
subsystem: testing
tags: [variational-laplace, benchmarks, runner-registry, spectral-dcm, task-dcm, latent-circuit, smoke-tests]

# Dependency graph
requires:
  - phase: 29-01
    provides: BenchmarkConfig VL fields (max_iter, hyperprior_*, prior_mean_a_offset) + vl pytest marker
  - phase: 29-03
    provides: TaskDCMForward.build_precision dt>=0.1 intractability guard (_TASK_PRECISION_MAX_DIM=5000)
  - phase: 28
    provides: model-agnostic VL engine (run_variational_laplace_generic, extract_vl_posterior_generic) + Spectral/Task/LatentCircuit forward models
provides:
  - "run_spectral_vl / run_task_vl / run_latent_circuit_vl: (BenchmarkConfig)->dict VL runners"
  - "3 additive RUNNER_REGISTRY entries under method=vl"
  - "tests/test_vl_runners_smoke.py: N=2/1-seed laptop smoke suite (~113s)"
  - "TaskDCMForward.predict integrate_ode step_size fix (unblocks task VL)"
affects: [phase-30-recovery-sweep, recovery_validation, figure-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VL runners mirror the existing (BenchmarkConfig)->dict SVI runner contract for drop-in registry use"
    - "Smoke tests prove plumbing (dict shape + finite metric), not recovery quality; xfail escape on legit non-convergence"

key-files:
  created:
    - benchmarks/runners/spectral_vl.py
    - benchmarks/runners/task_vl.py
    - benchmarks/runners/latent_circuit_vl.py
    - tests/test_vl_runners_smoke.py
  modified:
    - benchmarks/runners/__init__.py
    - src/pyro_dcm/inference/forward_models.py

key-decisions:
  - "context={'freqs': freqs} passed to spectral VL (plan said {}); SpectralDCMForward.predict requires it"
  - "task_vl: t_eval at TR resolution + dt=0.1 internal RK4 step so predicted BOLD rows match observed and T*N stays << 5000"
  - "LC smoke uses max_iter=4 (slowest fit) to keep the 3-runner suite under the 3-min laptop budget"

patterns-established:
  - "VL runner return dict: variant/method/n_regions + per-dataset lists + nested summary + metadata"

# Metrics
duration: 38min
completed: 2026-06-10
---

# Phase 29 Plan 04: VL Benchmark Runners Summary

**Three method="vl" benchmark runners (spectral_vl, task_vl, latent_circuit_vl) fitting via run_variational_laplace_generic, registered additively in RUNNER_REGISTRY, with a 113s laptop smoke suite proving the (BenchmarkConfig)->dict plumbing (VLINFRA-02).**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 3
- **Files modified:** 6 (4 created, 2 modified)

## Accomplishments

- `run_spectral_vl` — `SpectralDCMForward` + VL; per-dataset A-RMSE / coverage / correlation / convergence. Smoke: RMSE 0.0023, converged in 13 iters (~4s at N=2).
- `run_task_vl` — `TaskDCMForward` + VL at `dt=0.1` with BOLD sampled at `TR=2`, short duration so `T*N` stays well under the 5000 precision cap (guard never trips). Smoke: RMSE 0.0235, converged.
- `run_latent_circuit_vl` — thin wrapper reusing `_build_ground_truth` + `LatentCircuitForward` + the proven `lc_vl_acceptance_run.py` fit block; per-seed A-RMSE + magnitude-masked B-RMSE. Smoke: A-RMSE 0.0398, B-RMSE 0.0174. Module docstring documents the Mutagen `models/` ignore → M3 prerequisite (todo `mutagen-models-ignore`).
- Three additive `RUNNER_REGISTRY` entries under `method="vl"`; all existing entries untouched.
- `tests/test_vl_runners_smoke.py` — 3 `@pytest.mark.vl` smoke tests + a registry-callable check; full vl suite green in **113s** on laptop CPU (budget ~3 min).

## Task Commits

1. **Task 1: spectral_vl + task_vl runners (+ integrate_ode fix)** - `372e203` (feat)
2. **Task 2: latent_circuit_vl runner** - `6a09579` (feat)
3. **Task 3: registry entries + smoke tests** - `a731fd5` (feat)

## Files Created/Modified

- `benchmarks/runners/spectral_vl.py` - `run_spectral_vl(config)`: spectral CSD VL fit + metrics.
- `benchmarks/runners/task_vl.py` - `run_task_vl(config)`: task BOLD VL fit at dt=0.1.
- `benchmarks/runners/latent_circuit_vl.py` - `run_latent_circuit_vl(config)`: latent-circuit VL fit; Mutagen caveat doc note.
- `tests/test_vl_runners_smoke.py` - N=2/1-seed smoke suite + registry check.
- `benchmarks/runners/__init__.py` - 3 additive vl registry entries; explicit `BenchmarkConfig` re-export.
- `src/pyro_dcm/inference/forward_models.py` - `TaskDCMForward.predict`: `integrate_ode(..., step_size=...)` (was the invalid `options=` kwarg).

## Decisions Made

- **Spectral context:** the plan specified `context={}`, but `SpectralDCMForward.predict` reads `context["freqs"]`. Passed `context={"freqs": freqs}` (the engine injects `a_mask` itself). Without this the fit raises `KeyError`.
- **Task time grid:** `t_eval` at TR resolution (output times) + `dt=0.1` as the internal RK4 step. This makes the predicted-BOLD row count equal the observed-BOLD rows and keeps `T*N` (= 30·2 quick) far below the 5000 cap, so the [29-03-D1] guard never trips.
- **LC smoke budget:** the LC fit (N=4, J=1, dense time-domain precision, 36 params) is the slowest — ~229s at max_iter=16. The smoke test uses `max_iter=4` (plumbing, not recovery), keeping the 3-runner suite at 113s. No cluster job launched (full N×SNR sweep is Phase 30/M3).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed TaskDCMForward.predict integrate_ode call**
- **Found during:** Task 1 (task_vl first run)
- **Issue:** `TaskDCMForward.predict` called `integrate_ode(system, y0, t_eval, method="rk4", options={"step_size": self._dt})`, but `integrate_ode` has no `options` parameter (it takes `step_size=` directly, as `LatentCircuitForward._integrate` already does). Task VL had never been exercised through a runner before, so this latent bug surfaced now. Raised `TypeError: integrate_ode() got an unexpected keyword argument 'options'`.
- **Fix:** Changed the call to `integrate_ode(system, y0, self._t_eval, method="rk4", step_size=self._dt)`.
- **Files modified:** src/pyro_dcm/inference/forward_models.py
- **Verification:** task_vl smoke now fits (RMSE 0.0235, converged); precision-guard + csd-roundtrip tests still pass.
- **Committed in:** 372e203 (Task 1 commit)

**2. [Rule 3 - Blocking] Spectral runner context must carry freqs**
- **Found during:** Task 1 (spectral_vl)
- **Issue:** Plan's `context={}` would raise `KeyError('freqs')` inside `SpectralDCMForward.predict`.
- **Fix:** Passed `context={"freqs": freqs}` from the simulated CSD frequencies.
- **Files modified:** benchmarks/runners/spectral_vl.py
- **Verification:** spectral_vl smoke converges (RMSE 0.0023).
- **Committed in:** 372e203 (Task 1 commit)

**3. [Rule 1 - Lint] Explicit re-export of pre-existing unused import**
- **Found during:** Task 3 (ruff on the touched `__init__.py`)
- **Issue:** A pre-existing `from benchmarks.config import BenchmarkConfig` (unused, present on baseline a064e69) tripped `F401` once the file was re-formatted.
- **Fix:** Converted to `from benchmarks.config import BenchmarkConfig as BenchmarkConfig` (ruff's suggested explicit re-export) to keep the file ruff-clean without removing a public name. No consumer imports it from the runners package.
- **Files modified:** benchmarks/runners/__init__.py
- **Verification:** `ruff check benchmarks/runners/__init__.py` clean.
- **Committed in:** a731fd5 (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 lint)
**Impact on plan:** All necessary to make the runners execute and the touched file lint-clean. No scope creep; registry additions stayed purely additive.

## Issues Encountered

- **Pre-existing (out of scope, NOT regressions):** `tests/test_vl_forward_model_protocol.py` has 5 task-DCM failures from `simulate_task_dcm()`/`make_block_stimulus` signature drift (`dt_sim`/`dt_model` kwargs). Confirmed identical on baseline a064e69; the prompt flagged these explicitly as not-my-concern. The 29-04 runners deliberately mirror the WORKING `task_svi.py` call signatures, not the stale test, and so are unaffected. The 29-03 deliverables (precision-guard, csd-roundtrip) still pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VLINFRA-02 complete: the `(variant, "vl")` runner trio is wired into `RUNNER_REGISTRY` and proven on laptop. Phase 30's N×SNR multi-seed recovery sweep can consume them unchanged.
- **M3 prerequisite (carried):** the unanchored Mutagen `models/` ignore excludes `src/pyro_dcm/models/` from cluster sync, so any M3 latent-circuit VL run needs the anchored-ignore fix first (todo `mutagen-models-ignore`). Spectral/task VL runners are unaffected (they live under the synced `inference/`).
- **Pre-existing cleanup carried:** the `test_vl_forward_model_protocol.py` task signature-drift failures remain a worthwhile separate cleanup pass.

---
*Phase: 29-vl-validation-infra-bmr-rank*
*Completed: 2026-06-10*
