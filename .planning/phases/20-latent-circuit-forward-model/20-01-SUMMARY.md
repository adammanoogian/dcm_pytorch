---
phase: 20
plan: 01
subsystem: latent-circuit-forward-model
tags: [latent-circuit, ode, bilinear, hemodynamic-toggle, simulator, v0.6.0]

dependency-graph:
  requires:
    - "phases 1-14: coupled ODE system with bilinear neural state equation"
    - "phase 13: CoupledDCMSystem buffer/bilinear infrastructure"
  provides:
    - "CoupledDCMSystem(hemodynamic=False): N-dim latent-circuit ODE mode"
    - "simulate_latent_circuit(): end-to-end latent trajectory generator"
    - "make_stable_latent_circuit_A(): helper for stable DCM connectivity"
  affects:
    - "phase 20 plans 02-05: all downstream latent-circuit inference uses these primitives"
    - "phase 21: PCA extraction pipeline will feed trajectories_clean to this simulator"

tech-stack:
  added: []
  patterns:
    - "hemodynamic toggle: bool flag on CoupledDCMSystem, default=True for bit-exact backward compat"
    - "latent-circuit simulator mirrors task_simulator.py: _normalize_B_list reused, same bilinear path"
    - "initial state torch.zeros(N) not make_initial_state (5N) -- latent circuit has no hemo state"

key-files:
  created:
    - src/pyro_dcm/simulators/latent_circuit_simulator.py
    - tests/test_latent_circuit_forward.py
  modified:
    - src/pyro_dcm/forward_models/coupled_system.py
    - src/pyro_dcm/simulators/__init__.py

decisions:
  - id: "20-01-D1"
    description: "hemodynamic=False passed as keyword-only after stability_check_every to avoid positional ambiguity with hemo_params dict"
    rationale: "Preserves backward compat: all existing callers omit it; no positional breaks possible"
  - id: "20-01-D2"
    description: "simulate_latent_circuit reuses _normalize_B_list and _normalize_stimulus_to_input_fn from task_simulator.py (no duplication)"
    rationale: "DRY: the bilinear path is identical; private helpers imported directly to avoid copy-paste divergence"
  - id: "20-01-D3"
    description: "Initial state is torch.zeros(N) not make_initial_state (which returns 5N zeros)"
    rationale: "Latent-circuit state has no hemodynamic components; make_initial_state returns wrong shape for hemodynamic=False"
  - id: "20-01-D4"
    description: "make_stable_latent_circuit_A uses self_inhibition=1.0 Hz (vs 0.5 Hz SPM12 default)"
    rationale: "RNN latent states evolve faster than BOLD; stronger self-inhibition appropriate for latent-circuit timescale"

metrics:
  duration: "9 minutes"
  completed: "2026-05-24"
---

# Phase 20 Plan 01: Latent Circuit Forward Model Summary

**One-liner:** Hemodynamic toggle on CoupledDCMSystem + N-dim latent-circuit simulator with bilinear support.

## What Was Built

Added `hemodynamic: bool = True` keyword-only parameter to `CoupledDCMSystem.__init__()`. When `hemodynamic=False`, the module skips `BalloonWindkessel` construction entirely; the ODE state is neural activity `x` of shape `(N,)` and `forward()` returns `(N,)`. Full bilinear support (`B_list`, `n_driving_inputs`) is available in this mode. When `hemodynamic=True` (the default), behavior is bit-exact to pre-v0.6.0 callers -- all 15 pre-existing tests pass unchanged.

Created `src/pyro_dcm/simulators/latent_circuit_simulator.py` implementing:
- `simulate_latent_circuit()`: integrates `CoupledDCMSystem(hemodynamic=False)` over a fine time grid, adds Gaussian noise at the requested SNR, returns a dict with `'trajectories'` shape `(T, N)` and no hemodynamic or BOLD keys.
- `make_stable_latent_circuit_A()`: random sparse connectivity with `self_inhibition=1.0 Hz` diagonal, suitable for RNN latent timescales.

Updated `simulators/__init__.py` to re-export both new symbols.

Created `tests/test_latent_circuit_forward.py` with 9 tests covering all 6 must-have truths from the plan.

## Verification

All 6 must-have truths confirmed:
1. `CoupledDCMSystem(hemodynamic=False)` returns shape `(N,)` -- verified by `test_hemodynamic_false_returns_N_derivatives`.
2. `hemodynamic=True` is bit-exact to no-kwarg construction -- verified by `test_hemodynamic_true_bit_exact` using `torch.equal`.
3. `simulate_latent_circuit` returns `'trajectories'` shape `(T, N)` -- verified.
4. No `'bold'`, `'bold_clean'`, `'bold_fine'`, `'hemodynamic'`, `'neural'` keys -- verified.
5. Simulator matches direct ODE integration at `atol=1e-6` (rk4, same dt) -- verified.
6. Bilinear trajectories differ from linear by > 1e-4 -- verified.

Test run: **9/9 passed** in 1.77s.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 20-01-D1 | `hemodynamic=False` as keyword-only after `stability_check_every` | No positional break for existing callers |
| 20-01-D2 | Reuse `_normalize_B_list` / `_normalize_stimulus_to_input_fn` from task_simulator | DRY; bilinear path is identical |
| 20-01-D3 | Initial state `torch.zeros(N)` not `make_initial_state` (5N) | Wrong shape for no-hemo mode |
| 20-01-D4 | `self_inhibition=1.0 Hz` in `make_stable_latent_circuit_A` | RNN latent timescale faster than BOLD |

## Deviations from Plan

None -- plan executed exactly as written.

## Next Phase Readiness

- Phase 20 Plan 02 (prior recalibration) can now use `simulate_latent_circuit` to generate training data for prior calibration.
- Phase 20 Plans 03-05 (Pyro model, SVI fitting, recovery tests) depend on this simulator.
- The `CoupledDCMSystem(hemodynamic=False)` ODE is the forward model for all Phase 20+ latent-circuit inference.
