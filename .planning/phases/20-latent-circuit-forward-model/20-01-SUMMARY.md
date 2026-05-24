---
phase: 20-latent-circuit-forward-model
plan: 01
subsystem: forward-models
tags: [ode, bilinear, simulator, latent-circuit, direct-observation]
depends_on: []
provides:
  - CoupledDCMSystem hemodynamic toggle (hemodynamic=True/False)
  - simulate_latent_circuit function
  - make_stable_latent_circuit_A helper
affects:
  - 20-02 (Pyro generative model uses hemodynamic=False)
  - 20-03 (synthetic validation uses simulate_latent_circuit)
  - 20-04 (recovery benchmark uses simulator)
tech-stack:
  added: []
  patterns:
    - Toggle-based mode extension (hemodynamic bool flag)
    - Shared bilinear/linear branching across modes
key-files:
  created:
    - src/pyro_dcm/simulators/latent_circuit_simulator.py
    - tests/test_latent_circuit_forward.py
  modified:
    - src/pyro_dcm/forward_models/coupled_system.py
    - src/pyro_dcm/simulators/__init__.py
decisions:
  - hemodynamic=False branch in forward() duplicates bilinear logic rather than refactoring (preserves bit-exact hemodynamic=True path)
  - SNR computed as mean signal std / noise std (averaged across regions)
  - make_stable_latent_circuit_A auto-increases self-inhibition if initially unstable
metrics:
  duration: 22 min
  completed: 2026-05-24
---

# Phase 20 Plan 01: Forward Model & Simulator Summary

**One-liner:** CoupledDCMSystem hemodynamic toggle + latent circuit simulator producing (T, N) neural trajectories from bilinear ground truth.

## What Was Built

### CoupledDCMSystem hemodynamic toggle (Task 1)

Extended `CoupledDCMSystem` with `hemodynamic: bool = True` keyword-only parameter:

- **hemodynamic=True (default):** Bit-exact pre-Phase-20 behavior. State is (5N,), BalloonWindkessel constructed, forward() returns 5N derivatives.
- **hemodynamic=False:** State is (N,) neural-only. No BalloonWindkessel. forward() returns N-dim dx directly. Same bilinear/linear branching logic.
- **ValueError** raised if `hemo_params` given with `hemodynamic=False`.

### Latent circuit simulator (Task 2)

New `simulate_latent_circuit()` function:

- Uses `CoupledDCMSystem(hemodynamic=False)` for N-dim ODE integration.
- Returns dict with `trajectories` (T, N), `trajectories_clean` (T, N), `times`, ground-truth params.
- Supports bilinear mode (`B_list` + `stimulus_mod`).
- Gaussian noise added post-integration at specified SNR.
- Initial state is `zeros(N)` (not 5N `make_initial_state`).

New `make_stable_latent_circuit_A()` helper:

- Generates sparse stable A matrices with configurable density and self-inhibition.
- Guarantees stability via eigenvalue check with progressive diagonal strengthening.

## Verification Results

- 10 new tests all passing (test_latent_circuit_forward.py)
- 20+ existing tests passing (neural_state, balloon, coupled_system_bilinear, task_dcm_recovery)
- Zero edits to neural_state.py, balloon_model.py, bold_signal.py (OBS-04)
- Simulator-vs-ODE cross-validation passes at atol=1e-6 (SIM-02)
- Bilinear mode produces trajectories distinguishable from linear

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Duplicate bilinear logic in hemodynamic=False branch | Preserves bit-exact hemodynamic=True path per plan constraint |
| SNR = mean(signal_std per region) / noise_std | Consistent global noise level across regions |
| Auto-stabilize A in make_stable_latent_circuit_A | Prevents unstable ground truth that would produce divergent trajectories |

## Deviations from Plan

None -- plan executed exactly as written.

## Next Phase Readiness

Plan 20-02 (Pyro generative model) can proceed. The forward model foundation is in place:
- `CoupledDCMSystem(hemodynamic=False)` integrates N-dim latent states
- `simulate_latent_circuit()` generates ground-truth data for synthetic validation
- All interfaces tested and documented
