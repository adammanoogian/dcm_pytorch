---
phase: 01-neural-hemodynamic-forward-model
verified: 2026-03-25T22:04:53Z
status: passed
score: 10/10 must-haves verified
---

# Phase 1: Neural Hemodynamic Forward Model Verification Report

**Phase Goal:** Build the complete task-based DCM forward model pipeline from neural state equation through hemodynamics to BOLD signal plus a simulator that generates realistic synthetic data.
**Verified:** 2026-03-25T22:04:53Z
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Balloon-Windkessel ODE integrates without numerical instability for 500s | VERIFIED | test_500s_stability passes: no NaN/Inf, max state < 100 |
| 2 | Neural state dx/dt = Ax + Cu stable trajectories for stable A | VERIFIED | test_neural_state_stability and test_neural_state_stable_trajectory both pass finite bounded output |
| 3 | BOLD signal produces realistic percent signal change 0.5-5% | VERIFIED | test_bold_output_realistic_range and test_simulator_bold_range pass |
| 4 | ODE integrator supports Euler RK4 Dopri5 via torchdiffeq | VERIFIED | test_solver_selection_euler less than 1% error and test_solver_selection_rk4 less than 0.1% error pass |
| 5 | Simulator generates N-region BOLD given A C u hemo_params SNR | VERIFIED | test_simulator_500s test_simulator_5region test_simulator_output_shapes all pass |
| 6 | A matrix parameterization guarantees negative self-connections | VERIFIED | test_parameterize_A_always_negative_diagonal passes; code: A[diag_mask] = -torch.exp(A_free[diag_mask]) / 2.0 |
| 7 | Balloon-Windkessel uses SPM12 log-space convention lnf lnv lnq | VERIFIED | derivatives() operates in log-space; steady-state and known-value tests pass at 1e-10 |
| 8 | BOLD signal is zero at steady state v=1 q=1 | VERIFIED | test_steady_state_zero_bold passes with atol=1e-15 |
| 9 | Simulator SNR within 20 percent of requested value | VERIFIED | test_simulator_snr passes: empirical SNR within 8-12 for requested SNR=10 |
| 10 | All modules have future annotations and cite REF-XXX Eq. N | VERIFIED | All 17 source/test files contain future import; all 5 forward model files contain REF-001 or REF-002 citations |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Lines | Status |
|----------|-------|--------|
| src/pyro_dcm/forward_models/neural_state.py | 100 | VERIFIED - NeuralStateEquation + parameterize_A; imported and tested |
| src/pyro_dcm/forward_models/balloon_model.py | 165 | VERIFIED - BalloonWindkessel log-space ODE; imported and tested |
| src/pyro_dcm/forward_models/bold_signal.py | 72 | VERIFIED - bold_signal Buxton-form; imported and tested |
| src/pyro_dcm/forward_models/coupled_system.py | 138 | VERIFIED - CoupledDCMSystem nn.Module; imported by simulator and tests |
| src/pyro_dcm/utils/ode_integrator.py | 229 | VERIFIED - PiecewiseConstantInput integrate_ode make_initial_state; imported and tested |
| src/pyro_dcm/simulators/task_simulator.py | 393 | VERIFIED - simulate_task_dcm make_block_stimulus make_random_stable_A; imported and tested |
| pyproject.toml | -- | VERIFIED - hatchling build torch/pyro/torchdiffeq deps; package installs as pyro_dcm 0.1.0 |
| tests/test_neural_state.py | -- | VERIFIED - 8/8 pass |
| tests/test_balloon.py | -- | VERIFIED - 7/7 pass |
| tests/test_bold_signal.py | -- | VERIFIED - 6/6 pass |
| tests/test_ode_integrator.py | -- | VERIFIED - 16/16 pass |
| tests/test_task_simulator.py | -- | VERIFIED - 18/18 pass |

---

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| neural_state.py | A matrix parameterization | -torch.exp(A_free[diag_mask]) / 2.0 line 44 | WIRED |
| balloon_model.py | SPM12 log-space convention | lnf clamped at -14 chain-rule through exp | WIRED |
| coupled_system.py | neural_state.py | self.neural = NeuralStateEquation line 86 | WIRED |
| coupled_system.py | balloon_model.py | self.hemo = BalloonWindkessel lines 89-91 | WIRED |
| coupled_system.py | A/C as register_buffer | register_buffer calls lines 79-80 | WIRED |
| ode_integrator.py | torchdiffeq | from torchdiffeq import odeint odeint_adjoint line 19; jump_t line 167 | WIRED |
| task_simulator.py | coupled_system.py | CoupledDCMSystem instantiated line 158 | WIRED |
| task_simulator.py | bold_signal.py | bold_signal called line 192 | WIRED |
| task_simulator.py | ode_integrator.py | integrate_ode called line 168 | WIRED |

---

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| FWD-01: Balloon-Windkessel hemodynamic model as standalone ODE module | SATISFIED | BalloonWindkessel in balloon_model.py 165 lines; 7 unit tests passing |
| FWD-02: Neural state equation dx/dt = Ax + Cu | SATISFIED | NeuralStateEquation + parameterize_A in neural_state.py; 8 unit tests passing |
| FWD-03: BOLD signal equation | SATISFIED | bold_signal function in bold_signal.py; 6 unit tests including differentiability |
| FWD-04: ODE integrator wrapper with configurable solver | SATISFIED | integrate_ode supports dopri5 rk4 euler; PiecewiseConstantInput; 16 integration tests passing |
| SIM-01: Data simulator for task-based DCM | SATISFIED | simulate_task_dcm make_block_stimulus make_random_stable_A; 18 validation tests passing |

---

### Phase Success Criteria Verification

| Criterion | Status | Test Evidence |
|-----------|--------|---------------|
| SC1: Balloon-Windkessel ODE integrates without instability for 500s | PASSED | test_500s_stability: no NaN/Inf max_val < 100 |
| SC2: Neural dx/dt = Ax + Cu stable trajectories for stable A | PASSED | test_neural_state_stable_trajectory: finite max < 10 decay during rest |
| SC3: BOLD produces realistic percent signal change 0.5-5% | PASSED | test_bold_output_realistic_range and test_simulator_bold_range |
| SC4: ODE integrator supports Euler RK4 Dopri5 | PASSED | test_solver_selection_euler <1% test_solver_selection_rk4 <0.1% |
| SC5: Simulator generates N-region BOLD given A C u hemo_params SNR | PASSED | test_simulator_500s test_simulator_5region test_simulator_output_shapes |

---

### Anti-Patterns Found

None. Grep over all src/ files found zero TODO/FIXME/placeholder/stub patterns, empty returns, or missing implementations.

---

### Test Execution Summary

- tests/test_neural_state.py: 8 passed
- tests/test_balloon.py: 7 passed
- tests/test_bold_signal.py: 6 passed
- tests/test_ode_integrator.py: 16 passed (15.21s -- includes 500s stability simulation)
- tests/test_task_simulator.py: 18 passed (31.40s -- includes 500s end-to-end simulation)

Total: 55 passed, 0 failed, 0 skipped

---

### Notable Implementation Details Verified

1. torchdiffeq API: integrate_ode uses jump_t (not deprecated grid_points) internally. Public grid_points parameter preserved for callers.

2. SPM12 code defaults confirmed: DEFAULT_HEMO_PARAMS kappa=0.64 gamma=0.32 tau=2.0 alpha=0.32 E0=0.40. Discrepancy with Stephan 2007 paper values documented in source.

3. Log-space clamping: lnf >= -14 before exp; f >= 1e-6 before oxygen extraction. Verified by test_oxygen_extraction_clamp.

4. A and C stored as register_buffer in CoupledDCMSystem not nn.Parameter for Pyro compatibility in Phase 4.

5. BOLD constants k1=2.8 k2=2.0 k3=0.6 with E0=0.40 verified by test_bold_constants.

---

### Human Verification Required

None. All phase success criteria are covered by automated tests with measurable numerical thresholds.

---

_Verified: 2026-03-25T22:04:53Z_
_Verifier: Claude (gsd-verifier)_
