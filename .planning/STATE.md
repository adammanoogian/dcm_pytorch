# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Phase 2 (Spectral DCM Forward Model) -- In progress

## Current Position

**Milestone:** v0.1.0-foundation
**Phase:** 2 of 8 (Spectral DCM Forward Model) -- In progress
**Plan:** 02-02 complete (parallel with 02-01)
**Status:** In progress
**Last activity:** 2026-03-26 -- Completed 02-02-PLAN.md (empirical CSD computation)

Progress: [████░░░░░░░░░░░░░░░░] ~20% (4/~20 plans)

## Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| Package name: pyro_dcm | Clear identification as Pyro-based DCM | 2026-03-25 |
| src/ layout | project_utils standard, prevents import confusion | 2026-03-25 |
| Pyro (not sbi/BayesFlow) | Need explicit generative model for ELBO model comparison | 2026-03-25 |
| torchdiffeq (not diffrax) | PyTorch native, adjoint method, proven ecosystem | 2026-03-25 |
| Zuko (not nflows) | Cleaner API, actively maintained, Pyro-compatible | 2026-03-25 |
| Bilinear model first | Nozari 2024: linear suffices; Neural ODE deferred to v0.2 | 2026-03-25 |
| Static A first | Clean first paper; non-stationary A(t) is second contribution | 2026-03-25 |
| NumPyro for NUTS only | JAX speed for validation sampling, not primary inference | 2026-03-25 |
| SPM12 code hemo defaults | kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40 (not paper values) | 2026-03-25 |
| Simplified Buxton BOLD form | k1=7*E0, k2=2, k3=2*E0-0.2, V0=0.02 with E0=0.40 | 2026-03-25 |
| Log-space clamping | lnf >= -14, f >= 1e-6 before oxygen extraction to prevent NaN | 2026-03-25 |
| A/C as register_buffer | Pyro handles parameterization in Phase 4, not nn.Parameter | 2026-03-25 |
| torchdiffeq jump_t for discontinuities | v0.2.5 renamed grid_points to jump_t; our API preserves grid_points name | 2026-03-25 |
| Per-region SNR noise scaling | noise_std = signal_std / SNR per region for accurate SNR control | 2026-03-25 |
| Simulator accepts parameterized A (not A_free) | Direct control over connectivity values in simulations | 2026-03-25 |
| Welch CSD (not MAR) for empirical CSD | Standard signal processing; SPM matching via predicted CSD model | 2026-03-26 |
| np.interp for frequency grid interpolation | Linear interp on real/imag separately; simple, effective for smooth CSD | 2026-03-26 |
| numpy/scipy for CSD data prep, torch at boundary | CSD computation is data prep, not differentiable model | 2026-03-26 |

## Blockers

None currently.

## Key Risks

- ODE stiffness in Balloon model may need implicit solvers -- monitor for NaN gradients (Phase 1: no issues observed in 500s simulations)
- CSD normalization must exactly match SPM conventions or Phase 6 validation fails (Phase 2)
- Amortized guide may struggle with multi-modal posteriors in weakly identifiable configs (Phase 7)

## Architecture Notes

Three swappable module interfaces:

1. **ConnectivityPrior**: `StaticA` (v0.1) | `GPPriorA` | `SwitchingA` | `RNNPriorA` (v0.2)
2. **ObservationModel**: `BalloonBOLD` (task) | `SpectralCSD` (spDCM) | `FreqDomainLinear` (rDCM)
3. **InferenceGuide**: `MeanFieldGaussian` (baseline) | `NormalizingFlowGuide` (amortized)

## Phase 1 Deliverables (Complete)

- **Plan 01:** NeuralStateEquation, BalloonWindkessel, bold_signal (21 tests)
- **Plan 02:** CoupledDCMSystem, PiecewiseConstantInput, integrate_ode (16 tests)
- **Plan 03:** simulate_task_dcm, make_block_stimulus, make_random_stable_A (18 tests)
- **Total:** 55 tests, all passing

## Phase 2 Deliverables (In Progress)

- **Plan 02:** compute_empirical_csd, bold_to_csd_torch, default_welch_params (12 tests)

## Session Continuity

Last session: 2026-03-26T07:16:42Z
Stopped at: Completed 02-02-PLAN.md (empirical CSD computation)
Resume file: None

---
*Last updated: 2026-03-26 after completing 02-02-PLAN.md*
