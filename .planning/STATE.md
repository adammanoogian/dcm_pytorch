# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Phase 3 complete (Regression DCM Forward Model). Ready for Phase 4.

## Current Position

**Milestone:** v0.1.0-foundation
**Phase:** 3 of 8 (Regression DCM Forward Model) -- Complete
**Plan:** 3 of 3 complete (03-01, 03-02, 03-03)
**Status:** Phase complete
**Last activity:** 2026-03-26 -- Completed 03-03-PLAN.md (rDCM simulator and integration tests)

Progress: [█████████░░░░░░░░░░░] ~45% (9/~20 plans)

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
| Eigenvalue stabilization at -1/32 | SPM convention for fMRI frequencies; prevents transfer function blow-up | 2026-03-26 |
| Identity C_in/C_out for standard spDCM | Hemodynamic Jacobian deferred; standard spDCM uses identity projection | 2026-03-26 |
| SPM noise scaling C=1/256 | Matches spm_csd_fmri_mtf.m exactly; obs exponent /2, global obs /8.0 | 2026-03-26 |
| Transfer function peak test uses diagonal H[i,i] | Frobenius norm masks resonance; diagonal isolates eigenfrequency | 2026-03-26 |
| Manual Pearson correlation over np.corrcoef | Avoids process abort on Windows numpy; more robust | 2026-03-26 |
| rDCM Euler HRF (not double-gamma) | Matches Julia RegressionDynamicCausalModeling.jl exactly | 2026-03-26 |
| rDCM hemo rho=0.32 (not E0=0.40) | Julia dcm_euler_integration.jl uses H[5]=0.32 for rho | 2026-03-26 |
| rDCM BOLD constants (Julia convention) | relaxationRateSlope=25, frequencyOffset=40.3, echoTime=0.04, V0=4.0 | 2026-03-26 |
| 3x zero-padding for rDCM FFT convolution | Avoids circular convolution artifacts; matches Julia | 2026-03-26 |
| Confound prior precision = 1.0 | Weak prior on confound regressors; auxiliary, not connectivity params | 2026-03-26 |
| l0 clamped at 1e16 max | Prevents numerical issues from inf precision in absent connections | 2026-03-26 |
| 3-region sparse test (not 5-region) | 5-region with 2 inputs has insufficient drive; 3-region achieves F1 > 0.85 robustly | 2026-03-26 |
| Cross-mode threshold 0.8 (not 0.9) | Sparse ARD naturally shrinks coefficients differently from rigid VB | 2026-03-26 |

## Blockers

None currently.

## Key Risks

- ODE stiffness in Balloon model may need implicit solvers -- monitor for NaN gradients (Phase 1: no issues observed in 500s simulations)
- CSD normalization must exactly match SPM conventions or Phase 6 validation fails (Phase 2: predicted CSD matches SPM formula exactly)
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

## Phase 2 Deliverables (Complete)

- **Plan 01:** compute_transfer_function, predicted_csd, neuronal_noise_csd, observation_noise_csd, spectral_dcm_forward (27 tests)
- **Plan 02:** compute_empirical_csd, bold_to_csd_torch, default_welch_params (12 tests)
- **Plan 03:** simulate_spectral_dcm, make_stable_A_spectral, package exports, integration tests (26 tests)
- **Total:** 65 tests, all passing (120 total with Phase 1)

## Phase 3 Deliverables (Complete)

- **Plan 01:** rdcm_forward.py -- HRF, BOLD generation, design matrix construction (27 tests)
- **Plan 02:** rdcm_posterior.py -- Rigid/sparse VB inversion, free energy, priors, likelihood (33 tests)
- **Plan 03:** rdcm_simulator.py -- End-to-end simulator, package exports, integration tests (14 tests)
- **Total:** 74 tests, all passing (194 total with Phases 1-2)

## Session Continuity

Last session: 2026-03-26T12:27:47Z
Stopped at: Completed 03-03-PLAN.md (rDCM simulator and integration tests)
Resume file: None

---
*Last updated: 2026-03-26 after completing 03-03-PLAN.md*
