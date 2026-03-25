# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Phase 1 — Neural & Hemodynamic Forward Model + Task-DCM Simulator

## Current Position

**Milestone:** v0.1.0-foundation
**Phase:** 1 of 8 (Neural & Hemodynamic Forward Model)
**Plan:** 2 of 3 in phase (01-01 and 01-02 complete)
**Status:** In progress
**Last activity:** 2026-03-25 — Completed 01-02-PLAN.md

Progress: [██░░░░░░░░░░░░░░░░░░] ~10% (2/~20 plans)

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

## Blockers

None currently.

## Key Risks

- ODE stiffness in Balloon model may need implicit solvers — monitor for NaN gradients (Phase 1)
- CSD normalization must exactly match SPM conventions or Phase 6 validation fails (Phase 2)
- Amortized guide may struggle with multi-modal posteriors in weakly identifiable configs (Phase 7)

## Architecture Notes

Three swappable module interfaces:

1. **ConnectivityPrior**: `StaticA` (v0.1) | `GPPriorA` | `SwitchingA` | `RNNPriorA` (v0.2)
2. **ObservationModel**: `BalloonBOLD` (task) | `SpectralCSD` (spDCM) | `FreqDomainLinear` (rDCM)
3. **InferenceGuide**: `MeanFieldGaussian` (baseline) | `NormalizingFlowGuide` (amortized)

## Session Continuity

Last session: 2026-03-25T21:47:33Z
Stopped at: Completed 01-02-PLAN.md
Resume file: None

---
*Last updated: 2026-03-25 after completing 01-02-PLAN.md*
