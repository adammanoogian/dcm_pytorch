# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Phase 1 — Neural & Hemodynamic Forward Model + Task-DCM Simulator

## Current Position

**Milestone:** v0.1.0-foundation
**Phase:** 1 (not started)
**Status:** Ready to plan Phase 1

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

---
*Last updated: 2026-03-25 after initialization*
