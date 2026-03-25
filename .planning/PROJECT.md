# Pyro-DCM: Neural Amortized Dynamic Causal Modeling

## What This Is

A modular, research-grade Python framework for Dynamic Causal Modeling (DCM) that combines
biophysically grounded generative models with modern amortized variational inference. The
framework targets three DCM variants — task-based DCM, spectral DCM (spDCM), and regression
DCM (rDCM) — each reimplemented from first principles with full mathematical fidelity, then
extended with neural network inference guides that preserve scientific interpretability.
Built with Pyro (PyTorch PPL), torchdiffeq, and Zuko normalizing flows.

## Core Value

The A matrix (effective connectivity) remains an explicit, interpretable object with full
posterior uncertainty throughout inference — never absorbed into a latent space, never a
point estimate. This is the scientific meaning that must be preserved above all else.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implement Balloon-Windkessel hemodynamic forward model with torchdiffeq ODE integration
- [ ] Implement bilinear neural state equation (dx/dt = Ax + Cu) with explicit A matrix
- [ ] Implement BOLD signal equation mapping hemodynamic states to observations
- [ ] Implement cross-spectral density computation matching SPM conventions
- [ ] Implement spectral DCM transfer function H(w) and predicted CSD
- [ ] Implement regression DCM analytic frequency-domain likelihood
- [ ] Define Pyro generative models for all three DCM variants with proper priors
- [ ] Implement mean-field Gaussian variational guides (Laplace baseline)
- [ ] Build data simulators for all three variants with realistic SNR
- [ ] Parameter recovery tests: RMSE < 0.05, calibrated 95% CI coverage
- [ ] Cross-validate against SPM12 and tapas/rDCM reference implementations
- [ ] ELBO-based Bayesian model comparison across connectivity architectures
- [ ] Implement neural amortized guides (normalizing flows) for all variants
- [ ] Demonstrate amortized guide accuracy within 2x of per-subject NUTS
- [ ] Comprehensive benchmark suite comparing all inference methods

### Out of Scope

- Non-stationary A(t) extensions — deferred to v0.2, requires separate contribution
- Neural ODE replacements for biophysical forward model — deferred pending Nozari et al. (2024) evidence
- Clinical deployment or real-time processing — research tool only
- GUI or web interface — CLI/API only
- Multi-modal (EEG/MEG) DCM — different observation models entirely
- Structural connectivity integration (tractography priors) — future work

## Context

- Replaces SPM's MATLAB-only DCM with Python/Pyro implementation
- Three DCM variants: task-based (BOLD time series), spectral (resting-state CSD), regression (scalable frequency-domain)
- Every equation traces to a specific paper — see .planning/REFERENCES.md
- Bilinear model justified by Nozari et al. (2024): linear models suffice for macroscopic BOLD
- Architecture designed for swappable components: connectivity priors, observation models, inference guides
- Follows project_utils conventions: src/ layout, NumPy-style docstrings, ruff/mypy, pytest

## Constraints

- **Tech stack**: PyTorch + Pyro (required for SVI + neural guides with explicit generative model)
- **ODE solver**: torchdiffeq (PyTorch-native, adjoint method for memory efficiency)
- **Flow library**: Zuko (Pyro-compatible normalizing flows)
- **Mathematical fidelity**: Every function must cite [REF-XXX] and equation number from REFERENCES.md
- **No placeholders**: Every function computes real mathematics — no pass, no TODO stubs
- **Python 3.11+**: Modern type hints, src/ layout with pyproject.toml
- **Coding standards**: project_utils CODING_STANDARDS.md (NumPy docstrings, ruff, mypy, 88-char lines)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pyro over sbi/BayesFlow | Need explicit generative model for ELBO model comparison | -- Pending |
| torchdiffeq over diffrax | PyTorch native, adjoint method, proven ecosystem | -- Pending |
| Zuko over nflows/normflows | Cleaner API, actively maintained, Pyro-compatible | -- Pending |
| Bilinear over Neural ODE | Nozari 2024: linear suffices for macroscopic BOLD; v0.2 extension | -- Pending |
| src/ layout | project_utils standard, prevents import confusion | -- Pending |
| Static A first | Clean first paper; non-stationary A(t) is second contribution | -- Pending |
| NumPyro for NUTS only | JAX speed for validation sampling, not primary inference | -- Pending |

---
*Last updated: 2026-03-25 after initialization*
