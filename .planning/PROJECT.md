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

- Balloon-Windkessel hemodynamic forward model with torchdiffeq ODE integration — v0.1.0
- **Linear** neural state equation (dx/dt = Ax + Cu) with explicit A matrix — v0.1.0
- BOLD signal equation mapping hemodynamic states to observations — v0.1.0
- Cross-spectral density computation matching SPM conventions — v0.1.0
- Spectral DCM transfer function H(w) and predicted CSD — v0.1.0
- Regression DCM analytic frequency-domain likelihood — v0.1.0
- Pyro generative models for all three DCM variants with proper priors — v0.1.0
- Mean-field Gaussian variational guides (Laplace baseline) — v0.1.0
- Data simulators for all three variants with realistic SNR — v0.1.0
- Parameter recovery tests: RMSE < 0.05, calibrated 95% CI coverage — v0.1.0
- Cross-validated against SPM12 and tapas/rDCM reference implementations — v0.1.0
- ELBO-based Bayesian model comparison across connectivity architectures — v0.1.0
- Neural amortized guides (normalizing flows) for task and spectral DCM — v0.1.0
- Amortized guide accuracy within 2x of per-subject SVI — v0.1.0
- Comprehensive benchmark suite comparing all inference methods — v0.1.0
- 6 SVI guide types (AutoDelta, AutoNormal, AutoLowRankMVN, AutoMVN, AutoIAF, AutoLaplace) — v0.2.0
- 3 ELBO objectives (Trace_ELBO, TraceMeanField_ELBO, RenyiELBO) with compatibility enforcement — v0.2.0
- Shared fixture generation for reproducible benchmarks (3 variants x 3 sizes) — v0.2.0
- Real ELBO-based amortization gap metric — v0.2.0
- Multi-level coverage calibration curves (50%, 75%, 90%, 95%) per guide type — v0.2.0
- Tiered calibration sweep orchestrator with resume support — v0.2.0
- 9 publication-quality figure types (calibration, scaling, comparison, violin, Pareto, timing) — v0.2.0
- Practical recommendation guide with Mermaid decision tree — v0.2.0
- Benchmark narrative with zero TBD entries — v0.2.0

### Active

## Current Milestone: v0.3.0 Bilinear DCM Extension

**Goal:** Extend the neural state equation from the linear form `dx/dt = Ax + Cu` to
the full bilinear form `dx/dt = Ax + Σ_j u_j·B_j·x + Cu` (Friston, Harrison & Penny
2003, Eq. 1), propagating B-matrix modulatory inputs through the forward model,
Pyro generative model + priors, simulator, and recovery benchmark.

**Target features:**

- Bilinear neural state equation with `compute_effective_A(A, B_list, u_mod)`
- Bilinear `CoupledDCMSystem` accepting `B_list` and a modulatory input interpolant
- Pyro model sampling `B_free_j ~ N(0, 1/16)` per modulator with per-modulator masking
- Variable-amplitude event/epoch stimulus utilities (stick & boxcar)
- Bilinear simulator accepting `B_list` + `stimulus_mod`
- Bilinear recovery benchmark (3-region, 1 driving + 1 modulatory input)

**Explicitly deferred out of v0.3.0:**

- Group-level PEB-lite GLM (HEART2ADAPT-specific; not scoped to this single-subject toolbox)
- 4-node HEART2ADAPT circuit benchmark (study-specific)
- SPM12 cross-validation of bilinear DCM (requires MATLAB; v0.4+ candidate)
- NumPyro backends, regularization study, semi-amortized pipeline, amortized calibration (deferred to v0.4.0+)

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
- **Python 3.10+**: Modern type hints, src/ layout with pyproject.toml
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
| v0.3.0 scoped to bilinear only | Keeps milestone focused and shippable; HEART2ADAPT/PEB/SPM12 extensions land in v0.4+ | -- Pending |

---
*Last updated: 2026-04-17 after v0.3.0 milestone started*
