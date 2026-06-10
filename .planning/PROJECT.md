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

## Current State

**Shipped v0.6.0 (2026-06-10, scope-cut).** The framework now does DCM-based interpretability
of deep-learning models trained on neural data, with an SPM12-grade **Variational Laplace**
engine (`spm_nlsi_GN`: Gauss-Newton E-step, ReML M-step, SVD reduction, full posterior
covariance) as the default inference, generalized across spectral/task/latent-circuit DCM via a
`ForwardModel` protocol. Synthetic parameter recovery, Bayesian Model Reduction, and hybrid
VAE-DCM are validated; real-data application (real Cam-CAN M/EEG, real foundation-model runs,
SBI calibration) is built-but-un-run and **deferred to v0.7.0**. See
`.planning/MILESTONES.md` and `milestones/v0.6.0-MILESTONE-AUDIT.md`.

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
- Circuit Explorer JSON serializer (`CircuitViz`) for DCM configs + fitted posteriors — v0.4.0
- MNE/BIDS IO test suite + spectral & task DCM end-to-end pipeline demos (synthetic) — v0.5.0
- Direct-observation latent-circuit DCM + synthetic recovery (A-RMSE 0.026, B-RMSE 0.0048, pooled-R² 0.961) — v0.6.0
- **Variational Laplace inference engine** (SPM12 `spm_nlsi_GN`; full posterior covariance) as DCM default — v0.6.0
- `ForwardModel` protocol generalizing VL across spectral / task / latent-circuit DCM — v0.6.0
- Bayesian Model Reduction (Friston & Penny 2011), validated vs brute-force ELBO (~93× faster) — v0.6.0
- Hybrid VAE-DCM amortized inference (A-RMSE 0.076, masked sign recovery 0.77, 0.76 ms) — v0.6.0
- CT-RNN training + PCA latent-extraction baseline — v0.6.0
- Real foundation-model extractors (TRIBE v2 / LaBraM / BrainOmni) + real Schaefer parcellation *(infra; un-run)* — v0.6.0

### Active

## Paused Milestone: v0.3.0 Bilinear DCM Extension (Phase 16.1 incomplete)

> Phases 13-15 complete; Phase 16 acceptance failed (RECOV-04 B-RMSE) and the 16.1 diagnostic
> was never executed. The VL engine (v0.6.0) plausibly fixes this B-collapse cheaply — revisit
> after v0.7.0 if a clean v0.3.0 close is wanted. Not blocking.

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

## Current Milestone: v0.7.0 — Variational Laplace Validation (VL-validation-led)

**Goal:** Prove the Variational Laplace engine works *completely* on synthetic / known-ground-truth
problems before any real data. Establish a systematic validation matrix, cross-validate against
SPM12 `spm_nlsi_GN`, standardize VL+BMR model comparison, and harden numerical robustness. No
real data this milestone. Seed: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md` + v0.6.0 audit.

**Target features (confirmed scope):**

- **Synthetic recovery matrix (N × SNR)** — systematic parameter recovery across network sizes
  and noise, for spectral / task / latent-circuit forward models. CI coverage reported as a
  standard recovery metric (calibration rides along; not a separate SBC phase).
- **SPM12 cross-validation** — numeric agreement of VL output vs MATLAB `spm_nlsi_GN` on identical
  problems (prior-matched). Builds on the v0.1.0 SPM12 `.mat` export + MATLAB batch infrastructure.
- **VL + BMR model comparison** — standardize BMR-on-VL (relative-evidence ranking + separation
  gap); fix the absolute-ΔF Laplace-overconfidence via posterior tempering (todo
  `vl-overconfidence-for-bmr`).
- **Numerical robustness / edge cases** — convergence, the dt/precision-matrix intractability
  (Phase 28 note), stability-boundary handling, multi-restart determinism.

**Explicitly deferred (NOT v0.7.0 — gated on VL being proven first):**

- All real-data application — real Cam-CAN M/EEG (Phase 22 gates), real foundation-model runs
  (Phase 24), real-M/EEG demos. → v0.8.0+ once VL is validated.
- SBI reconciliation / SBC structural fix — SBI is a separate (uncalibrated) inference path,
  not VL. → later milestone.
- Posterior calibration/coverage as a standalone SBC dimension (light CI-coverage only, within
  the recovery matrix).
- Neural ODE extension; learned `C_obs`; nn4psych actor-critic networks.

**Infra prerequisite:** fix the Mutagen `models/` ignore (recreate session with anchored ignores)
before any M3 run touching `src/pyro_dcm/models/` — todo `mutagen-models-ignore`.

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
| Direct observation for RNN latents | No balloon-Windkessel when fitting DCM to RNN hidden states; hemodynamic model only for real BOLD | ✓ Good — `LatentCircuitForward` + synthetic recovery (v0.6.0) |
| Neural data prediction RNN (not behavioral) | RNN trained to predict neural activity, not behavioral choices; DCM then distills learned neural dynamics | — Pending — pipeline synthetic-validated; real-data → v0.7.0 |
| Variational Laplace as default inference (SPM12 `spm_nlsi_GN`) | Full posterior covariance / structured posterior closes the B-collapse mean-field SVI couldn't; no AutoLowRankMVN/AutoIAF guide needed | ✓ Good — closed Phase 20-05 (v0.6.0) |
| Scope-cut v0.6.0 at completion | Ship synthetic methodology + VL engine; defer real-data application rather than block on data/compute access | ✓ Good — honest milestone close (v0.6.0) |

---
*Last updated: 2026-06-10 — started milestone v0.7.0 (Variational Laplace Validation,
VL-validation-led; no real data). Confirmed scope: synthetic recovery matrix (N×SNR), SPM12
cross-validation, VL+BMR comparison, numerical robustness. Real-data + SBI deferred to v0.8.0+.
v0.3.0 relabeled Paused (Phase 16.1 incomplete).*
