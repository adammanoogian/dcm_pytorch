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

## Next Milestone: v0.7.0 — VL Validation + Real-Data Application (proposed)

**Goal:** Promote the v0.6.0 deliverables from synthetic-validated to real-data-validated, and
formalize the Variational Laplace engine with a systematic validation matrix. Seeded by the
v0.6.0 audit's deferred items and `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`.

**Candidate scope (to refine in `/gsd:new-milestone`):**

- **VL validation matrix** — recovery × N × SNR, SPM12 cross-check, calibration from the full
  covariance (Phase B of the draft).
- **VL + BMR model comparison** — relative-evidence ranking + separation gap; fix absolute-ΔF
  via posterior tempering (todo `vl-overconfidence-for-bmr`).
- **Real Cam-CAN M/EEG interpretability** — the deferred Phase 22 real-data gates (real training,
  source-localized ROIs); requires `camcan_loader.py` + DUA access.
- **Real foundation-model runs** — execute TRIBE v2 / LaBraM / BrainOmni extractors on real
  weights + cross-modal comparison (needs A100; first validate the parcellation nilearn path).
- **SBI reconciliation** — fix SBC structural calibration (stable-region prior / reparam) OR
  scope SBI as an optional speed-up benchmarked against calibrated VL.

**Infra prerequisite:** fix the Mutagen `models/` ignore (recreate session with anchored
ignores) before any M3 run touching `src/pyro_dcm/models/` — todo `mutagen-models-ignore`.

**Explicitly deferred beyond v0.7.0:**

- Neural ODE extension (Approach 2; separate milestone).
- nn4psych actor-critic networks (behavioral, not neural data).
- Learned `C_obs` for latent-circuit DCM (fixed at identity through v0.6.0).

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
*Last updated: 2026-06-10 after v0.6.0 milestone shipped (scope-cut). Validated section gained
v0.4.0/v0.5.0/v0.6.0 deliverables; next-milestone goals set to v0.7.0 (VL validation + real-data).
Note: v0.3.0 RECOV (Phase 16.1) remains genuinely incomplete and is still listed as the current
in-progress milestone above.*
