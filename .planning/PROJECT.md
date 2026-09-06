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

**v0.8.0 DCM for Evoked Responses (EEG/MEG ERP) complete (2026-06-26).** Pyro-DCM now has a full
time-domain ERP forward stack — a canonical-microcircuit (CMC) neural-mass model → extrinsic
laminar coupling + condition modulation + evoked integration (`spm_int_L` exp-Euler port) →
single-dipole lead-field → scalp ERP — **SPM12-parity-verified at every stage** (Phases 33-36, all
gsd-verifier-passed; 79 ERP tests green on M3). `ERPDCMForward` plugs into the existing
`ForwardModel` protocol so VL + amortized inference came for free. The headline artifact — a
5-source MMN precision-sweep demo showing monotone superficial-pyramidal-gain → attenuated-MMN (the
Adams/Ranlund mechanism the `actinf_physics` consumer imports) — ships gated behind the SPM parity
check. **Deferred (Phase 37, follow-up):** frontal-dominant *scalp* topography, which is an ECD
dipole-orientation phenomenon not recoverable in LFP-identity readout — needs a sensor montage +
verified MNI coords. Zero new dependencies; entirely additive (existing fMRI/spectral/rDCM/latent
paths bit-exact).

**v0.7.0 Variational Laplace Validation complete (2026-06-12).** The VL engine is now
validated across an N×SNR recovery matrix (spectral/task/latent-circuit), BMR relative-ranking
recovers true circuit structure (vs brute-force ELBO), and VL was cross-validated against MATLAB
SPM12 `spm_nlsi_GN` (model-ranking agreement 1.0; free energy within ~10% after matching the
noise model — the original ~270-nat "forward-model divergence" was 80% our own
MATLAB bridge omitting `Y.Q`; see the 2026-09-06 correction in STATE.md). All four phases (29-32)
passed verification. The framework remains an **fMRI / spectral DCM**; it has no time-domain
evoked-response (EEG/MEG ERP) capability yet — that is the v0.8.0 milestone.

**Prior: shipped v0.6.0 (2026-06-10, scope-cut).** SPM12-grade **Variational Laplace** engine
(`spm_nlsi_GN`) generalized across forward models via a `ForwardModel` protocol; synthetic
recovery, BMR, and hybrid VAE-DCM validated. See `.planning/MILESTONES.md`.

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
- VL N×SNR synthetic recovery matrix (spectral/task/latent-circuit; per-region R², masked sign, CI coverage, shrinkage) — v0.7.0
- BMR relative-evidence ranking recovers true circuit structure + agrees with brute-force ELBO (ρ=1.0) — v0.7.0
- VL cross-validated vs MATLAB SPM12 `spm_nlsi_GN` (ranking agreement 1.0; F equal up to constant offset) — v0.7.0
- VL numerical-robustness guards (convergence/determinism, dt≥0.1 precision-intractability guard, C-order CSD round-trip) — v0.7.0

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

## Current Milestone: v0.8.0 — DCM for Evoked Responses (EEG/MEG ERP)

**Goal:** Add the first time-domain evoked-response capability — a canonical-microcircuit (CMC)
neural-mass → extrinsic coupling + evoked integration → single-dipole lead-field → scalp-ERP
forward stack — generating MMN/P300 waveforms, SPM12-parity validated at every phase, reusing the
v0.7.0 VL + amortized inference. Forward + parity + synthetic only; no empirical ERP fitting.
Seed: `.planning/v0.8.0-EEG-ERP-SCOPE.md` + `.planning/research/v0.8.0/`.

**Target features (confirmed scope):**

- **CMC neural-mass forward model (single source)** — 4 laminar populations / 8 states, He/τ
  synaptic kernels, sigmoid firing, SPM `-exp` parameter transforms; parity vs `spm_fx_cmc.m`.
- **Exponential-Euler integrator** — a pure-torch port of SPM's `spm_int_L` (frozen-Jacobian
  local linearisation), the central new component; rk4/dopri5 will NOT match SPM for finite dt.
- **Extrinsic coupling + evoked integration (multi-source)** — forward/backward/lateral A,
  condition modulation B (incl. `diag(B)→G` precision path), input C; parity vs `spm_gen_erp.m`.
- **Single-dipole lead-field → scalp ERP** — `kron(P.J, L)` projection + deviant−standard
  difference wave; parity vs `spm_lx_erp.m` (LFP mode first, ECD via MATLAB-exported gain).
- **ERP-DCM model class + inference + MMN demo** — `erp_dcm_model.py` wired to VL + amortized
  via the `ForwardModel` protocol; 5-source auditory MMN network; precision-sweep (sp
  self-inhibition gain) → attenuated-MMN transfer curve (the Adams/Ranlund artifact).

**Decisions locked (milestone init 2026-06-25):**
- **CMC only** (not Jansen-Rit/ERP) — exposes superficial-pyramidal gain = precision, required by
  the downstream consumer.
- **Single-dipole-per-source** lead-field (not full montage/BEM).
- **VL + amortized + MMN demo** inference scope — synthetic/forward only.

**Explicitly deferred / out of scope:**

- Empirical M/EEG ERP **data fitting** — forward + parity capability first; fitting follows once
  validated.
- Full sensor montage / BEM head model, source localization / inverse, group PEB.
- Jansen-Rit/ERP and CMC_2014/TFM neural-mass variants.
- Full delay-operator path — delays forced off (D=1) for the first parity pass.

**Downstream consumer:** `actinf_physics` (Phase 133 / NEURO2-04) imports this model forward-only
to reproduce precision-attenuated MMN; keep this milestone domain-agnostic (general DCM-ERP).

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
| CMC (not Jansen-Rit/ERP) for v0.8.0 ERP | CMC exposes superficial-pyramidal gain = precision; the downstream Adams/Ranlund psychosis-MMN use case needs exactly that parameter | — Pending (v0.8.0) |
| New `spm_int_L` exp-Euler integrator (not torchdiffeq) for ERP | SPM integrates ERPs via frozen-Jacobian local linearisation; rk4/dopri5 converges to the true ODE but NOT to the SPM solution at finite dt → fails parity | — Pending (v0.8.0) |
| ERP forward implements existing `ForwardModel` protocol | Reuses VL + amortized inference with zero engine edits; LatentCircuitForward (v0.6.0) is the precedent | — Pending (v0.8.0) |
| Single-dipole lead-field; LFP-first parity | Sufficient for forward MMN/P300 + difference wave; ECD gain precomputed in MATLAB and exported via the validation/ bridge | — Pending (v0.8.0) |

---
*Last updated: 2026-06-25 — v0.7.0 (Variational Laplace Validation) verified complete (Phases
29-32). Started milestone v0.8.0 (DCM for Evoked Responses, EEG/MEG ERP; Phases 33-36). Confirmed
scope: CMC neural-mass forward + spm_int_L integrator + extrinsic/evoked + single-dipole
lead-field + ERP-DCM model class + MMN precision-sweep demo. SPM12-parity at every phase;
forward + synthetic only (empirical fitting deferred). Research: .planning/research/v0.8.0/.*
