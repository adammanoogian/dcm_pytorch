# Requirements: Pyro-DCM

**Defined:** 2026-03-25
**Core Value:** The A matrix (effective connectivity) remains an explicit, interpretable object with full posterior uncertainty throughout inference.

## v1 Requirements

Requirements for v0.1.0-foundation milestone. Each maps to roadmap phases.

### Infrastructure & Forward Models

- [ ] **FWD-01**: Implement Balloon-Windkessel hemodynamic model as standalone ODE module with parameters {kappa, gamma, tau, alpha, E0} per region, using Stephan et al. (2007) [REF-002] Eq. 2-5
- [ ] **FWD-02**: Implement neural state equation dx/dt = Ax + Cu with explicit A matrix (NxN effective connectivity) and C matrix (NxM driving input weights) per Friston et al. (2003) [REF-001] Eq. 1
- [ ] **FWD-03**: Implement BOLD signal equation y = V0[k1(1-q) + k2(1-q/v) + k3(1-v)] mapping hemodynamic states to observed BOLD per [REF-002] Eq. 6
- [ ] **FWD-04**: Build ODE integrator wrapper (torchdiffeq) solving coupled neural + hemodynamic system with configurable solver (Euler, RK4, Dopri5)
- [ ] **FWD-05**: Implement cross-spectral density computation from time series using multi-taper or Welch periodogram, matching SPM's spm_csd_mtf output format
- [ ] **FWD-06**: Implement spectral DCM transfer function H(w) = C(iwI - A)^-1 C' mapping neural dynamics to predicted CSD, including endogenous fluctuations and observation noise spectral models per [REF-010] Eq. 3-7
- [ ] **FWD-07**: Implement regression DCM analytic likelihood in frequency domain per Frassle et al. (2017) [REF-020] Eq. 4-15

### Probabilistic Framework

- [ ] **PROB-01**: Define Pyro generative model for task-based DCM with priors on A, C, hemodynamic parameters, and noise precision
- [ ] **PROB-02**: Define Pyro generative model for spectral DCM where likelihood is evaluated in frequency domain (predicted CSD vs observed CSD)
- [ ] **PROB-03**: Define Pyro generative model for regression DCM with analytic frequency-domain likelihood
- [ ] **PROB-04**: Implement mean-field Gaussian variational guide (Laplace-style) for each DCM variant as baseline inference method

### Simulation & Testing

- [ ] **SIM-01**: Build data simulator for task-based DCM: given A, C, input u(t), hemodynamic params, produce synthetic BOLD with realistic SNR
- [ ] **SIM-02**: Build data simulator for spectral DCM: given A and spectral noise params, produce synthetic CSD matrices
- [ ] **SIM-03**: Build data simulator for regression DCM: given A, C, produce synthetic frequency-domain data
- [ ] **REC-01**: Parameter recovery test for task-based DCM: simulate -> infer -> compare A_true vs A_posterior with RMSE < 0.05 and calibrated coverage
- [ ] **REC-02**: Parameter recovery test for spectral DCM with same criteria
- [ ] **REC-03**: Parameter recovery test for regression DCM with same criteria
- [ ] **REC-04**: Validate ELBO convergence for each DCM variant on simulated data with known ground truth

### Validation

- [ ] **VAL-01**: Cross-validate task-based DCM against SPM12 spm_dcm_estimate on at least one simulated dataset — posterior means within 10% relative error
- [ ] **VAL-02**: Cross-validate spectral DCM against SPM12 spm_dcm_csd on simulated data
- [ ] **VAL-03**: Cross-validate regression DCM against tapas/rDCM toolbox or published benchmarks from Frassle et al.
- [ ] **VAL-04**: Bayesian model comparison (ELBO ranking) correctly identifies true connectivity architecture in >80% of simulated cases across all three variants

### Amortized Inference

- [ ] **AMR-01**: Implement neural amortized guide (normalizing flow) for task-based DCM mapping observed BOLD -> posterior over (A, C, hemodynamic params)
- [ ] **AMR-02**: Implement neural amortized guide for spectral DCM mapping observed CSD -> posterior over (A, spectral params)
- [ ] **AMR-03**: Implement neural amortized guide for regression DCM
- [ ] **AMR-04**: Demonstrate amortized guide achieves accuracy within 2x of per-subject NUTS on held-out simulated subjects

### Metrics & Benchmarking

- [ ] **BNC-01**: Report per-variant: RMSE on A matrix, posterior calibration (coverage), ELBO, wall-clock time, number of gradient steps
- [ ] **BNC-02**: Compare mean-field guide vs amortized guide on all metrics
- [ ] **BNC-03**: Report amortization gap (per-subject ELBO minus amortized ELBO)

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Extensions

- **EXT-01**: Non-stationary A(t) with GP/RNN prior over time-varying connectivity
- **EXT-02**: Neural ODE extension replacing bilinear neural dynamics
- **EXT-03**: Rotational degeneracy resolution via structural masking or GNN constraints
- **EXT-04**: Windowed CSD or time-frequency CSD for non-stationary spectral DCM
- **EXT-05**: HMM/switching regime model for discrete state transitions in A(t)
- **EXT-06**: Scaling to large ROI sets (100+ regions) with sparse A architectures

## Out of Scope

| Feature | Reason |
|---------|--------|
| GUI / web dashboard | Research tool — API/CLI only |
| Real-time fMRI processing | Not a clinical deployment target |
| Structural connectivity (tractography priors) | Different modality, future work |
| Multi-modal (EEG/MEG) DCM | Different observation models entirely |
| Clinical classification pipelines | Beyond v0.1 scope |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FWD-01 | Phase 1 | Complete |
| FWD-02 | Phase 1 | Complete |
| FWD-03 | Phase 1 | Complete |
| FWD-04 | Phase 1 | Complete |
| SIM-01 | Phase 1 | Complete |
| FWD-05 | Phase 2 | Complete |
| FWD-06 | Phase 2 | Complete |
| SIM-02 | Phase 2 | Complete |
| FWD-07 | Phase 3 | Complete |
| SIM-03 | Phase 3 | Complete |
| PROB-01 | Phase 4 | Complete |
| PROB-02 | Phase 4 | Complete |
| PROB-03 | Phase 4 | Complete |
| PROB-04 | Phase 4 | Complete |
| REC-01 | Phase 5 | Pending |
| REC-02 | Phase 5 | Pending |
| REC-03 | Phase 5 | Pending |
| REC-04 | Phase 5 | Pending |
| VAL-01 | Phase 6 | Pending |
| VAL-02 | Phase 6 | Pending |
| VAL-03 | Phase 6 | Pending |
| VAL-04 | Phase 6 | Pending |
| AMR-01 | Phase 7 | Pending |
| AMR-02 | Phase 7 | Pending |
| AMR-03 | Phase 7 | Pending |
| AMR-04 | Phase 7 | Pending |
| BNC-01 | Phase 8 | Pending |
| BNC-02 | Phase 8 | Pending |
| BNC-03 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 29 total
- Mapped to phases: 29
- Unmapped: 0

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-27 after Phase 4 completion*
