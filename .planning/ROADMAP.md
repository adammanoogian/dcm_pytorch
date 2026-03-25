# Roadmap: Pyro-DCM v0.1.0-foundation

**Created:** 2026-03-25
**Milestone:** v0.1.0-foundation
**Total phases:** 8
**Total requirements:** 29

## Phase 1: Neural & Hemodynamic Forward Model + Task-DCM Simulator
**Status:** pending
**Requirements:** FWD-01, FWD-02, FWD-03, FWD-04, SIM-01
**Goal:** Build the complete task-based DCM forward model pipeline — from neural state equation through hemodynamics to BOLD signal — plus a simulator that generates realistic synthetic data.

**Success criteria:**
1. Balloon-Windkessel ODE integrates without numerical instability for 500s simulations
2. Neural state equation dx/dt = Ax + Cu produces stable trajectories for ||A|| < 1
3. BOLD signal equation produces realistic % signal change (0.5-5%)
4. ODE integrator wrapper supports configurable solver (Euler, RK4, Dopri5) via torchdiffeq
5. Simulator generates N-region BOLD time series given (A, C, u(t), hemo_params, SNR)

**Mathematical scope:**
- Bilinear neural model: dx/dt = Ax + Cu — [REF-001] Eq. 1
- Balloon model: ds/dt, df/dt, dv/dt, dq/dt — [REF-002] Eq. 2-5
- BOLD signal: y = V0[k1(1-q) + k2(1-q/v) + k3(1-v)] — [REF-002] Eq. 6
- ODE integration via torchdiffeq odeint with RK4 or Dopri5

**Key files:**
- `src/pyro_dcm/forward_models/neural_state.py`
- `src/pyro_dcm/forward_models/balloon_model.py`
- `src/pyro_dcm/forward_models/bold_signal.py`
- `src/pyro_dcm/utils/ode_integrator.py`
- `src/pyro_dcm/simulators/task_simulator.py`
- `tests/test_balloon.py`, `tests/test_neural_state.py`

---

## Phase 2: Spectral DCM Forward Model + CSD Computation + spDCM Simulator
**Status:** pending
**Requirements:** FWD-05, FWD-06, SIM-02
**Goal:** Build the spectral DCM observation pipeline — cross-spectral density computation from time series, transfer function mapping, and a simulator for synthetic CSD data.

**Success criteria:**
1. CSD computation from time series matches scipy.signal.csd output within numerical tolerance
2. Transfer function H(w) = C_out(iwI - A)^-1 C_in correctly predicts CSD peaks at expected frequencies
3. Endogenous fluctuation (1/f^alpha) and observation noise spectral models integrated
4. Simulator produces synthetic CSD given (A, noise params, frequency range)

**Mathematical scope:**
- Cross-spectral density via multi-taper/Welch — standard signal processing
- Spectral transfer function: g(w) = (iwI - A)^-1 — [REF-010] Eq. 3
- Predicted CSD: S(w) = |H(w)|^2 Sigma_neuronal(w) + Sigma_observation(w) — [REF-010] Eq. 4
- Neuronal fluctuation spectrum: 1/f^alpha — [REF-010] Eq. 5-6
- Observation noise spectrum — [REF-010] Eq. 7

**Key files:**
- `src/pyro_dcm/forward_models/csd_computation.py`
- `src/pyro_dcm/forward_models/spectral_transfer.py`
- `src/pyro_dcm/simulators/spectral_simulator.py`
- `tests/test_spectral.py`

---

## Phase 3: Regression DCM Forward Model + rDCM Simulator
**Status:** pending
**Requirements:** FWD-07, SIM-03
**Goal:** Build the regression DCM frequency-domain likelihood and simulator, including the analytic posterior and ARD sparsity priors.

**Success criteria:**
1. Frequency-domain likelihood computes correctly for known parameters
2. Region-wise factorization p(y_j | A_j, C_j) matches analytic formula from [REF-020]
3. Simulator produces synthetic frequency-domain data for given (A, C, noise)
4. Sparse A recovery: L1-penalized MAP on simulated data recovers zero-pattern with F1 > 0.85

**Mathematical scope:**
- DFT of convolution model — [REF-020] Eq. 4-8
- Analytic posterior per region — [REF-020] Eq. 11-14
- Free energy in closed form — [REF-020] Eq. 15
- ARD sparsity priors — [REF-020] Eq. 9-10

**Key files:**
- `src/pyro_dcm/forward_models/rdcm_likelihood.py`
- `src/pyro_dcm/simulators/rdcm_simulator.py`
- `tests/test_rdcm.py`

---

## Phase 4: Pyro Generative Models + Baseline Guides
**Status:** pending
**Requirements:** PROB-01, PROB-02, PROB-03, PROB-04
**Goal:** Wire the forward models into Pyro generative models with proper plate structure and priors, plus implement baseline mean-field Gaussian guides for each variant.

**Success criteria:**
1. Each DCM variant registered as Pyro model with correct plate structure
2. Prior samples from each model produce physically plausible BOLD/CSD/rDCM data
3. Mean-field Gaussian guide runs SVI without NaN gradients
4. ELBO decreases monotonically on simulated data (no instability)
5. Posterior means from SVI are in the right ballpark (within 20% of truth)

**Mathematical scope:**
- Prior specifications for A, C, hemodynamic params, noise — see research/FEATURES.md Section 4.1
- ELBO: E_q[log p(y|theta,m)] - KL[q(theta) || p(theta|m)]
- Mean-field Gaussian: q(theta) = prod_i N(mu_i, sigma_i^2)

**Key files:**
- `src/pyro_dcm/generative_models/task_dcm.py`
- `src/pyro_dcm/generative_models/spectral_dcm.py`
- `src/pyro_dcm/generative_models/regression_dcm.py`
- `src/pyro_dcm/guides/meanfield.py`
- `src/pyro_dcm/inference/svi_runner.py`
- `tests/test_pyro_models.py`

---

## Phase 5: Parameter Recovery Tests (All Three Variants)
**Status:** pending
**Requirements:** REC-01, REC-02, REC-03, REC-04
**Goal:** Rigorous parameter recovery validation — simulate with known ground truth, infer, and verify accuracy and calibration for all three DCM variants.

**Success criteria:**
1. Task-DCM: RMSE(A) < 0.05, 95% CI coverage in [0.90, 0.99]
2. Spectral DCM: RMSE(A) < 0.05, 95% CI coverage in [0.90, 0.99]
3. Regression DCM: RMSE(A) < 0.05, 95% CI coverage in [0.90, 0.99]
4. Convergence within 5000 SVI steps for task and spectral, closed-form for rDCM
5. Results documented with plots: true vs inferred A, posterior marginals, ELBO trace

**Test protocol per variant:**
1. Generate 50 synthetic datasets with known A (3-region, 5-region networks)
2. Run SVI to convergence, extract posterior means and 95% CIs
3. Compute RMSE, coverage, correlation(A_true, A_inferred)
4. Verify ELBO is higher for correctly specified model than misspecified

**Key files:**
- `tests/test_task_dcm_recovery.py`
- `tests/test_spectral_dcm_recovery.py`
- `tests/test_rdcm_recovery.py`
- `validation/recovery_results/`

---

## Phase 6: Validation Against SPM / Reference Implementations
**Status:** pending
**Requirements:** VAL-01, VAL-02, VAL-03, VAL-04
**Goal:** Cross-validate all three DCM variants against established reference implementations (SPM12, tapas/rDCM) and verify ELBO-based model comparison.

**Success criteria:**
1. Task-DCM posterior means within 10% relative error of SPM12 on same simulated data
2. Spectral DCM posterior means within 10% relative error of SPM12 spm_dcm_csd
3. Regression DCM results match tapas/rDCM toolbox within published tolerances
4. ELBO model ranking matches SPM free energy ranking on 3+ model comparison scenarios
5. Discrepancies documented with root-cause analysis

**Validation protocol:**
1. Export simulated datasets to .mat format for SPM12 processing
2. Run SPM12 DCM estimation in MATLAB, extract Ep.A and F (free energy)
3. Compare Pyro posterior vs SPM posterior element-wise
4. For rDCM: compare against tapas toolbox or Frassle et al. published benchmarks
5. For model comparison: compare ELBO ranking vs SPM free energy ranking

**Key files:**
- `validation/compare_spm.py`
- `validation/spm_reference_data/`
- `validation/VALIDATION_REPORT.md`

---

## Phase 7: Amortized Neural Inference Guides
**Status:** pending
**Requirements:** AMR-01, AMR-02, AMR-03, AMR-04
**Goal:** Implement normalizing flow guides that amortize inference across subjects — train once on simulated data, then do single-pass inference on new subjects.

**Success criteria:**
1. Normalizing flow guide trained on 10,000+ simulated datasets per variant
2. Amortized posterior RMSE within 1.5x of per-subject SVI RMSE
3. Amortization gap (per-subject ELBO - amortized ELBO) < 10% of per-subject ELBO
4. Inference time per subject < 1s (forward pass through flow)
5. Calibration maintained (coverage still in [0.85, 0.99])

**Architecture:**
- Summary network: 1D-CNN or Set Transformer over BOLD/CSD input
- Flow: MAF or Neural Spline Flow via Zuko
- Training: amortized SVI with ELBO objective over dataset of (data, params) pairs
- For rDCM: amortized guide may be optional given closed-form posterior

**Key files:**
- `src/pyro_dcm/guides/amortized_flow.py`
- `src/pyro_dcm/guides/summary_networks.py`
- `scripts/train_amortized_guide.py`
- `tests/test_amortized.py`

---

## Phase 8: Metrics, Benchmarks, and Documentation
**Status:** pending
**Requirements:** BNC-01, BNC-02, BNC-03
**Goal:** Comprehensive benchmarking comparing all inference methods across all DCM variants, plus API documentation and reproducibility scripts.

**Success criteria:**
1. Comprehensive benchmark table: RMSE, coverage, ELBO, wall time per variant x method
2. Advantage analysis: where does amortization help most? where is it unnecessary?
3. API documentation (docstrings + usage examples) following NumPy-style conventions
4. Reproducibility: single script runs all benchmarks from scratch
5. Draft methods section for paper

**Deliverables:**
- `benchmarks/` directory with all scripts and results
- `docs/03_methods_reference/` with mathematical derivations matching code
- Summary table comparing Pyro-DCM vs SPM12 vs tapas/rDCM

---

## Phase Dependency Graph

```
Phase 1 (Task forward model)
    |
    +---> Phase 2 (Spectral forward model)
    |         |
    |         +---> Phase 3 (rDCM forward model)
    |         |         |
    +----+----+---------+
         |
         v
    Phase 4 (Pyro generative models + guides)
         |
         v
    Phase 5 (Parameter recovery tests)
         |
         v
    Phase 6 (SPM validation)
         |
         v
    Phase 7 (Amortized inference)
         |
         v
    Phase 8 (Benchmarks + docs)
```

---
*Roadmap created: 2026-03-25*
*Last updated: 2026-03-25 after initialization*
