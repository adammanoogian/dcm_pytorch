# Requirements: Pyro-DCM v0.6.0 Latent Circuit DCM

**Defined:** 2026-05-24
**Core Value:** Distill black-box neural network computation into interpretable directed circuits with posterior uncertainty, using bilinear DCM as the interpretable model.

## v0.6.0 Requirements

Requirements for Latent Circuit DCM. Each maps to a roadmap phase.

### Direct Observation Forward Model

- [ ] **OBS-01**: `LatentCircuitSystem(nn.Module)` wraps `NeuralStateEquation` with N-dimensional state (no hemodynamic states). Accepts `A`, `B_list`, `C`, `input_fn`, `input_mod_fn`. Returns N derivatives (not 5N).
- [ ] **OBS-02**: `direct_observation(x, C_obs, noise_prec)` computes `y = C_obs @ x + noise`. When `C_obs=I`, reduces to identity observation `y = x + noise`.
- [ ] **OBS-03**: `LatentCircuitSystem` integrates via `torchdiffeq.odeint` on the same time grid as the RNN trajectories (no TR downsampling).
- [ ] **OBS-04**: Existing `NeuralStateEquation.derivatives` reused without modification. Zero edits to `neural_state.py`, `balloon_model.py`, `bold_signal.py`, or `coupled_system.py`.

### Latent Circuit Simulator

- [ ] **SIM-01**: `simulate_latent_circuit(A, C, B_list, u, dt, ...)` generates synthetic N-dimensional trajectories from known bilinear ground truth (no hemodynamic convolution). Returns `{'trajectories': (T, N), 'A': ..., 'B_list': ..., 'C': ...}`.
- [ ] **SIM-02**: Simulator output matches `LatentCircuitSystem` ODE integration at `atol=1e-6` given identical parameters (cross-validation test).

### Pyro Generative Model

- [ ] **MODEL-01**: `latent_circuit_dcm_model(observed, stimulus, ...)` Pyro model samples `A_free`, `C`, `B_free_j`, `noise_prec` with priors recalibrated for RNN-scale dynamics (not BOLD-scale; documented constant `LC_A_PRIOR_VARIANCE` separate from task DCM's `A_PRIOR_VARIANCE`).
- [ ] **MODEL-02**: Prior variance for A and B empirically calibrated on 5+ synthetic RNNs; documented with rationale (addresses pitfall LC4 -- BOLD priors cause shrinkage on RNN-scale data).
- [ ] **MODEL-03**: `create_guide()` auto-discovers all sample sites in `latent_circuit_dcm_model` without factory changes. Verified on AutoNormal, AutoLowRankMVN, AutoIAFNormal.
- [ ] **MODEL-04**: `C_obs` fixed at identity for v0.6.0 (no learned projection; addresses pitfall LC5 rotation ambiguity). Documented deferral of learned C_obs to v0.7.0+.
- [ ] **MODEL-05**: Multi-start SVI (>=10 random initializations, select by best ELBO) implemented as standard fitting procedure (addresses pitfall LC11; L&E uses 100 restarts).

### Synthetic Validation (Known-Connectivity Ground Truth)

- [ ] **SYNTH-01**: Parameter recovery on bilinear synthetic ground truth (N=4-8 nodes): A RMSE, B RMSE, sign recovery, 95% CI coverage. Uses same metric infrastructure as v0.3.0 RECOV benchmarks.
- [ ] **SYNTH-02**: Trajectory R-squared >= 0.95 on held-out trials for correctly-specified (bilinear) synthetic data.
- [ ] **SYNTH-03**: ELBO correctly selects the true N (number of nodes) from candidates N=2,4,6,8 on synthetic data where ground truth is N=4.

### CT-RNN Training

- [ ] **RNN-01**: `ContinuousTimeRNN(nn.Module)` implements `tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b)` with ReLU activation. Trainable via BPTT with behavioral cross-entropy loss on choice readout `z = W_out @ h`.
- [ ] **RNN-02**: Context-dependent decision-making (CDDM) task environment via neurogym. Two sensory modalities, context cue selects relevant modality.
- [ ] **RNN-03**: Train >= 20 RNNs (different random seeds, H=64-256 hidden units) on CDDM. Save trained weights and full h(t) trajectories per trial condition.
- [ ] **RNN-04**: Fixed-point analysis utilities: find fixed points via optimization (`torch.optim`), compute Jacobians (`torch.autograd.functional.jacobian`), eigenvalue decomposition (`torch.linalg.eig`). Used for linearization quality diagnostic (pitfall LC1).

### Latent Extraction & Dimensionality Reduction

- [ ] **DIM-01**: PCA-based dimensionality reduction: H-dimensional RNN hidden states -> N-dimensional DCM state space. Uses `sklearn.decomposition.PCA`.
- [ ] **DIM-02**: Output reconstruction R-squared gate: verify PCA-projected states reconstruct RNN outputs (behavioral readout) with R² >= 0.90 before fitting DCM (addresses pitfall LC3).
- [ ] **DIM-03**: Variance explained diagnostic: report cumulative variance explained vs N, recommend N where marginal gain < 5%.

### End-to-End Pipeline

- [ ] **PIPE-01**: Full pipeline: trained RNN -> extract h(t) -> PCA -> fit bilinear DCM -> posterior A, B_j matrices. Single-script demonstration.
- [ ] **PIPE-02**: Trajectory R-squared of bilinear DCM fit to nonlinear RNN latents reported per condition (context x coherence). Target: R² >= 0.80 (acknowledging model misspecification).
- [ ] **PIPE-03**: Linearization quality diagnostic: `||J(h*) - A_eff||_F / ||J(h*)||_F` at fixed points, documenting where bilinear approximation is valid (pitfall LC1).

### Bayesian Model Reduction

- [ ] **BMR-01**: `bayesian_model_reduction(posterior_mean, posterior_cov, prior_mean, prior_cov, reduced_prior_cov)` implements Friston & Penny (2011) analytic model evidence for reduced models. Cite REF-070.
- [ ] **BMR-02**: Circuit-size selection: fit full DCM (N=max), then analytically score all reduced architectures (connection subsets). Report log-evidence differences.
- [ ] **BMR-03**: BMR results compared to brute-force ELBO comparison (separate SVI fits per architecture) on synthetic data. Agreement validates the analytic approximation.

### TRIBE Foundation Model Use Case

- [ ] **TRIBE-01**: Extract latent representations from Meta TRIBE v2 (or a comparable open-source brain encoding model) for a stimulus set.
- [ ] **TRIBE-02**: Fit bilinear DCM to TRIBE's latent dynamics. Report trajectory R-squared and posterior A/B matrices.
- [ ] **TRIBE-03**: Demonstrate: "interpretable circuit distilled from a foundation model" — show which connections the DCM identifies and how they relate to known neuroscience (e.g., sensory→decision pathways).

### Comparison & Validation

- [ ] **COMP-01**: Quantitative comparison to Langdon & Engel 2025: bilinear DCM trajectory R² vs L&E nonlinear circuit R² on same RNN ensemble. Bilinear DCM adds posterior uncertainty (D-1) and explicit B_j (D-2) that L&E lacks.
- [ ] **COMP-02**: ELBO model comparison: linear vs bilinear DCM for same RNN (does B add anything?). Different N values. Different B_j mask topologies.
- [ ] **COMP-03**: Guide type comparison: posterior quality across AutoNormal, AutoLowRankMVN, AutoIAFNormal for latent circuit fitting. Coverage calibration on synthetic data.
- [ ] **COMP-04**: Misspecification analysis: systematic comparison of bilinear fit quality across RNN nonlinearity regimes (near-linear vs strongly nonlinear operating points).
- [ ] **COMP-05**: Qualitative comparison to TVB/Jirsa whole-brain approach: positioning section in methods/discussion. Scale (meso-circuit vs whole-brain), interpretability (A/B matrices vs neural mass params), inference (SVI vs simulation-based).

### Publication Artifacts

- [ ] **PUB-01**: Publication-quality figures: (1) pipeline schematic, (2) parameter recovery on synthetic, (3) trajectory fits per condition, (4) A + B_j heatmaps with credible intervals, (5) ELBO vs N curve, (6) BMR circuit selection, (7) L&E comparison, (8) misspecification regime analysis.
- [ ] **PUB-02**: Methods section (Markdown + LaTeX) covering: bilinear DCM equations, direct observation model (citing David et al. 2006 EEG/MEG DCM precedent), SVI inference, BMR (citing Friston & Penny 2011), validation strategy.
- [ ] **PUB-03**: REFERENCES.md updated with REF-070 through REF-076 (BMR, BMS, EEG DCM, Thomas 2023, DCM-RNN, Pinotsis 2013, Langdon & Engel 2025).

## Future Requirements (deferred)

### v0.7.0 Candidates (Neural ODE DCM — Approach 2)

- **NODE-01**: Replace bilinear neural state equation with learned Neural ODE `dx/dt = f_theta(x, u)`
- **NODE-02**: Rotational degeneracy mitigation (structural masking, coordinate regularization)
- **NODE-03**: Comparison of bilinear vs Neural ODE distillation quality

### v0.6.1 Candidates

- **LC-OBS-01**: Learned C_obs projection (jointly optimized with DCM parameters)
- **LC-PERT-01**: Perturbation validation: DCM connectivity changes -> RNN weight perturbation -> behavioral effect prediction
- **LC-TRIBE-02**: Fit DCM to multiple foundation models (BrainLM, POYO) for cross-model comparison

## Out of Scope

| Feature | Reason |
|---------|--------|
| Neural ODE distillation (Approach 2) | Separate milestone v0.7.0; different research question |
| Real neural recordings (monkey electrophysiology) | Ground truth unknown; establish method on synthetic/RNN/TRIBE first |
| Training RNN on BOLD signal | Would conflate neural and hemodynamic dynamics; RNN predicts behavior, hidden states ARE neural dynamics |
| Replicating L&E's Cayley Q exactly | Their code is MIT-licensed and available; we use PCA (simpler, standard) |
| Amortized guides for latent circuit fitting | Per-subject SVI first; amortized deferred to v0.6.1 |
| Dale's law constraints on CT-RNN | Optional biological constraint; defer to v0.6.1 |
| Group-level PEB analysis | Single-RNN fitting first; group analysis deferred |
| GUI / clinical deployment | PROJECT.md permanent out-of-scope |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| OBS-01 | Phase 20 | Pending |
| OBS-02 | Phase 20 | Pending |
| OBS-03 | Phase 20 | Pending |
| OBS-04 | Phase 20 | Pending |
| SIM-01 | Phase 20 | Pending |
| SIM-02 | Phase 20 | Pending |
| MODEL-01 | Phase 20 | Pending |
| MODEL-02 | Phase 20 | Pending |
| MODEL-03 | Phase 20 | Pending |
| MODEL-04 | Phase 20 | Pending |
| MODEL-05 | Phase 20 | Pending |
| SYNTH-01 | Phase 20 | Pending |
| SYNTH-02 | Phase 20 | Pending |
| SYNTH-03 | Phase 20 | Pending |
| RNN-01 | Phase 21 | Pending |
| RNN-02 | Phase 21 | Pending |
| RNN-03 | Phase 21 | Pending |
| RNN-04 | Phase 21 | Pending |
| DIM-01 | Phase 21 | Pending |
| DIM-02 | Phase 21 | Pending |
| DIM-03 | Phase 21 | Pending |
| PIPE-01 | Phase 22 | Pending |
| PIPE-02 | Phase 22 | Pending |
| PIPE-03 | Phase 22 | Pending |
| COMP-01 | Phase 22 | Pending |
| COMP-02 | Phase 22 | Pending |
| COMP-03 | Phase 22 | Pending |
| COMP-04 | Phase 22 | Pending |
| COMP-05 | Phase 22 | Pending |
| BMR-01 | Phase 23 | Pending |
| BMR-02 | Phase 23 | Pending |
| BMR-03 | Phase 23 | Pending |
| TRIBE-01 | Phase 24 | Pending |
| TRIBE-02 | Phase 24 | Pending |
| TRIBE-03 | Phase 24 | Pending |
| PUB-01 | Phase 25 | Pending |
| PUB-02 | Phase 25 | Pending |
| PUB-03 | Phase 25 | Pending |

**Coverage:**
- v0.6.0 requirements: 38 total
- Mapped to phases: 38/38
- Unmapped: 0

---
*Requirements defined: 2026-05-24*
*Last updated: 2026-05-24 -- traceability table populated by roadmapper (38/38 mapped to Phases 20-25)*
