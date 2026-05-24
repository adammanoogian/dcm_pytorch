---
type: research
scope: synthesis
milestone: v0.6.0
updated: 2026-05-24
---

# Research Summary: Latent Circuit DCM (v0.6.0)

**Project:** Pyro-DCM
**Milestone:** v0.6.0 -- Latent Circuit DCM
**Domain:** RNN-to-circuit distillation via bilinear DCM with direct observation
**Researched:** 2026-05-24
**Confidence:** HIGH (architecture + stack) / MEDIUM-HIGH (pitfall severity)

---

## Executive Summary

The v0.6.0 milestone adds Latent Circuit DCM to the existing Pyro-DCM framework: train a
continuous-time RNN on a cognitive task, extract hidden state trajectories, reduce
dimensionality with PCA, then fit bilinear DCM (dx/dt = Ax + sum_j u_j B_j x + Cu) to
those trajectories using a direct linear observation model in place of the
balloon-Windkessel hemodynamic chain. The scientific contribution is threefold: (1)
posterior uncertainty on circuit parameters -- something the canonical comparison target
Langdon & Engel (2025, Nature Neuroscience) does not provide; (2) explicit bilinear
parameterization of context-dependent connectivity changes via B_j matrices; and (3)
principled ELBO-based circuit architecture selection. Almost all required infrastructure is
already in the codebase. New code amounts to two forward-model files, one Pyro model, one
new rnn/ package (four files), and one simulator -- all built on top of the existing guide
factory, SVI runner, and bilinear NeuralStateEquation.

The recommended build order is inside-out: implement the direct observation model and latent
circuit simulator first (no RNN needed), validate bilinear DCM parameter recovery on
synthetic data via Path B (direct ODE simulation to DCM fit), then build the RNN pipeline,
and finally wire everything together for the L&E comparison. This order validates the core
scientific claim -- that bilinear DCM with Pyro SVI recovers interpretable circuit parameters
with calibrated posteriors -- before committing to the more complex RNN training
infrastructure. The RNN is a means to an end; the DCM model is the contribution.

The most critical risks are structural: (a) bilinear misspecification of nonlinear RNN
dynamics (fails for tanh-saturated or ReLU kink dynamics), (b) rotational degeneracy of the
PCA-projected state space (recovered A matrices are defined only up to an orthogonal
transform), (c) prior miscalibration inherited from the BOLD regime (SPM-calibrated priors
are the wrong scale for RNN hidden states -- a direct repeat of Phase 16.1 RECOV-04 that
must be prevented by design), and (d) PCA discarding task-relevant oblique dynamics
(Dubreuil et al. 2024 eLife). Each has a concrete prevention strategy documented below;
none is a showstopper for the milestone, but all must be pre-empted architecturally.

---

## Key Findings

### Recommended Stack

The existing PyTorch + Pyro + torchdiffeq + Zuko + scipy stack covers essentially all new
capability requirements. Total new direct dependencies are two optional packages in a new
[latent] extra in pyproject.toml: neurogym>=2.3 (Gymnasium-compatible, 33 standard tasks
including the exact CDDM task used by Langdon & Engel, verified at v2.3.1 on PyPI
2026-03-31) and scikit-learn>=1.3 (offline PCA analysis and explained variance diagnostics).
Both are stable, pure-Python, and have no C compilation issues.

**Core technologies:**

- **PyTorch 2.x (existing):** CTRNN as nn.Module with fixed-dt Euler integration
  (alpha=dt/tau, matching L&E); fixed-point finding via Adam on ||dx/dt||^2; eigenvalue
  analysis via torch.linalg.eig. No version change required.
- **Pyro 1.9+ (existing):** The new latent_circuit_dcm_model uses identical pyro.sample
  naming conventions so all 6 guide types and 3 ELBO variants inherit automatically.
  Zero changes to guides.py.
- **torchdiffeq (existing):** Same odeint call for the N-dimensional latent circuit ODE.
  Use fixed-dt Euler for RNN training (speed); adaptive stepping for DCM fitting (ODE
  accuracy affects parameter recovery).
- **neurogym>=2.3 (NEW):** Provides ContextDecisionMaking-v0 (the L&E task), Python >=3.10,
  framework-agnostic -- returns numpy arrays, no framework dependency.
- **scikit-learn>=1.3 (NEW):** PCA for offline hidden-state dimension selection via
  explained_variance_ratio_. Not for the differentiable path (use torch.pca_lowrank there).

Libraries explicitly rejected: trainRNNbrain (adds hydra-core, opinionated pipeline --
reference only for Euler/noise implementation details), latentcircuit (compare against, not
depend on), FixedPointFinder (stale 2022, TF-dependent -- reimplementable in ~50 lines of
PyTorch), PsychRNN (TF, unmaintained since 2021).

### Expected Features

**Must have (table stakes -- paper is incomplete without these):**

- **TS-1:** CT-RNN trainer on CDDM task (BPTT, Adam, ~100 units, ReLU)
- **TS-2:** Hidden state trajectory extraction across trial conditions
- **TS-3:** Dimensionality reduction H->N via PCA with output-R^2 quality gate
- **TS-4:** Direct observation model y = C_obs @ x + noise -- the core architectural change;
  replaces balloon-Windkessel + BOLD chain
- **TS-5:** SVI fitting of bilinear DCM to reduced trajectories (reuses all guide infra)
- **TS-6:** Trajectory reconstruction quality metrics (R^2 per node, aggregate vs N)
- **TS-9:** Synthetic validation on known-connectivity bilinear ground truth (same RECOV
  criteria as v0.3.0: RMSE, sign recovery, 95% CI coverage)
- **TS-10:** Publication figures 1-7 required; Figure 8 supplementary

**Should have (differentiators over L&E -- primary novelty claims):**

- **D-1:** Posterior uncertainty on A and B_j -- credible intervals on every connection
  weight (L&E gives point estimates + ensemble variability only, no per-parameter posteriors)
- **D-2:** Explicit bilinear B_j matrices -- context-dependent connectivity directly
  readable, not implicit in nonlinear activation f()
- **D-3:** ELBO-based architecture selection across N nodes, linear vs bilinear, B mask
  topology (L&E uses heuristic N=8; Bayesian model comparison can select N)
- **D-4:** Misspecification analysis -- systematic comparison of bilinear vs nonlinear L&E
  circuit fit quality across RNN ensemble
- **D-5:** Guide type comparison for latent circuit fitting (reuses v0.2.0 calibration infra)
- **TS-7:** Perturbation validation (L&E strongest validation: translate DCM B_j change to
  RNN weight space, verify behavioral prediction matches)
- **TS-8:** ELBO model comparison across circuit sizes N=2-10

**Defer to v0.7.0+ or post-paper:**

- Neural ODE distillation (AF-2), real neural recordings (AF-3), amortized guides (AF-5),
  full Cayley-parameterized Q embedding replicating L&E (AF-4), biologically constrained
  multi-area RNN (AF-6).

**MVP entry:** TS-9 + TS-4 + D-1 require no RNN (Path B only, Phase 1 alone is viable).
**Critical path:** TS-1 -> TS-2 -> TS-3 -> TS-4 -> TS-5. Parallel after TS-5: all of
TS-6, TS-7, TS-8, D-1 through D-5.

### Architecture Approach

Latent Circuit DCM is a fourth variant in the existing pattern (alongside task, spectral,
and rDCM). The observation path changes from a 5N-dimensional coupled ODE (neural +
hemodynamic) to an N-dimensional neural-only ODE with y = C_obs @ x. Everything else --
NeuralStateEquation, parameterize_A/B, compute_effective_A, integrate_ode, create_guide,
run_svi, extract_posterior_params -- is reused unchanged. A new rnn/ package lives outside
forward_models/ because the RNN is a data-generation tool, not a DCM generative model
component; its training is plain PyTorch supervised learning, completely separate from Pyro
SVI. Start with C_obs = I (full state observation, N_dcm = N_pca) to avoid the A/C_obs
rotation degeneracy; defer partial observation to v0.7.0+.

**Major new components:**

1. `forward_models/direct_observation.py` -- y = C_obs @ x; LOW complexity; a single
   matrix multiply replacing the entire balloon-Windkessel + BOLD signal chain
2. `forward_models/latent_circuit_system.py` -- N-dimensional ODE nn.Module for torchdiffeq;
   mirrors CoupledDCMSystem but with N-dimensional (not 5N) state vector
3. `models/latent_circuit_dcm_model.py` -- Pyro generative model; samples A, B_j, C; runs
   neural ODE; evaluates Gaussian likelihood on trajectories; MEDIUM complexity
4. `rnn/` package (4 files): ContinuousTimeRNN, rnn_trainer, latent_extraction,
   synthetic_rnn -- pure PyTorch, no Pyro
5. `simulators/latent_circuit_simulator.py` -- fast synthetic data path for unit tests and
   DCM model debugging without RNN training (enables Path B from day one)

**Nothing modified:** guides.py, neural_state.py, ode_integrator.py, balloon_model.py,
coupled_system.py, task_dcm_model.py. All existing tests continue to pass.

### Critical Pitfalls

PITFALLS.md identifies 14 new pitfalls specific to RNN distillation (beyond v0.3.0 B1-B14
which still apply). Top five ranked by impact on paper claim defensibility:

1. **LC4: Prior scale mismatch -- repeat of Phase 16.1 RECOV-04 (CRITICAL).** A_free ~
   N(0, 1/64) was calibrated for slow BOLD dynamics. RNN hidden states have entirely
   different magnitudes and timescales. Prevention: z-score RNN hidden states before DCM
   fitting; define A_PRIOR_VARIANCE_RNN and B_PRIOR_VARIANCE_RNN as required model arguments
   (never default to BOLD-calibrated priors). Empirically calibrate values in Phase 1.

2. **LC2: Rotational degeneracy of PCA state space (CRITICAL for claims).** Recovered A
   matrices from different RNN seeds live in arbitrary PCA coordinate systems and are not
   element-wise comparable. Prevention: Procrustes alignment for all cross-seed comparisons;
   validate through perturbation experiments (rotation-invariant) and eigenvalue spectra, not
   element-wise A_ij recovery.

3. **LC5: Observation model identifiability confound (HIGH -- architectural design gate).**
   If C_obs is a free sampled parameter alongside A, the model has an unconstrained rotation
   ambiguity where A and C_obs compensate for each other. Prevention: fix C_obs = I for
   v0.6.0. This decision must precede any fitting code being written.

4. **LC3: PCA discards task-relevant oblique dynamics (HIGH -- validation gate).** PCA
   maximizes variance; task-relevant dynamics can live in low-variance oblique directions
   (Dubreuil et al. 2024 eLife directly demonstrates this for PFC). Prevention: always
   check output-R^2 (task output reconstructability from top-N PCs) before fitting DCM.
   Gate: output-R^2 < 0.7 means PCA is insufficient for this task/RNN combination.

5. **LC1: Bilinear misspecification of nonlinear RNN dynamics (CRITICAL for defensibility).**
   The bilinear model is a first-order Taylor expansion. Fails for strongly saturated tanh,
   ReLU kink dynamics, and GRU gating. Prevention: use tanh CTRNN near x=0 for initial
   validation; report linearization quality index ||J(t) - A_eff(t)||_F / ||J(t)||_F with
   every claim; include L&E nonlinear model as comparison baseline.

Additional design-level pitfalls to address before coding begins:

- **LC11:** Multi-start SVI (10-20 seeds, best-ELBO selection) is non-optional for
  publication. The ELBO landscape has local optima from A rotations and A/B trade-offs.
- **LC12:** Direct observation removes hemodynamic smoothing; use LR=1e-4 and gradient
  clip=5.0 (vs LR=1e-3 and clip=10.0 for BOLD DCM).
- **LC6:** Context may be encoded as additive state offset, not bilinear B_j. Always fit
  linear-only (A + Cu) model alongside bilinear for ELBO comparison.
- **LC10:** dt must be a required explicit parameter. The BOLD-calibrated default dt=0.5s is
  wrong for RNN timescales and must not default silently.

---

## Implications for Roadmap

Based on the feature dependency graph (FEATURES.md), the build order (ARCHITECTURE.md), and
the phase-specific warning table (PITFALLS.md), the natural phase structure for v0.6.0 is
four phases. Phases 1 and 2 can run in parallel.

### Phase 1: Direct Observation Forward Model + Synthetic Validation

**Rationale:** The DCM model is the scientific contribution. It must pass parameter recovery
on synthetic bilinear ground truth before any RNN work begins. Phase 1 requires no new
dependencies, resolves LC5 and LC4 before they can propagate downstream, and produces Path B
-- the fast synthetic test path used throughout all subsequent phases.

**Delivers:**
- direct_observation.py, latent_circuit_system.py, latent_circuit_simulator.py,
  latent_circuit_dcm_model.py (C_obs=I, scalar noise prior)
- Empirically recalibrated A_PRIOR_VARIANCE_RNN and B_PRIOR_VARIANCE_RNN determined from
  calibration sweeps (not assumed from BOLD literature)
- Parameter recovery benchmark: known A -> DCM fit -> RMSE, sign recovery, 95% CI coverage;
  extended to bilinear B recovery (same RECOV criteria as v0.3.0)

**Addresses:** TS-4, TS-9, D-1 (posterior uncertainty is free with SVI)

**Pitfalls to resolve here:** LC5 first (fix C_obs=I), then LC4 (prior recalibration),
LC10 (dt as required argument), LC12 (LR=1e-4, gradient clip=5.0)

**Research flag:** NO -- standard Pyro pattern, directly analogous to task_dcm_model.py.

---

### Phase 2: CT-RNN Training + Latent Extraction

**Rationale:** Build the data pipeline only after the DCM model is validated on synthetic
data. The RNN is infrastructure. Can run in parallel with Phase 1 (confirmed independent in
ARCHITECTURE.md Path B vs Path A analysis).

**Delivers:**
- rnn/continuous_time_rnn.py (Euler, alpha=dt/tau, ReLU, configurable N_hidden)
- rnn/rnn_trainer.py (Adam MSE loop, early stopping; completely separate from Pyro SVI)
- rnn/latent_extraction.py (extract h(t), PCA, output-R^2 quality gate)
- rnn/synthetic_rnn.py (known-connectivity RNN with W_rec = Q @ A_true @ Q^T for
  rotation-invariant parameter recovery ground truth; LC2 mitigation)
- Benchmark: RNN reaches behavioral criterion on CDDM; output-R^2 >= 0.7 in top-N PCs;
  at least 5 RNN training seeds tested; fixed-point structure consistent across seeds

**Uses:** neurogym>=2.3, scikit-learn>=1.3, PyTorch DataLoader

**Pitfalls to resolve here:** LC9 (start with CDDM -- fixed-point task, not working memory),
LC14 (train >=5 seeds, check dynamical consistency), LC3 (output-R^2 gate before passing
trajectories to DCM)

**Research flag:** NO for CT-RNN training (standard supervised learning, well-referenced in
trainRNNbrain). YES for empirical calibration of the output-R^2 gate threshold on CDDM data
-- determine from data, not prior assumption.

---

### Phase 3: End-to-End Pipeline + L&E Comparison

**Rationale:** Wire Phases 1 and 2 together for the full Path A pipeline. This is the source
of the paper primary claims: bilinear DCM with Pyro posteriors fit to L&E-trained RNN,
compared to L&E nonlinear latent circuit.

**Delivers:**
- Full pipeline: train RNN -> extract h(t) -> PCA -> DCM fit -> posterior A/B with CIs and
  trajectory reconstruction
- Multi-start SVI (10-20 seeds, best-ELBO selection) -- non-optional for publication
- Procrustes-aligned cross-seed A matrix statistics
- Comparison to L&E LatentNet on the same trained RNN (Path C)
- ELBO model comparison: N=2-10 sweep, linear vs bilinear, B mask topology
- Figures 3, 4, 5 (DCM posterior with CIs, trajectory reconstruction, ELBO comparison)

**Pitfalls to resolve here:** LC1 (linearization quality index reported with every claim),
LC2 (Procrustes alignment for all cross-seed statistics), LC6 (linear-only baseline model
always fit alongside bilinear), LC11 (multi-start SVI), LC13 (B_j validated through
perturbation predictions, not element-wise comparison to W_rec)

**Research flag:** YES -- The Procrustes + DSA cross-seed A matrix validation framework
requires implementation choices not covered by existing Pyro-DCM infrastructure. A brief
targeted design step on the shared coordinate system is recommended before the Phase 3
benchmark spec is written.

---

### Phase 4: Perturbation Validation + Misspecification Analysis + Paper

**Rationale:** Perturbation validation (TS-7) is the strongest scientific claim and the most
technically complex to implement (must translate DCM B_j change to RNN weight space via PCA
projection). Misspecification analysis (D-4) is the honest self-assessment required for a
credible paper. Both are blocked on Phase 3.

**Delivers:**
- Perturbation pipeline: B_j modification -> PCA projection -> RNN weight perturbation ->
  behavioral change comparison (Figure 6)
- Misspecification analysis: bilinear R^2 vs L&E nonlinear R^2 across RNN ensemble; R^2 vs
  linearization quality index (Figure 8, supplementary)
- Guide type comparison (D-5): ELBO + calibration across 6 guide types on latent circuit
  model
- Figures 1, 2, 7, 8, supplementary figures; methods section draft

**Pitfalls to resolve here:** LC7 (perturbation is the primary validation metric -- not
element-wise A recovery), LC8 (held-out trial R^2 for N selection, not ELBO alone)

**Research flag:** YES -- The perturbation translation from bilinear B_j to RNN W_rec via
PCA basis is novel for this exact model combination. FEATURES.md flags TS-7 as technically
complex. A concrete design document specifying the rank-one mapping is needed before Phase 4
implementation begins.

---

### Phase Ordering Rationale

The inside-out order (forward model first, RNN second, integration third, analysis fourth)
is justified by three constraints from the research:

1. **Dependency:** TS-4 (direct obs model) is a prerequisite for all downstream work.
   TS-9 (synthetic validation) requires only TS-4 and can complete before TS-1 (RNN trainer).
   Phases 1 and 2 can run in parallel if two contributors are available.

2. **Risk front-loading:** LC4 (prior recalibration) and LC5 (observation model design) are
   architectural decisions that, if made wrong, corrupt all downstream results. Forcing Phase
   1 to resolve them before any RNN work begins prevents a repeat of the Phase 16.1 RECOV-04
   failure pattern.

3. **Test before integrate (project rule):** Path B (Phase 1 simulator to DCM fit) is the
   standalone test for the DCM model. Running Path A before Path B passes would violate the
   project rule against integrating untested modules.

### Research Flags

**Needs targeted research-phase during planning:**

- **Phase 3 benchmark design:** Procrustes + DSA cross-seed A matrix validation framework.
  Not in current Pyro-DCM infrastructure; needs a concrete design step before the Phase 3
  benchmark spec is written.
- **Phase 4 perturbation translation:** Mapping DCM bilinear B_j to RNN W_rec modifications
  via PCA basis. L&E uses their Cayley Q; the PCA-based adaptation for bilinear DCM is novel
  and not specified in any existing reference for this exact model combination.

**Standard patterns (skip research-phase):**

- **Phase 1 DCM model:** Follows task_dcm_model.py structure with simpler observation. No
  novel architecture; direct code reuse from existing modules.
- **Phase 2 CT-RNN training:** Standard supervised learning with verified reference
  implementation (trainRNNbrain Euler formulation).
- **ELBO model comparison:** Infrastructure exists from v0.2.0; extend to N sweep and
  linear-vs-bilinear comparison axes.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | neurogym v2.3.1 verified on PyPI; all existing stack coverage verified against codebase; 2 new optional deps only |
| Features | HIGH | L&E 2025 is the direct comparison paper; full PMC text extracted; feature boundaries and anti-features clearly scoped |
| Architecture | HIGH | Direct code analysis of all existing modules; reuse vs new-build table definitive; integration points verified against source |
| Pitfalls | MEDIUM-HIGH | LC1-LC7, LC10-LC12 are HIGH confidence (theory + Phase 16.1 experience + peer-reviewed evidence). LC8, LC9, LC13, LC14 are MEDIUM (plausible mechanisms not yet quantified for bilinear DCM on RNN data specifically) |

**Overall confidence:** HIGH on what to build and in what order. MEDIUM on exact pitfall
severity thresholds (linearization index cutoffs, output-R^2 gate value, N selection
behavior) -- these must be empirically calibrated during Phases 1 and 2, not assumed.

### Gaps to Address

- **Prior recalibration values (LC4):** Mechanism is clear; exact sigma_A and sigma_B for
  the RNN regime must be empirically determined in Phase 1 calibration sweeps. Cannot be
  pre-specified; depends on RNN hidden state statistics.

- **Output-R^2 gate threshold (LC3):** Recommended >= 0.7 is theoretical. Calibrate
  empirically on CDDM in Phase 2; may need task-specific adjustment.

- **Linearization quality threshold (LC1):** The > 0.5 risk cutoff is theoretical. Measure
  on CDDM-trained RNNs in Phase 3 to determine whether observed values require a methods
  caveat or an architectural fix.

- **Perturbation translation design (TS-7, LC7):** The exact mapping from B_j entries to RNN
  W_rec modifications via PCA basis is not specified in any existing reference for this exact
  model combination. Needs a design document before Phase 4 begins.

- **ELBO-selected N for CDDM:** Whether ELBO comparison confirms L&E heuristic N=8 or
  selects a different value is an open empirical question that v0.6.0 will answer.

---

## Sources

### Primary (HIGH confidence)

- Langdon & Engel (2025). Latent circuit inference from heterogeneous neural responses
  during cognitive tasks. Nature Neuroscience 28, 665-675.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/
- engellab/latentcircuit https://github.com/engellab/latentcircuit --
  reference implementation of Net + LatentNet (MIT, PyTorch)
- engellab/trainRNNbrain https://github.com/engellab/trainRNNbrain --
  CT-RNN training reference (Euler formulation, alpha=dt/tau, noise injection)
- Existing Pyro-DCM codebase: neural_state.py, task_dcm_model.py, coupled_system.py,
  guides.py, task_simulator.py (direct code analysis)
- Dubreuil, Valente, Beiran, Mastrogiuseppe, Ostojic (2024). Aligned and oblique dynamics
  in recurrent neural networks. eLife. [LC3 basis]
- Hristos, Jha, Engel (2024). Inferring context-dependent computations through linear
  approximations of PFC dynamics. Science Advances. PMC11654703. [LC1, LC6]
- Huang, Singh, Martinelli, Rajan (2025). Measuring and Controlling Solution Degeneracy
  across Task-Trained RNNs. NeurIPS 2025. arXiv:2410.03972. [LC14, LC2 DSA]
- neurogym PyPI v2.3.1 https://pypi.org/project/neurogym/ -- Python >=3.10 confirmed

### Secondary (MEDIUM confidence)

- Nozari et al. (2024). Macroscopic resting-state brain dynamics are best described by
  linear models. Nature Biomedical Engineering. PMC11357987. [LC1]
- Smith, Linderman, Sussillo (2021). JSLDS. NeurIPS. [LC1 comparator]
- Valente, Pillow, Ostojic (2022). Low-rank RNNs. NeurIPS. [LC1, LC3]
- Dubreuil et al. (2022). Theory of Gating in RNNs. PLoS CB. PMC9762509. [LC6]
- Pellegrino et al. (2024). SliceTCA. Nature Neuroscience. [LC3]
- v0.3.0 PITFALLS.md and Phase 16.1 RECOV-04 experience. [LC4, LC11]
- Langdon et al. (2025). Single-unit activations. Nature Machine Intelligence.
- Ji-An et al. (2025). Tiny RNNs. Nature.

### Tertiary (LOW confidence)

- Singh et al. (2020). MINDy. NeuroImage. [LC4 scale issues]
- State derivative normalization. arXiv:2401.02902. [LC10]
- Information criteria for dynamical systems (2025). [LC8]

---
*Research completed: 2026-05-24*
*Ready for roadmap: yes*
