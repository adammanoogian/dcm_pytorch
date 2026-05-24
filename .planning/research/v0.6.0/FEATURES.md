# Feature Landscape: Latent Circuit DCM (v0.6.0)

**Domain:** RNN-to-circuit distillation via bilinear DCM
**Researched:** 2026-05-24
**Reference paper:** Langdon & Engel (2025), Nature Neuroscience 28, 665--675

## Langdon & Engel 2025 Pipeline Summary

The canonical pipeline from Langdon & Engel (hereafter "L&E") operates in three
stages, each with specific equations and tooling. Understanding this pipeline is
essential because it defines the table stakes for any latent circuit paper --
and the places where bilinear DCM deviates from it.

### Stage 1: Train RNN on cognitive task

- **Architecture:** Continuous-time RNN with dynamics
  `tau * dy/dt = -y + f(W_rec @ y + W_in @ u)`, readout `z = W_out @ y`
- **Activation:** ReLU (primary) or Softplus (robustness test)
- **Task:** Context-dependent decision-making (CDDM): two sensory modalities
  (color/motion), context cue selects which modality is decision-relevant
- **Training:** BPTT via Adam on behavioral loss (cross-entropy on choice)
- **Scale:** N ~ 100-500 hidden units; train 200+ RNNs for ensemble analysis
- **Tooling:** `trainRNNbrain` package (PyTorch, Engel lab)

### Stage 2: Fit latent circuit to RNN responses

- **Observation model:** `y = Q @ x` (Eq. 1) -- Q is orthonormal N x n
- **Circuit dynamics:** `dx/dt = -x + f(w_rec @ x + w_in @ u)` (Eq. 2) -- nonlinear f()
- **Readout:** `z = w_out @ x` (Eq. 3)
- **Loss:** `L = sum_{k,t} ||y - Q @ x||^2 + ||z - w_out @ x||^2` (Eq. 7) --
  reconstruction error on neural activity AND behavioral output
- **Dimensionality:** n = 8 nodes for CDDM (2 context + 2 color + 2 motion + 2 choice),
  chosen to match total number of task inputs/outputs
- **Q constraint:** Orthonormality via Cayley transform of skew-symmetric matrices
- **Optimizer:** Adam, lr=0.02, minibatch 128 trials, stop at loss plateau
- **Ensemble:** 100 random initializations per RNN, select top 10 by validation R^2

### Stage 3: Validate and interpret

- **Conjugacy check:** Q^T @ W_rec @ Q approx w_rec (Eq. 5), r=0.89
- **Perturbation prediction:** delta_ij in circuit -> q_i @ q_j^T rank-one perturbation
  in RNN, measure behavioral effect on psychometric function
- **Comparison to linear methods:** Regression/correlation-based dim reduction FAILS
  to find the suppression mechanism that the latent circuit reveals
- **Ensemble clustering:** PCA on flattened w_rec across 200 RNNs, GMM clustering
  reveals 3 solution families

### What L&E does NOT provide

- No posterior uncertainty on circuit parameters (point estimates only)
- No model comparison across circuit architectures (fixed n=8 for CDDM)
- No systematic handling of model misspecification (ReLU circuit fit to ReLU RNN)
- No bilinear parameterization (input-dependent connectivity change is implicit in
  nonlinear f(), not explicit in B_j matrices)

---

## Table Stakes

Features users expect in a latent circuit DCM paper. Missing any of these means
the paper will be rejected or viewed as incomplete.

### TS-1: Continuous-time RNN trainer

| Aspect | Detail |
|--------|--------|
| **What** | Train a CT-RNN on a cognitive task (CDDM at minimum) with BPTT |
| **Why expected** | L&E and all related work start from a trained RNN; the RNN IS the ground truth |
| **Complexity** | Medium |
| **Existing code** | None in Pyro-DCM; need new `rnn/` module |
| **Dependencies** | PyTorch only (no Pyro needed) |
| **Notes** | L&E uses `trainRNNbrain` package. We can either depend on it or write our own minimal CT-RNN class. Writing our own is preferred for control over hidden state extraction. The RNN dynamics are `tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b)` with ReLU activation. Training objective is cross-entropy on behavioral choice readout `z = W_out @ h`. Must train multiple RNNs (>=20, ideally 100+) with different random seeds to show ensemble generality. |

### TS-2: Hidden state trajectory extraction

| Aspect | Detail |
|--------|--------|
| **What** | Extract h(t) trajectories from trained RNN across trial conditions |
| **Why expected** | These trajectories are the "data" that DCM fits to |
| **Complexity** | Low |
| **Existing code** | None; comes naturally from RNN forward pass |
| **Dependencies** | TS-1 |
| **Notes** | Save full h(t) at integration timesteps (not just output z). Need to organize by trial condition (context x coherence level). Typical: 500-1000 trials, T~100 timesteps, H~100-500 units. |

### TS-3: Dimensionality reduction (H -> N)

| Aspect | Detail |
|--------|--------|
| **What** | Reduce RNN hidden states from H dimensions to N DCM nodes |
| **Why expected** | DCM operates on small circuits (N=3-10); RNN has H=100-500 units |
| **Complexity** | Medium |
| **Existing code** | None |
| **Dependencies** | TS-2 |
| **Notes** | L&E uses a learned orthonormal Q (Cayley-parameterized, jointly optimized with circuit). For bilinear DCM, simpler approaches are viable: (1) PCA to top-N components, (2) task-variable regression to find task-aligned axes, (3) learned linear projection Q optimized alongside DCM. PCA is the standard baseline. The choice of N is a model selection question -- see TS-8. |

### TS-4: Direct observation model

| Aspect | Detail |
|--------|--------|
| **What** | `y = C_obs @ x + epsilon` (linear observation, no hemodynamic convolution) |
| **Why expected** | When fitting DCM to RNN latent states, there is no hemodynamic delay. The observation model is a direct linear mapping from DCM states to projected RNN states. |
| **Complexity** | Low |
| **Existing code** | Partially -- the bilinear Pyro model already has the neural state equation; need to bypass balloon-Windkessel and BOLD signal |
| **Dependencies** | Existing `NeuralStateEquation`, `parameterize_A`, `parameterize_B` |
| **Notes** | New Pyro model `latent_circuit_model` that samples A, B, C_obs and integrates `dx/dt = Ax + sum u_j B_j x + Cu` then evaluates Gaussian likelihood on projected RNN states. This is architecturally simple -- it is the task_dcm_model with the hemodynamic stage removed and the observation model replaced. |

### TS-5: Bilinear DCM fitting to RNN trajectories

| Aspect | Detail |
|--------|--------|
| **What** | SVI fitting of bilinear DCM to dimensionality-reduced RNN trajectories |
| **Why expected** | This is the core scientific contribution -- showing bilinear DCM can distill RNN dynamics |
| **Complexity** | Medium (leverages existing SVI infrastructure) |
| **Existing code** | Guide factory (6 types), SVI runner, ELBO objectives -- all exist |
| **Dependencies** | TS-3, TS-4, existing `guides.py` |
| **Notes** | Use existing `run_svi()` with new `latent_circuit_model`. The key novelty: B_j matrices explicitly capture how experimental context reshapes connectivity, vs L&E where this is implicit in nonlinear f(). Start with `auto_normal` guide; `auto_lowrank_mvn` for uncertainty quality. |

### TS-6: Trajectory reconstruction quality metrics

| Aspect | Detail |
|--------|--------|
| **What** | Quantify how well DCM-predicted trajectories match RNN trajectories |
| **Why expected** | Every distillation paper reports reconstruction quality |
| **Complexity** | Low |
| **Existing code** | None specific; standard torch operations |
| **Dependencies** | TS-5 |
| **Notes** | Metrics: (1) R^2 per node per condition on held-out trials, (2) normalized RMSE, (3) variance explained across conditions. Report per-node and aggregate. L&E uses correlation coefficient r. We should report R^2 (more standard) plus r for comparability. |

### TS-7: Perturbation validation

| Aspect | Detail |
|--------|--------|
| **What** | Show that connectivity changes predicted by DCM produce expected behavioral effects when applied back to the RNN |
| **Why expected** | L&E's strongest validation -- perturbation predictions confirm the circuit captures causal structure, not just correlations |
| **Complexity** | High |
| **Existing code** | None |
| **Dependencies** | TS-1, TS-5 |
| **Notes** | For bilinear DCM: perturb B_j entries, translate to RNN weight perturbation via Q (or PCA basis), re-run RNN, measure behavioral change. Compare predicted vs observed behavioral effect. This is scientifically critical but technically complex. For bilinear DCM, the mapping is: changing B_j[i,k] means changing how input u_j modulates the connection from node k to node i. The perturbation in RNN space depends on the dimensionality reduction method. |

### TS-8: Model comparison across circuit sizes

| Aspect | Detail |
|--------|--------|
| **What** | Compare ELBO across DCM models with different N (number of nodes) |
| **Why expected** | L&E fixes N=8 heuristically; Bayesian DCM can SELECT N via ELBO |
| **Complexity** | Medium |
| **Existing code** | ELBO model comparison already built and benchmarked |
| **Dependencies** | TS-5, existing ELBO comparison infrastructure |
| **Notes** | Fit bilinear DCM with N=2,4,6,8,10 nodes. Plot ELBO vs N. The ELBO naturally penalizes model complexity (more parameters = lower ELBO unless justified by data). This is a genuine advantage over L&E's point-estimation approach. |

### TS-9: Known-connectivity synthetic validation

| Aspect | Detail |
|--------|--------|
| **What** | Build a synthetic RNN with KNOWN bilinear structure, verify DCM recovers it |
| **Why expected** | Establishes that the method works when the model is correctly specified |
| **Complexity** | Medium |
| **Existing code** | `task_simulator.py` bilinear mode (but generates BOLD, need neural-state-only variant) |
| **Dependencies** | TS-4 |
| **Notes** | Create a small RNN (N=4-8 units) with dynamics `dx/dt = Ax + u_j B_j x + Cu + noise`. This IS a bilinear system, so DCM should recover A, B exactly (up to posterior uncertainty). Report RMSE on A, RMSE on B, sign recovery rate, 95% CI coverage -- same metrics as v0.3.0 RECOV criteria. This is the "easy" validation; the "hard" one is recovering structure from a nonlinear RNN (TS-11). |

### TS-10: Publication-quality figures

| Aspect | Detail |
|--------|--------|
| **What** | Standard figure set for a latent circuit paper |
| **Why expected** | Paper will not be publishable without these |
| **Complexity** | Medium |
| **Existing code** | Some plotting utilities from v0.2.0 benchmarks; CircuitViz JSON serializer |
| **Dependencies** | All other features |
| **Notes** | Required figures detailed below under "Paper Figures" section |

---

## Differentiators

Features that set bilinear DCM apart from L&E and other existing approaches.
These are the scientific contributions -- the reasons a reviewer would say
"this adds something new."

### D-1: Posterior uncertainty on circuit parameters (PRIMARY NOVELTY)

| Aspect | Detail |
|--------|--------|
| **What** | Full posterior distributions on A and B_j matrices via SVI |
| **Value proposition** | L&E gives point estimates only. Bilinear DCM with Pyro gives posterior means, variances, and credible intervals on every connection weight. This enables: (1) distinguishing strong from weak connections with uncertainty, (2) testing whether specific connections are significantly non-zero, (3) quantifying how uncertain the circuit topology is. |
| **Complexity** | Low (already built into SVI infrastructure) |
| **Existing code** | `extract_posterior()` in `guides.py`, all 6 guide types |
| **Dependencies** | TS-5 |
| **Notes** | This is the single biggest selling point. L&E fits 100 models and picks the best by R^2 -- they get ensemble variability but not parameter uncertainty per model. Pyro gives principled Bayesian posteriors. Show: (1) posterior credible intervals on A_ij and B_j[i,k], (2) posterior predictive trajectories (uncertainty bands on predicted x(t)), (3) shrinkage toward zero for irrelevant connections. |

### D-2: Explicit input-dependent connectivity (B_j matrices)

| Aspect | Detail |
|--------|--------|
| **What** | B_j matrices directly parameterize how experimental inputs reshape connectivity |
| **Value proposition** | In L&E, context-dependent connectivity changes are implicit in the nonlinear activation f(). You have to inspect w_rec and reason about how ReLU gating creates effective inhibition. In bilinear DCM, B_j[i,k] directly tells you: "input u_j strengthens/weakens the connection from node k to node i by this much." This is the native parameterization for context-dependent processing in neuroscience. |
| **Complexity** | Low (already built in v0.3.0) |
| **Existing code** | `parameterize_B`, `compute_effective_A`, bilinear `NeuralStateEquation` |
| **Dependencies** | None new |
| **Notes** | The key comparison figure: L&E's w_rec heatmap (static) vs DCM's A + u_j*B_j heatmaps (showing how connectivity CHANGES with context). This makes the context-switching mechanism visually obvious. For CDDM: B_context should suppress irrelevant sensory pathways -- the same mechanism L&E finds, but parameterized explicitly. |

### D-3: ELBO-based architecture selection

| Aspect | Detail |
|--------|--------|
| **What** | Principled Bayesian model comparison to select circuit size and topology |
| **Value proposition** | L&E chooses n=8 heuristically (matching number of task variables). ELBO comparison can: (1) confirm that 8 is optimal, or show a different N is better, (2) compare bilinear vs linear DCM for the same RNN (does B add anything?), (3) compare different B_j mask topologies. |
| **Complexity** | Low (infrastructure exists) |
| **Existing code** | ELBO comparison benchmarked in v0.2.0 |
| **Dependencies** | TS-5, TS-8 |
| **Notes** | Three comparison axes: (1) N selection (number of nodes), (2) linear vs bilinear (is B needed?), (3) B mask topology (which connections are modulated?). Each is a separate ELBO comparison. The linear-vs-bilinear comparison is particularly compelling: if ELBO prefers bilinear, it confirms context-dependent connectivity is needed. |

### D-4: Misspecification analysis (bilinear approximation of nonlinear RNN)

| Aspect | Detail |
|--------|--------|
| **What** | Systematic analysis of how well bilinear DCM approximates nonlinear RNN dynamics |
| **Value proposition** | This is the honest scientific question: bilinear DCM is a SIMPLER model than the ReLU circuit. How much information is lost? Where does the approximation break down? This analysis itself is a contribution -- nobody has systematically compared bilinear vs nonlinear distillation quality. |
| **Complexity** | High |
| **Existing code** | None |
| **Dependencies** | TS-5, TS-9 |
| **Notes** | Compare: (1) R^2 of bilinear DCM vs L&E nonlinear circuit on same RNN, (2) perturbation prediction accuracy of both models, (3) identify regimes where bilinear approximation fails (strong nonlinearity, saturation). The bilinear model is a first-order Taylor expansion of the nonlinear model around the fixed point; quantify the radius of validity. |

### D-5: Guide type comparison for circuit inference

| Aspect | Detail |
|--------|--------|
| **What** | Compare posterior quality across 6 guide types for latent circuit fitting |
| **Value proposition** | Which variational family best captures circuit parameter posteriors? Mean-field may miss correlations between A and B entries. IAF may capture them but be harder to train. This is a practical methodological contribution. |
| **Complexity** | Medium |
| **Existing code** | All 6 guide types, calibration sweep infrastructure |
| **Dependencies** | TS-5 |
| **Notes** | Run calibration analysis from v0.2.0 on latent circuit model. Report: (1) ELBO convergence, (2) posterior calibration, (3) coverage accuracy. Likely outcome: AutoLowRankMVN best balance of expressiveness and stability, but this needs to be demonstrated empirically for the new model. |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

### AF-1: Do NOT train RNN to predict BOLD signal

| Anti-Feature | Training RNN on hemodynamic (BOLD) data directly |
|--------------|--------------------------------------------------|
| **Why avoid** | The whole point is to fit DCM to NEURAL (latent) dynamics, not BOLD. If the RNN predicts BOLD, you'd need to deconvolve hemodynamics, adding complexity and circularity. The RNN should predict neural activity (spikes, LFP, or synthetic neural states). For the v0.6.0 milestone using synthetic data, the RNN IS the ground truth neural dynamics. |
| **What to do instead** | RNN predicts behavioral choice from task stimuli; hidden states ARE the neural dynamics. |

### AF-2: Do NOT implement Neural ODE distillation (v0.7.0 scope)

| Anti-Feature | Replacing bilinear DCM dynamics with a Neural ODE |
|--------------|---------------------------------------------------|
| **Why avoid** | Neural ODE (dx/dt = f_theta(x, u) where f is a neural network) is a different research question with different tradeoffs. It would be more expressive but less interpretable. It is explicitly deferred to v0.7.0 per PROJECT.md. Mixing it into v0.6.0 would dilute the bilinear contribution. |
| **What to do instead** | Keep dx/dt = Ax + sum u_j B_j x + Cu. Acknowledge Neural ODE as future work. |

### AF-3: Do NOT fit to real neural recordings

| Anti-Feature | Applying latent circuit DCM to monkey electrophysiology data in v0.6.0 |
|--------------|------------------------------------------------------------------------|
| **Why avoid** | Real data introduces observation noise, trial-to-trial variability, neuron dropout, and ground-truth-unknown confounds. The v0.6.0 milestone must establish the method on synthetic/RNN data where ground truth is known. Real data is a separate validation step (v0.8.0+). |
| **What to do instead** | All fitting on RNN-generated trajectories. Acknowledge real data application as next step. |

### AF-4: Do NOT replicate L&E's Cayley-parameterized Q exactly

| Anti-Feature | Reimplementing the full L&E latent circuit inference framework |
|--------------|---------------------------------------------------------------|
| **Why avoid** | L&E's code is available at `engellab/latent_circuit_inference`. Reimplementing it adds engineering burden without scientific contribution. The point is to COMPARE bilinear DCM to L&E, not to re-derive L&E. |
| **What to do instead** | Use PCA for dimensionality reduction (standard, transparent, reproducible). If a learned Q is needed for fair comparison, implement a simple linear projection optimized with DCM -- not the full Cayley machinery. |

### AF-5: Do NOT build amortized guides for latent circuit fitting

| Anti-Feature | Training amortized (neural network) guides for latent circuit DCM |
|--------------|-------------------------------------------------------------------|
| **Why avoid** | Per PROJECT.md, amortized guides are deferred. Per-subject SVI is sufficient for the paper. Amortization adds engineering complexity and makes the contribution harder to evaluate. |
| **What to do instead** | Use per-RNN SVI with existing guide factory. |

### AF-6: Do NOT over-engineer the RNN

| Anti-Feature | Building a complex multi-area, Dale's law, biologically constrained RNN |
|--------------|-------------------------------------------------------------------------|
| **Why avoid** | The RNN is a means to an end -- it generates latent dynamics for DCM to distill. A simple vanilla CT-RNN is sufficient and more interpretable. Biological constraints add parameters and make it harder to verify that DCM recovers the right structure. |
| **What to do instead** | Vanilla CT-RNN with ReLU, ~100 units, unconstrained W_rec. Match L&E for comparability. |

---

## Feature Dependencies

```
TS-1 (RNN trainer)
  |
  v
TS-2 (trajectory extraction)
  |
  v
TS-3 (dimensionality reduction) -----> TS-7 (perturbation validation)
  |                                       |
  v                                       v
TS-4 (direct observation model) -----> TS-10 (figures)
  |                                       ^
  v                                       |
TS-5 (bilinear DCM fitting) ------------>|
  |                                       |
  +---> TS-6 (reconstruction metrics) -->|
  |                                       |
  +---> TS-8 (model comparison) -------->|
  |                                       |
  +---> D-1 (posterior uncertainty) ----->|
  |                                       |
  +---> D-3 (ELBO architecture) -------->|
  |                                       |
  +---> D-4 (misspecification) ---------->|
  |
  v
TS-9 (synthetic validation) <-- uses TS-4 (direct obs model)

D-2 (explicit B_j) -- already built (v0.3.0)
D-5 (guide comparison) <-- uses TS-5 + existing calibration
```

**Critical path:** TS-1 -> TS-2 -> TS-3 -> TS-4 -> TS-5 -> TS-6/TS-7/TS-8

**Parallel after TS-5:** TS-6, TS-8, D-1, D-3, D-4, D-5 can all proceed
once SVI fitting works.

**Independent of RNN:** TS-9 (synthetic validation) only needs TS-4 and can
run in parallel with TS-1/TS-2/TS-3.

---

## Paper Figures (Required Set)

Based on L&E (2025), related work (Ji-An et al. 2025 Nature, Langdon et al.
2025 Nature Machine Intelligence), and the bilinear DCM novelty.

### Figure 1: Method schematic

**Content:** Pipeline diagram showing RNN -> trajectory extraction ->
dimensionality reduction -> bilinear DCM fitting -> interpretable circuit
**Purpose:** Introduce the method visually
**Panels:** (a) RNN architecture with task inputs, (b) hidden state trajectories
colored by condition, (c) PCA projection, (d) bilinear DCM equations,
(e) inferred A and B_j matrices as heatmaps
**Complexity:** Medium (Illustrator/TikZ)

### Figure 2: RNN task performance and dynamics

**Content:** (a) Psychometric curves showing RNN solves CDDM, (b) example
hidden state trajectories across conditions, (c) PCA of hidden states colored
by context/stimulus, (d) variance explained by PCA components
**Purpose:** Establish that the RNN has learned the task and has low-dimensional
dynamics worth distilling
**Complexity:** Low

### Figure 3: Bilinear DCM posterior -- the "money figure"

**Content:** (a) Posterior mean and 95% CI of A matrix entries (heatmap with
uncertainty), (b) Posterior of B_context matrix showing which connections are
modulated by context, (c) A_eff = A + u_context * B_context for each context
condition (two heatmaps), (d) comparison to L&E's w_rec (point estimate, no
uncertainty)
**Purpose:** Show the primary contribution -- posterior uncertainty + explicit
B_j
**Complexity:** Medium

### Figure 4: Trajectory reconstruction

**Content:** (a) True vs predicted trajectories per DCM node, per condition
(overlay with uncertainty bands from posterior predictive), (b) R^2 per node
per condition, (c) aggregate R^2 vs number of nodes N
**Purpose:** Quantify reconstruction quality
**Complexity:** Low-Medium

### Figure 5: Model comparison via ELBO

**Content:** (a) ELBO vs N (number of nodes) -- shows optimal circuit size,
(b) ELBO: linear DCM vs bilinear DCM -- shows B adds information,
(c) ELBO across B mask topologies
**Purpose:** Demonstrate principled architecture selection (key advantage over
L&E)
**Complexity:** Low

### Figure 6: Perturbation validation

**Content:** (a) Predicted behavioral effect of connectivity perturbation
(from DCM B_j), (b) Actual behavioral effect when perturbation applied to RNN,
(c) Predicted vs actual scatter plot across perturbation types
**Purpose:** Validate that inferred circuit has causal explanatory power
**Complexity:** High

### Figure 7: Synthetic validation (parameter recovery)

**Content:** (a) True vs recovered A entries with credible intervals,
(b) True vs recovered B entries with credible intervals, (c) Coverage
calibration plot (nominal vs empirical coverage), (d) RMSE distribution
across seeds
**Purpose:** Establish method correctness on known ground truth
**Complexity:** Medium (reuse v0.3.0 recovery plotting code)

### Figure 8: Misspecification analysis (supplementary)

**Content:** (a) R^2 of bilinear DCM vs nonlinear L&E circuit across RNN
ensemble, (b) When does bilinear fail? R^2 vs RNN nonlinearity strength,
(c) Posterior predictive check: are residuals structured when model is
misspecified?
**Purpose:** Honest assessment of bilinear approximation quality
**Complexity:** High

### Supplementary figures

- Guide type comparison (ELBO, calibration across guide types)
- Ensemble analysis across multiple RNNs
- Convergence diagnostics (ELBO traces)
- Prior sensitivity analysis

---

## Parameter Recovery Under Misspecification

When the ground truth is a nonlinear RNN and the model is bilinear DCM,
"parameter recovery" has a different meaning than in the well-specified case.

### Well-specified case (TS-9)

Ground truth is bilinear: `dx/dt = Ax + u_j B_j x + Cu + noise`.
Recovery means: DCM posterior on A contains true A, DCM posterior on B_j
contains true B_j, with calibrated coverage.

Standard RECOV metrics apply:
- A RMSE <= threshold
- B RMSE <= threshold on nonzero entries
- Sign recovery >= 80%
- 95% CI coverage >= 85%

### Misspecified case (D-4)

Ground truth is nonlinear: `dx/dt = -x + ReLU(W_rec x + W_in u)`.
There is no true A or B -- the RNN dynamics are nonlinear.

"Recovery" now means:
1. **Trajectory prediction quality** -- R^2 on held-out trials (the bilinear
   model captures the input-output mapping even if it gets the mechanism wrong)
2. **Effective connectivity correspondence** -- linearized RNN Jacobian
   `J = diag(f'(h*)) @ W_rec` at the fixed point should resemble DCM's A
   matrix. Correlation between J and posterior mean of A.
3. **Modulatory correspondence** -- the CHANGE in effective connectivity
   across contexts (J_context1 - J_context2) should resemble B_context.
   This is the key test: does bilinear DCM capture the context-dependent
   connectivity change that L&E finds via suppression?
4. **Perturbation predictive accuracy** -- does changing B_j in the DCM
   correctly predict behavioral effects of corresponding RNN perturbation?
5. **Posterior calibration under misspecification** -- posteriors may be
   overconfident because the model is wrong. Report calibration and note
   whether posteriors widen (desirable) or remain narrow (overconfident).

### What "good enough" looks like

- R^2 >= 0.7 on held-out trajectories (L&E reports r ~ 0.89 on training data
  for the nonlinear model; bilinear should be somewhat lower)
- Perturbation predictions qualitatively correct (same direction as RNN,
  even if magnitude differs)
- A matrix correlates with linearized RNN Jacobian at r >= 0.5
- B_context captures the dominant context-dependent connectivity change
- Posterior credible intervals include the linearized ground truth most of
  the time (no formal coverage guarantee under misspecification, but track it)

---

## What ELBO-Based Model Comparison Buys You Here

Three distinct comparison axes, each scientifically meaningful:

### Axis 1: Number of nodes (N)

- Fit bilinear DCM with N = 2, 3, 4, 5, 6, 8, 10
- ELBO increases with N until model complexity exceeds data support
- The peak identifies the "right" circuit size
- L&E cannot do this (no principled selection criterion beyond heuristic)

### Axis 2: Linear vs bilinear

- Same N, compare ELBO of `dx/dt = Ax + Cu` vs `dx/dt = Ax + u_j B_j x + Cu`
- If bilinear wins, context-dependent connectivity is supported by data
- If linear wins, the RNN uses a different mechanism (possible for some tasks)

### Axis 3: B mask topology

- Same N, same bilinear form, compare different B_j masks
- Which connections are modulated by context? Let ELBO decide.
- This is a form of Bayesian structure learning for the circuit

### Interpretive caution

ELBO comparison is approximate (variational gap). For critical claims,
supplement with posterior predictive checks and perturbation validation.
Do not claim "ELBO proves X" -- say "ELBO favors X, supported by
perturbation validation."

---

## Dimensionality Reduction Approaches (H -> N)

### Approach 1: PCA (RECOMMENDED for v0.6.0)

- Apply PCA to trial-averaged hidden state trajectories
- Take top-N principal components
- N selected by variance explained threshold or ELBO comparison
- **Pro:** Standard, reproducible, no optimization needed
- **Con:** PCA axes are not task-aligned; may split task-relevant variance
  across multiple components

### Approach 2: Task-variable regression

- Find linear axes in RNN space that best predict task variables
  (context, stimulus, choice)
- Use regression coefficients as basis vectors
- **Pro:** Task-interpretable dimensions
- **Con:** L&E shows regression-based methods miss circuit interactions

### Approach 3: Learned projection (jointly optimized)

- Optimize Q alongside DCM parameters in the Pyro model
- Q maps N-dimensional DCM states to H-dimensional RNN space
- **Pro:** Best possible fit; Q adapts to what DCM needs
- **Con:** More parameters, harder optimization, Q is not orthonormal
  unless constrained

### Recommendation

Use PCA as the primary approach for v0.6.0. It is transparent, standard,
and reproducible. The ELBO comparison (varying N) effectively selects how
many PCA components to keep. A learned projection is a natural extension
but adds optimization complexity -- defer to v0.6.1 if PCA results are
promising.

---

## MVP Recommendation

For a minimal viable paper, prioritize in this order:

### Wave 1 (core method)
1. **TS-9** Synthetic validation (bilinear ground truth -> bilinear DCM)
2. **TS-4** Direct observation model
3. **D-1** Posterior uncertainty (comes free with SVI)
4. **TS-10** Figures 3, 7 (posterior + recovery)

### Wave 2 (RNN pipeline)
5. **TS-1** RNN trainer (CDDM task)
6. **TS-2** Trajectory extraction
7. **TS-3** PCA dimensionality reduction
8. **TS-5** DCM fitting to RNN trajectories
9. **TS-6** Reconstruction metrics
10. **TS-10** Figures 1, 2, 4

### Wave 3 (model comparison + validation)
11. **TS-8** ELBO model comparison (N selection, linear vs bilinear)
12. **D-3** ELBO architecture selection analysis
13. **TS-7** Perturbation validation
14. **TS-10** Figures 5, 6

### Wave 4 (analysis + paper)
15. **D-4** Misspecification analysis
16. **D-5** Guide type comparison
17. **TS-10** Figures 8 + supplementary

### Defer to post-paper
- AF-2 (Neural ODE extension)
- AF-3 (Real neural data)
- AF-5 (Amortized guides)

---

## Sources

### Primary reference
- [Langdon & Engel (2025) - Latent circuit inference, Nature Neuroscience](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/) -- HIGH confidence, full paper extracted
- [engellab/latent_circuit_inference GitHub](https://github.com/engellab/latent_circuit_inference) -- HIGH confidence, reference implementation
- [engellab/trainRNNbrain GitHub](https://github.com/engellab/trainRNNbrain) -- HIGH confidence, RNN training package

### Related work
- [Langdon et al. (2025) - Single-unit activations, Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01127-2) -- MEDIUM confidence (paywall, extracted from search summaries)
- [Ji-An et al. (2025) - Discovering cognitive strategies with tiny RNNs, Nature](https://www.nature.com/articles/s41586-025-09142-4) -- MEDIUM confidence
- [Bilinear state space systems for nonlinear dynamical modelling](https://link.springer.com/article/10.1007/s12064-000-0001-9) -- MEDIUM confidence

### Dimensionality reduction
- [Dimensionality reduction in neuroscience, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6132250/) -- MEDIUM confidence

### Existing Pyro-DCM code
- `src/pyro_dcm/forward_models/neural_state.py` -- verified, bilinear NeuralStateEquation
- `src/pyro_dcm/models/task_dcm_model.py` -- verified, bilinear Pyro model
- `src/pyro_dcm/models/guides.py` -- verified, 6 guide types + SVI runner
