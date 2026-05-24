---
type: "research"
scope: "pitfalls"
milestone: "v0.6.0"
branch: "latent-circuit-dcm"
domain: "Fitting bilinear DCM to RNN hidden-state trajectories for circuit distillation"
updated: "2026-05-24"
confidence: "MEDIUM-HIGH for model misspecification and degeneracy; MEDIUM for prior/scale issues; LOW for task-design guidance"
scope_note: >
  Pitfalls specific to v0.6.0 Latent Circuit DCM -- fitting bilinear DCM
  (dx/dt = Ax + sum_j u_j B_j x + Cu) to trained RNN hidden-state
  trajectories via a direct observation model (no balloon-Windkessel).
  Does NOT repeat v0.3.0 bilinear pitfalls (those still apply; see
  .planning/research/v0.3.0/PITFALLS.md). Covers what appears NEW when
  fitting DCM to RNN dynamics instead of BOLD data.
---

# Pitfalls: Latent Circuit DCM (v0.6.0)

v0.3.0 bilinear DCM (dx/dt = Ax + sum_j u_j B_j x + Cu) is shipped for
BOLD data with hemodynamic observation model. v0.6.0 repurposes the same
bilinear dynamics but swaps the observation model:

    BOLD path:   x(t) -> balloon-Windkessel -> BOLD signal y(t)
    RNN path:    x(t) -> C_obs @ x(t) + noise = h_projected(t)

The DCM distills a trained RNN into a small interpretable circuit.
Ground truth for RNN distillation claims: Langdon & Engel (2025, Nature
Neuroscience), JSLDS (Smith, Linderman & Sussillo, NeurIPS 2021), LINT
(Valente, Pillow & Ostojic, NeurIPS 2022), MINDy (Singh et al., 2020).

Fourteen v0.3.0 pitfalls (B1-B14) still apply when fitting DCM to RNN
trajectories, but several change character significantly (see final section).

---

## Summary

| # | Pitfall | Severity | Primary Phase |
|---|---------|----------|---------------|
| LC1 | Bilinear misspecification of nonlinear RNN dynamics | Critical | Validation design |
| LC2 | Rotational degeneracy in RNN state space | Critical | Dimensionality reduction |
| LC3 | PCA discards task-relevant oblique dynamics | High | Dimensionality reduction |
| LC4 | Scale mismatch between RNN hidden states and DCM priors | High | Prior specification |
| LC5 | Observation model choice creates identifiability confound | High | Model definition |
| LC6 | Context encoded in state, not input-modulated connectivity | High | Bilinear model design |
| LC7 | Validation without a shared coordinate system | High | Recovery benchmark |
| LC8 | DCM node count overfitting / underfitting tradeoff | Medium | Model selection |
| LC9 | Task design determines distillability | Medium | RNN training |
| LC10 | Temporal resolution mismatch (RNN dt vs DCM dt) | Medium | Integration with existing ODE code |
| LC11 | Ensemble non-convergence from optimization landscape | Medium | Fitting pipeline |
| LC12 | Direct observation model removes hemodynamic smoothing | Medium | Numerical stability |
| LC13 | B_j matrices absorb nonlinearity residuals | Medium | Interpretation |
| LC14 | Solution degeneracy across RNN training seeds | Low | Experimental design |

---

## Critical Pitfalls

### LC1: Bilinear Misspecification of Nonlinear RNN Dynamics

**What goes wrong.** The RNN has dynamics dx/dt = -x + f(W_rec x + W_in u)
where f is a nonlinearity (tanh, ReLU, GRU gates). Bilinear DCM has
dx/dt = Ax + sum_j u_j B_j x + Cu. The bilinear form is a first-order
Taylor expansion of the RNN around a fixed point, augmented with input-
modulated connectivity. When the RNN dynamics are strongly nonlinear --
trajectories far from fixed points, transient amplification through non-
normal dynamics, saturation effects -- the bilinear approximation breaks
down. The recovered A and B matrices are meaningless projections of
nonlinear dynamics onto a linear subspace.

**When it works despite misspecification.** Published evidence suggests
bilinear/linear approximation works surprisingly well in specific regimes:

1. **Near fixed points.** When the RNN operates near stable fixed points
   and task computation involves transitions between attractor basins,
   the linearization around each fixed point is locally accurate. Bilinear
   DCM captures the input-modulated transitions. This is the regime where
   JSLDS (Smith et al. 2021) succeeds.

2. **Low-rank dynamics.** When the RNN uses rank-1 or rank-2 connectivity
   (LINT regime, Valente et al. 2022), the effective dynamics are
   inherently low-dimensional and often well-approximated by linear models.

3. **Slow/integrated variables.** Nozari et al. (2024) showed that spatial
   and temporal averaging linearizes macroscopic brain dynamics. For RNNs,
   the analog is that averaging over many hidden units (via PCA projection)
   may linearize the effective dynamics seen by DCM.

**When it fails.**

1. **Strong saturation.** tanh-based RNNs near saturation (|h| > 1.5) have
   derivatives approaching zero. The linearization predicts continued
   growth; the RNN saturates. Recovered A eigenvalues will be systematically
   wrong.

2. **ReLU kink dynamics.** ReLU RNNs exhibit different local linear maps on
   each side of the activation threshold. A single A matrix cannot capture
   piecewise-linear dynamics unless the trajectory stays in one region.

3. **GRU/LSTM gating.** Gated units implement multiplicative interactions
   between state and input that are fundamentally higher-order than
   bilinear. The bilinear B_j captures u*x interactions, but GRU gates
   involve u*x*sigmoid(x) -- a trilinear or higher interaction.

4. **Non-normal transient amplification.** Both Langdon & Engel (2025) and
   the context-dependent PFC study (Hristos et al. 2024, Science Advances)
   found non-normal dynamics with transient amplification in both {A_cx, B}
   and {A, B_cx} models. Bilinear DCM can represent non-normality through
   A, but transient amplification depends sensitively on matrix structure
   that bilinear fitting may not recover precisely.

**Warning signs.**
- Reconstructed trajectories diverge from RNN trajectories during transient
  periods but match during sustained/steady-state periods.
- R-squared of bilinear DCM fit drops below 0.5 on held-out trials.
- Recovered A matrix has different number of stable/unstable modes than the
  RNN Jacobian at the fixed point.
- Residuals are structured (not white noise) -- systematic misfit at
  specific trial epochs.
- Fit quality depends heavily on RNN activation function choice (works for
  tanh near origin, fails for ReLU).

**Prevention.**
1. **Compute RNN operating regime before fitting.** Histogram |h(t)| across
   training data. If >20% of values are in the saturated region (|tanh(h)|
   > 0.9 or h < 0 for ReLU), flag as high-risk for linear approximation.
2. **Linearization quality diagnostic.** At each time point, compute the
   RNN Jacobian J(t) = df/dx|_{x(t)} and the bilinear approximation
   A + sum_j u_j(t) B_j. Report ||J(t) - A_eff(t)||_F / ||J(t)||_F as
   a misspecification index. Values > 0.5 indicate poor local fit.
3. **Use continuous-time tanh RNNs (CTRNNs) for initial validation.** The
   CTRNN dx/dt = -x + tanh(W_rec x + W_in u) is closest to the bilinear
   regime near x=0 where tanh(x) approx x. Validate on CTRNN before
   attempting GRU/LSTM.
4. **Include a nonlinear baseline (Langdon & Engel model) for comparison.**
   If dx/dt = -x + f(w_rec x + w_in u) explains significantly more
   variance than bilinear DCM, the misspecification matters.

**Phase relevance.** Validation design -- must be addressed before
interpreting any DCM fit to RNN data. Gate: do not claim A/B recovery
without reporting misspecification index.

**Confidence.** HIGH on mechanism (well-established nonlinear dynamics
theory). MEDIUM on regime boundaries (depends on specific RNN architecture
and task).

---

### LC2: Rotational Degeneracy in RNN State Space

**What goes wrong.** For any invertible matrix P, the transformation
h -> P h, W_rec -> P W_rec P^{-1}, W_in -> P W_in, W_out -> W_out P^{-1}
preserves the RNN's input-output behavior exactly. This means the RNN's
"true" connectivity W_rec is defined only up to a similarity transform.
When fitting DCM to RNN hidden states h(t), the recovered A matrix is
A_DCM, which satisfies A_DCM approx Q^T W_rec Q (Langdon & Engel Eq. 5)
only if Q spans an invariant subspace. But PCA selects axes by variance,
not by invariance -- so the DCM A matrix lives in an arbitrary rotated
coordinate system.

**Consequences for bilinear DCM.**

1. **A and B cannot be compared to W_rec directly.** The recovered A is
   A_DCM = R^T W_rec R for some unknown rotation R that depends on the PCA
   basis, the training seed, and the data. Two RNNs trained on the same
   task with different seeds will have different W_rec (up to similarity
   transform) and different PCA bases, producing incomparable A_DCM
   matrices.

2. **B_j has the same degeneracy.** Bilinear modulation B_j represents
   input-modulated connectivity in the PCA-rotated basis. The
   interpretation "input j strengthens the connection from node 2 to
   node 3" is meaningless unless nodes 2 and 3 correspond to interpretable
   neural populations -- which they generally do not in PCA space.

3. **Validation requires basis alignment.** Comparing recovered A to
   ground-truth W_rec requires finding an alignment matrix that minimizes
   ||A_DCM - R^T W_rec R||_F over orthogonal R. This is the Procrustes
   problem, which has a closed-form solution but introduces its own biases
   (Procrustes always finds some alignment, even for unrelated matrices).

**How Langdon & Engel (2025) handle this.** They parameterize Q via the
Cayley transform Q = (I + A)(I - A)^{-1} pi_n where A = B - B^T is
skew-symmetric. Q is jointly optimized with the circuit parameters,
so the embedding is learned rather than fixed by PCA. They validate by
"conjugating the RNN connectivity matrices with the embedding matrix Q"
and checking that Q^T W_rec Q matches w_rec (correlation r = 0.89).
Critically, they also validate through perturbation experiments (Eq. 6)
-- translating circuit perturbations into rank-one modifications of the
RNN and verifying behavioral effects.

**Our approach vs theirs.** We use PCA then fit DCM in PCA space. This
is simpler but loses the invariant-subspace property. The rotation R is
fixed by PCA before DCM fitting begins, so DCM cannot discover that a
different rotation would produce a better-fitting bilinear model.

**Warning signs.**
- A_DCM recovered from two RNN seeds are dissimilar even after Procrustes
  alignment (correlation < 0.5).
- Sign structure of A_DCM changes when PCA is recomputed on different
  trial subsets.
- B_j matrices have no consistent pattern across seeds.
- Variance explained by top-N PCs is similar across seeds but A_DCM
  differs -- the issue is rotation, not dimensionality.

**Prevention.**
1. **Use Procrustes alignment for all cross-seed comparisons.** Compute
   optimal orthogonal R minimizing ||A_DCM^{(1)} - R A_DCM^{(2)} R^T||_F.
   Report aligned correlation, not raw correlation.
2. **Validate through perturbation, not parameter comparison.** Following
   Langdon & Engel: translate a DCM perturbation (e.g., lesion A_ij) back
   to the RNN via the PCA projection, run the perturbed RNN, and check
   that behavior changes as DCM predicts. This is rotation-invariant.
3. **Consider joint optimization of projection + DCM.** Instead of
   fixing PCA then fitting DCM, jointly optimize a linear projection
   C_embed alongside A, B, C. This is closer to Langdon & Engel but
   uses bilinear dynamics instead of their nonlinear f().
4. **Report eigenvalue structure, not element-wise A.** Eigenvalues of A
   are invariant under similarity transforms. Report eigenvalue spectrum,
   not individual A_ij entries, as the primary recovery metric.
5. **Known-connectivity synthetic RNN as ground truth.** Build an RNN
   with known low-rank W_rec where the ground truth A = Q_true^T W_rec
   Q_true is computable. Validate that DCM recovers this A after
   Procrustes alignment.

**Phase relevance.** Dimensionality reduction phase and validation
benchmark. Must be settled before any "parameter recovery" claims.

**Confidence.** HIGH on the mathematical degeneracy (textbook linear
algebra). MEDIUM on practical severity (depends on how well PCA
approximates an invariant subspace for the specific task).

---

## High-Severity Pitfalls

### LC3: PCA Discards Task-Relevant Oblique Dynamics

**What goes wrong.** PCA selects axes that maximize variance. But task-
relevant dynamics can be encoded in low-variance directions -- "oblique
dynamics" (Dubreuil et al. 2024, eLife). In oblique RNNs, the output
weights are nearly orthogonal to the largest principal components. This
means PCA projects out exactly the dynamics that matter for the task.

Concretely: a 3-node DCM fitted in the top-3 PCA dimensions may explain
90% of state variance but capture 0% of task-relevant computation if
the RNN uses oblique dynamics. The bilinear A and B matrices would
describe high-variance spontaneous fluctuations, not the task-driven
circuit.

**Published evidence.**
- **Dubreuil et al. (2024, eLife).** Aligned networks need 2 PCs for
  R^2 = 0.99 on output; oblique networks need 8+ PCs. "Output information
  is encoded in low-variance components that standard dimensionality
  reduction overlooks."
- **Langdon & Engel (2025).** Their learned Q matrix jointly optimizes
  for both activity reconstruction and task output fit, avoiding the
  PCA-variance trap. In PFC data, the task-relevant subspace explained
  only 11% and 7% of variance (monkeys A and F).
- **SliceTCA (Pellegrino et al. 2024, Nature Neuroscience).** PCA
  required substantially more components than sliceTCA to capture
  task-relevant structure, confirming PCA's variance bias.

**Warning signs.**
- High variance explained by N PCs (>90%) but poor task output
  reconstruction from those PCs (R^2 < 0.5 on behavioral output).
- Adding the N+1th PC dramatically improves output reconstruction --
  the task is in a low-variance dimension.
- DCM fitted in PCA space has poor predictive accuracy on held-out
  trials even when training fit is good.
- Systematically better fit on "spontaneous" (no-input) epochs than
  on task-driven epochs.

**Prevention.**
1. **Always check task-output reconstruction in PCA space.** Before
   fitting DCM, verify that W_out projected into the PCA subspace
   has non-negligible norm. Report R^2 of output reconstruction from
   top-N PCs as a quality gate.
2. **Consider task-informed dimensionality reduction.** Instead of PCA,
   use demixed PCA (dPCA) or targeted dimensionality reduction (TDR)
   that selects axes explaining task variables, not variance.
3. **Or: include output reconstruction loss in DCM fitting.** Add a
   term lambda * ||z_pred - W_out Q x_dcm||^2 to the ELBO, forcing
   DCM trajectories to predict task output. This mimics Langdon &
   Engel's joint fitting but within the Pyro framework.
4. **Report variance-explained AND output-R^2 for every DCM fit.**
   Variance alone is insufficient validation.
5. **Start with aligned-dynamics RNNs for initial validation.** Train
   RNNs with small output weights (aligned regime) where PCA works
   well, then test generalization to oblique.

**Phase relevance.** Dimensionality reduction and validation. Gate: if
output R^2 in PCA space < 0.7, PCA is insufficient and task-informed
reduction is required.

**Confidence.** HIGH. Dubreuil et al. (2024) directly demonstrates this
failure mode. Langdon & Engel's solution (learned embedding) is verified.

---

### LC4: Scale Mismatch Between RNN Hidden States and DCM Priors

**What goes wrong.** DCM priors (A_free ~ N(0, 1/64), B_free ~ N(0, 1.0))
were calibrated for BOLD data where:
- A matrix values are typically 0.05-0.5 Hz (slow hemodynamic timescale)
- State variables x are small perturbations around zero (BOLD percent
  signal change ~ 1-3%)
- Time constants are seconds (hemodynamic response ~ 5-6s)

RNN hidden states have completely different scales:
- W_rec values may be O(1) to O(1/sqrt(N)) depending on initialization
- Hidden states h(t) may span [-3, 3] (tanh) or [0, 10+] (ReLU)
- Time constants are set by the dt of the RNN (often 10-100ms for
  neural simulations, or arbitrary for cognitive tasks)

The A_free ~ N(0, 1/64) prior centers near zero with std = 0.125. But
if the ground-truth effective connectivity in the PCA-projected space
is O(1), this prior will heavily shrink A toward zero, producing
systematic underestimation. This is the same mechanism as the v0.3.0
B-RMSE shrinkage (Phase 16.1 RECOV-04 failure) but potentially worse
because the scale mismatch affects ALL parameters, not just B.

**Warning signs.**
- Posterior A_ij means systematically smaller than expected from W_rec.
- Posterior shrinkage ratio (std_post / std_prior) near 1.0 for many
  elements -- prior dominates, data is not informing the posterior.
- DCM fit quality improves dramatically when prior variance is widened
  by 10x-100x.
- Changing RNN hidden state normalization changes recovered A
  proportionally.

**Prevention.**
1. **Normalize RNN hidden states before DCM fitting.** Z-score each PC
   trajectory to zero mean, unit variance. This puts states in a
   standardized range comparable to DCM's assumptions. Document the
   normalization and its inverse (needed for perturbation experiments).
2. **Recalibrate priors for the RNN regime.** Replace A_free ~ N(0, 1/64)
   with A_free ~ N(0, sigma_A^2) where sigma_A is estimated from the
   RNN's Jacobian eigenvalues. Similarly for B and C.
3. **Empirical prior calibration.** Fit DCM to 5-10 synthetic RNNs with
   known parameters. Adjust priors so that posterior coverage is well-
   calibrated (90% CI captures truth in ~90% of cases).
4. **Time-scale normalization.** If the RNN uses dt_rnn, the DCM A matrix
   has units of 1/dt_rnn. Rescale A prior accordingly: if dt_rnn = 10ms,
   A values are ~100x larger than for BOLD with dt ~ 1s.
5. **Separate prior variance constant for RNN-mode.** Define
   A_PRIOR_VARIANCE_RNN and B_PRIOR_VARIANCE_RNN in the model, distinct
   from the BOLD-calibrated constants. Make this a required argument in
   the direct-observation model constructor so it cannot be accidentally
   defaulted.

**Phase relevance.** Prior specification -- must be resolved in model
definition phase, before any fitting runs. Risk of repeat of Phase 16.1
RECOV-04 failure.

**Confidence.** HIGH on the mechanism (direct consequence of prior
calibration for different data regimes). MEDIUM on the magnitude (depends
on specific RNN architecture and normalization choices).

---

### LC5: Observation Model Choice Creates Identifiability Confound

**What goes wrong.** Three observation model options exist, each with
different identifiability properties:

1. **Direct identity: y = x + noise.** Assumes DCM states ARE the PCA-
   projected RNN states. Simple but creates a hard constraint: the
   number of DCM nodes N must equal the number of PCA components kept.
   No freedom to have fewer DCM nodes than PCA dimensions. If N is
   too small, the model is misspecified; if N matches PCA dims, there
   may be too many parameters.

2. **Linear projection: y = C_obs @ x + noise.** DCM states x are latent;
   C_obs maps them to the observed PCA-projected states. More flexible:
   N_dcm < N_pca is possible. But C_obs introduces N_pca * N_dcm new
   parameters and creates an identifiability confound: A and C_obs can
   trade off (the same observations can be explained by different A with
   compensating C_obs).

3. **Learned nonlinear readout.** Most flexible but defeats the purpose
   of interpretable distillation.

**The Langdon & Engel approach.** They use y = Q x where Q is orthonormal.
This is equivalent to option 2 but with the constraint Q^T Q = I, which
eliminates the scaling ambiguity. The orthonormality constraint means that
Q is a rotation, not a general projection, preserving metric structure.
They optimize Q jointly with circuit parameters.

**Why this matters for bilinear DCM.** If using option 2 without
orthonormality, the A-C_obs degeneracy means that SVI can find solutions
where A is near-zero and C_obs is large (or vice versa). The "recovered
connectivity" A would be meaningless.

**Warning signs.**
- A matrix posterior mean near zero but C_obs values very large (or vice
  versa).
- Very different A matrices recovered when C_obs is initialized
  differently.
- ELBO landscape has multiple modes with similar ELBO but different A.
- Model fit is good (low residuals) but A doesn't match any reasonable
  connectivity.

**Prevention.**
1. **Start with direct identity (option 1).** For initial validation on
   synthetic RNNs, use N_dcm = N_pca and y = x + noise. This eliminates
   the C_obs confound entirely. Validate A/B recovery first, then
   optionally add C_obs.
2. **If using C_obs, constrain to orthonormal.** Parameterize C_obs via
   the Cayley transform (following Langdon & Engel) or via Householder
   reflections. This prevents the scaling degeneracy.
3. **Or: fix C_obs from PCA loadings.** Set C_obs to the PCA loading
   matrix and do not optimize it. This is equivalent to "PCA then fit
   in PCA space" with extra notation.
4. **Add a regularization term penalizing ||A|| + ||C_obs|| to break
   the degeneracy.** But this introduces a regularization hyperparameter.
5. **Report which observation model was used.** A/B recovery metrics
   are not comparable across observation model choices.

**Phase relevance.** Model definition phase. Decide observation model
before any fitting code is written.

**Confidence.** HIGH on the identifiability issue (standard linear
algebra). HIGH that identity model is safest for initial validation.

---

### LC6: Context Encoded in State, Not Input-Modulated Connectivity

**What goes wrong.** Bilinear DCM assumes context modulates connectivity:
A_eff(t) = A + sum_j u_j(t) B_j. This means context acts multiplicatively
on state through input-gated connectivity changes. But trained RNNs can
encode context in at least two other ways that bilinear DCM cannot capture:

1. **Context in hidden state.** The RNN maintains a persistent state
   representation of context (e.g., a sustained activity pattern
   indicating "attend to color"). The context does not modulate
   connectivity; it provides an additive bias to the dynamics. This is
   captured by the C*u term, not B_j.

2. **Context through gain modulation.** The context scales the gain of
   individual units (multiplicative modulation of single units, not
   connections). This is a diagonal B_j matrix, which v0.3.0
   discourages via Pitfall B5 (stability risk from positive diagonal).

3. **Context through gating.** GRU/LSTM gates implement context-dependent
   filtering that is not bilinear -- the gate is a function of BOTH
   input and state (sigmoid(W_gate @ [h; u])), making it a state-
   dependent modulation, not an input-dependent one.

**Published evidence.**
- **Hristos et al. (2024, Science Advances).** Found that two
  fundamentally different mechanisms of context-dependent input selection
  explain PFC responses equally well: context-dependent dynamics {A_cx, B}
  vs context-dependent inputs {A, B_cx}. Both are linear. The critical
  finding: "both mechanisms ultimately relied on a context-dependent
  realignment between the inputs and a subset of the modes."
- **Dubreuil et al. (2022, Theory of Gating in RNNs, PLoS CB).** Gating
  provides flexible control of collective dynamics timescales.
  Gain modulation at the network level is equivalent to switching on/off
  neuron subpopulations -- a mechanism that bilinear B_j (connection-
  level modulation) cannot represent without diagonal elements.

**When bilinear B_j works anyway.** If the RNN's context modulation
primarily affects inter-regional connection strengths (e.g., "attention
to color strengthens the V4-to-PFC pathway"), bilinear B_j is the correct
parameterization. This is the regime DCM was designed for. Empirically,
tasks with clear modulatory structure (e.g., "condition A strengthens
pathway X") produce RNN dynamics that bilinear DCM captures well.

**Warning signs.**
- B_j matrices are near-zero or have no consistent structure across seeds,
  but the RNN clearly uses context (behavioral accuracy differs across
  conditions).
- DCM model with B_j fits no better than linear-only (A + Cu) model --
  the ELBO improvement from adding B is negligible.
- Condition-averaged trajectories separate in PCA space via offset
  (additive, captured by C), not via different dynamics (multiplicative,
  captured by B_j).
- Task requires sustained context maintenance (working memory) rather
  than transient modulation.

**Prevention.**
1. **Pre-check: does the RNN show condition-dependent dynamics or just
   condition-dependent offsets?** Compute the Jacobian of the RNN at
   matched states under different conditions. If Jacobians are similar,
   context is additive (C pathway); if different, context modulates
   dynamics (B pathway).
2. **Include condition-dependent C as an alternative model.** Fit both
   bilinear (A + u_j B_j) and condition-offset (A + C_{cx}) models.
   Compare via ELBO model comparison (already in Pyro-DCM from v0.2.0).
3. **Allow B_j diagonal for RNN fitting (unlike BOLD DCM).** The
   stability concern (v0.3.0 Pitfall B5) is about BOLD hemodynamic
   blowup, which does not apply when the observation model is direct.
   Use the -exp transform on diagonal B elements to maintain stability:
   B_jii_eff = -exp(B_free_jii) / factor.
4. **Document bilinear DCM's representational limits clearly.** Bilinear
   captures input-gated connectivity changes. It does not capture
   state-gated modulation. This is a fundamental architectural
   limitation, not a bug.

**Phase relevance.** Model design and validation interpretation. Must be
considered when choosing which RNN tasks to validate against.

**Confidence.** HIGH on the mechanism. Hristos et al. (2024) directly
demonstrated the {A_cx, B} vs {A, B_cx} degeneracy. MEDIUM on practical
severity (task-dependent).

---

### LC7: Validation Without a Shared Coordinate System

**What goes wrong.** When fitting DCM to a trained RNN, the natural
validation question is "does DCM recover the RNN's connectivity W_rec?"
But this question is ill-posed because:

1. **Dimension mismatch.** W_rec is (H, H) where H is the number of
   hidden units (e.g., 100). A is (N, N) where N is the number of
   DCM nodes (e.g., 3-5). There is no element-wise comparison possible.

2. **Coordinate mismatch.** Even after projecting W_rec into the PCA
   subspace (A_proj = Q^T W_rec Q), the PCA basis Q is not guaranteed
   to span the same subspace as the DCM state space. If DCM uses a
   different effective coordinate system (e.g., because B_j absorbed
   some rotation), element-wise comparison of A and A_proj is meaningless.

3. **Nonlinearity absorption.** The "true" linearized connectivity seen
   by DCM is not W_rec but the Jacobian of the full RNN dynamics
   J = -I + diag(f'(W_rec h + W_in u)) @ W_rec, which depends on the
   operating point h(t). At different time points, J differs. DCM
   recovers a single time-averaged A, which may not match J at any
   specific time.

**What metrics SHOULD be used instead of element-wise A recovery.**

1. **Dynamical similarity.** Compare trajectories: given initial
   conditions and inputs, does DCM produce trajectories that match the
   RNN's? Use R^2 on held-out trials as the primary metric. This is
   coordinate-system-free.

2. **Perturbation prediction.** Following Langdon & Engel: remove or
   modify a connection in DCM, translate to RNN modification via the
   projection, and check if behavior changes match. This validates
   the functional role of specific A/B elements.

3. **Eigenvalue spectrum comparison.** Eigenvalues of A_DCM vs
   eigenvalues of Q^T J_avg Q (time-averaged Jacobian projected into
   PCA space). Eigenvalues are rotation-invariant.

4. **Sign/structure recovery on known-connectivity RNNs.** Build
   synthetic RNNs with known low-rank W_rec structure. After Procrustes
   alignment, check that the sign pattern of A_DCM matches the
   known ground truth.

**Warning signs.**
- High trajectory R^2 but low element-wise A recovery -- this is
  EXPECTED and correct. Do not treat it as a failure.
- Element-wise A recovery looks good on one seed but fails on another
  (rotational degeneracy, LC2).
- Perturbation predictions are wrong even when trajectory fit is good
  (overfitting to training trajectories).

**Prevention.**
1. **Define validation metrics BEFORE fitting.** Primary: trajectory R^2
   on held-out trials. Secondary: perturbation prediction accuracy on
   2-3 pre-specified perturbations. Tertiary: eigenvalue spectrum
   correlation. Do NOT use element-wise A RMSE as primary metric.
2. **Known-connectivity synthetic RNN as the validation anchor.** Build
   a CTRNN with W_rec = Q_true @ A_true @ Q_true^T (low-rank, known
   ground truth in the projected space). Validate that DCM recovers
   A_true after Procrustes alignment. This is the one case where
   element-wise comparison is meaningful.
3. **Report Procrustes-aligned metrics with significance test.** Compare
   Procrustes R^2 against a null distribution (random orthogonal
   matrices). Report p-value alongside correlation.

**Phase relevance.** Recovery benchmark design. This is the first thing
to settle -- all subsequent claims depend on the validation framework.

**Confidence.** HIGH on the issue (fundamental coordinate-system mismatch).
HIGH on the solution (trajectory-based and perturbation-based validation
are well-established in the Langdon & Engel framework).

---

## Medium-Severity Pitfalls

### LC8: DCM Node Count Overfitting / Underfitting Tradeoff

**What goes wrong.** The number of DCM nodes N determines model capacity:
- Too few nodes (N < task dimensionality): DCM cannot represent the
  dynamics. Bilinear approximation becomes worse because more nonlinear
  structure is compressed into fewer dimensions.
- Too many nodes (N >> task dimensionality): DCM overfits RNN noise and
  transient dynamics. B_j matrices fit noise rather than genuine
  modulation. ELBO improves but interpretation degrades.

Standard model selection tools (AIC, BIC) are unreliable for dynamical
systems because they assume i.i.d. samples, but dynamical trajectories
are temporally correlated. ELBO-based model comparison (already in
Pyro-DCM from v0.2.0) is better but still sensitive to observation model
specification.

**Published evidence.**
- **Information criteria for dynamical systems (2025).** "AIC and BIC
  assume uncorrelated samples, an assumption systematically violated by
  dynamical systems data." Model selection depends sensitively on
  sampling rate and system dimensionality.
- **Neural ODE autoencoders.** "RNN-based state-space autoencoders
  require many more latent dimensions than the synthetic systems they
  are attempting to model."

**Warning signs.**
- ELBO keeps improving as N increases (no clear elbow).
- B_j matrices have many non-zero elements at large N but only 1-2
  at small N (structure is N-dependent).
- Cross-validated trajectory R^2 peaks at intermediate N then decreases.
- Eigenvalues of A at large N include many near-zero values (excess
  dimensions are not dynamically active).

**Prevention.**
1. **Sweep N systematically (N = 2, 3, 4, 5, ..., 10) and report
   held-out trajectory R^2 as the model selection criterion.** Use
   held-out TRIALS, not held-out time points (temporal correlation makes
   time-point-level cross-validation misleading).
2. **Use ELBO model comparison across N values.** The Pyro-DCM v0.2.0
   infrastructure supports this directly. Report delta-ELBO between
   consecutive N values.
3. **Report effective dimensionality of the B_j structure.** At each N,
   how many B_j elements are non-zero (posterior 90% CI excludes zero)?
   Stable B structure across N values suggests the right N.
4. **Use the RNN's task-relevant dimensionality as a starting point.**
   Compute dPCA or TDR dimensionality of the RNN for the task. This
   gives an informed prior on N.
5. **Eigenvalue analysis of the RNN Jacobian.** The number of eigenvalues
   with |real part| > threshold gives a dynamics-informed estimate of N.

**Phase relevance.** Model selection phase; needed before benchmark
design.

**Confidence.** MEDIUM. Trade-off is well-established; specific guidance
for bilinear DCM on RNN data is novel (no published benchmarks).

---

### LC9: Task Design Determines Distillability

**What goes wrong.** Not all RNN tasks produce dynamics amenable to
bilinear DCM distillation. The choice of task and training objective
determines what dynamical structures the RNN learns:

1. **Fixed-point tasks** (e.g., Go/NoGo, categorization with sustained
   response): RNN dynamics converge to attractor states. Bilinear DCM
   captures transitions between attractors well. GOOD for distillation.

2. **Integration tasks** (e.g., evidence accumulation, random dot
   motion): RNN learns line attractors or slow manifolds. Linear DCM
   captures line attractors; bilinear captures how inputs modulate
   integration speed. GOOD for distillation.

3. **Oscillatory tasks** (e.g., motor timing, rhythm generation): RNN
   learns limit cycles. Linear DCM captures oscillations via complex
   eigenvalues of A. GOOD for distillation.

4. **Working memory tasks** (e.g., delayed match-to-sample): RNN may
   use nonlinear attractor mechanisms (bistable states) or persistent
   activity via near-unit eigenvalues. Bilinear DCM struggles with
   bistable attractors (requires two fixed points, which a single A
   cannot represent). MODERATE for distillation.

5. **Sequence generation / transient dynamics tasks**: RNN uses transient
   trajectories in high-dimensional space with no stable attractor. Linear
   approximation is poor because trajectories traverse multiple operating
   regimes. BAD for distillation.

**Prevention.**
1. **Start with context-dependent decision tasks** (Langdon & Engel's
   choice). These have clear modulatory structure (attention/context
   gates evidence flow) and the RNN typically learns low-rank dynamics.
2. **Verify dynamics type before fitting.** Run fixed-point finding
   (e.g., using optimization on ||f(h) - h|| for discrete-time or
   ||dh/dt|| for continuous-time). Count stable and unstable fixed
   points. If > 1 stable fixed point per condition, bilinear DCM will
   struggle.
3. **Document which task types are validated** in the v0.6.0 benchmark.
   Do NOT generalize results from one task type to all tasks.
4. **Consider JSLDS as a comparator for multi-attractor tasks.** JSLDS
   explicitly handles switching between linear regimes, which bilinear
   DCM cannot.

**Phase relevance.** RNN training phase and experimental design.

**Confidence.** MEDIUM. Theoretical grounding is solid (attractor
landscape theory), but empirical validation of distillability across
task types has not been published.

---

### LC10: Temporal Resolution Mismatch (RNN dt vs DCM dt)

**What goes wrong.** The existing DCM ODE integration uses dt = 0.5s
(task_dcm_model default) because BOLD data is sampled at TR = 1-3s
and the hemodynamic response is slow (~5s). For RNN hidden states:
- The RNN may use dt_rnn = 1ms-100ms (neural timescale) or arbitrary
  units.
- Dynamics can be much faster than hemodynamic timescales.
- The existing rk4 step size of 0.5s may be far too coarse for fast RNN
  dynamics, causing integration errors.

Conversely, if the RNN uses very fine dt (1ms over 10s = 10,000 steps),
the ODE integrator may need proportionally more steps, hitting the memory
pressure pitfall (v0.3.0 B10).

**Warning signs.**
- DCM trajectories lag behind RNN trajectories (integration too coarse).
- Reducing dt by 2x changes recovered A substantially (not converged).
- Very long fitting times due to fine-grained ODE integration.
- Numerical instability (NaN) at coarse dt that disappears at fine dt.

**Prevention.**
1. **Make dt a required parameter in the direct-observation model
   constructor**, not a default. Force the user to think about it.
2. **dt-invariance test.** Simulate at dt and dt/2; assert trajectory
   match to atol=1e-3. Include in the validation suite.
3. **Recommend normalizing RNN dynamics to a ~1s timescale.** If the
   RNN uses dt_rnn = 10ms, rescale: A_rescaled = A * (dt_rnn / 1.0).
   This puts the dynamics in a regime where dt = 0.01-0.05 is sufficient.
4. **Consider dopri5 (adaptive) instead of rk4 for RNN fitting.** The
   v0.3.0 rationale for rk4 (predictable runtime during SVI) may be
   less important here because RNN fitting is typically done per-RNN
   (not per-subject in a clinical pipeline). Adaptive stepping handles
   varying timescales automatically.

**Phase relevance.** Integration with existing ODE code. Must be
addressed in the direct-observation forward model implementation.

**Confidence.** HIGH. Standard numerical integration concern.

---

### LC11: Ensemble Non-Convergence from Optimization Landscape

**What goes wrong.** Langdon & Engel report fitting "100 latent circuit
models with different initialization" and selecting "10 with highest fit
quality." This implies the optimization landscape has many local minima.
Bilinear DCM fitted via SVI inherits this problem:

- The ELBO landscape for bilinear DCM on RNN data has local optima
  corresponding to different rotations of the A matrix (LC2), different
  allocations of dynamics between A and B (LC5), and different mappings
  between DCM nodes and RNN PCs.
- Single-run SVI may converge to a poor local optimum.
- The v0.3.0 Phase 16.1 RECOV-04 experience showed that even for BOLD
  data, guide initialization (init_scale) critically affects convergence.

**Warning signs.**
- Different random seeds for SVI produce very different A matrices.
- ELBO variance across seeds is high (>10% of mean ELBO).
- Some seeds produce NaN early; others converge but to different values.
- Best-seed ELBO is much better than median-seed ELBO (>5% gap).

**Prevention.**
1. **Multi-start SVI.** Run 10-20 SVI fits with different seeds. Select
   the fit with best ELBO. This is standard for nonlinear optimization
   and matches Langdon & Engel's approach.
2. **Report ensemble statistics.** For the selected best fit, report
   how many of the 10-20 runs converged to a similar solution (within
   Procrustes distance < threshold). If only 1-2 converge there, the
   result is fragile.
3. **Warm-start from multiple initializations.** Try (a) PCA-informed
   init: initialize A from the time-lagged correlation of PCA-projected
   states, (b) Random init, (c) Jacobian-informed init: initialize A
   from the time-averaged RNN Jacobian projected into PCA space.
4. **Phase 16.1 lesson: init_scale matters.** The RECOV-04 diagnostic
   showed that init_scale = 0.005 was too tight for B recovery. For
   RNN fitting, start with init_scale = 0.1 and sweep if needed.

**Phase relevance.** Fitting pipeline design.

**Confidence.** HIGH on the problem (Phase 16.1 experience + Langdon &
Engel's 100-restart strategy). MEDIUM on mitigation effectiveness.

---

### LC12: Direct Observation Model Removes Hemodynamic Smoothing

**What goes wrong.** In standard BOLD DCM, the balloon-Windkessel
hemodynamic model acts as a low-pass filter: fast neural dynamics are
smoothed into slow BOLD fluctuations. This smoothing has a regularizing
effect -- small errors in neural state dynamics are damped before reaching
the likelihood. With direct observation (y = x + noise), there is no
smoothing. Every time point of the ODE solution directly enters the
likelihood. Consequences:

1. **Gradient magnitudes are larger.** Each ODE integration error
   contributes directly to the ELBO gradient, not through a
   hemodynamic filter. Gradient clipping may need to be tighter.

2. **Sensitivity to initial conditions.** Without hemodynamic smoothing,
   the predicted trajectory is more sensitive to x(0). A small error in
   initial conditions grows exponentially (if max Re(eig(A)) < 0, it
   decays, but slowly for near-marginal eigenvalues).

3. **Temporal correlations in residuals.** ODE integration errors are
   temporally correlated (they propagate through the dynamics). The
   Gaussian likelihood treats each time point independently, which is
   misspecified. This doesn't affect posterior mode but inflates
   posterior confidence.

**Warning signs.**
- ELBO gradients are very large (>100) or very noisy (std > mean).
- SVI diverges at learning rates that worked for BOLD DCM.
- Posterior confidence intervals are too narrow (coverage < 50% at
  nominal 90%).
- Trajectory fit has correlated residuals (Durbin-Watson statistic
  significantly < 2).

**Prevention.**
1. **Use lower learning rates for direct-observation DCM.** Start at
   1e-4 instead of 1e-3. Increase gradient clipping from 10.0 to 5.0.
2. **Consider a multi-step prediction loss** instead of point-by-point
   comparison. Predict k steps ahead and average the loss over k values.
   This is more robust to ODE integration error.
3. **Add AR(1) noise model.** Instead of y = x + epsilon with i.i.d.
   epsilon, use y = x + epsilon_t where epsilon_t = rho * epsilon_{t-1}
   + eta_t. This accounts for temporal correlations. Sample rho as a
   Pyro parameter.
4. **Subsample time points for the likelihood.** Instead of using every
   ODE step, evaluate the likelihood every k-th step. This reduces the
   effective sample size and prevents overconfidence.
5. **Benchmark against BOLD DCM on the same data.** Run both direct and
   BOLD DCM (treating RNN states as "neural activity" and applying a
   synthetic hemodynamic response) to quantify the smoothing effect.

**Phase relevance.** Direct-observation model implementation.

**Confidence.** HIGH on the mechanism (loss of hemodynamic smoothing is a
direct consequence of the model change). MEDIUM on practical severity.

---

### LC13: B_j Matrices Absorb Nonlinearity Residuals

**What goes wrong.** When bilinear DCM is fitted to nonlinear RNN
dynamics, the B_j matrices have a dual role: (1) capturing genuine
input-modulated connectivity changes, and (2) absorbing the residual
misspecification from the bilinear approximation of nonlinear dynamics.
Because B_j enters the model multiplicatively (u_j * B_j * x), it can
partially compensate for nonlinear effects that correlate with the input
u_j.

Example: In a tanh RNN, when u_j is active, the RNN state moves to a
region where tanh(h) is more saturated. The true dynamics are slower
(gain < 1). Bilinear B_j can approximate this by adding negative
connectivity (reducing A_eff eigenvalues), but this "recovered B"
reflects tanh saturation, not genuine connectivity modulation.

**Warning signs.**
- B_j matrices have physiologically implausible structure (e.g.,
  uniformly negative, mimicking gain reduction).
- B_j magnitude correlates with the degree of nonlinearity in the RNN
  (larger B_j when RNN operates further from linear regime).
- Removing nonlinearity (replacing tanh with identity in the RNN)
  eliminates B_j recovery.
- B_j structure changes when the RNN activation function changes (tanh
  vs ReLU) even for the same task.

**Prevention.**
1. **Report LC1 misspecification index alongside B_j.** If the
   linearization quality is poor (index > 0.5), B_j interpretation is
   unreliable.
2. **Validate B_j through intervention, not recovery.** The meaningful
   question is not "does B_j match the RNN's modulation?" but "does
   modifying B_j in DCM predict the effect of modifying u_j in the RNN?"
3. **Compare B_j between bilinear DCM and Langdon & Engel model.** If
   the nonlinear model recovers substantially different modulation
   structure, bilinear B_j is contaminated by misspecification.
4. **Ablation test.** Fit DCM to an RNN with NO input modulation (purely
   context-independent dynamics). B_j should be near zero. If not, the
   B_j is absorbing nonlinearity, not modulation.

**Phase relevance.** Interpretation of results; validation benchmark
design.

**Confidence.** MEDIUM. Mechanism is plausible but the magnitude of
this effect has not been quantified in the literature.

---

### LC14: Solution Degeneracy Across RNN Training Seeds

**What goes wrong.** Different RNN training runs (same architecture, same
task, different random seed) can learn qualitatively different internal
solutions -- "solution degeneracy" (Huang et al., NeurIPS 2025). This is
distinct from rotational degeneracy (LC2, which is about coordinate-system
arbitrariness). Solution degeneracy means the actual dynamical strategy
differs: one RNN might use fixed-point attractors, another might use
limit cycles, for the same task.

If bilinear DCM is validated on one RNN seed and reports good recovery,
the result may not generalize to another RNN seed that learned a
fundamentally different solution.

**Published evidence.**
- **Huang et al. (2025, NeurIPS).** Three levels of degeneracy: behavior,
  neural dynamics, and weight space. Higher task complexity reduces
  dynamical degeneracy. Structural regularization (low-rank, sparsity)
  reduces degeneracy at all levels.
- **Contravariance principle.** Stronger feature learning reduces
  dynamical degeneracy but increases weight degeneracy. This means
  dynamics may be consistent across seeds even when weights differ
  (the ideal case for DCM distillation).

**Warning signs.**
- A matrices recovered from different RNN seeds have different
  eigenvalue spectra (not just rotations -- qualitatively different).
- Some RNN seeds produce DCM fits with high R^2, others with low R^2,
  even when all RNNs have similar task performance.
- Fixed-point analysis reveals different numbers of fixed points
  across seeds.

**Prevention.**
1. **Validate on multiple (>=5) RNN training seeds.** Report DCM fit
   quality statistics across seeds, not just one cherry-picked seed.
2. **Use regularization to reduce RNN solution degeneracy.** Apply
   L2 regularization or low-rank constraints during RNN training to
   push solutions toward a consistent dynamical regime.
3. **Check dynamical consistency before DCM fitting.** Use DSA
   (Dynamical Similarity Analysis) to cluster RNN seeds by dynamical
   similarity. Fit DCM separately to each cluster and compare.
4. **Report DSA distance between DCM-recovered dynamics across seeds.**
   This directly measures whether DCM captures a consistent underlying
   circuit despite RNN variability.

**Phase relevance.** Experimental design and benchmark reporting.

**Confidence.** MEDIUM. Huang et al. (2025) establishes the phenomenon
rigorously, but its impact on bilinear DCM specifically has not been
studied.

---

## v0.3.0 Pitfalls That Change Character Under RNN Fitting

Per instructions not to repeat v0.3.0 unless severity changes:

- **B1 (A_eff stability) -- CHANGES.** With direct observation and no
  hemodynamic smoothing, unstable A_eff causes ODE divergence that
  immediately hits the likelihood (no hemodynamic damping). However, RNN
  hidden states are bounded (tanh: [-1, 1]; ReLU: clipped by training),
  so the ground-truth dynamics are inherently stable. If DCM priors are
  properly calibrated (LC4), stability violations should be rarer.
  **Net effect: mechanism identical, frequency may differ.**

- **B2 (B non-identifiable under sparse events) -- BETTER.** RNN training
  can use arbitrary stimulus designs (not constrained by fMRI scanning
  protocol). Use frequent, sustained modulatory epochs with large effect
  sizes. This is a major advantage of the RNN distillation paradigm.
  **Net effect: addressable by task design.**

- **B5 (free B diagonal causes positive self-coupling) -- REVISIT.**
  For BOLD DCM, diagonal B was discouraged (Pitfall B5) because it
  destabilizes the hemodynamic ODE. For RNN fitting, diagonal B may
  be necessary to capture gain modulation (LC6). The stability concern
  is still real but the observation model change (no balloon-Windkessel)
  means the consequences are less catastrophic (direct observation can
  tolerate brief instability better than hemodynamic cascading).
  **Net effect: may need to allow diagonal B with -exp transform.**

- **B8 (prior variance calibration) -- SUPERSEDED by LC4.** The SPM-
  calibrated prior variance is wrong for RNN hidden states regardless
  of whether we use 1.0 or 1/64. LC4 addresses this comprehensively.
  **Net effect: prior recalibration is mandatory, not optional.**

- **B10 (per-step ODE cost) -- LIKELY WORSE.** RNN trajectories may
  require finer dt than BOLD, increasing step count. But RNN fitting
  is typically done on shorter time series (1-10s of task time vs 500s
  of fMRI), which partially compensates.
  **Net effect: wall-time impact depends on specific timescales.**

- **B11 (SVI convergence) -- LIKELY WORSE.** More parameters (if C_obs
  is added), more complex landscape (LC11), no hemodynamic smoothing
  (LC12) all conspire against convergence. Multi-start approach (LC11)
  is the primary mitigation.
  **Net effect: multi-start SVI is non-optional.**

---

## Phase-Specific Warning Summary

| Phase | Primary Pitfalls | Mitigation Order |
|-------|------------------|------------------|
| Model definition (direct obs model) | LC4, LC5, LC10, LC12 | LC5 first (observation model choice), then LC4 (priors), LC10 (dt), LC12 (gradients) |
| RNN training | LC9, LC14 | LC9 (task choice) before training; LC14 (multi-seed) during |
| Dimensionality reduction | LC2, LC3 | LC3 (PCA quality check) first; LC2 (rotation handling) in validation |
| DCM fitting pipeline | LC11, LC13 | LC11 (multi-start) in fitting; LC13 (interpretation) after |
| Validation / benchmark design | LC1, LC6, LC7, LC8 | LC7 (metrics) first; LC1 (misspecification) and LC6 (context coding) as diagnostics; LC8 (model selection) for N |

---

## Confidence Assessment

| Pitfall | Confidence | Basis |
|---------|-----------|-------|
| LC1 | HIGH mechanism, MEDIUM regime | Nonlinear dynamics theory; Langdon & Engel comparison |
| LC2 | HIGH | Textbook linear algebra; Langdon & Engel Cayley parameterization |
| LC3 | HIGH | Dubreuil et al. 2024 eLife; Langdon & Engel PFC results |
| LC4 | HIGH mechanism, MEDIUM magnitude | Prior calibration theory; Phase 16.1 RECOV-04 experience |
| LC5 | HIGH | Standard identifiability analysis |
| LC6 | HIGH | Hristos et al. 2024 Science Advances; Dubreuil et al. 2022 PLoS CB |
| LC7 | HIGH | Fundamental coordinate-system mismatch |
| LC8 | MEDIUM | Model selection theory; no published DCM-on-RNN benchmarks |
| LC9 | MEDIUM | Attractor landscape theory; no published distillability comparison |
| LC10 | HIGH | Standard numerical integration |
| LC11 | HIGH | Phase 16.1 experience + Langdon & Engel 100-restart strategy |
| LC12 | HIGH mechanism, MEDIUM severity | Direct consequence of observation model change |
| LC13 | MEDIUM | Plausible mechanism; unquantified in literature |
| LC14 | MEDIUM | Huang et al. 2025 NeurIPS; not studied for bilinear DCM |

---

## Sources

### Primary (HIGH -- published, peer-reviewed, directly relevant)

- **Langdon & Engel (2025).** "Latent circuit inference from heterogeneous
  neural responses during cognitive tasks." Nature Neuroscience.
  [PMC11893458](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/)
  [Nature](https://www.nature.com/articles/s41593-025-01869-7)
  Basis for: LC1, LC2, LC3, LC5, LC7, LC11.

- **Dubreuil, Valente, Beiran, Mastrogiuseppe, Ostojic (2024).** "Aligned
  and oblique dynamics in recurrent neural networks." eLife.
  [eLife](https://elifesciences.org/reviewed-preprints/93060)
  Basis for: LC3 (oblique dynamics).

- **Hristos, Jha, Engel (2024).** "Inferring context-dependent computations
  through linear approximations of prefrontal cortex dynamics." Science
  Advances.
  [PMC11654703](https://pmc.ncbi.nlm.nih.gov/articles/PMC11654703/)
  Basis for: LC1, LC6 ({A_cx, B} vs {A, B_cx} degeneracy).

- **Huang, Singh, Martinelli, Rajan (2025).** "Measuring and Controlling
  Solution Degeneracy across Task-Trained Recurrent Neural Networks."
  NeurIPS 2025.
  [arXiv 2410.03972](https://arxiv.org/abs/2410.03972)
  Basis for: LC14 (solution degeneracy), LC2 (DSA metric).

- **Smith, Linderman, Sussillo (2021).** "Reverse engineering recurrent
  neural networks with Jacobian switching linear dynamical systems."
  NeurIPS 2021.
  [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/8b77b4b5156dc11dec152c6c71481565-Abstract.html)
  Basis for: LC1 (JSLDS as comparator), LC9.

- **Valente, Pillow, Ostojic (2022).** "Extracting computational mechanisms
  from neural data using low-rank RNNs." NeurIPS 2022.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9877d915a4b4f00e85e7b4cfdf41e450-Abstract-Conference.html)
  Basis for: LC1 (LINT), LC3 (low-rank reduction).

- **Nozari et al. (2024).** "Macroscopic resting-state brain dynamics are
  best described by linear models." Nature Biomedical Engineering.
  [PMC11357987](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357987/)
  Basis for: LC1 (when linear approximation works).

### Secondary (MEDIUM -- relevant but extrapolated to this context)

- **v0.3.0 PITFALLS.md** (`.planning/research/v0.3.0/PITFALLS.md`):
  B1-B14. Basis for: prior calibration lessons, stability concerns.

- **Phase 16.1 RECOV-04 experience.** B-RMSE shrinkage from init_scale
  interaction. Direct project experience. Basis for: LC4, LC11.

- **Pellegrino et al. (2024).** "Dimensionality reduction beyond neural
  subspaces with slice tensor component analysis." Nature Neuroscience.
  [Nature](https://www.nature.com/articles/s41593-024-01626-2)
  Basis for: LC3 (PCA limitations).

- **Dubreuil et al. (2022).** "Theory of Gating in Recurrent Neural
  Networks." PLoS Computational Biology.
  [PMC9762509](https://pmc.ncbi.nlm.nih.gov/articles/PMC9762509/)
  Basis for: LC6 (gating vs bilinear modulation).

### Tertiary (LOW -- general or extrapolated)

- **Singh et al. (2020).** "Estimation and validation of individualized
  dynamic brain models with resting state fMRI." NeuroImage.
  MINDy framework. Basis for: LC4 (scale issues in neural ODE fitting).

- State derivative normalization (arXiv 2401.02902). Basis for: LC10
  (temporal resolution normalization).
