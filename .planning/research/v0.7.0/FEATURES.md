# Feature Landscape: v0.7.0 Variational Laplace Validation

**Domain:** Inference-engine validation for a DCM Variational Laplace reimplementation
**Researched:** 2026-06-10
**Confidence:** HIGH (grounded in codebase audit, existing test suite, SPM12 source, published
literature, and the Julia spectral-DCM cross-validation study as the closest precedent)

---

## Context: What Exists vs What Is Being Validated

The v0.6.0 VL engine (`variational_laplace.py`, `forward_models.py`, `csd_precision.py`,
`bmr.py`) already passes narrow smoke/recovery tests. v0.7.0 is validation-in-breadth:
systematically proving the engine is trustworthy across conditions before real data.

The four validation dimensions from the milestone scope map to feature categories below.

---

## Table Stakes

Features required for v0.7.0 to make any scientific claim about the VL engine. Absence of any
one of these means the milestone's headline result ("engine validated") is not defensible.

---

### TS-1: Recovery Matrix — Multi-Condition Sweep Harness

**What:** A harness that runs VL recovery across a Cartesian product of conditions and writes
per-cell results to a structured output (JSON + CSV). Minimum viable matrix:
- N (regions): 2, 3, 5 — latent-circuit and spectral forward models; task DCM at 3 only
  (task is ODE-based with intractable precision at N>5 under current identity-Q design)
- SNR: 2, 5, 10 — low / moderate / high
- Seeds per cell: ≥10 (fixed seed schedule 0–9 across all conditions for comparability)
- Forward models: all three (`LatentCircuitForward`, `SpectralDCMForward`, `TaskDCMForward`)

**Why required:** A single-seed, single-condition result (the current v0.6.0 state) cannot
distinguish "engine works" from "got lucky." The DCM validation literature (Razi et al. 2015,
identifiability work by Frässle et al. 2018, the Julia spectral DCM paper 2025) universally
uses multi-seed, multi-condition sweeps before claiming validation. MEDIUM confidence on exact
cell count; the 2×3×3 minimum is conservative and based on the existing 5-seed per condition in
`test_variational_laplace_recovery.py` scaled up.

**Metrics required per cell** (see TS-4):
- Median RMSE on A (off-diagonal; separately for diagonal)
- 10th-percentile RMSE (worst-case indicator)
- Masked sign recovery (|A_true| > 0.01 threshold)
- Credible interval coverage at 95% (from VL full covariance via sampling)
- Identifiability shrinkage ratio: mean(std_post / std_prior) across A free-parameters
- Mean free energy at convergence and number of iterations
- Fraction of seeds that converged (met stopping criterion within max_iter)

**Complexity:** Medium. The harness logic is straightforward; the cost is cluster time
(~2–4 hrs wall-clock for the full matrix on M3 SLURM). The individual VL call already works.

**Dependencies:** Existing VL engine; `make_stable_latent_circuit_A`, `make_stable_A_spectral`,
`make_stable_A_task`; `simulate_latent_circuit`, `simulate_spectral_dcm`, `simulate_task_dcm`.

---

### TS-2: Per-Cell Result Reporting — Correct Metric Suite

**What:** For every recovery cell, compute and record the full metric set that the DCM
literature considers necessary for claiming valid recovery:

| Metric | Why Required | Formula / Note |
|--------|-------------|----------------|
| RMSE(A) off-diagonal | Primary accuracy measure; all DCM papers | sqrt(mean((A_true - A_inferred)^2)) over off-diag |
| RMSE(A) diagonal | Self-connection separately (nonlinear parameterisation) | Same, over diagonal only |
| Masked sign recovery | Separates structural zeros from estimated zeros | fraction correct signs where \|A_true\| > 0.01 |
| Pearson r(A_true, A_inferred) | Scale-insensitive check on shape | existing `_pearson_corr` helper |
| Per-region R² mean ± std | Data-fit quality; Pitfall R2 shows pooled R² is misleading | 1 - SS_res/SS_tot per region, then aggregate |
| 95% CI coverage | Calibration; separates accuracy from uncertainty calibration | fraction of parameters where A_true in [lo, hi] |
| Identifiability shrinkage | Distinguishes "VL learned something" from "prior dominated" | mean(std_post_i / std_prior_i) per parameter |
| F trajectory convergent? | Engine health check | bool: F improves from iter 1 to best iter |

**Why required:** Using only RMSE or only correlation is insufficient. The DCM literature (Razi
2015, Frässle 2018, the probabilistic PPL paper PMC12133347) reports at minimum accuracy +
calibration + convergence. The Frässle identifiability work and the existing Pitfall R2 in
PITFALLS.md establish that masked sign recovery and per-region R² are specifically needed here
due to metric artifacts already encountered in this codebase.

**Complexity:** Low. All individual computations already exist in the codebase
(`_pearson_corr`, coverage logic in test files, `bayesian_model_reduction` posterior sampling).
New work: wiring them into a single `compute_recovery_metrics(result, gt, forward)` function.

**Dependencies:** TS-1.

---

### TS-3: SPM12 Cross-Validation — Posterior Mean Agreement

**What:** On at least one condition per forward model (recommendation: spectral N=3 SNR=5,
task N=3 SNR=5, latent-circuit N=3 SNR=10), run both dcm_pytorch VL and SPM12 `spm_nlsi_GN`
on identical synthetic data and compare posterior means (Ep.A).

**Comparison convention** (grounded in Julia spectral DCM paper and existing Phase 6 research):
- Compare in **free-parameter space** (dcm_pytorch `A_free` vs SPM `Ep.A`) to avoid the
  nonlinear diagonal transform amplifying differences.
- Acceptable tolerance: element-wise relative error < 10% for |A_free| > 0.01, absolute
  error < 0.02 for |A_free| ≤ 0.01. The Julia paper achieved O(10⁻³)–O(10⁻⁴) deviations
  with autodiff; our finite-difference Jacobian will be somewhat looser, hence the 10% bound.
- Comparison space must be documented and applied consistently (Pitfall S1).
- SPM priors must match dcm_pytorch priors exactly (Pitfall S2: `hyperprior_mean=8.0`,
  `hyperprior_precision=128.0`, `prior_mean_a_offset=a_mask/128`).

**What NOT to compare:**
- Absolute F values (Pitfall S3): SPM and dcm_pytorch use different normalisation constants
  in the SVD-reduced subspace and different Q precision initialisation. Absolute F difference
  of tens to hundreds of nats is expected and is not a failure.
- Posterior covariance Cp element-by-element: SPM returns full-space Cp; dcm_pytorch returns
  reduced-space covariance mapped back. Block structure may differ. The Julia paper explicitly
  focuses on means and free energy, not Cp.

**Model ranking test:** Run SPM and dcm_pytorch on 3 model pairs (full model vs one pruned
connection). Both engines must agree on which model has higher evidence. Disagreement on > 1
of 3 pairs = failure; this matches the "compare rankings not absolute values" convention
established in Phase 6 research.

**Complexity:** Medium-high. The SPM12 pipeline (export .mat, run MATLAB batch, load results)
is already specified in `06-RESEARCH.md` with complete code patterns. New work: adapting the
export for the spectral/latent-circuit case, adding hyperprior-matching parameters to the
export script, and adding the comparison assertion logic.

**Dependencies:** MATLAB R2022a + SPM12 available locally; `scipy.io.savemat` patterns from
Phase 6; Pitfall S1–S6 mitigations all required.

---

### TS-4: VL+BMR Model Comparison — Relative Evidence Ranking

**What:** Validate that BMR-on-VL correctly ranks reduced models (absent-connection prior vs
present-connection prior) by relative ΔF. The absolute-ΔF threshold regime is BROKEN (Pitfall
C1, empirically confirmed in job 55772525). The validated, defensible metric is:

**Separation gap:** `ΔF(pruning_absent_connection) - ΔF(pruning_present_connection) > 0`

For a 3-region synthetic circuit with known A structure, run the full VL, then apply BMR for
all 2^(N²-N) off-diagonal subsets. Assert:
- For each truly-absent connection: pruning it yields ΔF closer to 0 (less evidence loss)
  than pruning any truly-present connection.
- The ranking of all N² pruned models places absent-connection prunings above present-connection
  prunings. Formally: `rank(absent prunes) < rank(present prunes)` when sorted descending by ΔF.
- The separation gap `ΔF(best absent) - ΔF(worst present)` is > 3 nats (Jeffreys threshold
  for decisive evidence: Bayes factor ~20, equivalent to ΔF ≈ ln(20) ≈ 3 nats). This is the
  DCM convention for "decisive" model comparison per the Bayesian model selection literature.

**Report both:** untempered relative ranking (primary, known to work) AND tempered ranking if
tempering is implemented (exploratory, see TS-8).

**Complexity:** Low. `bayesian_model_reduction` and `bmr_circuit_selection` already exist and
pass their tests. New work: a multi-condition validation harness that checks the ranking
assertions across the recovery matrix conditions from TS-1, and the separation-gap threshold.

**Dependencies:** TS-1; `bmr.py`.

---

### TS-5: Numerical Robustness — Standard Edge-Case Suite

**What:** A pytest suite that exercises the VL engine on edge cases that are known failure modes
for Gauss-Newton / Laplace methods in DCM (from the PITFALLS.md N-series pitfalls):

| Edge Case | Test Description | Pass Criterion |
|-----------|-----------------|----------------|
| Near-stability boundary | A with max real eigenvalue ∈ [-0.05, 0] | Engine completes without NaN; reports non-convergence flag; does NOT raise |
| Nearly-absent connection (A_ij = 0.001) | Should be identifiable as "low" | Posterior std_post / std_prior > 0.7 (prior not overwhelmed; no spurious precision) |
| N=2 minimal network | Smallest possible DCM | Converges; posterior has finite F; no SVD rank-drop |
| Uniform prior (no mask) vs sparse prior | a_mask = ones vs a_mask = diagonal only | Both converge; sparse-mask result has narrower posterior on active parameters |
| Long time series T=1000 (latent circuit, dt=0.1) | Stress test precision matrix size | OOM-free; wall time < 5 min on M3 |
| ReML max iter reached | max_iter=5 (underfitting) | Returns result dict; free_energy list has exactly 5 entries; converged=False |
| Jacobian column norm near zero | A_free initialised to boundary | Warns (log) rather than crashes; posterior mean still finite |

**Complexity:** Low per test; medium total because each edge case needs a dedicated fixture.

**Dependencies:** Existing VL engine; `LatentCircuitForward` (confirmed OOM-free at T~60 for
the existing tests; T=1000 at dt=0.1 is the stress regime).

---

## Differentiators

Features that go beyond "does the engine work" and establish scientific credibility or enable
downstream use. Important for publication quality; not required for internal milestone closure.

---

### D-1: Credible Interval Coverage Calibration Curve

**What:** For each (N, SNR, forward-model) cell in the recovery matrix, compute empirical
coverage at multiple nominal levels: 50%, 75%, 90%, 95%. Plot expected vs observed coverage.
A well-calibrated VL posterior produces a curve close to the diagonal. Known behaviour from
the literature (Zeidman VL primer PMC10951963): VL Laplace is overconfident, so the coverage
curve will lie below the diagonal (observed < nominal), especially at high SNR.

**Why valuable:** Documents the known VL limitation rather than pretending it does not exist.
The calibration curve is the correct way to report this (it is what the Phase 11 calibration
analysis did for SVI guides in v0.2.0). Reviewers and users need to know whether the 95% CI
from VL is a "true 95% CI" or a "tighter-than-nominal 80% CI in disguise."

**Reporting convention:** Report separately for present connections (|A_true| > 0.01) vs absent
connections (|A_true| ≤ 0.01), because coverage behaviour differs: on present connections the
posterior is informative; on absent connections the posterior should revert toward the prior.

**Complexity:** Low. The CI computation already exists in `test_variational_laplace_recovery.py`
(sampling from `sigma_post` via `extract_vl_posterior`). New work: sweeping over nominal levels
and binning by connection strength.

**Dependencies:** TS-1, TS-2.

---

### D-2: Identifiability Shrinkage Heatmap

**What:** For each cell in the recovery matrix, compute and display the mean identifiability
shrinkage ratio `mean(std_post_i / std_prior_i)` as a heatmap over (N, SNR). A ratio near 1
means the posterior equals the prior (VL learned nothing). A ratio near 0 means the posterior
is much tighter than the prior (VL learned a lot).

The DCM identifiability literature (Frässle et al. 2015 PMC4335185, using profile likelihood)
established that dense networks at low SNR are prone to non-identifiability. This heatmap is
the VL-native equivalent of that diagnostic: it separates "engine working but truly
non-identifiable" from "engine broken."

**Complexity:** Low. The computation is a single line per cell after TS-1; the heatmap is
matplotlib. New work: aggregating the per-seed shrinkage into per-cell statistics.

**Dependencies:** TS-1.

---

### D-3: Free Energy Trajectory Diagnostic Plots

**What:** For each seed in each condition, store the full free energy trajectory
`result.free_energy` (already returned by the engine). Plot:
1. Typical trajectory (median seed for each cell) — should show monotonic increase then plateau.
2. Failure trajectory (worst-RMSE seed) — reveals whether failure is flat landscape or
   divergence.
3. Iteration count histogram across seeds — reveals whether max_iter is being hit (= more
   iterations needed) or whether convergence is fast (= well-conditioned problem).

**Why valuable:** The convergence criterion (4× ΔF < 0.1) does not imply parameter convergence
(Pitfall R5). These plots let users see whether the stopping criterion fires early (risky) or
after genuine stabilisation.

**Complexity:** Low. Engine already returns trajectories; matplotlib plotting.

**Dependencies:** TS-1.

---

### D-4: SPM12 Posterior Covariance — Diagonal Agreement Check

**What:** Even though full Cp comparison is not a table-stakes requirement (see TS-3), a
weaker check IS informative: compare the **diagonal** of Cp (marginal posterior variances) in
a corresponding common space. The diagonal can be extracted from SPM's sparse Cp matrix and
from dcm_pytorch's `sigma_post` without requiring the off-diagonal structure to match.

**Acceptable tolerance:** Factor-of-2 agreement on marginal posterior variances (i.e.,
`0.5 < std_post_i_ours / std_post_i_spm < 2.0` for all free parameters). This is intentionally
loose because VL's SVD projection changes the effective dimensionality of the posterior.

**Why valuable:** If marginal variances are systematically 5–10× smaller in dcm_pytorch, it
confirms the overconfidence hypothesis quantitatively. If they match, it suggests the
implementations agree at the marginal level despite differences in off-diagonal structure.

**Complexity:** Low-medium. SPM's Cp is returned as a sparse MATLAB matrix; `scipy.io.loadmat`
converts it to a dense numpy array after `full(DCM.Cp)` in the MATLAB script. The diagonal
extraction is trivial.

**Dependencies:** TS-3.

---

### D-5: Multi-Restart Analysis — Best vs Single-Start RMSE Gap

**What:** For a representative subset of cells (recommendation: spectral N=3/5, all SNR), run
VL with 3 restarts per seed: (a) default start (A_free = 0), (b) oracle start (A_free =
ground-truth projected into reduced space), (c) random perturbation (A_free ~ N(0, 0.1)).
Report the gap between best-restart and single-start RMSE.

**Why valuable:** Local optima are a known VL limitation (Pitfall N4). If the best-restart RMSE
is substantially lower than single-start RMSE, it quantifies how much performance is left on
the table by not multi-starting. This directly informs whether v0.8.0 should invest in
warm-starting or random multi-restart strategies. If the gap is small, no further investment
is needed.

**Complexity:** Medium. The `initial_p` API (commit 27b43fd) already supports warm-starting.
New work: the comparison logic and the oracle-start projection into the SVD-reduced space.

**Dependencies:** TS-1.

---

### D-6: Connection Density Sweep

**What:** Add density as a variable in the recovery matrix for the latent-circuit forward model
(which has full control over A structure): densities 0.25, 0.5, 0.75 (sparse, medium, dense).
The identifiability literature (Frässle 2015) documents that dense networks are prone to
non-identifiability; this sweep quantifies where the VL engine starts struggling.

**Complexity:** Low marginal cost if TS-1 harness already parametrises density. The
`make_stable_latent_circuit_A` function already accepts a density argument.

**Dependencies:** TS-1.

---

### D-7: Modulation Strength Sweep for BMR (B Matrix)

**What:** For the latent-circuit forward model (the one with the bilinear B path), sweep over
B modulation strengths: 0 (absent), 0.1, 0.3, 0.5, 0.7 (strong). Verify that BMR correctly
identifies which B edges are present vs absent as a function of their strength. Report the
minimum modulation strength at which BMR separation gap reliably exceeds 3 nats.

**Why valuable:** This is the v0.7.0 analogue of testing detection sensitivity. B-RMSE 0.0048
was reported at one operating point (SNR=10, N=3, strong B); we do not know the detection
threshold.

**Complexity:** Low marginal cost given TS-1/TS-4 harnesses.

**Dependencies:** TS-4.

---

## Anti-Features

Features to explicitly NOT build in v0.7.0. Each is either premature, a scope risk, or actively
harmful to the validation milestone.

---

### AF-1: Absolute ΔF Threshold for BMR Pruning

**What it would be:** A hard threshold on ΔF (e.g., "prune connection if ΔF > -3") used as
the primary BMR result.

**Why avoid:** Empirically broken due to Laplace overconfidence (Pitfall C1, job 55772525).
Absent connections score ΔF = -116; present connections score ΔF = -182,293. An absolute
threshold can never distinguish these at any reasonable setting.

**What to do instead:** Relative ranking + separation gap (TS-4). If absolute thresholding is
needed in a later milestone, it requires posterior tempering (TS-8) to be first validated.

---

### AF-2: Posterior Covariance Cp as Primary SPM12 Comparison Criterion

**What it would be:** Asserting element-wise Cp agreement (e.g., all Cp diagonal entries within
20%) as a pass/fail criterion for SPM cross-validation.

**Why avoid:** SPM's Cp lives in full parameter space; dcm_pytorch Cp lives in the SVD-reduced
subspace mapped back via the V matrix. The off-diagonal structure is not guaranteed to match
because the SVD reduction implicitly projects out low-variance dimensions in a way that differs
between the two implementations. The Julia spectral DCM paper explicitly validates only means
and free energy, not Cp. This would create unresolvable false failures.

**What to do instead:** Diagonal marginal variances with a loose tolerance (D-4), not
element-wise Cp.

---

### AF-3: Comparison of Absolute Free Energy Values Across Engines

**What it would be:** Asserting `|F_spm - F_dcmpytorch| < C` for some constant C.

**Why avoid:** Different SVD reduction dimensions, different ReML hyperparameter handling, and
different normalisation constants in the log-determinant terms make absolute F incomparable
(Pitfall S3). A constant offset of 50–200 nats is expected and not a bug.

**What to do instead:** Model ranking comparisons (TS-3): both engines must agree on which of
two models has higher evidence.

---

### AF-4: Standalone SBC (Simulation-Based Calibration with Rank Histograms)

**What it would be:** Full SBC per Talts et al. (2018) — simulating many (θ, y) pairs from the
joint prior, running VL on each y, computing the rank of θ in the posterior samples, and
checking uniformity.

**Why avoid in v0.7.0:** SBC requires the inference to be run ~500–1000 times to get a reliable
rank histogram. VL on spectral N=3 takes ~2 min; 500 runs = ~17 hours per condition. More
fundamentally, SBC validates a sampler; VL is a deterministic optimiser. The correct calibration
check for a deterministic posterior is coverage probability (TS-2), which is already in scope.
SBC is in scope for SBI validation (Phase D / v0.8.0), where NPE outputs a proper density
that can be sampled.

**What to do instead:** Coverage calibration curves at 50/75/90/95% nominal levels (D-1).

---

### AF-5: Real Data Application

**What it would be:** Applying the validated VL engine to real Cam-CAN MEG or fMRI data.

**Why avoid in v0.7.0:** Explicitly deferred to v0.7.0 Phase E (which may itself be a separate
milestone). The entire point of v0.7.0 is to validate the engine on synthetic ground truth
before trusting it on real data. Running on real data before completing the validation matrix
inverts this logic. Also, parcellation + data access (DUA) + nilearn installation are
unresolved blockers (from the v0.6.0 audit).

---

### AF-6: Normalising the Recovery Matrix into a Single Aggregate Score

**What it would be:** Reporting one number (e.g., "mean RMSE across all conditions = 0.08")
as the v0.7.0 headline result.

**Why avoid:** Pitfall R1 (Simpson's paradox across forward models and scales). A 3-region
latent-circuit model at SNR=10 and a 5-region spectral model at SNR=2 represent completely
different identifiability regimes. Averaging hides both the strong results and the weak results.

**What to do instead:** Per-cell matrix heatmap with separate per-forward-model summary.

---

## Feature Dependencies

```
TS-1 (recovery matrix harness)
  └── TS-2 (metric suite)             — requires harness to exist
  └── TS-4 (BMR ranking)              — runs on harness results
  └── TS-5 (numerical edge cases)     — parallel, uses same engine
  └── D-1 (coverage calibration)      — extends TS-2
  └── D-2 (shrinkage heatmap)         — extends TS-2
  └── D-3 (F trajectory plots)        — extends TS-1 output
  └── D-6 (density sweep)             — parametric extension of TS-1
  └── D-5 (multi-restart)             — extends TS-1 + initial_p API
TS-3 (SPM12 cross-validation)
  └── D-4 (Cp diagonal agreement)     — extends TS-3
TS-4 (BMR ranking)
  └── D-7 (modulation strength sweep) — extends TS-4 to B-sweep
TS-8 (posterior tempering)            — exploratory; extends TS-4 for absolute regime
```

---

## MVP Recommendation

For the core v0.7.0 claim ("VL engine validated on synthetic ground truth"):

**Build first (blockers):**
1. TS-1: Recovery matrix harness — without this nothing else is a matrix
2. TS-2: Metric suite function — needed before results can be reported
3. TS-3: SPM12 cross-validation — establishes the "SPM12-grade" claim
4. TS-4: BMR relative ranking harness — closes the known overconfidence todo
5. TS-5: Numerical robustness suite — gates reliability claim

**Build second (quality):**
6. D-1: Coverage calibration curve — needed to characterise overconfidence quantitatively
7. D-2: Shrinkage heatmap — separates "learned nothing" from "identifiability limit"
8. D-3: Free energy trajectory plots — diagnostic value for any failed cells

**Defer to later in v0.7.0 or v0.8.0:**
- D-4, D-5, D-6, D-7: Valuable but not required for the "engine validated" claim
- TS-8 (posterior tempering): Exploratory; needs careful calibration work; only if relative
  ranking alone is insufficient for BMR use cases

---

## Supplementary Feature: Posterior Tempering (TS-8, exploratory)

Not a table stake; included here for completeness because it is a confirmed open todo
(`vl-overconfidence-for-bmr`) and the milestone scope explicitly includes it as an option.

**What:** Inflate `sigma_post` by a temperature factor T > 1 before feeding it to BMR, to
recover the absolute-threshold regime. Two candidate approaches:
- **Fixed multiplicative temperature:** `sigma_tempered = T * sigma_post`. Simplest; requires
  a calibration pass to choose T.
- **Data-driven temperature:** Choose T so that the empirical 95% CI coverage from TS-2 equals
  the nominal 95%. Principled but requires running the recovery matrix first.

**Complexity:** Medium-high. The temperature schedule must be validated across conditions before
it can be used as a reporting criterion (Pitfall C2). The recovery matrix from TS-1 provides
the calibration data; the tempering logic is ~20 lines; but the validation loop is another
full recovery-matrix pass.

**Recommended order:** TS-1 → TS-2 → D-1 (coverage at nominal levels) → use D-1 output to
calibrate T → TS-8.

**Dependency:** TS-1, TS-2, D-1.

---

## Complexity Summary

| Feature | Category | Complexity | Phase |
|---------|----------|------------|-------|
| TS-1: Recovery matrix harness | Table stake | Medium | B |
| TS-2: Per-cell metric suite | Table stake | Low | B |
| TS-3: SPM12 cross-validation | Table stake | Medium-high | B |
| TS-4: BMR relative ranking | Table stake | Low | C |
| TS-5: Numerical robustness suite | Table stake | Low | B/D |
| D-1: Coverage calibration curve | Differentiator | Low | B |
| D-2: Shrinkage heatmap | Differentiator | Low | B |
| D-3: F trajectory plots | Differentiator | Low | B |
| D-4: Cp diagonal agreement | Differentiator | Low-medium | B |
| D-5: Multi-restart analysis | Differentiator | Medium | B |
| D-6: Connection density sweep | Differentiator | Low (marginal) | B |
| D-7: Modulation strength sweep | Differentiator | Low (marginal) | C |
| TS-8: Posterior tempering | Exploratory | Medium-high | C |
| AF-1: Absolute ΔF threshold | Anti-feature | — | — |
| AF-2: Cp element-wise comparison | Anti-feature | — | — |
| AF-3: Absolute F comparison | Anti-feature | — | — |
| AF-4: Full SBC | Anti-feature | — | — |
| AF-5: Real data application | Anti-feature | — | — |
| AF-6: Aggregate score | Anti-feature | — | — |

---

## Sources

### HIGH confidence (codebase + SPM12 source + verified literature)
- `src/pyro_dcm/inference/variational_laplace.py` — VL engine as implemented
- `src/pyro_dcm/model_selection/bmr.py` — BMR as implemented
- `tests/test_variational_laplace_recovery.py`, `tests/test_bmr_vs_elbo.py`,
  `tests/test_latent_circuit_vl.py` — current test baseline
- `.planning/milestones/v0.6.0-MILESTONE-AUDIT.md` — job 55772525 BMR overconfidence evidence
- `.planning/research/v0.7.0/PITFALLS.md` — C1–C2, R1–R5, N1–N5 pitfalls with empirical evidence
- `.planning/phases/06-validation-against-spm/06-RESEARCH.md` — SPM12 struct format, tolerances
- SPM12 local source (R2022a): `spm_nlsi_GN.m`, `spm_dcm_estimate.m`, `spm_dcm_fmri_csd.m`
- Zeidman, Friston & Parr (2024). A primer on Variational Laplace. NeuroImage.
  PMC10951963 — documents VL overconfidence (HIGH for overconfidence claim)
- Friston & Penny (2011). Post hoc Bayesian model selection. NeuroImage.
  REF-070 — BMR equations, ΔF as log Bayes factor

### MEDIUM confidence (verified from published paper abstracts / PMC)
- Pfeffer et al. (2025). Increasing spectral DCM flexibility and speed by leveraging Julia's
  ModelingToolkit. Imaging Neuroscience. PMC12330849. — O(10⁻³)–O(10⁻⁴) tolerance for
  posterior mean agreement; mean + free energy validated; Cp not compared
- Frässle et al. (2015). Assessing parameter identifiability for DCM. PMC4335185. — dense
  networks prone to non-identifiability; profile likelihood as diagnostic; SNR sweep methodology
- Razi et al. (2015). Construct validation of a DCM for resting state fMRI. PMC4295921. —
  multi-seed synthetic recovery as validation standard for spectral DCM

### LOW confidence (WebSearch-only, unverified details)
- General Bayesian calibration literature: coverage at multiple nominal levels as the standard
  check for deterministic posterior approximations (replaces full SBC for deterministic methods)
- Cold-posterior / temperature-scaling literature: tempering as a fix for Laplace overconfidence,
  but no DCM-specific validated schedule
