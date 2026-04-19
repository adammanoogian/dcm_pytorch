# Feature Landscape: Cross-Backend Inference Benchmarking for DCM

**Domain:** Probabilistic inference benchmarking / Dynamic Causal Modeling
**Researched:** 2026-04-06
**Milestone:** v0.2.0
**Overall confidence:** MEDIUM-HIGH

---

## Table Stakes

Features users expect from a credible inference benchmarking study.
Missing any of these undermines the paper's scientific credibility.

### TS-1: MCMC Gold Standard Reference

| Aspect | Detail |
|--------|--------|
| Feature | NUTS posteriors as ground-truth reference for all VI comparisons |
| Why Expected | Every serious VI benchmarking paper uses MCMC as the reference posterior. Reviewers will ask "how does this compare to NUTS?" |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | NumPyro integration (already planned), DCM model re-expression in NumPyro |

**Specification:**
- 4 parallel chains per dataset (community standard)
- R-hat < 1.01 convergence threshold (current best practice, tighter than the
  older 1.1 threshold per Vehtari et al. 2021)
- Bulk ESS > 400 and tail ESS > 400 per parameter (Stan recommendations)
- 200 warmup + 200-500 draws per chain (sufficient for DCM per the
  Friston/PPL benchmark: 200 warmup achieved convergence in <1 min with NumPyro)
- Store full posterior samples for downstream coverage/calibration analysis

**Evidence:** The 2025 J. Royal Society Interface paper on DCM in PPLs
demonstrated NUTS convergence with just 200 warmup samples using NumPyro,
achieving R-hat ~1.00 across all parameters. They used 4 chains.

**Scope note:** NUTS is feasible for spectral DCM (fast, no ODE). For task DCM,
each NUTS sample requires ODE integration -- expect 30-60 min per dataset with
3 regions. At 5-10 regions, task DCM NUTS may be computationally prohibitive.
Plan for spectral DCM NUTS as primary gold standard, task DCM NUTS as
stretch goal (small N only).

---

### TS-2: Extended Guide Variant Comparison

| Aspect | Detail |
|--------|--------|
| Feature | Benchmark multiple VI guide families beyond mean-field |
| Why Expected | Mean-field is known to underestimate posterior variance. A calibration study must test richer approximation families. |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | Existing Pyro generative models, Pyro autoguide API |

**Guide variants to benchmark (in order of increasing flexibility):**

1. **AutoDelta** -- MAP point estimate (no uncertainty). Complexity baseline.
2. **AutoNormal** (mean-field) -- Already implemented. Known to produce
   coverage 0.44-0.78 (current v0.1.0 numbers). Diagonal covariance ignores
   posterior correlations.
3. **AutoLowRankMultivariateNormal** -- Low-rank + diagonal covariance.
   Captures principal components of posterior correlation at moderate cost.
   Pyro recommends this as the natural upgrade from AutoNormal.
4. **AutoMultivariateNormal** (full-rank) -- Full Cholesky covariance.
   Can capture all posterior correlations but scales as O(D^2) in memory
   and is "difficult to fit in the high-dimensional setting" per Pyro docs.
   DCM with 3 regions has D~15-20 parameters -- full-rank is feasible.
   At 10 regions (D~120+), likely intractable.
5. **AutoIAFNormal** -- Inverse Autoregressive Flow. Non-Gaussian posterior
   approximation. Can capture multimodality and skewness that Gaussian
   families cannot.
6. **Amortized NSF** -- Already implemented (Zuko Neural Spline Flow).
   Conditional on data, single forward pass.

**Expected calibration ranking (hypothesis to test):**
AutoDelta (worst) < AutoNormal < AutoLowRank < AutoMultivariate ~= NUTS

Flow-based guides (IAF, NSF) may outperform Gaussian families on
multimodal posteriors but underperform on well-behaved unimodal posteriors
due to training difficulty.

**Complexity note:** AutoDelta, AutoNormal, AutoLowRank, and
AutoMultivariateNormal are all one-line guide instantiations in Pyro.
The main effort is the benchmark harness, not the guide implementation.

---

### TS-3: Coverage Calibration Analysis

| Aspect | Detail |
|--------|--------|
| Feature | Systematic coverage measurement and calibration diagnostics |
| Why Expected | The central scientific question of the paper: "how much accuracy/calibration does the variational approximation cost for DCM?" |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | All guide variants, NUTS reference, existing metrics.py |

**Required analyses:**

1. **Coverage-by-CI-level plot** (the "key figure"): For each method, plot
   empirical coverage (y-axis) vs nominal CI level (x-axis, 50% to 99%).
   A perfectly calibrated method lies on the diagonal. Under-confident
   methods (too-wide CIs) lie above; over-confident (too-narrow) below.
   Current v0.1.0 shows mean-field at 0.44-0.78 for 90% nominal --
   systematically overconfident.

2. **Per-parameter coverage**: Not just aggregate. Break down by parameter
   type (A diagonal, A off-diagonal, hemodynamic params, noise params).
   Different parameter groups may have different calibration properties.

3. **Coverage vs network size**: How does calibration degrade as D increases
   (3 -> 5 -> 10 regions)?

**Existing infrastructure:** `compute_coverage_from_ci()` and
`compute_coverage_from_samples()` in benchmarks/metrics.py already compute
coverage. Need to extend to multi-level CI and per-parameter breakdown.

---

### TS-4: ELBO Variant Comparison

| Aspect | Detail |
|--------|--------|
| Feature | Test different ELBO objectives for their effect on calibration |
| Why Expected | Different ELBO estimators have different variance/bias tradeoffs that affect posterior quality |
| Complexity | Low |
| Confidence | MEDIUM |
| Dependencies | Existing SVI runner, Pyro ELBO implementations |

**ELBO variants to compare:**

1. **Trace_ELBO** -- Standard Monte Carlo ELBO. Currently used.
2. **TraceMeanField_ELBO** -- Analytic KL when guide is mean-field.
   Lower variance gradient estimates. Only valid with AutoNormal.
3. **RenyiELBO (alpha < 1)** -- Tighter bound than standard ELBO.
   Alpha=0 is the IWAE objective. May produce better-calibrated
   posteriors by reducing the "variational gap."

**Not applicable:** TraceEnum_ELBO (for discrete latents, DCM has none).

**Expected finding:** TraceMeanField_ELBO should marginally improve
convergence with AutoNormal. RenyiELBO with alpha=0.5 may improve
calibration at cost of higher gradient variance.

---

### TS-5: Network Size Scaling Study

| Aspect | Detail |
|--------|--------|
| Feature | Test all methods at 3, 5, and 10 region network sizes |
| Why Expected | DCM users care about scalability. "Works on 3 regions" is insufficient for a methods paper. |
| Complexity | Medium-High |
| Confidence | HIGH |
| Dependencies | All guide variants, configurable n_regions in BenchmarkConfig |

**Known scaling behavior from literature:**

- **Traditional DCM (SPM VL)**: Practical limit ~10 regions. Combinatorial
  model space explosion.
- **rDCM**: Scales to 200+ regions (analytic solution). Already demonstrated
  with 66-region and 200-region networks.
- **NUTS**: Feasible at 3-5 regions for spectral DCM. At 10 regions, the
  parameter space (D~120+) may cause poor mixing. The PPL benchmark paper
  used a 9-dimensional system (3 neural mass populations, not brain regions).
- **Mean-field VI**: Scales well computationally but calibration degrades
  as D increases (more correlations to miss).
- **Full-rank VI**: O(D^2) cost. At D=120 (10 regions), the covariance
  matrix has 7,260 free parameters.
- **Amortized flows**: Training cost increases with D, but inference remains
  single-pass.

**Expected scaling results:**

| Method | 3 regions | 5 regions | 10 regions |
|--------|-----------|-----------|------------|
| NUTS | Gold standard | Feasible, slower | Stretch goal |
| AutoNormal | Fast, uncalibrated | Fast, worse calibration | Fast, poor calibration |
| AutoLowRank | Good balance | Good balance | Best VI option |
| AutoMultivariate | Best Gaussian | Feasible but slow | Likely intractable |
| Amortized NSF | Fast inference | Needs more training data | May need architecture changes |
| rDCM VB | Instant, overconfident | Good | Excellent scalability |

---

### TS-6: Amortization Gap Characterization

| Aspect | Detail |
|--------|--------|
| Feature | Properly measure and report the amortization gap |
| Why Expected | Already partially built (metrics.py has compute_amortization_gap). But v0.1.0 used synthetic ELBO (final_loss * 1.1). Must use real measured values. |
| Complexity | Low (fix existing implementation) |
| Confidence | HIGH |
| Dependencies | Fix amortized runners to compute real ELBO (v0.1.0 verification gap) |

**What to measure:**
- Per-subject ELBO from SVI (per-instance optimization)
- Per-subject ELBO from amortized guide (single forward pass, no refinement)
- Per-subject ELBO from amortized + refinement (few SVI steps on amortized init)
- Absolute gap: ELBO_amortized - ELBO_svi
- Relative gap: |absolute_gap| / |ELBO_svi|

**Known behavior from literature:**
- The amortization gap is guaranteed to be non-negative (amortized can never
  beat per-instance optimization on a single instance).
- Gap closes with more flexible inference networks (flows > Gaussian).
- Semi-amortized refinement (few gradient steps from amortized init)
  typically closes 60-90% of the gap.
- Gap is smaller for simpler models and larger training sets.

**Note:** The v0.1.0 verification report (08-VERIFICATION.md) identified this
as a blocker: both amortized runners use `svi_result["final_loss"] * 1.1`
as a placeholder. This must be fixed to compute real amortized ELBO via
`Trace_ELBO().differentiable_loss(model, guide, *model_args)`.

---

### TS-7: Cross-Backend Comparison Table

| Aspect | Detail |
|--------|--------|
| Feature | Unified table: 9+ methods x 3 variants x 3 network sizes |
| Why Expected | The deliverable of the paper. Readers want a single table showing method x variant x metric. |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | All guide variants implemented and benchmarked |

**Proposed table structure:**

| Method | Variant | N | RMSE | Coverage@90% | Corr | ELBO | Time (s) |
|--------|---------|---|------|-------------|------|------|----------|
| NUTS (4 chains) | spectral | 3 | ... | ... | ... | ... | ... |
| AutoDelta (MAP) | spectral | 3 | ... | N/A | ... | ... | ... |
| AutoNormal (MF) | spectral | 3 | ... | ... | ... | ... | ... |
| AutoLowRank | spectral | 3 | ... | ... | ... | ... | ... |
| AutoMultivariate | spectral | 3 | ... | ... | ... | ... | ... |
| AutoIAFNormal | spectral | 3 | ... | ... | ... | ... | ... |
| Amortized NSF | spectral | 3 | ... | ... | ... | ... | ... |
| Amortized + refine | spectral | 3 | ... | ... | ... | ... | ... |
| rDCM VB | rdcm | 3 | ... | ... | ... | ... | ... |
| ... | ... | 5 | ... | ... | ... | ... | ... |
| ... | ... | 10 | ... | ... | ... | ... | ... |

---

### TS-8: Reproducible Benchmark Runner

| Aspect | Detail |
|--------|--------|
| Feature | Single CLI command to reproduce all results |
| Why Expected | Scientific reproducibility. Already partially built. |
| Complexity | Low (extend existing CLI) |
| Confidence | HIGH |
| Dependencies | Existing run_all_benchmarks.py infrastructure |

**Already built:**
- `run_all_benchmarks.py` with --variant, --method, --quick flags
- RUNNER_REGISTRY dispatching to 7 runners
- JSON output with metadata (git hash, timestamp, versions)
- Publication-quality figure generation

**Needs extension:**
- New runners for each guide variant (AutoLowRank, AutoMultivariate, etc.)
- NumPyro NUTS runner
- Network size parameter (--n-regions 3 5 10)
- Amortized refinement runner
- Aggregation script to produce the unified comparison table

---

## Differentiators

Features that would set this work apart from existing DCM benchmarks.
Not expected by default, but significantly increase the paper's impact.

### D-1: Simulation-Based Calibration (SBC)

| Aspect | Detail |
|--------|--------|
| Feature | Full SBC analysis for each inference method |
| Value Proposition | SBC is the gold standard for validating Bayesian computation. Goes beyond simple coverage to detect systematic biases in the posterior approximation. |
| Complexity | Medium-High |
| Confidence | MEDIUM |
| Dependencies | All guide variants, significant compute (1000+ simulated datasets per method) |

**What SBC provides that coverage alone does not:**

1. **Rank histogram uniformity**: For each parameter, the rank of the true
   value among posterior samples should be uniformly distributed across
   simulated datasets. Non-uniformity reveals specific failure modes:
   - U-shaped histogram -> overconfident posterior
   - Inverted-U -> underconfident
   - Skewed -> biased posterior

2. **ECDF-based tests**: Plot empirical CDF of ranks against the uniform
   CDF. Deviations reveal systematic issues.

3. **Test quantity sensitivity**: Recent work (2025, Bayesian Analysis)
   shows that choice of test quantities matters. The joint log-likelihood
   is "an especially useful test quantity" that can detect problems
   invisible to marginal rank tests.

**Implementation approach:**
- Simulate 500-1000 datasets from the prior
- Run each inference method on each dataset
- Compute rank statistics for each parameter
- Plot rank histograms and ECDF difference plots
- Apply uniformity tests (KS test, chi-squared)

**Compute cost:** 1000 datasets x 9 methods x 3 variants = 27,000 inference
runs. Even at 10s per run, that is 75 hours. SBC is a stretch feature
that requires careful compute budgeting or restriction to a subset.

---

### D-2: Amortized + Refinement Pipeline

| Aspect | Detail |
|--------|--------|
| Feature | Semi-amortized inference: amortized initialization + few SVI refinement steps |
| Value Proposition | Best-of-both-worlds: amortized speed + per-instance accuracy. Directly addresses the amortization gap. |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | Existing AmortizedFlowGuide, SVI runner |

**Protocol:**
1. Run amortized guide to get initial posterior approximation
2. Initialize an AutoNormal guide at the amortized posterior mean/scale
3. Run 50-100 SVI refinement steps from this initialization
4. Measure ELBO, coverage, RMSE at steps 0, 10, 50, 100

**Expected behavior:**
- At step 0: amortized quality (fast but imperfect)
- At step 10-50: closes ~60-80% of the amortization gap
- At step 100: approaches per-instance SVI quality

**This answers a practical question users care about:** "Can I get NUTS-quality
posteriors at near-amortized speed by doing a few refinement steps?"

---

### D-3: Regularization / Parameterization Study

| Aspect | Detail |
|--------|--------|
| Feature | Systematic study of how parameterization and priors affect calibration |
| Value Proposition | Provides actionable guidance. "Use non-centered parameterization with sigma_A=0.3 for best calibration" is more useful than "AutoLowRank is better than AutoNormal." |
| Complexity | Medium |
| Confidence | MEDIUM |
| Dependencies | All guide variants, existing Pyro generative models |

**Factors to study:**

1. **Non-centered parameterization (NCP)**:
   Instead of `A_ij ~ Normal(0, sigma_A)`, use
   `A_raw_ij ~ Normal(0, 1)` and `A_ij = sigma_A * A_raw_ij`.
   NCP is standard best practice for NUTS and often helps VI.
   The 2025 DCM PPL paper found that initialization at prior tails
   (2.5th/97.5th percentile) solved convergence issues.

2. **Prior scale sensitivity**:
   Test sigma_A in {0.1, 0.3, 0.5, 1.0}. Current default is 0.5.
   Tighter priors may improve calibration by constraining posteriors;
   looser priors may underperform with mean-field.

3. **Prior predictive checks**:
   Sample A from prior, simulate BOLD/CSD, check if prior implies
   plausible data. "If the posterior sd is more than 0.1 times the
   prior sd, the prior is considered informative" (Stan wiki).

4. **Stability constraints**:
   Current approach: diagonal A_ii constrained to be negative (stability).
   Alternative: eigenvalue constraint on full A matrix.

---

### D-4: Practical Recommendation Guide

| Aspect | Detail |
|--------|--------|
| Feature | Decision tree / flowchart for "which method should I use?" |
| Value Proposition | Translates the benchmark into actionable guidance. The single most useful output for users. |
| Complexity | Low (after benchmarks complete) |
| Confidence | HIGH |
| Dependencies | All benchmarks complete with results |

**Format:** Decision tree as both text and figure.

**Proposed structure:**

```
Q: How many regions?
  > 10 regions:
    -> Use rDCM (only method that scales)
  3-10 regions:
    Q: Do you need uncertainty quantification?
      No:
        -> Use AutoDelta (MAP). Fastest, no uncertainty.
      Yes:
        Q: How much time do you have?
          < 1 minute per subject:
            -> Amortized NSF (if pre-trained guide available)
            -> Amortized + 50 refinement steps (best speed/accuracy)
          < 10 minutes per subject:
            -> AutoLowRank SVI (best calibration/speed tradeoff)
          Unlimited:
            Q: Do you need gold-standard posteriors?
              Yes:
                -> NUTS (4 chains, R-hat < 1.01)
              No:
                -> AutoMultivariate SVI (best VI calibration)
```

**Also include:**
- Method comparison radar chart (accuracy, calibration, speed, scalability)
- "When to use each method" summary table
- Warnings and caveats per method

---

### D-5: Per-Parameter Posterior Comparison Plots

| Aspect | Detail |
|--------|--------|
| Feature | Overlay posteriors from all methods for each A_ij element |
| Value Proposition | Visual comparison is more informative than aggregate metrics. Shows where methods agree/disagree. |
| Complexity | Low-Medium |
| Confidence | HIGH |
| Dependencies | All methods run on same datasets, posterior sample storage |

**Plot type:** For a single representative dataset, show:
- Violin/ridge plot of posterior for each A_ij element
- Each method as a different color
- True value marked as vertical line
- NUTS posterior as reference band

This is the "money figure" that reviewers and readers remember.

---

### D-6: Wall-Clock Timing Analysis

| Aspect | Detail |
|--------|--------|
| Feature | Detailed timing breakdown by component |
| Value Proposition | Users need to know not just total time, but what dominates. ODE solving? Gradient computation? Guide evaluation? |
| Complexity | Low |
| Confidence | HIGH |
| Dependencies | Existing timing infrastructure in runners |

**Already partially built:** Runners track wall time per dataset.

**Extension:** Break down into:
- ODE integration time (task DCM only)
- Forward model evaluation time
- Guide evaluation time
- Gradient computation time
- Per-SVI-step time
- Amortized: training time vs inference time

---

## Anti-Features

Things to explicitly NOT build. Common mistakes in this domain.

### AF-1: Do NOT Build Cross-PPL Runtime Benchmarking

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Benchmarking Pyro vs NumPyro vs PyMC vs Stan wall-clock times |
| Why Avoid | The 2025 DCM PPL paper already did this (NumPyro 16x faster than Stan). Repeating it adds no scientific value and opens a can of worms about fair comparison. |
| What to Do Instead | Use NumPyro for NUTS only (as validation reference). Focus on inference quality, not runtime comparison between PPLs. |

---

### AF-2: Do NOT Build Automatic Model Selection

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Automated pipeline that picks the "best" inference method per dataset |
| Why Avoid | Premature optimization. The benchmark establishes the tradeoffs; the user makes the decision based on their constraints. |
| What to Do Instead | Provide the recommendation guide (D-4) with clear criteria. |

---

### AF-3: Do NOT Build Real-Data Benchmarking

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Running benchmarks on real fMRI data |
| Why Avoid | With real data, ground truth is unknown. Cannot compute RMSE, coverage, or any recovery metric. Simulation-based benchmarking is the correct approach for validating inference methods. |
| What to Do Instead | Use simulated data with known ground truth. Note in the paper that real-data application is future work. |

---

### AF-4: Do NOT Build Posterior Recalibration

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Post-hoc recalibration of VI posteriors (conformal prediction, score calibration) |
| Why Avoid | While methods like CANVI and Bayesian Score Calibration exist, they add complexity and mask the underlying approximation error. The paper should characterize the error, not hide it. |
| What to Do Instead | Document the calibration gap honestly. Recommend richer guide families (low-rank, flows) rather than post-hoc fixes. Post-hoc calibration is a valid future extension. |

---

### AF-5: Do NOT Build Whole-Brain (50+ Region) Benchmarks

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Benchmarking at whole-brain scale (50-200 regions) |
| Why Avoid | Only rDCM scales to this range. Task and spectral DCM with ODE integration cannot handle 50+ regions with any VI method in reasonable time. Comparing rDCM-only at 50 regions adds little to the cross-method story. |
| What to Do Instead | Benchmark at 3, 5, 10 regions. Note rDCM scalability advantage in discussion. |

---

### AF-6: Do NOT Build Custom Guide Architectures

| Aspect | Detail |
|--------|--------|
| Anti-Feature | Designing novel guide architectures (structured guides, hierarchical guides, etc.) |
| Why Avoid | Scope creep. The paper is about characterizing existing methods, not inventing new ones. Pyro's autoguide API provides sufficient variety. |
| What to Do Instead | Use Pyro's built-in autoguides (AutoNormal, AutoLowRank, AutoMultivariate, AutoIAF). The AmortizedFlowGuide is the one custom guide already built. |

---

## Feature Dependencies

```
TS-8 (Benchmark Runner)
  |
  +-- TS-2 (Guide Variants)
  |     |
  |     +-- TS-3 (Coverage Calibration)
  |     |     |
  |     |     +-- D-1 (SBC) [stretch]
  |     |
  |     +-- TS-4 (ELBO Variants)
  |     |
  |     +-- D-5 (Per-Parameter Plots)
  |
  +-- TS-1 (NUTS Reference)
  |     |
  |     +-- TS-3 (Coverage Calibration)
  |     +-- D-5 (Per-Parameter Plots)
  |
  +-- TS-5 (Network Size Scaling)
  |     |
  |     +-- TS-3 (Coverage Calibration)
  |
  +-- TS-6 (Amortization Gap Fix)
  |     |
  |     +-- D-2 (Amortized + Refinement)
  |
  +-- TS-7 (Cross-Backend Table)
  |     |
  |     +-- D-4 (Recommendation Guide) [after all results]
  |
  +-- D-3 (Regularization Study) [parallel track]
  +-- D-6 (Timing Analysis) [low effort, parallel]
```

**Critical path:** TS-2 (guides) and TS-1 (NUTS) must come first.
Then TS-3 (calibration) depends on both. TS-7 (table) and D-4 (guide)
are synthesis steps that come last.

---

## MVP Recommendation

For MVP (minimum credible paper), prioritize:

1. **TS-6: Fix amortization gap** -- Remove the synthetic placeholder. Low
   effort, high impact (unblocks honest reporting).
2. **TS-2: Extended guide variants** -- Add AutoLowRank and AutoMultivariate
   to the benchmark. One-line guide changes, biggest calibration insight.
3. **TS-1: NUTS reference** -- NumPyro NUTS for spectral DCM at 3 regions.
   Gold standard comparison.
4. **TS-3: Coverage calibration plot** -- The key figure of the paper.
5. **TS-7: Cross-backend table** -- The key table of the paper.
6. **D-4: Recommendation guide** -- The key practical output.

Defer to post-MVP:
- **D-1 (SBC)**: Computationally expensive (27K+ inference runs). Important
  but not required for the initial paper. Can be added as supplementary.
- **D-3 (Regularization study)**: Interesting but orthogonal. Can be a
  separate section/paper.
- **TS-5 at 10 regions**: Start with 3 and 5 regions. 10-region benchmarks
  are expensive and may require algorithm-specific tuning.
- **D-2 (Amortized + refinement)**: Conceptually important but requires
  careful engineering of the initialization transfer.

---

## Sources

### HIGH Confidence
- [Pyro Automatic Guide Generation docs](https://docs.pyro.ai/en/dev/infer.autoguide.html) --
  AutoNormal, AutoLowRank, AutoMultivariate, AutoIAF specifications
- [Pyro SVI Tips and Tricks](https://pyro.ai/examples/svi_part_iv.html) --
  Guide progression, init_scale, learning rate strategies
- [NumPyro autoguide docs](https://num.pyro.ai/en/stable/autoguide.html) --
  NumPyro guide variants and NUTS configuration
- v0.1.0 benchmark results (benchmarks/results/, 08-VERIFICATION.md) --
  Current coverage numbers (0.44-0.78 for 90% nominal)

### MEDIUM Confidence
- [DCM in Probabilistic Programming Languages (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133347/) --
  NUTS vs ADVI vs Laplace for DCM, PPL runtime comparison, convergence
  with 200 warmup samples, full-rank ADVI equivalent to NUTS, mean-field
  ADVI underestimates variance
- [SBC: Choice of Test Quantities (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12490788/) --
  Modern SBC methodology, test quantity selection
- [Posterior SBC (2025)](https://arxiv.org/html/2502.03279v2) --
  Data-conditional SBC validation
- [Amortized VI: When and Why? (2023)](https://arxiv.org/abs/2307.11018) --
  Conditions for closing the amortization gap
- [CANVI: VI with Coverage Guarantees (2023)](https://arxiv.org/html/2305.14275v3) --
  Conformal calibration for amortized inference
- [R-hat improvements (Vehtari et al. 2021)](https://arxiv.org/pdf/1903.08008) --
  R-hat < 1.01 threshold, bulk/tail ESS

### LOW Confidence
- [Amortized VI Systematic Review (JAIR 2023)](https://dl.acm.org/doi/abs/10.1613/jair.1.14258) --
  Comprehensive review but PDF extraction failed; claims based on abstract
- [TREND: Transformer DCM (2024)](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00290/124203/) --
  Scaling to larger networks with amortized approach; full text not accessed
- [Stan Prior Choice Recommendations](https://github.com/stan-dev/stan/wiki/prior-choice-recommendations) --
  Prior sensitivity workflow; general Bayesian best practice, not DCM-specific
