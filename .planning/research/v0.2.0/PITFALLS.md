---
type: "research"
scope: "pitfalls"
milestone: "v0.2.0"
domain: "cross-backend inference benchmarking for probabilistic DCM"
updated: "2026-04-06"
confidence: "HIGH (verified against Baldy et al. 2025, Modrak et al. 2023, TVB paper 2025)"
---

# Pitfalls: Cross-Backend Inference Benchmarking for Pyro-DCM v0.2.0

This document catalogs specific mistakes that commonly occur when adding
cross-backend inference benchmarking to an existing probabilistic programming
project. Each pitfall is grounded in the current Pyro-DCM codebase state
(v0.1.0 shipped, 7 runners, coverage 0.44--0.78) and draws on the Baldy,
Woodman, Jirsa & Hashemi (2025) paper that benchmarked DCM inference in
NumPyro/PyMC/Stan -- the closest published analog to what v0.2.0 will do.

---

## Critical Pitfalls

Mistakes that produce scientifically misleading results or require major rework.

---

### P1: Treating 90% Coverage as a Universal Gold Standard

**What goes wrong:** The v0.1.0 benchmark targets 0.90 nominal coverage for
90% credible intervals. The temptation is to treat any method achieving <0.90
coverage as "broken" and any method achieving >=0.90 as "calibrated." This is
wrong for three independent reasons:

1. **Mean-field VI cannot achieve 0.90 coverage on correlated posteriors by
   design.** AutoNormal assumes independence between all parameters. For
   spectral DCM with N=10, the 143 latent dimensions have strong correlations
   (A_free elements interact through the eigenvalue constraint, noise
   parameters couple through the CSD likelihood). Mean-field marginals will be
   too narrow even if each marginal mean is correct. The Baldy et al. (2025)
   paper confirmed this: mean-field ADVI produced "uncorrelated and
   under-dispersed outputs" that systematically missed ground-truth parameters
   in their DCM ODE model.

2. **0.90 coverage with wrong posterior shape is worse than 0.78 coverage with
   correct shape.** A method could achieve 0.90 coverage by inflating all
   variances uniformly (over-dispersed in some dimensions, under-dispersed in
   others). Simpson's paradox: aggregated coverage can hit 0.90 while every
   individual parameter has wrong coverage.

3. **Coverage is sensitive to the test distribution.** Coverage computed over
   randomly sampled A matrices from `make_random_stable_A` may differ from
   coverage on realistic brain connectivity patterns. The Frassle et al.
   rDCM reliability paper showed that recovery performance depends heavily on
   the ground-truth connectivity structure.

**Warning signs:**
- A guide achieves exactly 0.90 coverage but posterior samples show no
  correlation between A_free elements
- Coverage varies wildly across seeds (std > 0.15)
- Coverage on diagonal vs off-diagonal A elements differs by >0.20

**Prevention:**
- Report coverage per-parameter, not just aggregated. Show calibration curves
  (expected vs observed coverage at 50%, 75%, 90%, 95%) separately for
  diagonal A, off-diagonal A, noise parameters.
- Use SBC (simulation-based calibration) rank histograms as the primary
  calibration diagnostic, not just marginal coverage. SBC detects shape
  problems that aggregate coverage misses.
- Accept that AutoNormal coverage ceiling is ~0.80--0.88 for correlated
  posteriors. Document this as a known limitation, not a bug.
- Compare coverage improvement from AutoNormal -> AutoLowRank ->
  AutoMultivariateNormal to show the coverage gain from capturing correlations.

**Phase relevance:** Calibration analysis phase. Must be addressed before any
cross-method comparison table is published.

**Confidence:** HIGH. Baldy et al. (2025) directly confirmed mean-field
underdispersion for DCM. Current v0.1.0 data (coverage 0.44--0.78) is
consistent with this mechanism.

---

### P2: Using NUTS as Gold Standard Without Verifying Convergence for DCM

**What goes wrong:** The plan assumes NumPyro NUTS provides ground-truth
posteriors to compare VI methods against. For DCM, NUTS can fail silently in
several ways:

1. **Multimodality from parameter degeneracy.** Baldy et al. (2025) found
   that 24 NUTS chains with diffuse priors converged to "different regions or
   modes of parameter space." For their 10-parameter neural mass model, some
   chains found local optima that "fail to recover perfectly true dynamics."
   DCM with N=10 has 100+ parameters -- the multimodality risk is higher.

2. **Standard R-hat misses multimodality.** R-hat < 1.01 per chain does NOT
   guarantee all chains found the same mode. Split-R-hat with 4 chains can
   report convergence even when 2 chains are in mode A and 2 in mode B. Baldy
   et al. found R-hat of 1.007--1.009 even when some chains had not found the
   global optimum.

3. **ODE integration creates correlated gradients.** The task DCM forward model
   integrates a coupled neural-hemodynamic ODE (5N states) over ~60--500s with
   dt=0.5. Leapfrog integration in HMC accumulates numerical error through the
   ODE solver, potentially causing energy conservation violations that manifest
   as high divergence counts rather than explicit NaN.

4. **ESS can be misleading for multimodal targets.** A chain stuck in one mode
   will report high ESS (low autocorrelation within-mode) while having zero
   coverage of the other mode.

**Warning signs:**
- Divergence count > 5% of total transitions
- ESS/total_samples < 0.25 (i.e., less than 25% efficiency)
- Different random seeds produce different NUTS posterior means (> 0.1 RMSE
  between seed runs)
- NUTS posterior is tighter than the VI posterior (should generally be the
  opposite)

**Prevention:**
- Run >= 8 chains (not just 4) with diverse initialization. Baldy et al.
  found that initializing from prior tails (below 2.5th, above 97.5th
  percentile) achieved "100% convergence" across 80 chains.
- Check R-hat AND cross-chain posterior overlap (e.g., energy distance between
  chains) to detect multi-modality.
- Set explicit convergence criteria before running: R-hat < 1.01, bulk ESS >
  400, tail ESS > 200, divergences < 1%.
- For task DCM with N >= 5, consider using the Laplace approximation as a
  secondary reference. Baldy et al. found Laplace "equivalent" to NUTS for
  their DCM, with RMSE = 0.097 for both.
- Store full NUTS traces (not just summaries) so multimodality can be diagnosed
  post-hoc.

**Phase relevance:** NumPyro backend implementation phase. Must be resolved
before NUTS results are used as the reference for any VI comparison.

**Confidence:** HIGH. Directly observed by Baldy et al. (2025) in a published
DCM inference benchmark. The 10-parameter system showed multimodality; our
100+ parameter system will be worse.

---

### P3: Unfair Cross-Framework Comparison (Pyro vs NumPyro)

**What goes wrong:** Pyro (PyTorch) and NumPyro (JAX) use different numerical
backends, different optimizers, different default settings, and different
compilation strategies. Naive comparison produces confounded results:

1. **Numerical precision defaults differ.** PyTorch defaults to float32; our
   codebase forces float64 via `.double()`. JAX defaults to float32 unless
   `jax.config.update("jax_enable_x64", True)` is set. If the NumPyro backend
   runs in float32 while Pyro runs in float64, CSD computation differences
   will dominate the comparison, not inference quality differences.

2. **Optimizer implementations differ.** Pyro's `ClippedAdam` and NumPyro's
   `Adam` are not identical. ClippedAdam includes gradient clipping; NumPyro's
   `optax.adam` does not by default. The learning rate decay schedule
   (`lrd` parameter) in `run_svi` has no direct NumPyro equivalent.

3. **JIT compilation confounds wall-clock time.** NumPyro JIT-compiles the
   model+guide on first call. Pyro does not JIT by default. If you include
   JIT compilation time in the first run, NumPyro looks slow. If you exclude
   it, NumPyro looks fast. The Baldy et al. (2025) paper found NumPyro was
   2--13x faster than Stan, but they measured end-to-end wall time, which
   includes compilation.

4. **ODE solver implementations differ.** `torchdiffeq` rk4 and `diffrax`
   Euler/RK4 have different numerical characteristics. Even with identical
   step sizes, floating-point accumulation differences over 1000+ steps
   produce different BOLD predictions. These differences propagate into ELBO
   and posterior estimates.

5. **Random number generation differs.** PyTorch and JAX have different PRNG
   algorithms. Same seed number produces different random draws. Any seed-level
   comparison is meaningless.

**Warning signs:**
- Wall-clock time comparison shows >10x difference (likely measuring
  compilation, not inference)
- ELBO values differ by >5% between backends for the same model specification
- One backend produces NaN while the other succeeds on the same data

**Prevention:**
- **Lock precision:** Set `jax_enable_x64 = True` for ALL NumPyro runs.
  Assert dtype == float64 at data boundaries.
- **Match optimizer settings:** Use the same learning rate, same clip norm,
  same decay schedule. Implement a thin wrapper that ensures identical
  hyperparameters regardless of backend.
- **Separate compilation from inference in timing.** Report: (a) compilation
  time, (b) per-step time, (c) total wall time. The meaningful comparison is
  per-step time x number of steps.
- **Do not compare ELBO values across backends.** Compare only downstream
  metrics (RMSE, coverage, correlation) computed identically from posterior
  samples.
- **Use identical ground-truth data.** Generate synthetic data once in numpy,
  save to disk, load in both backends. Do not regenerate with different PRNGs.
- **Match the forward model, not the code.** Verify that Pyro and NumPyro
  forward models produce the same predicted BOLD/CSD given identical
  parameters, within numerical tolerance (atol=1e-6).

**Phase relevance:** NumPyro backend implementation phase. These controls must
be designed into the runner infrastructure before any cross-backend runs.

**Confidence:** HIGH. Precision and optimizer differences are documented in
JAX and PyTorch documentation. Baldy et al. (2025) timing results demonstrate
the JIT compilation confound.

---

### P4: Conflating Amortization Gap with Approximation Gap

**What goes wrong:** The current `compute_amortization_gap` function in
`benchmarks/metrics.py` computes `elbo_amortized - elbo_svi`. This measures
the TOTAL inference gap, not the amortization gap specifically. The total gap
has two components:

- **Approximation gap:** The gap between the true posterior and the best
  possible approximation within the guide family (e.g., the best possible
  diagonal Gaussian). This exists even with infinite per-subject optimization.
- **Amortization gap:** The additional gap from using a shared amortized
  network instead of per-subject optimization within the same guide family.

The existing code conflates these. Worse, the `task_amortized.py` runner
computes a proxy amortization gap by scaling the SVI ELBO by the RMSE ratio
(line 409: `svi_result["final_loss"] * (1.0 + max(0.0, rmse_amort/rmse_svi -
1.0))`), which is not a valid measure of the ELBO gap at all.

**Warning signs:**
- "Amortization gap" is negative (amortized ELBO better than SVI ELBO) --
  this indicates SVI didn't converge, not that amortization helps
- Amortization gap is >50% of total ELBO -- likely includes approximation gap
- Gap doesn't decrease with more amortized training data

**Prevention:**
- Define the correct decomposition:
  ```
  Total gap = |ELBO_true - ELBO_amortized|
  Approx gap = |ELBO_true - ELBO_best_in_family|
  Amort gap = |ELBO_best_in_family - ELBO_amortized|
  ```
- To measure the amortization gap specifically: after amortized inference,
  fine-tune the flow parameters on the specific test subject for 100+ steps.
  The ELBO improvement from fine-tuning IS the amortization gap.
- Alternatively: compare amortized flow ELBO against per-subject flow
  optimization (same architecture, same number of flow transforms), NOT
  against AutoNormal SVI. AutoNormal vs flow is an approximation gap
  comparison, not an amortization gap comparison.
- Replace the RMSE-ratio proxy in `task_amortized.py` with actual ELBO
  evaluation using the wrapper model. This requires passing the packer and
  wrapper model through the runner, which the current architecture does not
  support.

**Phase relevance:** Amortization study phase. The metrics must be corrected
before any amortization gap figures are produced.

**Confidence:** HIGH. The decomposition is well-established in the SBI
literature (Cremer et al. 2018 "Inference Suboptimality in Variational
Autoencoders"). The existing code's RMSE-ratio proxy is verifiably incorrect.

---

### P5: SBC That Passes but Detects Nothing

**What goes wrong:** Simulation-based calibration is the recommended diagnostic
for posterior calibration, but naive SBC implementation has known blind spots:

1. **SBC with rank statistics alone cannot detect when the posterior equals the
   prior.** If the inference algorithm ignores the data entirely (a bug), SBC
   rank histograms will be perfectly uniform. Modrak et al. (2023) formally
   proved this and showed "wide sets of artificial counterexamples that
   incorrectly pass SBC."

2. **SBC is only as sensitive as the test quantities you choose.** Using only
   marginal ranks tests marginal calibration. Mean-field VI will pass marginal
   SBC even though joint calibration is wrong, because each marginal is a
   well-calibrated Normal -- it's the correlations that are missing.

3. **Insufficient SBC simulations produce noisy histograms.** With 20 datasets
   (the current `full_config` for task DCM), the rank histogram has only 20
   entries. You cannot distinguish a uniform distribution from a biased one
   with 20 samples.

**Warning signs:**
- SBC passes for a method known to be poorly calibrated (e.g., mean-field on
  correlated posterior)
- Rank histograms look uniform but the method's coverage is <0.70
- SBC uses the same prior for simulation and inference (will always pass)

**Prevention:**
- **Add data-dependent test quantities.** Modrak et al. (2023) recommend
  including the log-likelihood as a test quantity: `T(theta, y) = log p(y |
  theta)`. This catches algorithms that ignore the data.
- **Add joint test quantities.** For DCM: `T(theta) = max(real(eig(A)))` tests
  whether the posterior correctly captures the stability constraint. `T(theta)
  = ||A||_F` tests the overall connectivity magnitude.
- **Use enough simulations.** Minimum 100 for visual diagnostics (Talts et al.
  2018). Minimum 500 for reliable KS-test. This means SBC is a separate,
  expensive computation -- not embedded in the regular benchmark loop.
- **Use ECDF difference plots** (not just histograms). The ECDF-based test
  from the SBC R package has higher power than histogram visual inspection.
- **Consider posterior SBC** (Saila et al. 2025) for data-conditional
  calibration checks, especially for checking which parameterization (centered
  vs non-centered) works better for specific data.

**Phase relevance:** Calibration analysis phase. SBC should be implemented as a
standalone validation tool, separate from the benchmark runners.

**Confidence:** HIGH. Modrak et al. (2023) "Simulation-Based Calibration
Checking for Bayesian Computation: The Choice of Test Quantities Shapes
Sensitivity" published in Bayesian Analysis. Verified via PMC.

---

## High-Severity Pitfalls

Mistakes that cause significant delays or produce misleading sub-results.

---

### P6: Full-Rank Guide Memory Explosion at N=10

**What goes wrong:** AutoMultivariateNormal stores a full Cholesky factor of
shape (D, D). For our models at N=10:

- Task DCM: D=111 latent dims -> 6,327 guide parameters (Cholesky + loc)
- Spectral DCM: D=143 latent dims -> 10,439 guide parameters

This is manageable in isolation. But during SVI, each gradient step requires
computing gradients through the full covariance matrix AND the forward model:

- Task DCM forward pass: ODE integration of 50 states over ~120 steps
  (dt=0.5, duration=60s) = ~6,000 tensor operations
- Each operation generates a computational graph node
- Full-rank guide adds D^2/2 = ~6,000--10,000 additional parameters to
  backpropagate through

The combined memory footprint (forward model graph + full-rank guide) may
exceed GPU memory on consumer hardware (8GB) or cause severe slowdown on CPU.

Additionally, full-rank guides are known to have slow convergence in high
dimensions because the Cholesky factor has D*(D+1)/2 entries, many of which
interact. The optimization landscape for the Cholesky factor has O(D^3)
curvature, requiring careful learning rate tuning.

**Warning signs:**
- SVI wall time increases >4x when switching from AutoNormal to
  AutoMultivariateNormal at the same N
- GPU out-of-memory errors at N=10 with num_particles > 1
- Full-rank guide loss oscillates rather than decreasing
- Final ELBO is worse than AutoNormal (guide capacity too high for data)

**Prevention:**
- Use AutoLowRankMultivariateNormal with rank = floor(sqrt(D)) as the default
  structured guide. For N=10 spectral DCM: rank=11, requiring 1,859 guide
  parameters vs 10,439 for full rank.
- Only run AutoMultivariateNormal at N=3 and N=5. At N=10, use low-rank.
  Document this as an explicit design decision in the benchmark config.
- Profile memory before scaling: run one SVI step at N=10 with each guide
  type and measure peak memory.
- Consider AutoGuideList: full-rank for A_free (the scientifically important
  parameters), AutoNormal for noise parameters (less important to capture
  correlations).

**Phase relevance:** Guide variant implementation phase. Memory profiling must
happen before the full benchmark matrix is run.

**Confidence:** HIGH. Parameter counts computed directly from model structure.
Pyro documentation warns about high-dimensional full-rank guides.

---

### P7: Non-Centered Parameterization Applied When Centered is Better

**What goes wrong:** Non-centered parameterization (NCP) is often recommended
for hierarchical models, but it is NOT universally better. The centered vs
non-centered tradeoff depends on the data informativeness:

- **Data-rich regime (fMRI with 100+ time points per subject):** Centered
  parameterization is more efficient. The likelihood dominates the prior,
  and posterior geometry is well-conditioned.
- **Data-poor regime (short scans, low SNR):** Non-centered helps by
  decoupling the prior from the likelihood, eliminating funnel geometry.

For DCM specifically:
- Task DCM with 30+ TRs and SNR=5: moderately data-rich. Centered is likely
  fine for diagonal parameters; NCP may help for weakly identified
  off-diagonal connections.
- Spectral DCM with 32 frequency bins x 3x3 CSD: quite data-rich (288 real
  observations for 24 parameters at N=3). Centered is likely better.
- rDCM: explicitly data-rich by design (1000+ BOLD time points). Centered is
  strongly preferred.

The danger: applying NCP globally to all parameters when only a few benefit,
wasting optimization budget on reparameterization that makes convergence
slower for the well-identified parameters.

**Warning signs:**
- NCP model has higher ELBO (worse) than centered model on the same data
- NCP requires more SVI steps to converge than centered
- NCP posterior means are identical to centered but posterior widths differ
  (indicating the reparameterization didn't help, just changed the scale)

**Prevention:**
- Test both parameterizations on a reference dataset BEFORE running the full
  benchmark. Compare convergence speed and final ELBO.
- Consider the ASIS (Ancillarity-Sufficiency Interweaving Strategy) approach:
  alternate between centered and non-centered parameterizations during
  sampling. However, this adds implementation complexity.
- For the benchmark comparison: if testing NCP as a "regularization" strategy,
  compare it on the same data with the same optimizer settings. The only
  variable should be the parameterization.
- For Pyro SVI: NCP is less impactful than for MCMC because SVI already uses
  reparameterization gradients through the guide. The guide's variational
  distribution handles the posterior geometry. NCP primarily helps NUTS, not
  SVI.
- Start with centered for all Pyro SVI methods. Add NCP only for NUTS and
  only if divergences > 5%.

**Phase relevance:** Regularization study phase. This is a sensitivity analysis,
not a required feature for all methods.

**Confidence:** MEDIUM. The centered-vs-NCP tradeoff is well-established in the
Stan/MCMC literature. Its relevance to SVI with Pyro auto-guides is less
studied. The Baldy et al. (2025) paper did not explicitly discuss this
tradeoff for their DCM.

---

### P8: Prior Sensitivity Study That Biases Toward One Method

**What goes wrong:** When varying prior scales (e.g., A_free ~ N(0, sigma^2)
with sigma in {1/64, 1/16, 1/4, 1}), the comparison can be inadvertently
biased:

1. **Tight priors favor VI; diffuse priors favor MCMC.** VI with AutoNormal
   benefits from tight priors because the posterior is more Gaussian (closer
   to the guide family). MCMC can explore the full posterior regardless of
   prior width. If you declare a winner at tight priors, VI wins. If you
   declare a winner at diffuse priors, MCMC wins.

2. **SPM's prior is not neutral.** SPM uses N(0, 1/64) for A_free, which is
   quite tight (prior std = 0.125). This prior was tuned for Variational
   Laplace, which is similar to mean-field VI. Using SPM's prior favors
   mean-field VI in the comparison.

3. **The ground-truth simulation prior affects the result.** If you simulate
   data with A ~ Uniform over stable matrices but infer with A_free ~ N(0,
   1/64), the prior is misspecified. Methods that are robust to prior
   misspecification (MCMC, full-rank VI) will look better than methods
   that rely on the prior (mean-field VI, Laplace).

**Warning signs:**
- Method ranking changes completely when prior scale changes by 2x
- One method has uniformly best results at every prior scale (suspicious --
  the tradeoff should flip)
- Coverage increases above 0.95 at very tight priors (prior is doing the work,
  not the data)

**Prevention:**
- Use the SAME prior for ALL methods in the primary comparison. SPM's N(0,
  1/64) is the natural default since it matches the published DCM literature.
- Present prior sensitivity as a SEPARATE analysis, not interleaved with the
  primary method comparison.
- When varying priors, also vary the ground-truth simulation to match. If
  inference uses N(0, 1/16), simulate data from N(0, 1/16) so the prior is
  well-specified.
- Report the effective number of parameters (p_eff from WAIC or LOO-CV) as
  a diagnostic. If p_eff is much smaller than the latent dimension, the prior
  is dominating.

**Phase relevance:** Regularization study phase.

**Confidence:** MEDIUM. The bias mechanism is well-understood in Bayesian
statistics. Application to specific DCM comparison is extrapolated.

---

### P9: Simpson's Paradox in Aggregated Benchmark Tables

**What goes wrong:** The benchmark produces a 9-method x 3-variant x 3-size
table. Aggregating across variants or sizes can reverse method rankings:

Example: Method A has the best RMSE for task DCM and spectral DCM (2 of 3
variants), but Method B has the best RMSE for rDCM (1 of 3 variants). If you
average RMSE across all 3 variants, Method B might win overall because rDCM
produces much larger RMSE values that dominate the average.

Similarly: Method A is best at N=3 and N=5, but Method B is best at N=10.
If you average across sizes, Method B might win because N=10 problems have
larger absolute errors.

**Warning signs:**
- Method rankings differ between variant-specific and aggregated tables
- One variant or size dominates the aggregated metric
- "Best overall method" is never best for any individual variant

**Prevention:**
- **Never aggregate across variants.** Task DCM, spectral DCM, and rDCM are
  different problems with different dimensionalities and likelihood structures.
  Always present results per-variant.
- **If aggregating across sizes:** Use relative metrics (e.g., RMSE / RMSE_best
  at each size) rather than absolute metrics. This normalizes the scale.
- **Use rank-based aggregation.** Rank methods 1--9 within each (variant, size)
  cell, then compute mean rank across cells. This is robust to scale
  differences.
- **Show the full table.** A heatmap of (method, variant, size) -> metric is
  more informative than any single summary number.

**Phase relevance:** Results analysis and reporting phase.

**Confidence:** HIGH. This is a standard statistical pitfall. The specific risk
is high because rDCM RMSE (0.10--0.15 in v0.1.0) is 5--10x higher than
spectral DCM RMSE (0.01--0.02), which would dominate any naive average.

---

### P10: Amortized Guide Trained on Wrong Distribution

**What goes wrong:** The amortized flow guide learns p(theta | y) for data y
drawn from a training distribution p(y). If the test data comes from a
different distribution, the amortized posterior is unreliable. Specific risks:

1. **Training distribution too narrow.** The current inline training in
   `task_amortized.py` generates 50 training datasets with
   `make_random_stable_A(N, seed=seed+500+i)`. If this produces A matrices in
   a narrow range, the guide generalizes poorly to test A matrices outside
   that range.

2. **Training distribution doesn't match the benchmark evaluation.** If the
   benchmark evaluates on A matrices drawn from a different distribution
   (e.g., sparser A, or A with specific eigenvalue structure), the
   amortization gap measurement conflates "generalization gap" (wrong training
   distribution) with "amortization gap" (capacity limitation).

3. **SNR mismatch.** Training at SNR=5 but evaluating at different SNR levels
   means the guide has never seen the noise regime it's being tested on. The
   summary network's learned features may be specific to the training noise
   level.

**Warning signs:**
- Amortized RMSE is >3x SVI RMSE (suggests distributional mismatch, not just
  capacity limitation)
- Amortized coverage is <0.30 (the guide is confidently wrong, indicating
  out-of-distribution data)
- Amortized performance improves dramatically with 100 fine-tuning steps
  (the gap was from distribution mismatch, not capacity)

**Prevention:**
- Train the amortized guide on a WIDER distribution than the test distribution.
  Use broader priors for training: A_free ~ N(0, 1/16) for training even if
  the model uses N(0, 1/64).
- Explicitly verify overlap between training and test distributions. Plot
  histograms of eigenvalue spectra for training vs test A matrices.
- For the amortization gap study specifically: fine-tune the trained guide on
  each test subject for 50--200 steps. The ELBO improvement from fine-tuning
  is the clean amortization gap. If fine-tuning produces >80% of the total
  gap reduction, the original gap was distributional mismatch.
- Match training and test SNR exactly. If the benchmark varies SNR, train
  separate guides per SNR level or train on a SNR range.

**Phase relevance:** Amortization study phase. Training distribution design
must be settled before guide training begins.

**Confidence:** HIGH. The distributional mismatch problem is well-documented
in the amortized inference literature (Radev et al. 2020, Cranmer et al. 2020).
The current codebase's inline training with only 50 examples (CI mode) is
known to produce poor coverage (0.55--0.65, documented in STATE.md).

---

## Moderate Pitfalls

Mistakes that cause delays, wasted compute, or technical debt.

---

### P11: Combinatorial Explosion Without Compute Budget

**What goes wrong:** The v0.2.0 benchmark matrix is:
- 9 methods x 3 variants x 3 sizes x 5 seeds x 20 datasets = 8,100 runs
- At 30s average per run: ~68 hours of compute
- At 120s average (task DCM with ODE): ~270 hours of compute

Running the full matrix without prioritization means waiting days for results,
then discovering that 60% of the comparisons are uninformative (e.g., NUTS on
task DCM N=10 is prohibitively slow and doesn't finish).

**Warning signs:**
- Benchmark runner queued for >24 hours with no results
- >50% of runs fail or time out
- Results arrive but the interesting comparison wasn't in the first batch

**Prevention:**
- **Tier the comparisons:**
  - Tier 1 (must-have): All Pyro guides x spectral DCM x N=3. This is the
    cheapest, most informative comparison (spectral converges fast, N=3 is
    small).
  - Tier 2 (important): NUTS vs best-Pyro x all variants x N=3,5. Establishes
    the VI-vs-MCMC comparison.
  - Tier 3 (nice-to-have): Full matrix at N=10. Scalability analysis.
- **Set per-run timeouts.** If a single NUTS run on task DCM N=10 takes >1
  hour, record "timeout" rather than waiting indefinitely.
- **Run incrementally.** Save results after each (method, variant, size) cell.
  Don't require the full matrix to complete before analysis.
- **Eliminate invalid cells.** Not all method-variant combinations are valid:
  - rDCM analytic VB only works for rDCM (not task/spectral)
  - NUTS on task DCM N=10 may be infeasible (>1hr/chain)
  - Amortized guides need retraining per variant and size
  The actual valid matrix is ~60 cells, not 81.
- **Use quick mode for debugging.** Run with `--quick` (3 datasets, 500 steps)
  to verify all runners work before committing to full runs.

**Phase relevance:** Benchmark infrastructure phase. Compute budget must be
planned before starting full runs.

**Confidence:** HIGH. Computed directly from current config dimensions and
v0.1.0 timing data (task DCM ~1-2s/step, 3000 steps = 50+ minutes).

---

### P12: Reporting Averages Without Variance or Per-Dataset Breakdowns

**What goes wrong:** The current `run_task_svi` runner computes `mean_rmse` and
`std_rmse` across datasets. But the mean can hide important structure:

1. **Bimodal failure modes.** If 15/20 datasets produce RMSE=0.03 and 5/20
   produce RMSE=0.50 (ODE divergence, poor identifiability), the mean is
   0.15 and std is 0.20. The mean suggests mediocre performance when the
   method actually works well on identifiable problems and fails completely
   on non-identifiable ones.

2. **Seed sensitivity.** If results change dramatically across random seeds,
   the "mean" result is not representative. The existing rDCM decision
   documents this: "Seed 456 produces degenerate data; all masks converge to
   same diagonal solution" (STATE.md).

3. **Cherry-picking is tempting.** With 5 seeds, you can always find the seed
   that makes your method look best. Without pre-registration or reporting all
   seeds, results are not reproducible.

**Warning signs:**
- std_rmse > 0.5 * mean_rmse (high coefficient of variation)
- Removing 1--2 datasets changes the mean by >20%
- Results look great at seed=42 but terrible at seed=123

**Prevention:**
- **Report per-dataset scatter plots** (true vs inferred) alongside summaries.
  The v0.1.0 plotting module already generates these.
- **Report median and IQR** in addition to mean and std. Median is robust to
  outlier datasets.
- **Report failure rate** explicitly. "Method X: 85% success rate, RMSE=0.03
  (successful runs only)" is more informative than "Method X: RMSE=0.15
  (all runs including failures)."
- **Pre-register seeds.** Choose 5 seeds before running any method, use the
  same 5 seeds for all methods, report all 5.
- **Use paired comparisons.** Method A vs Method B on the same dataset is more
  informative than Method A average vs Method B average. Paired t-test or
  Wilcoxon signed-rank on per-dataset metric differences.

**Phase relevance:** Results analysis phase.

**Confidence:** HIGH. This is standard benchmarking methodology. The existing
codebase already tracks per-dataset results (`rmse_list`, `coverage_list`),
but the analysis currently only computes means.

---

### P13: Comparing Computational Cost Without Equating Convergence

**What goes wrong:** Comparing "Method A takes 10s, Method B takes 100s"
is meaningless if Method A ran 500 steps and Method B ran 5000 steps.
The question is not which method is faster per wall clock, but which method
produces better results per unit compute.

Specific confounds in the current codebase:
- Task SVI uses 3000 steps with lr_decay_factor=0.01. Spectral SVI uses 500
  steps. These are not the same computational budget.
- NUTS wall time includes warmup. With 200 warmup + 200 sampling (Baldy et
  al.'s configuration), half the time is warmup.
- Amortized inference wall time excludes training. The 0.01s inference time
  is only meaningful if you amortize the 5+ minute training time over many
  subjects.

**Warning signs:**
- The "fastest" method also has the worst RMSE (it simply ran fewer steps)
- NUTS appears much slower than SVI but produces much tighter posteriors
- Amortized appears much faster but the training time isn't counted

**Prevention:**
- **Report cost as effective-samples-per-second** for MCMC and
  ELBO-evaluations-per-second for SVI.
- **Show Pareto frontier plots:** accuracy (RMSE or coverage) vs computational
  cost (wall time). The useful comparison is whether a method is Pareto
  optimal, not whether it's fastest.
- **Equate convergence.** For SVI methods, run until ELBO converges (e.g.,
  relative change < 1e-4 over last 100 steps), not for a fixed number of
  steps. Report the convergence step count as well as wall time.
- **Report amortized training cost separately.** Frame it as: "Amortized
  inference takes 5 minutes to train + 0.01s per subject. At >30 subjects,
  amortized is cheaper than per-subject SVI."

**Phase relevance:** Benchmark infrastructure phase. Convergence detection
should be built into all runners.

**Confidence:** HIGH. This is a well-known benchmarking pitfall in the ML
literature.

---

### P14: Zuko NSF Spline Domain Truncation in New Guide Variants

**What goes wrong:** The current AmortizedFlowGuide uses Zuko NSF with spline
bins on [-5, 5] (documented in STATE.md: "Standardize to [-5, 5] for NSF
splines"). Values outside this range are identity-mapped, meaning the flow
has no expressiveness for extreme values.

When adding new guide variants (e.g., more powerful flows with more transforms
or different architectures), this domain truncation can cause:

1. **Loss of posterior tails.** Parameters that the prior places near the
   boundary of [-5, 5] in standardized space get identity-mapped, producing
   artificially sharp posterior tails.
2. **Inconsistent calibration.** The flow captures the posterior bulk but
   not the tails, producing good RMSE but poor coverage at high CI levels
   (95%, 99%).
3. **Gradient issues at domain boundaries.** Spline knots at +/-5 can produce
   gradient discontinuities, causing SVI instability.

**Warning signs:**
- Coverage at 90% CI is good but coverage at 99% CI is much worse than
  expected (indicating tail truncation)
- Posterior samples cluster at +/-5 in standardized space
- SVI loss spikes periodically (gradient discontinuity at spline boundary)

**Prevention:**
- Verify that the standardization statistics (mean_, std_ from packer) place
  >99% of training parameter values within [-4, 4] (leave margin for the
  boundaries).
- Monitor the fraction of posterior samples that hit the [-5, 5] boundary.
  If >1%, the standardization is too tight.
- Consider using unbounded flows (e.g., autoregressive affine transforms)
  as a comparison point. If unbounded flows have better tail calibration,
  the spline domain is the bottleneck.
- When adding new guide variants, re-run the standardization fit with the
  broader training distribution recommended in P10.

**Phase relevance:** Guide variant implementation phase.

**Confidence:** MEDIUM. The domain truncation is documented in the codebase.
The specific impact on tail calibration is extrapolated from the NSF
literature, not directly measured.

---

## Minor Pitfalls

Mistakes that cause annoyance or minor technical debt but are easily fixable.

---

### P15: Pyro Param Store Leaks Between Benchmark Runs

**What goes wrong:** The current benchmark runners call `pyro.clear_param_store()`
at the start of each dataset (see `task_svi.py` line 119). But if a runner
crashes mid-way, the param store may retain parameters from the previous run.
The next run's guide initialization will be contaminated.

This is especially problematic when comparing guide variants: if
AutoLowRankMultivariateNormal stores more parameters than AutoNormal, and the
previous run's AutoLowRank params leak into the current AutoNormal run, Pyro
may raise shape mismatch errors or silently use stale parameters.

**Prevention:**
- Wrap each benchmark run in a try/finally that always calls
  `pyro.clear_param_store()`.
- Add `pyro.clear_param_store()` at the START of each runner function, not
  just inside the per-dataset loop.
- Consider using `pyro.poutine.uncondition` to create isolated inference
  contexts.

**Phase relevance:** Benchmark infrastructure phase.

---

### P16: NumPyro Model Translation Errors

**What goes wrong:** Translating the Pyro models to NumPyro requires replacing:
- `pyro.sample` -> `numpyro.sample`
- `pyro.deterministic` -> `numpyro.deterministic`
- `torch.Tensor` operations -> `jax.numpy` operations
- `torchdiffeq.odeint` -> `diffrax` or manual integration

Each translation point is a potential bug. The most dangerous: subtle numerical
differences in ODE integration that change the posterior mode location.

**Prevention:**
- Write a shared test that runs both Pyro and NumPyro models on the same
  input data and asserts that forward model outputs match to within atol=1e-6.
- Translate one model at a time (spectral first -- no ODE, simplest).
- Use the existing `benchmarks/metrics.py` functions for evaluation so the
  evaluation code is shared across backends.

**Phase relevance:** NumPyro backend implementation phase.

---

### P17: LaTeX Table Formatting That Obscures Results

**What goes wrong:** The benchmark deliverable includes LaTeX tables for a
paper. Common formatting mistakes:

1. Reporting too many significant figures (RMSE=0.01234567 vs RMSE=0.012)
2. Not bolding the best method per column
3. Using different CI levels in different columns without clear headers
4. Mixing absolute and relative metrics in the same table

**Prevention:**
- Use 3 significant figures for all metrics.
- Bold the best value per column, underline the second-best.
- Clearly indicate CI level in column headers (e.g., "Cov90" not just "Cov").
- Separate absolute (RMSE, time) and relative (coverage, correlation) metrics
  into different tables or table sections.

**Phase relevance:** Documentation and reporting phase.

---

## Phase-Specific Warning Summary

| Phase Topic | Most Likely Pitfall | Severity | Mitigation Priority |
|---|---|---|---|
| Calibration analysis | P1 (coverage as gold standard) | CRITICAL | First |
| NUTS reference posterior | P2 (NUTS not converged) | CRITICAL | Before any comparison |
| NumPyro backend | P3 (unfair comparison), P16 (translation bugs) | CRITICAL | During implementation |
| Guide variants | P6 (memory at N=10), P14 (spline truncation) | HIGH | During implementation |
| Amortization study | P4 (gap conflation), P10 (wrong distribution) | CRITICAL | Before gap measurement |
| SBC validation | P5 (passes but detects nothing) | CRITICAL | During calibration |
| Regularization study | P7 (NCP when centered better), P8 (prior bias) | HIGH | Design before running |
| Benchmark infrastructure | P11 (compute budget), P13 (cost comparison) | HIGH | Before full runs |
| Results and reporting | P9 (Simpson's paradox), P12 (no variance) | HIGH | During analysis |
| Documentation | P17 (formatting) | LOW | Final stage |

---

## Sources

### Primary (HIGH confidence)

- Baldy, Woodman, Jirsa & Hashemi (2025). "Dynamic causal modelling in
  probabilistic programming languages." J R Soc Interface, 22(227):20240880.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133347/)
  -- DCM inference in NumPyro/PyMC/Stan, convergence criteria, ADVI failures
- Modrak et al. (2023). "Simulation-Based Calibration Checking for Bayesian
  Computation: The Choice of Test Quantities Shapes Sensitivity." Bayesian
  Analysis. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12490788/)
  -- SBC limitations and improvements
- "Bend to Mend: Toward Trustworthy Variational Bayes with Valid Uncertainty
  Quantification" (2025). [arXiv](https://arxiv.org/html/2512.22655)
  -- VB undercoverage mechanisms and fractional posterior correction

### Secondary (MEDIUM confidence)

- Pyro AutoGuide documentation.
  [Docs](https://docs.pyro.ai/en/stable/infer.autoguide.html)
  -- Guide variant parameter counts and memory scaling
- Saila et al. (2025). "Posterior SBC: Simulation-Based Calibration Checking
  Conditional on Data." [arXiv](https://arxiv.org/html/2502.03279v2)
  -- Data-conditional SBC for parameterization selection
- Lueckmann et al. (2021). "Benchmarking Simulation-Based Inference."
  [arXiv](https://arxiv.org/abs/2101.04653)
  -- SBI benchmarking methodology
- Stan User Guide: Centered vs Non-centered Parameterization.
  [Stan](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html)
  -- When each parameterization helps/hurts

### Codebase-Derived (HIGH confidence for current state)

- `benchmarks/metrics.py` lines 160-195: amortization gap computation
- `benchmarks/runners/task_amortized.py` line 409: RMSE-ratio proxy
- `benchmarks/config.py`: benchmark dimensions (3-50 datasets, 500-3000 steps)
- `.planning/STATE.md`: 102 decisions documenting known issues
- Current coverage range 0.44-0.78 from v0.1.0 benchmark results
