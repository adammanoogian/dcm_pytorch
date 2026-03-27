# Phase 5: Parameter Recovery Tests (All Three Variants) - Research

**Researched:** 2026-03-27
**Domain:** Bayesian parameter recovery validation for DCM models (task, spectral, regression)
**Confidence:** HIGH (codebase thoroughly analyzed; methodology grounded in established Bayesian validation literature)

## Summary

Phase 5 is a rigorous parameter recovery validation phase: simulate data from known ground truth parameters, run inference, and verify that the inferred parameters match the truth within documented tolerances. This is the standard scientific validation protocol for Bayesian generative models. The phase produces test scripts and validation results (plots, metrics), not new library modules.

Three DCM variants require recovery tests: (1) task-DCM via Pyro SVI with AutoNormal guide, (2) spectral DCM via Pyro SVI with AutoNormal guide, (3) regression DCM via analytic VB (`rigid_inversion`, `sparse_inversion`) as primary, with optional Pyro SVI comparison. The key metrics are RMSE(A) < 0.05, 95% credible interval coverage in [0.90, 0.99], and correlation(A_true, A_inferred) close to 1.0. ELBO model comparison tests verify that correctly specified models achieve higher ELBO than misspecified alternatives.

**Primary recommendation:** Use the existing simulators (`simulate_task_dcm`, `simulate_spectral_dcm`, `simulate_rdcm`) and inference infrastructure (`run_svi`, `create_guide`, `extract_posterior_params`, `rigid_inversion`, `sparse_inversion`) directly. Compute 95% credible intervals from AutoNormal's `guide.quantiles([0.025, 0.975])` for SVI models and from the analytic posterior covariance (`Sigma_per_region`) for rDCM. Run a reduced dataset count (10-20 per configuration) for CI tests, with the full 50 reserved for optional slow/validation runs.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyro-ppl | >=1.9 | SVI inference, AutoNormal guide, ELBO | Already in project; provides `guide.quantiles()` for CIs |
| torch | >=2.0 | Tensor computation, random seed control | Already in project |
| pytest | latest | Test framework with markers, parametrize | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | latest | Scatter plots, ELBO traces, calibration plots | Results visualization (optional for CI, required for documentation) |
| scipy.stats | (bundled) | `scipy.stats.norm.ppf` for analytic CIs, `scipy.stats.pearsonr` for correlation | Coverage computation from analytic VB |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `guide.quantiles()` | Manual loc +/- 1.96*scale | quantiles() handles transforms correctly; manual is fragile for constrained params |
| matplotlib | No plots (metrics only) | Plots are required by success criteria; matplotlib is already in CLAUDE.md tech stack |
| scipy.stats.norm | torch.distributions.Normal.icdf | Either works; scipy is already a dependency |

## Architecture Patterns

### Recommended Test Structure
```
tests/
    test_task_dcm_recovery.py       # REC-01: Task DCM recovery
    test_spectral_dcm_recovery.py   # REC-02: Spectral DCM recovery
    test_rdcm_recovery.py           # REC-03: rDCM recovery (analytic VB primary)
    test_elbo_model_comparison.py   # REC-04: ELBO convergence + model comparison
validation/                          # Optional: full 50-dataset runs with plots
    run_recovery_validation.py       # Slow validation script
    recovery_results/                # Generated plots and tables
```

### Pattern 1: Recovery Test Loop (SVI-based)

**What:** For each of N synthetic datasets, simulate data with known A, run SVI to convergence, extract posterior, compute metrics.

**When to use:** Task DCM and Spectral DCM recovery tests.

**Example:**
```python
def run_single_recovery_svi(
    model_fn, simulate_fn, simulate_kwargs, model_args_fn,
    num_steps=3000, lr=0.005, seed=None,
):
    """Run one recovery trial: simulate -> infer -> compare."""
    # 1. Simulate with known ground truth
    sim_result = simulate_fn(**simulate_kwargs, seed=seed)
    A_true = simulate_kwargs["A"]

    # 2. Build model args from simulation result
    model_args = model_args_fn(sim_result)

    # 3. Run SVI
    guide = create_guide(model_fn, init_scale=0.01)
    svi_result = run_svi(
        model_fn, guide, model_args,
        num_steps=num_steps, lr=lr, clip_norm=10.0,
    )

    # 4. Extract posterior
    posterior = extract_posterior_params(guide, model_args)

    # 5. Get A_inferred (median)
    A_free_median = posterior["median"]["A_free"]
    A_inferred = parameterize_A(A_free_median)

    # 6. Get 95% credible intervals via quantiles
    quantiles = guide.quantiles([0.025, 0.975], *model_args)
    A_free_lo = quantiles["A_free"][0]
    A_free_hi = quantiles["A_free"][1]
    # Note: CIs are on A_free, not parameterized A.
    # For off-diagonal: A_lo = A_free_lo, A_hi = A_free_hi
    # For diagonal: A_lo = -exp(A_free_hi)/2, A_hi = -exp(A_free_lo)/2
    # (monotone decreasing transform flips bounds)

    return {
        "A_true": A_true,
        "A_inferred": A_inferred,
        "A_free_lo": A_free_lo,
        "A_free_hi": A_free_hi,
        "losses": svi_result["losses"],
        "final_loss": svi_result["final_loss"],
    }
```

### Pattern 2: Coverage Computation

**What:** For each parameter element across N datasets, check whether the true value falls within the 95% credible interval. Coverage = fraction of datasets where true value is covered.

**When to use:** All three variants.

**Quantitative definition of "calibrated":** A 95% CI is calibrated if empirical coverage is in [0.90, 0.99]. This range accounts for:
- Finite sample variability (N=50 datasets: standard error of coverage ~ sqrt(0.95*0.05/50) ~ 0.03)
- Mean-field VI's known tendency to underestimate posterior variance (coverage may be slightly below 0.95)
- The range [0.90, 0.99] is the roadmap threshold, standard for VI validation

**Example:**
```python
def compute_coverage(results_list, element_mask=None):
    """Compute 95% CI coverage across recovery trials.

    Parameters
    ----------
    results_list : list of dict
        Each dict has 'A_true', 'A_free_lo', 'A_free_hi'.
    element_mask : torch.Tensor or None
        Which A elements to include (e.g., off-diagonal only).

    Returns
    -------
    float
        Fraction of (dataset, element) pairs where true value
        falls within [lo, hi].
    """
    covered = 0
    total = 0
    for r in results_list:
        A_true = r["A_true"]
        # For off-diagonal elements: CI is directly on A values
        # For diagonal: transform CI bounds
        lo = r["A_free_lo"]  # or transformed bounds
        hi = r["A_free_hi"]
        in_ci = (A_true >= lo) & (A_true <= hi)
        if element_mask is not None:
            in_ci = in_ci[element_mask]
        covered += in_ci.sum().item()
        total += in_ci.numel()
    return covered / total
```

### Pattern 3: rDCM Coverage from Analytic VB

**What:** rDCM's `rigid_inversion` returns per-region posterior mean (`mu_per_region`) and covariance (`Sigma_per_region`). Extract diagonal of covariance as variance, compute 95% CI as `mu +/- 1.96 * sqrt(diag(Sigma))`.

**When to use:** rDCM recovery tests (primary method).

**Critical detail:** The posterior parameters are per-region vectors that include A connections, C connections, and confound weights, in the order determined by `a_mask` and `c_mask`. To map back to the A matrix, use the same column-selection logic as in `rigid_inversion`.

**Example:**
```python
def extract_rdcm_credible_intervals(inv_result, a_mask, c_mask, nc=1):
    """Extract 95% CIs for A from rDCM analytic VB posterior.

    Returns
    -------
    A_mu : (nr, nr) posterior mean
    A_lo : (nr, nr) lower 95% CI bound
    A_hi : (nr, nr) upper 95% CI bound
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    A_mu = inv_result["A_mu"]
    A_lo = torch.zeros_like(A_mu)
    A_hi = torch.zeros_like(A_mu)

    for r in range(nr):
        mu_r = inv_result["mu_per_region"][r]
        Sigma_r = inv_result["Sigma_per_region"][r]
        std_r = torch.sqrt(torch.diag(Sigma_r))

        # Map back to A positions
        pos = 0
        for j in range(nr):
            if a_mask[r, j] > 0:
                A_lo[r, j] = mu_r[pos] - 1.96 * std_r[pos]
                A_hi[r, j] = mu_r[pos] + 1.96 * std_r[pos]
                pos += 1
        # Skip C and confound positions

    return A_mu, A_lo, A_hi
```

### Pattern 4: ELBO Model Comparison

**What:** Run SVI on the same data with two different models (correctly specified vs misspecified), compare final ELBO values. The correctly specified model should have lower loss (= higher ELBO, since loss = -ELBO).

**When to use:** REC-04 requirement.

**Misspecified model construction:** For each variant, create a misspecified model by using the wrong connectivity mask (e.g., removing a true connection or adding a spurious one). Alternatively, use a model with the wrong number of regions or wrong noise model.

**Example:**
```python
def test_elbo_correctly_specified_wins():
    """Correctly specified model achieves lower loss than misspecified."""
    # Generate data from 3-region model with specific A mask
    A_true = make_random_stable_A(3, seed=42)
    true_mask = (A_true != 0).float()

    sim_result = simulate_task_dcm(A_true, C, stim, ...)

    # Correctly specified: use true mask
    guide_correct = create_guide(task_dcm_model)
    result_correct = run_svi(
        task_dcm_model, guide_correct,
        (bold, stim, true_mask, c_mask, t_eval, TR, dt),
        num_steps=3000,
    )

    # Misspecified: use wrong mask (e.g., remove a connection)
    wrong_mask = true_mask.clone()
    wrong_mask[0, 1] = 0.0  # remove a true connection
    guide_wrong = create_guide(task_dcm_model)
    result_wrong = run_svi(
        task_dcm_model, guide_wrong,
        (bold, stim, wrong_mask, c_mask, t_eval, TR, dt),
        num_steps=3000,
    )

    # Correctly specified should have lower loss (higher ELBO)
    assert result_correct["final_loss"] < result_wrong["final_loss"]
```

### Anti-Patterns to Avoid

- **Testing on A_free instead of parameterized A:** The RMSE criterion is on the parameterized A matrix (with negative diagonal), not on A_free. Always apply `parameterize_A` before computing RMSE.
- **Computing CI on parameterized A directly from A_free CI:** The diagonal transform `a_ii = -exp(A_free_ii)/2` is monotone decreasing, so CI bounds must be flipped for diagonal elements. Off-diagonal elements are identity-transformed, so CI bounds pass through directly.
- **Running 50 datasets in CI:** This will be too slow for task-DCM SVI (50 * 3000 steps * ODE integration). Use 10-20 datasets for CI tests, mark 50-dataset tests as `@pytest.mark.slow`.
- **Ignoring NaN ELBO:** `run_svi` already raises `RuntimeError` on NaN, but recovery tests should catch and count failures rather than crashing on a single bad seed.
- **Using correlation alone without RMSE:** High correlation does not guarantee low RMSE (could be scaled/offset). Both metrics are needed.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Posterior quantiles from SVI | Manual loc +/- z*scale with transform logic | `guide.quantiles([0.025, 0.975])` | Handles all transforms correctly, tested by Pyro |
| Pearson correlation | numpy `corrcoef` | Torch-native `_pearson_corr` helper (already in `test_rdcm_simulator.py`) | Avoids numpy process abort on Windows (see existing helper) |
| Random stable A matrices | Custom generation | `make_random_stable_A`, `make_stable_A_spectral`, `make_stable_A_rdcm` | Already exist, already tested |
| Block stimulus | Custom generation | `make_block_stimulus`, `make_block_stimulus_rdcm` | Already exist with seed control |
| SVI training loop | Custom loop | `run_svi` with `create_guide` | Already handles ClippedAdam, LR decay, NaN detection, gradient clipping |

**Key insight:** Phase 5 is primarily a test/validation phase. Almost all infrastructure already exists from Phases 1-4. The new code is test orchestration, metric computation, and result aggregation -- not new inference or simulation code.

## Common Pitfalls

### Pitfall 1: Mean-Field VI Underestimates Posterior Variance
**What goes wrong:** AutoNormal (mean-field) assumes independence between parameters. The true posterior for DCM has correlations (especially between A elements in the same row). This causes credible intervals to be too narrow, leading to coverage below the nominal 95%.
**Why it happens:** Mean-field approximation ignores posterior correlations.
**How to avoid:** The [0.90, 0.99] coverage range already accounts for this. If coverage is below 0.90, increase SVI steps (more convergence = better calibration), increase data quality (higher SNR), or use larger networks (more data per parameter). Do NOT widen the acceptance range below 0.90.
**Warning signs:** Coverage consistently below 0.85 across multiple seeds.

### Pitfall 2: SVI Non-Convergence for Task DCM
**What goes wrong:** Task DCM involves ODE integration inside the SVI loop. If the A matrix sampled during SVI has large positive eigenvalues, the ODE can blow up, causing NaN losses.
**Why it happens:** Early SVI steps explore broad regions of parameter space; some produce unstable A matrices.
**How to avoid:** Use `init_scale=0.01` for the guide (already the default in `create_guide`). Use `clip_norm=10.0` (already default). Use `lr=0.005` or `lr=0.01` with `lr_decay_factor=0.01`. If still failing, reduce `dt` from 0.5 to 0.25 for coarser time grid (fewer ODE steps per SVI step).
**Warning signs:** RuntimeError("NaN ELBO") in `run_svi`.

### Pitfall 3: Confusing A_free with Parameterized A
**What goes wrong:** Computing RMSE on A_free instead of parameterized A, or computing coverage on the wrong space.
**Why it happens:** The Pyro model samples A_free, and `extract_posterior_params` returns A_free medians. The `parameterize_A` transform must be applied before comparing to ground truth.
**How to avoid:** Always apply `parameterize_A(A_free_median)` to get A_inferred. For coverage, either (a) work in A_free space and transform the ground truth A_true to A_free_true (inverting parameterize_A), or (b) transform the CI bounds to A space (flipping bounds for diagonal).
**Warning signs:** RMSE values that are way off, or diagonal coverage near zero.

### Pitfall 4: Slow Test Runtime in CI
**What goes wrong:** 50 datasets * 3 variants * 3000-5000 SVI steps = hours of compute, unsuitable for CI.
**Why it happens:** ODE-based task DCM is expensive per SVI step.
**How to avoid:** Use two tiers: (1) `@pytest.mark.slow` for full 50-dataset validation with 5000 steps, (2) default (no marker) for fast CI tests with 10 datasets, 2000 steps, small networks (3 regions). Structure tests so that `pytest -m "not slow"` runs in < 5 minutes.
**Warning signs:** CI pipeline taking > 10 minutes.

### Pitfall 5: ELBO Comparison Confounded by Guide Initialization
**What goes wrong:** Two models compared via ELBO, but different random seeds for guide initialization produce different results, making the comparison noisy.
**Why it happens:** SVI is stochastic; different guide initializations can converge to different local optima.
**How to avoid:** Use `pyro.set_rng_seed()` before each SVI run for deterministic comparison. Run both models with the same seed. Alternatively, run multiple seeds and compare average final ELBO.
**Warning signs:** Misspecified model occasionally beating correctly specified model.

### Pitfall 6: rDCM Sparse Inversion is Slow with Many Reruns
**What goes wrong:** `sparse_inversion` with `n_reruns=100` (default) is slow for 50 datasets.
**Why it happens:** Each rerun is a full VB iteration loop.
**How to avoid:** For CI tests, use `n_reruns=10`. For validation, use `n_reruns=20-50`. The existing test (`test_rdcm_simulator.py`) uses `n_reruns=20` as a reasonable tradeoff.
**Warning signs:** rDCM tests taking > 2 minutes for a single dataset.

## Code Examples

### Computing RMSE for A Matrix
```python
def compute_rmse_A(A_true: torch.Tensor, A_inferred: torch.Tensor) -> float:
    """RMSE between true and inferred A matrices."""
    return torch.sqrt(torch.mean((A_true - A_inferred) ** 2)).item()
```

### Computing Pearson Correlation (Existing Helper)
```python
# Already exists in tests/test_rdcm_simulator.py as _pearson_corr
def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1D tensors."""
    x_mean = x.mean()
    y_mean = y.mean()
    xd = x - x_mean
    yd = y - y_mean
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()
```

### Extracting 95% CI from AutoNormal Guide
```python
# After run_svi completes:
quantile_dict = guide.quantiles([0.025, 0.975], *model_args)
# quantile_dict["A_free"] has shape (2, N, N)
# quantile_dict["A_free"][0] = 2.5th percentile
# quantile_dict["A_free"][1] = 97.5th percentile
```

### Extracting 95% CI from rDCM Analytic VB
```python
# After rigid_inversion or sparse_inversion:
mu_r = result["mu_per_region"][r]          # (D_r,)
Sigma_r = result["Sigma_per_region"][r]    # (D_r, D_r)
std_r = torch.sqrt(torch.diag(Sigma_r))   # (D_r,)
lo_r = mu_r - 1.96 * std_r
hi_r = mu_r + 1.96 * std_r
```

### SVI Settings for Recovery (Recommended)
```python
# Task DCM: ODE-based, needs careful settings
TASK_SVI_SETTINGS = {
    "num_steps": 3000,      # 3000 for CI, 5000 for validation
    "lr": 0.005,            # Conservative for ODE stability
    "clip_norm": 10.0,
    "lr_decay_factor": 0.01,
    "num_particles": 1,     # 1 for speed, 4-8 for validation
}

# Spectral DCM: No ODE, faster convergence
SPECTRAL_SVI_SETTINGS = {
    "num_steps": 2000,      # 2000 for CI, 3000 for validation
    "lr": 0.01,             # Can be more aggressive
    "clip_norm": 10.0,
    "lr_decay_factor": 0.01,
    "num_particles": 1,
}

# rDCM via SVI (comparison only, not primary)
RDCM_SVI_SETTINGS = {
    "num_steps": 1000,      # Fast convergence (linear regression)
    "lr": 0.01,
    "clip_norm": 10.0,
    "lr_decay_factor": 0.01,
    "num_particles": 1,
}
```

### Network Configurations for Recovery Tests
```python
# 3-region (fast, for CI)
SMALL_NETWORK = {
    "n_regions": 3,
    "density": 0.5,
    "strength_range": (0.05, 0.25),
    "self_inhibition": 0.5,
}

# 5-region (slower, for validation)
MEDIUM_NETWORK = {
    "n_regions": 5,
    "density": 0.4,
    "strength_range": (0.05, 0.20),
    "self_inhibition": 0.5,
}
```

### Simulation Settings for Recovery Tests
```python
# Task DCM simulation
TASK_SIM_SETTINGS = {
    "duration": 300.0,   # 300s = 150 TRs at TR=2.0
    "dt": 0.01,          # Fine integration for simulation
    "TR": 2.0,
    "SNR": 5.0,          # Moderate noise
    "solver": "dopri5",
}

# But for model evaluation during SVI, use coarser grid:
TASK_MODEL_SETTINGS = {
    "dt": 0.5,           # Coarse integration for SVI speed
    "TR": 2.0,
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cook-Gelman-Rubin (2006) SBC | Talts et al. (2018) SBC with rank uniformity | 2018 | More rigorous calibration checking |
| Full SBC (hundreds of rank tests) | Coverage probability as simplified check | Pragmatic | For VI, coverage is more practical than full rank SBC since we have parametric posteriors |
| Single-dataset recovery | Multi-dataset (50+) with aggregate metrics | Standard practice | Controls for lucky/unlucky seeds |
| RMSE only | RMSE + coverage + correlation | Standard practice | RMSE alone misses calibration; correlation alone misses bias |

## Open Questions

### 1. Exact Coverage Computation for Diagonal Elements
**What we know:** Off-diagonal A elements have identity transform (A_ij = A_free_ij), so `guide.quantiles()` gives CI directly. Diagonal elements have the nonlinear transform `a_ii = -exp(A_free_ii)/2`.
**What's unclear:** Whether `guide.quantiles()` returns quantiles in the *unconstrained* (A_free) space or the *constrained* (A) space. If unconstrained, diagonal CI bounds need manual transformation with bound flipping.
**Recommendation:** Test empirically with a known case. If `guide.quantiles()` returns A_free quantiles, transform diagonal bounds manually: `A_lo_diag = -exp(A_free_hi_diag)/2`, `A_hi_diag = -exp(A_free_lo_diag)/2` (swap because transform is decreasing). Alternatively, compute coverage in A_free space by inverting the true A diagonal: `A_free_true_diag = log(-2 * A_true_diag)`.

### 2. Spectral DCM: CSD Noise for Recovery
**What we know:** `simulate_spectral_dcm` generates clean (noiseless) CSD from the forward model. The spectral DCM Pyro model has a `csd_noise_scale` parameter (HalfCauchy prior).
**What's unclear:** For recovery testing, should we add noise to the simulated CSD before inference? Without noise, the model may overfit trivially.
**Recommendation:** Add Gaussian noise to the decomposed real/imaginary CSD vector before inference, scaled to a target SNR. This makes the recovery test realistic and tests the noise model.

### 3. How Many Datasets for CI vs Validation
**What we know:** The roadmap says 50 datasets. Task DCM SVI with 3000 steps and ODE integration is slow (~30-60s per dataset on CPU). 50 datasets = 25-50 minutes.
**What's unclear:** Whether CI should run all 50 or a subset.
**Recommendation:** CI tests (`pytest` default): 10 datasets, 3-region only, 2000-3000 SVI steps. Validation (`@pytest.mark.slow`): 50 datasets, 3-region and 5-region, 5000 SVI steps. This keeps CI under 5-10 minutes while providing rigorous validation on demand.

### 4. ELBO as Model Comparison Metric
**What we know:** The ELBO is a lower bound on log marginal likelihood. For the same data and same guide family, higher ELBO (= lower SVI loss) indicates better model fit. This is theoretically justified for model comparison (Cherief-Abdellatif & Alquier, 2018).
**What's unclear:** With mean-field VI, the tightness of the ELBO bound varies between models. A model with more parameters may have a looser bound, confounding comparison.
**Recommendation:** Keep comparison simple: same guide family (AutoNormal), same number of SVI steps, same learning rate. Compare correctly specified vs misspecified (wrong mask) on same data. This is the standard DCM approach (Bayesian model comparison via free energy / ELBO).

## Sources

### Primary (HIGH confidence)
- Pyro AutoNormal source code (GitHub pyro-ppl/pyro): `quantiles()` method returns dict of quantile values; `median()` returns dict of median values. Parameters stored as `locs.<site_name>` and `scales.<site_name>`. `init_scale` default is 0.1 but project uses 0.01.
- Existing codebase: `src/pyro_dcm/models/guides.py` -- `create_guide`, `run_svi`, `extract_posterior_params` all verified.
- Existing codebase: `src/pyro_dcm/simulators/` -- all three simulators verified with seed control and comprehensive output dicts.
- Existing codebase: `src/pyro_dcm/forward_models/rdcm_posterior.py` -- `rigid_inversion` and `sparse_inversion` return `mu_per_region`, `Sigma_per_region`, `F_per_region`, `F_total`.
- Existing tests: `tests/test_rdcm_simulator.py` -- recovery pattern with correlation > 0.8 and F1 > 0.85 already established; includes `_pearson_corr` helper.
- Existing tests: `tests/test_svi_integration.py` -- SVI integration patterns established for all three variants.

### Secondary (MEDIUM confidence)
- [Pyro SVI Part IV](https://pyro.ai/examples/svi_part_iv.html) -- Tips on learning rate, init_scale, num_particles, ELBO monitoring. Verified via WebFetch.
- [Stan User's Guide: SBC](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html) -- SBC algorithm with rank computation and chi-squared uniformity test. Verified via WebFetch.
- [Cherief-Abdellatif & Alquier (2018)](https://arxiv.org/abs/1810.11859) -- ELBO consistency for model selection, robust to misspecification.

### Tertiary (LOW confidence)
- General VI coverage literature -- Mean-field VI typically underestimates posterior variance, leading to below-nominal coverage. This is well-known but hard to quantify precisely for DCM models specifically.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, APIs verified from source code
- Architecture: HIGH -- recovery test patterns well-established in existing test suite, code examples verified against codebase
- Pitfalls: HIGH -- based on direct codebase analysis (ODE stability, A_free transform, runtime estimates)
- Coverage methodology: MEDIUM -- standard Bayesian practice, but `guide.quantiles()` behavior for transformed parameters needs empirical verification
- SVI settings: MEDIUM -- based on Pyro docs and existing test patterns, but optimal settings are model-dependent

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable domain; recovery methodology does not change rapidly)
