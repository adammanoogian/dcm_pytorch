# Phase 11 Research: Calibration Analysis

**Researched:** 2026-04-12
**Domain:** Calibration benchmarking, publication figure generation, combinatorial benchmark orchestration
**Confidence:** HIGH (all infrastructure exists in codebase; Phase 11 is analysis on existing foundations)

---

## Executive Summary

Phase 11 is the data-generation and analysis phase of v0.2.0. All infrastructure is in
place: 6 guide types in `create_guide`, 3 ELBO objectives in `run_svi`, 4 SVI runners
plumbed with `guide_type`/`elbo_type` parameters, shared fixture generation, and
Predictive-based posterior extraction returning per-site samples. Phase 11 must (1)
orchestrate the combinatorial benchmark sweep (6 guides x 3 ELBOs x 3 variants x 3
sizes), (2) compute multi-level coverage calibration curves, (3) produce publication
figures (calibration curves, violin/ridge plots, Pareto frontiers), and (4) generate
a structured comparison table. The main technical challenges are combinatorial explosion
management, wall-clock timing decomposition, and multi-level per-parameter coverage
computation -- all tractable extensions of existing code.

The existing runners produce JSON results with `rmse_list`, `coverage_list`,
`correlation_list`, `time_list`, and per-element `a_true_list`/`a_inferred_list`. Phase
11 extends this in three directions: (a) coverage at 4 nominal levels (50%, 75%, 90%,
95%) instead of just 95%, (b) per-parameter breakdown (diagonal A vs off-diagonal A),
and (c) posterior samples stored for violin/ridge plots. The `compute_coverage_from_ci`
function in `benchmarks/metrics.py` already computes element-wise coverage from CI
bounds; the `compute_coverage_from_samples` function handles sample-based coverage at
arbitrary CI levels. The extension to multi-level calibration curves is mechanical.

No new external dependencies are needed. All plotting uses matplotlib (already
installed) with the existing `_apply_style()` pattern from `benchmarks/plotting.py`.
Seaborn is useful for violin plots but can be optional (matplotlib's `violinplot` works
as fallback).

---

## 1. Existing Infrastructure Inventory

### 1.1 What Already Exists (from Phases 8-10)

| Component | Location | Status |
|-----------|----------|--------|
| BenchmarkConfig with guide_type/elbo_type | `benchmarks/config.py` | Complete |
| 6 guide types in create_guide factory | `src/pyro_dcm/models/guides.py` | Complete |
| 3 ELBO objectives in run_svi | `src/pyro_dcm/models/guides.py` | Complete |
| GUIDE_REGISTRY / ELBO_REGISTRY | `src/pyro_dcm/models/guides.py` | Complete |
| Mean-field guard (TraceMeanField_ELBO + non-MF -> ValueError) | `run_svi()` | Complete |
| AutoMVN blocklist at N>7 | `create_guide()` | Complete |
| Predictive-based extract_posterior_params | `guides.py` | Complete |
| 4 SVI runners with guide_type/elbo_type | `benchmarks/runners/` | Complete |
| rDCM VB runners (rigid + sparse) | `benchmarks/runners/rdcm_vb.py` | Complete |
| SPM reference loader | `benchmarks/runners/spm_reference.py` | Complete |
| Fixture generation (3 variants x 3 sizes) | `benchmarks/generate_fixtures.py` | Script complete, not yet run |
| Fixture loading helpers | `benchmarks/fixtures.py` | Complete |
| Consolidated metrics (RMSE, coverage, Pearson, amort gap) | `benchmarks/metrics.py` | Complete |
| Existing plotting (scatter, metric strips, amort gap) | `benchmarks/plotting.py` | Complete |
| JSON results format | `benchmarks/results/` | Complete |
| run_all_benchmarks.py CLI | `benchmarks/run_all_benchmarks.py` | Complete |

### 1.2 What Phase 11 Must Add

| Component | Purpose | Complexity |
|-----------|---------|------------|
| Multi-level coverage computation | CAL-01: coverage at 50%/75%/90%/95% | Low |
| Per-parameter coverage breakdown | CAL-01: diagonal A vs off-diagonal A | Low |
| Calibration curve plotting | CAL-01: expected vs observed coverage figures | Medium |
| Tiered benchmark orchestrator | CAL-02: manage combinatorial sweep | Medium |
| Cross-method comparison table generator | CAL-03: structured table output | Low |
| Violin/ridge posterior overlay plots | CAL-04: per-A_ij element comparison | Medium |
| Wall-clock timing decomposition | CAL-05: forward/guide/gradient breakdown | Medium-High |
| Pareto frontier plotting | CAL-05: wall-time vs RMSE scatter | Low |
| Extended JSON results schema | Store multi-level coverage + samples | Low |

### 1.3 Critical Gap: Fixtures Not Yet Generated

The `generate_fixtures.py` script exists and is tested, but no `benchmarks/fixtures/`
directory exists on disk. Before any Phase 11 benchmarks can run, fixtures must be
generated:

```bash
python benchmarks/generate_fixtures.py --variant all --n-regions 3,5,10 \
    --n-datasets 50 --seed 42 --output-dir benchmarks/fixtures
```

This generates 9 subdirectories (3 variants x 3 sizes) x 50 datasets each = 450 .npz
files. This is a prerequisite for all Phase 11 work.

---

## 2. Combinatorial Explosion Management

### 2.1 The Full Matrix

The requirements specify: 6 guide types x 3 ELBO objectives x 3 DCM variants x 3
network sizes. Naive enumeration: 6 x 3 x 3 x 3 = 162 configurations. With 50
datasets each at ~5-30 seconds per run, the full sweep would take 10-80+ hours.

However, many cells are invalid or redundant:
- TraceMeanField_ELBO only works with auto_delta and auto_normal (2 of 6 guides)
- auto_mvn is blocked at N>7 (only runs at N=3, 5)
- rDCM uses analytic VB (no guide type or ELBO choice)
- Amortized runners have fixed guide (flow-based, not from create_guide factory)
- auto_delta gives point estimates (coverage is always 0 or 1 -- informative but fast)

### 2.2 Recommended Tiering Strategy

**Tier 1: Core Comparison (Fast, ~2 hours)**
All guides x trace_elbo x spectral DCM x N=3

Purpose: The "main result" comparing guide families on the easiest problem.
Configurations: 6 guide types x 1 ELBO x 1 variant x 1 size = 6 runs x 50 datasets.
Time estimate: 50 x ~7s x 6 = ~35 minutes (spectral SVI is fast).

**Tier 2: ELBO Effect (Medium, ~3 hours)**
Mean-field guides x 3 ELBOs x spectral DCM x N=3,5

Purpose: Isolate the effect of ELBO choice (only meaningful for mean-field guides).
Configurations: 2 guides x 3 ELBOs x 1 variant x 2 sizes = 12 runs x 50 datasets.
Note: TraceMeanField_ELBO only pairs with auto_delta/auto_normal. So this tests all
valid ELBO combinations for those guides.

**Tier 3: Scaling Study (Medium-Long, ~4 hours)**
All guides x trace_elbo x all variants x N=3,5 + mean-field/rDCM at N=10

Purpose: Network size scaling across all DCM variants.
Configurations (N=3,5):
- 6 guides x 1 ELBO x 2 SVI variants (task, spectral) x 2 sizes = 24 runs
- rDCM rigid + sparse x 2 sizes = 4 runs
Configurations (N=10):
- auto_normal + auto_delta x 1 ELBO x 2 SVI variants = 4 runs
- rDCM rigid + sparse x 1 size = 2 runs

Total: ~34 runs x 50 datasets.

**Tier 4: Full Sweep (Optional, for paper)**
All valid combinations from the remaining cells not in Tiers 1-3.
Purpose: Complete the cross-method table.

### 2.3 Valid Configuration Matrix

After accounting for constraints:

| Guide Type | trace_elbo | tracemeanfield_elbo | renyi_elbo | N=3 | N=5 | N=10 |
|------------|-----------|--------------------|-----------|----|----|----|
| auto_delta | YES | YES | YES | YES | YES | YES |
| auto_normal | YES | YES | YES | YES | YES | YES |
| auto_lowrank_mvn | YES | REJECTED | YES | YES | YES | YES |
| auto_mvn | YES | REJECTED | YES | YES | YES | BLOCKED |
| auto_iaf | YES | REJECTED | YES | YES | YES | YES* |
| auto_laplace | YES | REJECTED | YES | YES | YES | YES* |

*auto_iaf and auto_laplace at N=10 are technically allowed but may be very slow.
Success criteria say N=10 is "mean-field and rDCM only" -- auto_delta, auto_normal,
rDCM_rigid, rDCM_sparse. Recommend honoring this constraint.

For rDCM: guide_type and elbo_type are irrelevant (analytic VB). rDCM runs at all 3
sizes independently of the guide/ELBO matrix.

**Total valid SVI configurations (excluding rDCM and amortized):**
- At N=3,5: 6 guides x 2 valid ELBOs-per-guide (trace_elbo + {tracemeanfield or renyi}) = ~14 per variant per size
- At N=10: 2 guides x 3 ELBOs = 6 per variant

### 2.4 Orchestration Design

The current `run_all_benchmarks.py` runs a flat list of (variant, method) pairs with a
single guide_type and elbo_type. Phase 11 needs to either:

**Option A: Wrapper script that calls run_all_benchmarks.py in a loop.**
- Pro: No changes to existing CLI.
- Con: Each call re-initializes, no shared state.

**Option B: New calibration_sweep.py script with nested loops.**
- Pro: Manages tiering, result aggregation, and resume-on-failure.
- Con: New script, but it can reuse runner functions directly.

**Recommendation: Option B.** Create `benchmarks/calibration_sweep.py` that imports
runner functions from `RUNNER_REGISTRY`, iterates over the tiered configuration matrix,
and writes results to a structured JSON keyed by `{variant}_{guide_type}_{elbo_type}_{n_regions}`.
The script should support `--tier 1|2|3|all` and `--resume` (skip existing result keys).

---

## 3. Multi-Level Coverage Calibration (CAL-01)

### 3.1 What "Calibration Curve" Means Here

A calibration curve plots **nominal coverage level** (x-axis: 50%, 75%, 90%, 95%) vs
**observed empirical coverage** (y-axis: fraction of true values inside the CI). A
perfectly calibrated method lies on the y=x diagonal. Mean-field methods are expected
to fall below the diagonal (over-confident / too-narrow CIs).

This is NOT prediction calibration (Platt scaling, etc.) -- it is posterior credible
interval calibration in the simulation-based calibration (SBC) tradition.

### 3.2 Implementation

The existing `compute_coverage_from_samples` in `benchmarks/metrics.py` already accepts
`ci_level` as a parameter and has a z-score lookup table for 0.90, 0.95, 0.99. For
calibration curves, we need coverage at 0.50, 0.75, 0.90, 0.95.

**Required z-scores:**

| CI Level | z-score |
|----------|---------|
| 0.50 | 0.6745 |
| 0.75 | 1.1503 |
| 0.90 | 1.6449 |
| 0.95 | 1.9600 |

The z_table in `compute_coverage_from_samples` needs to be extended with 0.50 and 0.75
entries. This is a 2-line change.

**Alternative: Quantile-based coverage (preferred for non-Gaussian guides)**

For guides like AutoIAF that produce non-Gaussian posteriors, z-score-based CIs are
inappropriate. Better to use empirical quantiles from posterior samples:

```python
def compute_coverage_multi_level(
    true_vals: torch.Tensor,
    samples: torch.Tensor,
    ci_levels: list[float] = [0.50, 0.75, 0.90, 0.95],
) -> dict[float, float]:
    """Compute coverage at multiple nominal CI levels.

    Uses empirical quantiles from samples (not z-scores) for
    accuracy with non-Gaussian posteriors.
    """
    result = {}
    for level in ci_levels:
        alpha = (1.0 - level) / 2.0
        lo = torch.quantile(samples.float(), alpha, dim=0)
        hi = torch.quantile(samples.float(), 1.0 - alpha, dim=0)
        in_ci = (true_vals >= lo) & (true_vals <= hi)
        result[level] = in_ci.float().mean().item()
    return result
```

This is more general and correct for all guide types.

### 3.3 Per-Parameter Breakdown

The success criteria say "per-parameter coverage calibration curves." This means
computing coverage separately for:

1. **Diagonal A elements** (a_ii) -- self-connections, always negative via
   parameterization
2. **Off-diagonal A elements** (a_ij, i!=j) -- inter-region connections, can be
   positive or negative
3. **Hemodynamic parameters** (C, noise_prec for task; noise_a, noise_b, noise_c for
   spectral) -- if included in posterior extraction

For the A matrix, the breakdown is straightforward:

```python
diag_mask = torch.eye(N, dtype=torch.bool)
offdiag_mask = ~diag_mask

coverage_diag = compute_coverage_multi_level(
    A_true[diag_mask], samples[:, diag_mask],
)
coverage_offdiag = compute_coverage_multi_level(
    A_true[offdiag_mask], samples[:, offdiag_mask],
)
```

**Important:** Runners currently only extract and track A_free/A metrics. The full
posterior from `extract_posterior_params` includes all sites (C, noise_prec, etc.), but
runners only compute RMSE/coverage/correlation on A. For Phase 11, we should at minimum
report A diagonal vs A off-diagonal. Extending to non-A parameters is desirable but
lower priority -- the connectivity matrix is the scientific focus.

### 3.4 Calibration Curve Figure Specification

Each figure shows one guide type (or overlays multiple guides on one plot):

- **X-axis:** Nominal CI level (0.50, 0.75, 0.90, 0.95)
- **Y-axis:** Observed coverage (empirical fraction)
- **Reference line:** y = x diagonal (perfect calibration)
- **Traces:** One line per guide type, with confidence band (IQR across datasets)
- **Subplots:** One per DCM variant (task, spectral, rDCM) -- never aggregate across
  variants (P9 risk: Simpson's paradox)
- **Separate panels or colors for:** diagonal A vs off-diagonal A

**Figure layout recommendation:**
- 3x1 grid: rows = DCM variants (task, spectral, rDCM)
- Each subplot: 6 colored lines (one per guide type)
- All lines at N=3 on one figure, separate figure for N=5, N=10
- Total: 3 figures (one per network size) with 3 subplots each

---

## 4. Cross-Method Comparison Table (CAL-03)

### 4.1 Table Schema

The success criteria require: "6+ methods x 3 variants x 3 sizes reporting RMSE,
coverage@90%, Pearson correlation, and wall time."

**Method rows (8 methods total):**

| Method | variant=task | variant=spectral | variant=rDCM |
|--------|-------------|-----------------|-------------|
| AutoDelta (SVI, trace_elbo) | Yes | Yes | N/A |
| AutoNormal (SVI, trace_elbo) | Yes | Yes | N/A |
| AutoLowRankMVN (SVI, trace_elbo) | Yes | Yes | N/A |
| AutoMVN (SVI, trace_elbo) | Yes (N<=7) | Yes (N<=7) | N/A |
| AutoIAF (SVI, trace_elbo) | Yes | Yes | N/A |
| AutoLaplace (SVI, trace_elbo) | Yes | Yes | N/A |
| rDCM rigid (VB) | N/A | N/A | Yes |
| rDCM sparse (VB) | N/A | N/A | Yes |

Amortized runners could be included as additional rows if pre-trained guides exist.

**Column structure per cell:**
- RMSE: median (IQR) -- not just mean
- Coverage@90%: median (IQR) -- NOT "mean coverage"
- Pearson r: median (IQR)
- Wall time (s): median (IQR)

**Key constraint from STATE.md:** "Never aggregate across DCM variants." Each variant
gets its own table section.

### 4.2 Output Format

Generate three output formats:
1. **JSON** -- Machine-readable, for downstream analysis
2. **Markdown** -- For README/docs, rendered in GitHub
3. **LaTeX** -- For paper supplementary material

The JSON is primary (used by plotting code). Markdown and LaTeX are generated from JSON.

### 4.3 Median + IQR Reporting

STATE.md risk P12 says "report median+IQR, not just means." Current runners compute
`mean_rmse`, `std_rmse`, etc. Phase 11 analysis must compute:

```python
median = np.median(rmse_list)
q25 = np.percentile(rmse_list, 25)
q75 = np.percentile(rmse_list, 75)
iqr_str = f"{median:.4f} ({q25:.4f}-{q75:.4f})"
```

This is a post-processing step on the already-collected per-dataset lists.

---

## 5. Per-Parameter Posterior Comparison Plots (CAL-04)

### 5.1 What "Violin/Ridge Overlay" Means

For a representative dataset, show the posterior distribution of each A_ij element
across all methods. Each method contributes one violin (or ridge line). Ground truth is
marked with a vertical line or diamond marker.

For a 3-region network: 9 A_ij elements x 6+ methods = 9 subplots, each with 6+
violins.

### 5.2 Data Requirements

To produce violin plots, we need the full posterior samples (not just mean/std). The
Predictive-based `extract_posterior_params` already returns `samples` of shape
`(num_samples, N, N)` for A_free. The runners currently only store
`a_true_list`/`a_inferred_list` (means), not samples.

**Required change to runners:** For at least one "representative dataset" (e.g., dataset
index 0), store the full `A_free_samples` tensor (or a subsample of 500-1000 draws) in
the results JSON.

Alternatively, produce the violin plots at analysis time by re-running inference on a
single dataset with all guide types and collecting the raw samples. This avoids bloating
the JSON results.

**Recommendation: Re-run a single representative dataset at analysis time.** The violin
plot needs posterior samples from all methods on the SAME dataset. Running 6 guide types
x 1 dataset x 500 SVI steps takes ~1-2 minutes total. This is cleaner than storing
megabytes of samples in JSON.

### 5.3 Figure Layout

For N=3 (9 A_ij elements):

```
3x3 grid of subplots, one per A_ij element
  Each subplot: 6 violins (one per guide type)
  Horizontal line: ground truth A_ij value
  X-axis: method names (rotated)
  Y-axis: posterior value
  Title: A[i,j] (or A_free[i,j])
```

For the parameterized A (after applying the parameterize_A transform), diagonal elements
are always negative. The transform must be applied to posterior samples before plotting:

```python
# For each sample s in A_free_samples:
A_samples = parameterize_A(A_free_samples)  # apply transform element-wise
```

Actually, `parameterize_A` takes a full (N,N) tensor and applies different transforms to
diagonal vs off-diagonal. Need to verify it works on batched input or loop over samples.

### 5.4 Violin vs Ridge

**Violin plots (recommended):** Side-by-side comparison is natural. Each violin shows
the full posterior density. Seaborn's `violinplot` handles this well. Matplotlib's
built-in `violinplot` works but is less polished.

**Ridge plots:** Better when there are many methods (>8) or when distributions are
multimodal. More compact vertically.

**Recommendation:** Use violin plots for the main figure (6-8 methods is manageable).
If amortized methods are added (8+ methods), switch to ridge.

---

## 6. Wall-Clock Timing Decomposition (CAL-05)

### 6.1 The Challenge

The success criteria require "wall-clock timing breakdown (forward model, guide
evaluation, gradient)" per SVI step. Pyro's `svi.step()` is a single opaque call that
runs model + guide + loss + backward + optimizer update. Decomposing this requires
instrumenting the SVI loop.

### 6.2 Approach: Manual Timing with torch.profiler or record_function

Pyro's SVI step internally does:
1. **Model trace:** Run model forward (includes forward model: ODE integration for task,
   spectral transfer for spectral)
2. **Guide trace:** Run guide forward (sample from variational distribution)
3. **ELBO computation:** Compute log_prob differences
4. **Backward pass:** torch.autograd.backward on the loss
5. **Optimizer step:** Adam/ClippedAdam parameter update

For a clean breakdown, we need to time phases 1-5 separately. Options:

**Option A: Use `torch.autograd.profiler.record_function` context managers**

Wrap each phase in a named context:

```python
with torch.autograd.profiler.record_function("forward_model"):
    model_trace = poutine.trace(model).get_trace(*model_args)
with torch.autograd.profiler.record_function("guide_eval"):
    guide_trace = poutine.trace(guide).get_trace(*model_args)
```

Then use `torch.profiler.profile()` to collect events. This gives accurate breakdown but
requires rewriting the SVI step logic.

**Option B: Use Pyro's `poutine.trace` + manual timing**

Run model and guide traces manually with `time.time()` around each call:

```python
import pyro.poutine as poutine

# Forward model timing
t0 = time.time()
model_trace = poutine.trace(model).get_trace(*model_args)
t_forward = time.time() - t0

# Guide timing
t0 = time.time()
guide_trace = poutine.trace(guide).get_trace(*model_args)
t_guide = time.time() - t0

# ELBO + backward timing
t0 = time.time()
loss = elbo.differentiable_loss(model, guide, *model_args)
loss.backward()
t_gradient = time.time() - t0
```

This gives approximate breakdown but is simpler.

**Option C: Profile only a few representative steps**

Time the full `svi.step()` for overall wall time (already done). Then, for 5-10 steps
at convergence, run the manual decomposition from Option B. Report the proportional
breakdown.

**Recommendation: Option C.** The full SVI loop uses `svi.step()` for training. For
timing decomposition, run 10 additional "profiling steps" after training completes on a
single representative dataset. This gives the timing breakdown without slowing down the
main benchmark. The breakdown is reported as percentages of total step time:

```
Method       | Total (ms) | Forward (%) | Guide (%) | Gradient (%)
auto_normal  |   14.2     |   62%       |   8%      |   30%
auto_mvn     |   18.7     |   52%       |   21%     |   27%
auto_iaf     |   45.3     |   28%       |   48%     |   24%
```

### 6.3 Pareto Frontier

Plot wall-time-per-step (x-axis, log scale) vs RMSE (y-axis). Each point is one
guide type at one network size. The Pareto frontier connects methods that are not
dominated (no other method is both faster AND more accurate).

This is a standard scatter plot with Pareto front overlay. Straightforward matplotlib.

Color by guide type, shape by network size. Label points with method names.

---

## 7. rDCM Integration Notes

### 7.1 rDCM is Already in the Benchmark

The `rdcm_vb.py` runner handles both rigid and sparse rDCM. These runners do NOT use
`create_guide` or `run_svi` -- they call `rigid_inversion()` / `sparse_inversion()`
directly (analytic VB, no guide type or ELBO choice).

### 7.2 rDCM in the Comparison Table

rDCM appears as 2 rows (rigid + sparse) in the cross-method table. Its metrics
(RMSE, coverage, correlation, wall time) are directly comparable to SVI-based methods.
The coverage computation is already done via `compute_coverage_from_ci` with 95% CI from
the analytic posterior (mu +/- 1.96*sigma).

For multi-level coverage (CAL-01), rDCM's analytic Gaussian posterior makes z-score CI
computation exact. The same `compute_coverage_multi_level` function works with rDCM by
constructing samples from the known Gaussian:

```python
# For rDCM: analytic posterior is Gaussian, can generate exact samples
samples = mu + std * torch.randn(1000, *mu.shape)
```

Or directly compute coverage from the CDF (more precise):

```python
from scipy.stats import norm
coverage = 2 * norm.cdf(z_score * std_ratio) - 1
```

But sample-based is simpler and consistent with the SVI-based methods.

### 7.3 rDCM at N=10

rDCM is extremely fast (analytic solution, <1s per dataset) and scales well. Running
rDCM rigid + sparse at N=10 is expected to work without issues. The success criteria
explicitly include rDCM at N=10.

---

## 8. Results Schema Extension

### 8.1 Current JSON Schema (from benchmark_results.json)

```json
{
  "spectral": {
    "rmse_list": [...],
    "coverage_list": [...],
    "correlation_list": [...],
    "elbo_list": [...],
    "time_list": [...],
    "a_true_list": [[...], ...],
    "a_inferred_list": [[...], ...],
    "mean_rmse": float,
    "std_rmse": float,
    "mean_coverage": float,
    "mean_correlation": float,
    "mean_time": float,
    "metadata": {
      "variant": "spectral",
      "method": "svi",
      "n_regions": 3,
      ...
    }
  }
}
```

### 8.2 Extended Schema for Phase 11

The results key changes from `"{variant}"` to
`"{variant}_{guide_type}_{elbo_type}_{n_regions}"` to avoid key collisions:

```json
{
  "spectral_auto_normal_trace_elbo_3": {
    "rmse_list": [...],
    "coverage_multi": {
      "0.50": [...],
      "0.75": [...],
      "0.90": [...],
      "0.95": [...]
    },
    "coverage_diag_multi": {
      "0.50": [...],
      ...
    },
    "coverage_offdiag_multi": {
      "0.50": [...],
      ...
    },
    "correlation_list": [...],
    "elbo_list": [...],
    "time_list": [...],
    "a_true_list": [[...], ...],
    "a_inferred_list": [[...], ...],
    "summary": {
      "rmse_median": float,
      "rmse_q25": float,
      "rmse_q75": float,
      "coverage_90_median": float,
      "coverage_90_q25": float,
      "coverage_90_q75": float,
      ...
    },
    "metadata": {
      "variant": "spectral",
      "method": "svi",
      "guide_type": "auto_normal",
      "elbo_type": "trace_elbo",
      "n_regions": 3,
      ...
    }
  },
  "rdcm_rigid_vb_trace_elbo_3": {
    ...
  }
}
```

The `coverage_multi` dict replaces the single `coverage_list`. Each key is a CI level
string, each value is a per-dataset list. `coverage_diag_multi` and
`coverage_offdiag_multi` provide the per-parameter breakdown.

### 8.3 Result Naming Convention

```
{variant}_{guide_type}_{elbo_type}_{n_regions}
```

Examples:
- `spectral_auto_normal_trace_elbo_3`
- `task_auto_mvn_renyi_elbo_5`
- `rdcm_rigid_vb_na_10` (rDCM has no guide/ELBO choice; use "vb" and "na")

---

## 9. Plotting Infrastructure

### 9.1 Existing Style System

`benchmarks/plotting.py` has a `_apply_style()` function that tries SciencePlots first,
then seaborn-v0_8-whitegrid, then default matplotlib. Uses tab10 colorblind-friendly
palette. Figures saved via `_save_figure(fig, path, formats)` with configurable
formats (PNG default, PDF optional).

### 9.2 New Figure Functions Needed

| Function | Output | Inputs |
|----------|--------|--------|
| `plot_calibration_curves` | Expected vs observed coverage | Multi-level coverage results |
| `plot_posterior_violins` | Per-A_ij violin overlays | Posterior samples from all methods |
| `plot_pareto_frontier` | Wall-time vs RMSE scatter | Summary metrics per method |
| `plot_timing_breakdown` | Stacked bar chart | Per-component timing data |
| `generate_comparison_table` | Markdown + LaTeX string | Summary metrics |

### 9.3 Figure Catalog (Publication Set)

| Figure # | Title | Requirement |
|----------|-------|-------------|
| Fig 1 | Calibration curves (expected vs observed coverage) | CAL-01 |
| Fig 2 | Per-parameter calibration (diag vs offdiag) | CAL-01 |
| Fig 3 | Network size scaling (RMSE vs N for each guide) | CAL-02 |
| Fig 4 | Cross-method comparison (metric strip plots) | CAL-03 |
| Fig 5 | Posterior violin overlays (per A_ij) | CAL-04 |
| Fig 6 | Pareto frontier (wall-time vs RMSE) | CAL-05 |
| Fig 7 | Timing breakdown (stacked bar) | CAL-05 |
| Table 1 | Cross-method comparison table | CAL-03 |

All figures output as both PNG (for quick review) and PDF (for publication).

### 9.4 Color and Style Conventions

Use consistent color mapping across all figures:

```python
GUIDE_COLORS = {
    "auto_delta": "#1f77b4",      # blue
    "auto_normal": "#ff7f0e",     # orange
    "auto_lowrank_mvn": "#2ca02c", # green
    "auto_mvn": "#d62728",        # red
    "auto_iaf": "#9467bd",        # purple
    "auto_laplace": "#8c564b",    # brown
    "rdcm_rigid": "#e377c2",      # pink
    "rdcm_sparse": "#7f7f7f",     # gray
}
```

Consistent across ALL Phase 11 figures for reader orientation.

---

## 10. Wall-Clock Profiling Design

### 10.1 Profiling Function

```python
def profile_svi_step(
    model, guide, model_args, n_steps=10,
) -> dict[str, float]:
    """Time model/guide/gradient components of SVI steps.

    Runs n_steps of manual SVI decomposition after training,
    returns median timing for each component.
    """
    import pyro.poutine as poutine

    forward_times = []
    guide_times = []
    gradient_times = []

    for _ in range(n_steps):
        # Forward model
        t0 = time.time()
        poutine.trace(model).get_trace(*model_args)
        forward_times.append(time.time() - t0)

        # Guide evaluation
        t0 = time.time()
        poutine.trace(guide).get_trace(*model_args)
        guide_times.append(time.time() - t0)

        # ELBO + backward
        t0 = time.time()
        loss = elbo.differentiable_loss(model, guide, *model_args)
        loss.backward()
        gradient_times.append(time.time() - t0)

    return {
        "forward_median": np.median(forward_times),
        "guide_median": np.median(guide_times),
        "gradient_median": np.median(gradient_times),
    }
```

### 10.2 When to Profile

Run profiling on a single dataset after each guide type has been trained. This avoids
contaminating the training timing with profiling overhead. The profiling is separate from
the benchmark loop.

### 10.3 Expected Timing Patterns

Based on architecture knowledge:
- **auto_delta:** Fastest guide (no sampling, just return MAP). Forward model dominates.
- **auto_normal:** Fast guide (independent Normal sampling). Forward model dominates.
- **auto_lowrank_mvn:** Moderate guide (low-rank MVN sampling). Guide cost grows with D.
- **auto_mvn:** Expensive guide (full Cholesky sampling). Guide cost grows as O(D^2).
- **auto_iaf:** Most expensive guide (sequential autoregressive flow). Guide cost
  dominates at large D.
- **auto_laplace:** Guide is AutoMVN post-Laplace. Same cost as auto_mvn for sampling.

---

## 11. Runner Modifications Required

### 11.1 SVI Runners (task_svi, spectral_svi)

Current runners compute coverage at a single level (95% CI via quantile bounds). Phase
11 needs:

1. **Multi-level coverage:** Compute coverage at 50%, 75%, 90%, 95% from posterior
   samples (already available from `extract_posterior_params`)
2. **Per-parameter breakdown:** Separate diagonal and off-diagonal A elements
3. **Store guide_type/elbo_type in metadata** (already done in Phase 10)
4. **Store per-dataset posterior samples** for at least one representative dataset (for
   violin plots)

**Recommended approach:** Rather than modifying the existing runners (which are already
working and tested), create a new `calibration_runner.py` that wraps the existing runner
functions and adds the multi-level coverage computation post-hoc. The existing runners
return `a_true_list` and `a_inferred_list` -- but NOT posterior samples. The calibration
runner needs to call `extract_posterior_params` and retain the samples for each dataset.

Actually, looking more carefully: the runners already call `extract_posterior_params` and
get back `A_free_samples` -- they just compute coverage from quantiles and discard the
samples. The modification is to retain the samples for coverage_multi computation.

### 11.2 Minimal Runner Change

Add multi-level coverage computation inside the per-dataset loop, right after extracting
posterior samples. The samples are already available -- just compute coverage at 4 levels
instead of 1:

```python
# After getting A_free_samples from extract_posterior_params:
for ci_level in [0.50, 0.75, 0.90, 0.95]:
    alpha = (1.0 - ci_level) / 2.0
    lo = torch.quantile(A_free_samples.float(), alpha, dim=0)
    hi = torch.quantile(A_free_samples.float(), 1.0 - alpha, dim=0)
    A_lo, A_hi = _build_A_ci(lo, hi, N)
    cov = compute_coverage_from_ci(A_true, A_lo, A_hi)
    coverage_multi[ci_level].append(cov)
```

This is a ~15-line addition to each SVI runner, inside the existing per-dataset loop.

### 11.3 rDCM Runner Modification

rDCM runners use analytic CI (mu +/- 1.96*sigma). For multi-level coverage, use
different z-scores:

```python
z_scores = {0.50: 0.6745, 0.75: 1.1503, 0.90: 1.6449, 0.95: 1.9600}
for level, z in z_scores.items():
    A_lo = A_mu - z * A_std
    A_hi = A_mu + z * A_std
    cov = compute_coverage_from_ci(A_true, A_lo, A_hi)
    coverage_multi[level].append(cov)
```

---

## 12. Subtask Decomposition Recommendation

### Subtask 11-01: Extended Metrics + Calibration Sweep Script

**Scope:**
- Add `compute_coverage_multi_level` to `benchmarks/metrics.py`
- Add z-scores for 0.50, 0.75 to existing z_table
- Create `benchmarks/calibration_sweep.py` with tiered configuration matrix
- Generate fixtures (if not already present)
- Run Tier 1 (all guides x trace_elbo x spectral x N=3)
- Produce initial `calibration_results.json`

**Why first:** Establishes the results data that all subsequent analysis and plotting
depends on. The sweep script is the backbone of Phase 11.

### Subtask 11-02: Calibration Curves + Comparison Table

**Scope:**
- Add `plot_calibration_curves` to `benchmarks/plotting.py`
- Add `generate_comparison_table` (JSON/Markdown/LaTeX output)
- Run remaining tiers (2-3) of benchmark sweep
- Generate Figures 1-4, Table 1

**Why second:** The calibration curves and comparison table are the "main results" of
v0.2.0. They depend on having Tier 1-3 results.

### Subtask 11-03: Violin Plots + Timing + Pareto

**Scope:**
- Add `plot_posterior_violins` to `benchmarks/plotting.py`
- Implement `profile_svi_step` for timing decomposition
- Add `plot_pareto_frontier` and `plot_timing_breakdown`
- Generate Figures 5-7

**Why last:** The violin plots and timing breakdown are supplementary to the main
calibration results. They require re-running inference on specific datasets (for
violins) and implementing the timing profiler (for breakdown).

### Dependency Chain

```
11-01 (metrics + sweep) -> 11-02 (curves + table) -> 11-03 (violins + timing)
```

11-01 produces the data. 11-02 analyzes and visualizes the main results. 11-03 adds
supplementary visualizations and profiling.

---

## 13. Risk Assessment

### 13.1 Runtime Risk

The full sweep is computationally expensive. Mitigation:
- Tier the runs (Section 2.2)
- Use `--quick` mode for development (3-5 datasets, 500 steps)
- Support `--resume` in the sweep script
- Spectral DCM is fastest (~5-7s per dataset at N=3); use it for Tier 1

### 13.2 NaN/Divergence Risk

Some guide types may fail on specific datasets. The existing runners already handle this
with try/except and `n_failed` counting. The sweep script should:
- Continue on failure (don't abort the full sweep)
- Require >= 50% success rate per configuration (existing threshold)
- Flag failed configurations in the results JSON

### 13.3 AutoIAF Convergence Risk

AutoIAF is the most complex guide type and may need more SVI steps or different learning
rates to converge. The current 500-step default for spectral DCM may be insufficient.
Mitigation: For AutoIAF, use 1000 steps or auto-detect convergence (check loss plateau).

### 13.4 Storage Risk

50 datasets x 6 guide types x 3 ELBO types x 3 variants x 3 sizes = potentially very
large JSON. Mitigation: The tiered approach limits what actually runs. Per-element lists
(a_true_list, a_inferred_list) are the largest fields. For N=3, each dataset contributes
9 floats per list -- manageable. For N=10, each dataset contributes 100 floats -- still
reasonable.

### 13.5 parameterize_A Batching Risk

The violin plot needs `parameterize_A` applied to posterior samples. Currently
`parameterize_A` takes a single (N,N) tensor. May need to loop over samples or vectorize.
This is a minor engineering detail but worth noting.

---

## 14. Confidence Assessment

| Area | Confidence | Rationale |
|------|-----------|-----------|
| Multi-level coverage computation | HIGH | Simple extension of existing compute_coverage_from_ci/samples |
| Calibration curve plotting | HIGH | Standard matplotlib; reference implementations widely available |
| Combinatorial sweep orchestration | HIGH | Straightforward nested loops over existing runner functions |
| Cross-method comparison table | HIGH | Post-processing of JSON results |
| Violin/ridge plots | MEDIUM | Need posterior samples stored or re-run; parameterize_A batching untested |
| Wall-clock timing decomposition | MEDIUM | Pyro's SVI step is opaque; manual decomposition via poutine.trace should work but is not common practice |
| Pareto frontier | HIGH | Standard scatter plot with convex hull |
| AutoIAF convergence at N=5+ | MEDIUM | May need hyperparameter tuning specific to IAF |
| Full sweep runtime | LOW | Hard to estimate precisely; depends on hardware and N=10 configurations |

---

## Sources

- Existing codebase: `benchmarks/config.py`, `benchmarks/metrics.py`, `benchmarks/plotting.py`, `benchmarks/runners/*.py`, `src/pyro_dcm/models/guides.py`
- Phase 10 research: `.planning/phases/10-guide-variants/10-RESEARCH.md`
- v0.2.0 research: `.planning/research/v0.2.0/FEATURES.md`, `.planning/research/v0.2.0/PITFALLS.md`
- [Simulation-Based Calibration Checking for Bayesian Computation](https://arxiv.org/abs/2211.02383) -- Modrak et al. (2023)
- [Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability](https://arxiv.org/html/2310.13402) -- Coverage calibration methodology
- [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf) -- Kuleshov et al. (2018), expected vs observed coverage framework
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) -- torch.profiler usage for timing decomposition
- [Matplotlib Violin Plot Documentation](https://matplotlib.org/stable/gallery/statistics/violinplot.html)
- [Pyro SVI Documentation](https://docs.pyro.ai/en/stable/inference_algos.html) -- SVI.step() internals
