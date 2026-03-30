# Phase 8 Research: Metrics, Benchmarks, and Documentation

**Researched:** 2026-03-30
**Mode:** Ecosystem
**Overall confidence:** HIGH (infrastructure exists; Phase 8 is synthesis, not new algorithms)

---

## Executive Summary

Phase 8 is a synthesis phase -- the mathematics and algorithms are done. The work is
about measuring, documenting, and packaging what already exists. The codebase has 296
non-slow tests across 30 test files, three complete DCM variants (task, spectral, rDCM),
two inference methods per applicable variant (SVI per-subject, amortized flow), and
analytic VB for rDCM. The existing recovery tests (Phase 5) and amortized benchmark
(Phase 7) already compute the exact metrics needed (RMSE, coverage, ELBO, wall time,
amortization gap). Phase 8 elevates these from scattered test assertions into a unified
benchmark suite with persistent JSON results, publication-quality figures, and a
reproducible CLI entry point.

The documentation deliverable has three orthogonal components: (1) API reference
documentation ensuring every public function has complete NumPy-style docstrings with
examples, (2) a quickstart tutorial that researchers can copy-paste to go from
installation to posterior inspection in under 5 minutes, and (3) a paper-ready methods
section in both LaTeX and Markdown covering the mathematical framework, inference
algorithms, and benchmark results. The recent Baldy et al. (2025) paper "Dynamic causal
modelling in probabilistic programming languages" (J. R. Soc. Interface) provides a
direct methodological precedent -- they benchmark DCM inference across CmdStanPy, PyMC,
NumPyro, and BlackJAX with metrics including RMSE, ESS, wall time, and convergence
diagnostics. Our benchmark extends this by adding amortized inference and cross-spectral
density DCM variants.

The main risk is runtime: task DCM SVI takes ~1-2s per step on CPU, so a full benchmark
with 50 datasets x 3000 steps x multiple seeds could take 40+ hours. The benchmark
design must separate a "full fidelity" mode (for paper results) from a "quick" mode (for
CI/development validation). All other components are straightforward engineering.

---

## Technology Stack

### Benchmark Infrastructure

**CLI Framework: argparse (standard library)**
**Confidence:** HIGH

Use argparse, not Click or Typer. Rationale:
- Zero external dependencies (already used in `generate_training_data.py` and
  `train_amortized_guide.py` -- maintains consistency)
- Phase 8 adds no new core deps; the project already has argparse-based CLI patterns
- `--variant`, `--method`, `--quick` flags are simple enough for argparse
- No need for Click's decorator pattern for a single entry point script

**Results Storage: JSON**
**Confidence:** HIGH

Use `json` (standard library) for benchmark results, not pickle or custom formats.
- Human-readable, diff-friendly, version-controllable
- Load/replot without running benchmarks again
- Structure: `benchmarks/results/benchmark_results.json`
- Include metadata: timestamp, git hash, hardware info, Python/PyTorch versions

**Visualization: matplotlib + SciencePlots**
**Confidence:** HIGH (matplotlib already a dependency; SciencePlots is optional)

- matplotlib is already the project's plotting library (CLAUDE.md)
- SciencePlots (`pip install SciencePlots`) provides journal-quality styles via
  `plt.style.use('science')`. Requires LaTeX installation. Make it optional.
- Publication figures: bar charts (RMSE/time comparison), ELBO traces (line plots),
  posterior violin plots, scatter plots (true vs inferred A)
- Save as both PDF (vector, for paper) and PNG (raster, for README/docs)

**Benchmark optional dependency group in pyproject.toml:**

```toml
[project.optional-dependencies]
benchmark = [
    "matplotlib",
    "SciencePlots",
    "tabulate",  # terminal table formatting
]
dev = [
    "pytest",
    "ruff",
    "mypy",
]
```

Note: `matplotlib` and `scipy` are already core dependencies. `SciencePlots` and
`tabulate` are the only new additions, both optional.

### Documentation Tools

**Docstring Format: NumPy-style (already enforced by ruff)**
**Confidence:** HIGH

The project already uses NumPy-style docstrings with `convention = "numpy"` in ruff
config. Phase 8 ensures completeness and adds Examples sections where missing.

**Methods Section: Markdown primary, LaTeX secondary**
**Confidence:** HIGH

Write the methods section in Markdown first (readable in GitHub, usable in docs/), then
provide a LaTeX version for paper submission. No automated conversion tool needed --
the methods section is a one-time artifact, not a generated document.

---

## Benchmark Design

### Benchmark Matrix

The core benchmark is a 5x3 matrix (variants x methods), but not all cells are valid:

| | SVI (per-subject) | Amortized Flow | Analytic VB |
|------|-------------------|----------------|-------------|
| Task DCM | BNC-01, BNC-02 | BNC-02, BNC-03 | -- |
| Spectral DCM | BNC-01, BNC-02 | BNC-02, BNC-03 | -- |
| rDCM (rigid) | -- | -- | BNC-01 |
| rDCM (sparse) | -- | -- | BNC-01 |
| SPM12 (ref) | BNC-01 (from Phase 6) | -- | -- |

9 active cells. Each cell reports: RMSE(A), coverage, ELBO (or free energy), wall time,
gradient steps (where applicable).

### Metrics per Cell (BNC-01)

| Metric | Definition | Units |
|--------|-----------|-------|
| RMSE(A) | sqrt(mean((A_true - A_inferred)^2)) | unitless |
| Coverage | fraction of A elements with true value inside 90% CI | proportion [0,1] |
| ELBO | final negative ELBO loss (SVI) or free energy (rDCM) | nats |
| Wall time | total inference time per subject | seconds |
| Gradient steps | SVI steps to convergence | count |
| Correlation | Pearson r between A_true and A_inferred (flattened) | [-1, 1] |

### Amortization-Specific Metrics (BNC-02, BNC-03)

| Metric | Definition | Units |
|--------|-----------|-------|
| RMSE ratio | RMSE(amortized) / RMSE(SVI) | ratio (target < 1.5x) |
| Amortization gap | ELBO(SVI) - ELBO(amortized) | nats |
| Relative gap | gap / |ELBO(SVI)| | proportion (target < 10%) |
| Speed ratio | time(amortized) / time(SVI) | ratio (target << 1) |
| Coverage delta | coverage(amortized) - coverage(SVI) | proportion |

### Dataset Counts (Recommended)

Based on existing infrastructure runtime:

| Variant | Datasets | Rationale |
|---------|----------|-----------|
| Task DCM (SVI) | 20 | ~1-2s/step x 3000 steps x 20 = 17-33 hours |
| Task DCM (amortized) | 20 | Requires pre-trained guide; inference < 1s each |
| Spectral DCM (SVI) | 50 | ~0.01s/step x 500 steps x 50 = ~4 minutes |
| Spectral DCM (amortized) | 50 | Requires pre-trained guide; inference < 1s each |
| rDCM rigid | 50 | Analytic VB; ~1s per inversion |
| rDCM sparse | 50 | Analytic VB; ~2-5s per inversion |

Total estimated runtime (full fidelity): 18-34 hours, dominated by task DCM SVI.

Quick mode (for CI/development):

| Variant | Datasets | Steps |
|---------|----------|-------|
| Task DCM | 3 | 500 |
| Spectral DCM | 5 | 500 |
| rDCM | 5 | N/A (analytic) |

Quick mode runtime: ~30 minutes.

### SPM12 Row

Pull from `validation/VALIDATION_REPORT.md`. The SPM12 comparison table is structured
as: seeds [42, 123, 456], metrics [Max Rel Error, Mean Rel Error, Sign Agreement, F].
Map these into the benchmark table:
- RMSE: compute from exported element-wise errors (or note "pending MATLAB run")
- Coverage: N/A (SPM12 does not produce posterior samples in the same sense)
- ELBO: SPM12 free energy F (comparable to negative ELBO)
- Wall time: SPM12 estimation time (if available from MATLAB run)

If MATLAB is not available at benchmark time, include the row with "MATLAB required"
annotation. The benchmark runner should skip gracefully.

---

## Features Landscape

### Table Stakes (Must Have)

| Feature | Why Expected | Complexity | Source |
|---------|-------------|------------|--------|
| Benchmark results table (all variants x methods) | BNC-01 requirement | Medium | ROADMAP |
| JSON results file | Reproducibility, replotting | Low | 08-CONTEXT |
| CLI entry point with --variant, --method flags | 08-CONTEXT decision | Low | 08-CONTEXT |
| RMSE bar chart across methods | Standard benchmark visualization | Low | 08-CONTEXT |
| ELBO trace plots per variant | Convergence visualization | Low | Phase 5 pattern |
| Posterior violin plots | Uncertainty visualization | Medium | 08-CONTEXT |
| True vs inferred scatter plots | Recovery accuracy | Low | Phase 5 pattern |
| Wall time comparison table | Performance benchmark | Low | BNC-01 |
| Amortization gap analysis | BNC-03 requirement | Low | Phase 7 |
| Quickstart tutorial | 08-CONTEXT decision | Medium | 08-CONTEXT |
| API docstring completeness | Standard for released packages | Medium | 08-CONTEXT |
| Methods section (Markdown) | Paper-ready documentation | High | 08-CONTEXT |

### Differentiators (Value-Add)

| Feature | Value Proposition | Complexity | Notes |
|---------|------------------|------------|-------|
| Methods section (LaTeX) | Direct paper submission | Medium | Dual format from 08-CONTEXT |
| Hardware info in JSON metadata | Reproducibility across machines | Low | platform, torch.cuda, CPU |
| --quick flag for CI subset | Developer workflow | Low | Reduced datasets/steps |
| Graceful MATLAB skip | Works without MATLAB installed | Low | Auto-detect, annotate |
| SciencePlots styling | Journal-quality figures | Low | Optional dependency |
| Per-element A matrix heatmaps | Detailed connectivity analysis | Low | Useful for paper figures |
| Summary statistics table (mean +/- std) | Standard reporting | Low | Over seeds/datasets |
| Tabulate terminal output | Readable CLI output | Low | Optional dep |

### Anti-Features (Do NOT Build)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|----------|-------------------|
| Interactive dashboard (Streamlit/Dash) | Scope creep; CLI + saved figures sufficient | Static HTML or PDF report |
| Automated report generation (Jinja2 templates) | Over-engineering for a one-time report | Write report manually from JSON |
| Benchmark database (SQLite) | JSON is simpler, diff-friendly | JSON flat file |
| GPU benchmarking | Project targets CPU (research-grade, not production) | Note GPU capability in docs |
| Comparison vs sbi/BayesFlow | Different paradigm (likelihood-free); not apples-to-apples | Cite as related work |
| Auto-generated Sphinx API docs | Massive tooling overhead for 27 modules | Ensure docstrings are complete; API is the docs |
| Notebook-based tutorial (.ipynb) | Harder to maintain, harder to lint | .py script or .md quickstart |

---

## Architecture Patterns

### Benchmark Runner Architecture

```
benchmarks/
    run_all_benchmarks.py       # CLI entry point
    config.py                   # BenchmarkConfig dataclass
    runners/
        __init__.py
        task_svi.py             # Task DCM + SVI runner
        task_amortized.py       # Task DCM + amortized runner
        spectral_svi.py         # Spectral DCM + SVI runner
        spectral_amortized.py   # Spectral DCM + amortized runner
        rdcm_vb.py              # rDCM analytic VB runner
        spm_reference.py        # SPM12 results loader
    metrics.py                  # RMSE, coverage, correlation, gap
    plotting.py                 # All figure generation
    results/
        benchmark_results.json  # Raw results (gitignored except template)
```

**Pattern: Config Dataclass**

```python
@dataclass
class BenchmarkConfig:
    variant: str               # "task", "spectral", "rdcm"
    method: str                # "svi", "amortized", "vb"
    n_datasets: int
    n_regions: int = 3
    n_svi_steps: int = 3000
    seed: int = 42
    quick: bool = False        # Reduced params for CI
    output_dir: str = "benchmarks/results"
```

**Pattern: Per-Condition Runner Interface**

Each runner function has the same signature:

```python
def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run a single benchmark condition.

    Returns
    -------
    dict
        Keys: rmse_list, coverage_list, elbo_list, time_list,
              correlation_list, n_steps_list, metadata.
    """
```

This allows the CLI to dispatch to any runner via `{"task_svi": run_task_svi, ...}`.

**Pattern: Metrics Separation**

Move metric computation (RMSE, coverage, correlation) from test files to
`benchmarks/metrics.py`. The existing implementations in `test_task_dcm_recovery.py`,
`test_spectral_dcm_recovery.py`, and `test_rdcm_recovery.py` all have identical helper
functions (`compute_rmse_A`, `_pearson_corr`, coverage calculation). Consolidate into
one authoritative location. Tests can import from `benchmarks.metrics`.

**Anti-Pattern: Monolithic Script**

Do NOT put all benchmark logic in a single `run_all_benchmarks.py`. The existing
`test_amortized_benchmark.py` (587 lines) shows how unwieldy single-file benchmarks
become. Use the runners/ subpackage pattern above.

### Documentation Architecture

```
docs/
    00_current_todos/           # Existing (keep)
    01_project_protocol/        # Existing (keep)
    02_pipeline_guide/
        quickstart.md           # NEW: end-to-end tutorial
    03_methods_reference/
        methods.md              # NEW: paper-ready methods (Markdown)
        methods.tex             # NEW: paper-ready methods (LaTeX)
        equations.md            # NEW: key equations quick-reference
    04_scientific_reports/
        benchmark_report.md     # NEW: benchmark results narrative
```

**Pattern: Quickstart as Self-Contained Script**

The quickstart tutorial should be a runnable .md file (code blocks that can be
copy-pasted). Structure:

1. Installation (`pip install pyro-dcm`)
2. Simulate data (5 lines using task_simulator)
3. Run SVI inference (5 lines using run_svi)
4. Inspect posteriors (extract_posterior_params + matplotlib)
5. Compare models (run two models with different masks, compare ELBO)

Total: ~40 lines of user code, with explanatory text between blocks.

**Pattern: Methods Section Structure (following Baldy et al. 2025 and standard DCM papers)**

1. Generative Model
   - Neural state equation (Eq. 1, [REF-001])
   - Hemodynamic model (Eq. 2-6, [REF-002])
   - Spectral transfer function (Eq. 3-4, [REF-010])
   - Regression DCM (Eq. 1-8, [REF-020])
2. Inference
   - Stochastic Variational Inference (ELBO, mean-field guide)
   - Amortized Inference (summary networks, normalizing flows)
   - Analytic VB for rDCM (closed-form posterior)
3. Model Comparison
   - ELBO-based Bayesian model comparison
   - Free energy for rDCM
4. Implementation
   - Software stack (PyTorch, Pyro, torchdiffeq, Zuko)
   - Numerical stability measures
5. Benchmarks
   - Benchmark protocol
   - Results tables and figures

### Figures Architecture

```
figures/
    benchmark_rmse_comparison.pdf      # Bar chart: RMSE by variant x method
    benchmark_time_comparison.pdf      # Bar chart: wall time by variant x method
    benchmark_coverage_comparison.pdf  # Bar chart: coverage by variant x method
    elbo_trace_task.pdf                # Line plot: ELBO vs step for task DCM
    elbo_trace_spectral.pdf            # Line plot: ELBO vs step for spectral DCM
    posterior_violin_task.pdf           # Violin: posterior A elements, task DCM
    posterior_violin_spectral.pdf       # Violin: posterior A elements, spectral DCM
    true_vs_inferred_scatter.pdf       # Scatter: A_true vs A_inferred, all variants
    amortization_gap.pdf               # Bar chart: gap per variant
    a_matrix_heatmap.pdf               # Heatmap: true vs inferred A (example)
```

All figures saved as PDF (vector) + PNG (raster) dual format.

---

## Domain Pitfalls

### Critical Pitfalls

#### Pitfall 1: Task DCM Benchmark Runtime Explosion
**What goes wrong:** Task DCM SVI takes 1-2s per step. A naive benchmark with 50
datasets x 3000 steps = 42-83 hours. Adding multiple seeds/conditions multiplies this
further.
**Why it happens:** ODE integration in every SVI step is fundamentally expensive.
**Consequences:** Benchmark never completes, or is run with insufficient steps/datasets,
producing unreliable metrics.
**Prevention:**
- Design two modes: `--quick` (3 datasets, 500 steps, ~30 min) and full (20 datasets,
  3000 steps, ~33 hours).
- Run full benchmark on a workstation, save JSON results, commit results to repo.
- Quick mode validates pipeline correctness only, not metric accuracy.
- Consider: spectral DCM and rDCM can run with more datasets (they're fast) to
  compensate for task DCM's smaller sample.
**Detection:** If task DCM benchmark takes > 40 hours, reduce n_datasets or n_steps.

#### Pitfall 2: Amortized Benchmark Requires Pre-Trained Guide
**What goes wrong:** The amortized benchmark cannot run without a pre-trained
normalizing flow guide. Training the guide is itself hours-long (task: 14-28h,
spectral: ~30 min). If the benchmark script tries to train from scratch, runtime
doubles.
**Why it happens:** Amortized inference separates training cost (one-time, expensive)
from inference cost (per-subject, cheap). The benchmark should measure inference cost,
not training cost.
**Consequences:** Either the benchmark is impossibly slow, or it uses an untrained guide
producing meaningless metrics.
**Prevention:**
- Separate the benchmark into two phases: (a) pre-train guides using
  `scripts/train_amortized_guide.py`, save to `models/`; (b) load pre-trained guides in
  the benchmark runner.
- Provide pre-trained guide weights in the repo (or document how to generate them).
- The benchmark runner should fail gracefully if guide weights are missing: "Pre-trained
  guide not found at models/task_final.pt. Run scripts/train_amortized_guide.py first."
- In `--quick` mode, skip amortized benchmark entirely or use a CI-scale guide (200
  training examples, as in test_amortized_benchmark.py).
**Detection:** If amortized inference produces RMSE > 5x SVI, the guide is untrained or
poorly trained.

#### Pitfall 3: Non-Reproducible Results from Stochastic Inference
**What goes wrong:** SVI results vary across runs due to stochastic gradient descent.
Two runs with the same data but different seeds produce different posterior means. This
makes benchmark tables non-reproducible.
**Why it happens:** SVI uses stochastic gradients; Pyro's internal random state affects
guide initialization and ELBO estimation.
**Consequences:** Benchmark numbers change each run, undermining paper claims.
**Prevention:**
- Fix ALL random seeds: `torch.manual_seed(seed)`, `pyro.set_rng_seed(seed)`,
  `numpy.random.seed(seed)`.
- Report mean +/- std across datasets (not a single run).
- Store all per-dataset results in JSON; the summary table is computed from stored
  results.
- Record the exact seeds used in JSON metadata.
**Detection:** Run benchmark twice with same config; results should be identical.

### Moderate Pitfalls

#### Pitfall 4: Apples-to-Oranges ELBO Comparison
**What goes wrong:** Comparing ELBO across variants (task vs spectral vs rDCM) is
meaningless because they have different likelihoods, different data dimensionalities,
and different numbers of latent variables. Comparing ELBO between SVI and rDCM free
energy is also problematic: ELBO is a stochastic lower bound while free energy is an
analytic bound.
**Prevention:**
- Compare ELBO only within the same variant (task SVI vs task amortized).
- For cross-variant comparison, use RMSE and coverage (which are on comparable scales).
- For rDCM, report free energy separately from ELBO; note in the table footnotes that
  they are not directly comparable.
- The amortization gap (BNC-03) is computed within-variant: per-subject ELBO minus
  amortized ELBO for the same subject and data.

#### Pitfall 5: Coverage Depends on CI Level
**What goes wrong:** Coverage at 90% CI vs 95% CI gives very different numbers. The
existing codebase uses 90% CI in some places (test_amortized_benchmark.py: 1.645 z)
and 95% CI in others (recovery tests). Inconsistent CI level across the benchmark table
creates confusion.
**Prevention:**
- Standardize on 90% CI (1.645 * std) for all benchmark coverage metrics.
- Document the CI level in the table header and JSON metadata.
- The known mean-field underdispersion means 95% coverage will be systematically low
  (0.80-0.88 for SVI); 90% CI is more informative for mean-field methods.

#### Pitfall 6: Incomplete Docstrings Breaking Documentation Claims
**What goes wrong:** The ROADMAP success criterion says "API documentation (docstrings +
usage examples) following NumPy-style conventions." But current docstrings may lack
Examples sections, or some private functions may lack docstrings. Phase 8 claims
documentation is complete, but ruff only enforces some docstring checks.
**Prevention:**
- Run `ruff check --select D` on all source files and fix all violations.
- Add Examples sections to the top-level public API functions (those in __init__.py
  `__all__` lists): ~15 functions across forward_models, models, guides, simulators.
- Do NOT add examples to every internal helper; focus on the public API.
- The quickstart tutorial doubles as a usage example for the most common workflow.

### Minor Pitfalls

#### Pitfall 7: Large JSON Results File
**What goes wrong:** Storing per-dataset posterior samples in JSON produces massive
files (50 datasets x 9 A elements x 500 samples = 225K floats per condition).
**Prevention:** Store summary statistics (mean, std, quantiles) in JSON, not raw
samples. Store raw samples only if needed for posterior violin plots; use separate .pt
files for raw tensor data. Keep JSON under 1MB.

#### Pitfall 8: SciencePlots Requires LaTeX
**What goes wrong:** SciencePlots style `'science'` requires a working LaTeX
installation. On machines without LaTeX, `plt.style.use('science')` raises an error.
**Prevention:** Wrap style usage in try/except. Fall back to `plt.style.use('seaborn-v0_8-whitegrid')` or default matplotlib style if SciencePlots/LaTeX not available.
Log a warning rather than failing.

#### Pitfall 9: Methods Section Equations Drifting from Code
**What goes wrong:** The methods section presents equations that don't exactly match
the code implementation (e.g., different parameterization, different constants). Readers
who compare the paper equations to the source code find discrepancies.
**Prevention:** Every equation in the methods section must cite the same [REF-XXX] and
equation number as the corresponding code docstring. Add comments like
"See `src/pyro_dcm/forward_models/balloon_model.py` for implementation."

---

## Existing Code Reuse

### Functions to Consolidate into benchmarks/metrics.py

The following identical helper functions appear in 3+ test files and should be
consolidated:

| Function | Source Files | Action |
|----------|-------------|--------|
| `_pearson_corr(x, y)` | test_task_dcm_recovery, test_spectral_dcm_recovery, test_rdcm_recovery | Move to benchmarks/metrics.py |
| `compute_rmse_A(A_true, A_inferred)` | test_task_dcm_recovery, test_spectral_dcm_recovery | Move to benchmarks/metrics.py |
| Coverage computation (inline) | test_amortized_benchmark, test_spectral_dcm_recovery | Extract to benchmarks/metrics.py |
| `invert_A_to_A_free(A)` | scripts/generate_training_data.py | Already in scripts; benchmark can import |

### Existing Infrastructure to Reuse

| Component | Location | Reuse in Phase 8 |
|-----------|----------|------------------|
| Task DCM simulator | `pyro_dcm.simulators.task_simulator` | Generate benchmark datasets |
| Spectral DCM simulator | `pyro_dcm.simulators.spectral_simulator` | Generate benchmark datasets |
| rDCM simulator | `pyro_dcm.simulators.rdcm_simulator` | Generate benchmark datasets |
| SVI runner | `pyro_dcm.models.guides.run_svi` | Per-subject SVI benchmark |
| Guide factory | `pyro_dcm.models.guides.create_guide` | Per-subject SVI benchmark |
| Posterior extraction | `pyro_dcm.models.guides.extract_posterior_params` | Post-SVI analysis |
| Amortized flow guide | `pyro_dcm.guides.AmortizedFlowGuide` | Amortized benchmark |
| Training data generator | `scripts/generate_training_data.py` | Pre-generate benchmark data |
| Amortized training script | `scripts/train_amortized_guide.py` | Pre-train guides |
| Validation report | `validation/VALIDATION_REPORT.md` | SPM12 comparison row |
| rDCM rigid/sparse inversion | `pyro_dcm.forward_models.rdcm_posterior` | Analytic VB benchmark |

---

## Implications for Roadmap

### Suggested Plan Structure

Based on the three orthogonal deliverables (benchmarks, documentation, figures), the
phase naturally splits into 3 plans:

**Plan 1: Benchmark Infrastructure + Core Metrics (BNC-01, BNC-02, BNC-03)**
- Create benchmarks/ directory structure
- Implement metrics.py (consolidated RMSE, coverage, correlation, gap)
- Implement config.py (BenchmarkConfig dataclass)
- Implement per-variant runners (6 runners)
- Implement run_all_benchmarks.py CLI
- Implement --quick mode
- Test with quick mode (CI-suitable tests)
- Add `[benchmark]` extras to pyproject.toml

This is the heaviest plan. The runners contain the actual benchmark logic, but they are
largely assembly of existing components (simulators + inference + metrics). The novel
code is the CLI orchestration and JSON output.

**Plan 2: Figures + Plotting + Results Analysis**
- Implement plotting.py (all figure types)
- Generate full benchmark results (or quick-mode results)
- Create all figures (10+ plots)
- Write benchmark_report.md narrative
- SPM12 comparison table (from validation report)
- Amortization advantage analysis (BNC-02, BNC-03 narrative)

Depends on Plan 1 producing JSON results. Can run in quick mode for development, then
full mode for final results.

**Plan 3: API Documentation + Quickstart + Methods Section**
- Audit all public API docstrings for completeness
- Add Examples sections to top-level functions
- Write quickstart.md tutorial
- Write methods.md (Markdown)
- Write methods.tex (LaTeX)
- Write equations.md quick-reference
- Final pyproject.toml updates

Independent of Plans 1-2 (documentation work is parallelizable). The methods section
references benchmark results but does not depend on them being generated.

### Phase Ordering Rationale

Plan 1 must come first because Plans 2-3 depend on the benchmark infrastructure and
metrics module. Plan 2 depends on Plan 1's JSON output. Plan 3 is independent and
could theoretically run in parallel with Plan 2, but sequencing them allows the
methods section to reference final benchmark results.

### Research Flags

- **Plan 1:** Standard engineering; no research needed. All algorithms already
  implemented. The only question is "how many datasets per variant" which is a
  runtime trade-off, not a research question.
- **Plan 2:** Plotting is routine. May need to iterate on figure styles for paper
  quality, but no algorithmic research.
- **Plan 3:** Methods section writing is the most intellectually demanding part. Must
  carefully map code to equations. The Baldy et al. (2025) paper provides a structural
  template. No algorithmic research needed.

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|-----------|--------|
| Benchmark metrics | HIGH | Identical metrics already computed in Phase 5 + 7 tests |
| CLI design | HIGH | Pattern established by generate_training_data.py, train_amortized_guide.py |
| Figures/plotting | HIGH | matplotlib is the project standard; patterns are routine |
| JSON results format | HIGH | Standard practice; no novel design needed |
| Documentation structure | HIGH | Directory layout defined in CLAUDE.md |
| Methods section content | HIGH | All equations traced to REFERENCES.md; Baldy et al. 2025 provides template |
| Runtime estimates | MEDIUM | Based on Phase 5/7 observations (1-2s/step task DCM); actual runtime depends on hardware |
| SciencePlots compatibility | MEDIUM | Requires LaTeX; fallback to default matplotlib if unavailable |
| Quick mode thresholds | MEDIUM | CI-scale; needs empirical calibration during Plan 1 |

---

## Gaps to Address

1. **Pre-trained guide weights:** The amortized benchmark requires pre-trained guides.
   These must be generated before the benchmark runs. Decision: include guide training
   as a prerequisite step documented in the benchmark README, not as part of the
   benchmark runner itself.

2. **SPM12 results availability:** The validation report shows SPM12 cross-validation
   is "pending MATLAB run." If MATLAB is never run, the SPM12 row in the comparison
   table will show "N/A." This is acceptable -- the benchmark should note it clearly.

3. **Amortized coverage calibration:** Phase 7 showed CI-scale coverage of 0.55-0.65
   with 200 training examples. Full-scale training (10,000+ examples) should improve
   this to [0.85, 0.99], but this has not been empirically verified. The full benchmark
   will be the first time this is measured at scale.

4. **Task DCM RMSE with full SVI:** Phase 5 CI tests used 500 steps (pipeline
   validation only). The strict RMSE < 0.05 threshold was set for 3000-5000 steps.
   The full benchmark will be the first test of this threshold at scale.

---

## Sources

### Verified (HIGH confidence)
- Project codebase: all source files, tests, and planning documents read directly
- pyproject.toml: current dependencies and configuration
- validation/VALIDATION_REPORT.md: SPM12 cross-validation status
- CLAUDE.md: coding conventions and project structure

### Referenced Papers
- [Baldy et al. (2025)](https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0880) --
  DCM in probabilistic programming languages (benchmark structure precedent)
- [Kim & Welling (2018)](https://arxiv.org/abs/2209.10888) -- Amortization gap
  (JAIR systematic review 2022, updated survey)

### Tools Evaluated
- [SciencePlots](https://github.com/garrettj403/SciencePlots) -- matplotlib styles for
  scientific publication (v2.2.1)
- [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) -- NumPy docstring
  style guide
- argparse (Python stdlib) -- CLI framework (already used in project)

---

*Research completed: 2026-03-30*
