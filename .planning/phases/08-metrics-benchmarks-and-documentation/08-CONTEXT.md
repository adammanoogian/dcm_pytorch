# Phase 8: Metrics, Benchmarks, and Documentation - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive benchmarking comparing all inference methods (SVI, amortized, analytic VB) across all three DCM variants (task, spectral, rDCM), plus API documentation with quickstart tutorial, paper-ready methods section, and a reproducible benchmark runner.

</domain>

<decisions>
## Implementation Decisions

### Benchmark scope & metrics
- **Network size:** 3-region only (matches existing recovery tests, keeps runtime tractable)
- **Dataset count per condition:** Claude's discretion based on variant speed (spectral is fast, task DCM is slow)
- **Baseline comparison:** SVI only (no NUTS). NUTS is too expensive for task DCM and SVI is already well-established
- **SPM12 row:** Include SPM12 results from Phase 6 validation in the summary comparison table. Full Pyro-DCM vs SPM12 story in one place
- **Core metrics:** RMSE, coverage, ELBO, wall time per variant x method

### Documentation depth
- **Primary audience:** Researchers who want to run DCM with Pyro. They know DCM theory
- **Math in docs:** Key equations inline (neural state, balloon, BOLD, CSD), rest referenced via paper citations
- **Quickstart tutorial:** Yes, full end-to-end walkthrough: simulate data, run SVI, inspect posteriors, compare models
- **Methods section:** Paper-ready, APA format, in both LaTeX and Markdown. Equations, algorithm descriptions, benchmark tables formatted for journal submission

### Reproducibility design
- **Entry point:** Configurable CLI (`python benchmarks/run_all_benchmarks.py --variant task --method svi`). Filter by variant/method for partial reruns
- **MATLAB handling:** Skip gracefully. Auto-detect MATLAB availability; if missing, skip SPM12 rows and note 'MATLAB not available'. No error
- **Output:** Tables AND figures (bar charts, ELBO traces, posterior violin plots). Saved to figures/ directory
- **Runtime target:** No limit. Full fidelity for the paper. Run on a workstation
- **Raw results:** Save all metrics to `benchmarks/results/benchmark_results.json` for later analysis/replotting
- **Dependencies:** Optional extras in pyproject.toml (`pip install pyro-dcm[benchmark]`). matplotlib + benchmark-only deps separated from core
- **CI subset:** Claude's discretion on whether a `--quick` flag is practical

### Claude's Discretion
- Dataset count per benchmark condition (practical per variant)
- Whether to include a `--quick` CI subset flag
- Exact figure styles and layout
- Documentation page structure within docs/
- Compression of benchmark results

</decisions>

<specifics>
## Specific Ideas

- SPM12 comparison should be pulled from Phase 6 validation results (validation/VALIDATION_REPORT.md) into the unified benchmark table
- Quickstart should be the "hello world" of Pyro-DCM -- a researcher should be able to copy-paste and get results
- Methods section should be directly usable in a paper submission (LaTeX + Markdown dual format)

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 08-metrics-benchmarks-and-documentation*
*Context gathered: 2026-03-29*
