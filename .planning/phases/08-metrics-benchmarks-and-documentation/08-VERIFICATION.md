---
phase: 08-metrics-benchmarks-and-documentation
verified: 2026-04-02T13:55:38Z
status: gaps_found
score: 8/11 must-haves verified
gaps:
  - truth: CLI --quick produces JSON end-to-end and auto-generates figures
    status: partial
    reason: >
      CLI produces JSON correctly. But run_all_benchmarks.py never imports
      benchmarks.plotting or calls generate_all_figures(). config.save_figures
      is set at line 307 but never consumed. Figures in figures/ were generated
      separately, not by the CLI pipeline.
    artifacts:
      - path: benchmarks/run_all_benchmarks.py
        issue: config.save_figures set but generate_all_figures() never called.
    missing:
      - Import benchmarks.plotting and call generate_all_figures(results, args.output_dir)
        in main() after JSON save, conditioned on not args.no_figures
  - truth: Amortization gap computed from real amortized ELBO not a synthetic placeholder
    status: failed
    reason: >
      Both task_amortized.py and spectral_amortized.py compute the gap as
      svi_result[final_loss] * 1.1, a hardcoded 10% inflation of SVI ELBO.
      mean_correlation hardcoded to 0.0. BNC-02 and BNC-03 require actual measured values.
    artifacts:
      - path: benchmarks/runners/task_amortized.py
        issue: Line 390 uses svi_result[final_loss] * 1.1 as amortized_elbo. Line 423 hardcodes mean_correlation to 0.0.
      - path: benchmarks/runners/spectral_amortized.py
        issue: Line 372 same synthetic gap. Line 405 same mean_correlation placeholder.
    missing:
      - Compute actual amortized ELBO via Trace_ELBO().differentiable_loss(model, guide, *model_args)
        and pass to compute_amortization_gap() instead of svi_result[final_loss] * 1.1
      - Compute mean_correlation from actual recovered A values using pearson_corr()
  - truth: All public API functions have complete NumPy docstrings including Examples sections
    status: partial
    reason: >
      parameter_packing.py (TaskDCMPacker, SpectralDCMPacker) has 0 Examples
      sections across 365 lines and 8+ public methods. All other checked modules
      have Examples sections.
    artifacts:
      - path: src/pyro_dcm/guides/parameter_packing.py
        issue: 365 lines, 8 public methods, zero Examples sections.
    missing:
      - Add Examples sections to TaskDCMPacker.pack(), TaskDCMPacker.unpack(),
        SpectralDCMPacker.pack(), SpectralDCMPacker.unpack()
---

# Phase 8: Metrics, Benchmarks, and Documentation -- Verification Report

**Phase Goal:** Comprehensive benchmarking comparing all inference methods across all DCM variants, plus API documentation, quickstart tutorial, paper-ready methods section, and reproducibility scripts.
**Verified:** 2026-04-02T13:55:38Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Consolidated metrics (RMSE, coverage, correlation, gap) in metrics.py | VERIFIED | 195 lines, 5 real implementations, no stubs |
| 2 | BenchmarkConfig dataclass with quick/full factory methods | VERIFIED | 114 lines, quick_config() and full_config() present |
| 3 | CLI runner with --quick and --variant flags | VERIFIED | 352 lines, argparse wired, JSON output confirmed |
| 4 | CLI --quick produces JSON end-to-end; figures auto-generated | PARTIAL | JSON generated; plotting.generate_all_figures() never called by CLI |
| 5 | RUNNER_REGISTRY has 7 entries wired to real implementations | VERIFIED | 7 entries confirmed at runtime, all 7 runners imported and callable |
| 6 | Amortization gap computed from real amortized ELBO | FAILED | Both amortized runners use svi_result[final_loss] * 1.1 as placeholder |
| 7 | Docs: quickstart.md, methods.md, methods.tex, equations.md, references.bib | VERIFIED | All 5 exist in correct subdirs, 142-452 lines each |
| 8 | benchmark_report.md with unified comparison table and SPM12 row | VERIFIED | 287 lines, table at lines 70-79 including SPM12 rows |
| 9 | Amortization advantage analysis in benchmark_report.md | VERIFIED | Section 4 lines 111-159 analyzes gap with expected values |
| 10 | Figures saved as PDF and PNG | VERIFIED | 8 files in figures/ (4 figures x 2 formats), generated separately |
| 11 | All public API functions have complete docstrings with Examples | PARTIAL | parameter_packing.py: 0 Examples sections on 8+ methods |

**Score:** 8/11 truths verified

---

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| benchmarks/metrics.py | VERIFIED | 195 lines, 5 real implementations |
| benchmarks/config.py | VERIFIED | 114 lines, BenchmarkConfig with factories |
| benchmarks/run_all_benchmarks.py | PARTIAL | 352 lines; JSON OK; plotting never called |
| benchmarks/runners/__init__.py | VERIFIED | RUNNER_REGISTRY 7 entries at runtime |
| benchmarks/runners/task_svi.py | VERIFIED | 240 lines, real SVI loop |
| benchmarks/runners/spectral_svi.py | VERIFIED | 233 lines, real SVI loop |
| benchmarks/runners/rdcm_vb.py | VERIFIED | 480 lines, rigid + sparse VB |
| benchmarks/runners/task_amortized.py | PARTIAL | 453 lines; gap placeholder line 390; correlation 0.0 line 423 |
| benchmarks/runners/spectral_amortized.py | PARTIAL | 435 lines; same gap placeholder line 372; correlation 0.0 line 405 |
| benchmarks/runners/spm_reference.py | VERIFIED | 194 lines, parses VALIDATION_REPORT.md |
| benchmarks/plotting.py | VERIFIED | 490 lines, 5 public plot functions + generate_all_figures |
| tests/test_benchmark_metrics.py | VERIFIED | 135 lines, 9 tests, all passing |
| docs/02_pipeline_guide/quickstart.md | VERIFIED | 233 lines, complete 5-step tutorial |
| docs/03_methods_reference/methods.md | VERIFIED | 452 lines, cited equations |
| docs/03_methods_reference/methods.tex | VERIFIED | 222 lines |
| docs/03_methods_reference/equations.md | VERIFIED | 154 lines |
| docs/03_methods_reference/references.bib | VERIFIED | 142 lines |
| docs/04_scientific_reports/benchmark_report.md | VERIFIED | 287 lines, SPM12 rows, amortization section |
| figures/ (PDF + PNG pairs) | VERIFIED | 8 files (4 figures x 2 formats) |
| src/pyro_dcm/__init__.py exports | VERIFIED | extract_posterior_params + simulator functions in __all__ |
| src/pyro_dcm/guides/parameter_packing.py | PARTIAL | 365 lines; 0 Examples sections on 8+ public methods |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| run_all_benchmarks.py | RUNNER_REGISTRY | from benchmarks.runners import | WIRED | Dispatches runner(config) per combo |
| run_all_benchmarks.py | benchmarks/plotting.py | import + generate_all_figures() call | NOT_WIRED | save_figures config set but plotting never called |
| task_amortized.py | compute_amortization_gap() | real amortized ELBO | PARTIAL | Called with final_loss * 1.1 placeholder |
| spectral_amortized.py | compute_amortization_gap() | real amortized ELBO | PARTIAL | Same final_loss * 1.1 placeholder |
| task_svi.py | pyro_dcm.models.* | from pyro_dcm.models import | WIRED | task_dcm_model, create_guide, run_svi, extract_posterior_params |
| rdcm_vb.py | rdcm_posterior.* | from pyro_dcm.forward_models.rdcm_posterior import | WIRED | rigid_inversion + sparse_inversion |
| __init__.py | simulators | from pyro_dcm.simulators import | WIRED | simulate_rdcm, make_stable_A_rdcm, make_block_stimulus_rdcm |

---

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| BNC-01 | PARTIAL | Per-variant RMSE, coverage, ELBO, wall time reported. Amortized correlation hardcoded 0.0. |
| BNC-02 | BLOCKED | Amortized vs mean-field comparison uses synthetic ELBO. Real comparison requires actual amortized ELBO. |
| BNC-03 | BLOCKED | Amortization gap is svi_loss * 1.1, not actual per-subject vs amortized ELBO difference. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| benchmarks/runners/task_amortized.py | 390 | svi_result[final_loss] * 1.1 as amortized ELBO | Blocker | Fake amortized ELBO. BNC-03 cannot be satisfied. |
| benchmarks/runners/task_amortized.py | 423 | mean_correlation: 0.0 placeholder | Warning | Correlation column zeroed in benchmark table. |
| benchmarks/runners/spectral_amortized.py | 372 | svi_result[final_loss] * 1.1 as amortized ELBO | Blocker | Same fake amortized ELBO. |
| benchmarks/runners/spectral_amortized.py | 405 | mean_correlation: 0.0 placeholder | Warning | Same zero correlation placeholder. |
| benchmarks/run_all_benchmarks.py | 307 | save_figures set but plotting not called | Blocker | Figures not auto-generated by CLI pipeline. |
| src/pyro_dcm/guides/parameter_packing.py | -- | No Examples sections | Warning | 08-05 docstring completeness requirement not met. |

---

## Gaps Summary

**Gap 1 -- CLI plotting not wired:**
benchmarks/run_all_benchmarks.py sets config.save_figures at line 307 but never
imports benchmarks.plotting or calls generate_all_figures(). The 8 figures in
figures/ were generated by a separate script. The --no-figures flag has no effect
on the CLI pipeline. Fix: add from benchmarks import plotting and call
plotting.generate_all_figures(results, args.output_dir) after the JSON save,
guarded by if not args.no_figures.

**Gap 2 -- Synthetic amortization gap (BNC-02, BNC-03):**
Both task_amortized.py and spectral_amortized.py pass svi_result["final_loss"] * 1.1
as the amortized ELBO to compute_amortization_gap(). This is a hardcoded 10% synthetic
gap, not measured from the guide. The guides do perform forward passes and compute
RMSE/coverage, but their ELBO is never evaluated. Additionally, mean_correlation is
hardcoded to 0.0 rather than computed from the recovered A values. Fix: after the
amortized forward pass, evaluate the guide ELBO via
Trace_ELBO().differentiable_loss(amortized_model, guide, *model_args) and pass the
result to compute_amortization_gap(). Compute mean_correlation from the actual
recovered A_free values using pearson_corr().

**Gap 3 -- Missing Examples in parameter_packing.py (08-05):**
TaskDCMPacker and SpectralDCMPacker in src/pyro_dcm/guides/parameter_packing.py have
zero Examples sections across 365 lines and 8+ public methods. These classes are
exported via pyro_dcm.guides and used in both amortized benchmark runners. All other
modules checked have Examples sections. Fix: add minimal Examples to pack() and
unpack() on both classes.

---

*Verified: 2026-04-02T13:55:38Z*
*Verifier: Claude (gsd-verifier)*
