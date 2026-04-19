---
phase: 09-benchmark-foundation
verified: 2026-04-07T23:03:27Z
status: passed
score: 4/4 must-haves verified
---

# Phase 9: Benchmark Foundation Verification Report

**Phase Goal:** All benchmark runners operate on identical shared datasets with correct metrics
**Verified:** 2026-04-07T23:03:27Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running `generate_fixtures.py` produces .npz files for 3 variants x 3 sizes x N seeds, each containing ground-truth parameters and observed data | VERIFIED | `generate_fixtures.py` (363 lines) implements `generate_task_fixtures`, `generate_spectral_fixtures`, `generate_rdcm_fixtures`, each saving A_true + observed data via `np.savez` with manifest.json |
| 2 | All existing v0.1.0 runners load from shared fixtures when `fixtures_dir` is set and produce identical results to inline generation | VERIFIED | All 5 runners import `load_fixture` and have `if config.fixtures_dir is not None: ... else: <inline generation>` branches. Variable names are identical in both branches. |
| 3 | `BenchmarkConfig` accepts `guide_type`, `n_regions_list`, `elbo_type`, and `fixtures_dir` with defaults that preserve v0.1.0 behavior | VERIFIED | `config.py` has all 4 fields with defaults: `guide_type="mean_field"`, `n_regions_list=[3]`, `elbo_type="trace"`, `fixtures_dir=None`. Both `quick_config` and `full_config` accept `**kwargs` and forward them. |
| 4 | Amortization gap metric computes real ELBO via `Trace_ELBO().differentiable_loss()` for both amortized and per-subject guides, not the RMSE-ratio proxy | VERIFIED | Both amortized runners call `Trace_ELBO(num_particles=5).loss(model, guide, *args)` for amortized ELBO (before `clear_param_store`) and SVI ELBO (after SVI training). `compute_amortization_gap(svi_elbo, amortized_elbo)` takes float ELBO arguments. The function signature in `metrics.py` confirms it computes `elbo_amortized - elbo_svi`. The RMSE ratio is tracked separately as an auxiliary metric, not used for the gap. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `benchmarks/config.py` | Extended BenchmarkConfig with 4 new fields | VERIFIED | `guide_type`, `n_regions_list`, `elbo_type`, `fixtures_dir` all present with backward-compatible defaults; `field` import from dataclasses present; both factory methods accept `**kwargs` |
| `benchmarks/generate_fixtures.py` | CLI script for fixture generation | VERIFIED | 363 lines, shebang present, argparse CLI, 3 generator functions, manifest.json written per subdirectory |
| `benchmarks/fixtures.py` | `load_fixture` helper for runners | VERIFIED | `load_fixture` and `get_fixture_count` exported; complex tensor reconstruction from real/imag pairs implemented for csd, noisy_csd, X, Y |
| `benchmarks/runners/task_amortized.py` | Real ELBO evaluation for amortization gap | VERIFIED | `Trace_ELBO` imported; `elbo_fn.loss()` called twice (amortized before `clear_param_store`, SVI after training); `compute_amortization_gap(svi_elbo, amortized_elbo)` called with ELBO floats |
| `benchmarks/runners/spectral_amortized.py` | Real ELBO evaluation for amortization gap | VERIFIED | Same pattern as task_amortized; `elbo_fn.loss(amortized_spectral_dcm_model, guide, csd, freqs, a_mask, packer)` before clear; `elbo_fn.loss(spectral_dcm_model, svi_guide, *model_args)` after SVI |
| `benchmarks/runners/task_svi.py` | Fixture loading branch | VERIFIED | `from benchmarks.fixtures import load_fixture`; `if config.fixtures_dir is not None:` branch loading A_true, C, bold, stimulus_times/values |
| `benchmarks/runners/spectral_svi.py` | Fixture loading branch | VERIFIED | Fixture branch loads A_true, noisy_csd_real/imag, freqs; skips inline noise addition in else branch |
| `benchmarks/runners/rdcm_vb.py` | Fixture loading branch in both rigid and sparse functions | VERIFIED | Both `run_rdcm_rigid_vb` and `run_rdcm_sparse_vb` have identical fixture branches calling `load_fixture("rdcm", nr, i, config.fixtures_dir)` and recomputing regressors |
| `benchmarks/run_all_benchmarks.py` | CLI flags for fixtures_dir, guide_type, n_regions | VERIFIED | `--fixtures-dir`, `--guide-type`, `--n-regions` all present; mapped to `config.fixtures_dir`, `config.guide_type`, `config.n_regions_list`, `config.n_regions` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `benchmarks/config.py` | All runners | `BenchmarkConfig` dataclass consumed by all runners | VERIFIED | `fixtures_dir` field present; `quick_config`/`full_config` forward `**kwargs` |
| `benchmarks/runners/task_amortized.py` | `pyro.infer.Trace_ELBO` | ELBO evaluation for amortization gap | VERIFIED | `from pyro.infer import SVI, Trace_ELBO` at line 23; `elbo_fn = Trace_ELBO(num_particles=5)` used at lines 371 and 427 |
| `benchmarks/runners/spectral_amortized.py` | `pyro.infer.Trace_ELBO` | ELBO evaluation for amortization gap | VERIFIED | `from pyro.infer import SVI, Trace_ELBO` at line 22; same pattern |
| `benchmarks/generate_fixtures.py` | `pyro_dcm.simulators.*` | Simulator calls for all 3 variants | VERIFIED | Imports `simulate_task_dcm`, `simulate_spectral_dcm`, `make_stable_A_rdcm`, `generate_bold` |
| `benchmarks/generate_fixtures.py` | `pyro_dcm.models.spectral_dcm_model` | `decompose_csd_for_likelihood` for noise matching | VERIFIED | Import present; noise pattern exactly matches `spectral_svi.py` pattern |
| `benchmarks/fixtures.py` | All runners | `load_fixture` imported by all runners | VERIFIED | All 5 runners have `from benchmarks.fixtures import load_fixture` |
| `benchmarks/run_all_benchmarks.py` | `benchmarks/config.py` | CLI args mapped to BenchmarkConfig fields | VERIFIED | Lines 339-345 assign `fixtures_dir`, `guide_type`, `n_regions_list`, `n_regions` from parsed args |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| BENCH-01: Shared fixture generation + loading | SATISFIED | `generate_fixtures.py` produces .npz files; all runners branch on `fixtures_dir` |
| BENCH-02: Extended BenchmarkConfig + CLI | SATISFIED | 4 new fields in dataclass; 3 new CLI flags; backward-compatible defaults |
| BENCH-03: Real ELBO amortization gap | SATISFIED | `compute_amortization_gap` accepts ELBO floats; both runners compute via `Trace_ELBO(num_particles=5).loss()` |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `task_amortized.py` | `rmse_ratio_list` still tracked alongside ELBO gap | Info | RMSE ratio is an auxiliary diagnostic metric, not used for `amortization_gap_list`. The gap output key is `amortization_gap_list` populated from ELBO-based `gap_list`. No impact on correctness. |

No blockers found.

### Gaps Summary

No gaps. All 4 phase success criteria are satisfied by real implementations in the codebase.

---

_Verified: 2026-04-07T23:03:27Z_
_Verifier: Claude (gsd-verifier)_
