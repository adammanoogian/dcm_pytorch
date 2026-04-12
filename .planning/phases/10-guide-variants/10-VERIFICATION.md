---
phase: 10-guide-variants
verified: 2026-04-12T17:56:24Z
status: gaps_found
score: 3/4 must-haves verified
gaps:
  - truth: Existing runners gain guide_type parameterization via BenchmarkConfig without breaking v0.1.0 behavior
    status: failed
    reason: run_all_benchmarks.py CLI --guide-type flag defaults to mean_field (not in GUIDE_REGISTRY). config.guide_type=mean_field reaches create_guide() which raises ValueError. All SVI/amortized datasets fail, returning status=insufficient_data. BenchmarkConfig has correct default auto_normal but CLI overwrites it.
    artifacts:
      - path: benchmarks/run_all_benchmarks.py
        issue: Line 289 default=mean_field is not a valid GUIDE_REGISTRY key. Line 340 config.guide_type=args.guide_type overwrites BenchmarkConfig default. No --elbo-type CLI flag exists.
    missing:
      - Change --guide-type CLI default from mean_field to auto_normal at line 289
      - Add --elbo-type CLI flag (type=str, default=trace_elbo) and wire via config.elbo_type = args.elbo_type
---

# Phase 10: Guide Variants Verification Report

**Phase Goal:** Users can select from 6 guide types and 3 ELBO objectives for any DCM variant
**Verified:** 2026-04-12T17:56:24Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | create_guide(model, guide_type=X) returns working guide for all 6 types | VERIFIED | GUIDE_REGISTRY maps all 6 keys; test_guide_factory.py covers all 6 with class assertions and SVI smoke tests |
| 2 | SVI converges with each guide type on spectral DCM at N=3 | VERIFIED | run_svi complete with ClippedAdam, NaN detection, LR decay; test_elbo_variants.py checks all 14 valid combos for finite loss |
| 3 | Trace_ELBO, TraceMeanField_ELBO, RenyiELBO each produce valid SVI runs | VERIFIED | ELBO_REGISTRY covers 3 types; TraceMeanField guard enforced; RenyiELBO alpha=0.5 with min 2 particles |
| 4 | Existing runners gain guide_type via BenchmarkConfig without breaking v0.1.0 behavior | FAILED | All 4 runners correctly plumb config.guide_type/elbo_type; BenchmarkConfig defaults are correct; but CLI --guide-type default=mean_field overwrites those defaults and causes ValueError in create_guide for all SVI runners when script invoked with no arguments |

**Score:** 3/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/pyro_dcm/models/guides.py | create_guide, run_svi, extract_posterior_params | VERIFIED | 454 lines; all 3 functions implemented; GUIDE_REGISTRY 6 entries; ELBO_REGISTRY 3 entries; no stubs |
| src/pyro_dcm/models/__init__.py | Exports all 3 inference functions | VERIFIED | All 3 in __all__; imported from guides.py |
| benchmarks/config.py | BenchmarkConfig with guide_type, elbo_type | VERIFIED | 145 lines; guide_type=auto_normal, elbo_type=trace_elbo defaults |
| benchmarks/runners/spectral_svi.py | guide_type/elbo_type plumbing | VERIFIED | Lines 168, 176-177 wired |
| benchmarks/runners/task_svi.py | guide_type/elbo_type plumbing | VERIFIED | Lines 176, 184-185 wired |
| benchmarks/runners/spectral_amortized.py | guide_type/elbo_type for SVI comparison | VERIFIED | Lines 402, 410-411 wired |
| benchmarks/runners/task_amortized.py | guide_type/elbo_type for SVI comparison | VERIFIED | Lines 408, 416-417 wired |
| benchmarks/run_all_benchmarks.py | --guide-type with valid default; --elbo-type flag | FAILED | --guide-type default=mean_field not in GUIDE_REGISTRY; no --elbo-type flag |
| tests/test_guide_factory.py | Factory tests for all 6 types | VERIFIED | 217 lines; class, init_scale asymmetry, blocklist, kwargs, SVI smoke |
| tests/test_elbo_variants.py | ELBO combination tests | VERIFIED | 253 lines; 14 valid combos, 4 rejected, RenyiELBO, AutoLaplace post-guide |
| tests/test_posterior_extraction.py | Posterior extraction tests | VERIFIED | 239 lines; all 6 types, AutoDelta std=0, median compat, sample quantiles |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| create_guide | GUIDE_REGISTRY | dict lookup L149 | WIRED | ValueError with valid keys listed on unknown type |
| run_svi | ELBO_REGISTRY | conditional L281-327 | WIRED | RenyiELBO alpha=0.5, min 2 particles enforced |
| run_svi | guide_type | TraceMeanField guard L290-300 | WIRED | Raises ValueError for non-mean-field guides |
| spectral_svi | create_guide | config.guide_type | WIRED | Lines 165-170 |
| spectral_svi | run_svi | config.guide_type, config.elbo_type | WIRED | Lines 172-178 |
| extract_posterior_params | Predictive | pyro.infer.Predictive L423 | WIRED | Predictive sampling; median dict backward-compat |
| CLI --guide-type | BenchmarkConfig.guide_type | config.guide_type = args.guide_type L340 | BROKEN | default=mean_field not in GUIDE_REGISTRY |
| CLI --elbo-type | BenchmarkConfig.elbo_type | (no flag exists) | NOT_WIRED | --elbo-type flag absent from argparse |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| GUIDE-01: create_guide supports 6 guide types | SATISFIED | All 6 implemented and tested |
| GUIDE-02: ELBO variant comparison across 3 types | SATISFIED | All 3 in run_svi; 14 valid combos tested |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| benchmarks/run_all_benchmarks.py | 289 | default=mean_field for --guide-type | Blocker | Stale Phase 9 value; ValueError in create_guide for every SVI/amortized dataset with default CLI args |
| benchmarks/run_all_benchmarks.py | (absent) | Missing --elbo-type flag | Warning | elbo_type cannot be controlled from CLI |

No TODO/FIXME/placeholder/stub patterns found in guides.py, config.py, or any runner.

### Gaps Summary

One gap prevents full goal achievement. The core infrastructure is complete:

- create_guide: all 6 guide types with correct kwargs, init_scale asymmetry, auto_mvn blocklist at n_regions>7
- run_svi: all 3 ELBO types, TraceMeanField guard, RenyiELBO alpha=0.5 min 2 particles, AutoLaplace post-processing, NaN detection
- extract_posterior_params: Predictive-based, AutoDelta std=0, median backward-compat
- All 4 runners: guide_type and elbo_type plumbed from BenchmarkConfig
- BenchmarkConfig: guide_type=auto_normal and elbo_type=trace_elbo are correct defaults
- Tests: 3 test files, 217+253+239 lines, all 6 guide types, all 3 ELBOs, 14 valid combos

The gap is in benchmarks/run_all_benchmarks.py: the --guide-type CLI flag retains the Phase 9 default value mean_field, which is absent from GUIDE_REGISTRY. Line 340 unconditionally writes args.guide_type to config.guide_type, overwriting the correct auto_normal dataclass default. Any invocation without --guide-type auto_normal explicitly causes create_guide() to raise ValueError for every dataset. Runners catch the ValueError and increment n_failed, returning status=insufficient_data after all datasets fail.

Fix: two lines in run_all_benchmarks.py -- change line 289 default to auto_normal, add --elbo-type argument and wire to config.elbo_type.

---

_Verified: 2026-04-12T17:56:24Z_
_Verifier: Claude (gsd-verifier)_
