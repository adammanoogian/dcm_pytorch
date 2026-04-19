---
phase: 10-guide-variants
plan: 02
subsystem: inference
tags: [elbo, svi, pyro, guide-variants, benchmarks]
depends_on:
  requires: ["10-01"]
  provides: ["ELBO variant plumbing in run_svi", "TraceMeanField guard", "AutoLaplace post-processing", "Aligned BenchmarkConfig string keys"]
  affects: ["10-03"]
tech-stack:
  added: []
  patterns: ["ELBO_REGISTRY string-keyed dispatch", "mean-field guard pattern", "post-Laplace AutoMVN extraction"]
key-files:
  created:
    - tests/test_elbo_variants.py
  modified:
    - src/pyro_dcm/models/guides.py
    - src/pyro_dcm/models/__init__.py
    - benchmarks/config.py
decisions:
  - id: "elbo-registry-pattern"
    description: "String-keyed ELBO_REGISTRY dict maps elbo_type to Pyro ELBO classes, same pattern as GUIDE_REGISTRY"
    rationale: "Consistent dispatch pattern, easy to extend, no enum overhead"
  - id: "renyi-alpha-0.5"
    description: "RenyiELBO uses alpha=0.5 with forced minimum 2 particles"
    rationale: "alpha=0.5 is the standard Renyi divergence midpoint; 2 particles is the minimum for valid Renyi gradient estimation"
  - id: "post-laplace-in-result"
    description: "AutoLaplace post-processing stores AutoMVN guide in result['guide'] key"
    rationale: "Users need the post-Laplace guide for posterior queries, not the MAP guide"
metrics:
  duration: "7m 21s"
  completed: "2026-04-12"
---

# Phase 10 Plan 02: ELBO Variant Plumbing Summary

ELBO_REGISTRY with Trace/TraceMeanField/Renyi dispatch, mean-field guard, AutoLaplace post-processing, aligned BenchmarkConfig defaults.

## What Was Done

### Task 1: Add ELBO plumbing to run_svi with AutoLaplace handling

Extended `run_svi` with two new parameters (`elbo_type`, `guide_type`) and supporting infrastructure:

- **ELBO_REGISTRY**: String-keyed dict mapping `"trace_elbo"`, `"tracemeanfield_elbo"`, `"renyi_elbo"` to Pyro ELBO classes
- **elbo_type parameter**: Selects which ELBO objective to use (default `"trace_elbo"` for backward compatibility)
- **guide_type parameter**: Enables mean-field guard validation
- **Mean-field guard**: `TraceMeanField_ELBO` + non-mean-field guide raises `ValueError` with actionable error message
- **RenyiELBO**: Uses `alpha=0.5`, forces `num_particles >= 2` (Renyi needs multi-particle)
- **AutoLaplace post-processing**: After SVI loop, calls `guide.laplace_approximation(*model_args)` and stores resulting `AutoMultivariateNormal` in `result["guide"]`
- **BenchmarkConfig alignment**: `guide_type` default changed from `"mean_field"` to `"auto_normal"`, `elbo_type` default changed from `"trace"` to `"trace_elbo"`

### Task 2: Test all 14 valid and 4 rejected guide x ELBO combinations

Created `tests/test_elbo_variants.py` with 26 tests:

- 14 parametrized tests for all valid (guide, ELBO) combinations -- each produces finite loss
- 4 parametrized tests for rejected combinations (TraceMeanField_ELBO + non-mean-field) -- each raises ValueError
- Count assertions: exactly 14 valid and 4 rejected
- RenyiELBO alpha smoke test and min-particles enforcement
- AutoLaplace returns `AutoMultivariateNormal` post-guide
- Non-Laplace guides do not include `"guide"` in result
- Default `elbo_type` backward compatibility (no elbo_type argument works)
- Invalid `elbo_type` raises `ValueError`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| ELBO_REGISTRY string-keyed dispatch | Consistent with GUIDE_REGISTRY pattern; extensible, no enum overhead |
| RenyiELBO alpha=0.5, min 2 particles | Standard midpoint alpha; 2 particles is minimum for valid Renyi gradients |
| Post-Laplace guide in result["guide"] | Users need AutoMVN for posterior queries, not the MAP guide |
| BenchmarkConfig defaults aligned to registry keys | Prevents string mismatch between config and guides.py registries |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- `tests/test_elbo_variants.py`: 26/26 passed (4.26s)
- `tests/test_svi_integration.py`: 9/9 passed (33.60s)
- `tests/test_guide_factory.py`: 24/24 passed (2.51s)

## Commits

| Hash | Message |
|------|---------|
| 867cf43 | feat(10-02): add ELBO variant plumbing to run_svi with AutoLaplace handling |
| 5ef1905 | test(10-02): add tests for all 14 valid and 4 rejected guide x ELBO combos |

## Next Phase Readiness

Plan 10-03 (benchmark matrix runner) can proceed. All prerequisites delivered:
- `ELBO_REGISTRY` and `GUIDE_REGISTRY` provide string-keyed dispatch
- `BenchmarkConfig` uses aligned string keys
- `run_svi` accepts `elbo_type` and `guide_type` for full combinatorial control
