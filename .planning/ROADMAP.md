# Roadmap: Pyro-DCM

## Milestones

- **v0.1.0 Foundation** - Phases 1-8 (shipped 2026-04-03)
- **v0.2.0 Cross-Backend Inference Benchmarking** - Phases 9-12 (in progress)

## Overview

v0.2.0 extends the benchmark suite to answer: how much does the variational
approximation cost for fMRI DCM? The milestone adds 6 Pyro guide variants,
3 ELBO objectives, shared fixture generation for reproducibility, and a
systematic calibration study across guide types and network sizes. The output
is a practical recommendation guide and updated benchmark narrative with
real data replacing all v0.1.0 TBD entries.

<details>
<summary>v0.1.0 Foundation (Phases 1-8) - SHIPPED 2026-04-03</summary>

See `.planning/MILESTONES.md` for details. 8 phases, 26 plans, 127 commits.

</details>

## v0.2.0 Cross-Backend Inference Benchmarking

**Milestone Goal:** Establish inference quality guarantees across all Pyro
guide families, fix calibration (0.44-0.78 coverage to nominal levels),
and produce a practical recommendation guide for DCM users.

### Phases

- [ ] **Phase 9: Benchmark Foundation** - Shared fixtures, extended config, amortization gap fix
- [ ] **Phase 10: Guide Variants** - 6 guide types in create_guide factory, ELBO variant comparison
- [ ] **Phase 11: Calibration Analysis** - Coverage curves, scaling study, comparison table, timing, plots
- [ ] **Phase 12: Documentation** - Recommendation guide and updated benchmark narrative

## Phase Details

### Phase 9: Benchmark Foundation
**Goal**: All benchmark runners operate on identical shared datasets with correct metrics
**Depends on**: Phase 8 (v0.1.0 benchmark infrastructure)
**Requirements**: BENCH-01, BENCH-02, BENCH-03
**Success Criteria** (what must be TRUE):
  1. Running `generate_fixtures.py` produces .npz files for 3 variants x 3 sizes (3, 5, 10 regions) x N seeds, each containing ground-truth parameters and observed data
  2. All existing v0.1.0 runners load from shared fixtures (when fixtures_dir is set) and produce identical results to inline generation
  3. BenchmarkConfig accepts guide_type, n_regions_list, elbo_type, and fixtures_dir with defaults that preserve v0.1.0 behavior
  4. Amortization gap metric computes real ELBO via Trace_ELBO().differentiable_loss() for both amortized and per-subject guides, not the RMSE-ratio proxy
**Plans:** 3 plans
Plans:
- [ ] 09-01-PLAN.md -- Extend BenchmarkConfig + fix amortization gap ELBO proxy
- [ ] 09-02-PLAN.md -- Create fixture generation script and loading helper
- [ ] 09-03-PLAN.md -- Wire fixture loading into runners + CLI flags

### Phase 10: Guide Variants
**Goal**: Users can select from 6 guide types and 3 ELBO objectives for any DCM variant
**Depends on**: Phase 9 (shared fixtures and extended config)
**Requirements**: GUIDE-01, GUIDE-02
**Success Criteria** (what must be TRUE):
  1. `create_guide(model, guide_type=X)` returns a working guide for each of: AutoDelta, AutoNormal, AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoIAFNormal, AutoLaplaceApproximation
  2. SVI converges (ELBO decreases) with each guide type on spectral DCM at N=3
  3. Trace_ELBO, TraceMeanField_ELBO, and RenyiELBO(alpha=0.5) each produce valid SVI training runs with identical guide and data
  4. Existing runners gain guide_type parameterization via BenchmarkConfig without breaking v0.1.0 behavior
**Plans**: TBD

### Phase 11: Calibration Analysis
**Goal**: The calibration properties of every guide type are characterized across network sizes with publication-quality figures and tables
**Depends on**: Phase 10 (guide variants available for benchmarking)
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04, CAL-05
**Success Criteria** (what must be TRUE):
  1. Per-parameter coverage calibration curves (expected vs observed at 50%, 75%, 90%, 95%) exist for each guide type, showing whether AutoLowRank/AutoMVN improve over AutoNormal's 0.44-0.78 ceiling
  2. Benchmark results exist for 3 and 5 regions (all guides) and 10 regions (mean-field and rDCM only), with AutoMultivariateNormal excluded at N=10 due to memory constraints
  3. Cross-method comparison table reports RMSE, coverage@90%, Pearson correlation, and wall time for 6+ methods x 3 variants x 3 sizes, with results never aggregated across DCM variants
  4. Per-parameter posterior comparison plots (violin or ridge) overlay all methods per A_ij element for a representative dataset, with ground truth marked
  5. Wall-clock timing breakdown (forward model, guide evaluation, gradient) and Pareto frontier (wall-time vs RMSE) are generated, reporting median+IQR not just means
**Plans**: TBD

### Phase 12: Documentation
**Goal**: Users can select the right guide for their use case from a decision tree, and the benchmark narrative contains real results
**Depends on**: Phase 11 (calibration results available)
**Requirements**: DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. A decision tree guide exists that recommends a guide type given compute budget, network size, and DCM variant, with clear warnings about mean-field coverage ceilings and full-rank memory limits
  2. The benchmark narrative report has zero TBD entries -- all placeholders from v0.1.0 are replaced with v0.2.0 calibration results and method comparisons
**Plans**: TBD

## Progress

**Execution Order:** 9 -> 10 -> 11 -> 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 9. Benchmark Foundation | v0.2.0 | 0/3 | Not started | - |
| 10. Guide Variants | v0.2.0 | 0/TBD | Not started | - |
| 11. Calibration Analysis | v0.2.0 | 0/TBD | Not started | - |
| 12. Documentation | v0.2.0 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-04-07*
*Last updated: 2026-04-07*
