# Requirements: Pyro-DCM v0.2.0

**Defined:** 2026-04-07
**Core Value:** The A matrix (effective connectivity) remains an explicit, interpretable object with full posterior uncertainty throughout inference

## v0.2.0 Requirements

Requirements for Cross-Backend Inference Benchmarking. Each maps to roadmap phases.

### Pyro Guide Extensions

- [ ] **GUIDE-01**: `create_guide` factory supports 6 guide types: AutoDelta, AutoNormal, AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoIAFNormal, AutoLaplaceApproximation
- [ ] **GUIDE-02**: ELBO variant comparison across Trace_ELBO, TraceMeanField_ELBO, and RenyiELBO(alpha=0.5) with identical guide/data

### Benchmark Infrastructure

- [ ] **BENCH-01**: Shared `.npz` fixture generation script produces synthetic datasets for 3 DCM variants x 3 network sizes (3, 5, 10 regions) x N seeds, loadable by all runners
- [ ] **BENCH-02**: Extended `BenchmarkConfig` with `guide_type`, `n_regions_list`, `elbo_type`, and `fixtures_dir` fields (defaults preserve v0.1.0 behavior)
- [ ] **BENCH-03**: Amortization gap metric uses real ELBO via `Trace_ELBO().differentiable_loss(model, guide, *args)` instead of RMSE-ratio proxy

### Calibration Analysis

- [ ] **CAL-01**: Per-parameter coverage calibration curves showing expected vs observed coverage at 50%, 75%, 90%, 95% nominal levels for each guide type
- [ ] **CAL-02**: Network size scaling benchmarks at 3 and 5 regions for all guides, 10 regions for mean-field and rDCM only
- [ ] **CAL-03**: Cross-method comparison table: 6+ methods x 3 variants x 3 sizes reporting RMSE, coverage@90%, Pearson correlation, and wall time
- [ ] **CAL-04**: Per-parameter posterior comparison plots (violin or ridge overlay of all methods per A_ij element)
- [ ] **CAL-05**: Wall-clock timing breakdown (ODE/CSD forward, guide evaluation, gradient) and Pareto frontier (wall-time vs RMSE)

### Documentation

- [ ] **DOC-01**: Practical recommendation guide with decision tree for guide selection by compute budget, network size, and DCM variant
- [ ] **DOC-02**: Updated benchmark narrative report replacing all TBD entries from v0.1.0 with v0.2.0 results

## v0.3+ Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### NumPyro Backends

- **NUMPYRO-01**: NUTS reference posterior for spectral DCM (gold standard calibration anchor)
- **NUMPYRO-02**: NumPyro ADVI and Laplace runners to isolate Pyro-vs-NumPyro differences
- **NUMPYRO-03**: Pyro MCMC (HMC/NUTS) as alternative to NumPyro for MCMC without JAX reimplementation

### Regularization

- **REG-01**: Non-centered parameterization via `poutine.reparam(LocScaleReparam)`
- **REG-02**: Prior scale sensitivity sweep (1/256, 1/64, 1/32, 1/16, 1)
- **REG-03**: Prior predictive checks for physiological plausibility

### Amortized Refinement

- **AMORT-01**: Semi-amortized pipeline (amortized init + 50-100 SVI refinement steps)
- **AMORT-02**: Amortization gap closure curve at refinement steps 0, 10, 50, 100
- **AMORT-03**: Training data scaling study (200, 1000, 10000 simulations)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| JAX reimplementation of forward models | Crossing languages adds bottlenecks; maintain single PyTorch codebase |
| DCM_PPLs integration | ERP-DCM (Jansen-Rit), not fMRI-DCM; cannot be used as drop-in |
| Cross-PPL runtime benchmarking | Baldy et al. (2025) already did this; no added value |
| Real-data benchmarking | Ground truth unknown; simulation is correct approach |
| Novel guide architectures | Scope creep; use Pyro's built-in autoguides |
| Block-diagonal structured guide | Custom implementation, high effort, unclear gain over AutoLowRank |
| Full SBC (500+ datasets x 9 methods) | Computationally prohibitive; defer to supplementary |
| Automatic method selection | Benchmark characterizes tradeoffs; users decide |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GUIDE-01 | Phase 10 | Pending |
| GUIDE-02 | Phase 10 | Pending |
| BENCH-01 | Phase 9 | Complete |
| BENCH-02 | Phase 9 | Complete |
| BENCH-03 | Phase 9 | Complete |
| CAL-01 | Phase 11 | Pending |
| CAL-02 | Phase 11 | Pending |
| CAL-03 | Phase 11 | Pending |
| CAL-04 | Phase 11 | Pending |
| CAL-05 | Phase 11 | Pending |
| DOC-01 | Phase 12 | Pending |
| DOC-02 | Phase 12 | Pending |

**Coverage:**
- v0.2.0 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-04-07*
*Last updated: 2026-04-07 after Phase 9 completion*
