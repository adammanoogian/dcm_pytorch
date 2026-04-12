# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-06)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.2.0 Cross-Backend Inference Benchmarking

## Current Position

**Milestone:** v0.2.0 Cross-Backend Inference Benchmarking
**Phase:** 10 of 12 (Guide Variants) -- VERIFIED
**Plan:** 3/3 complete
**Status:** Phase 10 verified, ready for Phase 11
**Last activity:** 2026-04-12 -- Phase 10 verified (gap fixed: CLI defaults aligned)

Progress: [█████░░░░░] 50% (2/4 phases)

## Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| NumPyro deferred to v0.3+ | No reason to rewrite PyTorch forward models in JAX; adds bottlenecks | 2026-04-07 |
| Regularization deferred to v0.3+ | NCP, prior scale sensitivity orthogonal to main calibration story | 2026-04-07 |
| Amortized refinement deferred to v0.3+ | Semi-amortized pipeline requires Phase 10 guide variants first | 2026-04-07 |
| 4 phases for v0.2.0 | 12 requirements cluster into 4 natural delivery boundaries | 2026-04-07 |
| num_particles=5 for ELBO gap evaluation | Balances variance reduction vs cost for amortization gap metric | 2026-04-07 |
| kwargs passthrough in BenchmarkConfig factories | Forward-compatible extensibility without touching factory methods | 2026-04-07 |
| ELBO before clear_param_store ordering | Amortized guide params live in param store; must evaluate before clear | 2026-04-07 |
| rk4 solver for task fixture generation | dopri5 underflows with dt=0.01 + piecewise stimulus; rk4 is reliable | 2026-04-07 |
| No regressors stored in rDCM fixtures | Runners call create_regressors themselves; deterministic, saves space | 2026-04-07 |
| Duration override from fixture metadata | Fixtures may be generated with full-mode params; runners must use fixture duration to avoid shape mismatches | 2026-04-07 |
| PiecewiseConstantInput for fixture stimulus | task_dcm_model expects callable input_fn, not raw dict; fixtures must be wrapped | 2026-04-07 |
| AutoIAFNormal hidden_dim as list | Pyro's AutoRegressiveNN iterates over hidden_dims; int causes TypeError | 2026-04-12 |
| ELBO_REGISTRY string-keyed dispatch | Consistent with GUIDE_REGISTRY; extensible, no enum overhead | 2026-04-12 |
| RenyiELBO alpha=0.5, min 2 particles | Standard midpoint alpha; 2 particles is minimum for valid Renyi gradients | 2026-04-12 |
| Post-Laplace guide in result["guide"] | Users need AutoMVN for posterior queries, not the MAP guide | 2026-04-12 |
| Predictive-based extraction over guide.median() | Guide-agnostic; works for all 6 guide types identically | 2026-04-12 |
| Complex-site handling in extraction | Avoid float casting warning on complex predicted_csd | 2026-04-12 |
| Default num_samples=1000 for extraction | Balances accuracy vs speed for standard posterior queries | 2026-04-12 |

See STATE.md v0.1.0 decisions in git history.

## Blockers

None currently.

## Key Risks

- P1: Mean-field coverage ceiling ~0.80-0.88 is by design -- report per-parameter, not aggregate
- P6: AutoMultivariateNormal memory explosion at N=10 -- only run at N=3,5
- P9: Simpson's paradox in aggregated tables -- never aggregate across DCM variants
- P11: Combinatorial explosion -- tier the benchmark runs
- P12: Report median+IQR, not just means

## Session Continuity

Last session: 2026-04-12
Stopped at: Phase 10 verified, ready for Phase 11
Resume file: None

---
*Last updated: 2026-04-12 after Phase 10 verification*
