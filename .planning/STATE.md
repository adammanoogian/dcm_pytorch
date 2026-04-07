# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-06)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.2.0 Cross-Backend Inference Benchmarking

## Current Position

**Milestone:** v0.2.0 Cross-Backend Inference Benchmarking
**Phase:** 9 of 12 (Benchmark Foundation)
**Plan:** 1 of 3
**Status:** In progress
**Last activity:** 2026-04-07 -- Completed 09-01-PLAN.md (config extension + ELBO gap fix)

Progress: [█████████░] 93% (27/29 plans)

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

Last session: 2026-04-07
Stopped at: Completed 09-01-PLAN.md
Resume file: None

---
*Last updated: 2026-04-07 after 09-01 execution*
