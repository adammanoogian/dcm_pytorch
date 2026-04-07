# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-06)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.2.0 Cross-Backend Inference Benchmarking

## Current Position

**Milestone:** v0.2.0 Cross-Backend Inference Benchmarking
**Phase:** 9 of 12 (Benchmark Foundation)
**Plan:** --
**Status:** Ready to plan
**Last activity:** 2026-04-07 -- Roadmap created for v0.2.0 (4 phases, 12 requirements)

Progress: [░░░░░░░░░░] 0%

## Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| NumPyro deferred to v0.3+ | No reason to rewrite PyTorch forward models in JAX; adds bottlenecks | 2026-04-07 |
| Regularization deferred to v0.3+ | NCP, prior scale sensitivity orthogonal to main calibration story | 2026-04-07 |
| Amortized refinement deferred to v0.3+ | Semi-amortized pipeline requires Phase 10 guide variants first | 2026-04-07 |
| 4 phases for v0.2.0 | 12 requirements cluster into 4 natural delivery boundaries | 2026-04-07 |

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
Stopped at: Roadmap created for v0.2.0
Resume file: None

---
*Last updated: 2026-04-07 after roadmap creation*
