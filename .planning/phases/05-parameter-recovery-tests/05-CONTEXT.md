# Phase 5: Parameter Recovery Tests (All Three Variants) - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Rigorous parameter recovery validation for all three DCM variants. Simulate with known ground truth, infer, and verify accuracy and calibration. This phase produces test scripts and validation results, not new library modules. No cross-validation against SPM — that's Phase 6.

</domain>

<decisions>
## Implementation Decisions

### General Principle
- **Standard scientific validation protocol** matching the roadmap thresholds exactly
- Claude decides all test design details: network configs, SVI settings, statistical tests, plotting
- Follow the roadmap's stated test protocol: 50 synthetic datasets, 3-region and 5-region networks, RMSE/coverage/correlation metrics

### Claude's Discretion
- All test protocol details: exact network configurations, A matrix values, stimulus designs
- SVI hyperparameters for recovery (learning rate, steps, convergence criteria)
- rDCM: use analytic VB (Phase 3) as primary, Pyro SVI as comparison if time allows
- Statistical analysis: how to compute coverage, what plotting library to use
- Whether 50 datasets per variant is feasible in test runtime or needs to be reduced for CI
- How to handle test cases where recovery fails (flag vs fail vs skip)
- Results documentation format (plots, tables, markdown reports)

</decisions>

<specifics>
## Specific Ideas

- Roadmap thresholds: RMSE(A) < 0.05, 95% CI coverage in [0.90, 0.99], convergence within 5000 SVI steps
- Test protocol: generate 50 synthetic datasets with known A (3-region, 5-region), run SVI, extract posteriors, compute metrics
- For rDCM: primary recovery via analytic VB (rigid_inversion, sparse_inversion) — much faster than SVI
- ELBO model comparison: verify correctly specified model ranks above misspecified alternatives
- Results should include: true vs inferred A scatter plots, posterior marginals, ELBO traces, coverage calibration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-parameter-recovery-tests*
*Context gathered: 2026-03-27*
