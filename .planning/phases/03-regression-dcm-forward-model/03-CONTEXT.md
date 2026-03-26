# Phase 3: Regression DCM Forward Model + rDCM Simulator - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the regression DCM (rDCM) frequency-domain likelihood with analytic posterior, ARD sparsity priors, free energy computation, and a simulator for synthetic frequency-domain data. rDCM is a fundamentally different DCM variant — linear regression in the frequency domain with closed-form posteriors per region. No Pyro integration yet — that's Phase 4.

</domain>

<decisions>
## Implementation Decisions

### General Principle
- **Follow Frässle et al. (2017) [REF-020] conventions for all implementation choices.**
- **Use the Julia implementation as authoritative reference:** https://github.com/ComputationalPsychiatry/RegressionDynamicCausalModeling.jl
- This includes: HRF specification, design matrix construction, ARD update rules, free energy formula, convergence criteria
- Same "when in doubt, follow the reference" principle as Phases 1-2 with SPM

### Claude's Discretion
- All implementation details: HRF basis choice, ARD iteration count, convergence threshold, design matrix construction
- Tensor vs numpy decisions for the analytic posterior (rDCM is closed-form, may not need autograd)
- How to structure the VB iteration loop (if ARD requires iterative updates)
- Test case design and tolerance thresholds
- Whether to implement both sparse and non-sparse variants or just sparse

</decisions>

<specifics>
## Specific Ideas

- Primary paper reference: Frässle et al. (2017) [REF-020] Eq. 4-15
- Julia implementation: https://github.com/ComputationalPsychiatry/RegressionDynamicCausalModeling.jl
- MATLAB implementation: tapas/rDCM toolbox (https://github.com/translationalneuromodeling/tapas)
- The Julia implementation is likely cleaner and more readable than the MATLAB tapas code
- Every function must cite [REF-020] equation numbers
- Region-wise factorization: p(y_j | A_j, C_j) is the key computational structure
- Analytic posterior: p(theta_j | y_j) = N(mu_j, Sigma_j) with closed-form mu_j, Sigma_j
- Free energy F = sum_j F_j in closed form [REF-020] Eq. 15

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-regression-dcm-forward-model*
*Context gathered: 2026-03-26*
