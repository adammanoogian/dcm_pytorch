---
phase: 23
plan: 01
subsystem: model-selection
tags: [bmr, bayesian-model-reduction, free-energy, model-selection, laplace]
backfilled: 2026-06-09
dependency-graph:
  requires: []
  provides: [bayesian-model-reduction, make-reduced-prior-zero-connection, model-selection-subpackage]
  affects: [23-02, 23-03]
tech-stack:
  added: []
  patterns: [analytic-free-energy, laplace-approximation, precision-via-solve, slogdet-logdet, float64-numerics]
key-files:
  created:
    - src/pyro_dcm/model_selection/__init__.py
    - src/pyro_dcm/model_selection/bmr.py
    - tests/test_bmr.py
  modified:
    - .planning/REFERENCES.md
decisions:
  - "[23-01-D1] BMR delta_F uses the Laplace approximation evaluated at the full posterior mean (delta_F = log p(mu_f|m_r) - log p(mu_f|m_f) + 0.5*[log|Sigma_r| - log|Sigma_f|]), with no trace term, validated against the exact conjugate Gaussian Bayes factor."
  - "[23-01-D2] BMR antisymmetry (delta_F(full->reduced) = -delta_F(reduced->full)) holds only for equal-covariance prior pairs; the symmetry test exercises that case."
metrics:
  duration: ~3 min
  completed: 2026-05-27
---

# Phase 23 Plan 01: Bayesian Model Reduction Core Summary

**One-liner:** Analytic Bayesian Model Reduction that scores a reduced DCM prior from a single full-model inversion, computing the reduced posterior and the change in log model evidence without refitting.

## What Was Built

New `pyro_dcm.model_selection` subpackage with `bmr.py` implementing post hoc model
selection per [REF-070] Friston & Penny (2011):

1. **`bayesian_model_reduction`** -- given the full-model posterior (mean `mu_f`,
   covariance `Sigma_f`), the full-model prior, and a reduced prior, returns
   `(delta_f, reduced_posterior_mean, reduced_posterior_cov)`.
   - Step 1: reduced posterior via Bayes-rule precision update
     `Sigma_r_post^-1 = Sigma_f^-1 + Sigma_r0^-1 - Sigma_0^-1`, with the reduced
     mean from the corresponding information vector.
   - Step 2: change in log evidence via the Laplace approximation at `mu_f`
     ([23-01-D1]): `delta_f = 0.5*(log|Sigma_r_post| - log|Sigma_f| + log|Sigma_0|
     - log|Sigma_r0| - quad_reduced + quad_full)`, where the quadratic terms are
     the prior-mismatch penalties `(mu_f - mu_r0)' P_r0 (mu_f - mu_r0)` and
     `(mu_f - mu_0)' P_0 (mu_f - mu_0)`. No trace term.
   - Numerics: all inputs cast to `float64`; precisions formed via
     `torch.linalg.solve` (not `torch.inverse`); log-determinants via
     `torch.linalg.slogdet`. Positive-definiteness of the reduced posterior is
     guarded by a Cholesky check -- on failure the function warns and returns
     `delta_f = -inf` with NaN posterior tensors.

2. **`make_reduced_prior_zero_connection`** -- builds a reduced prior that prunes
   the parameters at the given indices: sets their prior mean to 0, zeroes their
   cross-covariances, and shrinks their variance to `shrinkage_variance`
   (default `1e-8`). All other parameters retain the original prior. This is the
   "switch a connection off" reduction underlying circuit pruning.

Both functions are re-exported from `pyro_dcm.model_selection.__init__`. REF-070
(Friston & Penny 2011, *Post hoc Bayesian model selection*, NeuroImage 56(4)) was
added to `.planning/REFERENCES.md`.

## Test Coverage

8 tests in `tests/test_bmr.py` (under `TestBayesianModelReduction`):

| Test | What It Verifies |
|------|-----------------|
| test_identical_priors_zero_delta | reduced prior == full prior gives delta_F == 0 and reduced posterior == full posterior |
| test_analytic_1d_case | delta_F matches the closed-form 1-D Gaussian Bayes factor |
| test_tight_reduction_on_correct_zero | shrinking a near-zero posterior parameter to N(0, 1e-8) increases evidence (delta_F > 0) |
| test_tight_reduction_on_nonzero | shrinking a far-from-zero posterior parameter to N(0, 1e-8) decreases evidence (delta_F < 0) |
| test_reduced_posterior_shape | returned reduced mean/cov have the input dimensionality |
| test_multidimensional_consistency | multivariate delta_F agrees with the 1-D marginal reduction of a single parameter |
| test_symmetry_of_delta_f | delta_F(full->reduced) == -delta_F(reduced->full) for equal-covariance priors [23-01-D2] |
| test_make_reduced_prior_zero_connection | helper zeroes mean, shrinks variance, clears cross-covariances at target indices, leaves others unchanged |

## Deviations from Plan

- The plan's Task 1 step 2 specified re-exporting a `bayesian_model_reduction_batch`
  function from `__init__.py`. This was **not implemented** -- no batch function
  exists in the shipped commit; only `bayesian_model_reduction` and
  `make_reduced_prior_zero_connection` are exported. Batch/exhaustive scoring was
  instead delivered in 23-02 as `enumerate_reduced_models` / `bmr_circuit_selection`.
- The plan sketched several candidate free-energy formulations (including a
  trace-term / SPM12 `spm_log_evidence` variant). The implementation settled on the
  Laplace formulation without a trace term ([23-01-D1]), validated against the exact
  conjugate Gaussian Bayes factor.

## Commits

| Hash | Message |
|------|---------|
| 97ef99a | feat(23-01): implement Bayesian Model Reduction core function and model_selection subpackage |
| f6409b2 | docs(23-01): update STATE.md with BMR implementation completion |

## Next Phase Readiness

The BMR core API (`bayesian_model_reduction`, `make_reduced_prior_zero_connection`)
is complete and tested, providing the foundation consumed by 23-02
(circuit-size selection via exhaustive enumeration and ranked delta-F scoring).
