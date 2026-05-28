---
phase: 21
plan: 03
subsystem: rnn-latent-analysis
tags: [fixed-point, jacobian, eigenvalues, pca, dimensionality-reduction, latent-extraction, r-squared-gate]
one-liner: "Fixed-point analysis via Adam on ||dh/dt||^2 + sklearn PCA pipeline with output-R2 gate for latent trajectory extraction (RNN-04, DIM-01/02/03)"

dependency-graph:
  requires:
    - "21-01 (ContinuousTimeRNN)"
    - "21-02 (train_rnn, eval_rnn_performance)"
  provides:
    - "find_fixed_points, compute_jacobian_at_fp, classify_stability"
    - "extract_trajectories, pca_reduce, output_r_squared_gate, variance_explained_diagnostic, zscore_trajectories"
  affects:
    - "21-04 (cluster training scripts that will use extract_trajectories)"
    - "22 (PIPE-03: linearization quality uses compute_jacobian_at_fp + classify_stability)"
    - "Phase 22 DCM fitting pipeline consumes pca_reduce / zscore_trajectories output"

tech-stack:
  added: []
  patterns:
    - "Lazy sklearn import inside pca_reduce (optional dep guard)"
    - "Adam optimization on ||dh/dt||^2 for fixed-point finding (no external library)"
    - "torch.autograd.functional.jacobian for exact automatic differentiation at FPs"
    - "torch.linalg.eig for complex eigenvalue stability classification"
    - "L2-distance deduplication of converged fixed points"
    - "Orthogonal embedding test construction for robust output-R2 fail case"

key-files:
  created:
    - src/pyro_dcm/rnn/fixed_point_analysis.py
    - src/pyro_dcm/rnn/latent_extraction.py
    - tests/test_fixed_point_analysis.py
    - tests/test_latent_extraction.py
  modified:
    - src/pyro_dcm/rnn/__init__.py

decisions:
  - id: "21-03-D1"
    decision: "Module docstring placed before from __future__ import annotations."
    rationale: "ruff E402 requires all imports to follow module-level docstring; PEP 257 module docstring must be the first statement. from __future__ goes after."
  - id: "21-03-D2"
    decision: "test_gate_fails_insufficient_components uses orthogonal basis embedding with equal per-factor variance."
    rationale: "Original test with mixing matrix still allowed 1 PC to capture >90% variance (correlated factors). Truly orthogonal equal-variance embedding forces PC1 to ~33% variance, guaranteeing gate fails with N=1."
  - id: "21-03-D3"
    decision: "classify_stability parameter named jacobian_matrix (not jacobian) to avoid shadowing torch.autograd.functional.jacobian."
    rationale: "fixed_point_analysis.py imports jacobian from torch.autograd.functional at module level; naming the parameter jacobian would shadow the import inside classify_stability."
  - id: "21-03-D4"
    decision: "extract_trajectories metadata stored under __meta__ key as plain Python dict (not np.ndarray)."
    rationale: "np.ndarray cannot store heterogeneous scalar types (float dt_seconds, float tau, float alpha). dict is the correct container; callers must isinstance-check. Documented in docstring."

metrics:
  duration: "~25 minutes"
  completed: "2026-05-25"
  tasks: 2/2
  tests-added: 26
  tests-passed: 26/26
---

# Phase 21 Plan 03: Fixed-Point Analysis and Latent Extraction Summary

Fixed-point analysis utilities (RNN-04) and latent extraction + PCA pipeline (DIM-01/02/03) for
CT-RNN latent circuit diagnostics. Adam on `||dh/dt||^2`, Jacobian via autograd, sklearn PCA with
output-R2 gate and z-score normalization.

## What Was Built

### Task 1: fixed_point_analysis.py and latent_extraction.py

**`src/pyro_dcm/rnn/fixed_point_analysis.py`** (new, ~175 lines):
- `find_fixed_points()`: Adam optimization on `||dh/dt||^2 = ||-h + f(W_rec@h + W_in@u + b)||^2`
  with n_inits random starts, early stopping at `loss < tol`, convergence threshold to keep
  candidates, and L2-distance deduplication via `_deduplicate_fixed_points()`.
- `compute_jacobian_at_fp()`: Uses `torch.autograd.functional.jacobian` to compute `d(dh/dt)/dh`
  at a given fixed point. Returns `(H, H)` tensor.
- `classify_stability()`: Uses `torch.linalg.eig` to classify stability from eigenvalue real parts.
  Returns `{"eigenvalues", "stable", "n_unstable", "max_real_part"}`.

**`src/pyro_dcm/rnn/latent_extraction.py`** (new, ~250 lines):
- `extract_trajectories()`: Runs RNN in eval mode on neurogym env, collects `(n_trials, T, H)`
  h(t) trajectories per condition. Stores `dt_seconds` metadata for Phase 22 time-grid alignment.
- `pca_reduce()`: Lazy-imports sklearn PCA, fits on training data, returns `(pca, projected)`.
- `variance_explained_diagnostic()`: Computes cumulative/marginal variance and `recommended_n`
  (first N where next component's marginal gain < 5%).
- `output_r_squared_gate()`: Reconstructs output `z_pred = h_projected @ (W_out @ components_.T).T`
  and computes R2 against true readout. Returns `{"r_squared", "passed", "threshold"}`.
- `zscore_trajectories()`: Per-PC z-score to zero mean/unit std with std clipped to `>= 1e-8`.
  Returns `(z_scored, means, stds)` for invertible normalization.

**`src/pyro_dcm/rnn/__init__.py`** (updated): Added all 8 new exports.

### Task 2: Test Files

**`tests/test_fixed_point_analysis.py`** (12 tests):
- `TestFindFixedPoints`: FP finding on zero-bias tanh RNN (verifies `||dh/dt|| < 1e-4`), return
  type/shape checks.
- `TestComputeJacobianAtFp`: Shape `(H, H)`, analytical check `J = -I + W_rec` at h=0, b=0 for tanh.
- `TestClassifyStability`: Stable (-I), unstable (diag with +0.5), marginal (zero eigenvalue),
  Python type assertions.
- `TestDeduplicateFixedPoints`: Identical merge → 1, distinct → 3, empty, contractive RNN → 1.

**`tests/test_latent_extraction.py`** (14 tests, `@pytest.mark.latent`):
- `TestPcaReduce`: Projected shape, variance ratios >= 0 summing <= 1, components_ shape.
- `TestVarianceExplainedDiagnostic`: Monotone cumulative, marginal >= 0, recommended_n >= 1, low-rank
  data recommends N <= 3.
- `TestOutputRSquaredGate`: Passes with full-rank embedding N=3, fails with N=1 (orthogonal equal-
  variance embedding ensures PC1 ~33%), result key types.
- `TestZscoreTrajectories`: Zero mean/unit std, returned stats match originals, inverse recovers
  original, near-zero std handled without NaN.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 21-03-D1 | Module docstring before `from __future__ import annotations` | ruff E402 requires imports after module docstring; PEP 257 says docstring is first statement |
| 21-03-D2 | Output-R2 fail test uses orthogonal equal-variance embedding | Correlated factors let 1 PC capture >90% variance; orthogonal blocks ensure PC1 ~33% |
| 21-03-D3 | `classify_stability` parameter named `jacobian_matrix` | Avoids shadowing module-level `jacobian` import from torch.autograd.functional |
| 21-03-D4 | `extract_trajectories` metadata stored as dict under `__meta__` key | np.ndarray cannot hold heterogeneous scalar types; dict is correct container |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Module docstring position caused ruff E402 errors**

- **Found during:** Task 1 ruff check
- **Issue:** `from __future__ import annotations` placed first, then module docstring, then imports.
  ruff E402 flagged all module-level imports as "not at top of file" because the docstring between
  `__future__` and imports broke import block contiguity.
- **Fix:** Moved module docstring to be the very first statement in both modules (before `from
  __future__ import annotations`), following PEP 257 + ruff conventions.
- **Files modified:** `fixed_point_analysis.py`, `latent_extraction.py`
- **Commit:** 1e78d02 (applied before commit)

**2. [Rule 1 - Bug] test_gate_fails_insufficient_components passed when it should fail**

- **Found during:** Task 2 test run
- **Issue:** The original test construction mixed 3 factors with a mixing matrix, creating correlated
  factors. One PC captured >90% variance (R2=0.985 with N=1), so the gate passed incorrectly.
- **Fix:** Replaced with orthogonal basis embedding where each factor occupies a disjoint H/3-sized
  block of the H-dimensional space, guaranteeing equal variance per factor and PC1 capturing ~33%.
- **Files modified:** `tests/test_latent_extraction.py`
- **Commit:** 4b5f537 (applied before commit, ruff --fix also cleaned unused `pytest` import)

## Next Phase Readiness

Phase 21 Plan 04 (cluster training scripts) is unblocked. All utilities needed by the pipeline are
now in place:
- `extract_trajectories` ready for post-training trajectory collection
- `pca_reduce` + `variance_explained_diagnostic` + `output_r_squared_gate` ready for DIM analysis
- `zscore_trajectories` ready for DCM prior alignment

Phase 22 (PIPE-03) requires `compute_jacobian_at_fp` + `classify_stability` for linearization
quality diagnostics -- both are complete and tested.
