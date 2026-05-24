---
phase: 21-ctrnn-training
plan: "01"
subsystem: rnn
tags: [pytorch, rnn, continuous-time-rnn, euler, bptt, neurogym, scikit-learn]

requires:
  - phase: 20-latent-circuit
    provides: latent_circuit_dcm_model + LC_A_PRIOR_VARIANCE; confirms rnn/ is independent module

provides:
  - ContinuousTimeRNN(nn.Module): Euler CT-RNN with alpha=dt/tau, ReLU/tanh, batched forward
  - src/pyro_dcm/rnn/ package: importable rnn sub-package
  - [latent] optional dep group: neurogym>=2.3, scikit-learn>=1.3
  - latent pytest marker: gates neurogym-requiring tests
  - 10-test unit suite: shapes, alpha, activations, BPTT, h0, noise, weights

affects:
  - 21-02 (rnn_trainer.py): imports ContinuousTimeRNN
  - 21-03 (fixed_point_analysis.py): imports ContinuousTimeRNN + accesses W_rec/W_in/b attributes
  - 21-04 (latent_extraction.py): imports ContinuousTimeRNN for trajectory extraction
  - 22-latent-dcm-fit: consumes trained RNN weights and trajectories

tech-stack:
  added:
    - neurogym>=2.3 (optional, [latent] group)
    - scikit-learn>=1.3 (optional, [latent] group)
  patterns:
    - Euler discrete-time CT-RNN: h[t+1] = (1-alpha)*h[t] + alpha*f(W_rec@h + W_in@u + b)
    - alpha = dt/tau stored as plain float attribute (non-learnable, v0.6.0 design)
    - Optional training noise on hidden states (enabled only when self.training and noise_std > 0)
    - Unbatched (T, M_in) input promoted to (T, 1, M_in) inside forward()
    - W_rec/W_in/W_out/b as nn.Parameter with Gaussian init std ~ 1/sqrt(fan_in)

key-files:
  created:
    - src/pyro_dcm/rnn/__init__.py
    - src/pyro_dcm/rnn/continuous_time_rnn.py
    - tests/test_ctrnn.py
  modified:
    - pyproject.toml

key-decisions:
  - "alpha = dt/tau is a fixed float attribute, not a learnable parameter (v0.6.0 design)"
  - "Euler integration chosen over torchdiffeq.odeint: faster, deterministic, matches Langdon & Engel 2025"
  - "Formal citation pending Phase 25 (PUB-03); interim cite as Langdon & Engel (2025)"
  - "W_rec init std = 1/sqrt(H) (stable Gaussian spectrum); W_in/W_out init std = 1/sqrt(fan_in)"

patterns-established:
  - "CT-RNN Euler loop: collect h per step into list, torch.stack at end -> (T, B, H)"
  - "Noise injected after activation inside loop, gated on self.training and noise_std > 0"

duration: 5min
completed: "2026-05-25"
---

# Phase 21 Plan 01: CT-RNN Package Summary

**ContinuousTimeRNN(nn.Module) with Euler alpha=dt/tau, batched forward pass, and 10-test suite confirming BPTT gradient flow, shape contracts, and training noise isolation.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-24T22:27:23Z
- **Completed:** 2026-05-24T22:33:13Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- `src/pyro_dcm/rnn/` package created and importable as `from pyro_dcm.rnn import ContinuousTimeRNN`
- `ContinuousTimeRNN(nn.Module)` implements `h[t+1] = (1-alpha)*h[t] + alpha*f(W_rec@h + W_in@u + b)` with `alpha = dt/tau`, matching Langdon & Engel (2025) / `trainRNNbrain` Euler formulation
- Forward pass accepts batched `(T, B, M_in)` and unbatched `(T, M_in)` input; returns `(z, h_traj)` with shapes `(T, B, M_out)` and `(T, B, H)`
- Gradient flows through all four parameters (W_rec, W_in, W_out, b) — BPTT-ready
- `pyproject.toml` updated with `[latent]` optional dep group and `latent` pytest marker
- 10-test unit suite (no neurogym dependency) runs in 1.4s; all pass; ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create rnn package + ContinuousTimeRNN module + pyproject.toml updates** - `f3968ed` (feat)
2. **Task 2: Create CT-RNN unit tests** - `1dd9eec` (test)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `src/pyro_dcm/rnn/__init__.py` - Package init; exports `ContinuousTimeRNN`
- `src/pyro_dcm/rnn/continuous_time_rnn.py` - `ContinuousTimeRNN(nn.Module)`; 113 lines; Euler loop, ReLU/tanh, noise injection, NumPy docstring
- `tests/test_ctrnn.py` - 10 unit tests; 215 lines; pure PyTorch
- `pyproject.toml` - Added `[latent]` optional dep group and `latent` pytest marker

## Decisions Made

- **alpha = dt/tau is a plain float, not `nn.Parameter`.** Fixed for v0.6.0; learnable timescales deferred to future work. Makes alpha inspection easy (`rnn.alpha`) and avoids accidental gradient computation through it.
- **Euler integration, not torchdiffeq.** neurogym produces fixed-dt observations; adaptive stepping adds cost without benefit. Matches Langdon & Engel (2025) `trainRNNbrain` exactly.
- **Formal reference ID deferred to Phase 25 (PUB-03).** Docstring cites "Langdon & Engel (2025)" as interim placeholder; `REF-` ID will be assigned when the full reference is added to REFERENCES.md.
- **W_rec init std = 1/sqrt(H).** Ensures the initial weight spectrum is near the edge of chaos (largest singular value ~ 1), following standard RNN initialization practice.

## Deviations from Plan

None - plan executed exactly as written. Ruff flagged three style issues in the test file (import order, docstring capitalization, line length) during Task 2 verification; all fixed inline before commit.

## Issues Encountered

Ruff I001 (isort) flagged the import block in `tests/test_ctrnn.py` after manual editing. Auto-fixed with `ruff check --fix` before final commit. No functional impact.

## User Setup Required

None - no external service configuration required. `neurogym` and `scikit-learn` are optional dependencies installed only when running Phase 21 neurogym-requiring tests (`pip install pyro-dcm[latent]`).

## Next Phase Readiness

- `ContinuousTimeRNN` is fully importable and tested; Plan 21-02 (`rnn_trainer.py`) can import and train it immediately
- `W_rec`, `W_in`, `W_in`, `b`, `f` attributes are accessible directly for fixed-point analysis (Plan 21-03)
- `[latent]` dep group is in pyproject.toml; neurogym-requiring tests in Plan 21-02 can use `@pytest.mark.latent`
- No blockers for subsequent Phase 21 plans

---
*Phase: 21-ctrnn-training*
*Completed: 2026-05-25*
