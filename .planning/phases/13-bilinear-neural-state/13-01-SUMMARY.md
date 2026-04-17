---
phase: 13-bilinear-neural-state
plan: 01
subsystem: forward_models
tags: [pytorch, einsum, bilinear-dcm, BILIN-01, BILIN-02, BILIN-07]

# Dependency graph
requires:
  - phase: 01-neural-hemodynamic-forward-model
    provides: parameterize_A, NeuralStateEquation (colocation target)
provides:
  - parameterize_B(B_free, b_mask) -> (J, N, N) masked modulatory tensor (BILIN-01)
  - compute_effective_A(A, B, u_mod) -> A + einsum('j,jnm->nm', u_mod, B) (BILIN-02)
  - BILIN-07 source-half rewrite: neural_state.py module + class docstrings correctly label A+Cu as linear
affects: [13-02, 13-03, 13-04, 14-bilinear-simulators, 15-bilinear-pyro-model, 16-bilinear-recovery]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stacked (J, N, N) tensor representation for modulatory B matrices (context-locked; flows through odeint_adjoint cleanly)"
    - "Mask-based B parameterization (pure elementwise mult, no -exp or tanh), with N(0, 1.0) prior (D1) doing regularization"
    - "Default-zero-diagonal safety via caller's b_mask.fill_diagonal_(0.0); explicit opt-in raises DeprecationWarning per Pitfall B5"
    - "Empty-J short-circuit: compute_effective_A returns A bit-exactly when B.shape[0] == 0 (no einsum call, no allocation)"

key-files:
  created:
    - tests/test_bilinear_utils.py
  modified:
    - src/pyro_dcm/forward_models/neural_state.py
    - src/pyro_dcm/forward_models/__init__.py

key-decisions:
  - "Local `import warnings` inside parameterize_B keeps warnings import off the module-load hot path for non-warning callers"
  - "Empty-J branch is explicit (`if B.shape[0] == 0: return A`) even though einsum over zero-length axis would work; avoids allocating (N,N) zero tensor and keeps bit-exact semantics"
  - "parameterize_B raises ValueError on shape mismatch BEFORE checking ndim, so the more-specific error message is surfaced first"

patterns-established:
  - "BILIN-* utility signatures: pure tensor in, pure tensor out, no nn.Module coupling"
  - "Deprecation-path testing uses pytest.warns with match= + a positive assertion that the flagged entry was not silently modified (warning is informational, not corrective)"
  - "J=0 edge case covered in tests for every new function that indexes a modulator axis"

# Metrics
duration: ~15min
completed: 2026-04-17
---

# Phase 13 Plan 01: Bilinear Utilities (parameterize_B + compute_effective_A) Summary

**Two pure-tensor bilinear utilities colocated with `parameterize_A`, plus BILIN-07 source-docstring rewrite declaring `A + Cu` as the linear form (not bilinear).**

## Performance

- **Duration:** ~15 min (implementation + verification; broad-regression `pytest` subset ~3 min was the dominant cost)
- **Started:** 2026-04-17
- **Completed:** 2026-04-17
- **Tasks:** 4 (3 code + 1 commit task; the commit task collapsed into per-step commits)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- `parameterize_B(B_free, b_mask)` added per BILIN-01: masked modulatory-matrix factory, (J, N, N) in and out, pure elementwise multiplication, DeprecationWarning on any non-zero b_mask diagonal entry (Pitfall B5).
- `compute_effective_A(A, B, u_mod)` added per BILIN-02: `A + torch.einsum("j,jnm->nm", u_mod, B)`, with explicit J=0 short-circuit that returns `A` bit-exactly without einsum allocation.
- `neural_state.py` module docstring + `NeuralStateEquation` class summary line rewritten per BILIN-07 (source half): `A + Cu` is now correctly labeled the **linear** form; the historical "bilinear" name is contextualized as the opt-in v0.3.0 extension path via `B` / `u_mod` kwargs.
- Both utilities re-exported via `pyro_dcm.forward_models.__all__` (alphabetical, within the Phase 1 section).
- `tests/test_bilinear_utils.py` created with 9 passing tests covering: shape preservation, mask-zero semantics, default-diagonal pattern, DeprecationWarning emission + non-corrective behavior, J=0 roundtrip (no warning emitted), shape-mismatch ValueError, B=0 bit-exactness, hand-computed einsum correctness to `atol=1e-12`, and J=0 short-circuit.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement parameterize_B, compute_effective_A, rewrite neural_state.py docstrings (BILIN-07 source half)** - `9e7f993` (feat)
2. **Task 2: Export utilities from forward_models/__init__.py** - `df1f15a` (feat)
3. **Task 3: Write tests/test_bilinear_utils.py with 9 coverage tests** - `fcedc56` (test)

**Plan metadata:** pending at the time of writing this summary (final `docs(13-01): complete utilities plan` commit).

## Files Created/Modified

- `src/pyro_dcm/forward_models/neural_state.py` - Added `parameterize_B` and `compute_effective_A` top-level functions with NumPy-style docstrings + REF-001 Eq. 1 citations; rewrote module docstring (lines 1-16) and `NeuralStateEquation` class summary line to correctly label A+Cu as the linear form; added `import warnings` at top of file. `parameterize_A` body and `NeuralStateEquation` method bodies untouched.
- `src/pyro_dcm/forward_models/__init__.py` - Extended the `neural_state` import block to include `compute_effective_A` and `parameterize_B`; inserted both names alphabetically into the "Phase 1" section of `__all__`.
- `tests/test_bilinear_utils.py` (new) - 9 unit tests across `TestParameterizeB` (6) and `TestComputeEffectiveA` (3). Follows project conventions: `from __future__ import annotations`, NumPy docstrings, absolute imports, `torch.float64` throughout, expected-vs-actual messages on plain asserts.

## Decisions Made

None beyond the planner's-discretion items called out in `13-CONTEXT.md`. Implementation used `torch.einsum("j,jnm->nm", u_mod, B)` (as suggested by the plan) rather than `tensordot`/explicit broadcasting; the grader was readability. The empty-J case is handled via an explicit `if` branch (returning `A` with zero allocations) rather than relying on einsum-over-zero-length-axis semantics; this preserves bit-exact equality with the input `A` for the linear-equivalent path.

## Deviations from Plan

None - plan executed exactly as written.

Plan 13-04 (non-source BILIN-07 sites: `CLAUDE.md` tree + `PROJECT.md` Validated line) was already merged on this branch before Plan 13-01 ran. This is noted not as a deviation but as branch context: 13-01 and 13-04 have non-overlapping file scopes, so the merge order was orthogonal.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Plan 13-02 ready:** `NeuralStateEquation.derivatives` extension to accept optional `B` and `u_mod` kwargs can now import `compute_effective_A` from its own module. Linear-short-circuit bit-exact invariance test (context decision) can be written against the `B=None` path.
- **Plan 13-03 ready:** `CoupledDCMSystem` stability monitor will consume `compute_effective_A` and compute `max(Re(eig(A_eff)))` at the configured cadence.
- **Plan 13-04 status:** Already complete (committed as `c69f455` before this plan; BILIN-07 source half is now also complete via Task 1 here). The full BILIN-07 requirement is closed with plans 13-01 + 13-04 both landed.
- **No blockers.** Phase 13 acceptance criterion #5 (forward-model source no longer mislabels `A + Cu` as bilinear) is satisfied in source.

---
*Phase: 13-bilinear-neural-state*
*Completed: 2026-04-17*
