---
phase: 13-bilinear-neural-state
plan: 02
subsystem: forward_models
tags: [pytorch, bilinear-dcm, BILIN-03, bit-exact, regression-test]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    plan: 01
    provides: compute_effective_A (called from bilinear branch of derivatives)
  - phase: 01-neural-hemodynamic-forward-model
    provides: NeuralStateEquation (class being extended)
  - phase: 02-task-dcm
    provides: make_random_stable_A (test-fixture generator at N=3 / N=5)
provides:
  - NeuralStateEquation.derivatives(x, u, *, B=None, u_mod=None) keyword-only bilinear extension (BILIN-03)
  - Literal linear short-circuit `return self.A @ x + self.C @ u` (grep-verified single site)
  - tests/test_linear_invariance.py (7 passing tests, rtol=0/atol=1e-10 gate on 3 CONTEXT-locked fixtures)
affects: [13-03, 14-bilinear-simulators, 15-bilinear-pyro-model, 16-bilinear-recovery]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Keyword-only bilinear-extension kwargs (`*` separator before `B` / `u_mod`) preserve every existing `derivatives(x, u)` call-site signature"
    - "Branch-on-None-first-then-empty-J pattern for the linear short-circuit: avoids `A + 0` allocation and guarantees byte-identical output to v0.2.0"
    - "Bit-exact regression testing via `torch.testing.assert_close(rtol=0, atol=1e-10)` against an explicit `_linear_reference(A, C, x, u)` helper"
    - "`torch.equal` (strict, no tolerance) for the no-kwarg-vs-B=None case because both paths execute the identical source line"

key-files:
  created:
    - tests/test_linear_invariance.py
  modified:
    - src/pyro_dcm/forward_models/neural_state.py

key-decisions:
  - "Literal short-circuit preserved verbatim: the grep `return self.A @ x + self.C @ u` matches exactly one line (the linear branch). No refactor into a helper, fused op, or cached expression."
  - "ValueError message includes the expected `u_mod` shape `(B.shape[0],)` per project convention (error messages must cite expected vs actual)."
  - "`test_bilinear_changes_output` uses `atol=1e-12` (tighter than the linear-invariance gate) because its expected values are exact arithmetic on small integers, and a loose gate would hide a genuine bilinear-branch regression."
  - "Parametrize decorator on `(N, seed)` pairs (not `product(N, seed)`) keeps the two random-fixture cases explicit rather than exploding to 4 fixture combinations."

patterns-established:
  - "BILIN-03 regression suite: 3 primary fixtures + 2 defensive cases + 2 bilinear-sanity cases = 7 tests covers the short-circuit gate from all angles"
  - "Explicit v0.2.0 reference helper (`_linear_reference`) gives the bit-exact comparison a named anchor that cannot be accidentally optimized to route through the new code path"

# Metrics
duration: ~14min
completed: 2026-04-17
---

# Phase 13 Plan 02: NeuralStateEquation Bilinear Extension + BILIN-03 Regression Test Summary

**`NeuralStateEquation.derivatives` gains keyword-only `B` and `u_mod` kwargs; when `B is None` or `B.shape[0] == 0` the method executes the literal `self.A @ x + self.C @ u` expression (bit-exact gate), and when `B` is non-empty it routes through `compute_effective_A`. Locked empirically by `tests/test_linear_invariance.py` at `rtol=0, atol=1e-10` on 3 CONTEXT-specified fixtures.**

## Performance

- **Duration:** ~14 min (implementation + verification; the 221s downstream regression sweep on `test_ode_integrator.py`/`test_task_simulator.py`/`test_task_dcm_model.py` was the dominant cost)
- **Started:** 2026-04-17
- **Completed:** 2026-04-17
- **Tasks:** 3 (1 source extension, 1 new test file, 1 metadata commit)
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments

- `NeuralStateEquation.derivatives` signature extended to `(self, x, u, *, B=None, u_mod=None)`; the leading positional args stay exactly `(x, u)`, so every existing call site (task-DCM model, simulator, ODE integrator tests) remains source- and behavior-compatible.
- Linear short-circuit gate implemented as a guard at the top of the method body: `if B is None or B.shape[0] == 0: return self.A @ x + self.C @ u`. The literal right-hand side appears exactly once in the file (grep-verified), and the branch-on-None comes first so that the hot path for v0.2.0 callers has zero attribute access on a possibly-`None` tensor.
- Bilinear path routes through `compute_effective_A(self.A, B, u_mod)` exactly as specified by 13-CONTEXT.md; shape validation is delegated to `compute_effective_A` (already tested by 13-01), and a clear `ValueError` is raised when `B` is non-empty but `u_mod` is `None` (the one error case that `compute_effective_A` cannot diagnose).
- `tests/test_linear_invariance.py` created with 7 passing tests:
  - `TestLinearInvariance` (5): hand-crafted 2-region at `atol=1e-10`, two `make_random_stable_A` parametrized fixtures (`N=3, seed=42` and `N=5, seed=7`) at `atol=1e-10`, an empty-J `(0, N, N)` fixture at `atol=1e-10`, and a strict `torch.equal` no-kwarg-vs-`B=None` case.
  - `TestBilinearPathSanity` (2): hand-computed bilinear output at `atol=1e-12`, and `ValueError` on `B` non-empty with `u_mod=None`.
- Docstring expansion on `derivatives`: added Parameters entries for `B` and `u_mod`, a note that the linear path evaluates the literal `self.A @ x + self.C @ u`, and a Raises entry for the `u_mod is None` case. The class summary line (landed by 13-01 Task 1) and module docstring are untouched.

## Task Commits

Each task committed atomically; no squashing:

1. **Task 1: Extend NeuralStateEquation.derivatives with bilinear path** - `55785de` (`feat(13-02): extend NeuralStateEquation.derivatives with bilinear path`)
2. **Task 2: Add tests/test_linear_invariance.py with atol=1e-10 gate** - `7289ff9` (`test(13-02): add test_linear_invariance.py with atol=1e-10 gate`)
3. **Task 3: Plan metadata commit** - pending at the time of this summary (final `docs(13-02): complete NeuralStateEquation extension plan` commit).

## Files Created/Modified

- `src/pyro_dcm/forward_models/neural_state.py` -- `NeuralStateEquation.derivatives` method signature expanded to accept keyword-only `B` and `u_mod`, body rewritten with linear short-circuit guard + bilinear branch. Class body and all other functions (`parameterize_A`, `parameterize_B`, `compute_effective_A`, module docstring) unchanged from the 13-01 state.
- `tests/test_linear_invariance.py` -- new file, 7 passing tests across `TestLinearInvariance` (5) and `TestBilinearPathSanity` (2). Follows project conventions: `from __future__ import annotations`, NumPy-style docstrings, absolute imports, `torch.float64` throughout, expected-vs-actual messages on plain asserts.

## Verification Evidence

- `pytest tests/test_linear_invariance.py -v` -> 7/7 passing.
- `pytest tests/test_neural_state.py tests/test_bilinear_utils.py tests/test_linear_invariance.py -v` -> 24/24 passing in 3.89s (pre-existing 17 tests unchanged + 7 new).
- `pytest tests/test_ode_integrator.py tests/test_task_simulator.py tests/test_task_dcm_model.py -x -q` -> 44/44 passing in 221.68s. No downstream regressions.
- `grep -n "return self.A @ x + self.C @ u" src/pyro_dcm/forward_models/neural_state.py` -> exactly one line (the linear short-circuit). No fused/refactored variants.
- `rg -n "linear form; bilinear B-matrix path" src/pyro_dcm/forward_models/neural_state.py` -> the 13-01 class-summary rename is intact (single match on the class summary line).
- `rg -n "[Bb]ilinear.*dx/dt = Ax \+ Cu" src/pyro_dcm/forward_models/neural_state.py` -> zero matches (the old misleading label is not present anywhere).
- Smoke test output:
  - `None path: tensor([ 0.9700, -0.0800], dtype=torch.float64)`
  - `empty-J path: tensor([ 0.9700, -0.0800], dtype=torch.float64)` (bit-identical to None path)
  - `Bilinear path: tensor([ 1.0700, -0.0800], dtype=torch.float64)` (extra `0.5 * 0.2 = 0.1` on `dx[0]` from `B[0, 0, 1] * x[1]`)

## Decisions Made

None beyond the planner's-discretion items already resolved in 13-CONTEXT.md. Implementation details:

- Branch ordering in the short-circuit is `B is None` first (identity check, no attribute access), then `B.shape[0] == 0` (explicit empty-J test). Either ordering would satisfy the bit-exact gate, but this ordering keeps the v0.2.0 hot path (where `B` is always `None`) at zero tensor-attribute accesses.
- The bilinear-sanity assertion (`test_bilinear_changes_output`) uses a tighter `atol=1e-12` than the linear-invariance gate because the expected values are exact arithmetic on small integers; loosening to `atol=1e-10` would have masked any future bilinear-branch regression that happened to stay within 10 ULPs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Plan 13-03 ready:** `CoupledDCMSystem` stability monitor can now assume `NeuralStateEquation.derivatives` accepts the bilinear kwargs; when `CoupledDCMSystem.forward` forwards `B` / `u_mod` through, the neural-state layer will route correctly. The linear short-circuit gate that 13-03 must preserve through the ODE-wrapper layer is now anchored by `tests/test_linear_invariance.py` (neural-state level); 13-03 adds an analogous BOLD-level bit-exact test.
- **Plans 14-16 ready:** the full forward-model bilinear path is now present at the neural-state layer. 14 (simulator) and 15 (Pyro model) can import `NeuralStateEquation` and pass `B` / `u_mod` through without any further forward_models edits in this plan.
- **BILIN-03 acceptance:** closed. Phase 13 acceptance criterion #3 (bit-exact linear invariance) has a structural guarantee (literal short-circuit) AND an empirical guarantee (`atol=1e-10` regression test on 3 fixtures). No v0.2.0 caller can see numerical drift from the extension.
- **No blockers.**

---
*Phase: 13-bilinear-neural-state*
*Completed: 2026-04-17*
