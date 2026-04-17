# Phase 13: Bilinear Neural State & Stability Monitor - Context

**Gathered:** 2026-04-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the forward-model layer of pyro_dcm to compute the Friston 2003 bilinear neural
state equation `dx/dt = (A + Σ_j u_j(t)·B_j)·x + C·u_driving(t)` end-to-end through
`NeuralStateEquation`, `CoupledDCMSystem`, and a stability monitor. Deliver the
supporting tensor utilities (`parameterize_B`, `compute_effective_A`) and a worst-case
stability test. Bit-exactly preserve linear behavior for existing callers when the new
bilinear arguments are omitted.

Out of scope for Phase 13: the Pyro generative model (Phase 15), the simulator and
stimulus utilities (Phase 14), the recovery benchmark (Phase 16), and any amortized-guide
updates (deferred to v0.3.1 per D5).

</domain>

<decisions>
## Implementation Decisions

### B-matrix representation

- **Stacked tensor `B: (J, N, N)`** is the representation that flows through the forward
  model and `CoupledDCMSystem`. All tensor leaves; clean for `torchdiffeq.odeint_adjoint`.
- Per-modulator site names (`B_free_0`, `B_free_1`, ...) are a Pyro-layer concern (Phase
  15); the forward model never sees site names and is indifferent to how the sampled
  tensor is constructed upstream.
- `B=None` or `J=0` (empty zeroth dim) is a supported input and triggers the linear
  short-circuit below.
- `compute_effective_A(A, B, u_mod) -> A_eff` signature: `A: (N,N)`, `B: (J,N,N)`,
  `u_mod: (J,)` -> `A_eff: (N,N)`. Implements `A + einsum('j,jnm->nm', u_mod, B)`.

### Modulator input routing

- **Single `PiecewiseConstantInput` with widened columns.** `input_fn(t) -> (M_driving +
  J_mod,)` returns concatenated driving + modulator values. `CoupledDCMSystem` slices
  internally: first `M_driving` columns go through `C @ u_drive`, remaining `J_mod`
  columns become `u_mod` for `A_eff`.
- Zero new stimulus primitives in Phase 13. `LinearInterpolatedInput` stays deferred
  (D2 -> v0.3.1 if needed).
- `CoupledDCMSystem.__init__` gains a single new int kwarg `n_driving_inputs: int | None
  = None`. When None (default), behavior is linear: all `input_fn(t)` columns go through
  `C`. When set, the split point is explicit.

### Linear short-circuit bit-exactness

- **`atol=1e-10` bit-exact gate.** When `B is None` or `B.shape[0] == 0`, the code path
  calls the original `A @ x + C @ u` directly -- not `A_eff = A + 0` followed by
  `A_eff @ x`. This guarantees zero numerical drift for v0.1/v0.2 callers.
- Dedicated regression test `tests/test_linear_invariance.py` asserts `atol=1e-10` on
  BOLD output between linear and `B=None` bilinear paths for at least 3 random seeds.

### Stability monitor behavior (BILIN-05)

- **Sink:** Python `logging` module. Named logger `pyro_dcm.stability` emits at
  `WARNING` level. Users silence via standard stdlib:
  `logging.getLogger('pyro_dcm.stability').setLevel(logging.ERROR)`.
- **Cadence:** every 10 ODE steps by default. Configurable via
  `CoupledDCMSystem(..., stability_check_every: int = 10)`. Set to `0` to disable
  entirely (no eigvals computed, zero overhead).
- **Disable switch:** constructor flag AND standard logging-level silencing. No env var.
- **Log-warn only; never raises** (D4). SVI divergent draws are expected and raising
  would corrupt gradient estimates.
- **Message format:** single-line `"Stability warning at t={t:.2f}s: max
  Re(eig(A_eff))={max_re:+.3f}; ||B·u_mod||_F={culprit_norm:.3f}"`.
- Trigger condition: `max(real(torch.linalg.eigvals(A_eff))) > 0` (strict, per D4).
- Monitor runs only when `B is not None and J > 0`; linear path is untouched.

### `parameterize_B` semantics

- **Location:** `src/pyro_dcm/forward_models/neural_state.py`, colocated with
  `parameterize_A`. Both are pure tensor utilities.
- **Signature:** `parameterize_B(B_free: Tensor (J,N,N), b_mask: Tensor (J,N,N)) -> Tensor (J,N,N)`.
  Operates on the full stacked tensor in one call.
- **Diagonal:** zeroed via mask by default. `b_mask[:, i, i] = 0` for all i in the default
  mask factory. If a caller explicitly supplies a `b_mask` with nonzero diagonal entries,
  `parameterize_B` emits a `DeprecationWarning` citing Pitfall B5 (the warning text names
  D4 stability risk).
- **Off-diagonal:** pure mask multiplication `B_free * b_mask`. No `-exp`, no `tanh`,
  no soft bounding. The `N(0, 1.0)` prior (D1) does the regularization work.
- **No `-exp` transform on diagonal** (even when user explicitly frees it). The mask
  path is the only legitimate way to enable self-modulators in v0.3.0; a future
  `parameterize_B_safe_diag` can be added in v0.4+ if self-modulator hypotheses become
  a primary use case.

### Doc-rename scope (BILIN-07)

- **Wide scope.** Fix all three stale sites in a single Phase 13 plan:
  1. `src/pyro_dcm/forward_models/neural_state.py` module docstring (lines 1-10) and
     `NeuralStateEquation` class docstring (around line 58): stop calling the `A + Cu`
     form "bilinear" -- it is linear. Reference the new true-bilinear code path for the
     Friston 2003 full form.
  2. `CLAUDE.md:67` directory-tree entry: `generative_models/` -> `models/` (correct
     the actual-path drift); add a one-line note pointing to v0.3.0 for the bilinear
     extension.
  3. `.planning/PROJECT.md:23` Validated-requirements line: rewrite "Bilinear neural
     state equation (dx/dt = Ax + Cu) with explicit A matrix -- v0.1.0" to "**Linear**
     neural state equation (dx/dt = Ax + Cu) with explicit A matrix -- v0.1.0". The
     true bilinear requirement will be added to Validated when v0.3.0 ships.
- **No test renames.** `test_neural_state.py` is neutral. Opportunistic renames of
  local helpers/variables with misleading "bilinear" naming are fine, but no formal
  task for test-file renames.
- **No full-repo grep audit.** Three known sites + opportunistic local rewording is
  sufficient. Future drift will be caught by Phase 15 and Phase 16 review.

### Claude's Discretion

- Internal tensor-op style for `compute_effective_A`: `einsum` vs `tensordot` vs
  explicit broadcasting. All equivalent; planner picks per readability.
- Exact formatting of stability-log message punctuation/order (as long as t, max_re,
  and culprit-norm are all present).
- Internal organization of `CoupledDCMSystem.forward` around the linear/bilinear
  branch: single `if` gate vs method-level dispatch vs `@functools.cached_property`
  flag -- planner's call.
- The `n_driving_inputs` kwarg default behavior ambiguity if `B is not None` but
  `n_driving_inputs is None`: planner decides whether to infer from `C.shape[1]` or
  raise a clear configuration error. Either is defensible.
- Whether to checkpoint a zero-copy buffer for `A_eff` to avoid per-step
  allocation in the ODE RHS.
- How the stability monitor's "every 10 steps" is counted for the non-fixed-step
  `dopri5` solver (adaptive step counts differ from rk4). Either count solver
  callbacks or wall-time sampling is acceptable.

</decisions>

<specifics>
## Specific Ideas

- The `pyro_dcm.stability` logger naming follows the project_utils convention
  `{package}.{subsystem}` -- consistent with any other library-emitted logs.
- The 3-sigma worst-case stability test (BILIN-06) draws `B = 3.0 * sqrt(1.0)` per
  free element (D1 prior variance = 1.0), sustained `u_mod = 1.0` for 500s. Pass
  criterion: no NaN in the full 5N state trajectory.
- Bit-exact regression in `test_linear_invariance.py` should cover at least: (a) a
  hand-crafted 2-region A/C, (b) a `make_random_stable_A(N=3, seed=42)` case, and
  (c) a `make_random_stable_A(N=5, seed=7)` case.

</specifics>

<deferred>
## Deferred Ideas

- `LinearInterpolatedInput` for smooth-ramp modulatory inputs -- v0.3.1 (SIM-06).
- `-exp` transform on B diagonal for guaranteed-stable self-modulators -- v0.4+ if
  self-modulator hypotheses become primary.
- Global env-var disable (`PYRO_DCM_STABILITY=off`) -- rejected as a reproducibility
  risk; not a deferred idea, intentionally not built.
- Full-repo audit for stale "bilinear" terminology -- Phase 15 and Phase 16 will
  each sweep their own file scope; no dedicated phase for this.
- `parameterize_B_safe_diag` with `-exp` transform -- v0.4+ candidate if needed.

</deferred>

---

*Phase: 13-bilinear-neural-state*
*Context gathered: 2026-04-17*
