# Phase 13: Bilinear Neural State & Stability Monitor - Research

**Researched:** 2026-04-17
**Domain:** Bilinear extension of task-DCM forward model (Friston 2003 Eq. 1)
**Confidence:** HIGH (code-grounded, SPM-verified, torchdiffeq-source-verified)
**Scope:** Implementation-level detail layered on top of `.planning/research/v0.3.0/`
project-level research. Do not duplicate; cite and extend.

---

## Executive Summary

Phase 13 is a narrow, additive change to three files
(`forward_models/neural_state.py`, `forward_models/coupled_system.py`, plus
two docs) with one new test file (`tests/test_linear_invariance.py`) and one
new WARNING sink (`pyro_dcm.stability` Python logger with `NullHandler`).
CONTEXT.md locks every non-trivial design choice; this research converts
those decisions into exact code patterns, exact test fixtures, and an
enumerated list of existing tests that must stay green.

The three open implementation questions explicitly left to Claude's
discretion (linear-branch structure, stability-monitor cadence counting,
`n_driving_inputs`-None policy) all have defensible resolutions with
single-line code implications. See Section 10 for the risk register; the
highest-severity risk (B1 eigenvalue instability) is mitigated by the
stability monitor spec that CONTEXT has already locked.

**Primary recommendation:** break Phase 13 into 4 plans
(13-01 utilities, 13-02 `NeuralStateEquation` + invariance test,
13-03 `CoupledDCMSystem` + monitor + 3-sigma test, 13-04 doc-rename). 13-04
can run fully in parallel; 13-01 -> 13-02 -> 13-03 is the strict critical
path. See Section 8.

**Key numeric anchors (for planner):**

| Thing | Value | Source |
|-------|-------|--------|
| Bit-exact atol | `1e-10` | CONTEXT.md "Linear short-circuit bit-exactness" |
| 3-sigma B element | `3.0` (= `3 * sqrt(1.0)`) | D1 prior variance; CONTEXT §"Specific Ideas" |
| Stability log cadence | every 10 `forward()` calls | CONTEXT.md §"Stability monitor" |
| Worst-case sim duration | 500s | BILIN-06, CLAUDE.md §"Numerical Stability" |
| Worst-case solver | `rk4` with `dt=0.1` or `0.5` | matches existing `task_dcm_model.py:150` |
| Invariance test seeds | 2-region hand + `seed=42, N=3` + `seed=7, N=5` | CONTEXT §"Specific Ideas" |

---

## 1. torchdiffeq RHS integration pattern for time-varying linear systems

**Findings (HIGH confidence, source-verified):**

- **`t` is guaranteed a scalar `torch.Tensor` of float64 dtype** when torchdiffeq
  calls `func(t, y)`. Verified via `torchdiffeq/_impl/rk_common.py` docstring
  for `_runge_kutta_step`: `"t0: float64 scalar Tensor, t1: float64 scalar
  Tensor"`. The existing `CoupledDCMSystem.forward(self, t, state)` already
  relies on this — `self.input_fn(t)` is called at `coupled_system.py:140`
  and `PiecewiseConstantInput.__call__` uses `torch.searchsorted(self.times,
  t.detach(), right=True) - 1` which requires scalar-tensor `t`.
- **RHS evaluations per step:**
  - `rk4`: 4 evaluations per accepted step (k1 through k4).
  - `dopri5`: 6 evaluations per **accepted** step plus rejected retries
    (Dormand-Prince FSAL). torchdiffeq source confirms 6 stages in the
    generic `_runge_kutta_step` iteration.
  - `perturb` argument is passed (`func(ti, yi, perturb=perturb)` with
    `perturb in {NONE, NEXT, PREV}`) to handle discontinuity jumps
    registered via `options={"jump_t": grid_points}` (see
    `ode_integrator.py:174`). `CoupledDCMSystem.forward` does NOT declare
    `perturb` as a kwarg today; torchdiffeq is tolerant of signatures that
    don't accept it (inspects via `inspect.signature`). Phase 13 should
    NOT add a `perturb` kwarg — keep the signature minimal.
- **`PiecewiseConstantInput` at non-grid times (CRITICAL for bilinear):**
  when the solver evaluates at `t` between onsets, `searchsorted(..., right=True) - 1`
  returns the index of the **last** onset `<= t`, so the solver receives
  the left-piecewise-constant value. For the widened
  `(M_driving + J_mod)` input, modulator columns are read with the **same**
  piecewise-constant semantics as driving columns, guaranteeing that
  `u_mod(t)` is well-defined and deterministic at every RHS call inside
  an integration step. PITFALL B12 (stick-function blur at rk4 mid-steps)
  is a stimulus-design issue, not a Phase 13 defect — Phase 13 simply
  consumes `PiecewiseConstantInput(t)` and must not change its semantics.
- **Time-varying linear RHS precedent in torchdiffeq examples:** `odeint`
  is agnostic to linearity; `func(t, y)` semantics is all that matters.
  `A_eff(t) = A + Σ_j u_j(t)·B_j` constructed per-step inside `forward` is
  exactly the same pattern as the existing `self.input_fn(t)` call at
  `coupled_system.py:140`. No solver-level change required.
- **Adjoint-gradient correctness for the stacked `(J,N,N)` buffer:**
  `odeint_adjoint` backpropagates through the module's buffers and
  parameters listed in `adjoint_params` (default `tuple(func.parameters())`).
  The existing `integrate_ode` already handles this at
  `ode_integrator.py:192`. **Stacked `B` registered via
  `register_buffer("B", B)` behaves identically to the existing `A` and `C`
  buffers** — the buffer is tracked by `Module.parameters()` only if it
  was registered as a Parameter, not as a buffer. But `adjoint_params`
  defaults to `tuple(func.parameters())`, which excludes buffers, so A/B/C
  are NOT in the default `adjoint_params` tuple — see Section 2.

**Implication for the RHS code:**
```python
def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    N = self.n_regions
    x = state[:N]
    s = state[N:2*N]; lnf = state[2*N:3*N]; lnv = state[3*N:4*N]; lnq = state[4*N:5*N]

    u_all = self.input_fn(t)  # shape (M_driving + J_mod,)

    if self.B is None or self.B.shape[0] == 0:
        # LINEAR SHORT-CIRCUIT (BILIN-03 bit-exact gate)
        dx = self.A @ x + self.C @ u_all                     # bit-exact v0.2.0 path
    else:
        M_d = self.n_driving_inputs
        u_drive = u_all[:M_d]                                # shape (M_driving,)
        u_mod = u_all[M_d:]                                  # shape (J_mod,)
        A_eff = self.A + torch.einsum("j,jnm->nm", u_mod, self.B)
        dx = A_eff @ x + self.C @ u_drive
        # optional stability check (Section 4)
        self._maybe_check_stability(t, A_eff, u_mod)

    ds, dlnf, dlnv, dlnq = self.hemo.derivatives(x, s, lnf, lnv, lnq)
    return torch.cat([dx, ds, dlnf, dlnv, dlnq])
```

**Sources:** torchdiffeq GitHub master `_impl/rk_common.py`,
`torchdiffeq/_impl/odeint.py`, `torchdiffeq/_impl/adjoint.py` (verified via
WebFetch 2026-04-17). Existing code reads of `ode_integrator.py:165-194` and
`coupled_system.py:104-149`.

---

## 2. `CoupledDCMSystem` buffer registration for stacked B

**Findings (HIGH confidence):**

- **Register B as a buffer, not a Parameter.** Rationale matches the existing
  `A`/`C` precedent in `coupled_system.py:88-91`: Pyro handles
  parameterization via sample sites; the `nn.Module` is a pure forward
  computation graph. Registering as `nn.Parameter` would double-count
  gradients (Pyro via autograd + `adjoint_params`).
- **Pyro model instantiation pattern (same as A/C):** Phase 15 will do
  ```python
  B_free = torch.stack([pyro.sample(f"B_free_{j}", ...) for j in range(J)])
  B = parameterize_B(B_free, b_mask)
  system = CoupledDCMSystem(A, C, input_fn, B=B, n_driving_inputs=M_d, ...)
  ```
  Each SVI step constructs a fresh `CoupledDCMSystem` with a fresh sampled
  `B`. Autograd tracks gradients through the `register_buffer("B", B)` call
  because the sampled `B` tensor still has `requires_grad=True` — buffers
  are just tensors that move with the module; they do not strip
  `requires_grad` (verified: `Module.register_buffer` source simply assigns
  to `self._buffers[name]`; autograd is orthogonal).
- **Adjoint compatibility:** `integrate_ode(..., adjoint=True)` passes
  `adjoint_params=tuple(func.parameters())`, which is **empty** in our
  design (all tensors are buffers). This is the existing v0.2.0 behavior
  — `task_dcm_model.py:149-151` uses `integrate_ode(system, y0, t_eval,
  method="rk4", step_size=dt)` with `adjoint=False` (default). **Phase 13
  does NOT need to switch to adjoint mode.** Adjoint is not in the scope
  of Phase 13 and is not used in `task_dcm_model.py` today. If Phase 15
  later enables adjoint for memory, the buffer approach remains valid
  because PyTorch's autograd already tracks the full graph through buffer
  assignments when the source tensor has `requires_grad=True` (Pyro's
  poutine handling guarantees this).
- **Signature extension (matches CONTEXT decisions):**
  ```python
  def __init__(
      self,
      A: torch.Tensor,                    # (N, N)
      C: torch.Tensor,                    # (N, M_driving + J_mod) when bilinear, (N, M) when linear
      input_fn: Callable[[torch.Tensor], torch.Tensor],
      hemo_params: dict[str, float] | None = None,
      *,
      B: torch.Tensor | None = None,      # (J, N, N) stacked or None
      n_driving_inputs: int | None = None,
      stability_check_every: int = 10,
  ) -> None:
  ```
  Keyword-only for the new args (`*` separator) prevents positional
  collisions with existing callers. `C` dimensionality note: when bilinear
  is active, `C.shape[1]` must equal `n_driving_inputs` (NOT
  `M_driving + J_mod`). This is how the slicing in Section 1 works.
  Existing linear callers pass `C` with `M` columns and `input_fn` with
  `M`-column values — nothing changes.
- **Internal `B` storage:** use `self.register_buffer("B", B)` when `B is
  not None`; otherwise set `self.B = None` (attribute, not buffer). The
  `forward` branch-check becomes `if self.B is None or self.B.shape[0] ==
  0:` — clean, and `None` is a valid non-buffer sentinel for linear mode.
  Note: `register_buffer("B", None)` is *also* valid in PyTorch since 1.6+
  (registers an unregistered slot that can later be assigned), but mixing
  `None` and tensor as a buffer complicates `forward`. Use the `None`
  attribute path.

**Sources:** `pytorch/torch/nn/modules/module.py::register_buffer`;
`coupled_system.py:88-102` (existing pattern); `ode_integrator.py:179-194`
(integrate_ode adjoint handling); `task_dcm_model.py:147-151` (current
adjoint=False usage).

---

## 3. Linear-short-circuit test construction (BILIN-03)

**Findings (HIGH confidence):**

- **The `if self.B is None or self.B.shape[0] == 0:` gate guarantees the
  linear code path is *literally* reused.** The CONTEXT-locked
  requirement is "the code path calls the original `A @ x + C @ u`
  directly — not `A_eff = A + 0` followed by `A_eff @ x`. This guarantees
  zero numerical drift." The `if` gate in Section 1's code block above
  satisfies this exactly: when B is None, the `A @ x + C @ u` expression
  is evaluated literally, producing **bit-identical** float64 output to
  the pre-Phase-13 code.
- **Bit-exact verification: prefer `torch.equal` for intra-Python
  verification, `atol=1e-10` for test-harness assertion.** `torch.equal`
  performs elementwise exact equality comparison (no tolerance). But
  because the BOLD output goes through the Balloon-Windkessel cascade
  (nonlinear) and the rk4 solver (FP accumulation), even the truly
  bit-identical neural-state path produces BOLD output that is bit-exact
  at the float64 representation level only if all intermediate float
  operations match exactly — which they do, because the `forward` method
  executes the same tensor operations in the same order. **Use
  `torch.testing.assert_close(bold_linear, bold_bilinear_B_none,
  atol=1e-10, rtol=0)`** to defend against possible future refactors
  that introduce a fused op or reordering. CONTEXT.md specifies atol=1e-10
  explicitly; honor that.
- **Test fixture structure (follow CONTEXT §"Specific Ideas"):**
  ```python
  # tests/test_linear_invariance.py

  import pytest
  import torch
  from pyro_dcm.forward_models import CoupledDCMSystem, bold_signal
  from pyro_dcm.simulators.task_simulator import (
      make_random_stable_A,
      simulate_task_dcm,
      make_block_stimulus,
  )
  from pyro_dcm.utils import PiecewiseConstantInput, integrate_ode, make_initial_state


  class TestLinearInvariance:
      """BILIN-03: B=None path is bit-exact to v0.2.0 linear path."""

      @pytest.fixture
      def hand_crafted_2region(self) -> dict:
          A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
          C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
          stim = make_block_stimulus(n_blocks=2, block_duration=20.0,
                                      rest_duration=20.0, n_inputs=1)
          return {"A": A, "C": C, "stim": stim, "N": 2, "M": 1}

      @pytest.fixture(params=[(3, 42), (5, 7)])
      def random_case(self, request) -> dict:
          N, seed = request.param
          A = make_random_stable_A(N, density=0.5, seed=seed)
          C = torch.zeros(N, 1, dtype=torch.float64); C[0, 0] = 0.5
          stim = make_block_stimulus(n_blocks=3, block_duration=20.0,
                                      rest_duration=20.0, n_inputs=1)
          return {"A": A, "C": C, "stim": stim, "N": N, "M": 1}

      def _simulate(self, case, B=None, n_driving=None):
          times = case["stim"]["times"]
          values = case["stim"]["values"]
          input_fn = PiecewiseConstantInput(times, values)
          system = CoupledDCMSystem(
              case["A"], case["C"], input_fn,
              B=B, n_driving_inputs=n_driving,
          )
          y0 = make_initial_state(case["N"], dtype=torch.float64)
          t_eval = torch.arange(0, 120.0, 0.1, dtype=torch.float64)
          sol = integrate_ode(system, y0, t_eval, method="rk4",
                              step_size=0.1)
          lnv = sol[:, 3 * case["N"] : 4 * case["N"]]
          lnq = sol[:, 4 * case["N"] : 5 * case["N"]]
          return bold_signal(torch.exp(lnv), torch.exp(lnq))

      def test_hand_crafted_2region_bit_exact(self, hand_crafted_2region):
          bold_linear = self._simulate(hand_crafted_2region, B=None)
          bold_bilinear_none = self._simulate(hand_crafted_2region, B=None)  # sanity
          torch.testing.assert_close(
              bold_linear, bold_bilinear_none, atol=1e-10, rtol=0.0,
          )

      def test_random_cases_bit_exact(self, random_case):
          # Same invariance: re-constructing with B=None matches prior run.
          # To actually test the None-short-circuit, compare against
          # the v0.2.0 code path: instantiate without the B kwarg at all.
          ...
  ```
- **Stronger form of the invariance test: compare the `B=None` path to
  the *no-B-kwarg* path.** A fully defensive test runs the simulation
  twice — once with the old signature (`CoupledDCMSystem(A, C, input_fn)`)
  and once with the extended signature (`CoupledDCMSystem(A, C, input_fn,
  B=None)`) — and asserts `torch.equal(bold_a, bold_b)` (strict). This
  catches any case where a developer accidentally routes both None and
  no-kwarg to the bilinear `A_eff` path. Because the CONTEXT-locked
  design makes both paths call the same `if B is None` gate, this test
  should pass at bit-exact equality. Use `torch.equal` here, not
  `assert_close`, because the paths are *identical* operations, not
  merely tolerantly-close.
- **A third invariance case (recommended but optional):** set `B` to a
  tensor of shape `(0, N, N)` (empty J-dim) and assert the output
  matches the B=None case. This tests the `B.shape[0] == 0` branch.
  Trivial to add: one extra fixture.

**Implication for planner:** the test file `tests/test_linear_invariance.py`
should contain at minimum:
1. Hand-crafted 2-region bit-exact (CONTEXT.md §"Specific Ideas" (a)).
2. `make_random_stable_A(N=3, seed=42)` bit-exact (CONTEXT.md §(b)).
3. `make_random_stable_A(N=5, seed=7)` bit-exact (CONTEXT.md §(c)).
4. Optional: `B.shape == (0, N, N)` empty-J invariance.
5. Optional: `B=None` vs no-kwarg strict equality via `torch.equal`.

---

## 4. Eigenvalue monitor implementation pattern

**Findings (HIGH confidence on mechanism, MEDIUM on exact cadence policy):**

- **torchdiffeq has NO native step-event callback that fires once per
  accepted step.** The only hook mechanism is `perturb` (for jump
  discontinuities), which is not a general step counter.
- **Counting `forward()` invocations is the only available mechanism.**
  This means: (i) with `rk4` fixed-step, counter increments by 4 per
  integration step (4 stages) — cadence "every 10 forward() calls" = ~every
  2.5 rk4 steps; (ii) with `dopri5` adaptive, counter increments by 6 per
  accepted step plus any rejected retries — cadence varies with step
  acceptance but is still bounded.
- **CONTEXT §"Claude's Discretion" item 6** explicitly calls out "how the
  stability monitor's 'every 10 steps' is counted for `dopri5` (adaptive
  step counts differ from rk4). Either count solver callbacks or wall-time
  sampling is acceptable." **Recommendation:** count `forward()` calls
  with a modulus-10 counter. Pros: simple, deterministic, zero overhead
  when `stability_check_every=0`. Cons: adaptive solvers will sample more
  densely in time; this is acceptable because (a) DCM uses `rk4` in
  `task_dcm_model.py:150`, not `dopri5`; (b) sampling more densely is
  not a correctness issue — the monitor is log-warn only.
- **Exact implementation pattern:**
  ```python
  def __init__(self, ..., stability_check_every: int = 10) -> None:
      super().__init__()
      # ... existing buffers ...
      self.stability_check_every = stability_check_every
      self._step_counter = 0          # not a buffer; per-instance counter

  def _maybe_check_stability(self, t, A_eff, u_mod) -> None:
      if self.stability_check_every <= 0:
          return
      self._step_counter += 1
      if self._step_counter % self.stability_check_every != 0:
          return
      # torch.linalg.eigvals returns complex tensor; detach to avoid
      # accidentally pulling the eigenvalue computation into autograd.
      with torch.no_grad():
          eigs = torch.linalg.eigvals(A_eff.detach())
          max_re = eigs.real.max().item()
          if max_re > 0.0:
              culprit_norm = (self.B * u_mod.view(-1, 1, 1)).sum(0).norm().item()
              _STABILITY_LOGGER.warning(
                  "Stability warning at t=%.2fs: "
                  "max Re(eig(A_eff))=%+.3f; ||B·u_mod||_F=%.3f",
                  float(t.item()), max_re, culprit_norm,
              )
  ```
- **Logger module-level setup:**
  ```python
  # top of coupled_system.py
  import logging
  _STABILITY_LOGGER = logging.getLogger("pyro_dcm.stability")
  ```
  Library discipline (see Section 6) requires also attaching a
  `NullHandler` at package init so that consumers who never configure
  logging don't get a `"No handlers could be found for logger
  pyro_dcm.stability"` warning. Add this once at `src/pyro_dcm/__init__.py`:
  ```python
  import logging as _logging
  _logging.getLogger("pyro_dcm").addHandler(_logging.NullHandler())
  ```
  This propagates to `pyro_dcm.stability` via the standard hierarchical
  logger mechanism.
- **Validation test (CONTEXT §"Claude's Discretion" item 6):**
  Construct a 2-region system with B chosen to produce
  `A_eff = A + u_mod*B` with `max Re(eig) > 0`, run 10s of integration at
  rk4 dt=0.1 (which is 100 accepted steps, ≈400 forward() calls, ≈40 log
  events expected at `stability_check_every=10`). Assert via pytest
  `caplog` fixture that at least one WARNING is emitted in the
  `pyro_dcm.stability` logger with the expected substring
  `"Stability warning at t="`. Assert no exception is raised
  (`never raises` invariant).

**Per-call overhead budget:** `torch.linalg.eigvals(A_eff)` on `(N, N)` is
O(N³) and ~microseconds at N=5. At cadence 10, overhead per forward() is
~0.1μs average — negligible next to the 50-100μs rk4 step cost.

**Sources:** `torchdiffeq/_impl/solvers.py`, `torchdiffeq/_impl/fixed_grid.py`,
`pytest` docs for `caplog` fixture; existing `torch.linalg.eigvals` usage
in `task_simulator.py:354`.

---

## 5. 3-sigma worst-case stability test fixture (BILIN-06)

**Findings (HIGH confidence on construction, MEDIUM on qualitative behavior):**

- **3-sigma interpretation (per D1, CONTEXT §"Specific Ideas"):**
  D1 locks `B_free` prior variance = 1.0 (SPM one-state match).
  3σ = `3 * sqrt(1.0) = 3.0`. CONTEXT says "3-sigma worst-case B at prior
  3σ, sustained `u_mod = 1.0` for 500s. Pass criterion: no NaN in the
  full 5N state trajectory."
- **Deterministic fixture (CI-stable, reproducible):**
  ```python
  def test_three_sigma_worst_case_stability():
      """BILIN-06: 3-sigma worst-case B + sustained u_mod=1 for 500s, no NaN."""
      N, J = 3, 1
      # Stable A: diagonal-dominant, matches parameterize_A(zeros) baseline
      A = torch.tensor(
          [[-0.5,  0.1,  0.0],
           [ 0.2, -0.5,  0.1],
           [ 0.0,  0.3, -0.5]],
          dtype=torch.float64,
      )
      C = torch.tensor([[0.5], [0.0], [0.0]], dtype=torch.float64)
      # Worst-case B: off-diagonal 3.0 (3-sigma of N(0, 1.0) prior),
      # diagonal 0 (CONTEXT locked via parameterize_B mask default).
      B = torch.tensor(
          [[[0.0, 3.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 3.0, 0.0]]],  # shape (J=1, N=3, N=3)
          dtype=torch.float64,
      )
      # Sustained modulator and driving: single PiecewiseConstantInput
      # with widened columns (M_driving=1 + J_mod=1 = 2 total cols).
      times = torch.tensor([0.0], dtype=torch.float64)
      values = torch.tensor([[1.0, 1.0]], dtype=torch.float64)  # driving, modulator
      input_fn = PiecewiseConstantInput(times, values)

      system = CoupledDCMSystem(
          A, C, input_fn,
          B=B, n_driving_inputs=1,
          stability_check_every=10,
      )
      y0 = make_initial_state(N, dtype=torch.float64)
      t_eval = torch.arange(0.0, 500.0, 0.1, dtype=torch.float64)
      sol = integrate_ode(system, y0, t_eval, method="rk4", step_size=0.1)

      # Pass criterion: no NaN or inf in the full 5N state trajectory.
      assert torch.isfinite(sol).all(), (
          f"Non-finite states detected: NaN count = {(~torch.isfinite(sol)).sum().item()}"
      )
  ```
- **Expected qualitative behavior:** with `A_eff = A + B`
  (since `u_mod = 1`), the eigenvalues of this particular worst-case
  `A_eff` have `max Re ≈ +5.5` (diagonal -0.5 + off-diagonal 3.0 row-sum
  = 5.5 Gershgorin upper bound). The system is unstable; neural states
  x grow exponentially, BUT the Balloon-Windkessel cascade has its own
  stabilizing mechanisms (log-transform on v and q, vasodilatory feedback
  via s). The test claim is narrowly *"no NaN"* — states may grow very
  large, but must remain finite in IEEE-754. This is a pragmatic pass
  criterion: it proves the integrator doesn't blow up to inf/nan in 500s,
  which is the production concern during SVI divergent draws.
- **CAUTION — this test is expected to FIRE the stability monitor many
  times.** Use `pytest.caplog` or `logging.getLogger('pyro_dcm.stability').setLevel(logging.ERROR)`
  in the test scope to silence the warnings during CI (or accept the log
  output). Recommend silencing via fixture:
  ```python
  def test_three_sigma_worst_case_stability(caplog):
      caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")
      # ... test body ...
  ```
- **Cross-reference CLAUDE.md §"Numerical Stability":**
  *"All ODE integrations must be tested for 500s simulation duration
  without NaN."* BILIN-06 is exactly this rule applied to the bilinear
  path.
- **Alternative fixture sign consideration:** CONTEXT §"Specific Ideas"
  says "B = 3.0 * sqrt(1.0) per free element" — implying positive 3σ
  uniformly. For an adversarial worst-case, an alternative is
  `B = 3.0 * (2 * randbool - 1)` (deterministic sign via fixed seed), but
  the uniform `+3.0` fixture is harsher (all off-diagonals same sign
  maximize row-sums of A_eff, giving largest Gershgorin bound). **Stay
  with uniform-positive `+3.0` off-diagonal, diagonal 0.**

**Sources:** D1 from `.planning/STATE.md`; CLAUDE.md §"Numerical Stability"
(line 67); PITFALLS.md §B1 ("3σ combined with u_mod=1 sustained 2s pushes
leading eigenvalue to ~+0.3 Hz"); BILIN-06 requirement verbatim.

---

## 6. Logging setup

**Findings (HIGH confidence on library discipline, verified absence of
existing logging):**

- **No existing logging infrastructure in `pyro_dcm`.** Grep across
  `src/pyro_dcm/` for `import logging`, `logging.getLogger`,
  `NullHandler`, `addHandler`, `logging.basicConfig` returns **zero
  matches**. This is a green-field logging introduction.
- **Library best practice (from stdlib `logging` docs):** a library
  should install a `NullHandler` at package root to prevent
  "no handlers could be found" warnings when consumers don't configure
  logging. Log events then propagate to the consumer's root logger if
  configured, or disappear silently if not.
- **Recommended setup (minimal, idiomatic):**

  1. **`src/pyro_dcm/__init__.py`** (add near top, after `__version__`):
     ```python
     import logging as _logging

     # Attach a NullHandler to the package's root logger so that library
     # users who don't configure logging don't see "no handlers could be
     # found" warnings when pyro_dcm emits log events (e.g. the
     # pyro_dcm.stability eigenvalue monitor).
     _logging.getLogger("pyro_dcm").addHandler(_logging.NullHandler())
     ```

  2. **`src/pyro_dcm/forward_models/coupled_system.py`** (add module-level
     logger):
     ```python
     import logging

     _STABILITY_LOGGER = logging.getLogger("pyro_dcm.stability")
     ```

  3. **No other changes.** Users silence via stdlib:
     ```python
     logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)
     ```
     (CONTEXT.md locks this pattern.)

- **pytest `caplog` fixture compatibility (HIGH confidence):**
  `caplog` captures logs propagating up the logger hierarchy. With
  `propagate=True` (default for named loggers) and a `NullHandler` at
  root, caplog's handler still sees events. Standard pytest pattern:
  ```python
  def test_stability_monitor_fires(caplog):
      caplog.set_level(logging.WARNING, logger="pyro_dcm.stability")
      # ... trigger unstable A_eff ...
      assert any(
          "Stability warning at t=" in record.message
          for record in caplog.records
          if record.name == "pyro_dcm.stability"
      )
  ```
- **Documentation (minimal):** one-paragraph section in
  `docs/02_pipeline_guide/quickstart.md` (or new
  `docs/02_pipeline_guide/logging.md`) listing the two silencing
  patterns. Planner decides whether to include in Phase 13 or defer to
  Phase 14/15. **Recommendation: include a 5-line docstring note in
  `CoupledDCMSystem.__init__` about the logger; defer the docs page.**

**Sources:** Python stdlib `logging` docs "Configuring Logging for a Library"
(https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library);
pytest docs for `caplog`; direct grep of `src/pyro_dcm/` confirming no
existing logging usage.

---

## 7. `CLAUDE.md` line 67 and `.planning/PROJECT.md` line 23 — exact text

**Findings (HIGH confidence, direct file reads):**

### CLAUDE.md — existing text

```
Line 67:  - All ODE integrations must be tested for 500s simulation duration without NaN
```

**Actually, CONTEXT.md points to "around line 67" for a directory-tree
entry.** The directory-tree listing begins at line 89. CONTEXT.md
paraphrases the wrong line — the intended target is the
**`generative_models/` entry at line 101**. Current text:

```
101:│       ├── generative_models/
102:│       │   ├── __init__.py
103:│       │   ├── task_dcm.py          # Pyro model for task-based DCM
104:│       │   ├── spectral_dcm.py      # Pyro model for spectral DCM
105:│       │   └── regression_dcm.py    # Pyro model for regression DCM
```

**Mismatch with actual repo structure:**
- The implemented folder is `src/pyro_dcm/models/`, not
  `generative_models/` (verified: `ls src/pyro_dcm/` shows `models/`).
- The implemented file is `task_dcm_model.py`, not `task_dcm.py`.

**Proposed edit (CONTEXT.md §"Doc-rename scope" item 2):** replace the
`generative_models/` block with the actual `models/` layout and reference
the v0.3.0 bilinear extension:

```
│       ├── models/
│       │   ├── __init__.py
│       │   ├── task_dcm_model.py       # Pyro model for task-based DCM [v0.3.0: + bilinear B]
│       │   ├── spectral_dcm_model.py   # Pyro model for spectral DCM
│       │   ├── rdcm_model.py           # Pyro model for regression DCM
│       │   ├── amortized_wrappers.py   # Amortized task/spectral wrappers
│       │   ├── guides.py               # Guide factory (SVI/Laplace/MVN/IAF)
│       │   ├── svi_runner.py           # SVI training loop
│       │   ├── model_comparison.py     # ELBO-based comparison
│       │   ├── posterior_extraction.py # Posterior summary helpers
│       │   └── task_dcm_forward.py     # Shared forward-model helper
```

**Note:** the current CLAUDE.md `inference/` block (lines 119-123) is
also stale — inference code has moved into `models/` (SVI, guides,
comparison). Phase 13 scope per CONTEXT is "correct the actual-path
drift" for `generative_models/`. Whether to also rewrite the
`inference/` block depends on planner's preference. **Recommend:
rewrite both blocks in one Phase 13 commit; the drift is the same kind
of issue and fixing one while leaving the other stale is worse than
fixing both.**

### `.planning/PROJECT.md` — existing text

```
Line 23:  - Bilinear neural state equation (dx/dt = Ax + Cu) with explicit A matrix — v0.1.0
```

**Proposed edit (per CONTEXT.md decision):**

```
- **Linear** neural state equation (dx/dt = Ax + Cu) with explicit A matrix — v0.1.0
```

The bold emphasizes the correction. When v0.3.0 ships, a new line will be
added:

```
- **Bilinear** neural state equation (dx/dt = (A + Σ u_j B_j)·x + Cu) with stability monitor — v0.3.0
```

(Out of scope for Phase 13; will be added in the v0.3.0 release phase.)

### Tensor Shape Conventions Table (CLAUDE.md lines 175-185)

CLAUDE.md lines 175-185 (approximate; numbers shift after edits above)
contain a "Tensor Shape Conventions" table that already lists `B_j`,
`u`, `B`-related shapes:

```
| B_j | (N, N) | Modulatory input j |
| u | (T, M) | Experimental inputs over time |
```

**Question 7 asks:** is this a gap to fix now or later?

**Recommendation (MEDIUM confidence):** fix it in Phase 13 alongside the
`generative_models/` edit. The correction:

- Change `B_j | (N, N) | Modulatory input j` to
  `B | (J, N, N) | Stacked modulatory matrices, J modulators` (matches
  CONTEXT.md §"B-matrix representation").
- Leave the `u | (T, M)` row unchanged (applies to both linear and
  bilinear; `M` becomes `M_driving + J_mod` in bilinear mode — a
  sublety that does NOT require table rewording).

Alternatively, defer to Phase 15 when B becomes a Pyro sample site.
CONTEXT.md explicitly scopes Phase 13's doc-rename to three sites; the
shape table is a fourth site. **Defer** to keep scope tight. Note in
`.planning/STATE.md` as a Phase 15 to-do.

**Sources:** direct reads of `CLAUDE.md`, `.planning/PROJECT.md`; CONTEXT.md
§"Doc-rename scope".

---

## 8. Wave structure & task breakdown proposal

**Recommended plan decomposition (4 plans, 3 waves):**

### Wave 1: Utilities + Docs (parallel)

- **Plan 13-01: `parameterize_B` + `compute_effective_A` utilities**
  - Requirements: BILIN-01, BILIN-02
  - Files modified: `src/pyro_dcm/forward_models/neural_state.py`,
    `src/pyro_dcm/forward_models/__init__.py` (exports)
  - New tests: `tests/test_bilinear_utils.py` (6-8 tests covering
    shapes, einsum correctness, mask diagonal default, DeprecationWarning
    on non-zero mask diagonal, zero-J edge case)
  - Estimated LoC: +80 src / +150 test
  - Dependencies: none
  - Parallelizable: yes (runs alongside 13-04)

- **Plan 13-04: Doc-rename (BILIN-07)**
  - Requirements: BILIN-07
  - Files modified:
    1. `src/pyro_dcm/forward_models/neural_state.py` (module docstring
       lines 1-10 + class docstring line 56-84)
    2. `CLAUDE.md` (generative_models/ block around line 101, plus
       optionally inference/ block around line 119)
    3. `.planning/PROJECT.md` (line 23)
  - No test changes.
  - Estimated LoC: +0 src code / +30 doc edits
  - Dependencies: none
  - Parallelizable: yes (runs alongside 13-01)

### Wave 2: Neural state equation (strict sequential)

- **Plan 13-02: `NeuralStateEquation` extension + invariance test**
  - Requirements: BILIN-03
  - Files modified: `src/pyro_dcm/forward_models/neural_state.py`
    (extend `__init__` and `derivatives`)
  - New tests: `tests/test_linear_invariance.py` (3 bit-exact cases
    per CONTEXT §"Specific Ideas")
  - Modified tests: none (existing `test_neural_state.py` must stay
    green unchanged)
  - Estimated LoC: +40 src / +120 test
  - Dependencies: 13-01 (needs `compute_effective_A` and
    `parameterize_B`)
  - Parallelizable: no (13-03 depends on this)

  **NOTE — scope clarification:** CONTEXT scopes the
  linear-short-circuit gate to `CoupledDCMSystem.forward` (Section 1's
  code block). `NeuralStateEquation.derivatives` also receives
  optional `B, u_mod` args (per BILIN-03 wording), but the CoupledDCMSystem
  is the single place where the runtime `input_fn(t)` is split into
  driving and modulator. **Resolution:** Phase 13-02 extends
  `NeuralStateEquation.derivatives(self, x, u, *, B=None, u_mod=None)`
  with the same `if B is None` short-circuit. The signature extension
  is minimal and keeps `NeuralStateEquation` self-contained for future
  direct-call testing. `CoupledDCMSystem` then forwards the computed
  `u_drive` and `u_mod` to `self.neural.derivatives(x, u_drive, B=self.B,
  u_mod=u_mod)`.

### Wave 3: CoupledDCMSystem + monitor + 3-sigma test (strict sequential)

- **Plan 13-03: `CoupledDCMSystem` extension + stability monitor + 3-sigma test**
  - Requirements: BILIN-04, BILIN-05, BILIN-06
  - Files modified:
    1. `src/pyro_dcm/forward_models/coupled_system.py` (extend
       `__init__`, `forward`, add `_maybe_check_stability`)
    2. `src/pyro_dcm/__init__.py` (add `NullHandler` setup)
  - New tests:
    1. `tests/test_bilinear_coupled_system.py` (or extension of
       `test_ode_integrator.py`): shape tests, linear-kwarg passthrough,
       B-buffer registration, `n_driving_inputs` slicing, stability
       monitor firing (caplog), `stability_check_every=0` zero-overhead
       path.
    2. `tests/test_bilinear_stability.py`: the BILIN-06 3-sigma test
       (Section 5).
  - Modified tests: none (existing `test_ode_integrator.py` tests must
    stay green unchanged)
  - Estimated LoC: +100 src / +200 test
  - Dependencies: 13-02 (needs extended `NeuralStateEquation`)
  - Parallelizable: no

### Parallelism summary

```
Wave 1 (parallel):
  ┌────────────────────┐   ┌────────────────────┐
  │ 13-01: Utilities   │   │ 13-04: Doc-rename  │
  └─────────┬──────────┘   └────────────────────┘
            │
            ▼
Wave 2 (sequential):
  ┌────────────────────────────────────┐
  │ 13-02: NeuralStateEquation +       │
  │        test_linear_invariance      │
  └─────────┬──────────────────────────┘
            ▼
Wave 3 (sequential):
  ┌────────────────────────────────────┐
  │ 13-03: CoupledDCMSystem + monitor  │
  │        + 3-sigma BILIN-06 test     │
  └────────────────────────────────────┘
```

- 13-04 has no code dependencies on any other plan; it can start Wave 1
  and finish before 13-01 ships. It is pure text editing.
- 13-01 can be further split into 13-01a (`parameterize_B`) and 13-01b
  (`compute_effective_A`), but each is ~20 LoC; keep together.
- 13-02 could be split into 13-02a (class extension) and 13-02b
  (invariance test), but the test is the acceptance gate for the
  extension — pair them.
- 13-03 is the largest plan but has a single natural scope
  (CoupledDCMSystem extension). Splitting would fragment the change and
  multiply review overhead. Keep together.

**Critical path:** 13-01 → 13-02 → 13-03. Total: 3 sequential waves,
with 13-04 overlapping Wave 1. Expected wall time ~1-1.5 days per plan
for experienced engineer.

---

## 9. Existing test file inventory

Tests that must stay green unchanged (BILIN-04 acceptance gate):

### Direct callers of touched classes/functions

| Test file | Touches | Count (approx) | Must stay green |
|-----------|---------|---------------|-----------------|
| `tests/test_neural_state.py` | `NeuralStateEquation`, `parameterize_A` | 8 tests (2 classes) | YES — BILIN-04 gate |
| `tests/test_ode_integrator.py` | `CoupledDCMSystem`, `PiecewiseConstantInput`, `integrate_ode` | ~15 tests, 6 `CoupledDCMSystem` constructor calls at lines 157, 246, 310, 348, 406, 450 | YES — BILIN-04 gate |
| `tests/test_task_simulator.py` | `simulate_task_dcm` (calls `CoupledDCMSystem` internally via `task_simulator.py:158`) | ~40 tests | YES — BILIN-04 gate |
| `tests/test_task_dcm_model.py` | `task_dcm_model` (calls `CoupledDCMSystem` at line 147) | ~10 tests | YES — BILIN-04 gate |
| `tests/test_task_dcm_recovery.py` | SVI on `task_dcm_model` | ~5 tests | YES — BILIN-04 gate |

### Indirect callers (less critical but verify)

| Test file | Reason | Must stay green |
|-----------|--------|-----------------|
| `tests/test_amortized_task_dcm.py` | Uses `amortized_task_dcm_model` which calls `CoupledDCMSystem` at `amortized_wrappers.py:128` | YES |
| `tests/test_amortized_benchmark.py` | End-to-end amortized task DCM | YES |
| `tests/test_svi_integration.py` | SVI smoke tests | YES |
| `tests/test_elbo_model_comparison.py` | ELBO on task DCM | YES |
| `tests/test_parameter_packing.py` | Uses `make_random_stable_A` fixture | YES |

### Architecturally insulated (do not touch, verify untouched)

| Test file | Why insulated |
|-----------|---------------|
| `tests/test_spectral_*.py`, `tests/test_csd_computation.py`, `tests/test_spectral_noise.py`, `tests/test_spectral_transfer.py` | Spectral DCM does not import `NeuralStateEquation` or `CoupledDCMSystem` (verified via grep) |
| `tests/test_rdcm_*.py` | rDCM is frequency-domain; does not use `CoupledDCMSystem` |
| `tests/test_balloon.py`, `tests/test_bold_signal.py` | Balloon + BOLD are pre-neural-state components; untouched |
| `tests/test_guide_factory.py`, `tests/test_spm_*validation.py`, `tests/test_tapas_rdcm_validation.py`, `tests/test_validation_export.py`, `tests/test_posterior_extraction.py`, `tests/test_model_ranking_validation.py`, `tests/test_summary_networks.py` | All inference-/validation-layer; do not touch forward-model extension semantics |

### Total test-run count for BILIN-04 gate

Counting from `wc -l`: `test_neural_state.py` 138 lines, `test_ode_integrator.py`
468 lines, `test_task_simulator.py` 541 lines, `test_task_dcm_model.py` 382
lines, `test_task_dcm_recovery.py` 500 lines = **2029 test lines** the
BILIN-04 gate must preserve as-is. The invariance + 3-sigma + monitor
tests add ~400 new test lines. Total test surface after Phase 13:
~3500 lines for the task-DCM forward-model stack.

**Recommended CI gate (planner):** after each plan lands, run
`pytest tests/test_neural_state.py tests/test_ode_integrator.py tests/test_task_simulator.py tests/test_task_dcm_model.py tests/test_task_dcm_recovery.py -v`
and verify zero failures, zero modifications to test source lines.

**Sources:** grep output in Section 0 of this research; direct `wc -l`
count; `.planning/research/v0.3.0/ARCHITECTURE.md` §"Test Impact Analysis".

---

## 10. Risk / pitfall check

### Project-level pitfalls (from `.planning/research/v0.3.0/PITFALLS.md`)

Mapping to Phase 13 plans:

| Pitfall | Severity | Phase 13 coverage |
|---------|----------|-------------------|
| **B1** A_eff(t) stability loss | CRITICAL | ADDRESSED by 13-03 (stability monitor BILIN-05 + 3-sigma test BILIN-06). Monitor fires strict `max Re > 0`; 3-sigma test asserts no NaN. **Not fully mitigated** — monitor is log-only; divergent SVI draws are accepted per D4. This is intentional (raising would break SVI gradients). |
| **B4** Stale "bilinear" docstrings | HIGH | ADDRESSED by 13-04 (doc-rename BILIN-07). Three locked sites + optional wide-scope CLAUDE.md cleanup. |
| **B9** Linear fixtures drift through bilinear path | MEDIUM | ADDRESSED by 13-02 (`test_linear_invariance.py` BILIN-03 at atol=1e-10). CONTEXT-locked "literal short-circuit" design makes this a structural guarantee, not a test-only guarantee. |
| **B12** stim_mod interpolation at rk4 mid-steps blurs sticks | MEDIUM | NOT addressed in Phase 13. Phase 13 consumes whatever `PiecewiseConstantInput` returns; stimulus primitives are Phase 14 (`make_event_stimulus`). **Flag for Phase 14.** |

### Pitfalls DEFERRED to later phases (not Phase 13)

| Pitfall | Severity | Deferred to |
|---------|----------|-------------|
| B2 (B non-identifiability under sparse designs) | CRITICAL | Phase 16 (recovery benchmark) |
| B3 (amortized guide shape failures) | CRITICAL | Deferred to v0.3.1 per D5 |
| B5 (free B diagonal breaks A_eff) | HIGH | PARTIAL — mask-based diagonal-zeroing in 13-01 (`parameterize_B` default) mitigates the benign case; explicit user override emits DeprecationWarning. Full stability monitor (13-03) catches the adversarial case. Phase 15 tightens. |
| B6 (permissive b_mask overfits) | HIGH | Phase 15 (Pyro model + priors) |
| B7 (sign-recovery metric meaningless near zero) | HIGH | Phase 16 (benchmark metrics) |
| B8 (YAML B prior variance wrong) | HIGH | Already resolved in D1 (CONTEXT.md); Phase 15 enforces N(0, 1.0). |
| B10, B11, B13, B14 | MEDIUM/LOW | Phase 16+ |

### Phase 13-specific risks NOT in project-level PITFALLS.md

Surfaced from the concrete CONTEXT decisions and code inspection:

1. **`n_driving_inputs` kwarg ambiguity** (CONTEXT §"Claude's Discretion" item 4).
   If user sets `B` but leaves `n_driving_inputs=None`, what happens?
   Options:
   - (a) Raise `ValueError("n_driving_inputs required when B is not None")`.
   - (b) Infer from `C.shape[1]` and emit `UserWarning` that the inference
     was made.
   - (c) Default `n_driving_inputs = C.shape[1]` silently.
   **Recommendation: (a) Raise.** Silent inference is dangerous because
   the stimulus `values` tensor must have exactly `C.shape[1] + B.shape[0]`
   columns; if the user widened `values` inconsistently, silent inference
   from `C` misses the error. Fail loud and make the user state the
   split explicitly. Docstring should include a worked example showing
   the consistency requirement.

2. **Counter is not a buffer — state leaks across odeint() calls.**
   `self._step_counter` is a plain Python int attribute, not a buffer.
   This means across multiple `integrate_ode(system, ...)` calls (e.g. in
   test loops), the counter persists. That's fine — the cadence is
   "every N forward() calls", monotonically increasing. No reset needed
   unless the user wants deterministic log-count per simulation. If
   deterministic per-sim logging is desired, add
   `def reset_step_counter(self): self._step_counter = 0`. **Recommend
   documenting the behavior; not adding an auto-reset.**

3. **`B` buffer tensor device drift.** If the user constructs
   `CoupledDCMSystem(A.cuda(), C.cuda(), input_fn, B=B_cpu)`, the
   `register_buffer` call keeps `B` on CPU while A/C are on GPU. The
   einsum will fail at the forward pass with a device-mismatch error.
   **Mitigation:** in `__init__`, explicitly move B to A.device with
   `B = B.to(A.device, A.dtype)`. This matches existing behavior in
   `task_simulator.py:156-157` for A and C. Document in `__init__`.

4. **`torch.linalg.eigvals` complex return type confusion.** The monitor
   does `eigs = torch.linalg.eigvals(A_eff); max_re = eigs.real.max().item()`.
   `eigs` is a **complex** tensor (torch.complex128 for float64 input).
   `.real` extracts the real part as float64. `.item()` requires a scalar
   — `.max()` returns 0-d tensor. This is correct but non-obvious to
   reviewers; add a one-line comment.

5. **`torch.no_grad()` in `_maybe_check_stability` is critical.**
   Without it, the eigenvalue computation would be tracked by autograd,
   adding complex-number gradient graph nodes with no semantic value and
   significant overhead during SVI backprop. **Verify the `with
   torch.no_grad():` wrapper is present** in Section 4's code and
   propagated into the plan's task-level verification steps.

6. **`_step_counter` under `odeint_adjoint`.** The adjoint method calls
   `forward()` a second time during backward (reverse-time integration).
   The counter will increment during reverse pass, doubling log volume
   and fragmenting log cadence. **Not critical for Phase 13** because
   `task_dcm_model.py` uses `adjoint=False`, but flag for Phase 15.
   Mitigation when adjoint is enabled: detect reverse pass via `t`
   direction and skip (`if not self.training or <reverse-detection>:`).
   Not in Phase 13 scope.

7. **Empty `J=0` path and empty `u_mod`.** `B=torch.zeros(0, N, N)` with
   `u_mod=torch.zeros(0)` must route through the linear short-circuit
   per CONTEXT §"Linear short-circuit". Verify the gate
   `if self.B is None or self.B.shape[0] == 0` handles both. Covered in
   Section 3's invariance-test extension.

### Summary risk register

| Risk | Severity | Phase 13 plan covering it | Status |
|------|----------|---------------------------|--------|
| A_eff instability at 3σ | Critical | 13-03 | Monitored + tested |
| Linear path fp drift | High | 13-02 | Structural gate + test |
| Stale "bilinear" docs | High | 13-04 | Rewritten |
| `n_driving_inputs` ambiguity | Medium | 13-03 | Raise ValueError; documented |
| Device drift on B | Low | 13-03 | `.to(A.device)` in __init__ |
| Counter under adjoint | Low | Deferred | Phase 15 concern |
| B5 partial mitigation | Medium | 13-01 (mask) + 13-03 (monitor) | Partial; Phase 15 tightens |

---

## Open Questions

1. **CLAUDE.md wide-scope doc fix (Section 7).** CONTEXT scopes the edit
   to `generative_models/`, but the `inference/` block is also stale
   (files moved into `models/`). Does the planner want to include both
   in Phase 13's 13-04, or keep 13-04 tight to the CONTEXT-locked
   3-site scope? **My recommendation:** include both blocks in 13-04;
   the drift is the same kind of issue. **Ask user / accept planner
   default.**

2. **Tensor Shape Conventions table in CLAUDE.md (Section 7).** Defer
   to Phase 15 or fix in 13-04? Recommendation is defer to keep scope
   tight. **Ask user / accept planner default.**

3. **`n_driving_inputs=None` policy (Section 10 risk 1).** Raise vs
   infer vs silent-default. Recommendation is raise ValueError. CONTEXT
   explicitly marks this as Claude's Discretion — planner chooses.
   **No blocker.**

4. **Docs page for `pyro_dcm.stability` logger (Section 6).** New
   `docs/02_pipeline_guide/logging.md` or inline docstring only?
   Recommendation: inline docstring in `CoupledDCMSystem.__init__`;
   defer standalone docs page to Phase 15 or v0.3.0 release. **No
   blocker.**

5. **Counter reset between `integrate_ode` calls (Section 10 risk 2).**
   Auto-reset or document persistent counter? Recommendation: document,
   don't auto-reset (keeps behavior simple and predictable). **No blocker.**

---

## Sources

### Primary (HIGH confidence — direct code/doc reads 2026-04-17)

- `src/pyro_dcm/forward_models/neural_state.py` — 108 lines, full read.
- `src/pyro_dcm/forward_models/coupled_system.py` — 150 lines, full read.
- `src/pyro_dcm/utils/ode_integrator.py` — 242 lines, full read.
- `src/pyro_dcm/forward_models/__init__.py` — full read (exports map).
- `src/pyro_dcm/__init__.py` — full read (confirmed no existing logging
  infrastructure).
- `src/pyro_dcm/simulators/task_simulator.py` lines 140-160, 290-420 —
  `CoupledDCMSystem` caller and `make_random_stable_A` fixture.
- `src/pyro_dcm/models/task_dcm_model.py:140-155` —
  `CoupledDCMSystem` caller in Pyro model.
- `src/pyro_dcm/models/amortized_wrappers.py:120-135` —
  `CoupledDCMSystem` caller in amortized model.
- `tests/test_neural_state.py` — 138 lines, full read.
- `tests/test_ode_integrator.py` — reads at lines 1-100, 140-230.
- `tests/conftest.py` — 69 lines, full read (test fixtures).
- `CLAUDE.md` — lines 60-130 (tech stack + directory structure).
- `.planning/PROJECT.md` — lines 15-50 (requirements).
- `.planning/phases/13-bilinear-neural-state/13-CONTEXT.md` — 162 lines,
  full read.
- `.planning/research/v0.3.0/STACK.md` — lines 1-120.
- `.planning/research/v0.3.0/ARCHITECTURE.md` — full read.
- `.planning/research/v0.3.0/PITFALLS.md` — full read.
- `.planning/research/v0.3.0/FEATURES.md` — lines 1-60.
- `.planning/research/v0.3.0/SUMMARY.md` — full read.
- torchdiffeq master `_impl/rk_common.py`
  (https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/rk_common.py)
  — WebFetch confirmed `t` is float64 scalar Tensor; rk4 = 4 stages,
  dopri5 = 6 stages per step.
- torchdiffeq `FURTHER_DOCUMENTATION.md` — WebFetch confirmed
  `adjoint_params` default is `tuple(func.parameters())` (excludes
  buffers).
- Python stdlib logging HOWTO "Configuring Logging for a Library"
  (https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library)
  — `NullHandler` recommendation.

### Secondary (MEDIUM confidence — project-level research citations)

- `.planning/STATE.md` — D1 (B prior variance = 1.0), D4 (monitor
  log-only never raises), D5 (amortized deferred to v0.3.1) via CONTEXT.
- SPM12 `spm_fx_fmri.m`, `spm_dcm_fmri_priors.m` — cited via
  PITFALLS.md, not re-verified here.

### Tertiary (LOW confidence — not load-bearing)

- Behavior of torchdiffeq `odeint_adjoint` in reverse-pass counter
  increment — inferred from API semantics; not empirically verified.
  Flagged as Phase 15 concern.

---

## Metadata

**Confidence breakdown:**
- torchdiffeq mechanics (Sec 1): HIGH (source-read).
- Buffer registration (Sec 2): HIGH (PyTorch source pattern).
- Linear short-circuit (Sec 3): HIGH (CONTEXT-locked design + test recipe).
- Stability monitor (Sec 4): HIGH on mechanism; MEDIUM on cadence policy
  under dopri5 (acceptable per CONTEXT Discretion).
- 3-sigma fixture (Sec 5): HIGH on construction; MEDIUM on "states may
  grow but remain finite" qualitative claim (untested until plan runs).
- Logging setup (Sec 6): HIGH (stdlib best practice).
- Doc-rename text (Sec 7): HIGH (direct file reads).
- Wave structure (Sec 8): HIGH (dependency-forced).
- Test inventory (Sec 9): HIGH (grep-confirmed).
- Risk register (Sec 10): HIGH on mechanism; MEDIUM on severity ratings.

**Research date:** 2026-04-17
**Valid until:** 2026-05-17 (30 days; stable deps; no published SPM/torchdiffeq
churn expected).
