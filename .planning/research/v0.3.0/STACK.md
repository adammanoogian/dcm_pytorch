# Stack Research — v0.3.0 Bilinear DCM Extension

**Domain:** Research-grade Dynamic Causal Modeling (bilinear extension of linear DCM)
**Researched:** 2026-04-17
**Confidence:** HIGH
**Scope:** What stack additions / changes are needed to go from `dx/dt = Ax + Cu` to `dx/dt = Ax + Σ_j u_j·B_j·x + Cu` with variable-amplitude modulatory events.

---

## Summary

**Verdict: NO new runtime dependencies required for v0.3.0.**

The existing stack (PyTorch 2.9, Pyro 1.9.1, torchdiffeq 0.2.5, scipy, matplotlib, pytest, ruff, mypy) already covers every capability the bilinear extension needs:

1. `torchdiffeq.odeint` already treats the RHS as `f(t, y)` with time dependence — no API change.
2. `PiecewiseConstantInput` already returns shape `(M,)` via `values[idx]` for arbitrary M, so adding a modulatory column to the existing stimulus table works with zero code change. **A new linear-interpolation sibling class is needed ONLY if v0.3.0 decides to support continuously-varying (non-step) modulatory amplitudes** — which the YAML's DCM.V1 explicitly allows ("variable-amplitude events").
3. `torch.linalg.eigvals` already imports without extra deps; it's used today in `make_random_stable_A` (`task_simulator.py:354`).
4. Pyro `AutoNormal` / `AutoMultivariateNormal` / `AutoIAFNormal` / `AutoLowRankMultivariateNormal` auto-discover **all** `pyro.sample` sites during their first model trace — including a new `B_free` site with shape `(M, N, N)` — with zero factory changes. This is the documented `AutoGuide._setup_prototype` behavior.
5. PEB-lite group analysis is explicitly deferred to v0.4+; no dep decision needed now. A brief recommendation is included at the end for future reference.

The v0.3.0 work is math + Pyro model plumbing, not infrastructure. The only **new in-repo primitive** likely to be added is a thin `LinearInterpolatedInput` class alongside `PiecewiseConstantInput` — but this is a ~40-line module inside `pyro_dcm.utils.ode_integrator`, not a dependency.

---

## Recommended Stack Additions

### Runtime dependencies

**None.** All bilinear-DCM capabilities are covered by existing pinned libraries.

### New in-repo primitives (no new deps)

| Primitive | File | Purpose | Why needed |
|-----------|------|---------|------------|
| `LinearInterpolatedInput` (optional) | `src/pyro_dcm/utils/ode_integrator.py` | Linear interp of stimulus values between grid points; returns shape `(M,)` per call via two `searchsorted` lookups + a lerp. | Only if DCM.V1 variable-amplitude modulatory events use continuous ramps; if variable-amplitude events are still piecewise-constant with differing heights, `PiecewiseConstantInput` already covers it (see §Integration Points). |
| `parameterize_B` helper | `src/pyro_dcm/forward_models/neural_state.py` | Optional mask/scale transform on `B_free` (SPM convention: `B` typically has **no** diagonal constraint, unlike `A`). | Keeps naming symmetry with existing `parameterize_A`; even if it's the identity function, documenting the choice is important. |
| `BilinearNeuralStateEquation` | `src/pyro_dcm/forward_models/neural_state.py` | Computes `(A + Σ_j u_j(t)·B_j) @ x + C @ u` given `A, B (shape M,N,N), C, u(t)`. | Core new math. Pure PyTorch; one `torch.einsum("j,jkl->kl", u, B)` call. |
| `A_eff(t)` eigenvalue monitor hook | `src/pyro_dcm/utils/diagnostics.py` or `simulators/task_simulator.py` | Evaluates `torch.linalg.eigvals(A + Σ_j u_j·B_j).real.max()` at each coarse time step during simulation and flags if any value >= 0. | Required by YAML DCM.V1 stability diagnostic. Uses `torch.linalg.eigvals` (already in PyTorch 2.x core). |

### Development tools

**No changes.** ruff + mypy + pytest configs stay as-is. Existing benchmark infrastructure (`benchmarks/run_all_benchmarks.py`, RUNNER_REGISTRY, .npz fixtures) receives a new `bilinear_task_dcm` runner entry — zero new tooling.

---

## Integration Points

### 1. torchdiffeq — no change, confirmed

**Claim verified (HIGH):** `odeint(func, y0, t_eval)` calls `func(t, y)` at every internal solver step. The RHS is free to inspect `t` and construct a time-varying `A_eff(t) = A + Σ_j u_j(t)·B_j` on the fly. This is exactly how the current `CoupledDCMSystem.forward(t, state)` already uses `self.input_fn(t)` at line 140 of `coupled_system.py`.

- torchdiffeq 0.2.5 (PyPI, released 2024-11-21) is the latest version; already pinned.
- `jump_t` option in `options={"jump_t": grid_points}` remains correct for discontinuities when the modulatory stimulus table contains step changes. `ode_integrator.py:174` already does this.
- **No API migration, no new solver, no version bump.**

### 2. Modulatory stimulus interpolation — `PiecewiseConstantInput` generalizes

**Claim verified (HIGH):** `PiecewiseConstantInput(times, values)` with `values` of shape `(K, M)` handles **any M** via `self.values[idx]` which returns shape `(M,)`. There is nothing in `ode_integrator.py:50-74` that assumes `M = 1`.

For v0.3.0, the stimulus table just needs to be widened from shape `(K, M_driving)` to `(K, M_driving + J_modulatory)`, with a convention that:
- Columns `0..M_driving-1` drive neural activity via `C @ u[0:M_driving]`.
- Columns `M_driving..M_driving+J-1` modulate connectivity via `B_j * u[M_driving+j]` for `j = 0..J-1`.

**When a new `LinearInterpolatedInput` primitive IS needed:**
- If DCM.V1's "variable-amplitude events" means a smooth ramp (e.g. parametric modulation of stimulus intensity across a block), step interpolation introduces artificial discontinuities that force the adaptive solver to restart unnecessarily often and may miss the intended cognitive-state continuity.
- Implementation is ~20 lines:
  ```python
  idx = torch.searchsorted(self.times, t.detach(), right=True) - 1
  idx = torch.clamp(idx, min=0, max=self.values.shape[0] - 2)
  t0, t1 = self.times[idx], self.times[idx + 1]
  alpha = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
  return (1 - alpha) * self.values[idx] + alpha * self.values[idx + 1]
  ```
- For this mode, **do not** pass `grid_points` to `integrate_ode` — the signal is smooth, so the adaptive solver should be allowed to choose its own step size.

**When `PiecewiseConstantInput` is sufficient:**
- If "variable-amplitude events" is interpreted as block-design with different heights per block (e.g. 100%, 50%, 75% contrast), encode the heights directly into `values[:, j]` and keep step interpolation. No new class needed.

**Recommended:** Build `LinearInterpolatedInput` in v0.3.0 because DCM.V1 explicitly calls out "variable-amplitude" distinct from "variable-block-height" and this is the cleaner primitive. It is strictly additive — `PiecewiseConstantInput` stays in place for block-design tasks.

### 3. Eigenvalue monitoring — `torch.linalg.eigvals` is sufficient

**Claim verified (HIGH):** `torch.linalg.eigvals` is in PyTorch 2.x core (confirmed locally: `torch 2.9.1+cpu` installed). It returns complex eigenvalues without requiring symmetry or stability. `make_random_stable_A` already uses it at `task_simulator.py:354`.

**Diagnostic pattern (for DCM.V1):**
```python
# Evaluate A_eff(t) stability on a coarse grid during simulation.
t_coarse = torch.arange(0, duration, 1.0, dtype=torch.float64)  # 1 s grid
u_coarse = torch.stack([stimulus(t) for t in t_coarse])         # (T_c, M)
A_eff = A[None] + torch.einsum("tj,jkl->tkl", u_coarse[:, M_drive:], B)
max_real = torch.linalg.eigvals(A_eff).real.max(dim=-1).values  # (T_c,)
unstable_fraction = (max_real >= 0).float().mean().item()
```

**No new dep.** If future needs include batched eigenvalue computations on GPU with very large N, the PyTorch `linalg` backend already handles that.

### 4. Pyro guides — automatic site discovery, zero factory changes

**Claim verified (HIGH, docs + source inspection):** Pyro `AutoGuide._setup_prototype` runs `poutine.trace(model)` on the first guide call and registers guide parameters for **every** `pyro.sample` site found. Adding a new `pyro.sample("B_free", ...)` to `task_dcm_model` is automatically discovered with no changes to `pyro_dcm.models.guides` (neither `create_guide` nor any registry entry).

**Important caveat (model structure is frozen after first trace):** The guide assumes model structure is static. Practically this is a non-issue for v0.3.0 because:
- Every call to the model uses the same `(N, M_driving, J_modulatory)` dimensions.
- `B_free` shape is `(J, N, N)` (or equivalently `(M_modulatory, N, N)` flattened via `.to_event(3)`), known at model construction.
- Different experiments construct **different guide instances**, which is already the pattern in `create_guide`.

**Recommended `B_free` prior specification (SPM convention, see SPM12 `spm_dcm_fmri_priors.m`):**
```python
B_free_prior = dist.Normal(
    torch.zeros(J, N, N, dtype=torch.float64),
    (1.0 / 64.0) ** 0.5 * torch.ones(J, N, N, dtype=torch.float64),
).to_event(3)
B_free = pyro.sample("B_free", B_free_prior)      # shape (J, N, N)
B_free = B_free * b_mask                           # structural mask
```

**Do NOT wrap in `pyro.plate("modulatory", J)`** just to index `B_j`. The `to_event(3)` approach is simpler, avoids a sequential-plate restriction in certain `AutoGuide` internals (`NotImplementedError` for sequential plates, per Pyro source), and keeps the latent shape contiguous for `AutoMultivariateNormal` / `AutoLowRankMultivariateNormal` covariance parameterization.

**Guide blocklist update in `guides.py`:** Consider lowering `_MAX_REGIONS["auto_mvn"]` from 7 to ~5 for bilinear DCM because the latent dimension now includes `N² + N·M + J·N² + 1` instead of `N² + N·M + 1`. For `N=5, J=2`: `25 + 5M + 50 + 1 ≈ 80+` latent dims, which is still fine for `auto_mvn`. For `N=7, J=2`: `49 + 7M + 98 + 1 ≈ 150+`, which starts costing O(150²) Cholesky factors per SVI step. **This is a performance tuning note, not a new dependency.**

### 5. PEB-lite group analysis (DEFERRED to v0.4+, brief note only)

When v0.4+ adds hierarchical group analysis, the right primitive depends on what level of PEB fidelity is needed:

| Option | When to choose | Why |
|--------|----------------|-----|
| **Native Pyro hierarchical model** | Want tight integration with existing guides/SVI, full control over PEB-specific design matrices. | Zero new deps, same SVI stack, same `create_guide` / `run_svi` tooling. Recommended. |
| **NumPyro + `MixedEffects` / handwritten** | Want NUTS validation of the group model (mirrors existing `nuts_validator.py` pattern for first-level DCM). | Uses already-pinned `numpyro` dep. No new install. |
| **Bambi 0.17.2** (PyMC-backed) | Want R-style `lme4` formula syntax for the group design, don't need tight DCM integration. | **Adds large deps:** PyMC, PyTensor, ArviZ. Formula interface is nice but doesn't translate cleanly to PEB's "design matrix over first-level posterior means" pattern. **Not recommended** for PEB — PEB is not a standard GLMM. |

**Recommendation for v0.4+:** Native Pyro hierarchical model. Defer the decision; no action in v0.3.0.

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `scipy.interpolate.interp1d` for modulatory stimulus | Breaks autograd — scipy has no PyTorch backend, and the stimulus is evaluated inside the ODE RHS which must stay differentiable for SVI. | Hand-rolled `LinearInterpolatedInput` using `torch.searchsorted` + tensor arithmetic. |
| `torch.nn.functional.interpolate` | Designed for image/grid upsampling, not for scattered-point interpolation at arbitrary solver-chosen times. Shape semantics don't match. | `torch.searchsorted` + manual lerp (~6 lines). |
| A new ODE solver library (`diffrax`, `torchode`) | torchdiffeq 0.2.5 already supports time-varying RHS, `jump_t`, and adjoint gradients. Migration is months of validation work for zero new capability. | Keep torchdiffeq. |
| Bambi / PyMC for PEB | Heavyweight dep (PyTensor compile step, ~30-60s per model), doesn't match PEB's design-matrix-over-first-level-posteriors structure, and would split the inference stack (Pyro for DCM, PyMC for group). | (v0.4+) Native Pyro hierarchical model. |
| `pyro.plate("modulatory", J)` around `B_j` sampling | Some `AutoGuide` subclasses raise `NotImplementedError` on sequential plates. Vectorized plates work but add shape-handling complexity with no benefit. | `dist.Normal(zeros(J,N,N), scale).to_event(3)`. |
| Sampling hemodynamic parameters (still) | v0.1.0 deliberately fixed these. Bilinear extension doesn't change that decision. | Keep hemodynamic params fixed at SPM12 defaults. |

---

## Version Compatibility

| Package | Version (installed / pinned) | Compatibility notes |
|---------|------------------------------|---------------------|
| torch | 2.9.1+cpu (installed); `>=2.0` (pinned) | `torch.linalg.eigvals`, `torch.einsum`, `torch.searchsorted` all stable since 2.0. |
| torchdiffeq | 0.2.5 (released 2024-11-21, latest) | `jump_t` API in `options={}` is stable; no change needed. |
| pyro-ppl | 1.9.1 (installed); `>=1.9` (pinned) | `AutoGuide._setup_prototype` behavior and `.to_event(n)` shape semantics both documented and stable. |
| zuko | `>=1.2` (pinned) | Not touched by v0.3.0 (amortized flows are v0.2.0 / v0.5+ territory). |
| numpyro | Existing pin | NUTS validation of bilinear task-DCM is optional for v0.3.0 (recommended: add as one validation test, not a blocking requirement). |

**No version bumps required.**

---

## Confidence

| Claim | Level | Source |
|-------|-------|--------|
| torchdiffeq RHS is `f(t, y)`, time-varying A_eff handled natively | HIGH | torchdiffeq README, `FURTHER_DOCUMENTATION.md`, existing `CoupledDCMSystem.forward` already uses `input_fn(t)`. |
| `PiecewiseConstantInput` generalizes to any M | HIGH | Direct source read: `ode_integrator.py:50-74`, shape is `values[idx]` with no M assumption. |
| `torch.linalg.eigvals` available, no new dep | HIGH | Already imported in `task_simulator.py:354`; PyTorch core. |
| Pyro `AutoNormal` auto-discovers new sample sites on first trace | HIGH | Pyro source inspection (`pyro/infer/autoguide/guides.py` `_setup_prototype`); confirmed behavior docs. |
| Sequential `pyro.plate` breaks some AutoGuides | HIGH | Pyro source: explicit `NotImplementedError` for sequential plates in `AutoContinuous`. |
| torchdiffeq 0.2.5 is latest stable | HIGH | PyPI 2024-11-21 release confirmed. |
| Bambi is wrong fit for PEB | MEDIUM | Bambi is a GLMM tool; PEB is a design-matrix-over-first-level-posteriors method. Architecturally a mismatch. Deferred decision; no hard verification yet because v0.4+ is out of scope. |
| `LinearInterpolatedInput` is needed for variable-amplitude events | MEDIUM | Depends on whether DCM.V1 "variable-amplitude" means continuous ramps or per-block heights. Recommend building it since cost is ~40 lines and it's strictly additive. |

---

## Sources

- [torchdiffeq GitHub README](https://github.com/rtqichen/torchdiffeq) — verified `odeint(func, y0, ...)` signature and time-varying RHS support.
- [torchdiffeq FURTHER_DOCUMENTATION.md](https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md) — verified `jump_t` option for discontinuities.
- [torchdiffeq on PyPI](https://pypi.org/project/torchdiffeq/) — confirmed 0.2.5 is latest (2024-11-21).
- [Pyro AutoGuide source (dev branch)](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/autoguide/guides.py) — confirmed `_setup_prototype` traces model on first call, freezes structure, registers all sample sites.
- [Pyro AutoGuide docs](https://pyro4ci.readthedocs.io/en/latest/infer.autoguide.html) — confirmed `AutoNormal` / `AutoMultivariateNormal` / `AutoIAFNormal` behavior.
- [Pyro tensor shapes tutorial](https://pyro.ai/examples/tensor_shapes.html) — verified `.to_event(n)` semantics for multi-dim tensor priors.
- [Bambi GitHub](https://github.com/bambinos/bambi) / [Bambi PyPI](https://pypi.org/project/bambi/) — noted current version (0.17.2) and PyMC backend for PEB-alternative evaluation only.
- Local source reads: `src/pyro_dcm/utils/ode_integrator.py`, `src/pyro_dcm/forward_models/coupled_system.py`, `src/pyro_dcm/forward_models/neural_state.py`, `src/pyro_dcm/models/task_dcm_model.py`, `src/pyro_dcm/models/guides.py`, `src/pyro_dcm/simulators/task_simulator.py`, `pyproject.toml`.
- Local env inspection: `torch 2.9.1+cpu`, `pyro 1.9.1`, `torchdiffeq 0.2.5`.

---
*Stack research for: v0.3.0 bilinear DCM extension of pyro_dcm*
*Researched: 2026-04-17*
