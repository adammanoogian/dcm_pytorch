# Phase 15: Pyro Generative Model with B Priors and Masks — Research

**Researched:** 2026-04-18
**Domain:** Pyro probabilistic programming; task-DCM generative model extension to the bilinear path
**Confidence:** HIGH (all critical code paths verified by direct source read; Pyro auto-discovery mechanism confirmed via `inspect.getsource(AutoGuide._setup_prototype)`)

---

## Executive Summary

1. **The Pyro per-modulator pattern is a strict Python `for j in range(J)` loop** that calls `pyro.sample(f"B_free_{j}", dist.Normal(0, B_PRIOR_VARIANCE**0.5).to_event(2))`. Pyro's `AutoGuide._setup_prototype` iterates over `prototype_trace.iter_stochastic_nodes()` and auto-registers every stochastic site — dynamic names like `B_free_0`, `B_free_1` are handled identically to static names like `A_free` or `C`. **Zero changes required to `create_guide`** (MODEL-06 is structurally free once the model-side loop is written correctly).

2. **Forward-model plumbing must call `CoupledDCMSystem` directly, not `simulate_task_dcm`.** The current linear `task_dcm_model` already takes this path (`task_dcm_model.py:147`). The bilinear extension adds (a) a `merge_piecewise_inputs(stimulus, stim_mod)` merge of driving + modulator inputs, (b) `B=B_stacked, n_driving_inputs=c_mask.shape[1]` kwargs on `CoupledDCMSystem`. `simulate_task_dcm` is NOT reusable — it runs a full simulator with noise, not a Pyro model.

3. **`B_PRIOR_VARIANCE = 1.0` module-level constant (MODEL-02) locks D1.** SPM12 `spm_dcm_fmri_priors.m` pC.B = B (variance 1.0 for the one-state model). The YAML "1/16" claim was audited wrong in v0.3.0 PITFALLS Section B8; D1 corrects it.

4. **Bilinear sites must be sampled with `.to_event(2)` on a `(N, N)` normal** — NOT unplated per free element. `parameterize_B` applies the mask AFTER sampling (masked elements remain in the trace but are zeroed deterministically downstream). This mirrors how `A_free` and `C` are sampled in the current linear model (`task_dcm_model.py:126-131, 138-143`). The flat-vector-of-free-entries alternative is rejected: `AutoGuide` would need shape-custom surgery per modulator, breaking MODEL-06's "no factory changes" gate.

5. **`TaskDCMPacker` refusal (MODEL-07) is a 3-line sample-site-name guard.** The packer's `pack()` and `unpack()` hardcode the keys `{"A_free", "C", "noise_prec"}` (`parameter_packing.py:117-121, 161-165`). A defensive pre-check at pack-time (or inside `amortized_task_dcm_model`) on input `params.keys()` / `trace.nodes` — `any(k.startswith("B_free_") for k in sites)` → raise `NotImplementedError("Bilinear amortized guides deferred to v0.3.1 per D5; use SVI paths.")` — closes MODEL-07 with no packer refactor.

**Primary recommendation:** Implement Phase 15 in **3 plans** (Wave 1: task_dcm_model + B_PRIOR_VARIANCE constant; Wave 2a parallel: extract_posterior_params + packer refusal; Wave 2b parallel: guide trace tests). Total ~600 LoC src changes, ~400 LoC tests. Critical path: 15-01 (model extension) unblocks everything else.

---

## 1. Pyro Model Structure — Per-modulator `B_free_j` Sampling

**Exact required pattern** (matches MODEL-01 "per-modulator loop" wording verbatim):

```python
# B_PRIOR_VARIANCE is a module-level constant in task_dcm_model.py
B_PRIOR_VARIANCE: float = 1.0
"""Prior variance on B_free elements (D1; SPM12 spm_dcm_fmri_priors.m pC.B = B)."""

def task_dcm_model(
    observed_bold: torch.Tensor,
    stimulus: object,               # driving input (PiecewiseConstantInput)
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    TR: float,
    dt: float = 0.5,
    *,
    b_masks: list[torch.Tensor] | None = None,   # list of (N,N) masks, one per modulator
    stim_mod: object | None = None,              # modulator PiecewiseConstantInput (J columns)
) -> None:
    ...
    # --- Existing A_free, C, noise_prec sampling unchanged ---

    # --- Bilinear block (ADDITIVE; only active when b_masks non-empty) ---
    B_list: list[torch.Tensor] = []
    if b_masks is not None and len(b_masks) > 0:
        _validate_bilinear_args(b_masks, stim_mod)   # see Section 5
        B_prior_std = (B_PRIOR_VARIANCE) ** 0.5      # = 1.0 under D1
        for j, b_mask_j in enumerate(b_masks):
            # (N, N) Normal prior; .to_event(2) matches A_free/C convention.
            B_free_j = pyro.sample(
                f"B_free_{j}",
                dist.Normal(
                    torch.zeros_like(b_mask_j),
                    B_prior_std * torch.ones_like(b_mask_j),
                ).to_event(2),
            )
            # parameterize_B emits DeprecationWarning on non-zero diagonal (MODEL-03).
            B_j = parameterize_B(
                B_free_j.unsqueeze(0),       # (1, N, N) stacked form for parameterize_B
                b_mask_j.unsqueeze(0),
            ).squeeze(0)                      # back to (N, N)
            B_list.append(B_j)
        B_stacked = torch.stack(B_list, dim=0)        # (J, N, N)
        pyro.deterministic("B", B_stacked)            # optional but useful for introspection
```

**Key design points:**

- **`f"B_free_{j}"` is mandatory** — no plate. Pyro's AutoGuide supports per-site guides but `pyro.plate` over modulators is NOT appropriate here because each modulator has its own distinct `b_mask_j` (structural) with potentially different sparsity patterns. The rDCM precedent (`rdcm_model.py:97-120`) uses a Python loop with per-region sample sites for exactly this reason (each region has different `D_r`).
- **`.to_event(2)` on `(N, N)`** — matches how `A_free` is sampled at `task_dcm_model.py:126-129`. Tells Pyro that the `(N, N)` matrix is a single observation event (not independent scalars to be plated).
- **`parameterize_B` expects 3-D `(J, N, N)` input.** Two options: (a) call it per-modulator with `unsqueeze(0)` / `squeeze(0)` as shown (simple), or (b) stack free matrices first and call `parameterize_B` once on `(J, N, N)` (cleaner — recommended for final implementation):

```python
# Cleaner alternative (recommended):
B_free_list = [pyro.sample(f"B_free_{j}", ...) for j in range(J)]
B_free_stacked = torch.stack(B_free_list, dim=0)                 # (J, N, N)
b_mask_stacked = torch.stack(list(b_masks), dim=0)               # (J, N, N)
B_stacked = parameterize_B(B_free_stacked, b_mask_stacked)       # (J, N, N)
pyro.deterministic("B", B_stacked)
```

**Sources:**
- `src/pyro_dcm/models/task_dcm_model.py:126-143` — existing `A_free`/`C` sample pattern.
- `src/pyro_dcm/models/rdcm_model.py:97-120` — per-iteration sample-site precedent.
- `src/pyro_dcm/forward_models/neural_state.py:62-153` — `parameterize_B` contract.

**Confidence:** HIGH (direct code read; pattern mirrors existing sites).

---

## 2. Forward-Model Plumbing Inside the Pyro Model

**Answer to: does the Pyro model reuse `simulate_task_dcm` or call `CoupledDCMSystem` directly?**

**It calls `CoupledDCMSystem` directly.** Current linear model already does
(`task_dcm_model.py:147-151`):

```python
system = CoupledDCMSystem(A, C, stimulus)
y0 = make_initial_state(N, dtype=torch.float64)
solution = integrate_ode(system, y0, t_eval, method="rk4", step_size=dt)
```

`simulate_task_dcm` is **not** reusable because it (a) uses dopri5 by default (Pyro uses rk4 for SVI stability per 04-RESEARCH.md Pitfall 1), (b) adds observation noise (the Pyro model gets noise via the likelihood site, not pre-noised), (c) returns a dict with many extra entries, (d) normalizes inputs the Pyro model's caller already handled.

**Bilinear extension of the call site** (new code replaces `task_dcm_model.py:147`):

```python
from pyro_dcm.utils.ode_integrator import merge_piecewise_inputs

# ... after sampling A_free, C, B_free_j, and building B_stacked ...

if b_masks is not None and len(b_masks) > 0:
    # Bilinear mode: merge driving + modulator into widened input.
    input_fn = merge_piecewise_inputs(stimulus, stim_mod)
    system = CoupledDCMSystem(
        A, C, input_fn, hemo_params=None,
        B=B_stacked,
        n_driving_inputs=c_mask.shape[1],
        stability_check_every=10,       # Section 9 discusses D4 interaction
    )
else:
    # Linear short-circuit (MODEL-04 bit-exact reduction gate).
    system = CoupledDCMSystem(A, C, stimulus)
```

**Why `merge_piecewise_inputs`?** `CoupledDCMSystem` in bilinear mode expects a SINGLE `input_fn(t) -> (M_drive + J,)` vector, slicing at `n_driving_inputs` internally (`coupled_system.py:286-297`). `merge_piecewise_inputs` in `utils/ode_integrator.py:244` handles the sorted-union-of-breakpoints correctly (and is already tested in Phase 14).

**Sources:**
- `src/pyro_dcm/models/task_dcm_model.py:145-153` — existing ODE call site.
- `src/pyro_dcm/forward_models/coupled_system.py:286-300` — bilinear forward pass with input splitting.
- `src/pyro_dcm/utils/ode_integrator.py:244-350` — `merge_piecewise_inputs`.
- `src/pyro_dcm/simulators/task_simulator.py:307-344` — reference for how Phase 14 wires B through (same pattern).

**Confidence:** HIGH (direct read of all three files; Phase 14 is the precedent).

---

## 3. Guide Auto-Discovery — Exact Mechanism

**Pyro's `AutoGuide._setup_prototype`** (verified via `inspect.getsource`):

```python
def _setup_prototype(self, *args, **kwargs):
    # run the model so we can inspect its structure
    model = poutine.block(self.model, self._prototype_hide_fn)
    self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(*args, **kwargs)
    ...
    for name, site in self.prototype_trace.iter_stochastic_nodes():
        for frame in site["cond_indep_stack"]:
            ...
```

**Key properties:**
- The guide re-runs the model under `poutine.trace` on its **first call** (lazy setup).
- `iter_stochastic_nodes()` yields every site created by `pyro.sample(name, ...)` regardless of name pattern.
- Dynamic names (e.g., `f"B_free_{j}"` producing `B_free_0`, `B_free_1`, ...) are treated identically to static names.
- `AutoNormal._setup_prototype` (verified):

```python
for name, site in self.prototype_trace.iter_stochastic_nodes():
    ...
    init_scale = torch.full_like(init_loc, self._init_scale)
    deep_setattr(self.locs, name, PyroParam(init_loc, constraints.real, event_dim))
    deep_setattr(self.scales, name, PyroParam(init_scale, self.scale_constraint, event_dim))
```

→ creates `self.locs.B_free_0`, `self.locs.B_free_1`, ... without any factory-level awareness.

**`AutoLowRankMultivariateNormal` and `AutoIAFNormal`** subclass `AutoContinuous`, which concatenates ALL continuous sites into a single `_latent` vector (via `_unpack_latent`). Site addition is automatic — dimension growth is handled at setup time.

**Minimal trace test (MODEL-06):**

```python
@pytest.mark.parametrize("guide_type", ["auto_normal", "auto_lowrank_mvn", "auto_iaf"])
def test_b_sites_auto_discovered(task_bilinear_data, guide_type):
    pyro.clear_param_store()
    guide = create_guide(task_dcm_model, guide_type=guide_type)
    # First call triggers _setup_prototype.
    guide(**task_bilinear_data_model_kwargs)  # or via one SVI step
    param_names = set(pyro.get_param_store().keys())
    J = len(task_bilinear_data["b_masks"])
    for j in range(J):
        assert any(f"B_free_{j}" in name for name in param_names), (
            f"B_free_{j} not auto-discovered by {guide_type}"
        )
```

Alternatively (more rigorous, no param-store coupling):

```python
guide(**kwargs)  # triggers setup
prototype_sites = set(guide.prototype_trace.nodes.keys())
for j in range(J):
    assert f"B_free_{j}" in prototype_sites
```

**Confidence:** HIGH (source code directly inspected; `_setup_prototype` runs once lazily and discovers all `pyro.sample` sites).

**Sources:**
- `pyro.infer.autoguide.guides.AutoGuide._setup_prototype` (verified locally via `inspect.getsource`).
- `pyro.infer.autoguide.guides.AutoNormal._setup_prototype` (same).
- [Pyro AutoGuide docs (stable)](https://docs.pyro.ai/en/stable/infer.autoguide.html)

---

## 4. B-Site Shape: Full `(N, N)` with `.to_event(2)`, Not Flat Free-Entries

**Two alternatives considered:**

| Alternative | Pros | Cons |
|-------------|------|------|
| **(A) Sample full `(N, N)` tensor, mask after** | Matches `A_free`/`C` pattern; trivial AutoGuide discovery; consistent with SPM12 `pC.B = B` which carries full matrix variance then multiplies by `B_mask` | Guide wastes capacity on masked-to-zero elements (stored scale is never exercised) |
| **(B) Flatten free entries, sample `(n_free,)`** | Zero "wasted" guide parameters | Requires per-modulator unpacker logic inside model; AutoGuide sees a variable-size vector per `j`; incompatible with MODEL-06 "auto-discovery across three guide families without factory changes" |

**Decision: Alternative A is mandatory** because MODEL-06 forbids guide-factory changes. The rDCM precedent (`rdcm_model.py`) takes Alternative B (variable `D_r` per region) precisely because rDCM can tolerate `create_guide` auto-discovering variable-size sites — but rDCM does NOT have to support `AutoLowRankMultivariateNormal`, which flattens all sites into one vector. Attempting B across all three guide families would break covariance structure and/or require custom `AutoContinuous` subclassing.

**rDCM precedent for "per-modulator loop" (MODEL-01 reference):**
- `rdcm_model.py:101-145` — Python `for r in range(nr)` loop with `f"theta_{r}"` and `f"noise_prec_{r}"` sites.
- Empirically confirmed to work with `AutoNormal` (`test_rdcm_model.py`), though rDCM does not exercise `AutoLowRankMVN` or `AutoIAFNormal` in the test suite.
- **For Phase 15, the precedent is "per-modulator loop with named sites" — the shape choice (full matrix, `.to_event(2)`) is Phase 15-specific.**

**Guide-capacity concern quantified:** At N=3, J=3, b_mask with zero diagonal only: free elements = 3 × 6 = 18. Full-matrix guide params = 3 × 9 = 27. Wasted capacity = 9 loc + 9 scale = 18 scalars. Negligible vs. A's 9-param and C's 3-param. For N=10, J=5: waste = 50, total = 450+500 = 950; still negligible.

**Sources:**
- `src/pyro_dcm/models/rdcm_model.py:97-165` — per-region loop precedent.
- `src/pyro_dcm/models/task_dcm_model.py:126-143` — full-matrix-with-mask pattern for A, C.
- `spm_dcm_fmri_priors.m` (v0.3.0 PITFALLS.md B8) — SPM stores full B with mask, not flat free vector.

**Confidence:** HIGH (rejected alternative B only on MODEL-06 constraint — if MODEL-06 were relaxed, B could be viable; but MODEL-06 is locked).

---

## 5. Edge Cases — `b_masks=None`, `b_masks=[]`, Shape Mismatches

**MODEL-04 edge-case matrix:**

| `b_masks` | `stim_mod` | Behavior |
|-----------|------------|----------|
| `None` | Any (ignored) | Linear short-circuit. No `B_free_j` sites sampled. Must exactly match pre-Phase-15 trace structure. |
| `[]` (empty list) | Any (ignored) | Equivalent to `None`. Normalize to `None` at function entry: `if b_masks == []: b_masks = None`. |
| `[mask_0, mask_1]` (len 2) | `stim_mod` with `(K, 2)` values | Bilinear; J=2. |
| `[mask_0]` (len 1) | `stim_mod is None` | `ValueError("stim_mod is required when b_masks is non-empty; got None.")` |
| `[mask_0, mask_1]` (len 2) | `stim_mod` with `(K, 3)` values | `ValueError("stim_mod.values.shape[1]=3 must match len(b_masks)=2.")` |
| `[mask_0]` where `mask_0.shape != (N, N)` | Any | `ValueError("b_masks[0].shape={...} must equal (N, N)={...}.")` |

**Recommended validation helper** (place near top of `task_dcm_model`, called only in bilinear branch):

```python
def _validate_bilinear_args(
    b_masks: list[torch.Tensor],
    stim_mod: object,
    N: int,
) -> None:
    if stim_mod is None:
        raise ValueError(
            "task_dcm_model: stim_mod is required when b_masks is non-empty; "
            "got None. Construct with make_epoch_stimulus (preferred per "
            "Pitfall B12)."
        )
    for j, m in enumerate(b_masks):
        if m.shape != (N, N):
            raise ValueError(
                f"task_dcm_model: b_masks[{j}].shape={tuple(m.shape)} must "
                f"equal (N, N)=({N}, {N})."
            )
    # Require stim_mod to be PiecewiseConstantInput (type-narrow).
    if not hasattr(stim_mod, "values"):
        raise TypeError(
            "task_dcm_model: stim_mod must be a PiecewiseConstantInput "
            "(has .values attr); got " f"{type(stim_mod).__name__}."
        )
    J = stim_mod.values.shape[1]
    if J != len(b_masks):
        raise ValueError(
            f"task_dcm_model: stim_mod.values.shape[1]={J} must equal "
            f"len(b_masks)={len(b_masks)}."
        )
```

**Raise-at-model-call-time, not at sample-site-time.** Validation before `pyro.sample` ensures the error message is not wrapped inside a Pyro trace stack. Matches `simulate_task_dcm` error behavior (`task_simulator.py:308-332`).

**Normalization of `[]` to `None` is explicit**, paralleling `_normalize_B_list` in `task_simulator.py:37-88` (which returns `None` for `[]`). The Pyro model should do the same at entry:

```python
if b_masks is not None and len(b_masks) == 0:
    b_masks = None  # MODEL-04 J=0 collapse
```

**Sources:**
- `src/pyro_dcm/simulators/task_simulator.py:37-88` — `_normalize_B_list` precedent.
- `src/pyro_dcm/simulators/task_simulator.py:308-332` — `stimulus_mod` validation pattern.

**Confidence:** HIGH.

---

## 6. Amortized Packer Refusal (MODEL-07)

**Existing `TaskDCMPacker` sample-site contract** (`parameter_packing.py:35-165`):

- `pack(params)` expects `params.keys() ⊇ {"A_free", "C", "noise_prec"}` (keys are positional via `.flatten()` order).
- `unpack(z)` returns exactly `{"A_free", "C", "noise_prec"}` — hardcoded in the `return` dict at line 161-165.
- `n_features = N*N + N*M + 1` — hardcoded at line 84. Cannot accommodate `J*N*N` bilinear terms.

**`amortized_task_dcm_model`** (`amortized_wrappers.py:155-219`):
- Samples single `_latent` site (`amortized_wrappers.py:73-79` in `_sample_latent_and_unpack`).
- `params["A_free"]`, `params["C"]`, `params["noise_prec"]` consumed in `amortized_task_dcm_model`.
- Does NOT have any awareness of `B_free_j` sites.

**Recommended refusal implementation (MODEL-07):**

Two complementary checks:

**(a) Hard refusal at `amortized_task_dcm_model` level** — rejects callers that would pass bilinear kwargs:

```python
def amortized_task_dcm_model(
    observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt, packer,
    *,
    b_masks=None, stim_mod=None,  # add these kwargs for API symmetry
):
    if b_masks is not None and len(b_masks) > 0:
        raise NotImplementedError(
            "amortized_task_dcm_model does not support bilinear (B) sample "
            "sites in v0.3.0 per D5. Bilinear amortized inference is "
            "deferred to v0.3.1 (see .planning/STATE.md D5). Use the SVI "
            "path via task_dcm_model + create_guide for bilinear DCM."
        )
    ...  # existing linear body
```

**(b) Defensive refusal inside `TaskDCMPacker.pack`** — catches indirect callers:

```python
def pack(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
    bilinear_keys = [k for k in params if k.startswith("B_free_")]
    if bilinear_keys:
        raise NotImplementedError(
            f"TaskDCMPacker refuses bilinear sample sites {bilinear_keys}; "
            f"bilinear amortized inference deferred to v0.3.1 per D5. Use "
            f"SVI via create_guide(task_dcm_model) for bilinear DCM."
        )
    ...  # existing body
```

**Why both?** Defense in depth. A developer may construct bilinear `params` dicts for offline analysis without going through `amortized_task_dcm_model`; the packer-level check catches that path. The wrapper-level check is the primary user-visible message.

**Unit test (MODEL-07):**

```python
def test_amortized_wrapper_refuses_bilinear():
    packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
    with pytest.raises(NotImplementedError, match="v0.3.1"):
        amortized_task_dcm_model(
            bold, stimulus, a_mask, c_mask, t_eval, 2.0, 0.5, packer,
            b_masks=[torch.ones(3, 3).fill_diagonal_(0.0)],
            stim_mod=some_stim_mod,
        )

def test_packer_refuses_bilinear_keys():
    packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
    with pytest.raises(NotImplementedError, match="v0.3.1"):
        packer.pack({
            "A_free": torch.zeros(3, 3),
            "C": torch.zeros(3, 1),
            "noise_prec": torch.tensor(10.0),
            "B_free_0": torch.zeros(3, 3),
        })
```

**Sources:**
- `src/pyro_dcm/guides/parameter_packing.py:35-165` — `TaskDCMPacker` contract.
- `src/pyro_dcm/models/amortized_wrappers.py:155-219` — `amortized_task_dcm_model`.
- `.planning/STATE.md:37` — D5 "Amortized-guide bilinear support deferred to v0.3.1".
- `.planning/research/v0.3.0/PITFALLS.md:168-212` — B3 amortized dim mismatch rationale.

**Confidence:** HIGH.

---

## 7. `extract_posterior_params` Extension (MODEL-05)

**Current definition** (`guides.py:354-454`):

- Uses `pyro.infer.Predictive(model, guide=guide, num_samples=num_samples)` to draw samples.
- Iterates over `samples.items()` (line 436) — every sample site in the trace — and computes `mean`, `std`, `samples` per site.
- Returns `result[site_name] = {"mean": ..., "std": ..., "samples": ...}` plus a top-level `median` dict mapping site → mean.

**Key insight: `extract_posterior_params` is ALREADY site-agnostic.** Because it iterates over `samples.items()`, any newly-appearing site (including `B_free_0`, `B_free_1`, ...) will be automatically included in the output dict with `mean`, `std`, `samples`.

**MODEL-05 thus requires NO code change** — only a documentation update and a test asserting the new keys appear. The "per-modulator `B_j` medians" are available as `posterior["B_free_0"]["mean"]`, etc. For users wanting the masked/parameterized `B_j`, the model already emits `pyro.deterministic("B", B_stacked)` (Section 1) which is captured by `Predictive` and appears as `posterior["B"]["mean"]` (shape `(J, N, N)`).

**Minimal docstring addition (recommended):**

```python
"""
...
    For bilinear task DCM models (v0.3.0+), per-modulator parameters appear
    under keys ``B_free_0``, ``B_free_1``, ..., ``B_free_{J-1}`` (raw free
    parameters) and ``B`` (shape ``(J, N, N)``, masked+parameterized).
    Compute per-modulator medians as ``posterior["B_free_j"]["mean"]`` or
    ``posterior["B"]["mean"][j]``. (MODEL-05)
"""
```

**Unit test (MODEL-05):**

```python
def test_extract_posterior_includes_bilinear_sites(task_bilinear_data):
    pyro.clear_param_store()
    guide = create_guide(task_dcm_model, guide_type="auto_normal")
    result = run_svi(
        task_dcm_model, guide,
        model_args=task_bilinear_data["model_args"],
        num_steps=20,  # smoke-level, no convergence check
    )
    posterior = extract_posterior_params(
        guide, task_bilinear_data["model_args"], num_samples=10,
    )
    J = len(task_bilinear_data["b_masks"])
    for j in range(J):
        assert f"B_free_{j}" in posterior, f"Missing B_free_{j} in posterior"
        assert posterior[f"B_free_{j}"]["mean"].shape == (3, 3)
    assert "B" in posterior, "Missing deterministic B site"
    assert posterior["B"]["mean"].shape == (J, 3, 3)
```

**Sources:**
- `src/pyro_dcm/models/guides.py:354-454` — full read.

**Confidence:** HIGH (already site-agnostic; zero code churn for MODEL-05 — only docs + test).

---

## 8. Test Strategy — Files and MODEL-* Coverage

**Existing test files** (verified):

| File | Lines | Current coverage |
|------|-------|-----------------|
| `tests/test_task_dcm_model.py` | 383 | Linear `task_dcm_model` trace, shapes, SVI smoke |
| `tests/test_guide_factory.py` | ~250 | `create_guide` all 6 guide types, toy 2-site model |
| `tests/test_amortized_task_dcm.py` | N/A | Amortized wrapper (linear-only) |
| `tests/test_parameter_packing.py` | N/A | `TaskDCMPacker` linear round-trip |
| `tests/test_posterior_extraction.py` | N/A | `extract_posterior_params` |
| `tests/test_bilinear_utils.py` | 186 | `parameterize_B`, `compute_effective_A` (from Phase 13) |
| `tests/test_linear_invariance.py` | N/A | bit-exact linear short-circuit (Phase 13) |

**Minimal Phase 15 test additions** (NEW or EXTEND existing):

| Test name | File | MODEL-* covered | Cost |
|-----------|------|-----------------|------|
| `test_B_PRIOR_VARIANCE_constant` | `test_task_dcm_model.py` (new class `TestBilinearStructure`) | MODEL-02 | 1-line assert |
| `test_linear_reduction_when_b_masks_none` | `test_task_dcm_model.py` | MODEL-01, MODEL-04 | ~30 LoC; trace equal to existing linear trace |
| `test_linear_reduction_when_b_masks_empty_list` | same | MODEL-04 | ~10 LoC |
| `test_bilinear_trace_has_B_free_sites` | same | MODEL-01 | ~30 LoC; trace check for `B_free_0..J-1` |
| `test_bilinear_masking_applied` | same | MODEL-03 | ~30 LoC; B matrix element equals 0 at mask-zero indices |
| `test_bilinear_stim_mod_required_error` | same | MODEL-04 | ~10 LoC; pytest.raises ValueError |
| `test_bilinear_stim_mod_shape_mismatch_error` | same | MODEL-04 | ~15 LoC |
| `test_bilinear_svi_smoke_3region_converges` | `test_task_dcm_recovery.py` OR new `test_bilinear_pyro_model.py` | MODEL-01, MODEL-04 | ~80 LoC; **target: <60s runtime** |
| `test_b_sites_auto_discovered_all_guides` | `test_guide_factory.py` (extend) | MODEL-06 | ~40 LoC parametrized 3-guide test |
| `test_extract_posterior_includes_bilinear_sites` | `test_posterior_extraction.py` (extend) | MODEL-05 | ~30 LoC |
| `test_amortized_wrapper_refuses_bilinear_kwargs` | `test_amortized_task_dcm.py` | MODEL-07 | ~20 LoC |
| `test_packer_refuses_bilinear_keys` | `test_parameter_packing.py` | MODEL-07 | ~15 LoC |

**Total: ~12 new tests, ~310 LoC.** Matches the "minimal set to close MODEL-01..07 without bloating suite" requirement in the research question.

**<60s SVI smoke-test budget:**
- 3-region, J=1 modulator, J=1 driving input, 30s simulated BOLD at TR=2.0 → T=15 BOLD samples.
- `step_size=0.5`, `duration=30.0` → 60 ODE steps per SVI step (rk4; 240 RHS evaluations).
- 50 SVI steps × (60 × 4 + bookkeeping) ≈ **~30-45s on CPU**, well within budget.
- Verification: existing linear SVI smoke (`test_task_dcm_model.py:310-344`) runs 50 steps on 3-region in <15s. Bilinear is 3.3-5.8× slower per Pitfall B10 (Section 9). Upper-bound estimate: 15s × 5 = 75s; LOWER if J=1 and step_size=0.5. Use 40 SVI steps instead of 50 to guarantee <60s.

**Convergence criterion:** mean of last 10 ELBO < mean of first 10 ELBO (same criterion as existing `test_svi_loss_decreases`).

**Sources:**
- `tests/test_task_dcm_model.py:310-344` — existing SVI smoke pattern.
- `tests/test_bilinear_utils.py` — `parameterize_B` tests already in place.
- v0.3.0 PITFALLS.md B10 — 3.3-5.8× cost multiplier.

**Confidence:** HIGH on test inventory (grep-verified); MEDIUM on <60s upper bound (runtime varies by machine; 40-step cap provides safety margin).

---

## 9. SVI Gradient/ODE Stability Under Bilinear B

**Risk analysis (D4 interaction):**

Early SVI iterations sample `B_free_j` from a guide initialized at `N(0, init_scale=0.01)`. Even at `init_scale=0.01`, the scale parameter is UNCONSTRAINED and adapted by gradient descent; by step 50-100 scales often grow to `O(0.1-0.5)` and individual draws reach `±1σ` of the prior (= ±1.0 under B_PRIOR_VARIANCE=1.0).

**Concrete stability calculation (Gershgorin bound):**
- `A` diagonal from `parameterize_A` at free=0: `a_ii = -0.5`.
- `B_free_j` sampled at ±1.0 with `b_mask` zero-diagonal: worst-case row-sum of `u_mod[j] * B[j]` at `u_mod=1.0` = `sum_{i≠j} |B_ij|`. For `N=3`, `J=1`, max row-sum ≈ 2.0 (two off-diagonal ±1 entries).
- `A_eff` max Gershgorin real part ≈ `-0.5 + 2.0 = +1.5` → **positive; stability monitor fires**.
- Over 60-step rk4 integration, eigenvalue stays ~1.5 for duration of u_mod=1 epoch (say 10s), producing `exp(1.5 × 10) ≈ 3e6` growth in `x(t)` during the modulator ON window — and BOLD derivatives explode accordingly.

**What happens to gradients in that case?**
- **Without guard:** `solution` contains `inf`/`nan`; `pyro.sample("obs", Normal(predicted_bold, ...).to_event(2), obs=...)` log-prob becomes `-inf`; loss is NaN. Pyro's SVI loop raises `RuntimeError("NaN ELBO at step {step}")` via the guard in `run_svi` (`guides.py:335-337`).
- **With D4 monitor (log-warn only):** identical to "without guard" — the monitor does NOT prevent NaN, it only logs. D4 explicitly chose log-warn over raise to avoid corrupting gradients (raise would interrupt `svi.step`; log is trace-safe).
- **`init_scale=0.01` is the only real protection.** Section 4 of `04-RESEARCH.md` (cited in `guides.py:133-137`) establishes this as the standard mitigation: start with small guide std so early draws stay near the prior mean (zero). For bilinear, this is even more important because B can couple to A via `A_eff`.

**Recommended mitigations for Phase 15:**

1. **Lower default `init_scale` for bilinear models** (NEW decision candidate, call this **L2**): when `b_masks is not None`, use `init_scale=0.005` (half the linear default). Documented in `create_guide` docstring; no new parameter needed — callers pass explicitly.

2. **NaN-safe likelihood** (precedent from `amortized_wrappers.py:143-145`):

```python
# In task_dcm_model, after computing predicted_bold:
if torch.isnan(predicted_bold).any() or torch.isinf(predicted_bold).any():
    # Replace with zeros (detached) → finite penalty, zero gradient.
    predicted_bold = torch.zeros_like(predicted_bold).detach()
```

   This is the SAME trick used in `amortized_wrappers.py` for untrained flows. Port to `task_dcm_model` bilinear branch. **Rejected alternative:** do nothing and rely on `run_svi` NaN check — this halts SVI entirely on any unlucky draw. Too brittle for 50-100 step smoke tests.

3. **`caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")` in SVI tests** — silence the monitor's expected spurious fires during test runs (same mitigation recommended in 14-RESEARCH.md R5).

4. **Gradient clipping** — already present in `run_svi` via `ClippedAdam({"clip_norm": 10.0})`. No change needed.

**Locked decision proposal (L2):** "init_scale default for bilinear mode is 0.005 (half the linear default); passed explicitly to `create_guide(...)` in Phase 15 SVI tests. Documented but NOT auto-switched inside `create_guide` — callers retain explicit control."

**Sources:**
- `src/pyro_dcm/forward_models/coupled_system.py:308-372` — `_maybe_check_stability`.
- `src/pyro_dcm/models/amortized_wrappers.py:143-145` — NaN protection precedent.
- `src/pyro_dcm/models/guides.py:133-137, 335-337` — init_scale rationale, NaN guard.
- `.planning/STATE.md:35` — D4 text.
- v0.3.0 PITFALLS.md B1, B5 — positive eigenvalue failure modes.

**Confidence:** MEDIUM on init_scale=0.005 being sufficient (empirical from linear path at init_scale=0.01; bilinear J=1 case analogous); HIGH on the NaN-safe likelihood port.

---

## 10. Recommended Plan Decomposition

**Three plans, two waves.** Mirrors Phase 14's wave structure (which also split utilities from primary integration).

```
Wave 1 (foundation, blocks all):
┌──────────────────────────────────────────────┐
│ 15-01: task_dcm_model bilinear extension      │
│ - Add B_PRIOR_VARIANCE module constant        │
│ - Add b_masks/stim_mod kwargs (MODEL-01/02/04)│
│ - Linear short-circuit + bit-exact trace      │
│ - Validation helper (edge cases)              │
│ - ~300 LoC src + ~200 LoC test                │
└──────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
Wave 2a (parallel):    Wave 2b (parallel):
┌───────────────────┐  ┌────────────────────────────┐
│ 15-02: Guide auto-│  │ 15-03: Packer refusal +    │
│ discovery tests + │  │ extract_posterior extension│
│ bilinear SVI smoke│  │ (MODEL-05, MODEL-07)       │
│ (MODEL-06)        │  │                            │
│ ~30 LoC src       │  │ ~50 LoC src + ~60 LoC test │
│ + ~150 LoC test   │  │                            │
└───────────────────┘  └────────────────────────────┘
```

**Plan scopes:**

### 15-01: task_dcm_model bilinear extension [Wave 1]

**Scope:**
- Add `B_PRIOR_VARIANCE = 1.0` module-level constant in `task_dcm_model.py` with docstring citing D1.
- Extend `task_dcm_model` signature with `*, b_masks=None, stim_mod=None` kwargs.
- Add `_validate_bilinear_args` helper (private module function).
- Add NaN-safe `predicted_bold` guard (ported from `amortized_wrappers.py:143-145`).
- Add per-modulator `B_free_j` sampling loop + `parameterize_B` call + stacked `B` deterministic site.
- Add `merge_piecewise_inputs` call + `CoupledDCMSystem(..., B=..., n_driving_inputs=...)` wiring.
- **Tests:** trace tests (new sites appear / don't appear), shape tests, edge-case ValueError tests, 3-region bilinear SVI smoke test with decreasing ELBO (MODEL-01, MODEL-02, MODEL-03 source-side, MODEL-04).

**Closes requirements:** MODEL-01, MODEL-02, MODEL-03 (model-side), MODEL-04 (smoke).

**Depends on:** Phase 13 (complete), Phase 14 (complete).

**LoC:** ~300 src + ~250 test. Timeline estimate: 1 day.

### 15-02: Guide auto-discovery verification [Wave 2a, parallelizable]

**Scope:**
- No src changes (MODEL-06 is passive — "works without factory changes").
- Add `tests/test_guide_factory.py::TestBilinearDiscovery` (or new `test_bilinear_guide_discovery.py`).
- Parametrized test across `["auto_normal", "auto_lowrank_mvn", "auto_iaf"]` — confirms `f"B_free_{j}"` sites appear in `guide.prototype_trace.nodes` and in `pyro.get_param_store()`.
- Second test: 20-step SVI with each guide type on 3-region J=1 bilinear data to confirm no runtime errors (per-guide smoke, not convergence).

**Closes requirements:** MODEL-06.

**Depends on:** 15-01 (bilinear model must exist first).

**LoC:** 0 src + ~150 test. Timeline estimate: 0.5 day.

### 15-03: Amortized refusal + extract_posterior extension [Wave 2b, parallelizable]

**Scope:**
- Add `NotImplementedError` check in `amortized_task_dcm_model` (new `b_masks=None, stim_mod=None` kwargs; raise when non-empty).
- Add `NotImplementedError` check in `TaskDCMPacker.pack` on bilinear keys.
- Extend `extract_posterior_params` docstring (no code change needed — already site-agnostic).
- **Tests:** two refusal tests (packer + wrapper), one extension test (`B_free_j` keys present in posterior dict).

**Closes requirements:** MODEL-05 (via test + docstring), MODEL-07.

**Depends on:** 15-01 (needs `task_dcm_model` with bilinear kwargs for the posterior-extraction test).

**LoC:** ~50 src + ~80 test. Timeline estimate: 0.5 day.

**Total Phase 15:** ~2 days wall time; 2 waves; 3 plans. Matches Phase 13's 4-plan and Phase 14's 2-plan footprint.

**Rejected alternative decompositions:**

- **Alt A: 1 plan** — Rejected. Too large a diff (~600 LoC src+test combined) for a single review. Also prevents parallel work by multiple developers on 15-02 and 15-03.
- **Alt B: 4 plans** (split 15-01 into B_PRIOR_VARIANCE-only + bilinear-extension) — Rejected. B_PRIOR_VARIANCE is a 3-line change that only makes sense in conjunction with first bilinear use. Artificial fragmentation.
- **Alt C: 15-02 and 15-03 merged** — Rejected. They touch different files, have no dependency on each other, and closing MODEL-06 independently of MODEL-05/07 gives cleaner traceability.

**Confidence:** HIGH on 3-plan structure (dependency-forced); MEDIUM on parallel execution (depends on whether 15-01 commits first).

---

## 11. Pitfall Catalog — Phase 15 Specific

### Inherited from v0.3.0 PITFALLS.md

| # | Title | Phase 15 impact | Mitigation |
|---|-------|-----------------|------------|
| **B1** | `A_eff(t)` loses negative-real-part eigenvalues under sustained large `u·B` | Early SVI iterations can sample B large; NaN BOLD | NaN-safe likelihood (Section 9 mitigation 2); init_scale=0.005 (L2) |
| **B3** | Amortized guide dimension mismatch | MODEL-07 direct mitigation | Explicit `NotImplementedError` in packer + wrapper (Section 6) |
| **B5** | Free B diagonal produces positive self-coupling | MODEL-03 direct mitigation | `parameterize_B` already emits DeprecationWarning on non-zero diag (Phase 13); model-level `b_masks[j].fill_diagonal_(0.0)` documented as recommended default |
| **B8** | SPM one-state B prior is variance 1, not 1/16 | MODEL-02 locks variance 1.0 (D1) | `B_PRIOR_VARIANCE = 1.0` module constant with docstring citing D1 |
| **B10** | Per-step ODE cost 3–6× linear | SVI smoke test runtime budget | Cap at 40 SVI steps (Section 8); N=3, J=1 only for smoke |

### Phase 15-specific risks (new)

**R1: Guide parameter name collision across modulator indices.**

If any bilinear site uses the name `B_free` (no suffix) alongside `B_free_0`, Pyro silently overwrites. Low probability with the literal `f"B_free_{j}"` pattern, but catchable:

```python
# Defensive assertion at model entry:
assert all(s != "B_free" for s in []), "B_free is not a valid site name; use B_free_0"
```

**Mitigation:** always use `f"B_free_{j}"` in the loop; never bare `B_free`. Unit test: trace contains `B_free_0`, not `B_free`.

**R2: `pyro.deterministic("B", ...)` collides with linear model's `A` and linear inference.**

The linear model emits `pyro.deterministic("A", ...)`. Adding `pyro.deterministic("B", B_stacked)` in bilinear mode is fine (different name), but if `b_masks=None`, **do NOT emit a `B` deterministic site** — it would make trace-comparison tests fail against the pre-Phase-15 baseline.

**Mitigation:** guard the `pyro.deterministic("B", ...)` call inside `if b_masks is not None and len(b_masks) > 0`. Unit test: linear-mode trace does NOT contain `"B"` site.

**R3: `create_guide` blocklist (`_MAX_REGIONS = {"auto_mvn": 7}`) does not account for J.**

With bilinear, effective parameter count grows by `J*N*N`. At `N=7, J=3`, total continuous sites in AutoMultivariateNormal = `7² + 7 + 1 + 3·49 = 204` → full-rank covariance is 204×204 = 41k params. Feasible but wasteful.

**Mitigation:** document in `create_guide` docstring that `auto_mvn` is not recommended for bilinear DCM with J > 1. Do NOT auto-block at this phase — too intrusive. Log a warning if `guide_type == "auto_mvn"` and any `B_free_j` site exists (post-setup check).

**R4: `b_masks` vs `stim_mod` ordering contract ambiguity.**

When J ≥ 2, `b_masks[0]` pairs with `stim_mod.values[:, 0]`, `b_masks[1]` with `stim_mod.values[:, 1]`, etc. This ordering is IMPLICIT in the `merge_piecewise_inputs` output. If user constructs `stim_mod` with columns in a different mental order than their `b_masks` list, silent misattribution.

**Mitigation:** document loudly in `task_dcm_model` docstring: "b_masks[j] MUST pair with stim_mod.values[:, j]; no column-name validation is performed." Unit test: two-modulator case with distinct b_masks verifies column 0 of stim_mod affects b_masks[0] only.

**R5: DeprecationWarning pollution in SVI test logs.**

If `b_masks` list is passed with any non-zero diagonal, `parameterize_B` emits `DeprecationWarning` on EVERY SVI step (once per model call). With 50 steps, logs get ~50 warnings.

**Mitigation:** in all Phase 15 tests, use `b_masks[j].fill_diagonal_(0.0)` per recommended default. Reserve non-zero-diagonal tests to a single explicit `pytest.warns(DeprecationWarning)` test in `test_bilinear_utils.py` (already present from Phase 13).

**R6: stability_check_every=10 repeated warnings during SVI.**

D4 + R5 of Phase 14 flagged that the stability monitor fires freely in SIM-05 with modest B. Under SVI with `B_free ~ N(0, 1.0)` prior and bilinear-large draws, the monitor will fire frequently. Not a bug (D4 chose log-warn only), but spams test logs.

**Mitigation:** in bilinear SVI smoke test, add autouse fixture:

```python
@pytest.fixture(autouse=True)
def _silence_stability_logger(caplog):
    caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")
```

Consistent with 14-RESEARCH.md recommendation.

**R7: `simulate_task_dcm` vs `task_dcm_model` silent dt mismatch.**

Recovery tests (Phase 16 future) use `simulate_task_dcm` at `dt=0.01` to produce BOLD, then feed into `task_dcm_model` at `dt=0.5`. With bilinear B and rk4, the model's coarse dt samples fewer points of `stim_mod` — if `stim_mod` is a stick function at 0.01-wide ticks, the model's rk4 may completely miss events.

**Mitigation:** Phase 14 already requires `make_epoch_stimulus` (boxcars) for modulators (Pitfall B12). Phase 15 tests use `make_epoch_stimulus` with epoch widths ≥ `max(dt)` = 0.5s. Document this in `task_dcm_model` bilinear docstring: "stim_mod epoch widths should be ≥ the model's `dt` parameter; use `make_epoch_stimulus`."

### Risk register summary

| Risk | Severity | Plan | Status |
|------|----------|------|--------|
| B1 eigenvalue blow-up | HIGH | 15-01 | NaN-safe likelihood + init_scale=0.005 |
| B3 amortized dim mismatch | HIGH | 15-03 | Explicit refusal |
| B5 diagonal positive self-coupling | HIGH | 15-01 | Phase 13 DeprecationWarning inherited |
| B8 B variance 1 not 1/16 | HIGH | 15-01 | B_PRIOR_VARIANCE constant + test |
| B10 ODE cost 3-6× | MEDIUM | 15-01 | 40-step SVI smoke cap |
| R1 site name collision | LOW | 15-01 | Literal `f"B_free_{j}"` only |
| R2 `B` determ in linear mode | LOW | 15-01 | Guard inside bilinear branch |
| R3 auto_mvn not sized for J | LOW | 15-02 | Docstring note |
| R4 b_masks/stim_mod ordering | MEDIUM | 15-01 | Docstring + 2-modulator unit test |
| R5 DeprecationWarning spam | LOW | all | Zero diagonal in tests |
| R6 stability log spam | LOW | 15-01, 15-02 | autouse caplog silencer |
| R7 dt mismatch | MEDIUM | 15-01 | Epoch widths ≥ dt; documented |

---

## 12. Locked Decision Proposals

Following the Phase 13/14 `L1/L2/L3` naming pattern, propose these plan-level decisions:

### L1 — B-site shape is full `(N, N)` with `.to_event(2)`

**Decision:** `pyro.sample(f"B_free_{j}", dist.Normal(0, B_PRIOR_VARIANCE**0.5).expand((N, N)).to_event(2))`, NOT a flat vector of free entries.

**Rationale:** (a) mirrors `A_free`/`C` pattern in existing `task_dcm_model`; (b) MODEL-06 requires auto-discovery across three AutoGuide families without factory changes — flat-free-vector would break `AutoLowRankMVN` concatenation; (c) SPM12 `pC.B = B` convention carries full matrix variance with mask applied downstream.

**Scope:** Plan 15-01.

### L2 — init_scale default for bilinear SVI tests = 0.005

**Decision:** Phase 15 SVI tests (and downstream recovery tests) pass `init_scale=0.005` (half the linear default) when `b_masks is not None`. `create_guide` API is NOT changed — callers pass explicitly.

**Rationale:** bilinear B introduces additional early-SVI instability risk (Section 9 Gershgorin analysis). Halving init_scale is the canonical mitigation per `04-RESEARCH.md` Pitfall 1; bilinear doubles down.

**Scope:** Plan 15-01 tests, plan 15-02 SVI smoke.

### L3 — `pyro.deterministic("B", ...)` emitted ONLY in bilinear branch

**Decision:** the model emits `pyro.deterministic("B", B_stacked)` only when `b_masks is not None and len(b_masks) > 0`. In linear mode, no `"B"` site exists in the trace (preserves bit-exact trace equality with pre-Phase-15 linear `task_dcm_model`).

**Rationale:** MODEL-04 requires the linear short-circuit to reduce to current model. Adding a new deterministic site in linear mode would break that invariant and break existing `test_task_dcm_model.py::test_model_trace_has_expected_sites`.

**Scope:** Plan 15-01.

**Status of these decisions:** proposed in this research; to be locked in Plan 15-01's CONTEXT.md or directly in the plan.

---

## 13. Test Matrix — MODEL-* Coverage

| MODEL-ID | Requirement (summary) | Test(s) | Plan | File |
|----------|----------------------|---------|------|------|
| MODEL-01 | Per-modulator `B_free_j` loop, Normal(0, 1.0), site-masking | `test_bilinear_trace_has_B_free_sites`, `test_bilinear_masking_applied`, `test_bilinear_svi_smoke_3region_converges` | 15-01 | `test_task_dcm_model.py` |
| MODEL-02 | `B_PRIOR_VARIANCE = 1.0` module constant, docstring, test | `test_B_PRIOR_VARIANCE_constant` | 15-01 | `test_task_dcm_model.py` |
| MODEL-03 | `b_mask` default-zero-diagonal; non-zero triggers DeprecationWarning | `test_nonzero_diagonal_triggers_deprecation_warning` (existing Phase 13); `test_bilinear_masking_applied` (model-level) | 15-01 | `test_bilinear_utils.py` (existing) + `test_task_dcm_model.py` |
| MODEL-04 | `b_masks=None`/`[]`/shape-mismatch edge cases; SVI smoke converges | `test_linear_reduction_when_b_masks_none`, `test_linear_reduction_when_b_masks_empty_list`, `test_bilinear_stim_mod_required_error`, `test_bilinear_stim_mod_shape_mismatch_error`, `test_bilinear_svi_smoke_3region_converges` | 15-01 | `test_task_dcm_model.py` |
| MODEL-05 | `extract_posterior_params` returns per-modulator `B_j` medians | `test_extract_posterior_includes_bilinear_sites` | 15-03 | `test_posterior_extraction.py` |
| MODEL-06 | `create_guide` auto-discovers `B_free_j` across 3 guide variants | `test_b_sites_auto_discovered_all_guides` (parametrized) | 15-02 | `test_guide_factory.py` |
| MODEL-07 | `amortized_wrappers.py` + `TaskDCMPacker` refuse bilinear sites | `test_amortized_wrapper_refuses_bilinear_kwargs`, `test_packer_refuses_bilinear_keys` | 15-03 | `test_amortized_task_dcm.py`, `test_parameter_packing.py` |

**Total: 12-14 new tests across 5 test files.** Every MODEL-ID has at least one direct test.

---

## 14. Open Questions

1. **Should `create_guide` accept a `bilinear_mode: bool` kwarg** that auto-applies `init_scale=0.005` and warns on `auto_mvn`? **Recommendation: NO** — keep the factory dumb; callers pass `init_scale` explicitly per L2.

2. **Should `task_dcm_model` accept `b_masks` as a stacked `(J, N, N)` tensor** in addition to `list[Tensor]`? MODEL-01 wording says "`b_masks` default shape `(N, N)`" which reads as list-of-masks. `simulate_task_dcm` already accepts both (`_normalize_B_list` at `task_simulator.py:37-88`). **Recommendation: accept both for API parity**; normalize at entry.

3. **Bit-exact reduction claim verification.** MODEL-04 says `b_masks=None` reduces to current linear model. The linear short-circuit preserves the exact `CoupledDCMSystem(A, C, stimulus)` call, but the Pyro trace picks up the `b_masks=None` kwarg value as an `_INPUT`. Trace equality is modulo input dict. **Recommendation: assert trace nodes (not inputs) match**, i.e., same set of sample/deterministic sites with same values. Handled by the linear short-circuit test already in the matrix.

4. **Should 15-02 and 15-03 be truly parallelizable or sequential?** Both depend only on 15-01. Execution parallel is cleanest; the two developers (or two serial sessions) cannot conflict on files (15-02 touches `test_guide_factory.py`; 15-03 touches `amortized_wrappers.py`, `parameter_packing.py`, `test_posterior_extraction.py`). **Recommendation: parallel.**

5. **Does MODEL-05 require `B_j` MEDIAN specifically, or just any per-modulator summary statistic?** `extract_posterior_params` returns "mean" (computed from Predictive samples; approximates median for symmetric posteriors per docstring line 405-407). **Recommendation: satisfied by existing `mean` field; docstring updated.** If true median is required, add `torch.quantile(samples, 0.5, dim=0)` as a new `median_true` field — 5-line change.

6. **Should Phase 15 include a CONTEXT.md from `/gsd:discuss-phase 15`?** The research question suggests no CONTEXT.md exists at research time. Given the 3 proposed `L1/L2/L3` decisions, **recommendation: run `/gsd:discuss-phase 15` to lock L1/L2/L3 before plan creation.**

---

## 15. Sources

### Primary (HIGH confidence — direct code/doc read 2026-04-18)

- `src/pyro_dcm/models/task_dcm_model.py` — 183 lines, full read (current linear model baseline).
- `src/pyro_dcm/models/guides.py` — 455 lines, full read (`create_guide`, `run_svi`, `extract_posterior_params`).
- `src/pyro_dcm/models/amortized_wrappers.py` — 272 lines, full read (MODEL-07 scope).
- `src/pyro_dcm/models/rdcm_model.py` — 120 lines sampled (per-region loop precedent).
- `src/pyro_dcm/models/spectral_dcm_model.py` — sample-site grep (spectral noise sites for cross-reference).
- `src/pyro_dcm/forward_models/neural_state.py` — 336 lines, full read (`parameterize_B`, `compute_effective_A`).
- `src/pyro_dcm/forward_models/coupled_system.py` — 373 lines, full read (bilinear integration gate, stability monitor).
- `src/pyro_dcm/simulators/task_simulator.py` — 1037 lines, relevant sections read (Phase 14 precedent for bilinear wiring).
- `src/pyro_dcm/utils/ode_integrator.py` — 344 lines sampled (`merge_piecewise_inputs` API).
- `src/pyro_dcm/guides/parameter_packing.py` — full read of `TaskDCMPacker` (MODEL-07 target).
- `tests/test_task_dcm_model.py` — 383 lines, full read.
- `tests/test_bilinear_utils.py` — sampled lines 1-150.
- `tests/test_guide_factory.py` — sampled lines 1-100.
- `.planning/STATE.md` — D1, D4, D5 decisions (lines 29, 35, 37).
- `.planning/research/v0.3.0/PITFALLS.md` — B1, B3, B5, B8, B10 full read.
- `.planning/phases/14-stimulus-utilities-and-bilinear-simulator/14-RESEARCH.md` — R1-R6 risk register section.
- Pyro source verified locally via `inspect.getsource(pyro.infer.autoguide.guides.AutoGuide._setup_prototype)` and `AutoNormal._setup_prototype`.

### Secondary (MEDIUM confidence — web/doc citations)

- [Pyro AutoGuide documentation (stable)](https://docs.pyro.ai/en/stable/infer.autoguide.html) — confirms general auto-discovery mechanism (though fetch returned 403; source verified locally instead).
- [Pyro autoguide guides module source (mirrored docs)](https://pyro4ci.readthedocs.io/en/latest/_modules/pyro/infer/autoguide/guides.html) — `_setup_prototype` mechanism reference.
- [Pyro Feature Request: Named Normal Autoguide (GitHub #2026)](https://github.com/pyro-ppl/pyro/issues/2026) — discusses per-site guide configuration (not directly blocking).

### Tertiary (LOW confidence — not load-bearing)

- SPM12 `spm_dcm_fmri_priors.m` prior convention (cited via v0.3.0 PITFALLS.md B8, not re-verified this phase).

---

## Metadata

**Confidence breakdown:**
- Model structure & B_free_j pattern (Section 1): HIGH — pattern verified against existing code (`A_free`, `C`, rDCM loop).
- Forward plumbing (Section 2): HIGH — Phase 14 simulator is direct precedent.
- Guide auto-discovery (Section 3): HIGH — Pyro source code directly inspected.
- B-site shape (Section 4): HIGH — MODEL-06 constraint forces A; rDCM precedent for loop.
- Edge cases (Section 5): HIGH — mirrors `_normalize_B_list` Phase 14.
- Packer refusal (Section 6): HIGH — defensive check is trivial.
- extract_posterior extension (Section 7): HIGH — already site-agnostic; zero churn.
- Test strategy (Section 8): HIGH on inventory; MEDIUM on <60s upper bound (machine-dependent).
- SVI stability (Section 9): MEDIUM — init_scale=0.005 is empirical extrapolation; NaN-safe likelihood is proven precedent.
- Plan decomposition (Section 10): HIGH — dependency-forced structure.
- Phase-specific pitfalls (Section 11): HIGH on inherited pitfalls; MEDIUM on Phase-15-new risks (speculative severities).
- Locked decision proposals (Section 12): MEDIUM — L2 in particular needs empirical confirmation in 15-01 tests.

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days; stable upstream deps — Pyro 1.9+, torchdiffeq unchanged; Phase 13/14 landed).

## RESEARCH COMPLETE
