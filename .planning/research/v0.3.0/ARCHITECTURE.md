# Architecture Research: Bilinear DCM Extension (v0.3.0)

**Domain:** Bilinear extension to existing linear DCM forward stack
**Researched:** 2026-04-17
**Confidence:** HIGH (grounded in direct file reads)
**Scope:** Integration strategy for B-matrix modulatory inputs into the existing
pyro_dcm architecture. Decides what to modify in place vs. add new vs. fork.

---

## Summary

The bilinear extension touches a narrow, well-bounded slice of the codebase:
the task-DCM chain (`neural_state -> coupled_system -> task_dcm_model ->
task_simulator -> amortized_wrappers -> benchmarks/runners/task_*`). Spectral
DCM and rDCM are architecturally independent — they never instantiate
`NeuralStateEquation` or `CoupledDCMSystem`, so they are untouched by the
extension. This dramatically reduces the blast radius.

**Central recommendation:** extend existing classes/functions with optional
`B_list` + `u_mod` parameters (defaulting to `None` = linear mode) rather than
forking new `Bilinear*` variants. The math is a clean superset
(`dx/dt = (A + Σ u_j·B_j) x + Cu` reduces exactly to `Ax + Cu` when
`B_list is None`), every existing call site defaults cleanly, and test churn
is zero for existing test suites.

**The only genuinely new artifact** is a small helper for the effective
A computation (`compute_effective_A`) plus a stimulus utility for the
modulatory input time-series. Everything else is additive parameters on
existing signatures.

**Build order** is dictated by the forward-data dependency: neural_state must
land first (it's imported by coupled_system, task_dcm_model, task_simulator,
and amortized_wrappers), then simulator (so recovery tests have ground truth),
then Pyro model (so SVI works), then benchmark (needs all three).

---

## File-by-File Change Plan

### Modified in place (backwards-compatible)

| File | Change | Breaking? | Rationale |
|------|--------|-----------|-----------|
| `src/pyro_dcm/forward_models/neural_state.py` | Add `parameterize_B(B_free, b_mask)` alongside `parameterize_A`. Add `compute_effective_A(A, B_list, u_mod) -> A_eff`. Extend `NeuralStateEquation.__init__` with optional `B_list=None, u_mod_fn=None`. Extend `derivatives(x, u)` to compute `(A + Σ u_mod[j]*B_list[j]) @ x + C @ u` when B is provided. | No — all new args default to `None` and the fallback path is exactly `self.A @ x + self.C @ u`. | `NeuralStateEquation` is an 8-line class; a subclass would duplicate ~all of it. Optional args with a None-branch is the minimum-churn path. Existing `test_neural_state.py` constructs with `(A, C)` only and will pass unchanged. |
| `src/pyro_dcm/forward_models/coupled_system.py` | Extend `CoupledDCMSystem.__init__` with optional `B_list=None, input_mod_fn=None`. Register `B_list` as a buffer list (or a stacked `B` tensor, shape `(K, N, N)`) when provided. Pass through to the inner `NeuralStateEquation`. The `forward(t, state)` method gains one line: evaluate `u_mod = input_mod_fn(t)` when non-None. | No — defaults preserve exact current behavior. Buffer registration must use `register_buffer` per-entry or stack to a `(K, N, N)` tensor to stay compatible with `torchdiffeq.odeint_adjoint`. | `test_ode_integrator.py` constructs `CoupledDCMSystem(A, C, input_fn)` in 6+ tests — all must keep passing. The module is itself a thin adapter; extending it is cheaper than subclassing and matches how `hemo_params` is already optional. |
| `src/pyro_dcm/models/task_dcm_model.py` | Add optional `b_masks: torch.Tensor \| None = None` (shape `(K, N, N)`) and `stim_mod: PiecewiseConstantInput \| None = None` to the signature. When `b_masks is not None`: sample `B_free ~ N(0, 1/16)` for each modulator, apply per-modulator masking, call `parameterize_B`, pass `B_list + stim_mod` into `CoupledDCMSystem`. When `None`: current linear code path runs unchanged. | No — the two new args default to `None`. Existing `test_task_dcm_model.py` calls (via `pyro.poutine.trace`) don't pass `b_masks` or `stim_mod` and continue to exercise the linear path. | Creating a separate `bilinear_task_dcm_model` would duplicate ~80 lines of identical prior sampling, ODE integration, BOLD extraction, and likelihood. The bilinear block is genuinely additive — 10 lines of conditional sampling + one extra kwarg into `CoupledDCMSystem`. Model comparison (ELBO) between linear and bilinear is naturally expressed as "same model, `b_masks=None` vs. `b_masks=<mask>`". |
| `src/pyro_dcm/simulators/task_simulator.py` | Extend `simulate_task_dcm` signature with optional `B_list=None, stimulus_mod=None` (dict or `PiecewiseConstantInput`). Build the `input_mod_fn` the same way `input_fn` is built. Forward to `CoupledDCMSystem(A, C, input_fn, hemo_params, B_list=..., input_mod_fn=...)`. Return dict gains `params['B']` and `params['stimulus_mod']` entries. Add module-level `make_event_stimulus(events, amplitudes, ...)` helper (stick/boxcar variable-amplitude). | No — new args default to `None`. Existing `test_task_simulator.py` calls (40+ of them) pass only `A, C, stimulus` and keep working. | Same logic as `task_dcm_model`: the pipeline (ODE integrate, downsample, noise) is identical for linear and bilinear; only the ODE RHS changes. Duplicating the simulator would produce two near-identical 200-line functions. |
| `src/pyro_dcm/models/amortized_wrappers.py` | `_run_task_forward_model` gains optional `B_list, stim_mod` kwargs; `amortized_task_dcm_model` threads them through. `TaskDCMPacker` (in `guides/parameter_packing.py`) may need a bilinear variant OR extension to pack `B_free` alongside `A_free, C, noise_prec`. | Low — if the amortized pipeline is not required in v0.3.0 (YAML doesn't list amortized training as a v0.3.0 deliverable), this file can be gated behind a "phase 2" follow-up. | Amortized wrappers currently reference `CoupledDCMSystem` directly. Keeping them pure linear while the core supports bilinear is acceptable if the milestone scope allows deferral. |
| `src/pyro_dcm/forward_models/__init__.py` | Export `parameterize_B`, `compute_effective_A` alongside existing exports. | No. | Single-line additions. |
| `src/pyro_dcm/models/__init__.py` (if it re-exports) | No change needed — `task_dcm_model` symbol unchanged. | No. | Signature extension, same symbol. |

### Added (new files)

| File | Purpose | Size Estimate | Rationale |
|------|---------|---------------|-----------|
| `benchmarks/runners/task_bilinear.py` | New benchmark runner following the `run_task_svi(config)` pattern. Simulates data with known `A + B + stim_mod`, runs SVI with `b_masks` supplied, extracts posterior, computes RMSE/coverage for both A and B. | ~250 lines (mirrors `task_svi.py` ~400 lines, minus boilerplate). | `task_svi.py` is tight on the linear path and its config/fixtures don't carry `B_list`. A parallel runner keeps the linear benchmark locked at v0.2.0 results and introduces the bilinear recovery metric independently. Reuses `benchmarks.metrics`, `benchmarks.config`, `benchmarks.fixtures` — no benchmark-infra changes needed. |
| `tests/test_bilinear_neural_state.py` | Unit tests for `parameterize_B`, `compute_effective_A`, and `NeuralStateEquation` in bilinear mode. | ~150 lines. | Pure additive — doesn't touch existing `test_neural_state.py` linear tests. |
| `tests/test_bilinear_task_dcm_model.py` | Model-trace tests for the bilinear path of `task_dcm_model` (sites `B_free_0..K-1` exist, shapes right, ODE stability). | ~200 lines. | Parallel to existing `test_task_dcm_model.py` but exercising `b_masks != None`. |
| `tests/test_bilinear_simulator.py` | End-to-end simulator tests with a modulatory input producing context-dependent BOLD. | ~150 lines. | Parallel to `test_task_simulator.py`. |
| `tests/test_task_bilinear_recovery.py` | Recovery benchmark test (RMSE on A and B, coverage). | ~180 lines. | New acceptance criterion for the milestone. |

### Unaffected (no changes)

| File / Directory | Why Not Touched |
|------------------|------------------|
| `src/pyro_dcm/forward_models/balloon_model.py`, `bold_signal.py` | Hemodynamic cascade depends only on neural activity `x`, not on how `x` was generated. The bilinear term changes `dx/dt`, not the x→BOLD map. |
| `src/pyro_dcm/forward_models/spectral_transfer.py`, `csd_computation.py`, `spectral_noise.py` | Spectral DCM uses a linearized transfer function `H(w) = (iwI - A)^-1` and cannot natively represent time-varying modulation — bilinear terms are fundamentally a time-domain construct. Out of scope for v0.3.0. |
| `src/pyro_dcm/forward_models/rdcm_forward.py`, `rdcm_posterior.py` | rDCM is a frequency-domain regression — bilinear terms would require a fundamental re-derivation (see [REF-022] if ever attempted). Out of scope. |
| `src/pyro_dcm/models/spectral_dcm_model.py`, `rdcm_model.py` | Never import `NeuralStateEquation` or `CoupledDCMSystem`. Verified: `spectral_dcm_model.py` imports only `parameterize_A` + `spectral_dcm_forward`; `rdcm_model.py` imports only `pyro` + `torch`. |
| `src/pyro_dcm/models/guides.py` | Guide factory is model-agnostic. An `AutoNormal` guide over `task_dcm_model` will auto-discover the new `B_free_*` sample sites without code changes. |
| `src/pyro_dcm/simulators/spectral_simulator.py`, `rdcm_simulator.py` | Not used in task-DCM pipeline. |
| `src/pyro_dcm/inference/svi_runner.py`, `model_comparison.py`, `nuts_validator.py` | Model-agnostic. Model comparison between linear and bilinear DCMs falls out naturally (two traces of the same `task_dcm_model` with different `b_masks`). |
| `benchmarks/runners/spectral_*.py`, `rdcm_vb.py`, `spm_reference.py` | Not task-DCM. |

---

## New Components (Justifications)

### `parameterize_B(B_free, b_mask) -> B` — in `neural_state.py`

**What:** Applies the structural mask to free B parameters and returns the
modulatory matrix ready for use. Unlike A, B has no diagonal constraint
(modulation is directional, not self-inhibitory), so this is essentially
`B_free * b_mask` with a tiny wrapper for symmetry of API.

**Why alongside `parameterize_A` (not new module):** A and B share the same
mathematical role — both are NxN connectivity matrices living in the neural
state equation. A utility module like `bilinear_transform.py` would be a
one-function file. The pair `parameterize_A` / `parameterize_B` reads more
naturally as "connectivity parameterizations in one place."

### `compute_effective_A(A, B_list, u_mod) -> A_eff` — in `neural_state.py`

**What:** Computes `A_eff(t) = A + Σ_j u_mod[j] · B_list[j]`. This is the
time-varying effective connectivity at a given instant.

**Why in `neural_state.py` (not its own module):** It's a 3-line function
whose sole caller is `NeuralStateEquation.derivatives`. Giving it its own
file would be over-structured. It belongs in the same module as the equation
it supports. Tests can target it directly since it's a pure function.

**Shape contract:**
- `A`: `(N, N)`
- `B_list`: `(K, N, N)` stacked tensor (preferred over Python list for
  `torchdiffeq` adjoint compatibility)
- `u_mod`: `(K,)` scalar modulator values at time `t`
- Returns `(N, N)`

Implementation is a single einsum: `A + torch.einsum('k,kij->ij', u_mod, B_list)`.

### `make_event_stimulus(onsets, durations, amplitudes, ...)` — in `task_simulator.py`

**What:** Variable-amplitude stick (delta) or boxcar (epoch) stimulus for
the modulatory input channel. Sibling of existing `make_block_stimulus`.

**Why alongside `make_block_stimulus`:** Both produce `dict[str, Tensor]`
consumable by `PiecewiseConstantInput`. Keeping stimulus helpers in one
place matches existing convention.

### `benchmarks/runners/task_bilinear.py`

**What:** New runner that mirrors `task_svi.py` but supplies `B_list`,
`stimulus_mod`, and `b_masks` through the simulate→infer→measure loop.

**Why new file (not extending `task_svi.py`):** The current
`task_svi.py::run_task_svi` is called by the benchmark orchestrator with a
fixed `config.py` contract. Adding a "bilinear mode" flag would ripple
through fixtures and plotting. A separate runner keeps v0.2.0 baseline
numbers locked and avoids cross-cutting changes to benchmark infrastructure.

---

## Data Flow Changes

### Propagation of new arguments

```
simulate_task_dcm(A, C, stimulus, B_list, stimulus_mod)
        │
        ▼
CoupledDCMSystem(A, C, input_fn, hemo_params, B_list, input_mod_fn)
        │
        ▼ .forward(t, state)
NeuralStateEquation.derivatives(x, u)
        │
        ├─ u_mod = self.input_mod_fn(t)    (if bilinear mode)
        ├─ A_eff = compute_effective_A(self.A, self.B_list, u_mod)
        └─ dx/dt = A_eff @ x + C @ u
```

### Pyro model flow

```
task_dcm_model(observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt,
               b_masks=None, stim_mod=None)
        │
        ├─ sample A_free ~ N(0, 1/64) → parameterize_A → A              (unchanged)
        ├─ sample C     ~ N(0, 1)                                       (unchanged)
        ├─ IF b_masks is not None:
        │     for j in range(K):
        │         sample B_free_{j} ~ N(0, 1/16)
        │         B_j = B_free_{j} * b_masks[j]                         (new)
        │     B_list = stack(B_0..B_{K-1})
        │ ELSE: B_list = None                                           (linear fallback)
        │
        ├─ CoupledDCMSystem(A, C, stimulus, B_list=B_list,
        │                    input_mod_fn=stim_mod)                     (extended)
        ├─ integrate_ode → extract lnv, lnq → bold_signal               (unchanged)
        └─ likelihood obs ~ N(predicted_bold, noise_std)                (unchanged)
```

### Tensor shape conventions (new)

| Tensor | Shape | Description |
|--------|-------|-------------|
| B_list | (K, N, N) | Stacked modulatory matrices (K modulators) |
| b_masks | (K, N, N) | Binary structural masks for B, one per modulator |
| u_mod | (K,) | Modulatory input values at a single time `t` |
| stim_mod values | (L, K) | Modulator time-series waypoints (L waypoints) |
| B_free | (K, N, N) | Free parameters before masking |

K = number of modulatory inputs (distinct from M = number of driving inputs).
Recommend K ≤ M in practice and the structural mask enforces
"modulator j can only modulate connections it's entitled to affect."

---

## Build Order (Dependency-Aware)

### Phase 1: `DCM.1` — Bilinear neural state (FOUNDATION — must land first)

**Deliverables:**
- `parameterize_B`, `compute_effective_A` in `neural_state.py`
- `NeuralStateEquation.__init__` accepts `B_list, input_mod_fn`
- `NeuralStateEquation.derivatives` uses `compute_effective_A` in bilinear mode
- `tests/test_bilinear_neural_state.py` passing
- `__init__.py` exports updated

**Why first:** 4 downstream modules (`coupled_system`, `task_simulator`,
`task_dcm_model`, `amortized_wrappers`) import from this file. Nothing else
can start until this is green.

**Exit gate:** Existing `test_neural_state.py` still passes unchanged
(backwards-compat proof).

### Phase 2: `DCM.2a` — CoupledDCMSystem extension

**Deliverables:**
- `CoupledDCMSystem.__init__` accepts `B_list, input_mod_fn`
- `forward(t, state)` calls the extended `NeuralStateEquation`
- Buffer registration correct for `odeint_adjoint`
- New bilinear test cases in `test_ode_integrator.py` OR a new
  `test_bilinear_coupled_system.py`

**Why before simulator/model:** Both downstream callers construct
`CoupledDCMSystem` directly.

**Exit gate:** Existing `test_ode_integrator.py` (6+ `CoupledDCMSystem`
tests) passes unchanged.

### Phase 3: `DCM.4` — Simulator extension (ground truth for recovery)

**Deliverables:**
- `simulate_task_dcm` signature extended
- `make_event_stimulus` helper
- `tests/test_bilinear_simulator.py` passing with a known
  `A + B + stim_mod` producing expected context-dependent BOLD

**Why before Pyro model:** Parameter recovery tests need a simulator that
produces data from known bilinear ground truth. The Pyro model is the
*inverse* of the simulator — you can't validate it without the forward.

**Exit gate:** Existing `test_task_simulator.py` (all 40+ tests) passes
unchanged.

### Phase 4: `DCM.3` — Stimulus utility polish (can be folded into DCM.4)

**Deliverables:**
- Amplitude-aware event/boxcar stimulus with tests.

**Why can-fold:** `make_event_stimulus` is small enough to land with the
simulator extension. Keeping it as a nominal separate phase helps
documentation but doesn't change the build graph.

### Phase 5: `DCM.2b` — Pyro generative model extension

**Deliverables:**
- `task_dcm_model` signature extended with `b_masks, stim_mod`
- `B_free_j ~ N(0, 1/16)` prior per modulator
- `tests/test_bilinear_task_dcm_model.py` — trace structure, shapes, stability

**Why after simulator:** The Pyro model tests condition on known parameters
and run under a trace poutine. They use fixtures built from
`simulate_task_dcm`, which needs the bilinear signature first.

**Exit gate:** Existing `test_task_dcm_model.py` (trace, stability, SVI
smoke tests) passes unchanged.

### Phase 6: `DCM.V1` — Recovery benchmark runner

**Deliverables:**
- `benchmarks/runners/task_bilinear.py`
- `tests/test_task_bilinear_recovery.py`
- Recovery metric thresholds (e.g., `A_RMSE < 0.05`, `B_RMSE < 0.08`,
  `coverage_95 in [0.90, 1.00]`)

**Why last:** Needs all prior phases green. It is the milestone's acceptance
test.

### Phase 7 (optional, may defer): Amortized wrapper extension

**Deliverables:**
- `_run_task_forward_model` accepts `B_list, stim_mod`
- `TaskDCMPacker` packs `B_free` alongside `A_free, C, noise_prec`
- Amortized recovery smoke test

**Why deferrable:** YAML doesn't list amortized training as a v0.3.0
requirement, and `guides/parameter_packing.py::TaskDCMPacker` (per grep)
currently packs only `A_free, C, noise_prec`. Extending it is a
non-trivial shape change that would benefit from its own phase.

---

## Backwards-Compatibility Analysis

### Existing callers of `NeuralStateEquation`

| Caller | Call signature | Breaks? |
|--------|----------------|---------|
| `coupled_system.py:97` | `NeuralStateEquation(self.A, self.C)` | No — 2-arg positional; new args are kwarg-only with defaults |
| `test_neural_state.py:80,95,102,131` | `NeuralStateEquation(test_A, test_C)` | No — same. 4 tests pass unchanged. |

### Existing callers of `CoupledDCMSystem`

| Caller | Call signature | Breaks? |
|--------|----------------|---------|
| `task_dcm_model.py:147` | `CoupledDCMSystem(A, C, stimulus)` | No — hemo_params already optional; new args kwarg-only |
| `task_simulator.py:158` | `CoupledDCMSystem(A_dev, C_dev, input_fn, hemo_params)` | No |
| `amortized_wrappers.py:128` | `CoupledDCMSystem(A, C, stimulus)` | No |
| `test_ode_integrator.py` (6 tests) | `CoupledDCMSystem(test_A, test_C, input_fn)` and variants | No |

### Existing callers of `task_dcm_model`

| Caller | Behavior | Breaks? |
|--------|----------|---------|
| `test_task_dcm_model.py` (3 test classes, ~10 tests) | Calls via `pyro.poutine.trace(...).get_trace(observed_bold=..., stimulus=..., a_mask=..., c_mask=..., t_eval=..., TR=..., dt=...)` — all keyword | No — new `b_masks`, `stim_mod` default to None |
| `test_task_dcm_recovery.py` | `svi.step(...)` with same kwargs | No |
| `benchmarks/runners/task_svi.py` | `run_svi(task_dcm_model, guide, model_args=(bold, stim, a_mask, c_mask, t_eval, TR, dt))` — positional tuple of 7 | No — positional tuple remains 7-tuple |

### Existing callers of `simulate_task_dcm`

| Caller | Typical call | Breaks? |
|--------|--------------|---------|
| `test_task_simulator.py` (~40 tests) | `simulate_task_dcm(A, C, stimulus, duration=..., seed=..., ...)` | No — `B_list, stimulus_mod` default None |
| `test_task_dcm_model.py` fixture | `simulate_task_dcm(A, C, stim, duration=30.0, ...)` | No |
| `test_task_dcm_recovery.py:189` | `simulate_task_dcm(...)` | No |
| `benchmarks/runners/task_svi.py` | `simulate_task_dcm(...)` | No |

**Conclusion:** Zero existing tests need updating. All churn is additive.

---

## Test Impact Analysis

### Tests that run UNCHANGED (must stay green — regression gates)

| Test file | # tests involving linear DCM | Gate status |
|-----------|------------------------------|--------------|
| `test_neural_state.py` | ~10 | Must pass unchanged |
| `test_ode_integrator.py` | ~15 (6 `CoupledDCMSystem`) | Must pass unchanged |
| `test_task_dcm_model.py` | ~10 | Must pass unchanged |
| `test_task_simulator.py` | ~40 | Must pass unchanged |
| `test_task_dcm_recovery.py` | ~5 | Must pass unchanged |
| `test_amortized_task_dcm.py` | ~N | Must pass unchanged (if amortized wrappers untouched this phase) |
| `test_spectral_*`, `test_rdcm_*`, `test_spm_*`, `test_tapas_*` | Many | Must pass unchanged (architecturally untouched) |

### Tests that are PURELY NEW (additive)

| Test file | Scope |
|-----------|-------|
| `test_bilinear_neural_state.py` | `parameterize_B` masking, `compute_effective_A` shape + values, `NeuralStateEquation` bilinear derivatives, linear fallback when `B_list=None` |
| `test_bilinear_coupled_system.py` (or additions to `test_ode_integrator.py`) | 500s ODE stability with bilinear term, adjoint-method compatibility |
| `test_bilinear_task_dcm_model.py` | Trace sites (`B_free_0..K-1` present), A + B diagonal/masking invariants, SVI smoke test |
| `test_bilinear_simulator.py` | Context-dependent BOLD magnitude (modulator ON vs OFF produces distinguishable traces), shapes, no-NaN at 500s |
| `test_task_bilinear_recovery.py` | RMSE + coverage on A and B across ~10 datasets |

### Tests that MIGHT need minor additions (not updates)

| Test file | Potential addition |
|-----------|---------------------|
| `test_ode_integrator.py` | One or two cases asserting `CoupledDCMSystem(A, C, input_fn)` ≡ `CoupledDCMSystem(A, C, input_fn, B_list=None, input_mod_fn=None)` — proves no behavioral drift. |
| `test_guide_factory.py` | Optional smoke test: `create_guide(task_dcm_model)` still discovers all sites when `b_masks` is passed. Not strictly required. |

**Blast radius summary:** 0 existing tests need edits; 5 new test files; ~10
optional additive smoke tests.

---

## Architectural Patterns (to follow)

### Pattern 1: None-default optional extension

**When:** Adding a new capability to an existing function/class where the
old behavior is a strict mathematical special case of the new behavior.

**Rule:** The new kwarg defaults to `None`. A single `if x is None: ...`
branch short-circuits to the old code path. Existing callers never see
new behavior.

**Used by:** All five `Modified in place` entries above.

### Pattern 2: Buffer for stacked tensors, not `nn.ParameterList`

**When:** Passing `B_list` through `CoupledDCMSystem` (which is an
`nn.Module` for `torchdiffeq.odeint_adjoint` compatibility).

**Rule:** Stack `B_list` into a single `(K, N, N)` tensor and call
`self.register_buffer("B", stacked)`. Avoid `nn.ParameterList` — the Pyro
model handles parameterization; the ODE module only *computes*.

**Why:** Matches how `A` and `C` are registered (see `coupled_system.py:90-91`).

### Pattern 3: Per-site Python loop for variable-count sample sites

**When:** Sampling `K` modulator B matrices inside a Pyro model where `K`
may differ across invocations.

**Rule:** Use a Python `for j in range(K):` loop with site names
`f"B_free_{j}"` — NOT `pyro.plate` over modulators, because each
modulator has its own mask and the plate assumes exchangeability.

**Precedent:** `rdcm_model.py:101-151` already uses this pattern for
per-region parameters, and its docstring explicitly documents why
`pyro.plate` is not used. The same justification applies here.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Forking a `BilinearNeuralStateEquation` class

**Why tempting:** Keeps the linear class "pure."

**Why wrong:** `NeuralStateEquation` is 8 lines. A subclass would
duplicate the constructor + override `derivatives`, yielding two
classes that share 90% of their surface and must be kept in sync.
`CoupledDCMSystem` would then need a branch on `isinstance(self.neural,
BilinearNeuralStateEquation)` — complexity explodes.

**Do instead:** Optional None-default args (Pattern 1).

### Anti-Pattern 2: Putting `compute_effective_A` in its own module

**Why tempting:** "One-function-per-file" looks tidy.

**Why wrong:** It's a 3-line pure function whose only caller is
`NeuralStateEquation.derivatives`. Separate module adds import overhead
and a second file to find when reading the equation.

**Do instead:** Co-locate in `neural_state.py` (Pattern: high cohesion).

### Anti-Pattern 3: Modulator as extra columns of existing C

**Why tempting:** "Just pack modulators into C."

**Why wrong:** C contributes additively (`+C @ u`) while B contributes
multiplicatively (`+(u_j B_j) @ x`). These are distinct operators in
Eq. 1. Conflating them produces an unidentifiable model and breaks
[REF-001] conformance.

**Do instead:** Keep C and B cleanly separated in both the forward model
and the Pyro priors.

### Anti-Pattern 4: Sampling a full `B ~ N(0, σ², shape=(K,N,N))` as one plate

**Why tempting:** Fewer sample sites → simpler guide.

**Why wrong:** Per-modulator structural masks (`b_masks[j]`) may differ,
and later model comparison (does modulator 0 affect connection i→j?)
requires per-modulator site identity.

**Do instead:** Loop over modulators with distinct site names (Pattern 3).

---

## Integration Points

### External (unchanged)

| Integration | Still works? |
|-------------|---------------|
| torchdiffeq `odeint_adjoint` | Yes — B registered as buffer |
| Pyro `AutoNormal` / `AutoLowRankMVN` / `AutoIAFNormal` guides | Yes — new `B_free_*` sites auto-discovered |
| Pyro `Trace_ELBO`, `TraceMeanField_ELBO` | Yes — model remains trace-friendly |
| `benchmarks.config.BenchmarkConfig` | Yes — new runner adds its own config fields |

### Internal (new cross-module dependencies)

| From → To | New Import |
|-----------|------------|
| `neural_state.py` → `torch.einsum` | native, no new dep |
| `task_dcm_model.py` → `parameterize_B`, `compute_effective_A` | new symbol imports |
| `task_simulator.py` → no new imports | existing `PiecewiseConstantInput` reused for `stim_mod` |
| `benchmarks/runners/task_bilinear.py` → `task_dcm_model` (with b_masks) | new runner only |

---

## Confidence Assessment

| Claim | Confidence | Evidence |
|-------|-----------|----------|
| Spectral DCM and rDCM do not use `NeuralStateEquation` or `CoupledDCMSystem` | HIGH | Direct grep + file read — `spectral_dcm_model.py` imports only `parameterize_A` + `spectral_dcm_forward`; `rdcm_model.py` imports neither. |
| Existing test fixture calls do not pass `B_list` / `b_masks` | HIGH | Read `test_neural_state.py`, `test_task_dcm_model.py`, `test_task_simulator.py` end-to-end. |
| Backwards-compat of optional None-default args | HIGH | Python positional/keyword arg semantics; Pyro trace mechanism does not record absent kwargs. |
| `parameterize_B` has no diagonal constraint | MEDIUM | Standard DCM convention (modulator doesn't self-inhibit — that's A's role). Final answer should be cross-checked against [REF-001] Eq. 1 conventions in REFERENCES.md at implementation time. |
| Stacking `B_list` as `(K, N, N)` buffer is adjoint-compatible | MEDIUM | By analogy with A, C (already buffers). Direct verification advisable during DCM.2a implementation. |
| `task_dcm_model` signature extension doesn't break `run_svi` positional tuple | HIGH | `model_args` in `run_svi` is passed via `svi.step(*model_args)` — new kwargs default to None, old 7-tuple remains valid. |
| Amortized wrappers can be deferred to a follow-up phase | MEDIUM | YAML scope doesn't list amortized training as v0.3.0 — user confirmation advisable. |
| `TaskDCMPacker` in `guides/parameter_packing.py` currently packs only `A_free, C, noise_prec` | HIGH | Direct grep: line 43 confirms `[A_free.flatten(), C.flatten(), log(noise_prec)]`. |
| Recommended `B_free ~ N(0, 1/16)` prior variance | LOW | Taken from user-provided YAML scope line; stricter than A's `N(0, 1/64)`, matching the Friston/SPM convention that B effects are smaller than baseline A. Should be cross-checked against SPM12 `spm_dcm_fmri_priors.m` during implementation. |

---

## Quality Gate Checklist

- [x] Integration points clearly identified file-by-file (Modified / Added / Unaffected tables)
- [x] New vs modified components explicit with 1-line rationale each
- [x] Build order justified (dependency graph + exit gates)
- [x] Backwards-compatibility analysis for all existing callers of the 4 central symbols
- [x] Grounded in actual file contents (`neural_state.py`, `coupled_system.py`, `task_dcm_model.py`, `task_simulator.py`, `test_neural_state.py`, `test_task_dcm_model.py`, `test_task_simulator.py`, `test_ode_integrator.py` refs, `amortized_wrappers.py`, `parameter_packing.py` refs, `spectral_dcm_model.py`, `rdcm_model.py`, `guides.py`, `task_svi.py` read directly)

---

## Sources

- Direct reads of repository source under `src/pyro_dcm/forward_models/`,
  `src/pyro_dcm/models/`, `src/pyro_dcm/simulators/`, `tests/`,
  `benchmarks/runners/` (HIGH confidence for all file-by-file claims).
- `.planning/PROJECT.md` for v0.3.0 scope definition.
- Provided YAML `GSD_pyro_dcm.yaml` for proposed file touches.
- [REF-001] Friston, Harrison & Penny (2003), Eq. 1 — for bilinear form
  mathematical reduction to linear when B is absent.

---
*Architecture research for: Bilinear DCM extension (v0.3.0)*
*Researched: 2026-04-17*
