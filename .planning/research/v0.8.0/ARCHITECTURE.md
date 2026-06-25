# Architecture — v0.8.0 DCM for Evoked Responses (CMC EEG/MEG ERP)

**Domain:** Time-domain CMC neural-mass → evoked → single-dipole lead-field → scalp-ERP
forward stack, bolted onto the existing Pyro-DCM framework with SPM12 parity per phase.
**Researched:** 2026-06-25
**Mode:** Architecture (integration design)
**Overall confidence:** HIGH (read against actual `src/pyro_dcm/` source; protocol methods,
`-exp` convention, packer/wrapper/simulator idioms, and SPM bridge all located in-tree)

---

## Executive Summary

The ERP stack is **almost entirely additive**. The repo already has the exact extension
seam it needs: a `ForwardModel` **Protocol** (`src/pyro_dcm/inference/forward_models.py`)
on which the model-agnostic Variational Laplace engine
(`_run_vl_generic` / `run_variational_laplace_generic`) dispatches. The VL engine never
references spectral/task internals directly — it only calls the protocol's eight members.
A new `ERPDCMForward` class implementing that protocol gives us VL reuse **for free**,
exactly as `LatentCircuitForward` (the v0.6.0 addition) did. The same is true of the
amortized path (`guides/parameter_packing.py` packer classes + `models/amortized_wrappers.py`
`_sample_latent_and_unpack`) and the SPM `.mat` bridge (`validation/export_to_mat.py` +
`validation/matlab_scripts/` + `cluster/scripts/spm_cross_validation.py`).

**The one true structural novelty** (confirmed against `utils/ode_integrator.py`): the
existing integrator is a thin torchdiffeq wrapper (`rk4`/`dopri5`). SPM integrates ERPs
with `spm_int_L` (frozen-Jacobian exponential-Euler). These are different algorithms;
torchdiffeq **cannot** reach SPM parity. So we add **one new integrator module**
(`utils/local_linearization.py`) rather than touching `ode_integrator.py`. Keeping
`ode_integrator.py` untouched is what preserves bit-exact backward compat for the
fMRI/spectral/rDCM/latent-circuit paths.

**Net file footprint:** ~8 new files + ~6 additive insertions into existing files (new
classes/functions only, no edits to existing class bodies). Zero churn to any existing
forward/inference/model path.

---

## 1. The Integration Seam (located in source)

### 1.1 `ForwardModel` protocol — the contract the CMC forward must satisfy

`src/pyro_dcm/inference/forward_models.py:30-117` defines a `@runtime_checkable`
`Protocol`. The VL core (`variational_laplace.py:970 _run_vl_generic`) calls exactly these
members and nothing else:

| Member | Signature | What ERP must return |
|--------|-----------|----------------------|
| `residual_is_complex` (property) | `-> bool` | **`False`** (time-domain real residuals, like `TaskDCMForward`/`LatentCircuitForward`) |
| `param_count(n_regions)` | `-> int` | total free params: A{1..4}+C+T(4)+G(4)+S(+M)+L+J(+R) |
| `pack_params(**kwargs)` | `-> Tensor` | flat vector (ordering frozen for parity) |
| `unpack_params(theta, n_regions)` | `-> dict[str,Tensor]` | named CMC param tensors |
| `build_prior_cov(n_regions, prior_variance, a_mask)` | `-> Tensor (n_params,)` | diagonal prior variances from `spm_cmc_priors.m` (§2.8 STACK); zero entries for absent connections so SVD reduction drops them |
| `build_precision(observed)` | `-> (list[Tensor], int)` | `([identity(ny)], 1)` v1; AR(1) `spm_Q(1/2,Ns)` later (STACK §5.1) |
| `predict(theta, observed, n_regions, **context)` | `-> Tensor` | flat vector matching `observed.reshape(-1)` |
| `build_result(theta_final, a_mask, n_regions, **context)` | `-> dict` | `{"theta_post":…, "predicted_output":…}` |

`context` is a free-form dict threaded through every call (`variational_laplace.py:1031`
sets `context["a_mask"]`; spectral adds `context["freqs"]`). **ERP uses `context` to carry
peristimulus time grid, condition design `X`, lead-field gain `G_ecd`, `dt`, and the
spatial-model flag (`"LFP"`/`"ECD"`).** No protocol change required.

Public entry: `run_variational_laplace_generic(forward_model, observed, a_mask, …,
context=…)` (`variational_laplace.py:1269`). The ERP forward plugs in here verbatim — this
is the D1 (VL posterior over CMC) reuse path with zero engine edits.

### 1.2 The `-exp(...)` convention — DO NOT reuse `parameterize_A` for CMC

`forward_models/neural_state.py:24 parameterize_A` negates the diagonal:
`a_ii = -exp(A_free_ii)/2` (fMRI self-inhibition stability). **CMC must NOT use this**
(STACK §2.8): CMC extrinsic connections are strictly positive `+exp(P.A{i})·E(i)` with
directional signs applied *structurally* in the equations of motion (forward `+`, backward
via `−A{3}`,`−A{4}`). A dedicated **`parameterize_cmc`** family lives in the new CMC module
and is the CMC analog of `parameterize_A`/`parameterize_B`. Intrinsic gains use the
permutation remap `j=[7 2 3 4 1 5 6 8 9 10]` with `G(:,j(i)) *= exp(P.G(:,i))` (only 4
free). This is the single most error-prone transform; it is isolated in one function and
parity-tested in Phase 33.

### 1.3 Amortized path seam

- `guides/parameter_packing.py` holds packer classes (`TaskDCMPacker`, `SpectralDCMPacker`,
  `LatentCircuitDCMPacker`) with identical `pack`/`unpack`/`fit_standardization`/
  `standardize`/`unstandardize` surface and a **log-space contract** for positive params.
  ERP adds **`ERPDCMPacker`** following the same surface; positive CMC params (the `exp`-
  transformed G/T/C/S scalings are already in *log* space as free params `P.*`, so they
  pack as-is — only any raw positive observation-noise scalar needs `log()`).
- `models/amortized_wrappers.py:51 _sample_latent_and_unpack` samples a single `_latent`
  site and unpacks via the packer; the wrapper then runs the *same* forward and conditions
  on `obs=`. ERP adds **`amortized_erp_dcm_model`** + an `_run_erp_forward_model` helper
  mirroring `_run_task_forward_model`. This is the D2 reuse path.

### 1.4 SPM `.mat` bridge seam

Parity fixtures follow the **Phase-32 cross-validation pattern** (the most recent parity
work): Python generates inputs/params → exports a `.mat` (`validation/export_to_mat.py`) →
a MATLAB script (`validation/matlab_scripts/run_spm_*.m`) runs the SPM reference on M3
(licensed MATLAB R2022a + Carrick spm12) → results compared in
`tests/test_*_spm_*_validation.py` and orchestrated by
`cluster/scripts/spm_cross_validation.py`. ERP adds parallel files (`export_erp_dcm`,
`run_spm_erp_dcm.m`, `test_spm_erp_dcm_validation.py`). The STACK "ECD gain export"
recommendation rides this exact bridge: precompute `G(:,:,i)` in MATLAB, save to `.mat`,
load into the torch lead field.

---

## 2. New & Modified Components (real `src/` paths)

### 2.1 NEW files (zero backward-compat risk — pure additions)

| File | Purpose | SPM source / REF |
|------|---------|------------------|
| `src/pyro_dcm/forward_models/cmc_neural_mass.py` | CMC state equations `f(x,u,P)`; 8 states/source; sigmoid `S(V)`; `parameterize_cmc` (A{1..4}, C, G perm-`j`, T, S, M transforms) | `spm_fx_cmc.m`; REF-ERP-001/002/003 |
| `src/pyro_dcm/forward_models/cmc_priors.py` | `spm_cmc_priors.m` prior means/variances + transform tables; feeds `build_prior_cov` | `spm_cmc_priors.m`; REF-ERP-003 |
| `src/pyro_dcm/forward_models/erp_input.py` | Gaussian-bump evoked drive `u(t)` (onset `M.ons`, dispersion `P.R`, 32-scale) | `spm_erp_u.m` |
| `src/pyro_dcm/utils/local_linearization.py` | **`spm_int_L` port**: frozen-Jacobian exp-Euler `Q=(matrix_exp(dt·D·J/N)−I)·inv(J)`; `solve` not `inverse`; float64 | `spm_int_L.m`, `spm_gen_erp.m` |
| `src/pyro_dcm/forward_models/erp_coupled_system.py` | Network assembly: extrinsic A fwd/bwd/lateral routing + per-condition B modulation (`spm_gen_Q`) over N sources; produces network RHS for the integrator | `spm_fx_cmc.m:68-82`, `spm_gen_Q.m` |
| `src/pyro_dcm/forward_models/erp_leadfield.py` | Lead field: `L_spatial` (LFP `diag(P.L)` / ECD `G·P.L`), `kron(P.J, L)` state expansion → scalp projection `y=(x−x0)@L.T` | `spm_lx_erp.m`, `spm_erp_L.m`, `spm_L_priors.m` |
| `src/pyro_dcm/simulators/erp_simulator.py` | `simulate_erp_dcm(...)`: params → per-condition scalp ERP + difference wave; mirrors `spectral_simulator.simulate_spectral_dcm` return-dict idiom | composes the above |
| `src/pyro_dcm/models/erp_dcm_model.py` | Pyro generative model: sample A/B/C/G/T/S/R/L/J from log-normal priors, deterministic transforms, forward, Gaussian likelihood on scalp residual; mirrors `spectral_dcm_model.py` | `spm_cmc_priors.m`, `spm_dcm_erp.m` |
| `scripts/demo_mmn_precision_sweep.py` | Headline demo: 5-source MMN network, sweep sp self-inhibition gain → attenuated MMN; `gain→|MMN|` transfer curve (D3) | FEATURES §5 |

> **Integrator placement decision:** the `spm_int_L` port goes in **`utils/`** (next to
> `ode_integrator.py`) because it is a generic numerical scheme taking `(f, x0, dt, N, D)`,
> not CMC-specific physics. This keeps `forward_models/` for equations and leaves the
> torchdiffeq wrapper bit-exact. (STACK offered `forward_models/erp_integrator.py` as an
> alternative; either is fine — recommend `utils/local_linearization.py`.)

### 2.2 MODIFIED files (ADDITIVE ONLY — new symbols appended, existing bodies untouched)

| File | Addition | Backward-compat guarantee |
|------|----------|---------------------------|
| `src/pyro_dcm/inference/forward_models.py` | **`class ERPDCMForward`** (new protocol impl) | Existing `SpectralDCMForward`/`TaskDCMForward`/`LatentCircuitForward` bodies unchanged; new class appended (same pattern as the v0.6.0 `LatentCircuitForward` addition) |
| `src/pyro_dcm/guides/parameter_packing.py` | **`class ERPDCMPacker`** | Existing packers untouched |
| `src/pyro_dcm/models/amortized_wrappers.py` | `amortized_erp_dcm_model` + `_run_erp_forward_model` | Existing wrappers/helpers untouched |
| `src/pyro_dcm/validation/export_to_mat.py` | `export_erp_dcm(...)` | Existing exporters untouched |
| `*/__init__.py` (forward_models, models, simulators, guides) | new exports | re-export only |
| `.planning/REFERENCES.md` | REF-ERP-001…006 entries | append only (keys after Zotero collation per CLAUDE.md) |

**Hard rule honored:** `ode_integrator.py`, `coupled_system.py`, `neural_state.py`,
`spectral_*`, `rdcm_*`, `task_dcm_model.py`, `spectral_dcm_model.py`,
`latent_circuit_dcm_model.py`, and the VL engine core are **never edited**. The CMC path is
reached only through *new* symbols. This is the bit-exact backward-compat guarantee the
project requires; it is verifiable by `git diff` showing only insertions in the four
modified `.py` files.

### 2.3 NEW validation fixtures / scripts

| File | Purpose |
|------|---------|
| `validation/matlab_scripts/run_spm_erp_dcm.m` | Run `spm_gen_erp`/`spm_lx_erp` reference on M3 from exported `.mat` |
| `tests/test_spm_erp_dcm_validation.py` | Parity assertions (single-source, multi-source, scalp) |
| `cluster/scripts/erp_cross_validation.py` | M3 entrypoint, mirrors `spm_cross_validation.py` record-don't-crash idiom |
| `tests/test_cmc_forward.py`, `tests/test_local_linearization.py`, `tests/test_erp_leadfield.py`, `tests/test_erp_dcm_recovery.py` | Standalone module tests (CLAUDE.md "test before integrate") |

---

## 3. Data Flow with Tensor Shapes

Extends the repo shape-convention table. New symbols: **N** = sources, **P=4** populations,
**S=8** states/source, **ns** = peristimulus time samples, **Nc** = scalp channels/sensors,
**Cnd** = conditions (standard, deviant), **n_inp** = stimulus inputs.

```
                                    erp_input.py            local_linearization.py
 params (per source/condition)  ──►  u(t) Gaussian bump  ──►  exp-Euler integrate  ──►  states
        │                              (ns, n_inp)              (frozen J at x0=0)        trajectory
        ▼                                                                                  │
 erp_coupled_system.py: assemble network RHS f(x,u,P)                                       ▼
   A{1..4}:(N,N) ea · B_cnd:(N,N) · C:(N,) · G_free:(N,4) · T_free:(N,4) · S:(N,) [· M:(N,)]
        │                                                                          erp_leadfield.py
        ▼                                                                                  ▼
   x_cmc per source (N, 8)  ── spm_vec column-major ──►  x_flat (8N,)            L_full = kron(J,L)
   states_traj per condition: (ns, N, 8)  ≡  (ns, 8N)                            J:(8,) · L_spatial
                                                                                  LFP diag (N,N) → L_full (N, 8N)
                                                                                  ECD  G·P.L (Nc,N) → L_full (Nc, 8N)
        ▼                                                                                  ▼
   scalp ERP per condition:   y_c = (states_c − x0) @ L_full.T        ──►        y_c (ns, Nc)
        ▼
   stacked observed/predicted:  (Cnd, ns, Nc)  ──► predict() returns .reshape(-1)
        ▼
   difference wave (MMN):  y_deviant − y_standard                     ──►        mmn (ns, Nc)
        ▼
   transfer curve (D3): sweep G(:,7) sp self-inhibition gain → |mmn| peak       (n_gain,)
```

**Boundary shape table**

| Stage | Tensor | Shape | dtype | Notes |
|-------|--------|-------|-------|-------|
| CMC state (per source) | `x_cmc` | `(N, 8)` | f64 | cols = ss/sp/ii/dp ×(V,I) |
| Flattened state | `x_flat` | `(8N,)` | f64 | **column-major** (`spm_vec`) — parity-critical |
| Evoked input | `u_erp` | `(ns, n_inp)` | f64 | `spm_erp_u`, 32-scaled |
| Frozen Jacobian | `J` | `(8N, 8N)` | f64 | `jacrev(f)` at x0, `−I·exp(−16)` reg |
| Update operator | `Q` | `(8N, 8N)` | f64 | `(matrix_exp(dt·D·J/N)−I)·inv(J)` via `solve` |
| State trajectory (per cond) | `states_c` | `(ns, 8N)` | f64 | one per condition |
| Spatial lead field | `L_spatial` | `(Nc, N)` ECD / `(N, N)` LFP | f64 | from `spm_erp_L` / exported gain |
| State→dipole weights | `J_contrib` (`P.J`) | `(8,)` | f64 | CMC default contributes state 3 (sp V) |
| Full lead field | `L_full` | `(Nc, 8N)` | f64 | `kron(J_contrib, L_spatial)` |
| Scalp ERP (per cond) | `y_c` | `(ns, Nc)` | f64 | `(states_c − x0) @ L_full.T` |
| Observed / predicted | `y` | `(Cnd, ns, Nc)` | f64 | `predict` flattens to `(Cnd·ns·Nc,)` |
| MMN difference wave | `mmn` | `(ns, Nc)` | f64 | deviant − standard |

**VL adapter specifics (`ERPDCMForward`):** `residual_is_complex=False`; `observed` =
`(Cnd, ns, Nc)`; `build_precision` returns `([identity(Cnd·ns·Nc)], 1)` (AR(1) deferred);
`predict` returns `y.reshape(-1)`. `a_mask` carries the extrinsic connection graph
(`(N,N)`); per-connection-type masks (fwd/bwd/lateral) and the condition design `X` ride in
`context`.

---

## 4. Build Order (Phases 33→36) — additive at every step

Ordered by the FEATURES dependency map (T1→T2→…→T12) and the rule that each phase ends on a
green SPM-parity gate before the next begins. Every phase is additive-only; backward compat
is re-verified by `git diff` (insertions only) + the full existing test suite staying green.

### Phase 33 — CMC core dynamics + exp-Euler integrator + single-source parity (T1, T2)
**The foundation; STACK flags this as the highest-risk implementer phase.**
- NEW: `forward_models/cmc_neural_mass.py` (`parameterize_cmc`, `f(x,u,P)`, sigmoid),
  `forward_models/cmc_priors.py`, `forward_models/erp_input.py`,
  `utils/local_linearization.py`.
- TEST: `tests/test_cmc_forward.py`, `tests/test_local_linearization.py`;
  `validation/matlab_scripts/run_spm_erp_dcm.m` (single-source), `export_erp_dcm` (additive
  to `export_to_mat.py`), `tests/test_spm_erp_dcm_validation.py` (single-source).
- GATE: single-source `spm_fx_cmc`+`spm_int_L` parity on frozen fixture (M3 SPM).
- **Additive:** all-new modules; touches only `export_to_mat.py` (append).
- **Research flag:** `matrix_exp`↔`spm_expm` tolerance must be *measured* (STACK §5.4);
  column-major `spm_vec` flatten ordering (STACK §2.1).

### Phase 34 — Network: extrinsic A + condition B + evoked integration + multi-source parity (T3–T7)
- NEW: `forward_models/erp_coupled_system.py` (fwd/bwd/lateral routing + per-condition B),
  `simulators/erp_simulator.py`.
- TEST: multi-source `tests/test_spm_erp_dcm_validation.py` extension vs `spm_gen_erp`.
- GATE: multi-source evoked parity for reference A/B/C on the 5-source MMN graph.
- **Additive:** all-new modules.
- **Research flag (STACK §5.2 / FEATURES anti-feature):** delay operator `spm_dcm_delay`
  deferred (D=1 first); add only if delay-free parity insufficient.

### Phase 35 — Lead field + scalp + MMN + Pyro model + VL adapter (T8, T9, T10, D1)
- NEW: `forward_models/erp_leadfield.py`, `models/erp_dcm_model.py`.
- MODIFY (additive): `inference/forward_models.py` (+`ERPDCMForward`).
- TEST: `tests/test_erp_leadfield.py`, `tests/test_erp_dcm_recovery.py`; scalp/difference-
  wave parity fixture. ECD gain `G(:,:,i)` precomputed in MATLAB, exported via `.mat`
  bridge, loaded into `erp_leadfield.py` (no MNE/FieldTrip in Python).
- GATE: scalp difference-wave parity vs `spm_lx_erp`; VL recovers planted CMC params via
  `run_variational_laplace_generic(ERPDCMForward(), …)`.
- **Additive:** one new class in `forward_models.py`; existing classes untouched.

### Phase 36 — Precision sweep demo + transfer curve + adapter API + amortized reuse (T11, T12, D2, D3, D4)
- NEW: `scripts/demo_mmn_precision_sweep.py`; adapter API mapping
  `(sp self-inhibition, A1/rIFG gain, B fwd/bwd)` → CMC params (D4, consumer-facing).
- MODIFY (additive): `guides/parameter_packing.py` (+`ERPDCMPacker`),
  `models/amortized_wrappers.py` (+`amortized_erp_dcm_model`).
- GATE: monotone `gain→|MMN|` attenuation curve (the actinf_physics hand-off artifact);
  amortized guide trains on `erp_simulator` draws.
- **Additive:** new packer + new wrapper; existing packers/wrappers untouched.
- **Defer if time-boxed:** D5 BMR over MMN modulation hypotheses reuses
  `model_selection/bmr.py` with no forward-stack change.

---

## 5. Anti-Patterns to Avoid (architecture-specific)

| Anti-pattern | Why bad | Instead |
|--------------|---------|---------|
| Editing `ode_integrator.py` to add exp-Euler | risks the fMRI/spectral/latent bit-exact paths; couples a parity-critical scheme to the torchdiffeq wrapper | new `utils/local_linearization.py` |
| Reusing `parameterize_A` for CMC A{1..4} | `-exp/2` diagonal negation is fMRI-only; CMC uses `+exp` with structural signs | new `parameterize_cmc` |
| Adding ERP params to an existing packer | `n_features` arithmetic differs; would break existing standardization stats | new `ERPDCMPacker` |
| Row-major flatten of `(N,8)` state | SPM `spm_vec` is column-major (state-blocked); breaks Jacobian/`kron` parity | flatten Fortran-order / operate per-column |
| `torch.inverse` for `(E−I)·inv(J)` | violates CLAUDE.md numerical rule; less stable | `torch.linalg.solve(J.T,(E−I).T).T` |
| Calling `integrate_ode(method="rk4")` for ERP | cannot reproduce `spm_int_L`; fails parity | `local_linearization` exp-Euler |
| Building a real head model in Python | pulls in FieldTrip/MNE; out of scope | LFP diag first; ECD gain exported from MATLAB via `.mat` bridge |

---

## 6. Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| `ForwardModel` protocol + VL dispatch reuse | HIGH | read `forward_models.py:30-117` + `_run_vl_generic`; `LatentCircuitForward` is a working precedent for a 3rd impl |
| `-exp` convention divergence (need `parameterize_cmc`) | HIGH | `neural_state.py:24` read; STACK §2.8 cross-confirms from SPM source |
| Integrator must be new (not `ode_integrator` edit) | HIGH | `ode_integrator.py` is a torchdiffeq wrapper; STACK §3 structural argument |
| Packer/wrapper additive seam | HIGH | three existing packers + wrapper idiom read directly |
| SPM `.mat` bridge extensibility | HIGH | Phase-32 harness (`spm_cross_validation.py`, `export_to_mat.py`, matlab_scripts) read |
| Exact tensor shapes at every boundary | MEDIUM-HIGH | derived from SPM source (STACK) + repo conventions; `Nc`/`Cnd` stacking layout should be locked by the Phase-33/35 parity fixtures |
| Phase→file mapping | HIGH | follows FEATURES dependency DAG + additive-only rule |

## Open Questions for Roadmap

1. **Observation stacking layout** `(Cnd, ns, Nc)` vs `(Cnd·ns, Nc)` — lock in Phase 35 so
   `predict`/`build_precision` agree with the parity fixture's `.mat` ordering.
2. **AR(1) precision** (`spm_Q(1/2,Ns)`) — identity v1; promote to a precision basis if
   F/evidence parity (not just forward parity) is required (STACK §5.1).
3. **Delay operator** — D=1 first; gate the `spm_dcm_delay` port on delay-free parity
   (STACK §5.2). Flag Phase 34 for possible deeper research here.
4. **Integrator home** — recommend `utils/local_linearization.py`; confirm with roadmapper
   vs STACK's `forward_models/erp_integrator.py` alternative (cosmetic).
</content>
</invoke>
