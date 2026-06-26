# Phase 35: Single-Dipole Lead-Field, Scalp Projection & ERPDCMForward — Research

**Researched:** 2026-06-26
**Domain:** Single-dipole lead-field projection (`spm_lx_erp.m` / `spm_erp_L.m`,
LFP-first) of the parity-verified per-source CMC trajectory into scalp ERP, the
deviant−standard difference wave, and an additive `ERPDCMForward` implementing the
existing `ForwardModel` protocol for free VL reuse. Parity vs frozen `spm_lx_erp` LFP
fixtures on M3.
**Confidence:** HIGH (every equation below transcribed from SPM12 source read
line-by-line this session at `C:/Users/aman0087/Documents/Github/spm12/toolbox/dcm_meeg/`
and the shipped Phase-33/34 torch + bridge code).

> This file is the **Phase-35 implementation concretes**. It does NOT re-survey the
> milestone — read `.planning/research/v0.8.0/{SUMMARY,ARCHITECTURE,PITFALLS}.md` for the
> why and `.planning/phases/34-.../34-RESEARCH.md` + `34-03-SUMMARY.md` for the locked
> conventions (8N column-major flatten, `(Cnd,ns,n,8)` state layout, the jacrev-vs-FD gate
> split, D=1 wrapper) this phase composes. It gives the *exact* lead-field algebra (with
> `spm_lx_erp.m`/`spm_erp_L.m`/`spm_L_priors.m` line refs), the *exact* protocol contract
> for `ERPDCMForward`, the *exact* repo seams to extend additively, the locked observation
> stacking layout, and the scalp parity-ladder spec — so the planner can cut 3 executable
> plans with no further source reading.

---

## Summary

Phase 35 turns the Phase-34 parity-verified per-source CMC trajectory (`(Cnd, ns, n, 8)`
source states, bit-close to `spm_gen_erp`) into the observed scalp ERP via a single
linear map and exposes the whole forward to Variational Laplace as a fourth `ForwardModel`
implementor. **Only three things are genuinely new**, and all are additive: (1) a pure-torch
lead-field builder `erp_leadfield.py` that constructs `L_full = kron(P.J, L_spatial)` and
projects `y = (x − x0) @ L_full.T`; (2) `class ERPDCMForward` appended to
`inference/forward_models.py` implementing the existing 8-member protocol (no protocol or
VL-engine edits — `LatentCircuitForward` is the precedent); and (3) a one-call extension of
`erp_simulator.py` + the `validation/` `.mat` bridge to project source states to scalp and
freeze the `spm_lx_erp` LFP lead field. Everything upstream — the network forward
(`cmc_network_f`), the condition-B mechanism (`apply_condition_modulation`), the exp-Euler
integrator (`integrate_local_linearization`), the column-major `cmc_flatten`, the D=1
`spm_fx_cmc_nodelay` wrapper, the 5-source MMN reference net (`_MS_*` constants in
`export_to_mat.py`) — is reused verbatim and is already proven bit-close to SPM.

The single most important source fact: for CMC, `spm_L_priors.m:108` sets
`pE.J = sparse(1,3,1,1,8)` → the **default observed state is column 3 (MATLAB, 1-indexed) =
index 2 (0-indexed) = superficial-pyramidal VOLTAGE**, *not* index 6 (deep-pyramidal
voltage). `spm_lx_erp.m:33` then forms `L = kron(P.J, L_spatial)` with `P.J` as the FIRST
kron argument, whose `(Nc, 8n)` block ordering (`P.J[s] * L_spatial` in column block `s`)
**exactly matches the column-major state-blocked flatten** (`cmc_flatten = x.T.reshape(-1)`,
flat index `= state*n + source`). In LFP mode (`spm_erp_L.m:105-118`) the spatial model is a
trivial diagonal `L_spatial = diag(P.L)`, default `P.L = ones` → identity, so the LFP scalp
ERP is literally each source's sp-voltage trace — the cleanest possible parity target, with
no head model, no `spm_cond_units`, no MNI coords. This is the Phase-35 gate (LEAD-05); the
ECD path (`spm_erp_L.m:43-77`, gain post `spm_cond_units`) needs a sensor montage + MNI coords
and is correctly deferred — `erp_leadfield.py` is built to *consume* an exported ECD gain from
day one so no rework is needed when Phase 36 supplies coords.

**Primary recommendation:** Build in strict wave order — Wave 1 (laptop, pure-torch):
`erp_leadfield.py` + `ERPDCMForward` + the structural guards (P.J=index-2, kron column-major,
LFP identity, difference-wave non-zero) + the VL round-trip; Wave 2 (M3, MATLAB): the
`spm_lx_erp` LFP-`L` + scalp-ERP + difference-wave fixtures; Wave 3 (laptop, against the
committed `.mat`): the scalp parity ladder (`L_full` exact → scalp ERP ≤1e-7 with the
inherited 3-way Jacobian split → difference-wave parity). Lock the observation stacking layout
as internal `(Cnd, ns, Nc)`, flat boundary C-order `reshape(-1)`, identity precision over
`Cnd*ns*Nc` BEFORE writing any scalp assertion.

---

## Source-resolved facts (the things Phase 35 must get exactly right)

### Fact 1 — `P.J` CMC default = state index 2 (sp voltage) [`spm_L_priors.m:106-109`]

```matlab
case{'CMC','TFM'}
    pE.J{end + 1} = sparse(1,3,1,1,8);          % :108  8-state one-hot at col 3
    pC.J{end + 1} = sparse(1,[1 7],1/32,1,8);   % :109  free at cols 1,7 var 1/32
```

`sparse(1,3,1,1,8)` is a `(1,8)` row vector with a single `1` at **MATLAB column 3** →
**0-indexed index 2**. The CMC 8-state column layout (from `cmc_neural_mass.py:20-33`,
proven Phase 33) is `[ss_V=0, ss_I=1, sp_V=2, sp_I=3, ii_V=4, ii_I=5, dp_V=6, dp_I=7]`. So
the default lead field reads **superficial-pyramidal voltage (index 2)** — physiologically
correct (EEG is dominated by L2/3 superficial-pyramidal depolarisation). The free states
(`pC.J` at MATLAB `[1 7]` → 0-indexed `[0, 6]`, var `1/32`) are ss_V and dp_V, but their
prior MEAN contribution is zero — they only matter if `J` is freed.

**GUARD (LEAD-02, hard):** `P_J_default` is the one-hot `e_2` (8-vector, `1.0` at index 2,
else 0). Assert it equals index 2, and assert it does NOT equal index 6 (the dp-voltage
inversion trap, pitfall C5.1). Cite `spm_L_priors.m:108`.

### Fact 2 — `L = kron(P.J, L_spatial)`, P.J FIRST, column-major block order [`spm_lx_erp.m:31-42`]

```matlab
L = spm_erp_L(P,dipfit);          % :31  lead field per SOURCE, (Nc, n)
if isnumeric(P.J)
    L = kron(P.J,L);              % :33  lead-field per STATE, (Nc, 8n)
```

`P.J` is `(1,8)`, `L_spatial` is `(Nc, n)`. `kron((1,8),(Nc,n))` is `(Nc, 8n)`. The MATLAB
kron block at state `s` is `P.J[s] * L_spatial`, occupying columns `[s*n : (s+1)*n]`. So
the full-lead-field column index is `state*n + source`. **This is exactly the
column-major state-blocked flatten** `cmc_flatten = x.T.reshape(-1)` (flat index
`state*n + source`, proven at N=5 in `34-03-SUMMARY.md` rung 2). The two orderings are
identical by construction — no transpose, no permutation.

**torch port:** `L_full = torch.kron(P_J.reshape(1, 8), L_spatial)` → `(Nc, 8n)`. (Both
operands must be 2-D; `torch.kron` exists in torch ≥2.0 — zero new deps.)

**GUARD (LEAD-02, hard):** build a *distinct-valued* `L_spatial` (e.g. `arange(Nc*n)+1`
reshaped) and a non-trivial `P.J`, compute `torch.kron(P_J.reshape(1,8), L_spatial)`, and
assert element-wise equality to the MATLAB-exported `kron(P.J,L)` fixture. A C-order
(`reshape`) flatten would put the block at `source*8 + state` and fail this — that is the
trap (pitfall C5.2).

### Fact 3 — LFP spatial model is a trivial diagonal [`spm_erp_L.m:105-118`, `spm_L_priors.m:84`]

```matlab
case{'LFP'}                       % :105
    m = length(P.L);              % :106
    try, n = dipfit.Ns; catch, n = m; end
    L = sparse(1:m,1:m,P.L,m,n);  % :112  (m, n) diagonal, gain P.L on the diagonal
```

`L_spatial = diag(P.L)`, shape `(Nc=m, n)`. Default `P.L = ones(1,m)` (`spm_L_priors.m:84`)
→ `L_spatial = I_n` (one channel per source, `Nc = n`). So in LFP mode:
- `L_full = kron(e_2, I_n)` → `(n, 8n)`; the sp-voltage state block (`s=2`) is `I_n`, all
  other blocks are 0.
- `y_scalp = (states − x0) @ L_full.T = states @ L_full.T` (CMC `x0 = 0`, M1) → `(ns, n)`,
  which is exactly the sp-voltage column of each source. **No head model, no
  `spm_cond_units`, no MNI coords.** This is the LEAD-05 parity target.

torch: `L_spatial = torch.diag(P_L)` where `P_L = ones(n)` default → `L_spatial = I_n`.

### Fact 4 — ECD spatial model (deferred gain export) [`spm_erp_L.m:43-77`]

```matlab
case{'ECD'}
    M = dipfit.datareg.fromMNI;                        % :57
    if dipfit.siunits, M = diag([1e-3 1e-3 1e-3 1])*M; end  % :59-60 MNI->m
    Lf = ft_compute_leadfield(transform_points(M,P.Lpos(:,i)'), dipfit.sens, dipfit.vol);
    G  = spm_cond_units(LastL);                        % :74  unit-conditioning rescale
    L(:,i) = G(:,:,i)*P.L(:,i);                         % :76  (Nc,3)@(3,) = (Nc,)
```

ECD needs `dipfit.datareg.fromMNI`, `dipfit.sens`, `dipfit.vol`, `P.Lpos` (MNI), and the
free 3-vector dipole moment `P.L` (`spm_L_priors.m:67`, `E.L=0, V.L=64`). The physical gain
`G(:,:,i)` is `(Nc,3)` **post `spm_cond_units`** (pitfall C5.3 — the exported gain MUST
include this rescale). `L_spatial = (Nc, n)`. The torch side reproduces ONLY
`kron(P.J, L_spatial)` + projection; `G` is precomputed in MATLAB and exported.

**Phase-35 scope decision (see Open Q1):** ECD requires a sensor montage + MNI coords, which
are flagged for Phase 36 (`ERPDCM-03`, "MNI coords flagged for verification before
hard-coding"). **Recommend Phase 35 ship the LFP-mode gate only** (the defensible,
head-model-free parity); build `erp_leadfield.py` to accept an exported ECD gain
`G (Nc,3,n)` from day one (a `spatial="ECD"` branch consuming `g_ecd` + `P.L`), but produce
the ECD gain fixture in Phase 36 where the MMN net coords + a template montage are locked.
This honours PITFALLS C5.4 ("for LFP-mode parity (recommended first)... start there") with
zero rework.

### Fact 5 — Projection sign & x0 [`spm_lx_erp.m` header, M1]

`spm_lx_erp` returns `G` where `y = G*x` (observer matrix, header `:2`). For a state row
trajectory `states (ns, 8n)`, the scalp ERP is `y = states @ L_full.T` → `(ns, Nc)`. CMC
`x0 = zeros` (M1, `cmc_steady_state`), so the `(x − x0)` baseline subtraction is a no-op but
should be written explicitly (`y = (states − x0) @ L_full.T`) for ECD generality and to
match the `ARCHITECTURE.md` boundary table. The dipole-moment sign (ECD) is free (`P.L` can
be ±) — irrelevant in LFP mode (default `P.L = +ones`), but the difference-wave sign
convention (deviant−standard) must be pinned end-to-end (see Fact 6 / Open Q2).

### Fact 6 — Difference wave (LEAD-03), and what Phase 35 can/cannot assert

The MMN difference wave is `y_scalp[deviant] − y_scalp[standard]` (subtraction order fixed
deviant−standard). In LFP identity mode with `P.J = sp voltage`, this is the deviant−standard
**sp-voltage difference per source**, non-zero iff `B` is wired (it is, in the `_MS_` ref net:
`_MS_B_EDGE=0.3` on edges + `_MS_B_DIAG=0.5` at precision nodes). **Phase 35 asserts:
(a) difference wave is non-zero (hard gate); (b) it matches the MATLAB-exported scalp
difference-wave fixture element-wise (≤1e-7).** The **negative-going / frontal-dominance**
sign assertion (pitfall S2) depends on the dipole ORIENTATION (ECD `P.L` sign) and MNI
coords, which do not exist until Phase 36 — so it is **deferred to Phase 36**. Phase 35 may
additionally record the *source-level* sign direction (deviant sp-voltage attenuated vs
standard at the precision nodes) as a diagnostic, but must NOT gate on a scalp polarity that
the LFP identity convention does not physically pin. State this in the `simulate_erp_dcm`
scalp-extension docstring so the planner does not over-scope (mirrors the Phase-34 boundary
note).

---

## The `ForwardModel` protocol contract for `ERPDCMForward` (LEAD-04)

The VL engine (`variational_laplace.py:970 _run_vl_generic` →
`run_variational_laplace_generic`) dispatches ONLY on the 8 protocol members
(`forward_models.py:30-117`); it sets `context["a_mask"]` (`:1031`) and calls
`param_count` → `build_prior_cov` → `_spm_svd` → `build_precision` → loop(`predict`) →
`build_result`. `LatentCircuitForward` (`forward_models.py:451-683`) is the exact precedent
for a third/fourth implementor with model-specific state held as constructor args. The ERP
class is **purely additive** — appended after `LatentCircuitForward`, zero edits to the
protocol or the engine.

| Member | ERP implementation | Source / precedent |
|--------|--------------------|--------------------|
| `residual_is_complex` (prop) | `False` (time-domain real) | like `Task`/`LatentCircuit` |
| `param_count(n)` | `4*n*n + n*n_inp + 4*n + 4*n + n + 2*n_inp` (A{1..4}+C+T+G+S+R); L/J held fixed in v1 (Open Q3) | `cmc_priors` schema |
| `pack_params(**kw)` | `cat([A_free(4nn), C_free(n·n_inp), T(4n), G(4n), S(n), R(2·n_inp)])` — **FROZEN ordering** | mirrors `LatentCircuit.pack` |
| `unpack_params(theta,n)` | reverse the frozen slice ordering → dict `{A_free(4,n,n), C_free, T, G, S, R}` | mirrors `:558-572` |
| `build_prior_cov(n,pv,a_mask)` | flatten cmc-prior variances IN PACK ORDER (A `mask/16`, C `mask/32`, T/G `1/32`, S `1/64`, R `1/16`); zero for absent connections so SVD drops them | `cmc_prior_moments` + `:574-598` |
| `build_precision(observed)` | `([eye(Cnd·ns·Nc, f64)], 1)` (identity v1; AR(1) `spm_Q` deferred, M3) | `LatentCircuit:600-605` |
| `predict(theta,observed,n,**ctx)` | per-condition integrate → project → stack `(Cnd,ns,Nc)` → `.reshape(-1)` (with `observed.ndim` guard) | `LatentCircuit:643-663` |
| `build_result(theta,a_mask,n,**ctx)` | `{"theta_post": {A,C,G,T,S,R,...,"B":...}, "predicted_output": (Cnd,ns,Nc)}` | `:665-682` |

**ERP-specific needs ride as CONSTRUCTOR ARGS (not protocol methods, pitfall B2):**
`l_full (Nc,8n)` precomputed lead field, `x_design (Cnd,n_effects)`, `b_masks (list[(n,n)])`,
`a_masks (tuple of 4 (n,n) — the fwd/fwd/bwd/bwd routing graph)`, `c_mask (n,n_inp)`,
integration grid `dt, ns, ons_ms, dur_ms, sus`. The Gaussian drive is rebuilt inside
`predict` from the unpacked `R` via `erp_gaussian_input` (or held fixed if R is frozen).

**Two contract subtleties (resolve in the plan):**
1. **The 4-block `a_mask`.** The protocol passes a single `(n,n)` `a_mask` via `context`;
   CMC has FOUR distinct extrinsic blocks (fwd sp→ss, fwd sp→dp, bwd dp→sp, bwd dp→ii).
   `ERPDCMForward` should store its OWN `a_masks` (tuple of 4 `(n,n)`) as a constructor arg
   and use them in `build_prior_cov`/`unpack`, IGNORING the engine-supplied scalar `a_mask`
   (or treating it as a compatibility no-op). Mirror how `TaskDCMForward` stores `c_mask`
   internally rather than reading it from context. **Decision needed (Open Q4).**
2. **FD-Jacobian flat-vector contract (pitfall B3).** The VL FD path
   (`variational_laplace.py:959-966`) calls `predict` with a FLAT `observed`; the main loop
   passes the `(Cnd,ns,Nc)` tensor. `predict` must mirror the `LatentCircuitForward.predict`
   `observed.ndim >= 2` guard (here `>= 3` → truncate to `(Cnd,ns)`; flat FD call → no
   truncation, the trajectory length already matches `ns`). Test `predict` under BOTH shapes.

---

## Locked observation stacking layout (research gap 4 — DECIDE BEFORE writing assertions)

**Decision (LOCK):** internal canonical tensor is **`(Cnd, ns, Nc)`**; the flat boundary is
**C-order `reshape(-1)`** → `(Cnd·ns·Nc,)`, condition-blocked (all of condition 0's `(ns,Nc)`
first, then condition 1's). `build_precision` returns `([identity(Cnd·ns·Nc)], 1)`. The
MATLAB `.mat` fixture stores the scalp ERP as a cell `{Cnd}` of `(ns, Nc)`; the torch test
`torch.stack`s to `(Cnd, ns, Nc)` and `reshape(-1)`s with the identical C-order.

**Justification:**
- The 3-D form keeps the condition axis EXPLICIT, which the difference wave
  (`states[1] − states[0]`) and the per-condition `apply_condition_modulation` loop both
  require — `(Cnd·ns, Nc)` pre-flattens that axis away and forces an error-prone reshape to
  recover it for differencing.
- `reshape(-1)` on `(Cnd, ns, Nc)` is unambiguous and matches the existing identity-precision
  idiom (`TaskDCMForward`/`LatentCircuitForward` both `observed.numel()` an identity); the
  VL engine's `observed.reshape(-1)` (`:1068`) and `predict.reshape(-1)` then agree
  element-for-element with no transpose.
- `(Cnd·ns, Nc)` and `(Cnd, ns, Nc)` are *byte-identical* under `reshape(-1)` (C-order), so
  the choice is purely about the internal axis the difference wave reads — keep it 3-D
  internally, flatten only at the `predict`/`build_precision` boundary.

Misalignment here (e.g. stacking `(ns, Cnd, Nc)` or Fortran-order flatten) produces a
clean-looking but wrong residual structure that passes shape tests and fails parity — this is
exactly research gap 4. **Write this layout into the `predict` docstring and the fixture
loader before any scalp assertion.**

---

## Plan decomposition recommendation

Three plans, three waves. Strict ordering: the pure-torch lead field + `ERPDCMForward` +
structural guards must land green on the laptop BEFORE the M3 fixture round-trip, so the
parity test has something to assert against and the kron/P.J logic is caught in isolation
before it compounds through the (already-verified) trajectory (V5 ladder discipline).

### Wave 1 — Pure-torch lead field + forward-model adapter (LAPTOP, sub-second tests) — Plan 35-01
Internally parallelizable; composes the frozen Phase-33/34 modules without editing them.
- **`forward_models/erp_leadfield.py`** (LEAD-01/02) — `cmc_default_pj()` (the `e_2` one-hot
  + the free-index `[0,6]` table), `build_lead_field(p_j, l_spatial)` (`torch.kron`),
  `lfp_spatial(p_l, n)` (`diag(P.L)`), `project_to_scalp(states, l_full, x0)`
  (`(states−x0) @ l_full.T`), and an `ecd_spatial(g_ecd, p_l)` branch (consumes an exported
  `G (Nc,3,n)` — built now, exercised Phase 36).
- **`inference/forward_models.py`** (LEAD-04, additive append) — `class ERPDCMForward`
  implementing the 8-member contract above; lazy-import the CMC network functions inside
  `predict`/`build_result` (mirror `TaskDCMForward`'s lazy `bold_signal`/`CoupledDCMSystem`
  import at `:398-400`) so import-time coupling stays zero.
- **`simulators/erp_simulator.py`** (extend, LEAD-03) — add a scalp path: project the
  existing `(Cnd, ns, n, 8)` source states through `l_full` to `(Cnd, ns, Nc)` and return
  `scalp` + the true `difference_wave_scalp = scalp[1] − scalp[0]`. Keep the existing
  source-state return keys (backward-compatible; new keys only).
- **`tests/test_erp_leadfield.py`** (laptop, no MATLAB) — the structural guards (below) +
  the **VL round-trip (LEAD-06)**: plant CMC params, `simulate_erp_dcm` → scalp obs,
  `run_variational_laplace_generic(ERPDCMForward(...), obs, ...)`, assert planted A/C/G/T
  recovered within tolerance (protocol confirmation, NOT a parity gate).

Gate: all laptop unit tests green; `git diff` shows only the new file + the appended class +
the simulator/`__init__` exports (no edits to existing class bodies). The Phase-33/34 suites
stay green (regression: `cmc_network_f`, the integrator, the bridge are untouched).

### Wave 2 — `spm_lx_erp` LFP-`L` + scalp-ERP MATLAB fixtures (M3, MATLAB) — Plan 35-02
Depends on Wave 1 (needs the reference net + P.J/P.L locked). Reuses the Phase-34
`erp_multisource_input.mat` / `_MS_` 5-source reference (N=5, Cnd=2).
- **`validation/export_to_mat.py`** (extend) — `export_erp_dcm_leadfield(...)` (or extend
  `export_erp_dcm_multisource`) to add the spatial spec: `dipfit.type='LFP'`,
  `dipfit.Ns=5`, `dipfit.Nc=5`, `P.L = ones(1,5)` (`(1,m)` double), `P.J = sparse(1,3,1,1,8)`
  encoded as a `(1,8)` one-hot. Cast all dims to double (the int64→`spm_Ce` footgun, commit
  a27828b). Freeze `dipfit`/`P.J`/`P.L` in `DCM.meta`.
- **`validation/matlab_scripts/run_spm_erp_dcm_leadfield.m`** (NEW) — mirror
  `run_spm_erp_dcm_multisource.m` scaffolding (env paths, loud try/catch, `getenv` IO,
  `M.f='spm_fx_cmc_nodelay'`, assert `nargout==2` and `x0==zeros(N,8)`). Compute the fixture
  arrays (below) via `spm_lx_erp` + reuse of the source trajectory.
- **`cluster/scripts/erp_cross_validation.py`** (extend) + **`.sbatch`** (REUSE) — add a
  lead-field mode mirroring the multi-source `main()` (export → `matlab -batch
  run_spm_erp_dcm_leadfield` → round-trip shape/meta check → JSON record, record-don't-crash,
  exit 0 on soft miss).

Gate: M3 job produces `validation/data/erp_leadfield_fixtures.mat` with `L_full (Nc,8N)`,
per-condition `y_scalp {Cnd}×(ns,Nc)`, `diff_wave (ns,Nc)`, and a provenance `meta`
(SPM `$Id`, `dt`, `ns`, `D=1`, `nargout==2`, `P.J`, `P.L`, `dipfit.type='LFP'`, `Nc`, the
`_MS_` edge list). Commit the `.mat` (V4 — reviewed change).

### Wave 3 — Scalp parity ladder (LAPTOP, against committed `.mat`) — Plan 35-03
Depends on Wave 1 + Wave 2. Assertions are torch-vs-frozen-arrays (deterministic f64), so the
suite RUNS AND PASSES on the laptop; `@pytest.mark.spm`/`slow` retained for the optional M3
re-run (inherited 33-03-D1/34-03-D2).
- **`tests/test_spm_erp_leadfield_validation.py`** (NEW) — the staged scalp ladder (LEAD-05):
  `L_full` element-wise → scalp ERP per condition (3-way Jacobian split) → difference-wave
  parity + non-zero. Reuse the multi-source loader pattern + `_reference_p()` from
  `tests/test_spm_erp_multisource_validation.py` (import the `_MS_` constants).

Gate (LEAD-05): `L_full` (torch `kron` vs exported) ≤1e-12 element-wise (it is exact kron of
the exported `L_spatial`); scalp ERP ≤1e-7 (carry the 3-way split, below); difference wave
≤1e-7 AND non-zero.

**Why this split:** Wave 1 is laptop pure-torch (the lead field is a pure linear map; the
forward adapter wraps the already-verified trajectory); Wave 2 is the only M3/MATLAB piece
(license-gated, record-don't-crash); Wave 3 is the gate. Splitting lets the P.J/kron guards
and the VL round-trip be written and merged before any MATLAB dependency exists. Waves are
sequential (each is a tier of the V5 ladder); Wave 1's modules can be built concurrently.

---

## Per-file implementation spec

### `src/pyro_dcm/forward_models/erp_leadfield.py` (NEW, LEAD-01/02) — laptop

```python
def cmc_default_pj() -> Tensor:
    """CMC default contributing-state vector P.J (spm_L_priors.m:108).

    Returns the (8,) one-hot e_2 (1.0 at index 2 = sp voltage, else 0). Free
    indices (pC.J) are [0, 6] (ss_V, dp_V) var 1/32 — held FIXED in v1.
    GUARD: index 2, NOT 6 (pitfall C5.1)."""

def lfp_spatial(p_l: Tensor, n: int) -> Tensor:
    """LFP spatial lead field diag(P.L), (Nc=m, n) (spm_erp_L.m:112).
    Default p_l = ones(n) -> identity. Nc == n (one channel per source)."""

def ecd_spatial(g_ecd: Tensor, p_l: Tensor) -> Tensor:
    """ECD spatial lead field L[:,i] = G[:,:,i] @ P.L[:,i] (spm_erp_L.m:76).
    g_ecd: (Nc,3,n) MATLAB-exported gain POST spm_cond_units (pitfall C5.3);
    p_l: (3,n) free dipole moments. Returns (Nc, n). Phase-36 exercised."""

def build_lead_field(p_j: Tensor, l_spatial: Tensor) -> Tensor:
    """Full per-state lead field L_full = kron(P.J, L_spatial) (spm_lx_erp.m:33).
    p_j: (8,); l_spatial: (Nc, n). Returns (Nc, 8n). torch.kron(p_j.reshape(1,8),
    l_spatial) — block s = P.J[s]*L_spatial at cols [s*n:(s+1)*n], matching the
    column-major cmc_flatten (Fact 2). GUARD vs exported kron (LEAD-02)."""

def project_to_scalp(states: Tensor, l_full: Tensor, x0: Tensor | None = None) -> Tensor:
    """Scalp ERP y = (states - x0) @ L_full.T (spm_lx_erp.m header).
    states: (ns, 8n) or (Cnd, ns, 8n); l_full: (Nc, 8n); x0: (8n,) default 0
    (CMC M1). Returns (..., Nc). float64-guarded."""
```

Shapes: `l_spatial (Nc,n)`, `l_full (Nc,8n)`, `states (ns,8n)`, `y (ns,Nc)`. Cite
`spm_lx_erp.m:31-42`, `spm_erp_L.m:76,112`, `spm_L_priors.m:84,108`. Do NOT import
`parameterize_A`. SPM source-file + line citation is the allowed citation (REF-ERP-* still
"verify" in Zotero — do not fabricate a bib key).

### `src/pyro_dcm/inference/forward_models.py` (APPEND `class ERPDCMForward`, additive) — laptop

Implements the 8-member contract in the table above. Constructor stores `l_full`, `x_design`,
`b_masks`, `a_masks (tuple[4]×(n,n))`, `c_mask`, `dt`, `ns`, `ons_ms`, `dur_ms`, `sus`.
`predict` (lazy-import `cmc_network_f`, `apply_condition_modulation`,
`integrate_local_linearization`, `cmc_flatten`/`cmc_unflatten`, `erp_gaussian_input`,
`project_to_scalp`):

```python
def predict(self, theta, observed, n_regions, **context):
    p = self.unpack_params(theta, n_regions)        # {A_free(4,n,n), C_free, T, G, S, R}
    inputs = erp_gaussian_input(self._pst, p["R"], self._ons, self._dur, self._sus)
    x0 = torch.zeros(8*n_regions, dtype=torch.float64)
    y_list = []
    for c in range(self._x_design.shape[0]):
        q = apply_condition_modulation(p, self._x_design[c])   # spm_gen_Q port
        f_c = lambda v, u, q=q: cmc_network_f(v, u, q, n_regions)
        traj = integrate_local_linearization(f_c, x0, inputs, self._dt)  # (ns, 8n)
        y_list.append(project_to_scalp(traj, self._l_full))             # (ns, Nc)
    y = torch.stack(y_list, dim=0)                  # (Cnd, ns, Nc)
    if observed.ndim >= 3:                          # main-loop call; FD passes flat
        y = y[:, : observed.shape[1]]
    return y.reshape(-1)
```

`build_prior_cov` flattens `cmc_prior_moments` variances IN PACK ORDER and zeroes absent
A/C entries (so `_spm_svd` drops them — same idiom as `LatentCircuit:574-598`).
`build_precision` returns `([torch.eye(observed.numel(), dtype=f64)], 1)`.

### `src/pyro_dcm/simulators/erp_simulator.py` (EXTEND, LEAD-03) — laptop

Add an optional `l_full` arg (or a `lead_field` dict). When supplied, project the existing
`(Cnd, ns, n, 8)` source states (flatten each `(n,8)→(8n,)` via `cmc_flatten`, or project the
already-flat `(ns,8n)` traj before the `(ns,n,8)` reshape) to `(Cnd, ns, Nc)` and add keys
`"scalp": (Cnd,ns,Nc)` and `"difference_wave_scalp": scalp[1]-scalp[0]`. Keep the existing
`states`/`pst`/`inputs`/`difference_wave` (source) keys untouched — new keys only. Docstring:
the scalp difference wave's NON-ZEROness is the Phase-35 gate; the negative-going/frontal SIGN
is Phase 36 (needs ECD orientation + MNI coords — Fact 6).

### `validation/export_to_mat.py` (EXTEND, additive) — laptop

`export_erp_dcm_leadfield(...)` reuses the `_MS_` reference net + `export_erp_dcm_multisource`
machinery, ADDING the spatial spec to the DCM struct: `dipfit.type='LFP'`, `dipfit.Ns=5`,
`dipfit.Nc=5`; `P.L = np.ones((1,5))`; `P.J = sparse one-hot (1,8) at col 3`. All dims double.
Freeze `dipfit`/`P.J`/`P.L` in `DCM.meta`. Single-source / multi-source defaults unchanged
(existing fixtures byte-identical).

### `validation/matlab_scripts/run_spm_erp_dcm_leadfield.m` (NEW, LEAD-05) — M3

Mirror `run_spm_erp_dcm_multisource.m` scaffolding. With `M.f='spm_fx_cmc_nodelay'`,
`M.x=zeros(N,8)`, asserting `nargout(M.f)==2`:

```matlab
% (1) the LFP lead field (the spm_lx_erp parity target)
dipfit.type = 'LFP'; dipfit.Ns = N; dipfit.Nc = N;
L_full = spm_lx_erp(P, dipfit);            % (Nc, 8N)  spm_lx_erp.m:31-33
% (2) per-condition scalp ERP: reuse the spm_gen_erp SOURCE trajectory, project
for c = 1:size(DCM.U.X,1)
    Qc        = spm_gen_Q(P, DCM.U.X(c,:));
    ysrc      = spm_int_L(Qc, M, DCM.U);   % (ns, 8N)  (Phase-34 verified)
    y_scalp{c}= ysrc * L_full';            % (ns, Nc)  spm_lx_erp header y=L*x
end
diff_wave = y_scalp{2} - y_scalp{1};       % deviant - standard
save(output_path, 'L_full', 'y_scalp', 'diff_wave', 'meta');
```

Record `meta.D=1`, `meta.nargout_Mf`, `meta.N`, `meta.Nc`, `meta.dt`, `meta.ns`,
`meta.P_J`, `meta.P_L`, `meta.dipfit_type='LFP'`, SPM `$Id` strings, the `_MS_` edge list.

### `tests/test_spm_erp_leadfield_validation.py` (NEW, LEAD-05) — laptop (gated on fixture)

Mirror `tests/test_spm_erp_multisource_validation.py`: `_FIXTURE_PATH` +
`skipif(not .mat.exists)`, a fixture loader, and `_reference_p()` reconstructing the EXACT `P`
by importing the `_MS_` constants + `_ms_log_block` + `_erp_gaussian_u_grid` from
`validation.export_to_mat` (identical P+drive → pitfall V1 satisfied). REUSE the
`_spm_diff_jacobian(dx=exp(-8))` helper + `_update_operator` + `integrate_local_linearization`.

---

## The parity gate as a concrete test spec (LEAD-05 / LEAD-02 / LEAD-03)

Fixture arrays in `erp_leadfield_fixtures.mat` (N=5, Nc=5, Cnd=2, LFP mode):

| Array | Shape | What it pins | Tolerance |
|-------|-------|--------------|-----------|
| `L_full` | `(Nc, 8N) = (5,40)` | `kron(P.J, L_spatial)` column-major state-block order | ≤1e-12 element-wise |
| `y_scalp` | `{Cnd}×(ns,Nc)=(128,5)` | scalp projection of the verified trajectory | scheme ~1e-13; FD-Jac ≤1e-8; jacrev ≤1e-7 |
| `diff_wave` | `(ns,Nc)=(128,5)` | deviant−standard scalp difference | ≤1e-7 AND non-zero |

**Staged scalp ladder (assert IN THIS ORDER — V5; a failure localises to one stage):**

1. **`L_full` exact** (no integrator, pure algebra — the kron/P.J guard, LEAD-02): build
   `L_spatial = diag(ones(5))` (LFP default), `p_j = cmc_default_pj()`,
   `build_lead_field(p_j, L_spatial)` vs exported `L_full` ≤1e-12. Plus the
   distinct-valued kron column-major check (Fact 2) and the `P.J == index 2, not 6` guard
   (Fact 1). **This is the single most important lead-field guard (C5).**
2. **Scalp ERP — scheme rung** (bit-exact, isolates the projection from the Jacobian method):
   drive the exp-Euler loop with SPM's OWN per-condition `Qupd` (from the Phase-34 fixture, or
   recompute) → source traj ~machine-eps → `project_to_scalp` → assert vs `y_scalp{c}` on
   RELATIVE error (34-03-D1: network states ~O(40), so gate `max|diff|/max|y| ≤ 1e-12`).
3. **Scalp ERP — FD-Jacobian rung** (≤1e-8): build the operator from the `spm_diff`-matched
   `J0` (the `_spm_diff_jacobian` helper) → project → ≤1e-8.
4. **Scalp ERP — shipped-jacrev rung (THE LEAD-05 GATE, ≤1e-7):** the full production
   `integrate_local_linearization(cmc_network_f, ...)` → `project_to_scalp` → assert vs
   `y_scalp{c}` ≤1e-7. **Finding (verify by measurement):** the Phase-34 shipped-jacrev
   *source* floor was 4.70e-8 (`34-03-SUMMARY` rung 6). The LFP default lead field is the
   IDENTITY (`P.L=ones` → `L_spatial=I`), so the projection does NOT amplify — the scalp
   jacrev floor stays ≈4.7e-8 **< 1e-7**. Unlike Phase 34 (where 4.7e-8 > 1e-8 forced the
   shipped path to be measured-not-gated), Phase 35's looser ≤1e-7 LEAD-05 tolerance lets the
   PRODUCTION integrator be GATED directly. Keep rungs 2–3 as diagnostic localisation rungs.
   (Caveat: if a non-identity LFP gain `P.L≠1` is ever used, the projection scales the floor
   by `max|P.L|` — MEASURE and re-confirm ≤1e-7 then.)
5. **Difference wave** (LEAD-03): `scalp[1] − scalp[0]` (production path) vs exported
   `diff_wave` ≤1e-7, AND assert `max|diff_wave| > 0` (non-zero — it is, since `B` is wired:
   `_MS_B_EDGE`/`_MS_B_DIAG`). The negative-going/frontal SIGN is **deferred to Phase 36**
   (Fact 6); Phase 35 may record the source-level sign direction as a non-gating diagnostic.

**Structural guards (laptop, no MATLAB — in `test_erp_leadfield.py`, Wave 1):**
- **P.J guard (LEAD-02):** `cmc_default_pj()` one-hot at index 2; assert `argmax == 2`, assert
  `!= 6`; cite `spm_L_priors.m:108`.
- **kron column-major (LEAD-02):** distinct-valued `L_spatial` + non-trivial `p_j`; assert
  `build_lead_field` block `s` occupies columns `[s*n:(s+1)*n]` and equals `p_j[s]*L_spatial`.
- **LFP identity:** `lfp_spatial(ones(n), n) == eye(n)`; `build_lead_field(e_2, I_n)` has the
  sp-voltage block `= I_n` and all other blocks `= 0`.
- **Projection through identity LFP** = sp-voltage trace: `project_to_scalp(states, kron(e_2,
  I_n))[..., j] == states sp-voltage of source j`.
- **Difference-wave non-zero** on the planted ref net (source + scalp).
- **float64** at the lead-field + projection boundary (pitfall N1).
- **`observed.ndim` guard** in `ERPDCMForward.predict` (both `(Cnd,ns,Nc)` and flat FD call).

Tolerances are element-wise forward agreement only (V2 — no absolute-F, no `Cp`; the forward
has no normalisation freedom). Anchor to the Phase-33/34 measured floors (`matrix_exp↔spm_expm`
8.6e-11; scheme 6.6e-14 single-source / 8.2e-13 rel at N=5; jacrev source 4.7e-8) as small
multiples (V3).

---

## VL round-trip spec (LEAD-06 — protocol confirmation, NOT a parity gate)

In `tests/test_erp_leadfield.py` (Wave 1, laptop, no MATLAB):

1. Plant CMC params on the `_MS_` 5-source net (or a smaller 2–3-source net for speed): a
   live A graph (`_MS_A_LIVE`/`_MS_A_DEAD` blocks), a C mask into bilateral A1, a non-trivial
   `B` (deviant), and a perturbed `G[:,0]` (precision knob) so recovery has signal.
2. `simulate_erp_dcm(p, x_design, n, l_full=kron(e_2, I_n))` → `obs (Cnd,ns,Nc)` (add light
   Gaussian noise so the residual is non-degenerate).
3. `result = run_variational_laplace_generic(ERPDCMForward(l_full=..., x_design=...,
   a_masks=..., b_masks=..., c_mask=..., dt=0.004, ns=128, ...), obs, a_mask=<union graph>,
   n_regions=n, ...)`.
4. Assert `result.theta_post["A"]` / `C` / `G` / `T` recovered within tolerance of planted
   (per-region R²-style or RMSE; reuse the VLREC metric idioms). This confirms the protocol
   wiring end-to-end (param_count → build_prior_cov → SVD → predict → ReML → build_result),
   NOT SPM parity. Document it as such (V1 — a torch-vs-torch round-trip is a complement, not
   the parity gate).

**Routing:** the VL round-trip on a 5-source × 2-cond × 128-step problem with the dense
`(Cnd·ns·Nc)=(1280)` identity precision and SVD over ~`4·25+...` params is borderline. A
single VL fit of this size is likely <3 min on laptop, but a **multi-restart or
multi-seed** round-trip MUST route to M3 (CLAUDE.md >3 min rule; subagents inherit it).
Recommend: a SINGLE-seed, ≤32-iter round-trip on a SMALL net (n=2 or 3) on laptop for the
LEAD-06 gate; defer any multi-seed recovery sweep to a (non-blocking) M3 job. **MEASURE the
laptop wall-time on first run; if >3 min, move to M3.**

---

## Compute routing

| Work | Where | Why |
|------|-------|-----|
| Wave 1 modules + `test_erp_leadfield.py` structural guards | **Laptop** | pure-torch; kron + projection are sub-millisecond; the guards reuse committed fixtures |
| Wave 1 VL round-trip (LEAD-06) | **Laptop (small net, single seed, ≤32 iter)** | one VL fit likely <3 min; MEASURE first run; any multi-seed/restart → M3 |
| Wave 2 MATLAB fixtures (`run_spm_erp_dcm_leadfield.m`) | **M3** | local MATLAB FlexLM unreachable; R2022a + Carrick spm12 on `comp`; submit via `cluster/sbatch/erp_cross_validation.sbatch` |
| Wave 3 `test_spm_erp_leadfield_validation.py` | **Laptop (against committed `.mat`)** + optional M3 re-run | assertions are torch-vs-frozen-arrays (deterministic f64) — the laptop run IS authoritative (33-03-D1/34-03-D2) |

ssh-agent unlocked at submit; Mutagen sync for code/results (not git push/pull). Fixtures are
tiny (`(5,40)` + `(128,5)`×2); the M3 job is minutes, well under `--time=01:00:00`. Heed the
Mutagen `models/` ignore footgun (anchor ignores) when syncing `src/pyro_dcm/models/`.

---

## Don't hand-roll

| Problem | Don't build | Use instead | Why |
|---------|-------------|-------------|-----|
| State→scalp lead field | a manual loop over states/sources | `torch.kron(p_j.reshape(1,8), l_spatial)` | one call; exactly matches `spm_lx_erp.m:33`; column-major block order is automatic (Fact 2) |
| LFP spatial model | a custom sparse builder | `torch.diag(p_l)` | `spm_erp_L.m:112` `sparse(1:m,1:m,P.L)` IS a diagonal; default `ones`→`I` |
| ECD physical gain | FieldTrip/MNE head model in Python | MATLAB-exported `G (Nc,3,n)` post `spm_cond_units` | zero new deps; reproducing `spm_cond_units` + `ft_compute_leadfield` is out of scope (anti-feature) |
| The integrator / network forward | a new ODE solver or re-derived EOM | `integrate_local_linearization` + `cmc_network_f` (Phase 33/34) | already bit-close to SPM (8 rungs green at N=5); editing them risks the parity gate |
| VL inference | a bespoke optimiser | `run_variational_laplace_generic(ERPDCMForward(), ...)` | the protocol gives VL + amortized for free (E3); zero engine edits |
| Precision matrix | AR(1) `spm_Q` now | identity `eye(Cnd·ns·Nc)` v1 | forward-only scope; AR(1) only if F/inference parity is later attempted (M3) |

---

## Common pitfalls (Phase-35-specific, from PITFALLS.md C5 + this session)

1. **Observing the wrong state (C5.1).** Using index 6 (dp voltage) or an even
   conductance index instead of index 2 (sp voltage) gives a physiologically-inverted scalp
   signal that passes shape tests. → `cmc_default_pj()` hard-asserts index 2 (Fact 1).
2. **C-order kron (C5.2).** A `reshape` flatten puts the lead-field block at `source*8+state`
   instead of `state*n+source`, silently mapping the lead field to the wrong states. → assert
   `build_lead_field` vs the exported kron with distinct-valued `L_spatial` (Fact 2).
3. **Missing `spm_cond_units` on ECD gain (C5.3).** Re-deriving the ECD gain without the
   unit-conditioning rescale gives a clean but wrong scalar scale. → export `G` from MATLAB
   POST `spm_cond_units`; Python only does kron+projection (deferred to Phase 36).
4. **Difference-wave sign over-claim (S2).** Asserting negative-going/frontal at the scalp in
   LFP identity mode pins a polarity the convention does not physically fix (needs ECD
   orientation). → Phase 35 gates NON-ZERO + parity only; sign deferred to Phase 36 (Fact 6).
5. **Observation-stacking misalignment (gap 4).** A wrong axis order or Fortran flatten
   produces a clean-looking wrong residual. → lock `(Cnd,ns,Nc)` + C-order `reshape(-1)`
   before any assertion (above).
6. **float32 creep (N1)** and **eig-clip reflex (N2).** Neither belongs in the CMC path;
   assert float64 at the lead-field boundary; never clip the CMC Jacobian.
7. **Editing the protocol or VL engine (B2).** ERP-specific needs ride as constructor args +
   `context`; the `ForwardModel` Protocol and `variational_laplace.py` are NEVER edited.
8. **FD flat-vector contract (B3).** `predict` must handle both `(Cnd,ns,Nc)` and the flat FD
   call — mirror the `LatentCircuitForward.predict` `observed.ndim` guard.

---

## Open questions / decisions for the planner

1. **ECD gain export in Phase 35 vs Phase 36.** ECD needs a sensor montage + MNI coords
   (flagged Phase 36 / `ERPDCM-03`, "verify coords before hard-coding"). **Recommend: Phase
   35 ships LFP-mode parity only (LEAD-05 gate); build `ecd_spatial()` to consume an exported
   `G (Nc,3,n)` but produce the ECD gain fixture in Phase 36.** This is the head-model-free,
   defensible gate (PITFALLS C5.4). Decision needed: confirm LFP-only for the Phase-35 gate.

2. **Difference-wave sign scope.** LEAD-03 text says "non-zero and negative-going," but the
   objective + Fact 6 defer the SIGN (needs orientation/coords) to Phase 36. **Recommend:
   Phase 35 gates NON-ZERO + element-wise parity vs `diff_wave`; record source-level sign as a
   non-gating diagnostic; assert negative-going/frontal in Phase 36 where ECD orientation
   exists.** Decision needed: confirm the LEAD-03 scope split.

3. **Which params are free in `pack`/`unpack` (and whether L/J/R are in the vector).** The
   milestone text lists `param_count` over `A+C+T+G+S+L+J+R`. **Recommend the canonical FROZEN
   ordering `A(4nn)+C+T(4n)+G(4n)+S(n)+R(2·n_inp)`** for the dynamics+input-timing, with **L
   and J held FIXED** (carried in the precomputed `l_full`) in v1 — the lead field is context,
   not a recovered param, for the forward-parity + round-trip. If the planner wants L/J freed
   for a richer round-trip, append them with their `spm_L_priors` variances (L `64`, J `1/32`
   at indices `[0,6]`) and zero them for the parity gate. Decision needed: lock the vector.

4. **The 4-block `a_mask` handling.** The protocol passes one `(n,n)` `a_mask`; CMC needs 4
   distinct extrinsic-block masks. **Recommend `ERPDCMForward` store its own `a_masks
   (tuple[4]×(n,n))` as a constructor arg and use them in `build_prior_cov`/`unpack`, treating
   the engine-supplied scalar `a_mask` as a compatibility no-op** (mirrors `TaskDCMForward`
   storing `c_mask` internally). Decision needed: confirm the 4-mask constructor contract.

5. **VL round-trip net size + routing.** **Recommend a SMALL net (n=2–3), single seed, ≤32
   iter on laptop for LEAD-06; MEASURE wall-time; any multi-seed/restart → M3.** Decision
   needed: net size for the round-trip.

6. **Reuse the Phase-34 source trajectory in the lead-field fixture.** The Wave-2 script can
   recompute `ysrc = spm_int_L(Qc,M,U)` (clean, self-contained) OR load the Phase-34
   `erp_multisource_fixtures.mat` `y{c}`. **Recommend recompute in
   `run_spm_erp_dcm_leadfield.m`** (one fixture, one provenance header, no cross-fixture
   coupling). Decision: confirm.

---

## Sources

### Primary (HIGH — SPM12 source read line-by-line this session at `C:/Users/aman0087/Documents/Github/spm12/toolbox/dcm_meeg/`)
- `spm_lx_erp.m` (`$Id: 7256`) — `L = spm_erp_L(P,dipfit)` (`:31`), `L = kron(P.J,L)` (`:33`,
  numeric-J branch), the observer `y = G*x` (header `:2-9`).
- `spm_erp_L.m` (`$Id: 7142`) — ECD branch `L(:,i)=G(:,:,i)*P.L(:,i)` with `spm_cond_units`
  (`:43-77`), LFP branch `L=sparse(1:m,1:m,P.L,m,n)` (`:105-118`), siunits MNI→m rescale
  (`:59-60`).
- `spm_L_priors.m` (`$Id: 7409`) — CMC `pE.J=sparse(1,3,1,1,8)` / `pC.J=sparse(1,[1 7],1/32)`
  (`:106-109`), LFP `pE.L=ones(1,m)` / `pC.L=ones*64` (`:84`), ECD `pE.L=zeros(3,n)` / `*64`
  (`:67`), single-J vector collapse (`:190-193`).

### Primary (HIGH — repo source / shipped Phase-33/34 artifacts read this session)
- `src/pyro_dcm/inference/forward_models.py:30-117` (the `ForwardModel` Protocol, 8 members),
  `:451-683` (`LatentCircuitForward` — the additive 4th-implementor precedent: `predict`
  `observed.ndim` guard, identity `build_precision`, lazy import idiom).
- `src/pyro_dcm/inference/variational_laplace.py:950-1068` (`_run_vl_generic` dispatch:
  `context["a_mask"]`, `param_count`→`build_prior_cov`→`_spm_svd`→`build_precision`→
  `observed.reshape(-1)`; the FD-Jacobian flat `predict` call `:959-966`).
- `src/pyro_dcm/forward_models/cmc_neural_mass.py` (`cmc_flatten=x.T.reshape(-1)` column-major,
  8-state column layout `:20-33`, `J_PERM`, `cmc_sigmoid`), `erp_coupled_system.py`
  (`cmc_network_f`, `apply_condition_modulation`, `parameterize_cmc_network`), `cmc_priors.py`
  (`cmc_prior_moments`, `cmc_steady_state=zeros`), `utils/local_linearization.py`
  (`integrate_local_linearization`, `_update_operator`), `simulators/erp_simulator.py`
  (`simulate_erp_dcm` → `(Cnd,ns,n,8)` + source `difference_wave`).
- `validation/export_to_mat.py:440-943` (`export_erp_dcm` / `export_erp_dcm_multisource`,
  `_MS_*` 5-source reference net, `_ms_log_block`, `_erp_gaussian_u_grid`, the int64→double
  `spm_Ce` footgun fix), `validation/matlab_scripts/run_spm_erp_dcm_multisource.m`
  (scaffolding + `spm_fx_cmc_nodelay` D=1 wrapper to mirror),
  `tests/test_spm_erp_multisource_validation.py` (the V5 ladder + `_spm_diff_jacobian` /
  `_reference_p` / fixture-loader pattern to reuse), `cluster/scripts/erp_cross_validation.py`
  + `cluster/sbatch/erp_cross_validation.sbatch` (M3 entrypoint to extend).
- `.planning/phases/34-.../34-RESEARCH.md` + `34-03-SUMMARY.md` (locked conventions: 8N
  column-major flatten, `(Cnd,ns,n,8)`, the jacrev-vs-FD 3-way gate split + measured floors
  matrix_exp 8.6e-11 / scheme 8.2e-13-rel / jacrev 4.7e-8, D=1 wrapper).
- `.planning/REQUIREMENTS.md:399-417` (LEAD-01..06).

### Peer milestone research (consumed, not re-derived)
- `.planning/research/v0.8.0/{SUMMARY,ARCHITECTURE,PITFALLS}.md` — C5 (lead-field traps),
  B2/B3 (additive protocol seam), V1-V5 (validation methodology), gap 4 (stacking layout),
  the boundary shape table.

## Metadata

**Confidence breakdown:**
- Lead-field algebra (`kron(P.J,L)`, P.J default index 2, LFP diag) — **HIGH**: transcribed
  from `spm_lx_erp.m:31-33` / `spm_erp_L.m:105-118` / `spm_L_priors.m:84,108`; the kron↔
  column-major correspondence proven against the shipped `cmc_flatten` (N=5 rung 2 green).
- `ERPDCMForward` protocol contract — **HIGH**: 8 members + `LatentCircuitForward` precedent
  read directly; ERP-specific needs ride as constructor/context (B2 honored).
- Observation stacking decision — **HIGH**: locked `(Cnd,ns,Nc)` C-order, justified against
  the identity-precision idiom + the difference-wave axis requirement.
- Scalp parity ladder + the ≤1e-7 production-path finding — **HIGH** for the structure;
  **MEDIUM** for the exact claim that the shipped jacrev scalp floor (≈4.7e-8) sits below 1e-7
  — Wave 3 must MEASURE it (LFP identity → no amplification is the load-bearing assumption).
- ECD deferral — **HIGH**: ECD needs a head model + coords absent until Phase 36; LFP-first is
  the documented recommendation (C5.4).

**Research date:** 2026-06-26
**Valid until:** stable (SPM12 frozen at `$Id 7256/7142/7409`; repo conventions stable) — ~30 days.

## RESEARCH COMPLETE
