# Phase 33: CMC Core Dynamics, spm_int_L Integrator & Single-Source Parity — Research

**Researched:** 2026-06-25
**Domain:** CMC neural-mass forward (4 populations / 8 states) + exponential-Euler
(`spm_int_L`) integrator + single-source SPM12 parity on frozen MATLAB fixtures
**Confidence:** HIGH (every equation and seam below transcribed from the actual SPM12
source at `C:/Users/aman0087/Documents/Github/spm12` and the actual repo source, read
line-by-line this session)

> This file is the **Phase-33 implementation concretes**. It does NOT re-survey the
> milestone — read `.planning/research/v0.8.0/{SUMMARY,STACK,PITFALLS,ARCHITECTURE}.md`
> for the why. This file gives the *exact* equations to port (with `spm_fx_cmc.m` line
> refs), the *exact* repo seams to extend, and the *exact* MATLAB fixture-generation
> script structure, so the planner can cut 2–4 executable plans with no further source
> reading.

---

## Summary

Phase 33 ports a **single source** (n=1) of the SPM12 canonical-microcircuit forward
(`spm_fx_cmc.m`) and the SPM12 evoked integrator (`spm_int_L.m`) into pure torch
(float64, zero new deps), and proves element-wise parity against frozen MATLAB fixtures
generated on M3 (R2022a + Carrick spm12). At n=1 the extrinsic `A{1..4}` blocks are
**identically zero** (`spm_fx_cmc.m:73-75` — the `else A = {0,0,0,0}` branch), so the
network coupling and condition modulation `B` are entirely deferred to Phase 34; Phase 33
is the intrinsic-dynamics + integrator + input + parity-harness foundation only.

The two load-bearing, non-obvious facts the implementer must not get wrong: (1) SPM
integrates the ERP with a **frozen-Jacobian exponential-Euler** scheme
(`spm_int_L.m:124-143`), NOT Runge-Kutta — so CMC must be routed through a NEW
`utils/local_linearization.py`, never `integrate_ode`; and (2) the CMC delay operator is
**default-ON** even for a single source (`spm_fx_cmc.m:226` calls `spm_dcm_delay`, which
inserts 1 ms intrinsic inter-population delays), so the fixture-generation MATLAB script
must explicitly force `D = identity` — the cleanest way is a 2-output wrapper m-file that
makes `nargout(M.f) == 2`, which sends `spm_int_L` down its `D = 1` branch
(`spm_int_L.m:112,117`).

**Primary recommendation:** Build in strict order — integrator (`local_linearization.py`)
+ CMC `f(x,u,P)` + `parameterize_cmc` permutation guard FIRST (laptop, pure-torch), then
the MATLAB fixture-gen + export bridge (M3), then the parity assertions. Write the
single-source parity test (`f`-field → `J0` → `Q_update` → `y_states`, the V5 ladder)
*before* any extrinsic coupling exists, so the integration-scheme mismatch (pitfall C1) is
caught in isolation.

---

## Source-resolved facts (the things the milestone left to "the implementer")

### Fact 1 — The exact `spm_fx_cmc` equations of motion (transcribe verbatim, 0-indexed)

Source: `spm12/toolbox/dcm_meeg/spm_fx_cmc.m`. State `x` shape `(n,8)`; column meaning
from `spm_fx_cmc.m:6-14`:

| 0-idx col | MATLAB col | population | quantity |
|-----------|-----------|------------|----------|
| `x[:,0]` | x(:,1) | spiny stellate (ss) | voltage |
| `x[:,1]` | x(:,2) | spiny stellate | conductance |
| `x[:,2]` | x(:,3) | superficial pyramidal (sp) | voltage |
| `x[:,3]` | x(:,4) | superficial pyramidal | conductance |
| `x[:,4]` | x(:,5) | inhibitory interneurons (ii) | current |
| `x[:,5]` | x(:,6) | inhibitory interneurons | conductance |
| `x[:,6]` | x(:,7) | deep pyramidal (dp) | voltage |
| `x[:,7]` | x(:,8) | deep pyramidal | conductance |

Fixed defaults (`spm_fx_cmc.m:47-49`):
```
E0 = [1, 0.5, 1, 0.5] * 200          # extrinsic rates (unused at n=1)
G0 = [4, 4, 8, 4, 4, 2, 4, 4, 2, 1] * 200   # 10 intrinsic strengths
T0 = [2, 2, 16, 28]                  # synaptic time constants (ms)
```

Sigmoid (`spm_fx_cmc.m:90-94`), applied elementwise to all 8 state columns:
```
R = (2/3) * exp(P.S)                 # P.S scalar per source; bias B=0
F = 1 / (1 + exp(-R * x))            # exp(-R*x + 0)
S = F - 1/2                          # F - 1/(1+exp(0)); the -1/2 is load-bearing
```

Input (`spm_fx_cmc.m:86,107`) — exogenous branch (the one that runs during ERP, because
`spm_gen_erp.m:48-54` removes `M.u`):
```
C = exp(P.C)
U = C * u * 32                       # u is the per-sample scalar from spm_erp_u (already 32*Gaussian)
```

Time-constant + intrinsic transforms (`spm_fx_cmc.m:114-160`):
```
T = (ones(n,1) * T0) / 1000          # ms -> seconds  (line 114)
T[:,i] *= exp(P.T[:,i])  for i in 0..3            # 4 free  (line 148-150)
G = ones(n,1) * G0                                  # (n,10)
j = [6,1,2,3,0,4,5,7,8,9]            # 0-indexed perm of MATLAB [7 2 3 4 1 5 6 8 9 10]
G[:, j[i]] *= exp(P.G[:,i])  for i in 0..3          # 4 free  (line 151-154)
if "M" in P: G[:,6] *= exp(-P.M * 32 * S[:,6])      # line 158-160 (deferred — optional)
```

Equations of motion (`spm_fx_cmc.m:171-198`), 0-indexed, `@` = extrinsic matmul (zero at
n=1):
```
# Granular — spiny stellate -> f[:,1]
uu = A1 @ S[:,2] + U
uu = -G[:,0]*S[:,0] - G[:,2]*S[:,4] - G[:,1]*S[:,2] + uu
f[:,1] = (uu - 2*x[:,1] - x[:,0]/T[:,0]) / T[:,0]

# Supragranular — superficial pyramidal -> f[:,3]
uu = -A3 @ S[:,6]
uu = G[:,7]*S[:,0] - G[:,6]*S[:,2] + uu
f[:,3] = (uu - 2*x[:,3] - x[:,2]/T[:,1]) / T[:,1]

# Supragranular — inhibitory interneurons -> f[:,5]
uu = -A4 @ S[:,6]
uu = G[:,4]*S[:,0] + G[:,5]*S[:,6] - G[:,3]*S[:,4] + uu
f[:,5] = (uu - 2*x[:,5] - x[:,4]/T[:,2]) / T[:,2]

# Infragranular — deep pyramidal -> f[:,7]
uu = A2 @ S[:,2]
uu = -G[:,9]*S[:,6] - G[:,8]*S[:,4] + uu
f[:,7] = (uu - 2*x[:,7] - x[:,6]/T[:,3]) / T[:,3]

# Voltages are integrals of conductances (line 195-198)
f[:,0] = x[:,1]; f[:,2] = x[:,3]; f[:,4] = x[:,5]; f[:,6] = x[:,7]
```
G-index map is MATLAB `G(:,k)` → Python `G[:,k-1]` (e.g. `G(:,7)` sp-self = `G[:,6]`).

**Flatten (parity-critical, `spm_fx_cmc.m:199` `spm_vec`):** MATLAB `spm_vec` is
column-major. Python: `x_flat = x.T.reshape(-1)` and `x = x_flat.reshape(8, n).T`. At n=1
this is the trivial 8-vector `[V_ss,I_ss,V_sp,I_sp,Iii,Cii,V_dp,I_dp]`.

### Fact 2 — The precision index, resolved unambiguously

`spm_fx_cmc.m:151` `j = [7 2 3 4 1 5 6 8 9 10]` (MATLAB, 1-indexed). Free param `P.G`
column **1** maps to `G(:,j(1)) = G(:,7)`, and `G(:,7)` is **`sp -> sp` superficial-
pyramidal self-inhibition** (`spm_fx_cmc.m:132`). Therefore:

- MATLAB `G(:,7)` ≡ Python `G[:,6]` ≡ **the sp self-inhibition / precision knob** — the
  SAME connection, just different indexing bases. There is no contradiction in the
  milestone docs; "G(:,7) 1-indexed" and "G[:,6] 0-indexed" name one column.
- The driving free parameter is `P.G` column **1** (MATLAB) ≡ `P_G[:, 0]` (Python).
- 0-indexed permutation array: `j = [6, 1, 2, 3, 0, 4, 5, 7, 8, 9]`.
- The modulatory `P.M` path (`spm_fx_cmc.m:159`) hits the SAME `G[:,6]`, gated by
  `S[:,6]` (deep-pyramidal voltage firing).
- **Permutation guard (CMC-02):** perturb `P_G[:,0]`, assert `G[:,6]` changes and `G[:,0]`
  does NOT.

(Lead-field `P.J` default = state index 2 / sp-voltage is a *different* index and belongs
to Phase 35 — out of scope here; do not conflate with the `G[:,6]` precision index.)

### Fact 3 — The exact `spm_int_L` update loop (port verbatim)

Source `spm_int_L.m:112-169`. With `dt = U.dt`, `N = 1` (default; `spm_gen_erp.m:84` calls
`spm_int_L(Q,M,U)` with nargin 3 → `spm_int_L.m:70` sets `N=1`), `D` = delay operator:
```
dfdx = J0 - I * exp(-16)                       # :126 regulariser, applied BEFORE Q
Q    = (spm_expm(dt*D*dfdx/N) - I) / dfdx       # :127  right-division = (E-I)*inv(J)
v    = x0_flat                                  # :131  spm_vec(M.x)
for i in range(ns):                             # :132
    u = U[i, :]
    for _ in range(N):                          # :141
        v = v + Q @ f(v, u)                     # :142
    y[i] = g(v, u)                              # :147  (g = identity for states)
y = real(y.T)                                   # :169  -> (ns, 8n)
```
Torch port (in `utils/local_linearization.py`):
```
J  = frozen_jacobian(f, x0_flat, u0)            # at x0=0, u0=0  (see Fact 5)
J  = J - torch.eye(8n, dtype=f64) * exp(-16.0)
E  = torch.matrix_exp(dt * D @ J / N)           # D = eye for Phase 33
Q  = torch.linalg.solve(J.T, (E - I).T).T       # (E-I) @ inv(J); NEVER torch.inverse
v  = x0_flat.clone()
for i in range(ns):
    for _ in range(N):
        v = v + Q @ f(v, U[i])
    y[i] = v
```
Notes:
- **Right-division (pitfall C2):** `(E-I)/dfdx` is `(E-I)*inv(J)`, NOT `inv(J)*(E-I)`. `J`
  is asymmetric. Use `torch.linalg.solve(J.T, (E-I).T).T`.
- `exp(-16)` shift is applied to `J` BEFORE forming both `E` and the `inv(J)` in `Q`.
- `spm_expm.m` is a degree-6 Padé + scaling-and-squaring (`spm_expm.m:38-63`);
  `torch.matrix_exp` is the same algorithm family (higher Padé degree for f64). Agreement
  must be MEASURED, not assumed (see parity gate, Q_update tier).

### Fact 4 — How to force D = identity in the fixtures (the delay default trap, M2)

`spm_fx_cmc` declares 3 outputs `[f,J,Q]` (`spm_fx_cmc.m:1`), and `spm_int_L.m:114`
`if nargout(f) >= 3` then `[fx,dfdx,D] = f(...)` — so by default it pulls
`D = spm_dcm_delay(P,M,J)` (`spm_fx_cmc.m:226`). For a single source `spm_dcm_delay.m`
still injects **1 ms intrinsic inter-population delays** (`spm_dcm_delay.m:60-82`, `di=1`),
so D ≠ I even at n=1. Two clean ways to force D=I, in order of preference:

1. **2-output wrapper m-file (recommended — gives EXACT D=I via SPM's own loop).** Add
   `validation/matlab_scripts/spm_fx_cmc_nodelay.m`:
   ```matlab
   function [f,J] = spm_fx_cmc_nodelay(x,u,P,M)
   [f,J] = spm_fx_cmc(x,u,P,M);
   ```
   Set `M.f = 'spm_fx_cmc_nodelay'`. Then `nargout(f) == 2` →
   `spm_int_L.m:117` branch → `D` stays `= 1` (initialized `spm_int_L.m:112`). Uses
   `spm_fx_cmc`'s own analytic Jacobian (`= spm_diff`, `spm_fx_cmc.m:208`).
2. **Strip `P.D`** so `spm_dcm_delay.m:55` `isfield(P,'D')` is false → `D = sparse(0)`
   (`:107`) → returned operator ≈ `J*inv(J) ≈ I` (`:176`) but with `exp(-16)` round-off
   (~1e-7, NOT exact). Inferior; use only as a cross-check.

**The fixture script must assert D=I** (e.g. export the wrapper path's effective operator,
or simply document that `nargout(M.f)==2`). Also force `M.N` irrelevant: `spm_gen_erp.m:29`
sets `M.N=0` but `spm_int_L` is called with nargin 3 so its `N=1`.

### Fact 5 — The frozen Jacobian / steady state / input, resolved

- **Steady state x0 = zeros(1,8) (M1).** `spm_gen_erp.m:80` calls `spm_dcm_neural_x`; for
  `spm_fx_cmc` it hits the `otherwise` branch (`spm_dcm_neural_x.m:70-72`) which does
  nothing → `x = M.x`, and `M.x` is initialized to `sparse(n,8)` zeros. At x=0, u=0:
  `S = F-1/2 = 0`, `U=0`, so `f(0,0)=0` — zeros IS the fixed point. Assert `x0 == 0`; do
  NOT port a Newton solver.
- **The frozen Jacobian J0 is df/dx at x=0, u=0.** Inside `spm_fx_cmc`'s own Jacobian
  (`spm_fx_cmc.m:206-208`) it uses `M.x` and `M.u`; ERP removes `M.u` so `u=0`. MATLAB
  fixture: `J0 = full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, Q, M, 1)))` → (8,8). Torch:
  evaluate the Jacobian of `f` w.r.t. flat state at `x0=0, u0=0` (autograd
  `torch.func.jacrev`, or finite difference — either is fine, the fixture pins it).
- **Gaussian input `spm_erp_u.m:42-64`** (port to `erp_input.py`), `t` in ms:
  ```
  t_ms  = t * 1000
  delay = M.ons[i] + 128 * P.R[i,0]          # M.ons default 60 ms
  scale = M.dur[i] * exp(P.R[i,1])           # M.dur default 16 (or 32, see N3)
  Ug    = exp(-(t_ms - delay)**2 / (2*scale**2))
  prop  = M.sus[i] * exp(P.R[i,2])           # M.sus default 0 -> prop 0
  Ug    = prop * cumsum(Ug)/sum(Ug) + Ug*(1-prop)
  u[:,i] = 32 * Ug                           # the 32-scaling lives HERE
  ```
  Default ERP grid (`spm_gen_erp.m:28-30`): `M.ns=128`, `U.dt=0.004` s (4 ms),
  `M.ons=60` ms → 512 ms peristimulus window.

### Fact 6 — `spm_cmc_priors.m` (direct transcription for `cmc_priors.py`)

Source `spm_cmc_priors.m`. Log-normal: prior mean of every *scaling* free param is 0,
variance as below. Single source (n=1, u=1 input):

| Param | Prior mean E | Prior var V | Source line |
|-------|--------------|-------------|-------------|
| `P.T` (4 free time consts) | `zeros(1,4)` | `1/32` each | `:121` |
| `P.G` (4 free intrinsic) | `zeros(n,4)` | `1/32` each | `:122` |
| `P.S` (sigmoid slope) | `0` | `1/64` | `:124` |
| `P.C` (input, per active entry) | `mask*32 - 32` | `mask/32` | `:114-116` |
| `P.A{1..4}` (extrinsic; **n>1 only**) | `mask*32 - 32` | `mask/16` | `:80-81` |
| `P.D` (delays) | `zeros(n,n)` | `Q/64` | `:123` — **omit/zero in Phase-33 fixtures** |
| `P.M` (modulatory; optional) | `0` | `mask/32` | `:71-72` |
| `P.R` (onset, dispersion) | `zeros(u,2)` | `[1/16, 1/16]` | `:133` |

`spm_cmc_priors.m:78-83` restructures A so `D{1}=D{2}=A{1}` (forward), `D{3}=D{4}=A{2}`
(backward) — Phase 34 detail. For Phase 33, `build_prior_cov` only needs the
single-source diagonal entries: 4×T + 4×G + 1×S + 1×C (+ R if input params packed).
Absent-connection variance = 0 so SVD reduction drops them (matches the existing
`build_prior_cov` contract, `inference/forward_models.py:59-70`).

---

## Plan decomposition recommendation

Three plans, three waves. Strict ordering: the pure-torch dynamics + integrator + the
permutation guard must land and be green on laptop BEFORE the M3 fixture round-trip, so the
parity test has something to assert against and the C1 mismatch is caught in isolation.

### Wave 1 — Pure-torch core (LAPTOP, <30 s unit tests) — Plan 33-01
Parallelizable internally; the four new modules have no cross-imports except the test.
- `utils/local_linearization.py` (CMC-03) — the `spm_int_L` port.
- `forward_models/cmc_neural_mass.py` (CMC-01, CMC-02) — `f(x,u,P)`, sigmoid,
  `parameterize_cmc` with the permutation `j`.
- `forward_models/cmc_priors.py` (CMC-04) — prior tables + `x0==0` assertion helper.
- `forward_models/erp_input.py` (CMC-05) — `spm_erp_u` Gaussian bump.
- `tests/test_cmc_forward.py` — permutation guard (`P_G[:,0]`→`G[:,6]`), float64 guard
  (CMC-07), no-eig-clip guard, `x0==0`.
- `tests/test_local_linearization.py` — internal-consistency (right-division shape,
  regulariser ordering, dtype). The matrix_exp-vs-spm_expm *measurement* lands in Wave 3
  (needs the MATLAB array).

Gate: all laptop unit tests green; `git diff` shows only NEW files + an append to
`forward_models/__init__.py`. No SPM yet.

### Wave 2 — Fixture generation + export bridge (M3, MATLAB) — Plan 33-02
Depends on Wave 1 (needs the single-source reference `P` struct shape locked).
- `validation/export_to_mat.py` += `export_erp_dcm(...)` (single-source DCM-erp `.mat`).
- `validation/matlab_scripts/run_spm_erp_dcm.m` — generates the 5 fixture arrays (below).
- `validation/matlab_scripts/spm_fx_cmc_nodelay.m` — the 2-output D=I wrapper.
- `cluster/scripts/erp_cross_validation.py` — M3 entrypoint (mirrors
  `spm_cross_validation.py`, record-don't-crash).
- `cluster/sbatch/erp_cross_validation.sbatch` — mirrors `spm_cross_validation.sbatch`
  (exports `MATLAB_PATH`, `SPM12_PATH`; `--partition=comp`).

Gate: M3 job produces `validation/data/erp_single_source_fixtures.mat` with the 5 arrays +
a provenance metadata header (SPM `$Id`, `dt`, `ns`, `M.ons/dur`, `D=1`, the exact `P`
struct).

### Wave 3 — Parity assertions (LAPTOP, SPM-gated → auto-skip; real run on M3) — Plan 33-03
Depends on Wave 1 + Wave 2.
- `tests/test_spm_erp_dcm_validation.py` — loads the `.mat`, asserts the V5 ladder
  (CMC-06). Marked `@pytest.mark.spm` + `@pytest.mark.slow`, skipif MATLAB unavailable.
- Extend `tests/test_local_linearization.py` — the MEASURED matrix_exp-vs-spm_expm floor
  on the exported `dt*J0`.

Gate (CMC-06): `f`-field ≤1e-10, `J0` ≤1e-10, `Q_update` ≤1e-9, `y_states` ≤1e-8.

**Why this split, not a single plan:** Wave 1 is laptop pure-torch with sub-second tests;
Wave 2 is the only M3/MATLAB piece (license-gated, slow, record-don't-crash); Wave 3 is the
gate. Splitting lets Wave 1 be reviewed/merged and the C1-isolation test be written before
any MATLAB dependency exists. Waves are sequential (each is a tier of the V5 ladder); there
is no phase-internal parallelism across waves, but Wave 1's four modules can be built
concurrently.

---

## Per-file implementation spec

### `src/pyro_dcm/utils/local_linearization.py` (NEW, CMC-03) — laptop
```python
def integrate_local_linearization(
    f: Callable[[Tensor, Tensor], Tensor],   # f(v_flat:(8n,), u:(n_inp,)) -> (8n,)
    x0: Tensor,                                # (8n,) f64, the frozen expansion point
    inputs: Tensor,                            # (ns, n_inp) f64, from erp_input
    dt: float,
    n_substeps: int = 1,                       # spm_int_L N; default 1
    delay_operator: Tensor | None = None,      # D; None -> identity (Phase 33)
    g: Callable | None = None,                 # output map; None -> identity (states)
) -> Tensor:                                   # (ns, 8n) f64 trajectory
```
Body = Fact 3 verbatim. Frozen Jacobian via `torch.func.jacrev(lambda v: f(v, inputs[0]))`
at `x0` with `u0 = zeros_like(inputs[0])` (Jacobian taken at u=0 per Fact 5). Hard-assert
`x0.dtype == torch.float64` and `f(x0, u0)` is finite. `Q` via
`torch.linalg.solve(J.T, (E - I).T).T`. NO eigenvalue clipping (CMC-07, pitfall N2). Keep
the module CMC-agnostic (takes `f`, not CMC internals) — it lives in `utils/` next to
`ode_integrator.py`, which it does NOT touch.

### `src/pyro_dcm/forward_models/cmc_neural_mass.py` (NEW, CMC-01/02) — laptop
```python
J_PERM = (6, 1, 2, 3, 0, 4, 5, 7, 8, 9)  # 0-indexed of MATLAB [7 2 3 4 1 5 6 8 9 10]
G0 = torch.tensor([4,4,8,4,4,2,4,4,2,1], dtype=f64) * 200
T0_MS = torch.tensor([2,2,16,28], dtype=f64)
E0 = torch.tensor([1,0.5,1,0.5], dtype=f64) * 200

def parameterize_cmc(P: dict, n: int) -> dict:
    """G:(n,10) after perm-j exp-scaling; T:(n,4) seconds; C=exp(P.C); A{1..4} or zeros.
    NOT parameterize_A — CMC uses +exp(P.A)*E(i), structural signs (spm_fx_cmc.m:69-72)."""

def cmc_sigmoid(x, P_S):  # R=(2/3)*exp(P_S); 1/(1+exp(-R*x)) - 1/2   (spm_fx_cmc.m:90-94)

def cmc_f(x_flat, u, P, n=1):  # -> (8n,) ; Fact 1 eqs; column-major flatten
```
Signatures consumed later by `ERPDCMForward` (Phase 35); Phase 33 only needs `cmc_f` +
`parameterize_cmc` standalone. Cite `spm_fx_cmc.m` line numbers per CLAUDE.md rule 2.
Docstrings cite REF-ERP-001 (David & Friston 2003) ONLY after Zotero confirms the key — do
NOT fabricate (see Open Questions).

### `src/pyro_dcm/forward_models/cmc_priors.py` (NEW, CMC-04) — laptop
Transcribe Fact 6 as data tables (dicts of name → (mean, var)). Provide
`cmc_prior_moments(a_mask, c_mask, n)` returning `(E, V)` dicts and a
`cmc_steady_state(n)` returning `zeros(n,8)` with an assertion + `spm_dcm_neural_x.m:70-72`
citation. Feeds a Phase-35 `build_prior_cov`; Phase 33 uses it only for fixture `P` and the
`x0==0` guard.

### `src/pyro_dcm/forward_models/erp_input.py` (NEW, CMC-05) — laptop
```python
def erp_gaussian_input(
    t_s: Tensor,           # (ns,) peristimulus time in SECONDS
    P_R: Tensor,           # (n_inp, 2 or 3)
    ons_ms, dur_ms, sus,   # M.ons (def 60), M.dur (def 16/32 — see N3), M.sus (def 0)
) -> Tensor:               # (ns, n_inp), 32-scaled
```
Body = Fact 5 verbatim, `t` converted to ms internally (`spm_erp_u.m:46`). Keep the
sustained-mix term even though default `sus=0` (pitfall N4).

### `src/pyro_dcm/validation/export_to_mat.py` (APPEND, CMC-06) — laptop
```python
def export_erp_dcm(P, M_meta, output_path):
    """Single-source CMC-erp .mat: P struct (A{} empty/zeros, C, G, T, S, R; NO D),
    M.ns, U.dt, M.ons, M.dur, x0=zeros(1,8). Scalars as np.array([[v]]); cell arrays
    for A{}; metadata header. Mirror export conventions in this file's docstring."""
```
Append only; existing exporters untouched (backward-compat B4-adjacent).

### `validation/matlab_scripts/run_spm_erp_dcm.m` (NEW, CMC-06) — M3
Mirror `run_spm_spectral_dcm_csd_injected.m` scaffolding: `getenv('SPM12_PATH')` with local
fallback, `addpath`, loud-failure `try/catch`, `getenv('DCM_INPUT_PATH'/'DCM_OUTPUT_PATH')`.
Then:
```matlab
spm('defaults','EEG');
load(input_path,'DCM'); P = DCM.P; M = DCM.M;   % M.f set to wrapper below
M.f = 'spm_fx_cmc_nodelay';                      % FORCE D=I (Fact 4)
M.x = zeros(1,8); M.n = 8; M.m = size(DCM.U.u,2);
x0 = spm_vec(M.x); u0 = sparse(M.m,1);
% (1) f-field at a FROZEN nonzero (x_test,u_test) for the per-transform check
f_field = spm_fx_cmc(x_test, u_test, P, M);
% (2) frozen Jacobian at x0,u0
J0 = full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, P, M, 1)));   % (8,8)
% (3) update operator (D=I,N=1), replicating spm_int_L:126-127
dfdx = J0 - eye(8)*exp(-16);
dtJ  = DCM.U.dt * dfdx;             % export this for matrix_exp measurement
Eexp = spm_expm(dtJ);              % export
Q_update = (Eexp - eye(8)) / dfdx;  % export
% (4) full trajectory via SPM's own integrator with D=I
y_states = spm_int_L(P, M, DCM.U);  % (ns,8)  (nargout(M.f)==2 -> D=1)
save(output_path,'f_field','J0','dtJ','Eexp','Q_update','y_states','meta');
```
Also write the `spm_fx_cmc_nodelay.m` helper (Fact 4). Assert in-script:
`assert(isequal(M.x, zeros(1,8)))` and document `D=1` in `meta`.

### `cluster/scripts/erp_cross_validation.py` + `cluster/sbatch/erp_cross_validation.sbatch` (NEW) — M3
Mirror `spm_cross_validation.py` / `.sbatch` exactly: `sys.path` insert, env knobs,
record-don't-crash (exit 0 on recorded miss, non-zero only on unexpected exception),
`--partition=comp`, `--mem=16G`, `export MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab`,
`export SPM12_PATH=/home/aman0087/fc37/Carrick/spm12`. Invoke MATLAB via
`subprocess.run([MATLAB_PATH, "-batch", matlab_cmd])` where
`matlab_cmd = "cd('...matlab_scripts'); setenv('DCM_INPUT_PATH','...'); setenv('DCM_OUTPUT_PATH','...'); run_spm_erp_dcm"`
(pattern from `validation/run_vl_validation.py:193-209`). `NEVER pip install in the array
job` (project rule — though this is a single task, keep the no-install discipline).

### Test files
- `tests/test_cmc_forward.py` (laptop) — permutation guard, sigmoid `-1/2`, units
  (`T/1000`), float64, no-eig-clip, `x0==0`, column-major flatten round-trip.
- `tests/test_local_linearization.py` (laptop + the MEASURED tier) — right-division
  orientation (asymmetric J test), regulariser-before-Q, dtype; the
  `torch.matrix_exp(dtJ)` vs exported `Eexp` measurement.
- `tests/test_spm_erp_dcm_validation.py` (SPM-gated) — the V5 ladder. `pytestmark =
  [pytest.mark.spm, pytest.mark.slow, pytest.mark.skipif(not check_matlab_available(), ...)]`
  exactly like `tests/test_vl_spm_cross_validation.py:33-40`.

---

## The parity gate as a concrete test spec (CMC-06)

Fixture arrays in `erp_single_source_fixtures.mat`:

| Array | Shape | What it pins | Torch tolerance |
|-------|-------|--------------|-----------------|
| `f_field` | (8,) | `cmc_f` at frozen nonzero `(x_test,u_test)` — isolates every transform/sigmoid/perm before the integrator | ≤ 1e-10 |
| `J0` | (8,8) | frozen Jacobian at x0=0,u0=0 | ≤ 1e-10 |
| `dtJ`, `Eexp` | (8,8) | `dt*(J0 - I*exp(-16))` and `spm_expm(dtJ)` — the matrix_exp MEASUREMENT | MEASURED (record floor; expect ~1e-12, do NOT assume) |
| `Q_update` | (8,8) | `(Eexp - I)*inv(dfdx)` right-division | ≤ 1e-9 |
| `y_states` | (ns,8) | full `spm_int_L` trajectory, D=I, known Gaussian `u` | ≤ 1e-8 |

Staged ladder (assert IN THIS ORDER — V5; a failure localizes to one stage):
`f_field` → `J0` → `matrix_exp(dtJ)` vs `Eexp` (measure) → `Q_update` → `y_states`.
If `J0` and `Q_update` match but `y_states` fails, the bug is loop ordering
(`v += Q@f` vs `v = Q@(v+f)`), not algebra (pitfall C1 prevention note).

Mandatory asserts baked into the gate:
- **D=1 + x0==0:** the test reads `meta.D == 1` and asserts the torch `x0` is exactly
  zeros(8) before integrating.
- **Permutation guard** (also in `test_cmc_forward.py`, laptop, no MATLAB needed): perturb
  `P_G[:,0]`, assert `parameterize_cmc` changes `G[:,6]` and not `G[:,0]`.
- **float64** at the integrator boundary (CMC-07).
- **matrix_exp floor is MEASURED**, not assumed — record `max|matrix_exp(dtJ)-Eexp|` in
  the test output; set `Q_update`/`y_states` thresholds as a small multiple of it (V3).
- **Tolerances are element-wise forward agreement** (no absolute-F, no `Cp` — V2; the
  forward model has no normalization freedom, so this is the strong defensible gate).

---

## Compute routing

| Work | Where | Why |
|------|-------|-----|
| Wave 1 modules + `test_cmc_forward.py` + non-MATLAB `test_local_linearization.py` | **Laptop** | pure-torch, sub-second unit tests, <<3 min |
| Wave 2 MATLAB fixture generation (`run_spm_erp_dcm.m`) | **M3** | local MATLAB license server unreachable (FlexLM -15); R2022a + Carrick spm12 verified on `comp` partition; submit via `cluster/sbatch/erp_cross_validation.sbatch` |
| Wave 3 `test_spm_erp_dcm_validation.py` | **Laptop (auto-skips)** + real run on **M3** | `@pytest.mark.spm` skipif-MATLAB-unavailable; the real assertion run rides the M3 job, mirroring Phase 32 |

ssh-agent must be unlocked at submit time to reach M3 (per Phase-32 workflow). Use Mutagen
sync for code deploy/results, NOT git push/pull (project rule). The fixture `.mat` is small
(8×8 + ns×8); the job is minutes, well under the sbatch `--time=01:00:00`.

---

## Open questions / decisions for the planner

1. **`x_test` / `u_test` for the `f_field` check.** Pick a fixed, reproducible nonzero
   state+input (e.g. `x_test = 0.1*ones(1,8)`, `u_test` = peak Gaussian value) and freeze it
   in `meta`. Recommendation: any fixed nonzero point works — it just exercises the sigmoid
   and all G/T transforms off the trivial zero. Lock it in the export so torch and MATLAB
   evaluate the SAME point.
2. **`M.dur` default = 16 vs 32 ms.** `spm_erp_u.m:29` defaults `M.dur=32` only when the
   length mismatches `M.ons`; the standard DCM-erp default passed by `spm_dcm_erp` is 16 ms.
   Decision: set `M.dur` EXPLICITLY in `export_erp_dcm` (don't rely on the fallback) and
   record it in `meta` (pitfall N3/V4). Recommend 16 ms.
3. **Jacobian method in torch (autograd vs finite-diff).** Either matches the fixture `J0`.
   `torch.func.jacrev` is exact and differentiable (helps Phase 35 VL). Recommend jacrev;
   the fixture pins correctness regardless.
4. **`spm_diff` for J0 in MATLAB.** Use
   `full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, P, M, 1)))` (arg index 1 = differentiate
   w.r.t. x), exactly as `spm_fx_cmc.m:208` does internally. The `spm_cat` is needed because
   `spm_diff` returns a cell for multi-block states.
5. **MNI coordinates / lead field / `P.J` — NOT needed in Phase 33.** No `dipfit`, no
   `spm_erp_L`, no scalp projection. Single-source LFP-of-states only. Defer all of Fact-2's
   `P.J` lead-field index to Phase 35.
6. **Zotero citations.** REF-ERP-001 (David & Friston 2003) etc. are flagged "verify" in the
   milestone research — do NOT add `\cite{}` keys or `.planning/REFERENCES.md` entries until
   the user confirms the paper is in Zotero (CLAUDE.md `.bib` rule). Docstrings may cite the
   SPM source file + line (allowed) but not a fabricated bib key.

---

## Sources

### Primary (HIGH — SPM12 source read line-by-line this session at `C:/Users/aman0087/Documents/Github/spm12`)
- `toolbox/dcm_meeg/spm_fx_cmc.m` (`$Id: 7279`) — state eqs (`:171-198`), sigmoid
  (`:90-94`), input (`:86,107`), perm `j` (`:151`), G map (`:126-135`), n>1 A branch
  (`:68-82`), modulatory M (`:158-160`), Jacobian (`:206-208`), delay call (`:226`).
- `toolbox/dcm_meeg/spm_cmc_priors.m` (`$Id: 7279`) — priors/variances (`:80-133`).
- `spm_int_L.m` (`$Id: 7143`) — exp-Euler loop (`:112-169`), N default (`:70`),
  regulariser + Q (`:126-127`), nargout/D branch (`:114-122`).
- `spm_expm.m` (`$Id: 5691`) — degree-6 Padé + scaling-and-squaring (`:38-63`).
- `toolbox/dcm_meeg/spm_erp_u.m` (`$Id: 7679`) — Gaussian bump (`:42-64`), ms timebase
  (`:46`), 32-scale (`:63`).
- `toolbox/dcm_meeg/spm_gen_erp.m` (`$Id: 6427`) — N=1 / dt=0.004 / ns=128 (`:28-30`), M.u
  removal (`:48-54`), steady state + integrate (`:80-84`).
- `toolbox/dcm_meeg/spm_dcm_neural_x.m` (`$Id: 6112`) — CMC otherwise→zeros (`:70-76`).
- `spm_dcm_delay.m` (`$Id: 7279`) — D from P.D (`:55-109`), no-P.D→sparse(0) (`:107`),
  intrinsic delay default `di=1` (`:60-82`).
- `toolbox/dcm_meeg/spm_gen_Q.m` (`$Id: 7279`) — B on A + diag→G(:,1) (`:45-67`) [Phase 34].

### Primary (HIGH — repo source read this session)
- `src/pyro_dcm/inference/forward_models.py:30-117` — ForwardModel Protocol (8 members).
- `src/pyro_dcm/forward_models/neural_state.py:24-59` — `parameterize_A` `-exp/2` (the
  convention CMC must NOT reuse).
- `src/pyro_dcm/utils/ode_integrator.py:1-70` — torchdiffeq wrapper (confirms a NEW sibling
  module, not an edit).
- `validation/export_to_mat.py:1-90` — `.mat` export conventions to mirror.
- `validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m` — env-var + loud-failure
  MATLAB script pattern.
- `validation/run_vl_validation.py:183-219` — `subprocess.run([MATLAB,"-batch",cmd])` +
  `setenv` DCM_INPUT/OUTPUT pattern.
- `cluster/scripts/spm_cross_validation.py:1-120` + `cluster/sbatch/spm_cross_validation.sbatch`
  — M3 entrypoint + `MATLAB_PATH`/`SPM12_PATH` exports + record-don't-crash.
- `tests/test_vl_spm_cross_validation.py:1-40` — `@pytest.mark.spm/slow` + skipif pattern.
- `pyproject.toml:85-93` — markers `slow`, `spm`, `vl`.
- `.planning/REQUIREMENTS.md:357-378` — CMC-01..07.

### Peer milestone research (consumed, not re-derived)
- `.planning/research/v0.8.0/{SUMMARY,STACK,PITFALLS,ARCHITECTURE}.md` — milestone context;
  C1–C5 pitfalls; additive seam; file list.

## Metadata
**Confidence breakdown:**
- CMC equations / integrator / priors / input: HIGH — transcribed from SPM12 source with
  line refs this session.
- D=1 forcing mechanism: HIGH — traced through `spm_int_L.m:114-122` nargout branch +
  `spm_dcm_delay.m` no-P.D path.
- matrix_exp vs spm_expm tolerance: MEDIUM — same algorithm family (HIGH); exact floor MUST
  be measured by `test_local_linearization.py` (the one explicit gap).
- Repo seams / bridge / markers: HIGH — read directly.

**Research date:** 2026-06-25
**Valid until:** stable (SPM12 frozen at `$Id 7279`; repo conventions stable) — ~30 days.

## RESEARCH COMPLETE
