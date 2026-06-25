# Technology Stack — v0.8.0 DCM for Evoked Responses (EEG/MEG ERP, CMC)

**Milestone:** v0.8.0 — add a time-domain CMC neural-mass → evoked → single-dipole
lead-field → scalp-ERP forward stack, validated against SPM12.
**Researched:** 2026-06-25
**Mode:** Ecosystem / feasibility (STACK dimension only)
**Overall confidence:** HIGH (SPM12 source read line-by-line; torch primitives verified
against the installed runtime, not training data)

---

## Executive Summary

**Zero new runtime dependencies.** The entire ERP forward stack (CMC state equations,
Gaussian-bump evoked input, exponential-Euler integration, single-dipole lead field,
scalp projection) is pure PyTorch over the existing stack. It needs exactly two linear-
algebra primitives beyond what the fMRI/spectral path already uses — `torch.matrix_exp`
and `torch.linalg.solve` — **both confirmed present in the installed `torch 2.9.1`**
(verified live, see below). Pyro, torchdiffeq, Zuko, scipy are untouched; the existing
Variational Laplace engine and `ForwardModel` protocol are reused verbatim.

**One non-obvious, load-bearing decision:** SPM12 does **not** integrate ERPs with a
Runge–Kutta solver. `spm_gen_erp.m` calls `spm_int_L.m`, an **exponential-Euler /
local-linearisation** scheme with a **frozen Jacobian** evaluated once at the steady
state. torchdiffeq's `rk4`/`dopri5` (what `TaskDCMForward` currently uses) will **not**
reproduce this and will fail bit-for-bit parity. For SPM parity we must port `spm_int_L`
directly in torch (`matrix_exp` + `solve`), **not** call `integrate_ode`. This is the
single most important finding for the Phase-33 implementer.

**MNE-Python is not needed this milestone.** The single-equivalent-dipole forward needs a
physical gain matrix only when projecting to a real sensor montage with a real head
model. For forward + SPM-parity + synthetic scope, use the `LFP` spatial model (trivial
diagonal gain, exact SPM parity, no head model) for first parity tests, and for `ECD`
parity export the precomputed per-source gain from SPM/MATLAB and reproduce
`kron(J, L)` in torch. MNE is required only to *compute* an ECD lead field from scratch
in Python — that is real-data fitting territory, explicitly out of scope.

---

## 1. New Dependencies — None (Justified)

| Need | Existing primitive | Status |
|------|--------------------|--------|
| CMC state equations (sigmoid, weighted sums, 2nd-order synaptic ODE) | `torch` elementwise + matmul | already available |
| Steady state for CMC | zeros — **no solve needed** (see §3) | trivial |
| Frozen Jacobian `df/dx` at x0 | `torch.func.jacrev` (functorch, in torch 2.x) or analytic | already available |
| Update operator `Q = (expm(dt·J) − I)·inv(J)` | `torch.matrix_exp`, `torch.linalg.solve` | **verified in torch 2.9.1** |
| Gaussian-bump evoked input `u(t)` | `torch.exp`, `cumsum` | already available |
| Single-dipole lead field `kron(J, L)` | `torch.kron`, matmul | already available |
| VL inference | existing `variational_laplace.py` + `ForwardModel` protocol | reuse |

**Live verification (not training data):**

```
torch 2.9.1+cpu
torch.matrix_exp           -> True
torch.linalg.matrix_exp    -> True
torch.linalg.solve         -> True
torch.matrix_exp(randn(3,3,f64)) -> torch.Size([3, 3])   # works
```

`pyproject.toml` already pins `torch>=2.0`; `matrix_exp`/`linalg.matrix_exp`/
`linalg.solve` have been stable public API since well before 2.0, so the `>=2.0` floor is
sufficient — **no version bump required**. `mne>=1.6` already exists as an *optional*
extra (`[project.optional-dependencies].mne`) and stays optional/out-of-scope.

**Net: do not touch `pyproject.toml` dependencies for v0.8.0.**

> Note on `spm_expm` vs `torch.matrix_exp`: SPM's `spm_expm` and torch's `matrix_exp`
> both use scaling-and-squaring with a Padé approximant, so they agree to ~1e-12 on the
> same `dt·J`. This is what makes "bit-for-bit-close" parity achievable without a custom
> matrix-exponential. (MEDIUM confidence on the exact 1e-12 figure — algorithm family is
> HIGH confidence; the bound should be asserted by the actual parity test, not assumed.)

---

## 2. CMC Mathematics — Transcribed from SPM12 Source

> Source of truth: `../spm12/toolbox/dcm_meeg/spm_fx_cmc.m`
> (`$Id: spm_fx_cmc.m 7279 2018-03-10`). Priors: `spm_cmc_priors.m`. Lead field:
> `spm_lx_erp.m` + `spm_erp_L.m` + `spm_L_priors.m`. Input: `spm_erp_u.m`.
> **Important correction to the brief:** CMC priors come from **`spm_cmc_priors.m`**, not
> `spm_erp_priors.m`. `spm_dcm_neural_priors.m:150-154` dispatches `'cmc' ->
> spm_cmc_priors`. `spm_erp_priors.m` is the *ERP/Jansen-Rit* model and must NOT be used.

### 2.1 Population / state structure (`spm_fx_cmc.m:6-14`, `:39-42`)

Four populations per source, **8 states per source**, arranged as (voltage, current/
conductance) pairs. `x` has shape `(n_sources, 8)`:

| state col | symbol | population | quantity |
|-----------|--------|-----------|----------|
| `x(:,1)` | V_ss | spiny stellate (granular L4, excitatory interneurons) | voltage |
| `x(:,2)` | I_ss | spiny stellate | current/conductance |
| `x(:,3)` | V_sp | superficial pyramidal (supragranular L2/3) | voltage |
| `x(:,4)` | I_sp | superficial pyramidal | current/conductance |
| `x(:,5)` | V_ii | inhibitory interneurons | current (labelled "current") |
| `x(:,6)` | I_ii | inhibitory interneurons | conductance |
| `x(:,7)` | V_dp | deep pyramidal (infragranular L5/6) | voltage |
| `x(:,8)` | I_dp | deep pyramidal | conductance |

`spm_dcm_x_neural.m:70-78` sets `m = 8`, `x = sparse(n, 8)`.
**Flattening (parity-critical):** MATLAB `spm_vec(x)` is column-major, so the flat state
vector is **state-blocked**: `[V_ss(all src); I_ss(all src); V_sp(all src); …]`. The torch
port must replicate this ordering (operate per-column on an `(n, 8)` tensor, or flatten
Fortran-order) wherever it compares state vectors, Jacobians, or builds `kron(J, L)`.

### 2.2 Fixed (default) parameters (`spm_fx_cmc.m:47-49`)

```
E = [1 1/2 1 1/2]*200    % extrinsic rates: [fwd_i, bwd_i, fwd_ii, bwd_ii]
G = [4 4 8 4 4 2 4 4 2 1]*200   % 10 intrinsic connection strengths
T = [2 2 16 28]          % synaptic time constants (ms): [ss, sp, ii, dp]
```
`T` is converted to seconds: `T = ones(n,1)*T/1000` (`:114`).

### 2.3 Sigmoid firing-rate function (`spm_fx_cmc.m:88-94`)

```
R = 2/3;                       % baseline slope
B = 0;                         % bias (background)
R = R .* exp(P.S);             % gain/slope, modulated by free param P.S
F = 1 ./ (1 + exp(-R*x + B));  % firing rate, applied elementwise to all 8 states
S = F - 1/(1 + exp(B));        % deviation from baseline  ==  sigmoid(R*x) - 1/2
```
`S(:,m)` is the presynaptic firing of population state `m`. `P.S` is the sigmoid-gain
free parameter (one per source; prior `E.S=0, V.S=1/64`, `spm_cmc_priors.m:124`). **This
is one of the three "superficial-pyramidal gain = precision" knobs** the downstream
psychosis-MMN consumer needs (see §2.7).

### 2.4 Input (`spm_fx_cmc.m:84-109`)

```
C = exp(P.C);                  % :86  input weights, log-normal
% during ERP integration M.u is removed (spm_gen_erp.m:48-54) -> exogenous branch:
U = C*u(:)*32;                 % :107 driving input -> enters granular (ss) layer only
```
The driving `u(t)` is the Gaussian bump from `spm_erp_u.m` (already 32-scaled internally),
so total drive scaling is `C·u·32` with `u` itself `32·`Gaussian.

### 2.5 Intrinsic-connection indexing (`spm_fx_cmc.m:126-154`)

10 intrinsic connections `G(:,1..10)`; free log-scaling applied through a permutation
`j = [7 2 3 4 1 5 6 8 9 10]` (`:151`):

```
for i = 1:size(P.G,2),  G(:,j(i)) = G(:,j(i)) .* exp(P.G(:,i));  end
```
`spm_cmc_priors.m:120-122` sets `m = 4`, `E.G = sparse(n,4)` — so **only 4 intrinsic
gains are free** (P.G columns 1-4), mapping to:

| P.G col | -> G index | connection | sign |
|---------|-----------|-----------|------|
| 1 | G(:,7) | sp -> sp (superficial pyramidal **self-inhibition**) | −ve |
| 2 | G(:,2) | sp -> ss | −ve |
| 3 | G(:,3) | ii -> ss | −ve |
| 4 | G(:,4) | ii -> ii self | −ve |

Time constants similarly: `T(:,i) = T(:,i).*exp(P.T(:,i))` for the 4 free `P.T`
(`:148-150`; `E.T=sparse(1,4), V.T=1/32`).

### 2.6 Extrinsic connections + lateral reduction (`spm_fx_cmc.m:68-82`)

```
A{1} = exp(P.A{1})*E(1);   % forward  sp -> ss   (*200)
A{2} = exp(P.A{2})*E(2);   % forward  sp -> dp   (*100)
A{3} = exp(P.A{3})*E(3);   % backward dp -> sp   (*200)
A{4} = exp(P.A{4})*E(4);   % backward dp -> ii   (*100)
% reciprocal (lateral) connections are halved-ish:
for i, L = (A{i}>exp(-8)) & (A{i}'>exp(-8));  A{i} = A{i}./(1 + 4*L);  end
```
`spm_cmc_priors.m:62-83`: `D{1}=D{2}=A{1}` (forward), `D{3}=D{4}=A{2}` (backward); priors
`E.A{i}=mask*32-32`, `V.A{i}=mask/16`. Absent connections -> `exp(-32) ≈ 0`.

### 2.7 Equations of motion (`spm_fx_cmc.m:163-198`) — the 2nd-order/convolution form

Each population is a critically-damped 2nd-order synaptic kernel written as a coupled
pair of 1st-order ODEs. With `Gk = G(:,k)`, `Sm = S(:,m)`, `Tk = T(:,k)`:

```
% Granular — spiny stellate (ss), driven by forward sp->ss and exogenous U
u      =  A{1}*S(:,3) + U;
u      = -G1.*S(:,1) - G3.*S(:,5) - G2.*S(:,3) + u;
f(:,2) = (u - 2*x(:,2) - x(:,1)./T1) ./ T1;        % :171-173

% Supragranular — superficial pyramidal (sp), receives backward dp->sp (−)
u      = -A{3}*S(:,7);
u      =  G8.*S(:,1) - G7.*S(:,3) + u;
f(:,4) = (u - 2*x(:,4) - x(:,3)./T2) ./ T2;        % :177-179

% Supragranular — inhibitory interneurons (ii), receives backward dp->ii (−)
u      = -A{4}*S(:,7);
u      =  G5.*S(:,1) + G6.*S(:,7) - G4.*S(:,5) + u;
f(:,6) = (u - 2*x(:,6) - x(:,5)./T3) ./ T3;        % :183-185

% Infragranular — deep pyramidal (dp), receives forward sp->dp (+)
u      =  A{2}*S(:,3);
u      = -G10.*S(:,7) - G9.*S(:,5) + u;
f(:,8) = (u - 2*x(:,8) - x(:,7)./T4) ./ T4;        % :189-191

% Voltage states are the integrals of the current states:
f(:,1) = x(:,2);  f(:,3) = x(:,4);  f(:,5) = x(:,6);  f(:,7) = x(:,8);   % :195-198
```

**The convolution kernel.** `dV/dt = I`, `dI/dt = (u − 2I − V/T)/T`. The author's own
notes (`:231-238`) show this realises `x = t·exp(k·t)` with `x'' = 2k·x' − k²·x`,
i.e. the **alpha-function synaptic kernel** with rate `k = −1/T`. This is the David &
Friston (2003) convolution-based neural-mass form. It is **linear in states given fixed
input** except for (a) the sigmoid `S(V)` and (b) the optional modulatory gain below.

**Modulatory gain (`spm_fx_cmc.m:156-160`):**
```
if isfield(P,'M'),  G(:,7) = G(:,7) .* exp(-P.M*32*S(:,7));  end
```
Deep-pyramidal firing `S(:,7)` down-modulates superficial-pyramidal self-inhibition
`G(:,7)`. Together with `P.S` (sigmoid gain) and the free `P.G` col-1 (`G(:,7)` baseline),
**these three control superficial-pyramidal gain = precision** — the quantity the
psychosis-MMN consumer reads out. (`E.M=0, V.M=mask/32`, `spm_cmc_priors.m:71-72`.)

### 2.8 Parameter transforms — the log/exp convention (summary table)

| Param | Transform in `spm_fx_cmc` | Prior (E, V) in `spm_cmc_priors` |
|-------|---------------------------|----------------------------------|
| `P.A{i}` extrinsic | `exp(P.A{i})·E(i)` | `mask·32−32`, `mask/16` |
| `P.C` input | `exp(P.C)` | `mask·32−32`, `mask/32` |
| `P.T` time const | `T0·exp(P.T)` (4 free) | `0`, `1/32` |
| `P.G` intrinsic | `G0·exp(P.G)` (4 free, perm `j`) | `0`, `1/32` |
| `P.S` sigmoid gain | `(2/3)·exp(P.S)` | `0`, `1/64` |
| `P.M` modulatory | `G7·exp(−P.M·32·S7)` | `0`, `mask/32` |
| `P.D` delays | delay operator (opt.) | `0`, `Q/64` |
| `P.B` trial-effect on A | additive **pre-exp**: `Q.A{j} += X·P.B{i}` (`spm_gen_Q.m:47`) | `0`, `mask/8` |
| `P.R` stim onset/disp | see §2.9 | `0`, `1/16` |

**Sign-convention warning for the port:** CMC extrinsic connections use **`+exp(P.A)`**
(strictly positive rates) with directional signs applied *structurally* in the equations
(forward `+`, backward via `−A{3}`, `−A{4}`). Do **not** reuse the existing fMRI
`parameterize_A` (which negates the diagonal via `−exp(...)` for self-inhibition
stability) — that convention does not apply to CMC. Build a dedicated CMC parameteriser.

### 2.9 Evoked input — Gaussian bump (`spm_erp_u.m:42-64`)

```
delay  = M.ons(i) + 128*P.R(i,1);     % ms; M.ons default 60
scale  = M.dur(i) * exp(P.R(i,2));    % M.dur default 16 (ms sd)
U      = exp(-(t - delay).^2 / (2*scale^2));   % t in ms
U      = prop*cumsum(U)/sum(U) + U*(1-prop);   % optional sustained mix (prop=M.sus, def 0)
u(:,i) = 32*U;
```
Pure-torch port; `t` is peristimulus time in ms.

### 2.10 Lead field & scalp projection (`spm_lx_erp.m`, `spm_erp_L.m`, `spm_dcm_erp.m`)

Per-source lead field then per-state expansion:
```
L = spm_erp_L(P, dipfit);     % (Nc x n) per source
L = kron(P.J, L);             % (Nc x 8n) per state    spm_lx_erp.m:31-32
```
- **`P.J` (state -> dipole contribution), CMC default** (`spm_L_priors.m:106-109`):
  `pE.J = sparse(1,3,1,1,8)` -> only **state 3 (superficial-pyramidal voltage)** contributes.
  `pC.J = sparse(1,[1 7],1/32,1,8)` -> states 1 (ss V) and 7 (dp V) get free contribution.
  Physiologically correct: scalp EEG dominated by L2/3 pyramidal depolarisation.
- **ECD single dipole** (`spm_erp_L.m:43-77`): `L(:,i) = G(:,:,i)·P.L(:,i)`, where
  `G(:,:,i)` is the `(Nc x 3)` physical gain from `ft_compute_leadfield` (FieldTrip head
  model) and `P.L(:,i)` is the free 3-vector dipole moment (`E.L=0, V.L=64`,
  `spm_L_priors.m:64-67`). This is exactly "single-equivalent-dipole-per-source."
- **LFP (default, no head model)** (`spm_erp_L.m:105-112`): `L = diag(P.L)` — trivial
  electrode gain, no physics.
- **Scalp prediction** (`spm_dcm_erp.m:286-289`): `y = R·(x − x0)·L'·U`. For synthetic
  forward (no inference) use `R=I`, `U=I`, `x0=0`: `y = (x − x0) @ L.T`.

**Recommendation for this milestone (forward + parity + synthetic, no real montage):**
1. First parity tests: **`LFP` spatial** — `L = diag(exp/identity P.L)`, zero head-model
   dependence, exact SPM parity, isolates the CMC dynamics.
2. ECD parity: **precompute `G(:,:,i)` in SPM/MATLAB** (`spm_erp_L`/`ft_compute_leadfield`),
   export via the existing `validation/` `.mat` bridge, and reproduce only `kron(J, L)` +
   projection in torch. **No MNE, no FieldTrip in Python.**

MNE-Python would only be needed to *compute* an ECD lead field from a head model natively
in Python (real-data fitting) — out of scope. Keep it the optional extra it already is.

---

## 3. Integration Scheme for SPM Parity

### 3.1 What SPM actually does

`spm_gen_erp.m:78-84`:
```
M.x  = spm_dcm_neural_x(Q, M);   % steady state
y{c} = spm_int_L(Q, M, U);       % integrate
```
- **Steady state for CMC = zeros.** `spm_dcm_neural_x.m` only runs a Newton fixed-point
  solve for conductance models (`spm_fx_cmm`/`spm_fx_mfm`). CMC (`spm_fx_cmc`) hits the
  `otherwise` branch (`:70-72`) which does nothing -> `x0 = zeros(n, 8)`. **No solve to
  port.**
- **`spm_int_L.m` = exponential-Euler with a FROZEN Jacobian** (`:124-165`):
  ```
  dfdx = dfdx - I*exp(-16);                       % :126 regulariser
  Q    = (spm_expm(dt*D*dfdx/N) - I) / dfdx;      % :127  == (expm(dt·D·J) − I)·inv(J)
  for i = 1:ns                                    % per time bin
      u = U.u(i,:);
      for j = 1:N,  v = v + Q*f(v,u,P,M);  end    % :141-143
      y(:,i) = g(v,u,P,M);
  end
  ```
  `J = df/dx` is evaluated **once** at the expansion point `x0` (`:114-122`) and reused
  for every step. `D` is the delay operator from `spm_dcm_delay` (`=1` if delays absent).
  Defaults: `N = 1`, `dt = U.dt` (= data sampling interval; `spm_gen_erp` default
  `U.dt = 0.004 s`; in fitting `dt = xY.dt`).

This is a **local-linearisation / matrix-exponential** integrator, exact for the linear
part of the dynamics and first-order in the input. It is **not** RK4/dopri5.

### 3.2 Mapping to torch — DO NOT use torchdiffeq

The existing `integrate_ode(...method="rk4")` used by `TaskDCMForward`/`LatentCircuitForward`
**cannot** reproduce `spm_int_L` and will break parity. Implement `spm_int_L` directly as
a new small pure-torch integrator (e.g. `forward_models/erp_integrator.py` or
`utils/local_linearization.py`):

```python
# x0 = zeros(n, 8) for CMC; f, x0, dt, N, D from the CMC forward model
J  = jacobian(f, x0)                       # torch.func.jacrev, frozen at x0
J  = J - torch.eye(n) * exp(-16.0)         # SPM regulariser, spm_int_L.m:126
E  = torch.matrix_exp(dt * D @ J / N)      # verified API in torch 2.9.1
Q  = torch.linalg.solve(J.T, (E - I).T).T  # == (expm − I) @ inv(J), stable form
v  = x0.flatten()
for i in range(ns):
    u = U[i]
    for _ in range(N):
        v = v + Q @ f(v, u)
    y[i] = g(v, u)
```
Notes:
- `(E − I)/dfdx` in MATLAB is *right* division `= (E − I)·inv(J)`. Use
  `solve(J.T, (E−I).T).T` (or `(E−I) @ torch.linalg.inv(J)`) — never `torch.inverse`
  (per CLAUDE.md numerical-stability rule).
- `torch.matrix_exp` ≈ `spm_expm` (both scaling-and-squaring + Padé) -> parity to ~1e-12.
- **dt:** match the synthetic dataset's sampling interval exactly (use the same `xY.dt`
  the MATLAB side uses, e.g. 1–4 ms). **N = 1.** **D = 1** for the first delay-free parity
  pass; add `spm_dcm_delay` only once the delay-free forward matches.
- Keep the whole thing `float64` (matches the repo convention and SPM's `double`).
- This integrator is differentiable (autograd through `matrix_exp`/`solve`), so the
  existing VL engine's finite-difference *or* autograd Jacobian path both work; the
  `ForwardModel.predict` contract (return a flat vector matching `observed.reshape(-1)`)
  is satisfied unchanged.

### 3.3 New `ForwardModel` implementation

Add an `ERPDCMForward` class implementing the existing protocol
(`inference/forward_models.py`): `residual_is_complex = False` (time-domain real
residuals, like `TaskDCMForward`); `build_precision` returns a single identity `Q`
(SPM uses an AR(1) `spm_Q(1/2,Ns)` serial-correlation component — for closer parity the
precision basis can later carry that AR(1) matrix, but identity is the right v1). Parameter
packing must carry `A{1..4}`, `C`, `T`(4), `G`(4), `S`, optional `M`, plus spatial `L`,
`J`, and stimulus `R` — following the transforms in §2.8.

---

## 4. Citations to Add (flag for manual Zotero addition)

Per CLAUDE.md: **do not fabricate `.bib` keys.** The following papers must be added to the
project Zotero folder by the user; Better BibTeX will then export keys. Map each to a
`[REF-xxx]` in `.planning/REFERENCES.md` at implementation time.

| Suggested REF | Paper | Covers | In Zotero? |
|---------------|-------|--------|-----------|
| REF-ERP-001 | David O, Friston KJ (2003). *A neural mass model for MEG/EEG: coupling and neuronal dynamics.* NeuroImage 20:1743-1755 | convolution-based NMM, alpha kernel, 2nd-order ODE form (cited in every source file header) | **verify** |
| REF-ERP-002 | Bastos AM, Usrey WM, Adams RA, Mangun GR, Fries P, Friston KJ (2012). *Canonical microcircuits for predictive coding.* Neuron 76:695-711 | the canonical microcircuit, superficial-pyramidal gain = precision | **verify** |
| REF-ERP-003 | Moran RJ, Pinotsis DA, Friston KJ (2013). *Neural masses and fields in dynamic causal modeling.* Front. Comput. Neurosci. 7:57 | CMC parameterisation, DCM-for-ERP families, transforms | **verify** |
| REF-ERP-004 | Pinotsis DA, Moran RJ, Friston KJ (2012). *Dynamic causal modeling with neural fields.* NeuroImage 59:1261-1274 | CMC / neural-field lineage | **verify** |
| REF-ERP-005 | Kiebel SJ, David O, Friston KJ (2006). *Dynamic causal modelling of evoked responses in EEG/MEG with lead fields and spatial priors.* NeuroImage 30:1273-1284 | ECD lead field, spatial model (cited in `spm_erp_L.m:23`) | **verify** |
| REF-ERP-006 | Friston K, Mattout J, Trujillo-Barreto N, Ashburner J, Penny W (2007). *Variational free energy and the Laplace approximation.* NeuroImage 34:220-234 | VL / `spm_nlsi` (likely already in Zotero from v0.7.0) | **likely present** |
| REF-ERP-007 | Garrido MI, Kilner JM, Stephan KE, Friston KJ (2009). *The mindful brain: MMN as a model.* / Garrido et al. (2007) DCM-MMN | MMN application context for the downstream consumer | optional |

**SPM source files to cite alongside each equation** (per CLAUDE.md rule 2):
`spm_fx_cmc.m` (state eqs), `spm_cmc_priors.m` (priors/transforms), `spm_gen_erp.m` +
`spm_int_L.m` (integration), `spm_erp_u.m` (input), `spm_lx_erp.m` + `spm_erp_L.m` +
`spm_L_priors.m` (lead field), `spm_dcm_erp.m` (driver).

---

## 5. Open Questions / Flags for Roadmap

1. **AR(1) precision parity.** SPM uses `xY.Q = {spm_Q(1/2,Ns,1)}` (AR-1 serial
   correlation, `spm_dcm_erp.m:118`). v1 can use identity precision; flag a later phase to
   add the AR(1) component if residual-level parity demands it. (LOW risk for forward
   parity; matters for F/evidence parity.)
2. **Delay operator `spm_dcm_delay`.** Deferred (D=1) for first parity. If SPM runs with
   non-trivial extrinsic delays, the delay-absorbed Jacobian must be ported. Flag as a
   sub-task gated on the delay-free forward matching.
3. **Channel modes `M.U` / feature selection `spm_fy_erp`.** Relevant to *inference*
   parity (data projection), not forward generation. Synthetic-only scope can set `U=I`;
   flag for the empirical-data milestone (out of scope here).
4. **`torch.matrix_exp` vs `spm_expm` exact tolerance** is asserted as ~1e-12 from shared
   algorithm family (MEDIUM confidence) — the parity test should *measure* it rather than
   assume it.

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Zero new deps | HIGH | torch 2.9.1 primitives verified live |
| CMC state/kernel/sigmoid math | HIGH | `spm_fx_cmc.m` read line-by-line, line refs given |
| Parameter transforms/priors | HIGH | `spm_cmc_priors.m` + `spm_L_priors.m` transcribed |
| Integration scheme (spm_int_L) | HIGH | `spm_int_L.m` + `spm_gen_erp.m` read; torchdiffeq-unsuitability is structural |
| MNE not needed | HIGH | lead-field path traced; only `ft_compute_leadfield` (ECD) needs a head model, exportable from SPM |
| matrix_exp↔spm_expm exact tolerance | MEDIUM | same algorithm family; should be measured by the parity test |
| Citation keys | N/A | flagged for manual Zotero addition; never fabricated |
