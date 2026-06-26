# Phase 34: Extrinsic Coupling, Condition B & Multi-Source Evoked Integration — Research

**Researched:** 2026-06-26
**Domain:** Hierarchical CMC network — extrinsic forward/backward/lateral coupling
(`spm_fx_cmc.m` n>1 branch), condition-specific `B` modulation incl. the
`diag(B)→G(:,1)→G[:,6]` precision path (`spm_gen_Q.m`), and C-driven evoked
integration over the peristimulus window (`spm_gen_erp.m`), parity vs frozen
multi-source MATLAB fixtures (delays off, D=1).
**Confidence:** HIGH (every equation below transcribed from the actual SPM12 source at
`C:/Users/aman0087/Documents/Github/spm12/toolbox/dcm_meeg/` and the shipped Phase-33
torch code, read line-by-line this session).

> This file is the **Phase-34 implementation concretes**. It does NOT re-survey the
> milestone — read `.planning/research/v0.8.0/{SUMMARY,STACK,PITFALLS,ARCHITECTURE}.md`
> for the why and `.planning/phases/33-.../33-RESEARCH.md` + the three `33-0{1,2,3}-SUMMARY.md`
> for the proven single-source baseline this phase composes. It gives the *exact*
> state-coupling map (with `spm_fx_cmc.m` line refs), the *exact* `spm_gen_Q` dual
> mechanism, the *exact* repo seams to extend additively, and the multi-source fixture
> + parity-ladder spec, so the planner can cut 3 executable plans with no further source
> reading.

---

## Summary

Phase 34 lifts the Phase-33 single-source CMC forward to an **N-source hierarchical
network**. Three things change relative to Phase 33, and *only* three: (1) the four
extrinsic blocks `A{1..4}` become non-zero `(N,N)` matrices that couple sources via four
specific population-to-population routes (`spm_fx_cmc.m:68-82,171-198`); (2) a
condition-specific modulation `B` is folded into the **free (log-space) parameters**
*before* exponentiation, hitting both all four `A{j}` *and* the superficial-pyramidal
self-inhibition precision knob via `diag(B)→Q.G(:,1)→G[:,6]` (`spm_gen_Q.m:41-67`); and
(3) the evoked generator loops conditions, re-deriving per-condition parameters and
re-integrating with the **already-shipped Phase-33 exp-Euler integrator** at network
scale (`spm_gen_erp.m:69-86`). Everything else — the integrator, the sigmoid, the
intrinsic EOM, the Gaussian input, the column-major flatten, the `D=1` nargout-aware
wrapper — is reused verbatim from Phase 33, which proved them bit-identical to SPM12.

The single most important inherited fact (Phase 33-03 decision D2, carried forward
explicitly): **SPM's `spm_int_L` freezes its Jacobian via `spm_diff` forward differences
(`dx=exp(-8)`), NOT exact autodiff.** The shipped `integrate_local_linearization` uses
exact `torch.func.jacrev`, which is *more accurate* than SPM and at single-source level
left a measured 4.7e-8 floor on `y_states` (above the 1e-8 bit-close gate). At network
scale (8N×8N Jacobian) that FD-truncation floor will be **larger**. Therefore the
Phase-34 multi-source trajectory gate must, exactly as Phase 33-03 did, be asserted on a
**scheme rung** (drive the loop with SPM's own frozen `Q_update`, isolates loop ordering,
bit-exact ~1e-13) and an **FD-Jacobian rung** (replicate `spm_diff` `dx=exp(-8)` to build
the operator, ≤1e-8), with the shipped-`jacrev` path recorded as a *measured* floor — not
gated at 1e-8. Do not attempt to gate the shipped exact-AD integrator at ≤1e-8 against an
SPM FD-Jacobian fixture; that is a known, documented numerical-method divergence, not a
bug.

**Primary recommendation:** Build in strict wave order. Wave 1 (laptop, pure-torch,
sub-second tests): the network forward `cmc_network_f` (the 4 extrinsic terms added to the
Phase-33 EOM), the `spm_gen_Q` port `apply_condition_modulation`, the extrinsic
parameteriser (A-block `exp*E` + lateral `(1+4L)` reduction), `erp_simulator`, and the
**C4 `diag(B)→G[:,6]` guard test** plus the `cmc_network_f(n=1)==cmc_f` bit-exact guard.
Wave 2 (M3, MATLAB): the multi-source fixtures — per-condition `spm_gen_Q` `Q.A{1..4}` +
`Q.G(:,1)`, the per-condition frozen `Q_update`, and the `spm_gen_erp` trajectory — all
with `M.f='spm_fx_cmc_nodelay'` (D=1) and `M.x==zeros(N,8)`. Wave 3 (laptop, against the
committed `.mat`): the staged parity ladder (`spm_gen_Q` algebra → network `J0` → network
`Q_update` → multi-source trajectory).

---

## Source-resolved facts (the things Phase 34 must get exactly right)

### Fact 1 — Extrinsic topology: the exact state-coupling map (`spm_fx_cmc.m:68-82,171-198`)

The `n > 1` branch (`spm_fx_cmc.m:68-72`) builds four blocks from the free log-params and
the fixed extrinsic rates `E0 = [1, 0.5, 1, 0.5]*200` (`:47`):

```
A{1} = exp(P.A{1})*E(1)   % E(1)=200  forward   sp -> ss
A{2} = exp(P.A{2})*E(2)   % E(2)=100  forward   sp -> dp
A{3} = exp(P.A{3})*E(3)   % E(3)=200  backward  dp -> sp
A{4} = exp(P.A{4})*E(4)   % E(4)=100  backward  dp -> ii
```

**Lateral / reciprocal reduction** (`:79-82`), applied AFTER `exp*E`, INSIDE `spm_fx_cmc`
(so it also applies to the per-condition `Q.A`):

```
for i = 1:4
    L    = (A{i} > exp(-8)) & (A{i}' > exp(-8));   % (N,N) element-wise boolean
    A{i} = A{i} ./ (1 + 4*L);                       % halve-ish reciprocal pairs
end
```

`L` is an `(N,N)` boolean (a connection is "reciprocal" iff both `A{i}[r,c]` and its
transpose entry exceed `exp(-8)`); the division is element-wise. At `N=1` the scalars are
0, `0>exp(-8)` is false → no-op → the Phase-33 path is unchanged.

The four blocks enter the equations of motion at **specific (origin population firing →
target conductance row)** routes (`spm_fx_cmc.m:171,177,183,189`). Firing `S` is the
sigmoid output `(N,8)`; the matmuls use the **voltage** firing columns of the origin
population. 0-indexed for the torch port:

| Block | MATLAB term | origin firing | target `f` row | sign | Python (0-idx) |
|-------|-------------|---------------|----------------|------|----------------|
| `A{1}` fwd sp→ss | `+A{1}*S(:,3)` into `f(:,2)` | sp voltage `S(:,3)` | ss conductance | `+` | `f[ss_I] += A1 @ S[:,2]` |
| `A{2}` fwd sp→dp | `+A{2}*S(:,3)` into `f(:,8)` | sp voltage `S(:,3)` | dp conductance | `+` | `f[dp_I] += A2 @ S[:,2]` |
| `A{3}` bwd dp→sp | `-A{3}*S(:,7)` into `f(:,4)` | dp voltage `S(:,7)` | sp conductance | `−` | `f[sp_I] -= A3 @ S[:,6]` |
| `A{4}` bwd dp→ii | `-A{4}*S(:,7)` into `f(:,6)` | dp voltage `S(:,7)` | ii conductance | `−` | `f[ii_I] -= A4 @ S[:,6]` |

(0-indexed conductance rows: ss_I=1, sp_I=3, ii_I=5, dp_I=7; firing cols: sp-V=2, dp-V=6.)
**Forward originates from superficial-pyramidal voltage firing `S[:,2]`; backward from deep
pyramidal voltage firing `S[:,6]`.** Each `A{i}` is `(N,N)`, each `S[:,col]` is `(N,)`, the
matmul yields `(N,)`. These four terms are the ONLY additions to the Phase-33
`cmc_f` body; everything intrinsic (the `-G*S` sums, the `(u - 2x - x/T)/T` kernel, the
voltage integrals) is bit-identical.

**Input `C` enters spiny-stellate only** (`:86,107,171`): `U = exp(P.C)*u*32` adds to
`f(:,2)` (ss) only — already correct in the shipped `cmc_f` (`big_u` enters `f1`). For
`N>1`, `C` is `(N, n_inp)`, `u` is `(n_inp,)`, `big_u = (C @ u)*32` is `(N,)`. Sources with
no input have `P.C` ≈ `mask*32-32` so `exp(P.C) ≈ 0` (handled by the C mask in the fixture).

### Fact 2 — `spm_gen_Q`: the dual `B` mechanism (`spm_gen_Q.m:24-81`) — pitfall C4 / EVOK-02

`spm_gen_Q(P, X)` builds the condition-specific parameter struct `Q` from the free params
`P` (which carries a list `P.B` of `(N,N)` between-trial-effect matrices) and a design row
`X` (length = number of between-trial effects). It operates **entirely in free/log space,
BEFORE `parameterize_cmc` exponentiates**:

```
Q = rmfield(P,'B')                                  % :26-30  drop B
try Q.C = Q.C(:,:,1) + X(1)*P.C(:,:,2); end          % :35-37  only if C has a 2nd page
for i = 1:length(X)
    for j = 1:length(Q.A)
        Q.A{j} = Q.A{j} + X(i)*P.B{i};               % :47  SAME B{i} added to ALL 4 A blocks
    end
    if isfield(P,'M'),  Q.M = Q.M + X(i)*P.N{i};  end % :59-61  modulatory (omit in ref net)
    if isfield(Q,'G'),  Q.G(:,1) = Q.G(:,1) + X(i)*diag(P.B{i});  end  % :65-67  PRECISION PATH
end
```

Two load-bearing details for the planner:

1. **`B` hits ALL FOUR `A{j}` identically** (`:45-47`): one `(N,N)` matrix `B{i}` is added
   to `Q.A{1}, Q.A{2}, Q.A{3}, Q.A{4}` — additive, in log space, pre-exp. (NMDA `AN`/`BN`
   and `int` branches `:49-54,71-79` do not apply to the plain CMC reference — `P.AN`/`int`
   are absent.)
2. **The diagonal of `B` modulates the precision knob** (`:65-67`): `Q.G(:,1)` is column 1
   of the **free** `P.G` (the 4-column free intrinsic param). `diag(P.B{i})` is the `(N,)`
   self-connection vector. After `parameterize_cmc`, `G(:,j(1)) = G(:,7)` (MATLAB) =
   `G[:,6]` (Python) `*= exp(Q.G[:,0])` — the sp→sp self-inhibition = superficial-pyramidal
   gain = **precision** (Bastos 2012 / Adams 2013). Omitting `diag(B)→Q.G[:,0]` destroys
   the MMN precision mechanism the whole milestone exists to demonstrate (EVOK-02 says this
   omission must be explicitly tested against).

   The C-effect line `:35-37` only fires when `P.C` has a 2nd page (`size(P.C,3)>=2`); for a
   single-page reference `C` the `try` silently skips. The torch port should mirror: apply
   it only if `C` has a 2nd condition slice, else skip.

**Order of operations (critical):** in `spm_gen_erp`, `spm_gen_Q` runs FIRST (free-space B
folding) → THEN `spm_int_L → spm_fx_cmc` exponentiates `exp(Q.A{j})*E(i)`, applies the
`(1+4L)` lateral reduction, and `G[:,6] *= exp(Q.G[:,0])`. So the torch pipeline per
condition is: `Q = apply_condition_modulation(P, X_c)` → `params = parameterize_cmc_network(Q, N)`
→ integrate. The precision modulation flows `diag(B) → Q.G[:,0] → exp → G[:,6]`.

### Fact 3 — `spm_gen_erp`: the evoked loop (`spm_gen_erp.m:69-86`)

```
U.u = spm_erp_u((1:M.ns)*U.dt, P, M)     % :44  the Gaussian bump grid (condition-INDEPENDENT)
for c = 1:size(X,1)                       % :72  one row of the design matrix X per condition
    Q    = spm_gen_Q(P, X(c,:));          % :76  Fact 2
    M.x  = spm_dcm_neural_x(Q, M);        % :80  steady state -> zeros(N,8) for CMC (M1)
    y{c} = spm_int_L(Q, M, U);            % :84  exp-Euler integrate -> (ns, 8N)
end
```

Maps onto the Phase-33 integrator at network scale with **zero integrator changes**:

```python
inputs = erp_gaussian_input(t_s, P_R, ...)          # (ns, n_inp), shared across conditions
x0 = cmc_steady_state(N).T.reshape(-1)               # zeros(8N,) column-major
traj = []
for X_c in X:                                        # X: (Cnd, n_effects)
    Q = apply_condition_modulation(P, X_c)           # free-space B folding (Fact 2)
    f_c = lambda v, u: cmc_network_f(v, u, Q, N)     # Q differs per condition
    traj.append(integrate_local_linearization(f_c, x0, inputs, dt))   # (ns, 8N)
```

Note: because `Q.A`/`Q.G` differ per condition, the **frozen Jacobian `J0` differs per
condition** — the integrator (which freezes `J` at `x0` internally) is called once per
condition, correctly re-freezing each time. The Gaussian input grid is condition-
independent (`spm_erp_u` depends on `P.R`, which `B` does not touch), so compute it once.
The steady state is re-solved per condition but is always `zeros(N,8)` for CMC.

### Fact 4 — D=1 forcing reuses the Phase-33 nargout-aware wrapper UNCHANGED

`validation/matlab_scripts/spm_fx_cmc_nodelay.m` already exists and is the load-bearing
mechanism. Its `if nargout < 2` guard is what prevents the `spm_diff(M.f,...)` → wrapper →
`spm_fx_cmc(2 outputs)` infinite recursion (Phase-33 found the unconditional 2-output
version OOM'd). **Reuse it verbatim for multi-source**: set `M.f='spm_fx_cmc_nodelay'`,
which makes `nargout(M.f)==2`, sending `spm_int_L:114` down its `:117` branch and keeping
`D=1` (`spm_int_L:112`). The mechanism is N-independent — no wrapper change. The
multi-source fixture script MUST set this and assert `nargout(M.f)==2` and
`isequal(M.x, zeros(N,8))` exactly as `run_spm_erp_dcm.m:83-89` does for N=1.

### Fact 5 — Inherited Jacobian-method carry-forward (Phase 33-03 D2/D3) — gate design

`spm_int_L` freezes `J0 = spm_diff(@spm_fx_cmc, x0, u0, Q, M, 1)` — a **one-sided
forward-difference** Jacobian with default step `dx=exp(-8)`. The shipped
`integrate_local_linearization` uses exact `torch.func.jacrev`. At N=1 this produced:
`J0` (jacrev vs `spm_diff`) floor **5.6e-4**; `y_states` (shipped jacrev) floor **4.7e-8**;
but with the Jacobian method held to SPM's (`spm_diff` FD replicated) the forward is
**bit-exact (0.0)** and the scheme path (SPM's own frozen `Q`) reproduces `y_states` to
**6.6e-14**. The measured `matrix_exp↔spm_expm` floor is **8.6e-11**.

**Consequence for Phase 34:** at 8N×8N the FD-truncation of the shipped exact-AD path will
exceed 4.7e-8. So the multi-source trajectory gate must split, exactly like Phase 33-03:
- **Scheme rung** (bit-exact): drive the torch exp-Euler loop with SPM's OWN exported
  per-condition `Q_update` → isolates loop ordering from the Jacobian method (~1e-13).
- **FD-Jacobian rung** (≤1e-8): build the operator with a `spm_diff`-matched forward-
  difference Jacobian (the test already has `_spm_diff_jacobian(f, x0, dx=exp(-8))` in
  `tests/test_spm_erp_dcm_validation.py:114-129` — REUSE it).
- **Shipped-jacrev rung** (measured floor, recorded, NOT gated at 1e-8): the production
  integrator's exact-AD trajectory — documented as more-accurate-than-SPM.

Do not relitigate this; it is settled. The planner should write the gate with all three
rungs and a loose recorded ceiling on the jacrev rung.

---

## Plan decomposition recommendation

Three plans, three waves. Strict ordering: the pure-torch network forward + `spm_gen_Q`
port + the C4 guard must land and be green on the laptop BEFORE the M3 fixture round-trip,
so the parity test has something to assert against and the B-wiring / coupling logic is
caught in isolation before it compounds through the integrator (V5).

### Wave 1 — Pure-torch network core (LAPTOP, sub-second unit tests) — Plan 34-01
Internally parallelizable; the new module composes the frozen Phase-33 `cmc_neural_mass` /
`erp_input` / `local_linearization`.
- `forward_models/erp_coupled_system.py` (EVOK-01/02/03) — `parameterize_cmc_network`
  (A-block `exp*E` + lateral `(1+4L)`), `apply_condition_modulation` (the `spm_gen_Q`
  port), `cmc_network_f` (the 4 extrinsic terms added to the Phase-33 EOM).
- `simulators/erp_simulator.py` (EVOK-04) — `simulate_erp_dcm(...)` → per-condition
  per-source LFP dict + difference-wave hook (source-state difference; scalp diff is
  Phase 35).
- `tests/test_erp_coupled_system.py` (laptop, no MATLAB) — the structural guards:
  C4 `diag(B)→G[:,6]` guard + the omit-diag negative test; `cmc_network_f(n=1)==cmc_f`
  bit-exact (reuse the Phase-33 fixture); lateral `(1+4L)` triggers on a 2-source
  reciprocal pair; C→ss-only; the fwd-from-`S[:,2]` / bwd-from-`S[:,6]` adjacency table;
  float64 guard.

Gate: all laptop unit tests green; `git diff` shows only NEW files + appends to
`forward_models/__init__.py` / `simulators/__init__.py`. No SPM yet. The single-source
Phase-33 parity suite stays green (regression: `cmc_f` and the integrator are untouched).

### Wave 2 — Multi-source fixture generation + export bridge (M3, MATLAB) — Plan 34-02
Depends on Wave 1 (needs the reference network `P`/`A`/`B`/`X` shapes locked).
- `validation/export_to_mat.py` += multi-source path in (or alongside) `export_erp_dcm`:
  cell arrays `P.A{1..4}` `(N,N)`, `P.B{...}` `(N,N)`, `U.X` `(Cnd, n_effects)`,
  `M.x=zeros(N,8)`, `M.n=8N`, `M.f='spm_fx_cmc_nodelay'`.
- `validation/matlab_scripts/run_spm_erp_dcm_multisource.m` — generates the fixture arrays
  (below); reuses `spm_fx_cmc_nodelay.m` (no change).
- `cluster/scripts/erp_cross_validation.py` += a multi-source mode (or a sibling entry);
  `cluster/sbatch/erp_cross_validation.sbatch` reused.

Gate: M3 job produces `validation/data/erp_multisource_fixtures.mat` with the per-condition
arrays + a provenance `meta` (SPM `$Id`, `dt`, `ns`, `ons`, `dur`, `D=1`, `nargout_Mf==2`,
`x0==zeros(N,8)`, the exact reference `P`/`A`/`B`/`X`, the locked edge list).

### Wave 3 — Multi-source parity ladder (LAPTOP, against committed `.mat`) — Plan 34-03
Depends on Wave 1 + Wave 2. Mirrors the Phase 33-03 decision: the assertions are
torch-vs-frozen-arrays (deterministic float64), so the suite RUNS AND PASSES on the laptop;
`@pytest.mark.spm`/`slow` retained for the optional M3 re-run.
- `tests/test_spm_erp_multisource_validation.py` — the staged ladder (EVOK-05): `spm_gen_Q`
  algebra → network `J0` (FD-matched) → network `Q_update` → multi-source trajectory
  (scheme + FD-Jacobian rungs + measured jacrev floor).

Gate (EVOK-05): `Q.A{1..4}` ≤1e-12 element-wise, `Q.G[:,0]` ≤1e-12; network `J0`
(FD-matched) ≤1e-10; `Q_update` ≤1e-9; trajectory (scheme rung) ~1e-13, (FD-Jacobian rung)
≤1e-8; shipped-jacrev recorded.

**Why this split:** Wave 1 is laptop pure-torch with sub-second tests (the network forward
+ the entire B mechanism are isolable from the integrator); Wave 2 is the only M3/MATLAB
piece (license-gated, record-don't-crash); Wave 3 is the gate. Splitting lets the C4
B-wiring guard be written and merged before any MATLAB dependency exists. Waves are
sequential (each is a tier of the V5 ladder); Wave 1's modules can be built concurrently.

---

## Per-file implementation spec

### `src/pyro_dcm/forward_models/erp_coupled_system.py` (NEW, EVOK-01/02/03) — laptop

Composes the frozen Phase-33 `cmc_neural_mass` (`cmc_sigmoid`, `cmc_flatten`,
`cmc_unflatten`, `J_PERM`, `G0`, `T0_MS`, `E0`, `parameterize_cmc`) without editing it.

```python
def parameterize_cmc_network(p: dict[str, Tensor], n: int) -> dict[str, Tensor]:
    """Extend parameterize_cmc with the n>1 extrinsic blocks + lateral reduction.

    A[i] = exp(P.A[i]) * E0_pair[i], then A[i] /= (1 + 4*L) with
    L = (A[i] > exp(-8)) & (A[i].T > exp(-8))  (spm_fx_cmc.m:68-82).
    E0_pair = [200, 100, 200, 100] (E0[0],E0[1],E0[0],E0[1]).
    Returns {"T":(n,4), "G":(n,10), "C":(n,n_inp), "A":(4,n,n), "S":(n,1)}.
    Reuses parameterize_cmc for T/G/C/S; only the A path is new.
    p["A"]: list/tensor of 4 (n,n) free log-params (zeros default)."""

def apply_condition_modulation(p: dict, x_design: Tensor) -> dict:
    """Port spm_gen_Q.m:24-67 (Fact 2). Free/log space, pre-exp.

    Q = copy(p) without "B"; for i,Xi in enumerate(x_design):
      for j in range(4): Q["A"][j] = Q["A"][j] + Xi * p["B"][i]    # :47
      Q["G"][:,0] = Q["G"][:,0] + Xi * diag(p["B"][i])             # :66 precision path
    C-effect (:35-37) only if p["C"].ndim==3 and size>=2 (else skip).
    p["B"]: list of (n,n); x_design: (n_effects,)."""

def cmc_network_f(x_flat: Tensor, u: Tensor, p: dict, n: int) -> Tensor:
    """Network EOM: Phase-33 cmc_f intrinsic body + 4 extrinsic A@S terms (Fact 1).

    f[ss_I] += A1 @ S[:,2]; f[dp_I] += A2 @ S[:,2];
    f[sp_I] -= A3 @ S[:,6]; f[ii_I] -= A4 @ S[:,6].
    MUST reproduce cmc_f bit-exactly at n=1 (A=zeros -> terms vanish): guard test.
    Calls parameterize_cmc_network; reuses cmc_sigmoid / cmc_flatten / cmc_unflatten.
    float64-guarded (raise TypeError on non-f64, mirroring cmc_f)."""
```

Shapes: `x_flat (8n,)`, `S (n,8)`, `A[i] (n,n)`, extrinsic term `(n,)`. Cite
`spm_fx_cmc.m:68-82,171-198` and `spm_gen_Q.m:41-67` per CLAUDE.md rule 2. Do NOT import
`parameterize_A`. Do NOT cite a fabricated bib key (REF-ERP-* still "verify" in Zotero;
SPM source-file + line citation is allowed).

> **Design decision (flag to planner):** keep the Phase-33 `cmc_f` BYTE-FROZEN and add a
> *new* `cmc_network_f` (duplicating ~12 lines of intrinsic EOM but adding the 4 A@S
> terms), pinned by a `cmc_network_f(n=1) == cmc_f` bit-exact guard test that also
> re-asserts the committed Phase-33 `y_states` fixture. This preserves the proven
> single-source gate with zero edit risk. The alternative — extending `cmc_f` in place
> with the A@S terms (mathematically zero at n=1) — is also valid and avoids duplication
> but edits a frozen, parity-gated function. Recommend the new-function path; see Open
> Question 1.

### `src/pyro_dcm/simulators/erp_simulator.py` (NEW, EVOK-04) — laptop

Mirror `simulators/spectral_simulator.simulate_spectral_dcm` (returns a dict).

```python
def simulate_erp_dcm(
    p: dict[str, Tensor],      # T,G,C,S,R,A(list[4]),B(list) free params
    x_design: Tensor,          # (Cnd, n_effects) between-trial design
    n: int,
    ns: int = 128, dt: float = 0.004,
    ons_ms: float = 60.0, dur_ms: float = 16.0, sus: float = 0.0,
) -> dict:
    """Per-condition per-source LFP via the spm_gen_erp loop (Fact 3).

    Returns {"states": (Cnd, ns, n, 8), "pst": (ns,) peristimulus secs,
             "inputs": (ns, n_inp), "difference_wave": states[dev]-states[std]}.
    The difference-wave hook differences SOURCE states (sp voltage, col 2);
    scalp projection + the true MMN difference wave are Phase 35 (LEAD-03)."""
```

`pst = arange(1,ns+1)*dt - ons_ms/1000` (`spm_gen_erp.m:35`). Uses `erp_gaussian_input`
(frozen Phase-33), `apply_condition_modulation`, `parameterize_cmc_network`,
`integrate_local_linearization`, `cmc_steady_state`. Reshape the integrator's `(ns,8n)`
back to `(ns,n,8)` via the column-major inverse `cmc_unflatten`.

### `validation/export_to_mat.py` (EXTEND `export_erp_dcm`, additive) — laptop

The existing `export_erp_dcm` (`:509`) hardcodes N=1 (no `A`/`B`, `M.x=zeros(1,8)`). Add a
multi-source path — recommend an optional `n`/`A`/`B`/`X` argument set (single-source
defaults unchanged → existing Phase-33 fixture byte-identical) OR a sibling
`export_erp_dcm_multisource`. The cell arrays `P.A{1..4}` and `P.B{...}` must be encoded as
MATLAB cells (object ndarray): `np.empty((1,4),dtype=object)` filled with `(N,N)` float64
blocks, mirroring how the task exporter handles `B{}`. `U.X` is `(Cnd, n_effects)` float64
(cast to double — the int64→`spm_Ce` footgun from Phase 32, commit a27828b). `M.x` =
`zeros(N,8)`, `M.n = 8N` (double), `M.m = n_inp`, `M.f = 'spm_fx_cmc_nodelay'`. Store the
locked edge list + `X` + the exact `P` in `DCM.meta` for provenance (V4).

### `validation/matlab_scripts/run_spm_erp_dcm_multisource.m` (NEW, EVOK-05/06) — M3

Mirror `run_spm_erp_dcm.m` scaffolding (env paths, loud-failure try/catch, `getenv` IO).
Then, with `M.f='spm_fx_cmc_nodelay'`, `M.x=zeros(N,8)`, asserting `nargout(M.f)==2` and
`isequal(M.x,zeros(N,8))`:

```matlab
% (1) spm_gen_Q per condition: the B-wiring guard (C4 / EVOK-05 part 1)
for c = 1:size(DCM.U.X,1)
    Qc = spm_gen_Q(P, DCM.U.X(c,:));
    QA{c} = Qc.A;            % cell of 4 (N,N) free-log-space blocks  (spm_gen_Q:47)
    QG{c} = Qc.G(:,1);       % (N,) free precision column             (spm_gen_Q:66)
end
% (2) per-condition frozen Jacobian + update operator (the SCHEME rung anchor, Fact 5)
for c = 1:size(DCM.U.X,1)
    Qc   = spm_gen_Q(P, DCM.U.X(c,:));
    x0   = spm_vec(M.x); u0 = sparse(M.m,1);
    J0c  = full(spm_cat(spm_diff(@spm_fx_cmc, x0, u0, Qc, M, 1)));   % (8N,8N) spm_diff FD
    dfdx = J0c - eye(8*N)*exp(-16);
    Qupd{c} = (spm_expm(DCM.U.dt*dfdx) - eye(8*N)) / dfdx;           % right-division
    J0{c} = J0c;
end
% (3) the multi-source evoked trajectory (EVOK-05 part 2)
y = spm_gen_erp(P, M, DCM.U);     % cell {Cnd} of (ns, 8N)
save(output_path,'QA','QG','J0','Qupd','y','meta');
```

Record `meta.D=1`, `meta.nargout_Mf=nargout(M.f)`, `meta.N`, `meta.edges`, `meta.X`,
`meta.dt/ns/ons/dur`, the SPM `$Id` strings, and `meta.x0=zeros(N,8)`.

### `cluster/scripts/erp_cross_validation.py` (EXTEND) + `.sbatch` (REUSE) — M3

Add a multi-source mode mirroring the existing single-source `main()` (export → `matlab
-batch run_spm_erp_dcm_multisource` → round-trip shape/meta check → JSON record,
record-don't-crash, exit 0 on soft miss). Reuse `cluster/sbatch/erp_cross_validation.sbatch`
(`MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab`,
`SPM12_PATH=/home/aman0087/fc37/Carrick/spm12`, `--partition=comp`, `--mem=16G`, no pip).

### `tests/test_spm_erp_multisource_validation.py` (NEW, EVOK-05) — laptop (gated on fixture)

Mirror `tests/test_spm_erp_dcm_validation.py` structure; REUSE its `_spm_diff_jacobian(f,
x0, dx=exp(-8))` helper (`:114-129`) and `_update_operator`. Gate on
`erp_multisource_fixtures.mat` availability (not MATLAB), `@pytest.mark.spm`/`slow`. The
ladder rungs below.

---

## The parity gate as a concrete test spec (EVOK-05 / EVOK-06)

Fixture arrays in `erp_multisource_fixtures.mat` (N=5, Cnd=2 for the reference MMN net):

| Array | Shape | What it pins | Tolerance |
|-------|-------|--------------|-----------|
| `QA` | `{Cnd}` × `{4}` × `(N,N)` | `spm_gen_Q` B→all-A folding (free log space) | ≤1e-12 element-wise |
| `QG` | `{Cnd}` × `(N,)` | `spm_gen_Q` `diag(B)→Q.G[:,0]` precision path | ≤1e-12 element-wise |
| `J0` | `{Cnd}` × `(8N,8N)` | network frozen Jacobian (FD-matched) — `cmc_network_f`==`spm_fx_cmc` | ≤1e-10 |
| `Qupd` | `{Cnd}` × `(8N,8N)` | right-division at network scale (C2) | ≤1e-9 |
| `y` | `{Cnd}` × `(ns,8N)` | multi-source evoked trajectory (`spm_gen_erp`) | scheme ~1e-13; FD-Jac ≤1e-8 |

Staged ladder (assert IN THIS ORDER — V5; a failure localises to one stage):

1. **`spm_gen_Q` algebra** (no integrator, no MATLAB-numerics ambiguity): for each
   condition, `apply_condition_modulation(P, X_c)` reproduces `QA{c}` (all 4 blocks) AND
   `QG{c}` element-wise ≤1e-12. **This is the single most important B-wiring guard (C4).**
   Include a NEGATIVE assertion: an `apply_condition_modulation` variant that skips the
   `diag(B)→G[:,0]` line produces a `Q.G[:,0]` that does NOT match `QG{c}` (proves the
   precision path is load-bearing, EVOK-02).
2. **Network `J0` (FD-matched)**: `_spm_diff_jacobian(lambda v: cmc_network_f(v, u0, Q_c,
   N), x0, dx=exp(-8))` vs `J0{c}` ≤1e-10 — proves `cmc_network_f` IS `spm_fx_cmc` at N>1
   with the Jacobian method held to SPM's (Fact 5).
3. **Network `Q_update`**: `_update_operator(J0{c}, dt, 1, None)` (right-division
   `solve(dfdx.T,(E-I).T).T`) vs `Qupd{c}` ≤1e-9.
4. **Trajectory — scheme rung** (bit-exact, isolates C1 loop ordering): drive the exp-Euler
   loop with SPM's OWN `Qupd{c}` → reproduces `y{c}` to ~1e-13.
5. **Trajectory — FD-Jacobian rung** (≤1e-8): build the operator from the FD-matched `J0`
   and integrate → reproduces `y{c}` to ≤1e-8.
6. **Trajectory — shipped-jacrev rung** (MEASURED, recorded, loose ceiling): full
   `integrate_local_linearization(cmc_network_f, ...)` — record the floor (expected > 4.7e-8;
   exact-AD is more accurate than SPM, NOT a bug — Fact 5).

Structural guards (laptop, no MATLAB — in `test_erp_coupled_system.py`, Wave 1):
- **C4 precision guard:** perturb `diag(B)`, assert `parameterize_cmc_network(Q)` changes
  `G[:,6]` (sp self-inhibition) and the omit-diag path does not.
- **`cmc_network_f(n=1) == cmc_f`** bit-exact (reuse committed Phase-33 fixture).
- **Lateral `(1+4L)`** triggers on a 2-source reciprocal pair; absent on a one-way pair.
- **C→ss-only:** input perturbation moves `f[ss_I]` only.
- **Adjacency table:** forward terms read `S[:,2]`, backward read `S[:,6]` (Fact 1 map).
- **float64** at the network-forward boundary.
- **D=1 + x0==0:** read `meta.D==1`, `meta.nargout_Mf==2`; assert torch `x0==zeros(8N)`.

Tolerances are element-wise forward agreement only (V2 — no absolute-F, no `Cp`; the
forward has no normalisation freedom). Anchor the trajectory tolerances to the Phase-33
measured floors (`matrix_exp↔spm_expm` 8.6e-11; scheme 6.6e-14) as small multiples (V3).

---

## Compute routing

| Work | Where | Why |
|------|-------|-----|
| Wave 1 modules + `test_erp_coupled_system.py` | **Laptop** | pure-torch; 5-source × 2-cond × 128-step × 40×40 matrix_exp is sub-second; `cmc_network_f(n=1)` guard reuses the committed fixture |
| Wave 2 MATLAB fixtures (`run_spm_erp_dcm_multisource.m`) | **M3** | local MATLAB license unreachable (FlexLM -15); R2022a + Carrick spm12 on `comp`; submit via `cluster/sbatch/erp_cross_validation.sbatch` |
| Wave 3 `test_spm_erp_multisource_validation.py` | **Laptop (against committed `.mat`)** + optional M3 re-run | assertions are torch-vs-frozen-arrays (deterministic f64), like Phase 33-03 D1 — the laptop run IS authoritative |

ssh-agent unlocked at submit; Mutagen sync for code/results (not git push/pull). Fixture is
small (per-condition `(40,40)` + `(128,40)`); the job is minutes, well under `--time=01:00:00`.

---

## Open questions / decisions for the planner

1. **`cmc_network_f` new-function vs extend `cmc_f` in place.** Recommend a NEW
   `cmc_network_f` in `erp_coupled_system.py` (keeps the Phase-33 byte-frozen, zero
   regression risk; ~12 lines duplicated, pinned by the `n=1`-equals-`cmc_f` guard). The
   alternative (add the 4 A@S terms to `cmc_f`, mathematically zero at n=1) avoids
   duplication but edits a parity-gated function. Either preserves single-source parity
   bit-exactly. **Decision needed before Wave 1.**

2. **Exact 8N state-vector flatten/block order.** Column-major `spm_vec` (state-blocked:
   `[V_ss(all src); I_ss(all src); V_sp(all src); ...]`) is already implemented as
   `cmc_flatten = x.T.reshape(-1)` / `cmc_unflatten = x_flat.reshape(8,n).T` and proven at
   N=1. **Confirm it holds at N>1** (it does — the transpose makes it state-blocked over
   sources) and that the per-condition `J0{c}` fixture (built by `spm_diff` on the same
   `spm_vec` order) aligns. The Wave-3 `J0` rung is the alignment check; if it fails by a
   permutation, the flatten order is the suspect.

3. **The reference 5-source MMN edge list — LOCK IT IN THE FIXTURE.** Phase 34 needs a
   concrete valid multi-source A/B/C. The canonical Garrido/Ranlund hierarchy (topology
   only; MNI coords are **Phase 36 / ERPDCM-03**, NOT 34 — do not hard-code coords here):
   - **Forward (A1,A2):** A1L→STGL, A1R→STGR, STGL→rIFG, STGR→rIFG.
   - **Backward (A3,A4):** the reverse — rIFG→STGL, rIFG→STGR, STGL→A1L, STGR→A1R.
   - **Lateral (reciprocal, triggers `1+4L`):** STGL↔STGR (a clean reciprocal test pair).
   - **Input C:** drives A1L and A1R (bilateral auditory recipients) only.
   - **Condition B (deviant X=[1] vs standard X=[0]):** a single `(5,5)` `B{1}` on the
     forward/backward edges, with non-zero `diag(B)` at rIFG + bilateral A1 (the precision
     nodes). **Decision: freeze the exact `A`/`B`/`C` masks + `X` in `export_erp_dcm` /
     `DCM.meta` before the Wave-2 fixture run** (V4 — fixture regeneration is a reviewed
     change). A smaller 2-source reciprocal net is also worth a fixture for the lateral
     guard in isolation; the planner may add it.

4. **Defer the delay path (D≠1) — confirmed Phase 34 keeps D=1.** EVOK-06 forces D=1 via
   `spm_fx_cmc_nodelay` (Fact 4) and asserts it in the fixture script. The full
   `spm_dcm_delay` port stays deferred (milestone anti-feature). No action beyond
   asserting `nargout(M.f)==2` and `meta.D==1`.

5. **Difference-wave observable at Phase 34.** `erp_simulator`'s difference-wave hook
   operates on **source states** (sp voltage), since the lead field / scalp projection is
   Phase 35 (LEAD-01/03). The negative-going / frontal-dominance assertions (S2) belong to
   Phase 35/36 once `spm_lx_erp` exists — Phase 34 only needs a non-trivial source-level
   difference between the two conditions (which exists iff B is wired). State this in the
   `simulate_erp_dcm` docstring so the planner doesn't over-scope.

6. **Jacobian-method gate split (inherited, settled).** Do NOT gate the shipped exact-AD
   integrator at ≤1e-8 against an SPM `spm_diff`-FD-Jacobian fixture. Use the three-rung
   split (scheme / FD-Jacobian / measured-jacrev) per Fact 5. The planner should copy the
   Phase 33-03 ladder pattern, not reinvent the tolerance.

7. **`spm_gen_Q` C-effect (`:35-37`) and `M`/`int` branches.** The reference net uses a
   single-page `C` (skip the C-effect) and has no `P.M`/`P.AN`/`int` fields (skip those
   branches). Port `apply_condition_modulation` to mirror the `try`/`isfield` guards so it
   no-ops cleanly; do NOT implement the NMDA `AN`/`BN` or `int` paths this phase.

---

## Sources

### Primary (HIGH — SPM12 source read line-by-line this session at `C:/Users/aman0087/Documents/Github/spm12`)
- `toolbox/dcm_meeg/spm_fx_cmc.m` (`$Id: 7279`) — extrinsic blocks + E pairing (`:47,68-72`),
  lateral `(1+4L)` reduction (`:79-82`), the four extrinsic→EOM routes
  (`:171,177,183,189`), C→ss input (`:86,107,171`), intrinsic perm `j` (`:151`), modulatory
  M (`:158-160`), Jacobian via `spm_diff(M.f,...)` (`:206-208`), delay call (`:226`).
- `toolbox/dcm_meeg/spm_gen_Q.m` (`$Id: 7279`) — B→all-A folding (`:41-47`), C-effect
  (`:35-37`), modulatory M (`:59-61`), the `diag(B)→Q.G(:,1)` precision path (`:65-67`),
  NMDA/`int` branches not used (`:49-54,71-79`).
- `toolbox/dcm_meeg/spm_gen_erp.m` (`$Id: 6427`) — the condition loop (`:69-86`), input
  grid + peristimulus time (`:35,44`), `M.u` removal (`:48-54`), per-condition steady state
  + integrate (`:80,84`).
- `spm_int_L.m` (`$Id: 7143`) — exp-Euler loop, `exp(-16)` regulariser + right-division
  `Q=(E-I)/dfdx` (`:126-127`), `spm_diff`-frozen Jacobian, N-default (`:70`), nargout/D
  branch (`:112-122`).
- `spm_dcm_neural_x.m` (`$Id: 6112`) — CMC otherwise→zeros (`:70-76`).
- `spm_erp_u.m` (`$Id: 7679`) — Gaussian bump (`:42-64`), ms timebase, 32-scale.

### Primary (HIGH — repo source / shipped Phase-33 artifacts read this session)
- `src/pyro_dcm/forward_models/cmc_neural_mass.py` — `cmc_f`, `parameterize_cmc`,
  `cmc_sigmoid`, `cmc_flatten/unflatten`, `J_PERM=(6,1,2,3,0,4,5,7,8,9)`, `G0`, `T0_MS`,
  `E0` (the frozen single-source forward Phase 34 composes; A returned as `zeros(4,n,n)`).
- `src/pyro_dcm/utils/local_linearization.py` — `integrate_local_linearization`,
  `_update_operator` (the spm_int_L port; reused unchanged; `jacrev`-frozen Jacobian).
- `src/pyro_dcm/forward_models/{cmc_priors,erp_input}.py` — `cmc_steady_state` (zeros),
  `cmc_prior_moments` (A moments at n>1 already present, `:73-77`), `erp_gaussian_input`.
- `validation/export_to_mat.py:442-650` — `export_erp_dcm` (single-source; the cell/scalar
  savemat conventions + the int64→double `spm_Ce` footgun fix to mirror).
- `validation/matlab_scripts/{run_spm_erp_dcm.m, spm_fx_cmc_nodelay.m}` — the fixture
  scaffolding + the load-bearing nargout-aware D=1 wrapper (reused unchanged).
- `cluster/scripts/erp_cross_validation.py` + `cluster/sbatch/erp_cross_validation.sbatch`
  — the M3 entrypoint + record-don't-crash idiom to extend.
- `tests/test_spm_erp_dcm_validation.py:81-129` — the V5 ladder + `_spm_diff_jacobian(f,x0,
  dx=exp(-8))` helper to REUSE for the network `J0` rung.
- `src/pyro_dcm/simulators/spectral_simulator.py:37-153` — the `simulate_*` return-dict
  idiom `erp_simulator` mirrors.
- `.planning/phases/33-.../33-03-SUMMARY.md` — the AD-vs-FD-Jacobian carry-forward (D2/D3),
  measured floors (matrix_exp 8.6e-11; scheme 6.6e-14; jacrev y_states 4.7e-8).
- `.planning/REQUIREMENTS.md:380-396` — EVOK-01..06.

### Peer milestone research (consumed, not re-derived)
- `.planning/research/v0.8.0/{SUMMARY,STACK,PITFALLS,ARCHITECTURE}.md` — C4 (B dual
  mechanism), M2 (delay default), V1-V5 (validation methodology), the additive seam.

## Metadata
**Confidence breakdown:**
- Extrinsic topology / state-coupling map (Fact 1): HIGH — transcribed from
  `spm_fx_cmc.m:68-82,171-198` with the firing-column / target-row mapping resolved.
- `spm_gen_Q` dual mechanism (Fact 2): HIGH — `spm_gen_Q.m:41-67` read line-by-line; the
  `diag(B)→Q.G[:,0]→G[:,6]` precision path confirmed against the `J_PERM` remap.
- Evoked loop + integrator reuse (Fact 3): HIGH — `spm_gen_erp.m:69-86`; the integrator is
  the proven Phase-33 port, reused unchanged.
- D=1 wrapper reuse (Fact 4): HIGH — the shipped `spm_fx_cmc_nodelay.m` nargout mechanism
  is N-independent.
- Jacobian-method gate split (Fact 5): HIGH — settled by Phase 33-03 decisions D2/D3 with
  recorded floors.
- Reference 5-source edge list: MEDIUM — the canonical Garrido/Ranlund topology is
  well-known but the exact masks must be LOCKED in the fixture before Wave 2 (Open Q3);
  MNI coords explicitly deferred to Phase 36.

**Research date:** 2026-06-26
**Valid until:** stable (SPM12 frozen at `$Id 7279/7143/6427`; repo conventions stable) — ~30 days.

## RESEARCH COMPLETE
