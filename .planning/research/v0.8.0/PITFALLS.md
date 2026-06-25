# Domain Pitfalls — v0.8.0 DCM for Evoked Responses (CMC EEG/MEG ERP)

**Domain:** Time-domain CMC neural-mass → evoked → single-dipole lead-field → scalp-ERP
forward stack, bit-for-bit-close to SPM12 `../spm12/toolbox/dcm_meeg/`, synthetic /
forward-only, reusing the existing VL + amortized inference.
**Researched:** 2026-06-25
**Overall confidence:** HIGH (every parity-critical claim traced to SPM12 source read
line-by-line; integration-scheme + delay-default corroborated by Ozaki-1992 /
van Wijk et al. 2021 literature; existing-codebase traps traced to the actual
`inference/forward_models.py` + Phase-32 findings)

**Owning phases:** 33 = CMC forward (single-source dynamics + `spm_int_L`); 34 =
extrinsic coupling + evoked input + condition modulation `B`; 35 = lead-field +
scalp projection; 36 = full model class + MMN precision-sweep demo.

> **How to read this file.** Pitfalls are ordered by parity-breaking severity. Every
> entry gives **warning signs** (detect early), **prevention** (actionable, becomes a
> success-criterion guard), and **owning phase**. The five CRITICAL pitfalls (C1–C5) are
> the ones that *silently pass smoke tests and fail parity*; they should each become an
> explicit, fixture-backed test requirement in the roadmap.

---

## The single most important framing

This milestone's whole value is **mathematical equivalence to SPM12**, the same
discipline Phase 32 used. Phase 32 proved that even a "matched" cross-engine comparison
left a **constant 270-nat free-energy offset** and a **systematic posterior-mean
divergence** from tiny forward-model differences. The lesson transfers directly: *small,
silent forward-model discrepancies are the default outcome, not the exception.* Every
CMC sub-component (sigmoid, permutation `j`, integrator, lead field) must be frozen
against MATLAB-exported fixtures **at the smallest possible granularity**, because once
they compound through the integrator they are nearly impossible to bisect. The
test architecture must make divergence visible **per-component**, not just at the scalp.

---

## Critical Pitfalls

Mistakes that silently pass smoke tests, break SPM parity, and are extremely hard to
bisect once compounded through the integrator. Each must become a fixture-backed gate.

### C1 — Integration-scheme mismatch (rk4/dopri5 vs `spm_int_L` exponential-Euler) [HEADLINE]

**What goes wrong.** `spm_gen_erp.m:84` integrates with `spm_int_L.m`, an
**exponential-Euler / local-linearisation (Ozaki 1992)** scheme with a **frozen Jacobian**
evaluated once at the expansion point `x0`:
`Q = (spm_expm(dt·D·J/N) − I)·inv(J)`, then `v ← v + Q·f(v,u)` per time bin
(`spm_int_L.m:126-143`). The existing `TaskDCMForward`/`LatentCircuitForward` call
`integrate_ode(..., method="rk4", step_size=dt)`. **RK4 and exponential-Euler-with-frozen-J
produce different trajectories** for the *same* ODE and same `dt` — they are different
integrators, not different step sizes of the same one. A torch CMC forward that "looks
right" (right peak latency, plausible shape) will diverge from SPM by a few percent that
compounds over 128 time bins, and no amount of shrinking `dt` makes rk4 converge to
`spm_int_L` (they converge to the *true* ODE solution but not to each other for finite
`dt`, and SPM's `dt`=4 ms is *not* in the small-`dt` regime).

**Warning signs (detect early):**
- Per-source LFP traces have correct shape but a consistent ~1–5% amplitude/latency drift
  vs SPM that *grows* toward the end of the peristimulus window.
- The drift is *insensitive* to reducing the torch ODE step but *sensitive* to switching
  the integrator family.
- `torch.allclose` to SPM at the first 1–2 time bins but failing by bin 50+.

**Prevention (actionable):**
- **Do NOT route CMC through `integrate_ode`/torchdiffeq.** Port `spm_int_L` directly as
  a new pure-torch integrator (`forward_models/erp_integrator.py`):
  `J = jac(f, x0); J = J − eye(n)·exp(−16); E = torch.matrix_exp(dt·D·J/N);
  Q = solve(J.T, (E−I).T).T; loop v += Q @ f(v,u)`. Use `torch.linalg.solve`
  (CLAUDE.md rule), never `inverse`. Keep `float64`. `N=1`, `D=1` for first pass.
- **The early-detection fixture (write this test FIRST, Phase 33):** a *single-source*
  CMC with **fixed reference parameters** (prior mean `P=0`, default `E/G/T`). Export
  from MATLAB, via the existing `validation/` `.mat` bridge, **three** frozen arrays for
  this exact config:
  1. `J0` — the frozen Jacobian `df/dx` at `x0=zeros(8,1)` (call `spm_fx_cmc` with
     2 outputs, or `spm_diff`),
  2. `Q_update` — the update operator `(spm_expm(dt·J0)−I)/J0`,
  3. `y_states` — the full `spm_int_L` state trajectory `(ns, 8)` for a known Gaussian `u`.
  Assert torch reproduces `J0` (≤1e-10), `Q_update` (≤1e-9, this isolates
  `matrix_exp`↔`spm_expm`), and `y_states` (≤1e-8) **at the single-source level before
  any extrinsic coupling exists**. This catches the integrator mismatch in isolation;
  if `J0` and `Q_update` match but `y_states` does not, the bug is the *loop ordering*
  (`v += Q@f` vs `v = Q@(v+f)` etc.), not the algebra.
- Assert the `exp(−16)` Jacobian regulariser is applied **before** forming `Q` (it shifts
  every eigenvalue; omitting it changes `inv(J)`).

**Owning phase:** **33** (and it gates everything downstream).

---

### C2 — `spm_expm` vs `torch.matrix_exp`, and `(expm−I)/J` right-division

**What goes wrong.** Two sub-traps inside C1's `Q`:
1. **Right-division.** MATLAB `(spm_expm(...) − I)/dfdx` is `(E−I)·inv(J)`
   (right inverse), *not* `inv(J)·(E−I)`. `J` is not symmetric for CMC, so the two differ.
   A naive port writes `inv(J) @ (E−I)` and silently gets a transposed-ish wrong operator
   that still produces a smooth, plausible ERP.
2. **`spm_expm` is not `expm`.** SPM's `spm_expm` handles the augmented/sparse case and
   uses scaling-and-squaring + Padé; `torch.matrix_exp` is the same algorithm family and
   agrees to ~1e-12 on a dense matrix — *but only if you feed it the same matrix*. If `D`
   (delay operator) or the `exp(−16)` shift is applied inconsistently, the inputs differ.

**Warning signs:**
- `Q_update` fixture (C1) matches under one operand order and fails under the other —
  if it "passes" only after you flipped a transpose by trial-and-error, you have a latent
  sign/orientation bug that will resurface with `n>1` sources.
- `matrix_exp` vs `spm_expm` discrepancy >1e-9 on the *same* exported `dt·J` → you are not
  feeding identical matrices (check the `exp(−16)` shift and `dt·D/N` scaling).

**Prevention:**
- Implement as `Q = torch.linalg.solve(J.T, (E − I).T).T` (the stable form of
  right-division) and assert it against the MATLAB `Q` fixture directly (C1 item 2).
- Add a *standalone* `matrix_exp ↔ spm_expm` test on a frozen `dt·J0` array exported from
  MATLAB; **measure** the tolerance (STACK.md flags the 1e-12 figure as MEDIUM confidence
  — the test asserts it, never assumes it).

**Owning phase:** **33**.

---

### C3 — Parameter-transform / units / permutation traps (silent, smoke-test-proof)

**What goes wrong.** A cluster of off-by-one / wrong-transform errors, every one of which
yields a *finite, smooth, plausible* ERP that passes a NaN/shape smoke test but fails
parity. From `spm_fx_cmc.m` + `spm_cmc_priors.m`:

| # | Trap | SPM ground truth | Failure mode if wrong |
|---|------|------------------|-----------------------|
| a | **Intrinsic permutation `j`** | `j=[7 2 3 4 1 5 6 8 9 10]`; only **4** free `P.G` cols map via `G(:,j(i))*=exp(P.G(:,i))`, so `P.G(:,1)→G(:,7)=sp self-inhibition` (`spm_fx_cmc.m:151-154`) | If you apply `P.G(i)` to `G(i)` directly, the *precision knob* (the whole point of the milestone) modulates the wrong connection. Smoke test passes; MMN sweep is meaningless. |
| b | **Number of free intrinsic/τ params = 4, not 10/8** | `E.G=sparse(n,4)`, `E.T=sparse(1,4)` (`spm_cmc_priors.m:120-122`) | Packing 10 G or 8 T free params breaks the VL prior layout and silently over-parameterises. |
| c | **Time-constant units** | `T=[2 2 16 28]` ms → `T/1000` to seconds (`:114`), then `T*=exp(P.T)` | Forgetting `/1000` makes synaptic kernels 1000× too slow; ERP becomes a flat ramp. Forgetting the rate is `k=−1/T` (alpha kernel `x''=2k x'−k² x`) inverts the dynamics. |
| d | **Sigmoid is dimensionless deviation-from-baseline** | `R=(2/3)·exp(P.S)`; `F=1/(1+exp(−R·x))`; `S=F−1/2` (`:90-94`) — the **−1/2** baseline subtraction is load-bearing | Using raw `1/(1+exp(−Rx))` (no −1/2) injects a constant 0.5 drive into every population → wrong steady state, wrong amplitude. `x` here is **mV-scale voltage state**, fed *dimensionlessly* into the sigmoid via `R`. |
| e | **The `(2/3)` baseline slope** | `R=2/3` *then* `R*=exp(P.S)` | Dropping the 2/3 or applying `exp(P.S)` to the wrong base rescales every firing rate. |
| f | **Extrinsic sign convention** | extrinsic = `+exp(P.A)·E(i)` (strictly positive), directional signs applied **structurally** (`+A{1}`, `−A{3}`, `−A{4}` in the eqs) | Reusing the fMRI `parameterize_A` (which does `−exp(x)/2` on the diagonal for stability) is **WRONG for CMC** — STACK.md §2.8 flags this explicitly. Build a dedicated CMC parameteriser. |
| g | **Input scaling chain** | `C=exp(P.C)`; exogenous branch `U=C·u·32` (`:86,107`); and `u` itself is already `32·`Gaussian (`spm_erp_u.m:63`) | Double-counting or dropping a 32 scales the whole ERP. Also: during ERP integration `M.u` is *removed* (`spm_gen_erp.m:48-54`) so the **exogenous** branch (`U=C·u·32`), not the endogenous (`u·512`), is the one that runs. |
| h | **State flattening order** | MATLAB `spm_vec` is **column-major** → flat state is *state-blocked* `[V_ss(all src); I_ss(all src); …]` (STACK.md §2.1) | A C-order (`reshape`) flatten misaligns the Jacobian, `kron(J,L)`, and any state-vector comparison vs MATLAB. |

**Warning signs:** the single-source fixture (C1) passes for the *steady state* (all zeros)
but the *driven* trajectory amplitude is off by a clean factor (×2, ×0.5, ×32, ×1000) or
the wrong intrinsic connection responds to a `P.G` perturbation. A clean multiplicative
offset is the signature of a units/scaling trap; a wrong-connection response is the
signature of the permutation trap.

**Prevention:**
- Build a **per-transform unit test** that perturbs each free param one at a time and
  asserts the resulting `f(x,u,P)` (the *derivative field*, exported from `spm_fx_cmc`
  with a frozen non-zero `x` and `u`) matches MATLAB element-wise (≤1e-10). This tests
  the algebra *before* the integrator and isolates every entry in the table above.
  Crucially, perturb `P.G(:,1)` and assert **`G(:,7)` (not `G(:,1)`) changes** — a direct
  guard for the permutation.
- Pin a comment citing `spm_fx_cmc.m` line numbers next to each transform (CLAUDE.md
  rule 2). Do **not** import `parameterize_A` from `forward_models/neural_state.py`.

**Owning phase:** **33** (transforms/sigmoid/permutation/units), **34** (input scaling +
`M.u` removal, extrinsic sign).

---

### C4 — Extrinsic-coupling & condition-modulation `B` wiring (wrong population, wrong space)

**What goes wrong.** CMC extrinsic connections have a **specific population topology** that
is easy to mis-wire, and the condition modulation `B` is applied in a **non-obvious place
and space**:

1. **Forward/backward/lateral rules** (`spm_fx_cmc.m:69-72, 171-191`):
   - `A{1}` forward `sp→ss` (`+`, into granular), `A{2}` forward `sp→dp` (`+`, into deep).
   - `A{3}` backward `dp→sp` (`−`, into superficial pyramidal),
     `A{4}` backward `dp→ii` (`−`, into inhibitory interneurons).
   - **Forward connections originate from superficial pyramidal firing `S(:,3)`; backward
     from deep pyramidal firing `S(:,7)`.** Wiring forward from the wrong source population,
     or sending backward into ss instead of sp/ii, gives a plausible-but-wrong network.
   - **Lateral/reciprocal reduction** (`:79-82`): any connection that is reciprocal
     (`A{i}>exp(−8)` *and* `A{i}'>exp(−8)`) is divided by `(1+4·L)`. Omitting this makes
     reciprocally-connected sources ~5× too strongly coupled.
2. **`C` input enters spiny-stellate (granular) only** — `U=C·u·32` is added to the `ss`
   equation (`:171`), not to pyramidal cells. Driving the wrong population is a classic
   ERP-DCM error.
3. **`B` condition modulation is applied in LOG space, pre-exp, and to *all* `A{j}`
   simultaneously, AND to intrinsic gain via its diagonal** (`spm_gen_Q.m:45-67`):
   ```
   Q.A{j} = Q.A{j} + X(i)*P.B{i}     % every j (fwd i, fwd ii, bwd i, bwd ii) gets SAME P.B{i}
   Q.G(:,1) = Q.G(:,1) + X(i)*diag(P.B{i})   % the DIAGONAL of B also modulates intrinsic G col 1
   ```
   `Q.G(:,1)` is `P.G` column 1 → permutation → `G(:,7)` = **sp self-inhibition =
   precision**. So a *single* `B` matrix simultaneously (a) scales all four extrinsic
   connection types and (b) modulates superficial-pyramidal gain via its diagonal. This
   is the exact mechanism by which "the deviant modulates precision." Wiring `B` only to
   `A` (forgetting the `diag(B)→G` path), or applying it post-exp (multiplicative instead
   of additive-in-log), **destroys the MMN precision mechanism** the milestone exists to
   demonstrate.

**Warning signs:**
- The standard-condition ERP is correct but the deviant−standard difference wave is the
  wrong sign, the wrong shape, or absent.
- Perturbing `diag(P.B)` has no effect on superficial-pyramidal gain (means the
  `diag(B)→G(:,1)→G(:,7)` path is missing).
- A 2-source reciprocal network is ~5× over-coupled (missing lateral reduction).

**Prevention:**
- **Multi-source fixture (Phase 34):** export `spm_gen_Q` output `Q` for a known `P`,
  `X=[1]` (deviant) and assert torch reproduces `Q.A{1..4}` *and* `Q.G(:,1)` element-wise.
  This is the single most important `B`-wiring guard.
- Encode the forward/backward/lateral topology as a **named, asserted adjacency** (a table
  test mapping each `A{i}` to its `(source pop, target pop, sign)`), not as ad-hoc indexing.
- Assert `C` enters only the `ss` equation; assert the lateral `(1+4L)` reduction triggers
  on a reciprocal test pair.
- Implement `B` as **additive in log-space pre-exp** on `A` and as `+diag(B)` on `G(:,1)`,
  mirroring `spm_gen_Q`; never as a multiplicative post-exp factor.

**Owning phase:** **34**.

---

### C5 — Lead-field / dipole / `kron(J,L)` & single-dipole simplification traps

**What goes wrong.** The scalp projection has four independent traps:

1. **Which state is observed — `P.J`.** For CMC, `pE.J = sparse(1,3,1,1,8)`
   (`spm_L_priors.m`, CMC branch) → **only state 3 = superficial-pyramidal voltage**
   contributes by default; `pC.J = sparse(1,[1 7],1/32,1,8)` allows states 1 (ss V) and
   7 (dp V) free. Observing deep pyramidal voltage (state 7), or a conductance state
   (even index), or using the ERP-model `J` (9 states, index pattern differs) gives a
   wrong, physiologically-inverted scalp signal. **EEG is dominated by L2/3 superficial
   pyramidal depolarisation — state 3, not 7.**
2. **`kron(J,L)` structure & state order.** `spm_lx_erp.m:31` builds `L = kron(P.J, L)`
   (per-state lead field, `(Nc, 8n)`). The `kron` ordering must match the column-major
   state-blocked flatten (C3h). A C-order port silently maps the lead field to the wrong
   states.
3. **Single-dipole ECD: orientation, scaling, sign.** `spm_erp_L.m:75-77`:
   `L(:,i) = G(:,:,i)·P.L(:,i)` where `G(:,:,i)` is the `(Nc,3)` physical gain and
   `P.L(:,i)` the free **3-vector dipole moment** (`E.L=0, V.L=64`). `G` passes through
   `spm_cond_units` (`:74`) — a unit-conditioning rescale. Reproducing only `kron(J,L)`
   in torch while exporting `G` from MATLAB (the STACK.md recommendation) means the
   **exported `G` must already include `spm_cond_units`**; re-deriving the gain without it
   gives a clean but wrong scalar scale. Dipole **orientation sign** is free (`P.L` can be
   ±) so a sign-flipped moment produces a sign-flipped ERP that still "fits" — dangerous
   for the difference-wave sign (see S8).
4. **Single-dipole simplification risk.** The milestone fixes one ECD per source. SPM's
   ECD with `siunits` applies a `diag([1e-3 1e-3 1e-3 1])` MNI→m rescale (`spm_erp_L.m:59-60`)
   and a `fromMNI` transform. For LFP-mode parity (recommended first) `L=diag(P.L)` is
   trivial and sidesteps all of this — **start there**.

**Warning signs:**
- Scalp ERP shape correct but global polarity inverted (wrong `P.J` state or sign).
- Per-source LFP matches SPM but scalp projection is off by a constant scale (missing
  `spm_cond_units` / `siunits` rescale).
- The difference wave (MMN) sign flips when you change nothing but the dipole moment sign.

**Prevention:**
- **Phase 35: start with LFP spatial model** (`L=diag(P.L)`, no head model) — exact SPM
  parity, isolates dynamics from the lead field. Only then add ECD.
- For ECD, **export `G(:,:,i)` (post-`spm_cond_units`) from MATLAB** via the `validation/`
  bridge and reproduce *only* `kron(P.J, L)` + projection in torch (no FieldTrip/MNE).
  Freeze a `spm_lx_erp` output `L` fixture `(Nc, 8n)` and assert element-wise.
- **Hard-code and assert `P.J` default = unit at state index 3** (0-based: 2), with line
  citation to `spm_L_priors.m`. Add a guard test that the observed state is superficial
  pyramidal voltage, not deep.
- Pin the dipole-moment sign convention in the fixture so the difference-wave sign (S8) is
  reproducible.

**Owning phase:** **35**.

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or partial parity — not silent killers but
costly.

### M1 — Initial conditions / steady state assumed non-trivial

**What goes wrong.** `spm_gen_erp.m:80` calls `spm_dcm_neural_x(Q,M)` for the steady state
*per condition*. For **CMC this returns zeros** — `spm_dcm_neural_x` only runs a Newton
fixed-point solve for conductance models (`spm_fx_cmm`/`spm_fx_mfm`); CMC hits the
`otherwise` branch and `x0 = zeros(n,8)` (STACK.md §3.1). Porting a Newton solver "to be
safe" wastes effort and risks a *different* x0 than SPM (which would shift the frozen
Jacobian expansion point → C1 divergence).

**Warning signs:** torch x0 is a small non-zero vector; frozen Jacobian differs from the
exported `J0` fixture in low-order digits.

**Prevention:** **assert `x0 == zeros(n,8)` for CMC** with a citation to the
`spm_dcm_neural_x` `otherwise` branch; do not implement a fixed-point solve this milestone.
The C1 `J0` fixture already pins the expansion point.

**Owning phase:** **33**.

### M2 — Delay operator `D` deferred but silently nonzero in the reference

**What goes wrong.** STACK.md/FEATURES.md correctly defer the full delay operator (`D=1`).
But **`spm_dcm_delay` (polynomial delay) is the *default* for the CMC model**
(confirmed by van Wijk et al. 2021 / search). If the MATLAB reference fixtures are
generated through the normal `spm_dcm_erp` path *with delays on*, the exported trajectory
will embed a non-identity `D` and a delay-free torch port will never match it — you'll
chase a phantom C1 bug.

**Warning signs:** single-source `J0`/`Q_update` match but `y_states` is consistently
phase-shifted/latency-shifted vs SPM; the shift looks like a small fixed delay.

**Prevention:** **generate the reference fixtures with delays explicitly disabled**
(set `M.pF`/delay params so `spm_dcm_delay` returns `D=1`, or call `spm_int_L` on a `Q`
with `P.D=0` and confirm `nargout`<3 path). Document in the fixture-generation MATLAB
script that `D=1`. Add `spm_dcm_delay` port only as a gated sub-task *after* the delay-free
forward matches, and re-export fixtures with delays on for that step.

**Owning phase:** **34** (flagged), revisited if parity demands.

### M3 — AR(1) precision vs identity (free-energy parity, not forward parity)

**What goes wrong.** SPM ERP inference uses `xY.Q = {spm_Q(1/2,Ns,1)}` (AR-1 serial
correlation, `spm_dcm_erp.m`). The existing time-domain `ForwardModel.build_precision`
returns **identity**. For *forward* parity this is irrelevant; for *free-energy / evidence*
parity (if the demo ever compares F or runs VL against SPM) it reintroduces exactly the
Phase-32 constant-offset problem — and an AR(1) vs identity precision mismatch is not even
a *constant* offset, it changes the residual weighting.

**Warning signs:** forward traces match SPM but any F-based or VL-posterior comparison
diverges in a residual-shape-dependent way.

**Prevention:** v1 uses identity precision (correct for forward-only scope). **If inference
parity is attempted, add the AR(1) `spm_Q(1/2,Ns)` precision basis** to the ERP
`build_precision` and re-validate. Document that absolute-F is off-limits as a parity gate
(see S/validation section) regardless.

**Owning phase:** **36** (only if D1/D2 inference is in scope; otherwise documented + deferred).

### M4 — `dt` mismatch with the reference dataset

**What goes wrong.** `spm_int_L` uses `dt = U.dt` (default `0.004 s`; in fitting `xY.dt`).
The update operator `Q` depends on `dt` through `matrix_exp(dt·J)`. A torch `dt` that
differs from the MATLAB fixture's `dt` (e.g. 1 ms vs 4 ms, or seconds-vs-ms confusion in
`spm_erp_u` where `t` is converted `t*1000`) breaks parity completely.

**Warning signs:** `Q_update` fixture fails by a large factor; trajectory has wrong number
of samples.

**Prevention:** thread a single `dt` (and `ns`, `M.ons`, `M.dur`) constant from the fixture
metadata; assert torch `ns == fixture ns` and `dt == fixture dt`. Keep `spm_erp_u` time in
**ms** internally (`t*1000`) exactly as SPM.

**Owning phase:** **34** (input) / **33** (integrator).

---

## Minor Pitfalls

Annoyances that are fixable but waste time.

### N1 — `float32` creeping in
SPM is `double`; the repo convention is `float64`. A `float32` tensor anywhere in the
integrator drops parity to ~1e-4, swamping the ~1e-9 tolerances. **Prevention:** assert
`dtype==float64` at the `ForwardModel` boundary. **Phase 33.**

### N2 — Eigenvalue clipping reflex from the fMRI path
CLAUDE.md says "clip eigenvalues of A: real<0". That rule is for the **fMRI neural state
matrix A**, not the CMC Jacobian. The CMC `spm_int_L` path uses the `exp(−16)` regulariser
instead and does **not** clip eigenvalues. Applying the fMRI clip to the CMC Jacobian
changes `Q` and breaks parity. **Prevention:** the CMC integrator uses only the `exp(−16)`
shift; no eigenvalue clipping. **Phase 33.**

### N3 — `M.ons`/`M.dur` defaults
`spm_erp_u` defaults `M.dur=32` if mismatched length (`:29`) and `delay = M.ons + 128·P.R(i,1)`
with `M.ons` default 60 ms. Hard-coding the wrong onset shifts the bump. **Prevention:**
carry `M.ons`, `M.dur`, `M.sus` from fixture metadata. **Phase 34.**

### N4 — `prop`/sustained-input mix
`spm_erp_u.m:62`: `U = prop·cumsum(U)/sum(U) + U·(1−prop)` with `prop=M.sus·exp(P.R(i,3))`,
default `M.sus=0`. Easy to drop the sustained term; harmless at default but wrong if the
demo enables it. **Prevention:** port the full expression; default `M.sus=0`. **Phase 34.**

---

## Backward-Compatibility Pitfalls (additive-only, bit-exact existing paths)

The CLAUDE.md additive-only rule means the existing fMRI/spectral/rDCM/latent-circuit paths
must remain **bit-exact**. The ERP work touches shared infrastructure, so:

### B1 — Reusing `parameterize_A` and perturbing it
`inference/forward_models.py:19` imports `parameterize_A`/`parameterize_B` from
`neural_state.py`. CMC needs a *different* extrinsic parameterisation (C3f). **Do NOT edit
`parameterize_A`** to "generalise" it for CMC — that would change the spectral/task/latent
forward outputs and break Phase-32-validated parity. **Prevention:** add a *new*
`parameterize_cmc_extrinsic` function; leave the existing one untouched. Guard with the
existing spectral/task recovery tests (must stay green, unchanged). **Phase 33.**

### B2 — Extending the `ForwardModel` protocol
The new `ERPDCMForward` must implement the **existing** protocol
(`residual_is_complex=False`, `predict` returns a flat vector matching
`observed.reshape(-1)`, `build_precision` returns `([Q], 1)`) **without changing the
Protocol signature**. Adding required methods to the `Protocol` would break
`SpectralDCMForward`/`TaskDCMForward`/`LatentCircuitForward` (which are
`@runtime_checkable`-checked). **Prevention:** ERP-specific needs (lead field, `P.J`,
stimulus `R`) live as **constructor args / extra `**context`**, not new protocol methods.
The VL engine (`variational_laplace.py`) must not be edited for ERP. **Phase 36.**

### B3 — The FD-Jacobian flat-vector contract
The VL engine's finite-difference path calls `predict` with a **flat** `observed`
(see `LatentCircuitForward.predict`'s `observed.ndim>=2` guard). The ERP `predict` must
handle both the `(T, Nc)` main-loop call and the flat FD call without reshaping
incorrectly. **Prevention:** mirror the existing `observed.ndim` guard pattern exactly;
test `predict` under both call shapes. **Phase 36.**

### B4 — `pyproject.toml` / dependency drift
STACK.md: **zero new deps**. Do not bump torch, do not promote `mne` from optional.
Adding a dep to satisfy ERP would be a non-additive change to the environment. **Prevention:**
CI/lock check that `pyproject.toml` deps are unchanged. **Phase 33.**

---

## Validation-Methodology Pitfalls (vs-SPM, not vs-torch)

This is where Phase 32's hard-won lessons apply most directly.

### V1 — Self-referential tests (torch-vs-torch masquerading as parity)
**What goes wrong.** The most dangerous failure mode: writing tests that compare the torch
forward against *another torch computation* (e.g. "the integrator matches torchdiffeq",
"the difference wave is non-zero", "recovery of synthetic-from-torch params works"). All of
these can pass with a forward model that is *internally consistent but wrong vs SPM*.
Phase 32 is the cautionary tale — VL fit *its own* generated data perfectly yet diverged
from SPM because the forward models were not byte-identical.

**Warning signs:** every test passes but no test loads a MATLAB-exported array.

**Prevention:** **the parity gate must compare against frozen MATLAB fixtures** (`J0`,
`Q_update`, `f(x,u,P)` field, `spm_gen_Q` output, `spm_int_L` trajectory, `spm_lx_erp`
lead field, full scalp ERP) exported via the `validation/` `.mat` bridge. Recovery/round-trip
tests are *complements*, never the parity gate. Make "loads a `.mat` SPM fixture" a literal
checklist item for each parity test.

### V2 — Choosing the wrong parity statistic / absolute-F gate
**What goes wrong.** Phase 32 proved a **constant ~270-nat free-energy offset** between
engines from a fixed normalisation convention; the strict-5%-absolute-F gate was
**infeasible by construction**. Repeating that with ERP (e.g. gating on absolute F, or on
element-wise posterior covariance `Cp`) will fail for reasons that are *not* bugs.

**Warning signs:** a parity test fails by a *constant* additive offset across seeds/configs
(`std≈0`) — that is a convention difference, not an error.

**Prevention:** for **forward** parity, gate on **element-wise trajectory/lead-field/ERP
agreement** at the documented tolerance (the forward model has no normalisation freedom —
this is the strong, defensible gate). For any **inference** comparison, gate on
**ΔF / model ranking** and **posterior-mean Ep within tolerance**, **never absolute F,
never element-wise Cp** (the Phase-32 S3 rule). Document the expected constant-offset up front.

### V3 — Tolerance set too loose (or too tight) / not measured
**What goes wrong.** Picking `atol=1e-2` "to be safe" hides real divergence; picking `1e-14`
fails on legitimate `matrix_exp↔spm_expm` Padé differences. STACK.md flags the
`matrix_exp↔spm_expm` ~1e-12 figure as MEDIUM confidence.

**Prevention:** **measure** tolerances empirically from the standalone `matrix_exp↔spm_expm`
and `J0` tests, then set per-component tolerances as a *small multiple* of the measured
floor (e.g. `J0` ≤1e-10, `Q_update` ≤1e-9, trajectory ≤1e-8, scalp ERP ≤1e-7 to absorb
compounding). Record the measured value in the test, not just the threshold.

### V4 — Fixture drift / regeneration without provenance
**What goes wrong.** MATLAB fixtures regenerated with different `dt`, `ons`, SPM version, or
delays-on silently change; tests "pass" against the new wrong baseline.

**Prevention:** freeze fixtures with a **metadata header** (SPM version `$Id`, `dt`, `ns`,
`M.ons/dur`, `D` on/off, the exact `P` struct, MATLAB version). The Phase-32 runs used
MATLAB R2022a + Carrick `spm12` on M3 — pin the same. Check the fixture provenance into the
repo alongside the `.mat`. Treat fixture regeneration as a reviewed change.

### V5 — Compounding hides the source of divergence
**What goes wrong.** Only testing at the scalp ERP means a 1e-3 divergence could come from
*any* of sigmoid/permutation/integrator/lead-field; bisecting requires re-deriving each
stage.

**Prevention:** the **staged fixture ladder** — assert at *every* boundary in order:
`f(x,u,P)` field → `J0` → `Q_update` → single-source trajectory → `spm_gen_Q` Q →
multi-source trajectory → `spm_lx_erp` L → scalp ERP → difference wave. A failure localises
to one stage. This ladder *is* the recommended test architecture for the milestone.

---

## MMN-Demo Scientific Pitfalls

The demo must demonstrate the *real* precision→attenuation mechanism, not a fitting artifact.

### S1 — Precision effect is a fitting artifact, not the mechanism
**What goes wrong.** If the MMN sweep is produced by *fitting* and reading back a parameter,
a regularisation/identifiability artifact can masquerade as the precision effect. Even
forward-only, sweeping the *wrong* knob (C3a permutation bug → modulating `G(:,1)` instead
of `G(:,7)`) produces *some* monotone curve that looks like the result but isn't the
mechanism.

**Warning signs:** the `gain → |MMN|` curve exists but is insensitive to which connection
you label "precision"; or the attenuation direction reverses when you fix the permutation.

**Prevention:** the sweep must be a **pure forward** sweep of the *verified* superficial-
pyramidal self-inhibition `G(:,7)` (via `P.G(:,1)` *after* the permutation guard C3a, plus
`P.M`). Assert the swept parameter actually changes `G(:,7)` (reuse the C3a guard). The
mechanism direction is fixed by physiology (FEATURES.md §4): **increasing sp self-inhibition
→ lower superficial-pyramidal gain → down-weighted prediction errors → smaller deviant
response → attenuated MMN.** Assert *monotone attenuation* in that direction. **Phase 36.**

### S2 — Difference-wave sign
**What goes wrong.** The MMN is `deviant − standard`, a fronto-central **negativity**
(~100–250 ms). The sign depends on (a) the subtraction order, (b) the dipole moment sign
(C5.3), and (c) `P.J` state choice (C5.1). Any of these flips can produce a *positive*
"MMN", which is wrong.

**Warning signs:** difference wave peaks positive, or its sign flips when the dipole moment
sign flips (means the sign is not physically pinned).

**Prevention:** **pin the convention end-to-end**: subtraction order `deviant − standard`;
dipole-moment sign frozen in the fixture (C5); `P.J` = superficial pyramidal (C5.1). Assert
the canonical MMN is **negative-going** and **larger over frontal (rIFG) sources** than
purely sensory ones (FEATURES.md §5 acceptance criterion). Cross-check the *standard* and
*deviant* ERPs each match SPM before differencing (a correct difference of two correct
waves is the only trustworthy difference). **Phase 36.**

### S3 — Demo not anchored to a frozen-parameter SPM reference
**What goes wrong.** Showing a plausible MMN figure without a frozen-`(A,B,C,G,T,R)` SPM
comparison means the headline artifact is unvalidated — exactly the trap the milestone
exists to avoid.

**Prevention:** the **forward-parity acceptance** (FEATURES.md §5): at a fixed reference
parameter set, per-source LFPs *and* the scalp difference wave match `spm_gen_erp` +
`spm_lx_erp` within documented tolerance on frozen fixtures, *before* any sweep figure is
produced. The sweep is only credible once the single reference point is SPM-validated.
**Phase 36** (depends on the V5 ladder being green through Phase 35).

---

## Phase-Specific Warning Table (roadmap guards)

| Phase | Topic | Highest-priority pitfalls | Mandatory fixture/guard |
|-------|-------|---------------------------|--------------------------|
| **33** | CMC single-source forward + `spm_int_L` | **C1, C2, C3, M1, N1, N2, B1, B4** | `f(x,u,P)` field, `J0`, `Q_update`, single-source trajectory; `x0==0`; permutation guard `P.G(:,1)→G(:,7)`; `float64`; no eig-clip; `parameterize_A` untouched |
| **34** | extrinsic coupling + evoked input + `B` | **C4, M2, M4, N3, N4, C3(input/sign)** | `spm_gen_Q` Q fixture (A + diag→G); forward/backward/lateral adjacency table; lateral `(1+4L)`; `C`→ss only; delays-off in fixtures; `dt`/`ons` from metadata |
| **35** | lead field + scalp projection | **C5** | LFP-first (`L=diag(P.L)`); `spm_lx_erp` L fixture; `P.J`=state 3 guard; exported `G` post-`spm_cond_units`; pinned dipole sign; `kron` column-major order |
| **36** | model class + MMN demo + (opt.) inference | **S1, S2, S3, B2, B3, M3** | forward-parity at frozen ref (LFP + diff wave); monotone attenuation of verified `G(:,7)`; negative-going frontal MMN; protocol unchanged; FD flat-vector contract; AR(1) only if inference attempted |

---

## Sources

**Primary (HIGH confidence — SPM12 source read line-by-line at `../spm12/toolbox/dcm_meeg/`):**
- `spm_fx_cmc.m` (`$Id: 7279 2018-03-10`) — CMC state eqs, sigmoid, permutation `j`,
  extrinsic topology, modulatory gain.
- `spm_cmc_priors.m` (`$Id: 7279`) — log-normal transforms, 4 free G/T, priors.
- `spm_int_L.m` (`$Id: 7143 2017-07-29`) — exponential-Euler / frozen-Jacobian integrator,
  `(expm(dt·J)−I)/J`, `exp(−16)` regulariser.
- `spm_gen_erp.m` (`$Id: 6427`) + `spm_gen_Q.m` (`$Id: 7279`) — generation loop, steady
  state, condition modulation `B` on A and `diag(B)→G(:,1)`.
- `spm_erp_u.m` (`$Id: 7679`) — Gaussian evoked input, ms timebase, 32-scaling, sustained mix.
- `spm_lx_erp.m` (`$Id: 7256`) + `spm_erp_L.m` (`$Id: 7142`) + `spm_L_priors.m` (`$Id: 7409`)
  — lead field, `kron(J,L)`, ECD/LFP, `P.J` CMC default (state 3).
- Peer research: `.planning/research/v0.8.0/STACK.md`, `FEATURES.md`.
- `.planning/phases/32-spm12-cross-validation/32-SPM-CROSSVAL-FINDINGS.md` — constant
  270-nat F offset, systematic forward-model divergence, S3 absolute-F rule.

**Corroborating literature (MEDIUM confidence — community-known DCM-ERP reproduction gotchas):**
- van Wijk BCM et al. (2021/2020), *A fast and robust integrator of delay differential
  equations in DCM for electrophysiological data*, NeuroImage — confirms `spm_int_L` is
  Ozaki (1992) local linearisation, that integration error can reach ERP-signal amplitude
  for long delays, and that the **polynomial delay (`spm_dcm_delay`) is the default for CMC**
  (M2). https://www.sciencedirect.com/science/article/pii/S1053811921008405
- SPM M/EEG course, *DCM for evoked responses* (Auksztulewicz).
  https://www.fil.ion.ucl.ac.uk/spm/course/slides21-meeg/13_DCM_ERP.pdf
- SPM docs, *DCM for evoked responses*.
  https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/dcm/dcm_erp/
