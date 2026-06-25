# Research Summary: v0.8.0 DCM for Evoked Responses (CMC EEG/MEG ERP)

**Project:** Pyro-DCM
**Milestone:** v0.8.0 -- time-domain CMC neural-mass to evoked to single-dipole
lead-field to scalp-ERP forward stack, validated against SPM12
**Domain:** DCM for evoked responses (EEG/MEG ERP), canonical microcircuit (CMC),
mismatch negativity (MMN) precision-sweep demo
**Researched:** 2026-06-25
**Confidence:** HIGH

---

## Executive Summary

v0.8.0 adds a complete time-domain ERP forward stack to the Pyro-DCM framework under
the same SPM12-parity discipline used through Phase 32. The canonical microcircuit
(CMC) model (David & Friston 2003; Bastos et al. 2012) has four populations per
cortical source -- spiny stellate, superficial pyramidal, inhibitory interneurons, deep
pyramidal -- giving 8 states per source and a specific laminar extrinsic routing
(forward: sp->ss/dp; backward: dp->sp/ii). The milestone covers four phases: Phase 33
(CMC single-source dynamics + exponential-Euler integrator, parity vs spm_fx_cmc /
spm_int_L), Phase 34 (extrinsic A/B/C + evoked integration over a multi-source
network, parity vs spm_gen_erp), Phase 35 (single-dipole lead-field + scalp
projection, parity vs spm_lx_erp), and Phase 36 (erp_dcm_model.py Pyro model + VL /
amortized wiring + MMN precision-sweep demo as the hand-off artifact for the
downstream actinf_physics consumer). The headline scientific deliverable is a
five-source auditory MMN network (bilateral A1, bilateral STG, rIFG) that generates
the canonical deviant minus standard difference wave and produces a quantitative
precision -> MMN-amplitude attenuation curve by sweeping superficial-pyramidal
self-inhibition gain (G[:,7] in MATLAB / G[:,6] in 0-indexed Python), the Adams 2013 /
Ranlund 2016 aberrant-precision mechanism.

The single most important finding for every phase implementer: SPM12 does NOT
integrate ERPs with Runge-Kutta. spm_gen_erp.m calls spm_int_L.m, an
exponential-Euler / local-linearisation (Ozaki 1992) integrator with a FROZEN
Jacobian evaluated once at the expansion point x0 = zeros(n, 8) and reused for every
time bin. The update operator is Q = (matrix_exp(dt*D*J/N) - I)*inv(J), applied as
v += Q @ f(v, u) per step. torchdiffeq rk4/dopri5 converges to the true ODE solution
for small dt, but NOT to the spm_int_L solution for finite dt (SPM default dt = 4 ms
is not a small-dt regime). This integrator mismatch produces smooth, plausible, wrong
ERP traces that pass all NaN/shape tests and fail the parity gate by a growing
latency/amplitude drift that is insensitive to reducing ODE step size. The fix is a
new pure-torch module utils/local_linearization.py implementing the spm_int_L
algorithm using torch.matrix_exp and torch.linalg.solve. This file is the central new
component of the milestone; it must exist and be fixture-verified before any other ERP
work proceeds.

The stack requires zero new runtime dependencies. torch.matrix_exp and
torch.linalg.solve are confirmed present in the installed torch 2.9.1 and have been
stable API since well before the >=2.0 floor in pyproject.toml. The entire ERP
forward (CMC state equations, Gaussian-bump evoked input, lead-field projection) is
pure torch. The new ERPDCMForward class implements the existing ForwardModel protocol,
giving VL inference (run_variational_laplace_generic) and the amortized
packer/wrapper path as free reuse with zero engine edits. All existing
fMRI/spectral/rDCM/latent-circuit paths remain bit-exact; all new code is additive
(new files or new symbols appended to existing files, never editing existing class
bodies).

---

## Key Findings

### Stack

**Zero new runtime dependencies (HIGH confidence).** The full ERP forward stack needs
exactly two linear-algebra primitives beyond the fMRI/spectral path: torch.matrix_exp
and torch.linalg.solve. Both confirmed present in torch 2.9.1 (verified live). The
existing >=2.0 pin is sufficient; do not touch pyproject.toml. MNE-Python (already an
optional extra) stays optional and out-of-scope.

**One new integrator, pure torch (HIGH confidence).** spm_int_L.m (exponential-Euler,
frozen Jacobian) is the SPM integration scheme. The torch port pattern:

    J  = jacrev(f, x0)                            # frozen at x0 = zeros(n, 8)
    J  = J - eye(8*n, dtype=float64) * exp(-16)   # spm_int_L:126 regulariser
    E  = torch.matrix_exp(dt * D @ J / N)         # verified in torch 2.9.1
    Q  = torch.linalg.solve(J.T, (E - I).T).T     # (expm - I) @ inv(J), right-division
    v  = x0.reshape(-1)
    for i in range(ns):
        v = v + Q @ f(v, U[i])
        y[i] = g(v, U[i])

Use float64 throughout. N=1, D=identity for the first delay-free parity pass. Never
use torch.inverse; use linalg.solve(J.T,...).T for the right-division (CLAUDE.md rule).

torch.matrix_exp vs spm_expm agreement is ~1e-12 on the same input matrix (same
scaling-and-squaring + Pade algorithm family), but this is MEDIUM confidence on the
exact figure; Phase 33 must MEASURE it rather than assume it.

**CMC math transcribed from spm_fx_cmc.m line-by-line (HIGH confidence).** Key source
files: spm_fx_cmc.m (state equations, sigmoid, 10-intrinsic remap, extrinsic
topology), spm_cmc_priors.m (4 free G/T, log-normal transforms), spm_gen_erp.m +
spm_int_L.m (integration), spm_erp_u.m (Gaussian-bump, ms timebase, 32-scaling).
CMC steady state = zeros; no Newton solve needed.

**Sign-convention trap for extrinsic connections:** CMC uses +exp(P.A{i})*E(i) with
directional signs applied structurally in the equations of motion (forward +, backward
via -A{3}, -A{4}). The fMRI parameterize_A negates the diagonal via -exp/2. These are
different conventions. The CMC port requires a dedicated parameterize_cmc function;
importing parameterize_A is explicitly wrong.

**Citations flagged for Zotero (do not fabricate bib keys):**

| Suggested REF | Paper | In Zotero? |
|---------------|-------|-----------|
| REF-ERP-001 | David O, Friston KJ (2003). Neural mass model for MEG/EEG. NeuroImage 20:1743. | verify |
| REF-ERP-002 | Bastos AM et al. (2012). Canonical microcircuits for predictive coding. Neuron 76:695. | verify |
| REF-ERP-003 | Moran RJ, Pinotsis DA, Friston KJ (2013). Neural masses and fields in DCM. Front. Comp. Neurosci. 7:57. | verify |
| REF-ERP-004 | Pinotsis DA, Moran RJ, Friston KJ (2012). DCM with neural fields. NeuroImage 59:1261. | verify |
| REF-ERP-005 | Kiebel SJ, David O, Friston KJ (2006). DCM of evoked responses with lead fields. NeuroImage 30:1273. | verify |
| REF-ERP-006 | Friston K et al. (2007). Variational free energy and Laplace approximation. NeuroImage 34:220. | likely present |
| REF-MMN-001 | Adams RA et al. (2013). Computational anatomy of psychosis. Front. Psychiatry 4:47. | verify |
| REF-MMN-002 | Ranlund S et al. (2016). Impaired prefrontal synaptic gain in psychosis. Hum. Brain Mapp. 37:351. | verify |
| REF-MMN-003 | Garrido MI et al. (2009). The mismatch negativity: a review. Clin. Neurophysiol. 120:453. | verify |
| REF-MMN-004 | Garrido MI et al. (2007/2009). DCM of evoked potentials. NeuroImage. | verify |

---

### Features

**Table stakes (all required for a credible ERP-DCM):**

- T1: CMC population ODE, single source, 8 states (ss/sp/ii/dp voltage+conductance),
  sigmoid S(V) = 1/(1+exp(-Rx)) - 1/2 (the -1/2 baseline subtraction is load-bearing),
  second-order synaptic kernel, log-exp-scaled G/T/C; parameterize_cmc with permutation
  remap j=[7 2 3 4 1 5 6 8 9 10] (only 4 free G cols, only 4 free T).
- T2: Single-source SPM parity for spm_fx_cmc + spm_int_L on frozen MATLAB fixtures.
- T3: Extrinsic coupling A: forward (sp->ss / sp->dp, +), backward (dp->sp / dp->ii, -),
  lateral (reciprocal-reduced by 1/(1+4L) for each reciprocal pair).
- T4: Condition-specific modulation B: additive in log-space pre-exp on A{1..4} AND via
  diag(B)->G(:,1)->G(:,7) MATLAB / G(:,6) 0-indexed (sp self-inhibition = precision).
  This dual mechanism (spm_gen_Q.m:45-67) is the core of the MMN precision effect;
  omitting the diag->G path destroys the precision mechanism.
- T5: Input C + Gaussian stimulus u(t) (onset M.ons=60 ms, dispersion P.R, 32-scaled,
  enters spiny-stellate layer only); spm_erp_u port.
- T6: Evoked integration over peristimulus time (default dt=4ms, ns=128, ~512 ms) ->
  per-source, per-condition LFP timeseries.
- T7: Multi-source evoked parity vs spm_gen_erp for a reference A/B/C on the 5-source
  MMN graph.
- T8: Single-dipole-per-source lead field; kron(P.J, L_spatial) where CMC default P.J
  = state index 2 (superficial-pyramidal voltage, 0-indexed). LFP mode first (trivial
  diagonal), then ECD via MATLAB-exported gain.
- T9: Deviant minus standard difference wave (the MMN): negative-going, frontal peak.
- T10: erp_dcm_model.py Pyro model class with log-normal priors (A,B,C,G,T,R) and
  Gaussian likelihood on scalp residual.
- T11: Superficial-pyramidal self-inhibition G[:,6] / P.G[:,0] (via permutation remap)
  plus P.M exposed as named, sweepable precision parameter.
- T12: Precision-sweep MMN demo + transfer curve: monotone gain -> |MMN| attenuation.

**Differentiators (high-value, largely free reuse):**

- D1: VL posterior over CMC connectivity (zero engine edits; plug ERPDCMForward into
  existing run_variational_laplace_generic).
- D2: Amortized flow guide for ERP-DCM (reuse guides/amortized_flow.py + ERPDCMPacker).
- D3: Quantitative gain -> |MMN| transfer curve (directly consumable by actinf_physics
  consumer as a function, not just a figure).
- D4: Consumer-facing adapter API mapping (sp_inhibition_gain, a1_b_gain, rifg_b_gain,
  fwd_bwd_flag) -> CMC params (thin map for the Phase-133 adapter).
- D5: BMR model-comparison over MMN modulation hypotheses (forward-only vs backward-only
  vs intrinsic-gain): reuses existing BMR, no forward-stack change.
- D6: Circuit-explorer viz of the 5-source network with condition-modulated edges.

**Anti-features (out of scope this milestone):**

- Empirical ERP data fitting (real MMN recordings)
- Full sensor montage / BEM forward / FieldTrip head model in Python
- Source localization / inverse problem
- Group PEB / hierarchical between-subject modeling
- Jansen-Rit / ERP 3-population model (CMC only for precision mechanism access)
- CMC_2014 / thalamo-cortical / TFM variants
- Full delay differential-equation delay operator (D=1 for first parity pass)

**Canonical 5-source MMN network (Garrido/Ranlund -- verify coords before hard-coding):**

| Source | Abbrev | Approx MNI (mm) | Role |
|--------|--------|-----------------|------|
| Left primary auditory cortex | A1 L | (-42, -22, 7) | input-recipient (C drives here) |
| Right primary auditory cortex | A1 R | (46, -14, 8) | input-recipient (C drives here) |
| Left superior temporal gyrus | STG L | (-61, -32, 8) | mid-hierarchy |
| Right superior temporal gyrus | STG R | (59, -25, 8) | mid-hierarchy |
| Right inferior frontal gyrus | rIFG | (46, 20, 8) | top of hierarchy; key psychosis node |

**The precision mechanism (concrete, from spm_fx_cmc.m + spm_gen_Q.m):**
G[:,6] (0-indexed Python; G[:,7] MATLAB 1-indexed) is the sp->sp self-inhibition.
Free parameter P.G[:,0] maps to it via permutation remap j(1)=7. P.M adds
dp-firing-dependent modulation. Condition modulation B also reaches it via
diag(B)->G(:,1)->G(:,6). Increasing sp self-inhibition -> lower superficial-pyramidal
gain -> down-weighted prediction errors -> attenuated deviant response -> attenuated
MMN. This is Adams 2013 aberrant-precision and Ranlund 2016: impaired sp self-
inhibition gain in rIFG and bilateral A1 in psychosis and relatives.

---

### Architecture

**The integration seam already exists.** The repo has a @runtime_checkable
ForwardModel Protocol in src/pyro_dcm/inference/forward_models.py:30-117. The VL core
(_run_vl_generic / run_variational_laplace_generic) dispatches only via this
protocol's eight members. ERPDCMForward is the fourth implementor (after
SpectralDCMForward, TaskDCMForward, LatentCircuitForward), following the exact
precedent set by the v0.6.0 LatentCircuitForward addition.

**Protocol contract for ERPDCMForward:**

| Member | ERP return |
|--------|-----------|
| residual_is_complex | False (time-domain real residuals) |
| param_count(n_regions) | A{1..4} + C + T(4) + G(4) + S + optional M + L + J + R |
| pack_params / unpack_params | flat vector; ordering frozen for parity |
| build_prior_cov | diagonal prior variances from spm_cmc_priors.m |
| build_precision | ([identity(Cnd*ns*Nc)], 1) v1; AR(1) spm_Q deferred |
| predict | y.reshape(-1) where y shape (Cnd, ns, Nc) |
| build_result | {"theta_post": ..., "predicted_output": ...} |

ERP-specific needs (lead field, P.J, stimulus R, condition design X, dt, spatial model
flag) ride as constructor args and context dict -- no new protocol methods, no engine
edits.

**New files (8 pure additions, zero backward-compat risk):**

| File | Purpose | SPM source |
|------|---------|-----------|
| src/pyro_dcm/forward_models/cmc_neural_mass.py | CMC state eqs f(x,u,P); sigmoid S(V); parameterize_cmc with perm j | spm_fx_cmc.m |
| src/pyro_dcm/forward_models/cmc_priors.py | Prior means/variances + transform tables | spm_cmc_priors.m |
| src/pyro_dcm/forward_models/erp_input.py | Gaussian-bump evoked drive u(t) (onset, dispersion, 32-scale) | spm_erp_u.m |
| src/pyro_dcm/utils/local_linearization.py | spm_int_L port: frozen-Jacobian exp-Euler; solve not inverse; float64 | spm_int_L.m |
| src/pyro_dcm/forward_models/erp_coupled_system.py | Network: extrinsic A fwd/bwd/lateral + per-condition B (spm_gen_Q) | spm_fx_cmc.m:68-82, spm_gen_Q.m |
| src/pyro_dcm/forward_models/erp_leadfield.py | kron(P.J, L_spatial) state expansion; LFP diag / ECD G*P.L; scalp proj | spm_lx_erp.m, spm_erp_L.m |
| src/pyro_dcm/simulators/erp_simulator.py | simulate_erp_dcm(...): per-condition scalp ERP + difference wave | composes the above |
| src/pyro_dcm/models/erp_dcm_model.py | Pyro generative model; log-normal priors; Gaussian likelihood | spm_cmc_priors.m, spm_dcm_erp.m |
| scripts/demo_mmn_precision_sweep.py | 5-source MMN; sweep sp self-inhibition; gain->|MMN| transfer curve | FEATURES section 5 |

**Modified files (additive only -- existing class bodies untouched):**

| File | Addition | Backward-compat |
|------|----------|----------------|
| src/pyro_dcm/inference/forward_models.py | class ERPDCMForward appended | Existing 3 forward classes unchanged |
| src/pyro_dcm/guides/parameter_packing.py | class ERPDCMPacker appended | Existing packers untouched |
| src/pyro_dcm/models/amortized_wrappers.py | amortized_erp_dcm_model + _run_erp_forward_model appended | Existing wrappers untouched |
| src/pyro_dcm/validation/export_to_mat.py | export_erp_dcm(...) appended | Existing exporters untouched |
| */__init__.py (forward_models, models, simulators, guides) | new exports only | re-export only |
| .planning/REFERENCES.md | REF-ERP-001..006 + REF-MMN-001..004 appended | after Zotero collation |

Hard rule honored: ode_integrator.py, coupled_system.py, neural_state.py, all
spectral/rDCM/task forward models and model classes, and the VL engine core are NEVER
edited. Verifiable by git diff showing only insertions in the four modified .py files.

**Tensor shape boundary table (N=sources, ns=time samples, Nc=channels, Cnd=conditions):**

| Stage | Tensor | Shape | dtype | Notes |
|-------|--------|-------|-------|-------|
| CMC state per source | x_cmc | (N, 8) | f64 | cols = ss/sp/ii/dp x(V,I) |
| Flattened state | x_flat | (8N,) | f64 | column-major (spm_vec) -- parity-critical |
| Evoked input | u_erp | (ns, n_inp) | f64 | spm_erp_u, 32-scaled |
| Frozen Jacobian | J | (8N, 8N) | f64 | jacrev(f) at x0, -I*exp(-16) reg |
| Update operator | Q | (8N, 8N) | f64 | (matrix_exp(dt*D*J/N)-I)*inv(J) via solve |
| State trajectory per cond | states_c | (ns, 8N) | f64 | one per condition |
| Spatial lead field | L_spatial | (Nc, N) ECD / (N, N) LFP | f64 | from spm_erp_L / exported gain |
| State->dipole weights | J_contrib (P.J) | (8,) | f64 | CMC default: state index 2 (sp V) |
| Full lead field | L_full | (Nc, 8N) | f64 | kron(J_contrib, L_spatial) |
| Scalp ERP per cond | y_c | (ns, Nc) | f64 | (states_c - x0) @ L_full.T |
| Observed / predicted | y | (Cnd, ns, Nc) | f64 | predict flattens to (Cnd*ns*Nc,) |
| MMN difference wave | mmn | (ns, Nc) | f64 | deviant minus standard |

SPM validation bridge: Parity fixtures follow the Phase-32 pattern: Python exports a
.mat (additive to export_to_mat.py) -> MATLAB script runs SPM reference on M3 (R2022a
+ Carrick spm12) -> Python asserts in tests/test_spm_erp_dcm_validation.py. The ECD
lead-field gain G(:,:,i) (post-spm_cond_units) is precomputed in MATLAB and exported
via this bridge; Python reproduces only kron(P.J, L) + projection.

New validation / test files:

| File | Purpose |
|------|---------|
| validation/matlab_scripts/run_spm_erp_dcm.m | Run spm_gen_erp/spm_lx_erp reference on M3 from exported .mat |
| tests/test_spm_erp_dcm_validation.py | Parity assertions (single-source, multi-source, scalp) |
| cluster/scripts/erp_cross_validation.py | M3 entrypoint, mirrors spm_cross_validation.py |
| tests/test_cmc_forward.py | Per-transform perturbation test; permutation guard |
| tests/test_local_linearization.py | matrix_exp vs spm_expm tolerance measurement |
| tests/test_erp_leadfield.py | P.J default guard; kron column-major order check |
| tests/test_erp_dcm_recovery.py | VL round-trip on synthetic ground truth |

---

### Pitfalls

**Five CRITICAL pitfalls** (silently pass smoke tests; fail SPM parity; must each
become an explicit fixture-backed gate before phase completion):

**C1 -- Integration-scheme mismatch (the headline risk, Phase 33).** RK4/dopri5 and
spm_int_L are different algorithms. Using torchdiffeq for ERP produces smooth,
plausible-looking traces that diverge from SPM by a growing drift over the peristimulus
window; no step-size reduction fixes it. Detection: per-source LFP has correct shape
but ~1-5% amplitude/latency drift that grows toward the end of the window, insensitive
to ODE step reduction. Prevention: implement utils/local_linearization.py as the
spm_int_L port; do NOT route CMC through integrate_ode. Write this test FIRST in
Phase 33: export three frozen MATLAB arrays from a single-source reference config:
(1) J0 (frozen Jacobian at x0=0), (2) Q_update (update operator), (3) y_states (full
trajectory for a known Gaussian u). Assert torch reproduces J0 (<=1e-10), Q_update
(<=1e-9), y_states (<=1e-8) before any extrinsic coupling exists.

**C2 -- (expm - I)/J right-division and spm_expm agreement (Phase 33).** MATLAB
(spm_expm(...) - I)/dfdx is right-division = (E - I)*inv(J), NOT inv(J)*(E-I). J is
not symmetric for CMC. A naive port writes inv(J) @ (E-I) and silently gets a wrong
transposed-ish operator. Implement as torch.linalg.solve(J.T, (E-I).T).T. The
exp(-16) Jacobian regulariser must be applied BEFORE forming Q. Measure (not assume)
matrix_exp vs spm_expm tolerance on a frozen exported dt*J0 array.

**C3 -- Parameter-transform / units / permutation traps (Phase 33/34).** Eight sub-
traps, each yielding a finite, smooth, plausible ERP that passes NaN/shape tests:
(a) Intrinsic permutation j: P.G[:,0] -> G[:,6] (sp self-inhibition), NOT G[:,0].
(b) Only 4 free G cols, 4 free T -- not 10 or 8.
(c) Time-constant units: T=[2,2,16,28] ms -> /1000 to seconds, then *exp(P.T).
(d) Sigmoid baseline subtraction: S = 1/(1+exp(-Rx)) - 1/2; the -1/2 is load-bearing.
(e) Baseline slope: R=2/3 THEN R*=exp(P.S); not R=1.
(f) Extrinsic sign convention: +exp(P.A)*E(i) with structural signs; NOT fMRI -exp/2.
(g) Input scaling chain: C=exp(P.C); exogenous branch U=C*u*32; u is already 32*Gaussian.
(h) State flattening: column-major (Fortran-order, state-blocked); NOT C-order.
Prevention: per-transform unit test perturbing each free param one at a time, asserting
f(x,u,P) derivative field matches MATLAB element-wise (<=1e-10). Critically, perturb
P.G[:,0] and assert G[:,6] (not G[:,0]) changes -- a direct guard for the permutation.

**C4 -- Extrinsic coupling and condition-modulation B wiring (Phase 34).** Forward
connections originate from superficial-pyramidal firing S(:,3); backward from deep
S(:,7). Lateral/reciprocal connections require 1/(1+4L) reduction factor. Input C
enters spiny-stellate only. The B modulation is additive in log-space pre-exp on ALL
A{1..4} AND via diag(B)->G(:,1)->G(:,6) (precision); omitting the diag->G path
destroys the MMN precision mechanism. Prevention: export spm_gen_Q output for a known
P, X=[1] (deviant) and assert torch reproduces Q.A{1..4} AND Q.G(:,1) element-wise.

**C5 -- Lead-field / kron(J,L) and single-dipole traps (Phase 35).** CMC default P.J
= state index 2 (superficial-pyramidal voltage, 0-indexed); observing deep pyramidal
(index 6) produces a physiologically-inverted EEG. kron(J,L) must match the column-
major state-blocked flatten (C3h). For ECD: exported G(:,:,i) must include
spm_cond_units rescale. Prevention: start with LFP spatial model (trivial diagonal)
for Phase 35 first parity; add ECD only after LFP parity confirmed. Hard-code and
assert P.J default = unit at state index 2.

**Five moderate pitfalls:**

- M1: CMC steady state = zeros; do not implement Newton solver (assert x0==zeros with
  citation to spm_dcm_neural_x.m otherwise branch). Phase 33.
- M2: Delay operator silently nonzero: SPM polynomial delay is the CMC default. Generate
  fixtures with delays explicitly disabled; assert D=1 in the MATLAB script. Add
  spm_dcm_delay port only after delay-free parity confirmed. Phase 34.
- M3: AR(1) precision vs identity: identity v1 is correct for forward-only scope; if
  inference/F parity attempted, add spm_Q(1/2,Ns) AR(1) precision basis. Phase 36.
- M4: dt mismatch: thread dt/ns/M.ons/M.dur as single constants from fixture metadata;
  keep spm_erp_u time in ms internally. Phase 33/34.
- N1/N2: float32 creep (drops parity to ~1e-4; assert dtype==float64 at ForwardModel
  boundary) and eigenvalue-clipping reflex (the fMRI eig-clip rule does NOT apply to the
  CMC Jacobian; use the exp(-16) shift only). Phase 33.

**Four backward-compatibility pitfalls:**

- B1: Do NOT edit parameterize_A for CMC; add parameterize_cmc_extrinsic as a new
  standalone function. Existing spectral/task recovery tests must remain green.
- B2: Do NOT add required methods to the ForwardModel Protocol; ERP-specific needs ride
  as constructor args + context.
- B3: ERP predict must handle both (Cnd, ns, Nc) call and flat FD call shapes; mirror
  the LatentCircuitForward observed.ndim guard.
- B4: Zero new deps; do not bump torch, do not promote MNE from optional.

**Validation-methodology pitfalls (Phase-32 lessons applied directly):**

- V1: Self-referential torch-vs-torch tests are not parity gates. Every phase gate must
  compare against frozen MATLAB-exported arrays.
- V2: Do not gate on absolute free energy (Phase-32 proved ~270-nat constant offset).
  For forward parity: element-wise trajectory/ERP agreement. For inference parity:
  delta-F / model ranking; never absolute F or element-wise Cp.
- V3: Measure tolerances empirically from J0 and Q_update fixtures. Suggested tiers:
  J0 <=1e-10, Q_update <=1e-9, single-source trajectory <=1e-8, scalp ERP <=1e-7.
- V4: Freeze fixture metadata header (SPM $Id, dt, ns, M.ons/dur, D on/off, exact P
  struct, MATLAB R2022a + Carrick spm12). Treat fixture regeneration as reviewed change.
- V5: Staged fixture ladder -- assert at every boundary in order: f(x,u,P) field -> J0
  -> Q_update -> single-source trajectory -> spm_gen_Q Q -> multi-source trajectory ->
  spm_lx_erp L -> scalp ERP -> difference wave. Testing only at scalp makes bisection
  intractable.

**MMN demo scientific pitfalls:**

- S1: Sweep must be pure-forward of verified G[:,6] (after permutation guard). Assert
  P.G[:,0] perturbation changes G[:,6], not G[:,0].
- S2: MMN sign must be pinned end-to-end: subtraction order deviant-standard; dipole-
  moment sign frozen in fixture; P.J = sp voltage (state index 2). Assert negative-going
  and frontal dominance.
- S3: The sweep figure is only credible once the fixed-reference SPM parity gate
  (LFP + diff wave match spm_gen_erp/spm_lx_erp) is green.

---

## Implications for Roadmap

Phase numbering continues from 32. v0.8.0 covers Phases 33-36.

### Suggested Phase Structure

---

#### Phase 33 -- CMC core dynamics + exp-Euler integrator + single-source parity (T1, T2)

**Rationale:** The absolute foundation. Every downstream phase requires a verified
single-source CMC forward and a verified spm_int_L port. The integration-scheme
mismatch (C1) is the most dangerous pitfall in the milestone; it must be caught in
isolation before extrinsic coupling compounds it. Phase 32 proved that small forward
differences compound silently.

**Delivers:**
- forward_models/cmc_neural_mass.py -- CMC state equations, sigmoid, parameterize_cmc
  with permutation remap j
- forward_models/cmc_priors.py -- prior means/variances from spm_cmc_priors.m
- forward_models/erp_input.py -- Gaussian-bump evoked drive
- utils/local_linearization.py -- spm_int_L port (frozen-Jacobian exp-Euler, solve not
  inverse, float64)
- validation/export_to_mat.py additive: export_erp_dcm (single-source config)
- validation/matlab_scripts/run_spm_erp_dcm.m (single-source, D=1 explicit)
- tests/test_cmc_forward.py -- per-transform perturbation test; permutation guard
- tests/test_local_linearization.py -- matrix_exp vs spm_expm tolerance measurement
- tests/test_spm_erp_dcm_validation.py (single-source: J0, Q_update, y_states assertions)

**Features:** T1, T2

**Must-avoid pitfalls:** C1, C2, C3 (all sub-traps), M1, M4, N1, N2, B1, B4

**Parity gate:** Single-source spm_fx_cmc + spm_int_L on frozen MATLAB fixture. Assert
sequentially: (1) f(x,u,P) derivative field <=1e-10 vs MATLAB, (2) J0 <=1e-10,
(3) Q_update <=1e-9 (MEASURES matrix_exp vs spm_expm tolerance; do not assume 1e-12),
(4) full state trajectory y_states (ns,8) <=1e-8. Fixture metadata must include D=1
assertion and x0==0 check.

**Additive guarantee:** all-new files; touches only export_to_mat.py (append).

**Research flag:** The matrix_exp vs spm_expm tolerance is MEDIUM confidence; the
test_local_linearization.py test must measure it empirically and record the measured
floor. This sets the tolerance floor for all downstream phases.

---

#### Phase 34 -- Extrinsic coupling + condition B + evoked integration + multi-source parity (T3-T7)

**Rationale:** Builds the hierarchical network on the Phase-33 verified foundation.
Introduces the most complex coupling logic: CMC laminar routing rules, lateral reduction
factor, and the dual B-modulation mechanism (extrinsic + diag->G = the precision path).
The delay operator must be explicitly disabled in all fixture-generation scripts (M2 --
polynomial delay is the CMC default in normal SPM paths).

**Delivers:**
- forward_models/erp_coupled_system.py -- extrinsic A{1..4} fwd/bwd/lateral routing +
  per-condition B via spm_gen_Q; C input enters ss only; lateral (1+4L) reduction
- simulators/erp_simulator.py -- simulate_erp_dcm(...) returning per-condition ERP dict
- Extension of tests/test_spm_erp_dcm_validation.py -- multi-source assertions vs
  spm_gen_erp
- cluster/scripts/erp_cross_validation.py -- M3 entrypoint for multi-source parity jobs

**Features:** T3, T4, T5, T6, T7

**Must-avoid pitfalls:** C4, M2, M4, N3, N4, C3 sub-traps f/g (extrinsic sign, input
scaling)

**Parity gate:** (1) spm_gen_Q fixture: torch Q.A{1..4} and Q.G[:,0] match exported
MATLAB values element-wise -- the critical B-wiring guard. (2) Multi-source evoked
trajectory vs spm_gen_erp for the 5-source MMN reference A/B/C on frozen fixture
(delays explicitly off). (3) C->ss-only assertion. (4) Lateral (1+4L) reduction
triggers on a reciprocal test pair.

**Additive guarantee:** all-new modules.

**Research flag:** Confirm the fixture-generation MATLAB script can explicitly disable
delays before the first fixture run. If delay-free parity is insufficient, gate
spm_dcm_delay port as a separate sub-task with re-exported fixtures.

---

#### Phase 35 -- Lead field + scalp projection + difference wave + parity vs spm_lx_erp (T8, T9, T10, D1)

**Rationale:** Turns per-source LFPs into the observed scalp ERP and produces the MMN
difference wave. The P.J default (state index 2 = sp voltage, not dp voltage) and the
kron(J,L) column-major ordering are the most dangerous traps here (C5). Start with LFP
spatial model (trivial diagonal) to isolate dynamics from lead-field physics.

**Delivers:**
- forward_models/erp_leadfield.py -- L_spatial (LFP diag / ECD G*P.L), kron(P.J,
  L_spatial), projection y = (x - x0) @ L_full.T
- models/erp_dcm_model.py (VL adapter version: forward + ERPDCMForward wiring)
- Additive insertion into inference/forward_models.py: class ERPDCMForward
- tests/test_erp_leadfield.py -- P.J default guard; kron column-major order check
- tests/test_erp_dcm_recovery.py -- VL round-trip on synthetic ground truth
- MATLAB fixtures: spm_lx_erp LFP lead field L (Nc, 8n); ECD gain G(:,:,i) post-
  spm_cond_units exported separately via .mat bridge

**Features:** T8, T9, T10 (VL adapter), D1

**Must-avoid pitfalls:** C5, B2, B3

**Parity gate:** (1) LFP lead field: scalp ERP matches spm_gen_erp + spm_lx_erp
(LFP mode) within <=1e-7. (2) P.J guard: assert observed state is index 2 (sp V), not
index 6 (dp V). (3) kron column-major order verified against exported L_full (Nc, 8n)
fixture. (4) Difference wave is non-zero and negative-going. (5) VL recovers planted
CMC params via run_variational_laplace_generic(ERPDCMForward(), ...) on synthetic
ground truth (confirms protocol; not a parity gate).

**Research flag:** Lock observation stacking layout (Cnd, ns, Nc) vs (Cnd*ns, Nc) in
predict / build_precision / MATLAB .mat output before writing scalp-ERP parity
assertions.

---

#### Phase 36 -- Pyro model class + precision-sweep demo + amortized wiring + transfer curve (T10-T12, D1-D4)

**Rationale:** Completes the milestone and produces the actinf_physics hand-off
artifact. The precision-sweep demo is only credible once the Phase 35 parity gate is
green. The amortized path (D2) and adapter API (D4) are near-free reuse of existing
infrastructure. BMR model comparison (D5) is deferred if time-boxed.

**Delivers:**
- models/erp_dcm_model.py (full Pyro generative model: log-normal priors, Gaussian
  likelihood, B modulation)
- Additive insertions: guides/parameter_packing.py (ERPDCMPacker), models/
  amortized_wrappers.py (amortized_erp_dcm_model, _run_erp_forward_model)
- scripts/demo_mmn_precision_sweep.py -- 5-source MMN; sweep G[:,6] (via P.G[:,0]) at
  rIFG and bilateral A1; gain->|MMN| transfer curve; overlay at low/baseline/high gain
- Consumer adapter API (D4): maps (sp_inhibition_gain, a1_b_gain, rifg_b_gain,
  fwd_bwd_flag) -> CMC params

**Features:** T10 (full), T11, T12, D1, D2, D3, D4; optional D5 (BMR)

**Must-avoid pitfalls:** S1, S2, S3, B2, B3, M3

**Parity gate and demo acceptance criteria:**
1. At fixed reference (A,B,C,G,T,R), per-source LFPs and scalp difference wave match
   spm_gen_erp + spm_lx_erp (LFP mode) within Phase 35 tolerances on frozen fixtures.
   This gate must be green BEFORE producing any sweep figure.
2. Sweep is a pure-forward sweep of verified G[:,6]; assert P.G[:,0] perturbation
   changes G[:,6] (reuse Phase 33 permutation guard).
3. gain->|MMN| curve is monotone-decreasing (increasing sp self-inhibition -> attenuated
   MMN).
4. Difference wave is negative-going and larger over frontal (rIFG) sources than
   purely sensory ones.
5. Amortized guide trains on erp_simulator draws without errors.

**Research flag:** If inference comparison against SPM is attempted, add
spm_Q(1/2,Ns) AR(1) precision basis and re-validate; document that absolute-F is not
a valid criterion per V2 and Phase-32 findings.

---

### Critical Path

The critical path has no phase-level parallelism because each phase contributes a tier
of the V5 staged fixture ladder:

    Phase 33: spm_int_L port + single-source parity (J0, Q_update, y_states gates)
        |
        v
    Phase 34: extrinsic coupling + B modulation + multi-source evoked integration
              (spm_gen_Q fixture Q.A/Q.G; multi-source trajectory gates)
        |
        v
    Phase 35: lead field + scalp projection + difference wave
              (spm_lx_erp L fixture; LFP scalp ERP; negative-going diff wave)
        |
        v
    Phase 36: Pyro model + precision sweep demo
              (frozen-ref SPM match before sweep; monotone attenuation; frontal negativity)

The M3 cluster is required for all MATLAB fixture-generation jobs (R2022a + Carrick
spm12). Python unit tests that are fast (single-source forward, integrator unit tests)
may run locally; any multi-source integration sweep projected >3 min routes to M3.

**Per-phase parity gate summary:**

| Phase | SPM reference | Tolerances | Key fixture arrays |
|-------|--------------|------------|--------------------|
| 33 | spm_fx_cmc + spm_int_L (single source, D=1) | J0 <=1e-10, Q <=1e-9, traj <=1e-8 | J0 (8,8), Q_update (8,8), y_states (ns,8) |
| 34 | spm_gen_Q + spm_gen_erp (5-source, D=1) | Q.A/Q.G element-wise, traj <=1e-8 | spm_gen_Q Q struct, network trajectory (ns,5,8) |
| 35 | spm_lx_erp LFP mode | scalp ERP <=1e-7 | L_full (Nc, 8n), y_c (ns, Nc) per condition |
| 36 | full pipeline at fixed ref params (LFP) | same as Phase 35 | repeat Phase 35 frozen-ref + monotone curve |

**Five mandatory guard tests (cannot be skipped):**

1. Phase 33: test_spm_erp_dcm_validation.py single-source -- loads MATLAB J0/Q/trajectory
   fixtures; asserts element-wise agreement at documented tolerances.
2. Phase 33: test_cmc_forward.py per-transform perturbation -- perturbs P.G[:,0], asserts
   G[:,6] (not G[:,0]) changes; confirms the precision-mechanism connection.
3. Phase 34: spm_gen_Q fixture assertion -- torch Q.A{1..4} and Q.G[:,0] match exported
   MATLAB values element-wise.
4. Phase 35: P.J default guard (test_erp_leadfield.py) -- assert state index 2 (sp V),
   not 6 (dp V); kron column-major order check vs exported L_full.
5. Phase 36: frozen-ref SPM forward parity (LFP + diff wave green) before demo runs;
   monotone attenuation assertion in sweep.

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Stack (zero deps, spm_int_L algorithm, matrix_exp availability) | HIGH | torch 2.9.1 primitives verified live; spm_int_L.m read line-by-line with line refs |
| CMC math (state equations, sigmoid, permutation, transforms) | HIGH | spm_fx_cmc.m + spm_cmc_priors.m transcribed with line refs in STACK.md section 2 |
| Integration scheme (torchdiffeq unsuitability) | HIGH | structural: different algorithm family; SPM dt=4 ms not in small-dt regime |
| matrix_exp vs spm_expm exact tolerance | MEDIUM | same algorithm family HIGH; exact bound must be measured by Phase 33 test |
| Features (table stakes, precision mechanism, B dual-path) | HIGH | spm_gen_Q.m + spm_fx_cmc.m read; precision traced to G[:,7] in source |
| Architecture (protocol reuse, additive seam, file list) | HIGH | forward_models.py:30-117 + _run_vl_generic read; v0.6.0 LatentCircuitForward precedent |
| Pitfalls (C1-C5, V1-V5) | HIGH | every claim traced to SPM12 source line-by-line or Phase-32 empirical finding |
| MNI source coordinates | MEDIUM | commonly cited Garrido/Ranlund values; must verify vs primary papers before hard-coding |
| Delay operator default status | MEDIUM | van Wijk et al. 2021 confirms polynomial delay is CMC default; needs explicit verification in fixture script |
| Citation keys | N/A | flagged for manual Zotero addition; never fabricated |

**Overall confidence: HIGH**

### Gaps to Address During Planning

1. **matrix_exp vs spm_expm tolerance (gap 1, Phase 33):** The STACK.md ~1e-12 figure is
   MEDIUM confidence. Phase 33 must export a frozen dt*J0 from MATLAB, compute
   spm_expm(dt*J0) in MATLAB, and assert against torch.matrix_exp(dt*J0) to measure the
   actual tolerance floor. This becomes the basis for the per-component tolerance table
   used by all downstream phases.

2. **MNI source coordinates (gap 2, Phase 34/36):** The 5-source network coordinates are
   the commonly cited Garrido/Ranlund values. Verify exact mm against primary papers
   (REF-MMN-003/REF-MMN-004) before hard-coding into fixtures. Flag for Zotero addition;
   do not hard-code from memory.

3. **Delay operator default status in fixture scripts (gap 3, Phase 34):** Confirm the
   MATLAB fixture-generation script can set P.D=0 (or equivalent) and verify
   spm_dcm_delay returns D=1 for those parameters before the first fixture run. Document
   in fixture metadata header. If the normal spm_dcm_erp path cannot easily disable
   delays, investigate directly in spm_gen_erp.m / spm_int_L.m.

4. **Observation stacking layout (gap 4, Phase 35):** Lock the (Cnd, ns, Nc) stacking
   order in predict / build_precision / MATLAB .mat output as part of Phase 35 planning
   before writing parity assertions. Misalignment produces a clean-looking but wrong
   residual structure.

5. **Zotero citations (gap 5, before any REF-xxx entries):** REF-ERP-001 to REF-ERP-006
   and REF-MMN-001 to REF-MMN-004 must be confirmed in the project Zotero folder. Do not
   fabricate Better BibTeX keys; do not add REF-xxx to .planning/REFERENCES.md or
   docstrings until the paper is in Zotero.

---

## Sources

### Primary (HIGH confidence -- SPM12 source read line-by-line at ../spm12/toolbox/dcm_meeg/)

- spm_fx_cmc.m ($Id: 7279 2018-03-10) -- CMC state equations, sigmoid, permutation j,
  extrinsic topology, modulatory gain
- spm_cmc_priors.m -- log-normal transforms, 4 free G/T, priors
- spm_int_L.m ($Id: 7143 2017-07-29) -- exponential-Euler / frozen-Jacobian integrator
- spm_gen_erp.m ($Id: 6427) + spm_gen_Q.m ($Id: 7279) -- evoked generation loop,
  steady state, B-modulation on A and diag->G(:,1)
- spm_erp_u.m ($Id: 7679) -- Gaussian evoked input, ms timebase, 32-scaling
- spm_lx_erp.m ($Id: 7256) + spm_erp_L.m ($Id: 7142) + spm_L_priors.m ($Id: 7409) --
  lead field, kron(J,L), ECD/LFP, P.J CMC default (state 3)
- spm_dcm_neural_x.m -- steady-state logic (CMC hits otherwise branch -> zeros)
- .planning/phases/32-spm12-cross-validation/32-SPM-CROSSVAL-FINDINGS.md -- constant
  270-nat F offset, systematic divergence, V2/S3 absolute-F rule

### Primary (HIGH confidence -- codebase read)

- src/pyro_dcm/inference/forward_models.py:30-117 -- ForwardModel Protocol;
  _run_vl_generic dispatch
- src/pyro_dcm/inference/variational_laplace.py -- VL engine;
  run_variational_laplace_generic
- src/pyro_dcm/forward_models/neural_state.py:24 -- parameterize_A fMRI sign
  convention; confirms CMC must NOT reuse it
- src/pyro_dcm/guides/parameter_packing.py -- packer idiom; three existing packers
- src/pyro_dcm/models/amortized_wrappers.py -- wrapper idiom; _sample_latent_and_unpack
- src/pyro_dcm/utils/ode_integrator.py -- torchdiffeq wrapper; confirms must not be
  modified for ERP
- src/pyro_dcm/validation/export_to_mat.py + validation/matlab_scripts/ +
  cluster/scripts/spm_cross_validation.py -- Phase-32 SPM bridge pattern
- torch 2.9.1+cpu runtime -- live verification of matrix_exp, linalg.solve

### Secondary (MEDIUM confidence -- corroborating literature)

- van Wijk BCM et al. (2021). A fast and robust integrator of delay differential
  equations in DCM for electrophysiological data. NeuroImage. -- confirms spm_int_L is
  Ozaki (1992) local linearisation; polynomial delay is CMC default.
- Adams RA et al. (2013). The computational anatomy of psychosis. Front. Psychiatry
  4:47. [flag for Zotero: REF-MMN-001]
- Ranlund S et al. (2016). Impaired prefrontal synaptic gain in psychosis. Hum. Brain
  Mapp. 37:351. [flag for Zotero: REF-MMN-002]
- Garrido MI et al. (2009). The mismatch negativity: a review. Clin. Neurophysiol.
  120:453. [flag for Zotero: REF-MMN-003]
- Garrido MI et al. (2007/2009). DCM of evoked potentials. NeuroImage. [flag for
  Zotero: REF-MMN-004]
- Bastos AM et al. (2012). Canonical microcircuits for predictive coding. Neuron 76:695.
  [flag for Zotero: REF-ERP-002]
- David O, Friston KJ (2003). A neural mass model for MEG/EEG. NeuroImage 20:1743.
  [flag for Zotero: REF-ERP-001]

---

*Research completed: 2026-06-25*
*Ready for roadmap: yes*
