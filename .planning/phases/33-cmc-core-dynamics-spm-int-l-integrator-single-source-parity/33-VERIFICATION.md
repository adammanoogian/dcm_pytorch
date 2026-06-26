---
phase: 33-cmc-core-dynamics-spm-int-l-integrator-single-source-parity
verified: 2026-06-26T09:22:12Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 33: CMC Core Dynamics, spm_int_L Integrator and Single-Source Parity -- Verification Report

**Phase Goal:** A verified single-source CMC forward (4 populations, 8 states) and a verified
spm_int_L exponential-Euler integrator exist, with single-source SPM12 parity proven on frozen
MATLAB fixtures -- catching the rk4 vs spm_int_L integration-scheme mismatch in isolation before
any extrinsic coupling can compound it.

**Verified:** 2026-06-26T09:22:12Z
**Status:** PASSED
**Re-verification:** No -- initial verification

---

## Test Run Output

    python -m pytest tests/test_local_linearization.py tests/test_cmc_forward.py tests/test_spm_erp_dcm_validation.py -q

Result: **25 passed in 8.50s** (7 integrator + 9 CMC forward + 9 parity ladder)

The parity ladder RUNS on the laptop (keyed on fixture availability, not MATLAB availability).
Decision 33-03-D1 documents this explicitly.

---

## Goal Achievement

### Observable Truths

| Num | Truth | Status | Evidence |
|-----|-------|--------|----------|
| 1 | P_G col 0 changes G col 6 not G col 0 (permutation guard) | VERIFIED | test_permutation_guard passes; direct Python check confirms |
| 2 | sigmoid -1/2 baseline; R=2/3 at P_S=0 | VERIFIED | test_sigmoid_baseline passes; S(0)=0 exactly |
| 3 | CMC steady state zeros(1,8); f(0,0)==0 | VERIFIED | test_steady_state_zero and test_steady_state_assert pass |
| 4 | Integrator right-division via torch.linalg.solve; exp(-16) before Q | VERIFIED | test_right_division_orientation and test_regulariser_before_q pass |
| 5 | float64 enforced; no eigenvalue-clip | VERIFIED | test_float64_enforced and test_no_eig_clip pass |
| 6 | SPM12 parity gate green at documented tolerances | VERIFIED | All 9 parity rungs pass; measured values in table below |
| 7 | Existing paths bit-exact (additive-only) | VERIFIED | 0-line diff on neural_state.py, ode_integrator.py, pyproject.toml; 97 existing tests pass |

**Score: 7/7 truths verified**

---

## Per-Requirement Checklist (CMC-01..07)

### CMC-01: cmc_neural_mass.py -- single-source CMC state equations

**Status: SATISFIED**

- src/pyro_dcm/forward_models/cmc_neural_mass.py -- 212 lines, commits 2d4f16f / 870a2f4
- 4 populations (spiny stellate ss, superficial pyramidal sp, inhibitory interneurons ii, deep pyramidal dp), 8 states
- Second-order synaptic kernel: f[0]=x[1], f[2]=x[3], f[4]=x[5], f[6]=x[7]
- Sigmoid S=1/(1+exp(-Rx))-1/2 with R=(2/3)*exp(P.S) -- cites spm_fx_cmc.m:90-94
- Equations of motion cite spm_fx_cmc.m:171-198 per equation
- Module constants G0 (10-element), T0_MS (4-element), E0 (4-element) per spm_fx_cmc.m:47-49
- cmc_f returns shape (8,) float64 at n=1

### CMC-02: parameterize_cmc with J_PERM permutation guard

**Status: SATISFIED**

- J_PERM = (6,1,2,3,0,4,5,7,8,9) confirmed in source (direct Python call)
- Maps MATLAB j=[7 2 3 4 1 5 6 8 9 10] (1-indexed) to 0-indexed Python; J_PERM[0]=6
- Free P.G col 0 drives G col 6 (sp self-inhibition / precision knob)
- test_permutation_guard PASSES: P_G col 0 += 0.5 changes G col 6; G col 0 allclose to baseline
- test_extrinsic_convention PASSES: parameterize_A token absent from source; A=zeros at n=1
- Uses +exp(P.*) log-normal scaling; NOT the fMRI -exp/2 convention

### CMC-03: local_linearization.py ports spm_int_L

**Status: SATISFIED**

- src/pyro_dcm/utils/local_linearization.py -- 165 lines, commit 2d4f16f
- integrate_local_linearization(f, x0, inputs, dt, n_substeps=1, delay_operator=None, g=None)
- Frozen Jacobian via torch.func.jacrev(lambda v: f(v, u0))(x0)
- Regulariser: dfdx = jacobian - identity * math.exp(-16.0) BEFORE forming E and Q (line 79)
- Right-division: torch.linalg.solve(dfdx.T, rhs).T (line 85) -- no torch.inverse
- D=identity when delay_operator=None (test_identity_delay_default PASSES)
- NOT routed through torchdiffeq/integrate_ode -- new sibling; ode_integrator.py unchanged
- Citations: spm_int_L.m:112-169, :126-127, :132-147, :70; Ozaki (1992) by name; no [REF-xxx]

### CMC-04: cmc_priors.py provides prior moments + zero steady state

**Status: SATISFIED**

- src/pyro_dcm/forward_models/cmc_priors.py -- 104 lines, commit 62336a3
- cmc_prior_moments: T zeros(n,4) var 1/32; G zeros(n,4) var 1/32; S zeros(n,1) var 1/64; C mean c_mask*32-32; R zeros(n_inp,2) var 1/16
- Cites spm_cmc_priors.m :114-116, :121, :122, :124, :133
- cmc_steady_state(n) returns torch.zeros(n, 8, dtype=float64) with assert all-zeros
- Cites spm_dcm_neural_x.m:70-72 (CMC otherwise branch -- no Newton solve)
- test_steady_state_assert PASSES: shape (1,8), dtype float64, all zeros

### CMC-05: erp_input.py provides Gaussian evoked drive

**Status: SATISFIED**

- src/pyro_dcm/forward_models/erp_input.py -- 75 lines, commit 62336a3
- erp_gaussian_input(t_s, p_r, ons_ms=60.0, dur_ms=16.0, sus=0.0) -> (ns, n_inp)
- ms timebase: t_ms = t_s * 1000.0 (cites spm_erp_u.m:46)
- Sustained-mix kept even at sus=0 (pitfall N4); dur_ms=16 EXPLICIT (pitfall N3)
- 32-scaling: 32.0 * bump (cites spm_erp_u.m:63)
- test_erp_input_peak PASSES: peak at t_ms approx 60, max value 32.0, dtype float64, shape (128,1)

### CMC-06: Single-Source SPM12 Parity Gate (the milestone crux)

**Status: SATISFIED**

Fixture: validation/data/erp_single_source_fixtures.mat (11 KB, committed to git)
M3 job 57884677, MATLAB R2022a + Carrick spm12.
SPM file IDs: spm_int_L.m 7143 2017-07-29 / spm_fx_cmc.m 7279 2018-03-10
D=1, x0=zeros(8), dt=0.004, ns=128, dur=16, u_test=32.0 confirmed from meta.
checks_pass=true in cluster/results/erp_cross_validation_57884677.json.

Parity ladder (tests/test_spm_erp_dcm_validation.py) -- 9 rungs ALL PASS on laptop:

| Rung | What isolated | max diff | Gate | Result |
|------|--------------|----------|------|--------|
| pre-assert | D==1, x0==zeros(8), u_test=32, dt=0.004, float64 | mandatory | mandatory | PASS |
| 1. f_field | cmc_f transforms/sigmoid/J_PERM/units | 5.821e-11 | <=1e-10 | PASS |
| 2a. J0 (spm_diff FD) | cmc_f == spm_fx_cmc, method matched | 0.000e+00 | <=1e-10 | PASS (bit-exact) |
| 2b. J0 (jacrev) MEASURED | exact-AD vs SPM FD floor | 5.556e-04 | recorded | FD truncation |
| 3. matrix_exp MEASURED | torch.matrix_exp vs spm_expm | 8.556e-11 | <1e-9 | PASS |
| 4. Q_update | right-division (E-I) inv(dfdx) | 2.745e-15 | <=1e-9 | PASS |
| 5a. y_states (scheme) | loop ordering with SPM own Q_update | 6.573e-14 | <=1e-8 | PASS |
| 5b. y_states (FD-Jac) | end-to-end with spm_diff Jacobian | 1.161e-10 | <=1e-8 | PASS |
| 5c. y_states (jacrev) MEASURED | propagated FD truncation | 4.692e-08 | <1e-6 | PASS |

Matrix_exp floor MEASURED: 8.556e-11 < 1e-9 (satisfies pitfall V3).

Jacrev-vs-spm_diff assessment: The shipped integrator uses exact jacrev; SPM uses spm_diff forward
differences (dx=exp(-8)). The 4.7e-8 y_states floor is entirely the propagated FD truncation from
SPM own spm_diff. Acceptable because: (a) cmc_f IS spm_fx_cmc bit-exact (rung 2a: 0.0); (b) the
integration scheme IS bit-exact (rung 5a: 6.6e-14); (c) floor is measured and documented (V3);
(d) 4.7e-8 is below the Phase-35 gate of 1e-7; (e) exact AD is more accurate than SPM FD.
Decision 33-03-D2 documents this. NOT a gap.

### CMC-07: float64 enforced; NO eigenvalue-clip on CMC Jacobian

**Status: SATISFIED**

- cmc_f raises TypeError on float32 x_flat (test_float64 PASSES)
- integrate_local_linearization raises TypeError on float32 x0 (test_float64_enforced PASSES)
- test_no_eig_clip PASSES: Jacobian with eigenvalue +0.5 -> trajectory grows (not clipped)
- No eigenvalue / eig_clip / eigvalues tokens in local_linearization.py

---

## Required Artifacts

| Artifact | Status | Lines |
|----------|--------|-------|
| src/pyro_dcm/utils/local_linearization.py | VERIFIED | 165 |
| src/pyro_dcm/forward_models/cmc_neural_mass.py | VERIFIED | 212 |
| src/pyro_dcm/forward_models/cmc_priors.py | VERIFIED | 104 |
| src/pyro_dcm/forward_models/erp_input.py | VERIFIED | 75 |
| src/pyro_dcm/forward_models/__init__.py | VERIFIED | +18/0- append-only |
| tests/test_cmc_forward.py | VERIFIED | 144 |
| tests/test_local_linearization.py | VERIFIED | 204 |
| tests/test_spm_erp_dcm_validation.py | VERIFIED | 383 |
| validation/data/erp_single_source_fixtures.mat | VERIFIED | 11 KB |
| validation/export_to_mat.py | VERIFIED | append-only |
| validation/matlab_scripts/spm_fx_cmc_nodelay.m | VERIFIED | 35 |
| validation/matlab_scripts/run_spm_erp_dcm.m | VERIFIED | 170+ |
| cluster/scripts/erp_cross_validation.py | VERIFIED | 245 |
| cluster/sbatch/erp_cross_validation.sbatch | VERIFIED | 70+ |
| cluster/sbatch/erp_parity_test.sbatch | VERIFIED | 50+ |
| cluster/results/erp_cross_validation_57884677.json | VERIFIED | checks_pass=true |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| local_linearization.py | f(v,u) callable | v = v + q_op @ f(v, u_i) loop (line 161) | WIRED |
| local_linearization.py | torch.linalg.solve | torch.linalg.solve(dfdx.T, rhs).T (line 85) | WIRED |
| cmc_neural_mass.py | G col 6 via J_PERM[0]=6 | mult col J_PERM[:4] = exp(p_g) scatter (line 137) | WIRED |
| test_spm_erp_dcm_validation.py | erp_single_source_fixtures.mat | scipy.io.loadmat in fixture | WIRED |
| test_spm_erp_dcm_validation.py | integrate_local_linearization + cmc_f | imported + called in rungs 5b/5c | WIRED |
| validation/export_to_mat.py::export_erp_dcm | run_spm_erp_dcm.m | load(input_path, DCM) in MATLAB | WIRED (M3 job 57884677) |

---

## Additive-Only Verification

git diff e91ce2e HEAD shows ONLY new source files, __init__.py append (+18/0-), export_to_mat.py
append, MATLAB/cluster/test files, planning docs. No edits to existing forward models.

Confirmed 0-line diff vs baseline: neural_state.py, ode_integrator.py, pyproject.toml.

Regression smoke: 97 existing tests pass (test_neural_state, test_balloon, test_ode_integrator,
test_csd_computation, test_validation_export, test_rdcm_forward, test_spectral_dcm_model).

---

## Citations Verification

All 4 new source files use SPM source file + line range + author/year-in-prose only.
No [REF-xxx] patterns, no fabricated BibTeX keys, no REFERENCES.md edits (grep returns empty).

Examples: spm_int_L.m:126-127, Ozaki (1992); spm_fx_cmc.m:90-94, David, O. & Friston, K.J. (2003);
spm_cmc_priors.m:114-116, spm_dcm_neural_x.m:70-72; spm_erp_u.m:42-64.

---

## Anti-Patterns Found

None. No TODO/FIXME/placeholder. No torch.inverse (only torch.linalg.solve). No eigenvalue
clipping. No parameterize_A import in cmc_neural_mass.py. pyproject.toml unchanged (zero new deps).

---

## Gaps Summary

None. Phase 33 goal achieved. All 7 CMC requirements satisfied with direct test evidence and code
inspection. The jacrev-vs-spm_diff nuance (4.7e-8 y_states floor) is a measured numerical-method
floor, not a gap -- the forward and scheme are bit-exact to SPM, and the shipped integrator is
more accurate than SPM FD.

---

*Verified: 2026-06-26T09:22:12Z*
*Verifier: Claude (gsd-verifier)*
