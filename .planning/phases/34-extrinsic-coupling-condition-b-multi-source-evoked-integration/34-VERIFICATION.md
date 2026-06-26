---
phase: 34-extrinsic-coupling-condition-b-multi-source-evoked-integration
verified: 2026-06-26T10:32:05Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 34: Extrinsic Coupling and Evoked Integration Verification Report

**Phase Goal:** The hierarchical CMC network (extrinsic fwd/bwd/lateral A, condition-specific B incl. the diag(B) to G precision path, C-driven evoked integration) produces per-source per-condition LFPs matching spm_gen_erp on frozen multi-source fixtures (delays off, D=1).
**Verified:** 2026-06-26T10:32:05Z
**Status:** PASSED
**Re-verification:** No - initial verification

---

## Test Run Output

All tests run on laptop against committed fixture (no MATLAB required).

- Phase-34 suite (test_erp_coupled_system.py + test_spm_erp_multisource_validation.py + test_export_erp_multisource.py): **19 passed in 12.70s**
- Phase-33 regression (test_cmc_forward.py + test_local_linearization.py + test_spm_erp_dcm_validation.py): **25 passed in 26.20s**
- Combined Phase-33+34 scope: **44 passed in 19.77s**

### Measured max|diff| per rung (test_spm_erp_multisource_validation.py -s)

| Rung | max-diff | Gate | Result |
|------|----------|------|--------|
| pre: meta D==1, nargout_Mf==2, N==5, x0==zeros(5,8), float64 | -- | hard asserts | PASS |
| 1: spm_gen_Q Q.A{1..4} B->all-A fold (free log) | 0.000e+00 | <=1e-12 | PASS |
| 1: spm_gen_Q Q.G(:,1) diag(B)->precision col | 0.000e+00 | <=1e-12 | PASS |
| 1b: diag->G NEGATIVE (omit-diag breaks deviant QG) | 5.000e-01 (with-diag=0.0) | mismatch required | PASS |
| 2: network J0 (spm_diff FD, dx=exp(-8)) | 0.000e+00 | <=1e-10 | PASS |
| 3: network Q_update (right-division, C2) | 1.706e-12 | <=1e-9 | PASS |
| 4: trajectory SCHEME (SPM own Qupd) | rel 8.162e-13, abs 3.298e-11 | rel <=1e-12 | PASS |
| 5: trajectory FD-Jacobian (spm_diff-matched J0) | 1.255e-10 | <=1e-8 | PASS |
| 6: trajectory shipped-jacrev | 4.698e-08 | MEASURED NOT gated (<=1e-5 ceiling) | PASS (recorded) |

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | apply_condition_modulation reproduces spm_gen_Q QA all four A blocks AND QG element-wise <=1e-12 | VERIFIED | Rung 1: max-diff=0.000e+00; confirmed by direct Python call |
| 2 | Omitting diag(B)->G[:,0] destroys deviant QG match (precision path load-bearing) | VERIFIED | Rung 1b mismatch=5.000e-01=X*diag(B); with-diag=0.000e+00; structural negative green |
| 3 | cmc_network_f(n=1) == cmc_f bit-exactly (extrinsic terms vanish at n=1) | VERIFIED | test_network_f_n1_bit_exact max-diff=0.0; confirmed by direct call |
| 4 | Network J0 (spm_diff FD) matches fixture <=1e-10, proving cmc_network_f IS spm_fx_cmc at N>1 | VERIFIED | Rung 2: max-diff=0.000e+00 |
| 5 | Multi-source trajectory matches spm_gen_erp (scheme machine-epsilon, FD-Jacobian <=1e-8) | VERIFIED | Rung 4 rel 8.162e-13; Rung 5 1.255e-10; Rung 6 measured 4.698e-08 not gated |
| 6 | Fixture meta.D==1, nargout_Mf==2, x0==zeros(5,8), float64 confirmed | VERIFIED | test_pre_asserts green; fixture spot-check confirms all fields |

**Score:** 6/6 truths verified

---

## Per-Requirement (EVOK-01..06) Checklist

### EVOK-01: Extrinsic coupling topology wired

File: src/pyro_dcm/forward_models/erp_coupled_system.py (257 lines)

- Forward: +A[0]@S[:,2] into ss (f1), +A[1]@S[:,2] into dp (f7) - lines 231, 247.
- Backward: -A[2]@S[:,6] into sp (f3), -A[3]@S[:,6] into ii (f5) - lines 236, 241.
- Lateral: (1+4L) reciprocal reduction in parameterize_cmc_network lines 103-106.
- C drives spiny-stellate only via big_u = (c @ u_vec) * 32.0 entering uu_g only - lines 223, 230.
- n=1 bit-exact guard: max-diff=0.0 confirmed by independent Python call.
- Tests: test_forward_backward_adjacency, test_lateral_reciprocal_reduction, test_input_drives_ss_only - ALL GREEN.
- Citations: spm_fx_cmc.m:68-82,171-198 with SPM Id spm_fx_cmc.m 7279 inline.

**Status: VERIFIED**

### EVOK-02: Dual-B precision mechanism

Files: src/pyro_dcm/forward_models/erp_coupled_system.py, tests/test_erp_coupled_system.py, tests/test_spm_erp_multisource_validation.py

- B folds into all four A{1..4} at line 161 (spm_gen_Q.m:47).
- diag(B[i]) added to free Q.G[:,0] at line 164 (spm_gen_Q.m:65-67), routes via J_PERM[0]==6 to G[:,6] (sp self-inhibition). J_PERM[0]=6 confirmed in cmc_neural_mass.py line 55.
- POSITIVE structural guard test_c4_precision_guard: Q[G][:,0] == P[G][:,0] + X*diag(B0) AND parameterised G[:,6] moves vs baseline - GREEN.
- NEGATIVE structural guard test_c4_negative_omit_diag: omit-diag leaves G[:,6] identical to baseline - GREEN.
- Fixture-anchored negative rung 1b: omit-diag produces 5.000e-01 mismatch = X*diag(B) to 1e-12 - GREEN.
- Independent Python call: with-diag G[:,6]=[1079.9,1319.0] vs baseline [800,800]; without-diag G[:,6]=[800,800] unchanged.

**Status: VERIFIED**

### EVOK-03: C drives ss only; evoked integration

Files: src/pyro_dcm/forward_models/erp_coupled_system.py, src/pyro_dcm/simulators/erp_simulator.py (125 lines)

- C enters only uu_g (ss conductance numerator), not uu_sp/uu_ii/uu_dp.
- simulate_erp_dcm loops conditions, applies apply_condition_modulation per condition, calls integrate_local_linearization (Phase-33 integrator) per condition.
- Smoke test confirms shapes (2, ns, n, 8) / (ns,) / (ns, n_inp), all-finite trajectories.

**Status: VERIFIED**

### EVOK-04: simulate_erp_dcm and difference-wave hook

Files: src/pyro_dcm/simulators/erp_simulator.py, tests/test_erp_coupled_system.py

- Returns {states: (Cnd, ns, n, 8), pst: (ns,), inputs: (ns, n_inp), difference_wave: states[1]-states[0]}.
- test_simulate_erp_dcm_smoke: with B wired, dw[:,:,2] != 0 (non-zero sp-voltage difference) - GREEN.
- Control: with B=0, difference_wave == zeros_like(dw) exactly - GREEN.
- Scope boundary documented: source-level states only; scalp lead-field + true MMN are Phase 35.

**Status: VERIFIED**

### EVOK-05: SPM12 parity gate (the crux)

Files: tests/test_spm_erp_multisource_validation.py (534 lines, 8 tests), validation/data/erp_multisource_fixtures.mat (86247 bytes, committed)

- Fixture committed to git, fixture-keyed skip guard (not MATLAB-keyed, per 33-03-D1).
- spm_gen_Q algebra: Q.A{1..4} max-diff=0.000e+00 (<=1e-12), Q.G(:,1) max-diff=0.000e+00 (<=1e-12). Both exactly 0.
- NEGATIVE rung 1b: omit-diag residual 5.000e-01 = X*diag(B) exactly to 1e-12. Precision path load-bearing.
- Network J0: max-diff=0.000e+00 (<=1e-10). cmc_network_f IS spm_fx_cmc at N=5.
- Q_update: max-diff=1.706e-12 (<=1e-9). Production right-division operator at network scale.
- Trajectory scheme rung: rel=8.162e-13, abs=3.298e-11. Gated on RELATIVE error (rel <=1e-12) - see assessment.
- Trajectory FD-Jacobian rung: max-diff=1.255e-10 (<=1e-8). End-to-end integrator with Jacobian method held to SPM.
- Trajectory jacrev rung: floor=4.698e-08. MEASURED NOT gated (loose <=1e-5 ceiling). Propagated spm_diff FD truncation; exact AD more accurate than SPM (Fact 5, 33-03-D2/D3). Not a bug.
- Reconstructed P imports Wave-2 exporter locked topology constants (pitfall V1 satisfied - identical P+drive).

**Status: VERIFIED**

#### Scheme-rung relative-vs-absolute gating (34-03-D1) assessment

**Soundness verdict: SOUND.**

The 5-source network states reach max amplitude 40.267 (standard) and 40.400 (deviant) - confirmed by direct scipy.io.loadmat inspection. At this magnitude, float64 accumulation noise over 128 steps scales the absolute floor to 3.298e-11, while the relative error stays at 8.162e-13 (machine epsilon). The relative gate (<=1e-12) is the correct scale-invariant loop-ordering invariant - calibration to network magnitude, not gate-loosening to force green.

Two independent confirmations rule out loop-ordering bugs (pitfall C1 - v = Q@(v+f) vs v = v + Q@f):
(a) Rung 3 Q_update matches operator to 1.706e-12, confirming the operator is correct.
(b) Rung 5 independently-built FD-Jacobian operator passes at 1.255e-10 <= 1e-8, confirming the loop is correct.
A real ordering bug would diverge catastrophically, not at machine epsilon. This decision is sound.

### EVOK-06: D=1 forced and asserted

Files: validation/matlab_scripts/run_spm_erp_dcm_multisource.m, validation/data/erp_multisource_fixtures.mat, tests/test_spm_erp_multisource_validation.py

- MATLAB script sets M.f = spm_fx_cmc_nodelay (nargout-aware 2-output wrapper, REUSED UNCHANGED from 33-02). Asserts nargout(M.f)==2 AND isequal(M.x, zeros(N,8)) before generating fixtures (script lines 107-110).
- Fixture meta.D=1, meta.nargout_Mf=2, meta.x0=zeros(5,8) - confirmed by direct scipy.io.loadmat inspection.
- test_pre_asserts: checks D==1, nargout_Mf==2, N==5, x0==zeros(5,8) - GREEN.
- Torch-side x0: cmc_steady_state(n) returns zeros(n,8) (Phase-33 M1 guarantee), re-asserted from fixture meta.
- Full delay path deferred by design (Phase 35+).

**Status: VERIFIED**

---

## Required Artifacts

| Artifact | Status | Evidence |
|----------|--------|----------|
| src/pyro_dcm/forward_models/erp_coupled_system.py | VERIFIED | 257 lines, real math, spm_fx_cmc.m + spm_gen_Q.m with Id + line numbers; 3 substantive functions |
| src/pyro_dcm/simulators/erp_simulator.py | VERIFIED | 125 lines, real implementation, imports Phase-33 integrator |
| tests/test_erp_coupled_system.py | VERIFIED | 329 lines, 9 tests GREEN; C4 precision guard + omit-diag negative + 6 structural guards + smoke |
| tests/test_spm_erp_multisource_validation.py | VERIFIED | 534 lines, 8 tests GREEN; 8-rung fixture-keyed parity ladder |
| tests/test_export_erp_multisource.py | VERIFIED | 72 lines, 2 tests GREEN; loadmat round-trip + B-folding teeth |
| validation/export_to_mat.py | VERIFIED | APPENDED only (294 lines added after line 650; single-source exporter byte-untouched) |
| validation/matlab_scripts/run_spm_erp_dcm_multisource.m | VERIFIED | 226 lines; nargout + x0==zeros asserts; per-condition QA/QG/J0/Qupd/y saved |
| validation/data/erp_multisource_fixtures.mat | VERIFIED | 86247 bytes, committed to git; D=1, nargout_Mf=2, N=5, Cnd=2 |
| src/pyro_dcm/forward_models/__init__.py | VERIFIED | Append-only (9 lines added) |
| src/pyro_dcm/simulators/__init__.py | VERIFIED | Append-only (simulate_erp_dcm added; ruff whitespace cosmetics only on existing lines) |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| apply_condition_modulation | parameterize_cmc_network -> G[:,6] | diag(B) added to free G col 0; J_PERM[0]=6 routes to G[:,6] | WIRED |
| cmc_network_f | cmc_neural_mass.cmc_f intrinsic body | duplicated EOM + 4 extrinsic A@S terms; bit-exact at n=1 (max-diff=0.0) | WIRED |
| simulate_erp_dcm | integrate_local_linearization | per-condition closure f_c at line 110 of erp_simulator.py | WIRED |
| test_spm_erp_multisource_validation | validation/data/erp_multisource_fixtures.mat | scipy.io.loadmat; torch-vs-frozen-MATLAB (pitfall V1 satisfied) | WIRED |
| _reference_p | validation.export_to_mat topology constants | imports _MS_*, _ms_log_block, _erp_gaussian_u_grid directly | WIRED |

---

## Additive-Only Verification

git diff --stat 9890337 HEAD -- src/pyro_dcm/forward_models/cmc_neural_mass.py src/pyro_dcm/utils/local_linearization.py
Result: (empty - zero changes to frozen Phase-33 files)

git diff --stat 9890337 HEAD -- src/
  src/pyro_dcm/forward_models/__init__.py           |   9 +
  src/pyro_dcm/forward_models/erp_coupled_system.py | 256 ++++++
  src/pyro_dcm/simulators/__init__.py               |   7 +-
  src/pyro_dcm/simulators/erp_simulator.py          | 124 ++++
  4 files changed, 394 insertions(+), 2 deletions(-)

The 2 deletions in simulators/__init__.py are ruff isort trailing-whitespace fixes on two existing comment lines - no semantic change, no existing export removed. cmc_neural_mass.py, local_linearization.py, and all fMRI/spectral/rDCM paths are byte-identical.

---

## Citation Compliance

- SPM source file + line + Id revision cited inline throughout erp_coupled_system.py and erp_simulator.py.
- No fabricated [REF-xxx] identifiers in any new file.
- No edits to .planning/REFERENCES.md.
- Author/year (David and Friston 2003, NeuroImage 20, 1743-1755) cited in module docstring of erp_coupled_system.py.
- float64 enforced throughout with TypeError guard at the network-forward boundary.

---

## Human Verification

None required. All critical behaviors are verified programmatically against frozen MATLAB-generated fixtures. The parity ladder is deterministic float64 against committed arrays.

---

*Verified: 2026-06-26T10:32:05Z*
*Verifier: Claude (gsd-verifier)*
