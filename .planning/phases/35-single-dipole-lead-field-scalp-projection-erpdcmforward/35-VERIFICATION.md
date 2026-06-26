---
phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward
verified: 2026-06-26T13:57:02Z
status: passed
score: 6/6 must-haves verified
human_verification: []
---

# Phase 35: Single-Dipole Lead-Field, Scalp Projection & ERPDCMForward -- Verification Report

**Phase Goal (ROADMAP):** Per-source LFPs become the observed scalp ERP via a single-dipole
lead-field (kron(P.J,L_spatial), LFP-first), the deviant-standard difference wave is produced,
and ERPDCMForward (implementing the existing ForwardModel protocol) gives VL inference as free
reuse -- scalp-ERP parity proven vs spm_lx_erp on frozen fixtures.

**Verified:** 2026-06-26T13:57:02Z
**Status:** PASSED
**Re-verification:** No -- initial verification
---

## Test Execution Evidence

**M3 job 57901337** (comp partition, erp_pytest.sbatch): **65 passed in 106.73s, exit 0**.

Scope confirmed from cluster/sbatch/erp_pytest.sbatch TEST_TARGET (line 38):
test_local_linearization.py, test_cmc_forward.py, test_spm_erp_dcm_validation.py,
test_erp_coupled_system.py, test_spm_erp_multisource_validation.py,
test_export_erp_multisource.py, **test_erp_leadfield.py**, **test_spm_erp_leadfield_validation.py**.

Phase-35 files in scope:
- test_erp_leadfield.py (15 tests: structural guards + protocol tests + VL round-trip)
- test_spm_erp_leadfield_validation.py (6 tests: pre-asserts + 5-rung parity ladder)

tests/test_export_erp_leadfield.py (3 tests) is not in the M3 sbatch TEST_TARGET
but passed locally per 35-02-SUMMARY -- deterministic, no MATLAB dependency.
---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | build_lead_field(p_j, l_spatial) = kron(P.J, L_spatial) column-major; P.J FIRST; block s at cols [s*n:(s+1)*n] = P.J[s]*L_spatial | VERIFIED | erp_leadfield.py:158 torch.kron(p_j.reshape(1,_N_STATES), l_spatial); test_build_lead_field_kron_column_major asserts all 8 state blocks; rung 1 vs SPM: max-diff=0.000e+00 |
| 2 | cmc_default_pj() one-hot at index 2 (sp_V), asserted argmax==2 AND != 6 | VERIFIED | _PJ_STATE=2 at line 53; test_cmc_default_pj_is_state_index_2 triple guard (argmax==2, argmax!=6, sum==1); pre_asserts in validation ladder confirm on frozen fixture |
| 3 | lfp_spatial(ones(n),n) == eye(n); build_lead_field(e_2, I_n) sp-voltage block (s=2) == I_n, all others == 0 | VERIFIED | erp_leadfield.py:102-103 torch.diag(p_l); test_lfp_spatial_default_identity + test_build_lead_field_lfp_identity_blocks confirm identity at cols [10:15] |
| 4 | project_to_scalp(states, l_full, x0) = (states - x0) @ l_full.T; x0 defaults to zeros; through identity LFP, scalp[:,j] == source j sp-voltage | VERIFIED | erp_leadfield.py:194 (states-x0) @ l_full.transpose(-2,-1); test_project_to_scalp_through_identity_lfp asserts y[:,j] == states[:,2*n+j]; batched tested |
| 5 | ERPDCMForward appended after LatentCircuitForward; all 8 ForwardModel members; zero edits to Protocol, VL engine, sibling classes | VERIFIED | forward_models.py:693; git diff --stat cf6cc93: 325 insertions, 0 deletions; variational_laplace.py + cmc_neural_mass.py + erp_coupled_system.py + local_linearization.py: 0 diff |
| 6 | Scalp ERP vs spm_gen_erp + spm_lx_erp <=1e-7 (rung 4); diff wave <=1e-7 AND non-zero (rung 5); predict ndim guard works; VL round-trip recovers planted params | VERIFIED | Rung 4: floor 6.379e-11 <=1e-7 GATE PASS (~1600x margin); rung 5: max-diff=1.265e-11, max-diff_wave=3.076e-2>0; ndim guard: torch.equal confirmed; LEAD-06: R^2 0.535->0.679, walltime 87.4s <180s |

**Score:** 6/6 truths verified
---

## Required Artifacts

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|----------|
| src/pyro_dcm/forward_models/erp_leadfield.py | 5 lead-field functions; torch.kron; SPM citations; no [REF-xxx] | VERIFIED | 194 lines; all 5 functions; citations spm_lx_erp.m:33 / spm_erp_L.m:112/76 / spm_L_priors.m:108; no [REF-xxx] |
| src/pyro_dcm/inference/forward_models.py | ERPDCMForward appended; all 8 protocol members; additive only | VERIFIED | Line 693; git diff: 325 insertions, 0 deletions; _ERP_DEAD_FREE=-32.0 at line 690; all 8 members present |
| src/pyro_dcm/simulators/erp_simulator.py | optional l_full arg; scalp/difference_wave_scalp keys; existing keys unchanged | VERIFIED | 31 insertions, 1 deletion (comment rewrite only); l_full arg at line 59; scalp keys at lines 148-152 |
| src/pyro_dcm/forward_models/__init__.py | 5 new names exported | VERIFIED | Lines 25-30 (imports) and 84-88 (in __all__): all 5 names |
| tests/test_erp_leadfield.py | Structural guards + protocol tests + VL round-trip | VERIFIED | 15 test functions; @pytest.mark.vl @pytest.mark.slow on round-trip; in M3 sbatch scope |
| tests/test_spm_erp_leadfield_validation.py | 5-rung parity ladder; fixture-keyed NOT MATLAB-keyed | VERIFIED | pytestmark skipif keys on _FIXTURE_PATH.exists() lines 138-145; 6 tests; skip does NOT trigger |
| validation/data/erp_leadfield_fixtures.mat | L_full (5,40), y_scalp {2}x(128,5), diff_wave (128,5); meta provenance | VERIFIED | shapes confirmed by loadmat; meta D=1, nargout_Mf=2, N=5, Nc=5, dipfit_type=LFP, P_J argmax=2, P_L=ones(5), x0=zeros; committed 170c11a |
| validation/export_to_mat.py | export_erp_dcm_leadfield additive; LFP spatial spec; all-double | VERIFIED | 244 insertions, 0 deletions; test confirms dipfit.type=LFP, P.J idx-2, P.L ones(1,5) |
| validation/matlab_scripts/run_spm_erp_dcm_leadfield.m | spm_lx_erp call; nargout==2; D=1; x0==zeros asserts | VERIFIED | Lines 106-125: all three asserts present; L_full = spm_lx_erp(P, dipfit) |
| cluster/sbatch/erp_cross_validation_leadfield.sbatch | comp partition; no pip | VERIFIED | File exists; 35-02-SUMMARY confirms comp/16G/1h, no pip |
| cluster/scripts/erp_cross_validation.py | --mode leadfield dispatches main_leadfield() | VERIFIED | Line 786 choices includes leadfield; lines 795-796 dispatch |
---

## Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|---------|
| erp_leadfield.py:build_lead_field | torch.kron(p_j.reshape(1,8), l_spatial) | column-major state-block kron | WIRED | Lines 158-160 |
| erp_leadfield.py:cmc_default_pj | index 2, not 6 | _PJ_STATE=2; one-hot construction; triple test guard | WIRED | Lines 53, 77-79 |
| ERPDCMForward.predict | project_to_scalp + integrate_local_linearization + apply_condition_modulation + cmc_network_f | lazy import; per-condition loop; torch.stack to (Cnd,ns,Nc); reshape(-1) | WIRED | Lines 918-964 |
| ERPDCMForward.predict ndim guard | identical flat output for 3-D and flat observed | if observed.ndim >= 3: y = y[:, :observed.shape[1]] | WIRED | Lines 962-963; test asserts torch.equal(y_main, y_fd) |
| simulate_erp_dcm scalp path | project_to_scalp on flat (ns,8n) traj; stacked to (Cnd,ns,Nc) | if l_full is not None: guard; project + stack + difference | WIRED | Lines 132-152; test confirms shape + non-zero + B-omitted control |
| test_lead06_vl_roundtrip | run_variational_laplace_generic(ERPDCMForward(...), scalp_obs) | @pytest.mark.vl; import from variational_laplace | WIRED | Lines 471-479; result keys finite; R^2 lift asserted |
| Parity ladder skip condition | keys on _FIXTURE_PATH.exists() (committed .mat), NOT MATLAB | pytest.mark.skipif(not _FIXTURE_PATH.exists(), ...) | WIRED | Lines 138-145; fixture committed 170c11a |

---

## LEAD Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|---------|
| LEAD-01: erp_leadfield.py builds L_full=kron(P.J,L_spatial) column-major; LFP diagonal; y=(x-x0)@L_full.T; SPM citations | SATISFIED | All three functions correct; SPM citations in every docstring; no [REF-xxx] |
| LEAD-02: P.J default=state index 2 asserted (not 6); kron column-major verified vs exported L_full fixture | SATISFIED | Triple guard in test; rung 1 max-diff=0.0; identity block at cols [10:15] confirmed |
| LEAD-03: difference wave non-zero asserted; negative-going sign (per REQUIREMENTS.md) | NON-ZERO: SATISFIED. SIGN: documented Phase-36 carryover (35-01-D2, 35-03-D2) | Non-zero asserted (B-wired vs B-omitted; rung 5 max-diff_wave=3.076e-2); parity <=1e-7; sign=-1 diagnostic; ROADMAP goal met |
| LEAD-04: ERPDCMForward appended; 8-member ForwardModel protocol; identity precision v1; zero edits to Protocol/VL engine | SATISFIED | 325 insertions, 0 deletions; all 8 members; build_precision returns eye(Cnd*ns*Nc,f64) |
| LEAD-05 (PARITY GATE): scalp ERP vs frozen fixtures <=1e-7 | SATISFIED | Rung 4: floor 6.379e-11 <=1e-7 GATE PASS (~1600x margin); fixture-keyed; M3 job 57901337 |
| LEAD-06: VL round-trip recovers planted CMC params | SATISFIED | n=2; R^2 0.535->0.679; walltime 87.4s; protocol confirmation |
---

## LEAD-05 Production Gate Reasoning Audit

The LEAD-05 gate at <=1e-7 for the production jacrev integrator path is sound:

1. Phase-34 measured source jacrev floor: 4.70e-8 (34-03-SUMMARY rung 6).
   Propagated spm_diff FD truncation vs exact torch.func.jacrev (exact AD is
   MORE accurate than SPM FD -- Fact 5, not a bug).
2. LFP lead field amplification: P.L=ones(5) -> L_spatial=I_5 (identity, no gain).
   build_lead_field(e_2, I_5) selects ONLY the sp-voltage column (state s=2);
   every other state block is exactly zero.
3. Measured scalp jacrev floor: 6.379e-11 -- BELOW the expected ~4.7e-8.
   The identity lead field selects only the sp-voltage state, which carries LESS
   of the propagated spm_diff FD truncation than the worst-case source state.
4. Gate margin: 6.4e-11 / 1e-7 = ~1600x below tolerance.
5. Caveat baked into test (lines 518-521): a non-identity P.L != 1 scales the
   floor by max|P.L| -- must re-measure before gating with a non-LFP spatial model.

Reasoning verdict: Sound. The identity-projection no-amplification argument is correct,
the measured floor confirms it (6.4e-11 is actually below 4.7e-8 source floor), and
the caveat is documented in the test body for Phase-36 ECD work.
---

## LEAD-03 Scope Split Assessment

The PLAN (35-01-D2, 35-03-D2) explicitly defers the negative-going/frontal SIGN
of the difference wave to Phase 36. The REQUIREMENTS.md says the difference wave
must be asserted non-zero and negative-going. The scope split is sound because:

- In LFP identity mode, the sign of the scalp difference wave at precision nodes
  depends on CMC A/B parameterization and ECD dipole orientation (angle in MNI space),
  neither of which exists in Phase 35.
- The _MS_ 5-source reference net produces a deviant-minus-standard difference that is
  non-zero and matches SPM diff_wave element-wise (max-diff=1.265e-11 <=1e-7) --
  sufficient for the Phase-35 forward-parity goal.
- The peak channel sign=-1 is printed as a non-gating diagnostic in rung 5,
  providing early signal for Phase 36.
- Phase 36 will add ECD gain export + ecd_spatial + MNI coords to lock the
  physiologically correct sign convention.

Verdict: SOUND scope split. The ROADMAP goal (difference wave is produced) is fully
met. The REQUIREMENTS.md negative-going addition is a Phase-36 item.

---

## Additive-Only Verification

git diff --stat cf6cc93 HEAD -- src/ validation/ (7 files, 1039 insertions, 1 deletion):

- src/pyro_dcm/forward_models/__init__.py: 13 insertions (re-export 5 new names)
- src/pyro_dcm/forward_models/erp_leadfield.py: 194 insertions (NEW)
- src/pyro_dcm/inference/forward_models.py: 325 insertions (ERPDCMForward append)
- src/pyro_dcm/simulators/erp_simulator.py: 31 insertions, 1 deletion (comment rewrite)
- validation/data/erp_leadfield_fixtures.mat: binary NEW (committed 170c11a)
- validation/export_to_mat.py: 244 insertions (export_erp_dcm_leadfield append)
- validation/matlab_scripts/run_spm_erp_dcm_leadfield.m: 233 insertions (NEW)

Frozen modules (0 diff since cf6cc93): variational_laplace.py, cmc_neural_mass.py,
erp_coupled_system.py, local_linearization.py -- all confirmed.

Phase 33/34 regression: M3 job 57901337 ran test_spm_erp_multisource_validation.py
and test_spm_erp_dcm_validation.py in the 65-passed scope. No regressions.

---

## Anti-Patterns Scan

| File | Finding | Severity | Assessment |
|------|---------|----------|------------|
| erp_leadfield.py | No TODO/FIXME/placeholder; all 5 functions compute real math | None | Clean |
| ERPDCMForward.predict lines 955-956 | NaN guard: if not torch.isfinite(traj).all() traj = zeros | WARNING | Defensive guard, not a stub. Returns zero-gradient signal on integrator divergence. Acceptable for v1 |
| ecd_spatial in erp_leadfield.py | Built now, EXERCISED in Phase 36 in module docstring | INFO | Full einsum matvec at line 132; deferral is of the exercise (fixture+test), not the code |

---

*Verified: 2026-06-26T13:57:02Z*
*Verifier: Claude (gsd-verifier), model: claude-sonnet-4-6*