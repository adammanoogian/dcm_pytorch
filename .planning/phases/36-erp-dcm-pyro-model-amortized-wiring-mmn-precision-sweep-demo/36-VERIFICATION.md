---
phase: 36-erp-dcm-pyro-model-amortized-wiring-mmn-precision-sweep-demo
verified: 2026-06-26T00:00:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
erpdcm04_amendment:
  status: user_approved_deferral
  not_a_gap: true
---

# Phase 36: ERP-DCM Pyro Model, Amortized Wiring and MMN Precision-Sweep Demo -- Verification Report

**Phase Goal:** A Pyro generative ERP-DCM model, amortized inference wiring, and the headline
5-source MMN precision-sweep demo -- gated behind a green fixed-reference SPM forward-parity check.
**Verified:** 2026-06-26
**Status:** PASSED
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | erp_dcm_model samples 7 named CMC sites + scalp_noise_scale, conditions Gaussian obs_erp, delegates forward entirely to simulate_erp_dcm (no second forward assembly) | VERIFIED | erp_dcm_model.py 259 lines; simulate_erp_dcm( at line 236; zero hits for integrate_local_linearization or project_to_scalp; AutoNormal auto-discovers all sites including B_free_0 (MODEL-06). M3 job 57904695: test_erp_dcm_model green (3 tests). |
| 2 | ERPDCMPacker pack order identical to ERPDCMForward.pack_params (A_free|C_free|T|G|S|R), identity reshape (no .exp()), n_features equals param_count; amortized flow guide trains without error (B fixed via ERPDCMForward.predict) | VERIFIED | ERPDCMForward.pack_params (lines 811-818) and ERPDCMPacker.pack (lines 501-510) identical order. _run_erp_forward_model calls forward.predict at line 353. M3 job 57903632: slow amortized test 91.25s exit 0. M3 job 57904695: test_amortized_erp green. |
| 3 | build_mmn_5source_network returns 5-source auditory-MMN topology byte-identical to locked _MS_* fixture constants (element-wise asserted), C into bilateral A1, deviant B at precision nodes, NO MNI coords | VERIFIED | forward_models/mmn_reference.py 322 lines; lazy-reads _MS_* from validation.export_to_mat (single source of truth); test_topology_equality_vs_ms_constants and test_no_mni_coords_emitted green. M3 job 57904695. |
| 4 | MMN sweep asserts monotone gain->|MMN| attenuation AND windowed (90-120 ms) negative MMN minimum attenuating with gain; frontal scalp dominance recorded honestly as rIFG/A1 ratio + ECD-deferral note (user-approved amendment) | VERIFIED (amended scope) | run_precision_sweep asserts mmn_t non-increasing (lines 459-465) and win_t < 0 at all gain points + win_mag non-increasing (lines 469-482). rIFG/A1 ratio printed + ECD note lines 609-614. 7-point sweep monotone 1.54e-5 to 1.12e-6; windowed min negative at all 7 points. |
| 5 | mmn_cmc_params sets FREE P.G[:,0] at {rIFG,A1L,A1R} (never G[:,6] directly), flowing to G[:,6] via J_PERM[0]=6; returns complete bundle {p, a_masks, b_masks, c_mask, x_design, l_full} | VERIFIED | mmn_reference.py lines 292-296: g[node, 0] = sp_inhibition_gain at each precision node. G[:,6] never indexed directly. test_sp_inhibition_gain_moves_g6_not_g0 green. M3 job 57904695. |
| 6 | demo_mmn_precision_sweep.py runs run_parity_gate() FIRST (line 597), raises SystemExit on absent fixture and RuntimeError on divergence > 1e-7, calls make_figure() ONLY after gate + guard + sweep pass (line 616) | VERIFIED | main() lines 596-616 linear sequence. Gate raises before any figure. Measured: scalp ERP 6.379e-11, diff wave 1.265e-11, both <= 1e-7. figures/mmn_precision_sweep.{png,pdf} exist on disk. |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|----------|
| src/pyro_dcm/models/erp_dcm_model.py | Pyro generative ERP-DCM (7 sample sites + scalp_noise_scale; reuses simulate_erp_dcm; Gaussian obs_erp) | VERIFIED | EXISTS 259 lines (min 80). Contains simulate_erp_dcm(. No second forward loop. Exported from models/__init__.py. |
| src/pyro_dcm/guides/parameter_packing.py | class ERPDCMPacker appended (pack/unpack/fit_standardization/n_features) | VERIFIED | class ERPDCMPacker at line 429. n_features = 4NN+NM+4N+4N+N+2M. Identity reshapes; no exp at unpack. |
| src/pyro_dcm/guides/summary_networks.py | class ErpSummaryNet appended (MLP, float64) | VERIFIED | class ErpSummaryNet at line 145. MLP (Cnd*ns*Nc,) -> embed_dim. Float64. |
| src/pyro_dcm/models/amortized_wrappers.py | amortized_erp_dcm_model + _run_erp_forward_model appended (B fixed; reuses ERPDCMForward.predict) | VERIFIED | amortized_erp_dcm_model at line 365; _run_erp_forward_model calls forward.predict at line 353. B excluded from packer, fixed in ERPDCMForward. |
| src/pyro_dcm/forward_models/mmn_reference.py | build_mmn_5source_network + mmn_cmc_params | VERIFIED | EXISTS 322 lines (min 90). Both functions present and exported from forward_models/__init__.py. |
| tests/test_erp_dcm_model.py | structural trace + AutoNormal auto-discovery + SVI-smoke | VERIFIED | EXISTS 159 lines. 3 tests. M3 job 57904695: green. |
| tests/test_amortized_erp.py | ERPDCMPacker round-trip (3 laptop) + amortized flow-trains-without-error (slow/M3) | VERIFIED | EXISTS 175 lines. 4 tests. M3 jobs 57904695 + 57903632: all green. |
| tests/test_mmn_reference.py | topology equality vs _MS_* + permutation + adapter-bundle + fwd/bwd control | VERIFIED | EXISTS 207 lines. 7 tests. M3 job 57904695: green. |
| scripts/demo_mmn_precision_sweep.py | gate -> sweep -> assert (monotone/windowed-negative) -> figure | VERIFIED | EXISTS 620 lines (min 120). Gate-before-figure ordering confirmed in main() lines 597-616. |
| figures/mmn_precision_sweep.{png,pdf} | publication-quality gain->|MMN| transfer curve + diff-wave overlay | VERIFIED | Both exist on disk (gitignored; script is source of truth). Two-panel figure confirmed. |

---

### Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|----------|
| erp_dcm_model.py | simulate_erp_dcm | Single call at line 236; no second forward assembly | WIRED | grep confirms single call; zero hits for integrate_local_linearization or project_to_scalp in file |
| amortized_erp_dcm_model | ERPDCMForward.predict | _run_erp_forward_model calls forward.predict at line 353 | WIRED | Confirmed in code; B stays fixed inside ERPDCMForward |
| ERPDCMPacker.pack | ERPDCMForward.pack_params | Both flatten [A_free, C_free, T, G, S, R] in identical order | WIRED | test_packer_order_matches_forward_pack_params asserts torch.equal; M3 green |
| mmn_cmc_params(sp_inhibition_gain) | G[:,6] via J_PERM[0]=6 | Sets FREE P.G[node,0]; parameterize_cmc_network maps to G[:,6] | WIRED | Code lines 292-296; tests assert G[:,6] changes and G[:,0] unchanged |
| build_mmn_5source_network() | _MS_* topology constants | Lazy-reads from validation.export_to_mat inside _ms_topology() | WIRED | test_topology_equality_vs_ms_constants asserts torch.equal element-for-element |
| run_parity_gate() | erp_leadfield_fixtures.mat | Loads via _load_fixture; raises SystemExit on absent (line 297) | WIRED | fixture_path.exists() check line 296; raises BEFORE make_figure |
| demo.main() | gate -> figure ordering | run_parity_gate() line 597 -> sweep -> make_figure() line 616 | WIRED | Linear call sequence; no path to make_figure if gate raises |

---

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ERPDCM-01: models/erp_dcm_model.py Pyro generative model, log-space priors (A,B,C,G,T,R), Gaussian scalp likelihood, B modulation | SATISFIED | None |
| ERPDCM-02: amortized path -- ERPDCMPacker + amortized_erp_dcm_model additively appended; flow guide trains on erp_simulator draws without error | SATISFIED | None |
| ERPDCM-03: 5-source auditory MMN network (A1 L/R, STG L/R, rIFG) with fwd/bwd/lateral graph, C into bilateral A1, deviant B; MNI coords flagged | SATISFIED | None |
| ERPDCM-04: precision sweep with gain->|MMN| curve; monotone attenuation + frontal-dominant negative-going difference wave (AMENDED: windowed-negative + honest ECD note) | SATISFIED (amended) | None -- user-approved deferral of frontal-dominance to ECD phase |
| ERPDCM-05: consumer adapter (sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag) -> CMC params for actinf_physics | SATISFIED | None |
| ERPDCM-06: MMN sweep figure gated behind green fixed-reference SPM forward-parity check BEFORE any output | SATISFIED | None |

---

### Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| erp_dcm_model.py | B_PRIOR_VARIANCE = 1/8 flagged MUST-VERIFY in module docstring | Warning | Documented provisional; low-stakes for fixed-B headline demo; not a blocker |
| All Phase 36 files | No TODO/FIXME/placeholder/return null patterns found | -- | Clean |

No blocker anti-patterns. B-prior variance flag is a documented future action item, not a functional defect.

---

### Additive-Only Verification

git diff --stat 39b43d4..54e765d on forward stack subdirectories:

- forward_models/mmn_reference.py: +322 lines (NEW file)
- forward_models/__init__.py: +7 lines (APPENDED exports only)
- simulators/, utils/, inference/: 0 changes

Phase 33/34/35 forward stack (cmc_neural_mass.py, erp_coupled_system.py, erp_leadfield.py,
local_linearization.py, erp_simulator.py, forward_models.py core) is byte-untouched.
The ForwardModel protocol is untouched. All Phase 36 changes are new files and additive appends.

---

### M3 Test Execution Evidence

| M3 Job | Target | Result |
|--------|--------|--------|
| 57904695 | Full v0.8.0 ERP scope (11 test files incl. test_erp_dcm_model, test_amortized_erp, test_mmn_reference) | 79 passed in 88.29s, exit 0 |
| 57903632 | tests/test_amortized_erp.py tests/test_erp_dcm_model.py (@pytest.mark.slow amortized flow path) | 7 passed in 91.25s, exit 0 |

All three Phase 36 test files confirmed green on M3.

---

### ERPDCM-04 Amendment -- Documented Deferral, NOT a Gap

The plan literal criterion required machine-asserting a frontal-dominant negative-going difference
wave. During Task 2, a 189-tuning forward-only search found best rIFG/A1 ratio = 0.062
(A1 dominates by >= 16x, ~2000x at baseline). Root cause: the LFP-identity lead field gives
one channel per source at raw sp-voltage; input node A1 (direct evoked drive) necessarily
dominates rIFG (3 synapses downstream). True scalp frontal-dominance is an ECD
dipole-orientation effect -- frontal dipoles project to frontal electrodes; A1 dipoles tangential.
This was explicitly deferred from Phase 35 (35-01-D1). ecd_spatial() exists but requires
an (Nc,3,n) dipole-gain fixture that does not exist.

The executor stopped (Rule-4 blocker). The orchestrator chose Option A: ship the honest LFP
source-space demo now; ECD frontal-topography becomes a follow-up phase.

Delivered in place of the original criterion:
1. Monotone gain->|MMN| attenuation -- machine-asserted (raises RuntimeError on violation)
2. Windowed (90-120 ms) negative MMN deflection that attenuates with gain -- machine-asserted
3. rIFG/A1 ratio printed honestly at each gain point (~5e-4 at baseline)
4. ECD-deferral scope note in script output (lines 609-614) and figure caption (lines 577-583)

User-approved scope change recorded in 36-03-SUMMARY, ROADMAP Phase 36 entry,
and script run_precision_sweep() docstring. NOT a verification gap.

---

### Zotero Action Required (Non-Blocking)

Six papers cited by author/year in Phase 36 source files but not yet confirmed in Zotero.
No [REF-xxx] BibTeX keys fabricated (CLAUDE.md compliance). Add before any manuscript citation:

1. Adams, R.A. et al. (2013), The computational anatomy of psychosis, Front. Psychiatry 4, 47
2. Ranlund, S. et al. (2016), Impaired prefrontal synaptic gain..., Hum. Brain Mapp. 37, 351-365
3. Garrido, M.I. et al. (2009), The mismatch negativity: a review..., Clin. Neurophysiol. 120, 453-463
4. David, O. & Friston, K.J. (2003), A neural mass model for MEG/EEG..., NeuroImage 20, 1743-1755
5. Bastos, A.M. et al. (2012), Canonical microcircuits for predictive coding, Neuron 76, 695-711
6. Kiebel, S.J. et al. (2006), Dynamic causal modelling of evoked responses..., NeuroImage 30, 1273-1284

---

## Overall Assessment

Phase 36 (v0.8.0 CAPSTONE) goal is fully achieved. All 6 ERPDCM requirements are satisfied.
The phase delivered: a Pyro generative ERP-DCM model reusing the parity-gated forward verbatim;
an amortized flow path with ERPDCMPacker matching ERPDCMForward pack order exactly; a public
5-source MMN network builder byte-identical to the SPM12 fixture; a consumer adapter routing
sp_inhibition_gain through the correct permutation; and a gated demo script whose parity check
(scalp ERP 6.379e-11, diff wave 1.265e-11, both <= 1e-7) runs before any figure is produced.
The ERPDCM-04 frontal-dominance deferral is an honest, user-approved scope adjustment -- not a gap.

---

*Verified: 2026-06-26*
*Verifier: Claude (gsd-verifier)*
