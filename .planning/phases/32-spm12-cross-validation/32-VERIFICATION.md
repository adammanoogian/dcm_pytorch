---
phase: 32-spm12-cross-validation
verified: 2026-06-12T06:57:09Z
status: passed
score: 3/3 requirements satisfied (8/8 observable truths verified)
notes: >-
  The strict-5-percent-absolute-F gate and the 10-percent-Ep gate are NOT met.
  These are recorded as DOCUMENTED FINDINGS per a binding user decision (research
  pitfall S3: a constant 269.895-nat F normalization offset makes absolute-F
  infeasible by convention; deterministic forward-model divergence in posterior
  means). They are NOT scored as gaps. The defensible cross-model criterion
  (ranking agreement = 1.0 across all 5 seeds) is met. See
  32-SPM-CROSSVAL-FINDINGS.md.
---

# Phase 32: SPM12 Cross-Validation Verification Report

**Phase Goal:** VL output is cross-validated against MATLAB spm_nlsi_GN on a
prior-matched spectral DCM problem, agreeing on posterior means in free-parameter
space, on matched-problem free energy, and on model ranking -- reusing the
existing validation/ SPM bridge.

**Verified:** 2026-06-12T06:57:09Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Same-CSD bridge export_spectral_dcm_csd_for_spm injects Python (F,N,N) CSD into SPM | VERIFIED | export_to_mat.py:243; writes DCM.Y.csd=observed_csd (no transpose, S4) + DCM.Y.Hz at L337-339; complex128/float64 casts L305-306 |
| 2 | C-order/S4 round-trip green (existing + new injected-CSD contract) | VERIFIED | test_csd_corder_roundtrip.py + test_csd_injection_roundtrip.py pass (9 passed, incl. asymmetry guard) |
| 3 | compare_free_energies is strict-5pct, single-problem-only, S3-walled by docstring | VERIFIED | compare_results.py:277; docstring forbids cross-DIFFERENT-model use (S3) L292-294; within_tolerance gate L327 |
| 4 | run_vl_validation.py orchestrator exists with prior-matched VL-to-SPM behavior | VERIFIED | run_vl_spectral_dcm_validation L267; hyperprior_mean=8.0, prior_mean_a_offset=a_mask/128 L348-350; theta_post A_free L353 |
| 5 | MATLAB injection script calls spm_nlsi_GN on injected CSD, env-overridable SPM12 path | VERIFIED | run_spm_spectral_dcm_csd_injected.m; getenv SPM12_PATH L29; spm_nlsi_GN(DCM.M,DCM.U,Y) L149; MAR recompute skipped L82-91 |
| 6 | MATLAB/SPM12 paths env-overridable (config.MATLAB_PATH, SPM12_PATH) | VERIFIED | config.py:52 MATLAB_PATH from os.environ.get; SPM12_PATH passed to MATLAB child env (run_vl_validation.py:199-204) |
| 7 | S3 held in code: no element-wise Cp, no absolute-F-across-models | VERIFIED | run_vl_validation.py -- all Cp mentions are prohibition docstrings/comments; only compare_model_ranking (relative) + single-problem compare_free_energies. Same in spm_xval_multiseed.py:18-19,183 |
| 8 | Real M3 run JSONs carry ranking 1.0, constant ~269.895-nat F offset, Ep numbers | VERIFIED | spm_cross_validation_56407192.json (ranking 1.0); spm_xval_multiseed_56407635.json (f_offset_mean=269.8947, f_offset_std=0.0, f_offset_is_constant=true, ranking 1.0 x5 seeds) |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Status | Details |
| --- | --- | --- |
| validation/export_to_mat.py::export_spectral_dcm_csd_for_spm | VERIFIED | 435 lines; new fn L243, BOLD-only exporter untouched |
| validation/compare_results.py::compare_free_energies | VERIFIED | 329 lines; fn L277, documented dict, S3 docstring |
| validation/run_vl_validation.py::run_vl_spectral_dcm_validation | VERIFIED | 413 lines; wires all key links |
| validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m | VERIFIED | 197 lines; injects CSD, env SPM12 path, saves Ep_A/Cp/F |
| tests/test_csd_injection_roundtrip.py | VERIFIED | 139 lines; passes (vl) |
| tests/test_compare_free_energies.py | VERIFIED | 66 lines; passes (vl) |
| tests/test_vl_spm_cross_validation.py | VERIFIED | 158 lines; collects + skips cleanly without MATLAB |
| cluster/scripts/spm_cross_validation.py + spm_xval_multiseed.py | VERIFIED | 209 / 196 lines; call orchestrator, record-dont-crash |
| cluster/sbatch/spm_cross_validation.sbatch + spm_xval_multiseed.sbatch | VERIFIED | 76 / 76 lines |
| cluster/results JSONs (56407192, 56407635) | VERIFIED | Present, exit 0, consistent with findings |

### Key Link Verification

| From | To | Via | Status |
| --- | --- | --- | --- |
| run_vl_validation.py | run_variational_laplace SPM-matched priors | hyperprior_mean=8.0, prior_mean_a_offset=a_mask/128 | WIRED (L348-350) |
| run_vl_validation.py | export_spectral_dcm_csd_for_spm + MATLAB injected script | same-CSD injection + subprocess | WIRED (L184, L197-204) |
| run_vl_validation.py | theta_post A_free vs spm Ep_A | compute_free_param_comparison (S1) | WIRED (L353, L364) |
| run_vl_validation.py | compare_free_energies + compare_model_ranking | matched-F gate + relative ranking (S3-safe) | WIRED (L367, L399) |
| export_to_mat.py | DCM.Y.csd / DCM.Y.Hz | savemat, C-order, no transpose | WIRED (L337-339) |
| MATLAB script | spm_nlsi_GN on injected CSD | inject before estimation, skip MAR recompute | WIRED (L149) |

### Requirements Coverage

| Requirement | Status | Notes |
| --- | --- | --- |
| VLSPM-01 | SATISFIED | End-to-end M3 run on reciprocal-asymmetric N=2; hE=8.0, a_mask/128 priors confirmed |
| VLSPM-02 | SATISFIED (documented findings) | Ranking agreement 1.0 (defensible criterion). Strict Ep-10pct / abs-F-5pct misses recorded as findings (constant 269.895-nat offset, deterministic forward-model divergence) per binding user decision; S3 held in code |
| VLSPM-03 | SATISFIED | New orchestrator + comparator added; bridge reused; round-trip tests green |

### Anti-Patterns Found

None blocking. All Cp / element-wise / absolute-F occurrences in the changed
validation/ + cluster/ Python are prohibition docstrings/comments, not actual
forbidden comparisons. No TODO/FIXME/placeholder stubs in the delivered paths.

### Automated Checks

- Fast -m vl tests: 9 passed in 13.04s (csd_injection_roundtrip, csd_corder_roundtrip, compare_free_energies).
- SPM-gated test: 1 skipped, 0 errors (test_vl_spm_cross_validation.py auto-skips, no MATLAB locally).
- ruff: All checks passed on all 8 changed validation/ + cluster/ Python files.

### Documented Findings (NOT gaps -- binding user decision)

1. Model-ranking agreement = 1.0 across all 5 seeds; the defensible cross-engine criterion. Met.
2. vl_F minus spm_F = exactly 269.895 nats, std = 0.0 (f_offset_is_constant=true). Engines compute free energy identically up to a fixed normalization constant; strict-absolute-F-5pct is infeasible by convention (pitfall S3). Recorded.
3. Posterior means diverge deterministically (identical across seeds; analytic CSD is noise-free). VL tracks ground truth closer (off-diag vs-true ~1pct); SPM lands systematically off-truth: a genuine, quantified forward-model difference, not noise or a bug. Recorded.

### Human Verification Required

None required for the verdict. The strict-gate misses are conclusively explained
by the recorded findings and are consistent across single-seed and 5-seed M3 runs.

### Gaps Summary

No gaps. All deliverables exist, are substantive, and are wired; fast tests are
green; ruff clean; SPM-gated test skips correctly; result JSONs are present and
internally consistent with the findings (ranking 1.0, constant 269.895-nat F
offset with std 0, documented Ep numbers); S3 is held in code (only relative
compare_model_ranking + single-problem compare_free_energies, no element-wise
Cp, no absolute-F-across-models). The strict-5pct-F and 10pct-Ep misses are
expected/documented per the binding user decision and are scored as
MET-with-documented-finding, not as gaps.

---

_Verified: 2026-06-12T06:57:09Z_
_Verifier: Claude (gsd-verifier)_
