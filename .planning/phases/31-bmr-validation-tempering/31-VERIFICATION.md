---
phase: 31-bmr-validation-tempering
verified: 2026-06-11T14:34:38Z
status: passed
score: 3/3 must-haves verified (VLBMR-01, VLBMR-02, VLBMR-03)
re_verification: false
test_run:
  command: "pytest tests/test_bmr_vlbmr01_recovery.py tests/test_bmr_vs_vl_refit.py tests/test_bmr_tempering_calibration.py -m vl -q"
  result: "8 passed in 96.42s"
  breakdown: "VLBMR-01: 2 passed (N2, N4); VLBMR-02: 1 passed; VLBMR-03: 5 passed"
---

# Phase 31: BMR Validation & Posterior Tempering Verification Report

**Phase Goal:** Bayesian Model Reduction is validated as a defensible model-comparison tool -- relative-evidence ranking recovers the true circuit structure with a reported separation gap and agrees with brute-force ELBO -- with posterior tempering offered only as an exploratory, PD-safe restore of an absolute-deltaF regime calibrated against the Phase 30 coverage output.

**Verified:** 2026-06-11T14:34:38Z  
**Status:** passed  
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | BMR relative ranking recovers true sparse circuit structure (top-K == true edges) on a real spectral VL posterior, with positive separation gap | VERIFIED | test_bmr_vlbmr01_recovery.py 2 passed (N2 K2, N4 K6); set-equality gate lines 128-130, positive-gap assert line 140, cut-at-K assert line 146 |
| 2 | Analytic BMR agrees with brute-force VL refit on reduced-model RANK (present-edge prune costs more than absent; worst-model agreement); Spearman report-only | VERIFIED | test_bmr_vs_vl_refit.py 1 passed; rank gates lines 282-317; Spearman report-only lines 319-334; rho=1.0 |
| 3 | Posterior tempering is EXPLORATORY, PD-safe, coverage-matched, side-by-side annotation; absolute-deltaF never gated | VERIFIED | test_bmr_tempering_calibration.py 5 passed; PD guard lines 120-133; T-selection lines 136-178; side-by-side finite-gap lines 181-207; cluster JSON note + C2c |

**Score:** 3/3 truths verified (8/8 tests pass)

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| benchmarks/bmr_recovery.py | VERIFIED | All 5 functions present (lines 47, 113, 140, 200, 307). Full-cov slice sigma_post[:d,:d] line 193 (NOT diagonal-only). Tempering via temper_vl_posterior line 360 |
| tests/test_bmr_vlbmr01_recovery.py | VERIFIED | mark.vl line 64; SEEDS=range(42,47) line 60; reciprocal edges lines 67-69; passes 2/2 |
| tests/test_bmr_vs_vl_refit.py | VERIFIED | mark.vl line 235; refit a_mask-zero lines 217-219; rank gates only; passes 1/1 |
| tests/test_bmr_tempering_calibration.py | VERIFIED | pytestmark vl line 41; PD-guard shape/factor asserts lines 131-133; passes 5/5 |
| tests/test_bmr_vs_elbo.py (UNMODIFIED) | VERIFIED | No phase-31 commit touches it (git log main..HEAD empty); git diff HEAD empty; last commit be6a982 (phase 23) |
| cluster/scripts/bmr_tempering_calibration.py | VERIFIED | _DT_TASK=0.1 line 105 + guard line 166; temper_vl_posterior lines 245/415; ast.parse OK; no pip |
| cluster/sbatch/bmr_tempering_calibration.sbatch | VERIFIED | No --array; --partition=comp line 32; only a NEVER-pip comment (no pip command) |
| cluster/results/bmr_tempering_calibration_56397206.json | VERIFIED | On disk (5905 bytes). status=ok; tempering_factor=2.0; coverage_trace; stress_cell side-by-side rankings; held_out_cell C2c (cross_condition_non_pd=true); top-level note EXPLORATORY |

### Key Link Verification

| From | To | Status | Details |
|------|----|--------|---------|
| bmr_tensors_from_vl_result | rank_connections | WIRED | line 193 slices full A_free covariance, consumed by tests calling rank_connections |
| tempered_vs_untempered_ranking | temper_vl_posterior (PD guard) | WIRED | line 360; NO hand-rolled torch.linalg.cholesky in tempering path |
| cluster samples_fn | temper_vl_posterior | WIRED | cluster script line 245 (per-T cov tempering before MVN draw) |
| select_tempering_factor | compute_coverage_from_samples | WIRED | bmr_recovery.py line 275; coverage-matching vs Phase 30 recovery_matrix.json |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| VLBMR-01 (primary recovery) | SATISFIED | Truth 1; 2/2 tests pass |
| VLBMR-02 (BMR vs brute-force VL agreement) | SATISFIED | Truth 2; 1/1 test pass; ELBO test untouched |
| VLBMR-03 (exploratory tempering) | SATISFIED | Truth 3; 5/5 tests pass + cluster JSON harvested status=ok |

### Critical Cross-Cutting Checks (project central failure mode)

| Check | Status | Evidence |
|-------|--------|----------|
| NO absolute-deltaF as pass/fail criterion (Pitfall C1) | PASS | Only numeric deltaF assertion is separation_gap > 0 (sign/positivity, NOT a nat magnitude) |
| All tempering routes through temper_vl_posterior (no hand-rolled Cholesky) | PASS | grep cholesky returns only docstrings; test torch.linalg.cholesky is PD-confirmation of guard output, not a tempering step |
| Three laptop VL tests actually pass on this machine | PASS | 8 passed in 96.42s (under ~4-min budget) |
| Reciprocal-edge ground-truth deviation documented + does not undermine claim | PASS | Recorded 31-01-D1 / 31-02-D1; claim still recovers TRUE structure (now reciprocal -- legitimate spectral-DCM identifiability fix; feed-forward chains CSD-invisible) |
| cluster JSON is Mutagen artifact (untracked) -- on-disk only | PASS | File present on disk; not required committed |

### Anti-Patterns Found

None blocking. The sbatch contains a NEVER-pip comment (not a pip command); test files use type-ignore[import-untyped] on pyro_dcm imports (established 30-01-D4 convention, pyproject untouched). No TODO/FIXME/placeholder/empty-return stubs in any phase-31 artifact.

### Human Verification Required

None. All must-haves are programmatically verifiable and verified.

### Gaps Summary

No gaps. All three requirements satisfied with real, substantive, wired implementations. The single substantive deviation -- switching the prescribed feed-forward / single-edge ground truth to RECIPROCAL edges -- is a documented, legitimate spectral-DCM identifiability fix (a strictly lower-triangular A yields a CSD bit-identical to the empty graph, so BMR has no signal to rank). Recorded in both 31-01 and 31-02 SUMMARYs; does NOT undermine the VLBMR-01/02 claim (relative ranking still recovers the TRUE structure, which is simply reciprocal). The central failure mode -- absolute-deltaF as a pass/fail criterion -- is absent from all three test files (only a positivity check on separation_gap), and every tempering operation routes through the PD-guarded temper_vl_posterior with no hand-rolled Cholesky.

---
*Verified: 2026-06-11T14:34:38Z*  
*Verifier: Claude (gsd-verifier)*
