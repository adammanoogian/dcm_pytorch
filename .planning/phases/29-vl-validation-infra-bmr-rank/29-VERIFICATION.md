---
phase: 29-vl-validation-infra-bmr-rank
verified: 2026-06-10T20:09:35Z
status: passed
score: 6/6 success criteria verified
re_verification: false
vl_test_result: "17 passed, 0 failed (pytest -m vl); split into 3 runs: unit 9, smoke 3, determinism 5"
gaps: []
human_verification: []
---

# Phase 29: VL Validation Infra + BMR Rank Verification Report

**Phase Goal:** The benchmark + BMR layers expose everything downstream validation
needs -- three registered VL runners, VL-aware BenchmarkConfig, the corrected
RELATIVE-ranking BMR API, a PD-safe posterior-tempering primitive, and the
precision-intractability/dt guards -- with VL convergence + multi-restart determinism
proven across all three forward models. Laptop-runnable; NO cluster requirement.

**Verified:** 2026-06-10T20:09:35Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Success Criteria (per-criterion)

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | VLINFRA-01: BenchmarkConfig optional VL fields, None defaults | met | benchmarks/config.py L89-92: max_iter, hyperprior_mean, hyperprior_precision, prior_mean_a_offset all `... | None = None`, appended after fixtures_dir; documented L54-72 incl. dt>=0.1 note. quick_config/full_config forward **kwargs unchanged. |
| 2 | VLINFRA-02/-05: 3 VL runners registered, files import VL engine, vl marker, MATLAB_PATH | met | benchmarks/runners/__init__.py L38-40 registers ("spectral","vl"),("task","vl"),("latent_circuit","vl"). 3 runner files exist (188/231/227 lines), each imports run_variational_laplace_generic (L41/L46/L46) + matching forward model. pyproject.toml L92 registers vl marker. config.py L52-57 defines MATLAB_PATH (env-overridable, default R2022a binary). |
| 3 | VLINFRA-03: rank_connections RELATIVE ranking + separation gap (NOT absolute threshold) | met | model_selection/bmr.py L431-549: K single-prune BMR calls via _single_prune_costs, sort ascending by prune_delta_f, separation_gap = max gap_to_next, separation_after_rank. Docstring Notes L491-501 explicitly states absolute delta-F is NOT a pass/fail criterion (cites job 55772525). Re-exported __init__.py L14,23. 4 ranking tests pass. |
| 4 | VLINFRA-04: temper_vl_posterior scales cov by temp + Cholesky PD guard raises loud | met | bmr.py L552-613: float64 cast, factor<=0 raises (L594), sigma_tempered=factor*sigma symmetrized (L600-601), torch.linalg.cholesky guard raising ValueError with shape+factor in message (L603-611). Re-exported __init__.py L15,24. PD-guard tests pass. |
| 5 | VLROBUST-01/-02: determinism suite (3 models, fixed-seed + multi-restart) + task precision guard | met | tests/test_vl_determinism.py (403 lines): _fit_spectral/_fit_task/_fit_latent helpers, test_{spectral,task,latent}_vl_deterministic_fixed_seed, test_different_seeds_differ, test_multistart_schedule_reproducible -- all 5 pass in 86.8s. Guard in inference/forward_models.py TaskDCMForward.build_precision (class L267, method L347): _TASK_PRECISION_MAX_DIM=5000 (L27), raises ValueError with BOTH actual ({ny}) and expected (5000) sizes + "dt >= 0.1" hint (L380-387); docstring documents dt>=0.1 floor (L355-359). |
| 6 | VLREC-05: C-order CSD round-trip regression test exists | met | tests/test_csd_corder_roundtrip.py (92 lines): asymmetric complex CSD round-trip via (j,i,w) C-order map + compute_csd_precision block-structure (non-transposed) assertion. Both tests pass. |

**Score:** 6/6 success criteria met.

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| benchmarks/config.py | VERIFIED | 4 VL fields, documented |
| config.py (MATLAB_PATH) | VERIFIED | env-overridable constant L52-57 |
| pyproject.toml (vl marker) | VERIFIED | L92 |
| src/pyro_dcm/model_selection/bmr.py | VERIFIED | rank_connections L431, temper_vl_posterior L552 |
| src/pyro_dcm/model_selection/__init__.py | VERIFIED | both re-exported in import block + __all__ |
| src/pyro_dcm/inference/forward_models.py | VERIFIED | TaskDCMForward.build_precision guard L347-389 |
| benchmarks/runners/{spectral,task,latent_circuit}_vl.py | VERIFIED | 188/231/227 lines, import VL engine + forward models |
| benchmarks/runners/__init__.py | VERIFIED | 3 vl registry entries L38-40 |
| tests/test_bmr_rank_connections.py | VERIFIED | 133 lines, 5 tests pass |
| tests/test_csd_corder_roundtrip.py | VERIFIED | 92 lines, 2 tests pass |
| tests/test_task_precision_guard.py | VERIFIED | 52 lines, 2 tests pass |
| tests/test_vl_runners_smoke.py | VERIFIED | 94 lines, 3 tests pass |
| tests/test_vl_determinism.py | VERIFIED | 403 lines, 5 tests pass |
| docs/03_methods_reference/vl_determinism_notes.md | VERIFIED | 97 lines |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| runners/__init__ RUNNER_REGISTRY | run_{spectral,task,latent_circuit}_vl | (variant,"vl") keys | WIRED (L38-40) |
| *_vl.py runners | VL engine + forward models | run_variational_laplace_generic + Forward classes | WIRED |
| bmr.rank_connections | bayesian_model_reduction | _single_prune_costs (K calls) | WIRED |
| model_selection package | rank_connections/temper_vl_posterior | re-export | WIRED |
| test_task_precision_guard | TaskDCMForward.build_precision | oversized ny -> ValueError expected vs actual | WIRED (test green) |
| test_vl_determinism | VL engine + 3 forward models | run_variational_laplace_generic + manual_seed | WIRED |

### VL Test Result

pytest -m vl collects 17 tests (791 deselected). Run split to keep each invocation
laptop-fast:
- unit (test_bmr_rank_connections + test_csd_corder_roundtrip + test_task_precision_guard): 9 passed in 3.44s
- test_vl_runners_smoke.py: 3 passed in 115.0s
- test_vl_determinism.py: 5 passed in 86.8s

Total: 17/17 vl-marked tests pass. No failures, no xfails, no unknown-marker warnings.

### Anti-Patterns Found

None. No TODO/placeholder/empty-return stubs in the verified artifacts. All functions
compute real mathematics with cited references (REF-070 for BMR).

### Pre-Existing Debt (NOT a Phase 29 gap)

tests/test_vl_forward_model_protocol.py -- 5 task-DCM failures
(test_task_is_forward_model, test_task_param_count, test_task_pack_unpack_roundtrip,
test_task_residual_is_real, test_task_dcm_vl_recovery). Root cause: simulate_task_dcm()
signature drift (TypeError: unexpected keyword argument 'dt_sim') plus make_block_stimulus
drift. Confirmed reproducing (5 failed, 6 passed in 17.0s). This file is NOT vl-marked,
so it is excluded from the -m vl suite entirely. Documented as pre-existing debt against
baseline commit a064e69; out of Phase 29 scope.

### Human Verification Required

None. All criteria verified programmatically against the codebase and the green vl suite.

### Gaps Summary

No gaps. All six success criteria (VLINFRA-01..05, VLROBUST-01/-02, VLREC-05) are met in
the actual code, all key links are wired, and the full 17-test vl suite is green on laptop
with no cluster requirement -- satisfying the Phase 29 goal.

---

_Verified: 2026-06-10T20:09:35Z_
_Verifier: Claude (gsd-verifier)_
