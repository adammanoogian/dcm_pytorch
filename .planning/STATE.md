# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-10)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Planning v0.7.0 — VL validation matrix + deferred real-data application (real Cam-CAN M/EEG, real foundation-model runs, SBI calibration). Seed: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`.

## Current Position

**Milestone:** v0.7.0 Variational Laplace Validation (VL-validation-led).
**Phase 32 — SPM12 Cross-Validation: IN PROGRESS (wave-2 code-complete; M3 run PENDING) — the LAST v0.7.0 phase.**
**32-03 CODE-COMPLETE 2026-06-11 (VLSPM-03, the Phase 32 deliverable — M3 RUN PENDING orchestrator):**
`run_vl_spectral_dcm_validation(seed=42, n_regions=2, max_iter=64)` in NEW `validation/run_vl_validation.py`
fits the Phase 28 VL engine on a reciprocal-ASYMMETRIC N=2 spectral problem (A[0,1]=0.15/A[1,0]=0.10,
post-override stability re-check; reciprocal mandatory per Phase 31 identifiability, asymmetry gives S4
teeth) with SPM-matched priors (hyperprior_mean=8.0, precision=128.0, prior_mean_a_offset=a_mask/128, S2),
injects the IDENTICAL observed_csd into SPM via the 32-01 bridge (`export_spectral_dcm_csd_for_spm` +
`run_spm_spectral_dcm_csd_injected.m`), runs `spm_nlsi_GN`, and compares: Ep free-space 10%
(`compute_free_param_comparison`, S1 — `theta_post["A_free"]` not parameterized A), matched-F strict 5%
(`compare_free_energies`, the HEADLINE gate on the identical CSD), relative cross-model ranking over 3
a_masks (`compare_model_ranking` >=0.80, S3-safe — NO absolute-F-across-models, NO element-wise Cp).
SPM-gated test `tests/test_vl_spm_cross_validation.py` (spm+slow, skipif(not check_matlab_available()))
HARD-asserts all three + S4 asymmetry + the no-Cp negative guard; **collects + SKIPS cleanly on this laptop
(1 skipped, 0 errors — local FlexLM -15 unreachable).** The REAL `spm_nlsi_GN` run is M3 sbatch
`cluster/sbatch/spm_cross_validation.sbatch` (comp, 16G, 1h, no pip, exports
MATLAB_PATH=/usr/local/matlab/r2022a/bin/matlab + SPM12_PATH=/home/aman0087/fc37/Carrick/spm12) →
`cluster/scripts/spm_cross_validation.py` (record-don't-crash per 31-03-D3: a gate miss incl. real 5%-F
miss is RECORDED in JSON + exit 0; only unexpected exception exits 1) → `cluster/results/
spm_cross_validation_<jobid>.json`. The `.m` SPM12 addpath is now `getenv('SPM12_PATH')` (local default kept;
ONLY change to the 32-01 file). MATLAB_PATH from config (env-overridable); SPM12_PATH on the subprocess child
env — SAME orchestrator runs laptop + M3. ruff clean on all 3 changed Python files; mypy delta is only the
pre-existing bare-dict [type-arg] + pyro_dcm [import-untyped] (no new class, 32-01-D3/32-02-D2). Decisions
32-03-D1..D4. Commits fd88aea (Task1 orchestrator), 62555ae (.m SPM12_PATH), 6f5dcfb (Task2 SPM-gated test),
4234502 (Task3 cluster harness). **HEADLINE matched-F relative_error: PENDING — orchestrator submits the
sbatch + harvests the JSON.** **Next: orchestrator submits `cluster/sbatch/spm_cross_validation.sbatch`,
harvests `cluster/results/spm_cross_validation_<jobid>.json`, populates the headline number in SUMMARY+STATE,
then `/gsd:verify-phase 32` → v0.7.0 close-out.**
**Phase 32 — SPM12 Cross-Validation: IN PROGRESS (wave-1) — the LAST v0.7.0 phase.**
**32-01 DONE 2026-06-11 (the SAME-CSD injection bridge):** a Python analytic `(F,N,N)` complex CSD
now injects element-identical into the SPM12 `DCM.Y.csd`/`DCM.Y.Hz` struct (C-order, NO transpose,
S4-guarded), and a forked MATLAB batch makes `spm_dcm_fmri_csd` fit THAT injected CSD instead of
recomputing it from BOLD via MAR — the precondition for the strict 5%-matched-nat free-energy gate
(32-02's `compare_free_energies`) used in Plan 32-03. `export_spectral_dcm_csd_for_spm()` in
`validation/export_to_mat.py` (casts complex128/float64; BOLD-only `export_spectral_dcm_for_spm`
untouched) + `validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m` (verify+inject csd/Hz,
conditional cell-wrap, bypass MAR `spm_dcm_fmri_csd_data`, print Ep.A(1,2) vs Ep.A(2,1), identical
Ep_A/Cp/F save block as `run_spm_spectral_dcm.m`) + `tests/test_csd_injection_roundtrip.py`
(2 @pytest.mark.vl, transpose guard `loaded[0,0,1] != loaded[0,1,0]`, freqs float64). Gate
(`test_csd_corder_roundtrip`) confirmed GREEN BEFORE building; combined 4 vl tests pass (2.97s);
ruff clean on both changed files. MATLAB script UNEXECUTED (R2022a installed but license checkout
failed `-15,10032`; full estimation is 32-03 by design) — verified by grep (22 tokens present).
Decisions 32-01-D1/D2/D3. Commits a4582a3 (Task 1), 158bc22 (Task 2), d72a514 (Task 3). NO file
overlap with 32-02 (parallel wave). **Next: 32-03 strict-5%-matched-F cross-validation consumes
this bridge + `compare_free_energies` under MATLAB+SPM12.**
**32-02 DONE 2026-06-11 (VLSPM-02, strict-5% matched-F gate):** added
`compare_free_energies(vl_free_energy, spm_F, rel_tolerance=0.05)` to `validation/compare_results.py`
— a SINGLE-matched-problem relative-tolerance comparator returning
`{vl_free_energy, spm_F, relative_error, within_tolerance, rel_tolerance}` with the 5% default as a
HARD pass/fail gate (`rel_err < rel_tolerance`), per the BINDING user decision (overrides research's
softer "report descriptively"). Docstring forbids S3 cross-model absolute-F use (cross-model path
stays `compare_model_ranking`, relative ranking) and ties the 5% target to same-CSD injection
(Plan 32-01). No existing function modified. `tests/test_compare_free_energies.py` (5
`@pytest.mark.vl` tests, 1.48s laptop): within/outside-tol, custom tolerance, zero-`spm_F`
div-by-zero guard, and a cross-model-ranking-is-separate-path test pinning the S3 boundary. The
`within_tolerance` return key is the contract consumed by Plan 32-03 (`run_vl_validation.py`).
ruff clean on both files; mypy delta 15→16 = the same bare-`dict` `[type-arg]` every sibling
comparator emits (no new error category). Decisions 32-02-D1/D2. Commits 4e1ed26 (feat), 1a2a096
(test). Branch: gsd/phase-32-spm12-cross-validation. **NOTE: Plan 32-01 (parallel wave, separate
agent) edits `validation/export_to_mat.py` + MATLAB/test files; no file overlap with 32-02.**
**Phase 31 — BMR Validation + Tempering: ✅ COMPLETE & VERIFIED PASSED 2026-06-11 (3/3 plans, 3/3 must-haves, 8 vl tests green on-machine).** VLBMR-01/02/03 all Complete. `31-VERIFICATION.md` status: **passed**. Tempering delivered strictly exploratory/PD-safe; absolute ΔF never gated anywhere. **Key identifiability finding (31-01-D1 / 31-02-D1):** spectral DCM cannot identify a lone off-diagonal A entry (CSD bit-identical to empty graph) — VLBMR-01/02 use RECIPROCAL-edge ground truth; the claim "relative ranking recovers TRUE structure" holds, the true structure is reciprocal. **Next: Phase 32 (SPM12 cross-validation, local/MATLAB) — the LAST v0.7.0 phase; `/gsd:plan-phase 32`. Needs MATLAB + SPM12.**
**Phase 31 — BMR Validation + Tempering: 3/3 PLANS DONE (31-01, 31-02, 31-03).**
**31-03 DONE 2026-06-11 (VLBMR-03, EXPLORATORY — NOT a headline claim):** posterior tempering,
calibrated by coverage-matching against Phase 30 coverage_95 and routed ENTIRELY through the
PD-guarded `temper_vl_posterior` (no hand-rolled Cholesky anywhere), restores 95% coverage on the
task-N4 overconfidence stress cell (M3 job 56397206: single re-fit seed untempered cov95 0.875 →
tempered 1.0 at chosen T=2.0) while leaving the BMR ranking UNCHANGED (tempered top-K == untempered
[12,11,3,14,7,13], sep_gap 5.13e5 nats unchanged). The SAME T=2.0 breaks PD on the task-N2
posterior — the concrete C2c cross-condition hazard, surfaced as `cross_condition_non_pd=true`
(status=ok, not aborted). `benchmarks/bmr_recovery.py` + `select_tempering_factor`
(smallest-T-in-band coverage matcher; closest-to-target surfaced if none in band, never raises) +
`tempered_vs_untempered_ranking` (side-by-side rank_connections via temper_vl_posterior).
`tests/test_bmr_tempering_calibration.py` (`@pytest.mark.vl`, 5 tests, 4.90s laptop). M3 glue
`cluster/scripts/bmr_tempering_calibration.py` + `.sbatch` (single task, no pip, dt>=0.1, rk4).
Absolute delta-F NEVER gated (C1/C2). ruff+mypy clean (laptop); ruff/ast/bash clean (cluster).
Commits 974fe58 (helpers+test), 2598d0f (cluster), 004a0ab (harvest 56397206). **Next: Phase 31 is
complete (3/3 plans); run `/gsd:verify-phase 31` then proceed to Phase 32 (SPM12 cross-validation,
local/MATLAB) or `/gsd:complete-milestone` planning.**
**31-01 DONE 2026-06-11 (VLBMR-01, the PRIMARY defensible result):** BMR relative-evidence ranking on a
real spectral-DCM VL posterior recovers the true sparse circuit structure 5/5 across seeds for N=2 and
N=4 (top-K essential off-diagonal edges == true present edges, positive separation_gap, cut at K), NEVER
gating on absolute delta-F (pitfall C1). `benchmarks/bmr_recovery.py` (sparse ground-truth A builder +
`bmr_tensors_from_vl_result` full A_free covariance slice `sigma_post[:N*N,:N*N]` + C-order `offdiag_indices`)
+ `tests/test_bmr_vlbmr01_recovery.py` (`@pytest.mark.vl`, 2 params x 5 seeds, 112.7s laptop). **KEY FINDING:
a feed-forward chain A is UNIDENTIFIABLE by spectral DCM — its CSD is bit-identical to the empty graph
(rel diff 0.0), collapsing A_free to zero; ground truth must use RECIPROCAL edges.** ruff+mypy clean.
Commits c081c5c (helpers), 766dd24 (test). **Note: 31-02 (`bc0e33f test(31-02)`) full-model spectral VL
fit + analytic BMR over reduced set also already present on this branch.** **Next: 31-03 (tempering
calibration) reuses `benchmarks/bmr_recovery.py` + the reciprocal-edge ground truth.**
**Phase 30 — Recovery Matrix Sweep: ✅ COMPLETE & VERIFIED PASSED 2026-06-11 (3/3 plans, 5/5 criteria).**
**TASK GAP CLOSED 2026-06-11:** the 4 task cells that errored ("underflow in dt 0.0", torchdiffeq 0.2.5/
torch 2.10 adaptive `dopri5` underflow in `simulate_task_dcm` ground-truth gen) were fixed by switching
task ground-truth sim to fixed-step `rk4` (`_run_task_cell`, commit c0a7616) + broadened per-seed except,
re-run on M3 (job 56372816), and harvested. **FULL MATRIX: 10/10 cells classified, 0 ERRORED — 6 PASS, 4
identifiability-limit-with-evidence.** Task DCM recovers cleanly at N=2 (sign_masked 1.0, A-RMSE ~0.04;
cell 5 PASS, cell 4 marginal coverage 0.75); task N=4 is a documented identifiability limit (sign 0.57,
coverage 0.0, conv 0.4) — a real finding, not a harness failure. VLREC-01..04 + VLROBUST-03 all Complete.
`30-VERIFICATION.md` status: **passed**. **Next: `/gsd:plan-phase 31` (BMR validation + tempering — task
N=4 is the natural overconfidence stress case for tempering calibration).**
**30-03 DONE 2026-06-10:** harvest + classifier + report over the COMPLETE M3 sweep (jobs 56346424 + 56372816).
`benchmarks/recovery_matrix_thresholds.py` (documented per-cell thresholds RMSE_A 0.05 / sign 0.80 /
coverage 0.85, SHRINKAGE_SOFT_TARGET 0.7 informational; `classify_cell` -> pass | identifiability_limit
WITH evidence, never raises on a failing cell, skips missing metrics) + `cluster/scripts/
recovery_matrix_aggregate.py` (glob per-cell JSON excl. local; classify; write recovery_matrix.csv/json;
eig_clamp/boundary regime from max_real_eig_list + shrinkage; fail-loud on zero files; parameterized
report path) + `tests/test_recovery_matrix_aggregate.py` (4 vl tests, ~5s) + `30-RECOVERY-MATRIX-REPORT.md`.
**REAL RESULTS: 5 PASS (spectral 0-3, latent cell 9), 1 IDENTIFIABILITY-LIMIT (latent cell 8 N=4 SNR=1:
RMSE 0.0501 marginally over provisional 0.05; sister cell 9 passes — audit-outlier case), 4 ERRORED (task
cells 4-7: torchdiffeq 'underflow in dt 0.0', surfaced not dropped). eig_clamp held (0 in-band draws); low
shrinkage at high SNR documented as expected Laplace overconfidence (VLROBUST-03).** ruff+mypy clean.
Commits 29978dd (thresholds), 32a93ed (aggregator), 819201e (tests), eb30643 (real-results report+matrix).
**VLREC-04 + VLROBUST-03 satisfied → Phase 30 CLOSED. Next: `/gsd:plan-phase 31` (tempering calibration).**
**30-02 DONE 2026-06-10:** recovery-matrix sweep driver + M3 submission. `benchmarks/recovery_matrix_grid.py`
(GRID constants N{2,4}×SNR{1,3}×{spectral,task,latent}, `enumerate_cells`/`cell_for_index` → 10 stable cells
[spectral 4 + task 4 + latent 2; latent N-axis collapsed to fixed N=4], `run_one_cell` reusing Phase 29 VL
simulate/forward symbols + per-cell SNR injection + near-boundary-A exclusion + per-region R2/shrinkage) +
`cluster/scripts/recovery_matrix_cell.py` (env-driven SLURM entrypoint, status=error never aborts the array,
per-cell JSON) + `cluster/sbatch/recovery_matrix_sweep.sbatch` (array 0-9, NO pip, dt≥0.1, comp/8h/16G).
LOCAL faithfulness pre-check PASSED (task 0 spectral N=2 SNR=1 quick max_iter=6: 10/10 seeds, 28.7s, valid
per-cell JSON w/ populated metrics). **SUBMITTED M3 array job 56346424 (120 fits, all 10 tasks RUNNING incl.
latent cells 8/9; Mutagen models/ fix confirmed in place on M3).** Results → `cluster/results/
recovery_matrix_56346424_<0..9>.json` (synced back via Mutagen). ruff+mypy clean. Commits 5c7547a (grid),
10918ce (cluster). **Next: 30-03 harvest+classifier AFTER job 56346424 completes.**
**30-01 DONE 2026-06-10:** hardened per-cell recovery-metric assembler `benchmarks/recovery_matrix_metrics.py`
(+ `tests/test_recovery_matrix_metrics.py`, 7 vl tests, 3.2s laptop). `assemble_cell_metrics()` turns one VL
runner result into per-region R2 (NOT pooled — consumed from a driver `r2_per_region_list` built via
`compute_trajectory_r_squared(pooled=False)`), MASKED sign recovery (reuses `masked_sign_recovery`,
guards sign(0)), 95% coverage, RMSE median/IQR, std_post/std_prior shrinkage; flat JSON, no tensors leak.
Plus `exclude_near_boundary_A`/`resample_A_until_accepted` (reject max-Re-eig in [-0.05,0], eig_clamp
non-injectivity N2; band constants `NEAR_BOUNDARY_LO/HI`) and `snr_for_model` (task/latent `{'SNR':snr}`
vs spectral `{'noise_log_amplitude':-log(snr)}`). ruff+mypy clean. VLREC-02 + VLREC-03 hardening proven.
Commits 6ae82cd (lib), 9fb55f4 (tests). **Next: 30-02 sweep driver (M3 cluster).**
**Phase 29 — VL Validation Infrastructure & BMR Rank Functions: ✅ COMPLETE 2026-06-10** (5/5 plans;
verifier passed 6/6; 17/17 vl tests green; all laptop). Phase 30 PREREQUISITES before the M3 sweep
launch (30-02): (1) fix the Mutagen `models/` ignore (recreate
`dcm-pytorch` session with anchored ignores) — required for latent-circuit M3 runs only; spectral/task
sweeps are unaffected; (2) decide the sweep grid (N values × SNR values × seeds) and confirm the
multi-hour cluster job. Phase 32 (SPM12, local/MATLAB) can run in parallel with 30.
**29-05 DONE:** VL determinism regression suite (`tests/test_vl_determinism.py`, 5 `@pytest.mark.vl`
tests, ~2m42s laptop): fixed-seed determinism across spectral/task/latent-circuit (same seed ->
posterior means equal within atol 1e-8, bitwise preferred), seed-sensitivity guard, and multi-restart
reproducibility (pitfall N4: fixed restart-seed schedule -> same winner). Methods note
`docs/03_methods_reference/vl_determinism_notes.md` documents the within-machine determinism contract +
non-determinism sources (BLAS order, float64 accumulation, rk4 ODE, FD step N5) + cross-machine caveat.
VLROBUST-01 delivered. Commits ed71f9c, b0bfd6e.
**29-04 DONE:** three `method="vl"` benchmark runners (`run_spectral_vl`, `run_task_vl`,
`run_latent_circuit_vl`) following the `(BenchmarkConfig)->dict` contract, registered additively in
`RUNNER_REGISTRY`; N=2/1-seed laptop smoke suite (`tests/test_vl_runners_smoke.py`) green in 113s.
Fixed a blocking `TaskDCMForward.predict` bug (`integrate_ode` took `step_size=`, not `options=`).
VLINFRA-02 delivered. Commits 372e203, 6a09579, a731fd5.
**29-01 DONE:** VL config foundation — optional None-default VL fields on `BenchmarkConfig`
(`max_iter`, `hyperprior_mean`, `hyperprior_precision`, `prior_mean_a_offset`; zero behavior change),
centralized env-overridable `MATLAB_PATH` in root `config.py`, registered `vl` pytest marker.
VLINFRA-01 + VLINFRA-05 delivered.
**29-02 DONE:** `rank_connections()` (relative single-prune BMR ranking + separation gap; absolute
delta-F never a pass/fail rule per job 55772525) and `temper_vl_posterior()` (temperature scale +
loud Cholesky PD guard, calibration deferred to Phase 31) added to `model_selection/bmr.py`,
re-exported, 5 vl unit tests pass on a known circuit. VLINFRA-03/04 delivered.
Roadmap drafted: 4 phases (29-32), 19/19 requirements mapped. Critical path 29 -> 30 -> 31; Phase
32 (SPM12, local/MATLAB) runs in parallel with Phase 30 (recovery sweep, M3 cluster). Confirmed
scope: synthetic recovery matrix (N×SNR), SPM12 cross-validation (user has MATLAB), VL+BMR
comparison + overconfidence fix (relative ranking only), numerical robustness. **No real data;
SBI deferred to v0.8.0+.** Phase numbering continues from 28 → v0.7.0 = Phases 29-32.
**Next:** `/gsd:plan-phase 29`. **PREREQUISITE for Phase 30 latent-circuit M3 runs:** fix the
Mutagen `models/` ignore (recreate session with anchored ignores).

<details><summary>v0.6.0 — SHIPPED 2026-06-10 (scope-cut), archived + tagged</summary>
**Phase:** All 34 plans executed (Phases 20-27 + retroactive Phase 28 VL engine).
**Status:** Goal-backward audit (`.planning/v0.6.0-AUDIT.md`) found every plan executed but the
**real-data scientific claims undelivered** (pivoted to synthetic or built-but-not-run). v0.6.0
**scope-cut** to its delivered core; real-data application **deferred to v0.7.0** (recorded as
deferred, NOT failed). User-approved both decisions 2026-06-10.

  - ✅ **Delivered:** Phase 20 synthetic recovery (via VL: A-RMSE 0.026, B-RMSE 0.0048, pooled-R²
    0.961, BMR 3/3) · Phase 21 CT-RNN · Phase 23 BMR (~93× faster) · Phase 27 pub artifacts ·
    **Phase 28 SPM12-grade VL inference engine** (the path that actually delivered v0.6.0 inference).
  - ⚠️ **Synthetic/infra-only → v0.7.0:** Phase 22 (pivoted from real Cam-CAN to synthetic OU;
    gates 1-2 unmet) · Phase 24 (real extractors + real parcellation built, never run on real
    weights; M/EEG pipeline scripts deleted in merge).
  - ✅ **Phase 25 HVAE-02 CONFIRMED 2026-06-10** — eval-only re-run (job 56331599) on the trained
    checkpoint reproduced RMSE 0.0761 + unmasked 0.4425 exactly, and **masked sign recovery 0.7745
    > 0.6 → PASS**. Phase 25 now 4/4. The 0.4425 was purely the `sign(0)` artifact.
  - ❌ **→ v0.7.0:** Phase 26 SBI SBC failed 2/9 (structural); real-M/EEG demo unplanned.
**Last activity:** 2026-06-10 -- Milestone audit + scope-cut: ROADMAP reconciled (stale
"0/N Planned" → audited status), Phase 28 VL consolidation written, 4 gaps triaged.

> **✅ Planning-doc drift RESOLVED 2026-06-10.** The post-consolidation Variational Laplace work
> (ForwardModel protocol, SVI→VL default, ReML M-step, `spm_dcm_csd_Q` precision, SVD reduction,
> SPM12 hyperpriors, analytical hemodynamic Jacobian, `LatentCircuitForward`) is now documented as
> **retroactive Phase 28** (`.planning/phases/28-variational-laplace-engine/28-CONSOLIDATION-SUMMARY.md`)
> inside v0.6.0 — it's what delivered v0.6.0's inference. The forward-looking VL *validation matrix*
> + *real-data application* is reserved for v0.7.0 (`.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`,
> Phases B–E) via `/gsd:new-milestone`.

</details>

**Milestone status:**
- v0.1.0 ✅ | v0.2.0 ✅ | v0.4.0 ✅ (Phase 17) | v0.5.0 ✅ (Phases 18-19) | **v0.6.0 ✅ shipped 2026-06-10 (scope-cut)**
- v0.3.0 ⏸ still in progress — Phase 16.1 (RECOV-04 B-RMSE diagnostic) never executed; the only genuinely-open prior milestone.
- v0.7.0 📋 next — VL validation + deferred real-data (`/gsd:new-milestone`).

## Decisions

- **[32-01-D1] The injected `DCM.Y.csd` is cell-wrapped `{squeeze(csd)}` only if `~iscell` in the MATLAB script.** `scipy.io.savemat` writes a bare numeric `(Nf,n,n)` complex array, but `spm_dcm_fmri_csd` expects `DCM.Y.csd` as a one-element cell block; the conditional wrap is idempotent and the comment cites `spm_dcm_fmri_csd.m` (when `DCM.Y.csd` is populated, the internal `spm_dcm_fmri_csd_data` MAR estimation is skipped and the supplied CSD used directly). NO transpose on either side: `DCM.Y.csd(w,i,j)` 1-based == Python `observed_csd[w-1,i-1,j-1]`, guarded by `tests/test_csd_injection_roundtrip.py`.
- **[32-01-D2] BOLD-less export synthesizes a `(len(freqs)*8, N)` float64 zeros `DCM.Y.y` placeholder.** Once `DCM.Y.csd` is injected the `y` values are unused, but `DCM.v`/`DCM.n` must stay valid; `v = len(freqs)*8` matches the spectral `order=8` so the struct is internally consistent. `bold_data` (optional) overrides only the `y` shape, never the injected csd.
- **[32-01-D3] mypy `np.ndarray [type-arg]` + `scipy [import-untyped]` on the new function/test are PRE-EXISTING file conventions, not gated.** Every function in `export_to_mat.py` uses bare `np.ndarray` (15 baseline errors → 20 after, all the same `[type-arg]` class) and the repo ships no scipy stubs. Honored the existing style rather than diverging one function; ruff is clean on both changed files. The plan's `validation/` ruff sweep also surfaced pre-existing dirt in `run_rdcm_validation.py`/`run_validation.py` (untouched files) — not in scope (consistent with 30-01-D4 scoping).
- **[32-01-D4] The MATLAB injection script is UNEXECUTED at 32-01.** MATLAB R2022a is installed locally but `matlab -batch` failed a license checkout (`-15,10032`), and full SPM estimation is Plan 32-03 by design (plan delivers script + sanity check only). Verified by grep (`DCM.Y.csd`, `DCM.Y.Hz`, `spm_dcm_fmri_csd`, `results.Ep_A/Cp/F`). CARRY-FORWARD to 32-03: confirm `DCM.Y.csd`-populated actually bypasses `spm_dcm_fmri_csd_data` in this SPM12 build, and that the `Ep.A(1,2)`/`Ep.A(2,1)` readout matches the injected asymmetric ground truth (0.15 / 0.10).
- **[32-02-D1] Strict 5% relative-F is the HARD default gate for VL-vs-SPM matched-F (BINDING user decision).** `compare_free_energies(vl_free_energy, spm_F, rel_tolerance=0.05)` returns `within_tolerance = bool(rel_err < rel_tolerance)` with `rel_err = abs(vl-spm)/max(abs(spm),1e-12)` — a pass/fail gate, NOT a descriptive report (overrides the research's softer fallback). It is single-problem-only (same priors/data/model, same CSD); its docstring forbids S3 cross-model absolute-F use, and cross-model agreement stays `compare_model_ranking` (relative ranking), pinned by `test_cross_model_ranking_is_separate_path`. The 5% target is only meaningful when both F are on the IDENTICAL CSD (same-CSD injection, Plan 32-01). The `within_tolerance` key is a contract consumed by Plan 32-03 (`run_vl_validation.py`) — do not change the signature/return.
- **[32-02-D2] No new mypy override; `compare_free_energies` returns bare `dict` to match every existing sibling comparator.** `compare_posterior_means`/`compare_model_ranking`/`compute_free_param_comparison` all annotate `-> dict:`; the new function follows the module's established pattern. mypy baseline 15→16 errors, the single delta being the same `[type-arg]` on bare `dict` the whole file already emits (no new error category; pre-existing scipy-stub + bare-generic noise). Scoped to the plan's files, consistent with 30-01-D4. The new test file introduces zero mypy errors of its own; ruff clean on both.
- **[32-03-D1] MATLAB binary from `config.MATLAB_PATH`; SPM12 via the `SPM12_PATH` child env var (single name across .m, sbatch, subprocess).** `run_vl_validation.py` resolves `[str(MATLAB_PATH), "-batch", ...]` and passes `dict(os.environ)` (carrying the sbatch-exported `SPM12_PATH`) to the MATLAB subprocess; the `.m` reads `getenv('SPM12_PATH')` with a local-default fallback (the ONLY change to the 32-01 file, loud `~exist` guard kept). One env-var name spans the .m, the sbatch, and the subprocess so laptop + M3 share one code path (the addendum prose's `DCM_SPM12_PATH` was reconciled to `SPM12_PATH`, matching the must-have/.m/sbatch/orchestrator instruction).
- **[32-03-D2] Cross-model ranking uses 3 a_mask scenarios, RELATIVE delta-F only (S3).** full-reciprocal (correct) / single-direction ([1,0] only) / diagonal-only, each re-fit on BOTH engines; `compare_model_ranking` compares only the relative ordering of F (key `"pyro_elbo"` = VL `free_energy[-1]`, higher=better). NEVER absolute F across masks, NEVER element-wise Cp. Single-direction may rank near diagonal-only — itself a valid agreement signal per the Phase 31 identifiability finding.
- **[32-03-D3] Pre-existing mypy bare-dict `[type-arg]` + `pyro_dcm`/scipy `[import-untyped]` are not gated.** Every `validation/` comparator returns bare `dict`; `pyro_dcm` ships no `py.typed`. Honored the file convention (consistent with 32-01-D3 / 32-02-D2); ruff clean on all 3 changed Python files.
- **[32-03-D4] The real `spm_nlsi_GN` cross-validation executes on M3; the laptop SPM-gated test auto-skips — both enforce the SAME orchestrator.** Local FlexLM -15 unreachable; MATLAB R2022a + SPM12 verified on the M3 comp partition. The strict 5% matched-F gate is HARD-asserted in the laptop test (which SKIPS without a license) and RECORDED (`matched_f_relative_error`, record-don't-crash per 31-03-D3) in the M3 JSON — both true: the test enforces it, the run reports the real number. A genuine 5%-F miss is a finding to ESCALATE, not to silently relax (the user chose the strict gate + same-CSD path to make it achievable).
- **[31-03-D1] `temper_vl_posterior` cannot break PD by positive scaling alone; the guard fires only on an already-indefinite input.** A positive scalar times a PD matrix stays PD, so an "over-large T" never breaks a clean posterior. The laptop PD-guard test (`tests/test_bmr_tempering_calibration.py`) therefore feeds a deliberately indefinite covariance (a symmetric matrix with one negative eigenvalue) so the Cholesky genuinely fails, asserting the message names the shape `(3,3)` and `tempering_factor=100.0`. The realistic PD break is captured on the cluster as the C2c cross-condition mode (T=2.0 calibrated on task-N4 breaks PD on task-N2). The plan's "over-large T that breaks PD" is realized exactly this way.
- **[31-03-D2] Chosen T is the smallest coverage-RAISING candidate even when the coarse ladder overshoots the band (in_band=False).** On the task-N4 stress re-fit seed, the (1,2,5,10,20,50,100) ladder jumps from coverage 0.875 (T=1) straight to 1.0 (T=2), so no candidate lands inside [0.90,0.98]; `select_tempering_factor` returns the closest-to-target (T=2.0, coverage 1.0) with `in_band=False` and never raises. The band [0.90,0.98] is a documented EXPLORATORY choice (research Open Question 3), not a validated schedule; a finer ladder would be needed to hit it exactly. Reported, not gated. The tempered top-K is identical to the untempered ([12,11,3,14,7,13]) — mild tempering preserves the BMR structure.
- **[31-03-D3] Cross-condition non-PD (C2c) is RECORDED as a structured result, not raised.** The first M3 run (job 56396691) aborted with status=error when T=2.0 broke PD on the held-out task-N2 posterior. Fixed (Rule 1, in the Task 2 cluster script): the held-out untempered ranking is computed unconditionally and only the tempered path is wrapped in a `ValueError` guard, recording `cross_condition_non_pd=true` / `topk_preserved=false` / `non_pd_message`, so the already-successful stress-cell calibration persists and the job finishes status=ok (job 56397206). The C2c is the scientifically interesting outcome (a T tuned on one condition is not PD-safe on another) — surfaced as data, never lost as a crash. Tempering remains EXPLORATORY; absolute delta-F never gated.
- **[31-02-D1] VLBMR-02 COMPLETE (2/2) — reciprocal-edge ground truth makes the brute-force VL-refit present>absent gate pass.** `tests/test_bmr_vs_vl_refit.py` (`@pytest.mark.vl`, commits bc0e33f Task 1, ac69897 Task 2; 1 passed ~35-41s laptop, ruff+mypy clean). The plan's prescribed SPARSE single-edge spectral ground truth was UNIDENTIFIABLE and the brute-force gate failed: (a) the **hemodynamic** spectral forward (`spectral_dcm_forward(hemodynamic=True)`, default) is insensitive to single off-diagonal A entries on a near-diagonal base (CSD diff ≈0; rel-diff 8e-32 vs 0.23 neural-only) → sparse/chain A → A_free collapses to 0, all ΔF degenerate (SAME phenomenon as 31-01-D1); (b) a denser non-reciprocal A fit is non-identifiable/rotated (true A[1,0]=0.4 recovers A_free[idx3]≈0; mass lands on A[1,2]/A[0,1]; overconfident posterior loads the "absent" edges) so the brute-force refit ranked absent>present (C1/S3 + 29-02-D1). **Fix (adopted 31-01-D1's identifiability pattern):** RECIPROCAL-edge ground truth (0↔1 + 1↔2 present at 0.3/0.25, 0↔2 absent). With it A is recoverable and BOTH methods rank present>absent with worst single-prune-model agreement and Spearman ρ=1.0 (BMR present -3.54e6 < absent -2.43e6; brute-force present -5.89 < absent -2.26). Worst-model gate restricted to the like-for-like single-prune subset (two-prune model reported but excluded — S3/C1 dimensionality confound). RANK-only, never absolute-ΔF equality. `test_bmr_vs_elbo.py` (SVI) untouched. Reciprocal-edge spectral ground truth is now the shared identifiability pattern across 31-01 + 31-02. **Carry-forward:** brute-force present>absent ordering is fragile on non-identifiable spectral ground truth; any task/latent cross-model confirmation must use identifiable topology + route to M3 (`@pytest.mark.slow`, >3-min laptop).
- **[31-01-D1] A feed-forward chain is UNIDENTIFIABLE by spectral DCM; VLBMR-01 ground truth uses RECIPROCAL edges.** The plan's feed-forward chain (N=2 `[(1,0)]`, N=4 `[(1,0),(2,1),(3,2)]`) produces a stationary CSD bit-identical to the empty graph (`||csd_chain-csd_zero||/||csd_zero|| = 0.0`), so VL collapses A_free to exactly zero and every single-prune delta-F is 0.0 (the spurious top-K `[3,7,11]` is float sign-noise = the transpose of the true edges, NOT an index bug — the S4 round-trip guard held). Switched to reciprocal edges (N=2 `[(0,1),(1,0)]` K=2; N=4 reciprocal chain K=6); recovery is 5/5. Real spectral-DCM identifiability property, carried forward to 31-03. Builder, plumbing, gate semantics, and the never-absolute-delta-F contract unchanged.
- **[31-01-D2] N=2 saturated-reciprocal has no absent prunable edge → `separation_after_rank==K` cut is degenerate, asserted conditionally.** With both off-diagonals present (`K == N*(N-1)`) there is no essential/non-essential boundary, so the cut lands at rank 1, not K. Gated that assertion on `has_absent_edges = K < N*(N-1)`; recovery + positive separation_gap asserted unconditionally. N=4 (K=6 < 12) exercises the full cut==K gate.
- **[31-01-D3] BMR separation_gap magnitudes (1e4–1e6 nats) are RELATIVE-ranking signal only.** They reflect VL Laplace overconfidence (pitfall C1); correctly NOT gated as absolute thresholds. This is the overconfidence regime tempering (31-03) targets.
- **[30-03-D1] `classify_cell` is pass-or-documented-limit, never silent (VLREC-04).** A cell PASSES iff every PRESENT check (RMSE_A <= 0.05, masked sign >= 0.80, coverage_95 >= 0.85) passes; a check whose metric is `None` is SKIPPED (`pass=None`), never auto-failed; a failing cell returns `status="identifiability_limit"` WITH an evidence block (shrinkage/coverage/RMSE IQR/convergence) — it NEVER raises. The classifier raises only on structurally malformed input (a contracted key absent). Thresholds are provisional documented defaults (no Fisher-info bound yet); SHRINKAGE_SOFT_TARGET 0.7 is informational evidence only (low shrinkage = expected Laplace overconfidence, job 55772525), never a gate.
- **[30-03-D2] Boundary regime characterized from `raw.max_real_eig_list`, not an explicit field.** The per-cell JSON has NO `boundary_rejections` field; VLROBUST-03 characterization uses the accepted-draw max-real-eig distribution (proximity to the `[-0.05,0]` band; 0 accepted draws in-band → exclusion held) plus the shrinkage-below-soft-target overconfidence flag. The aggregator excludes `recovery_matrix_local_*.json` and parses the array index from the `_<idx>.json` suffix (cell 9 used the array PARENT job id 56346424, cells 0-8 used per-task ids 5634644X).
- **[30-03-D3] Task-DCM cells (4-7) errored at the simulator, surfaced as errored cells.** All four task cells failed with torchdiffeq `underflow in dt 0.0` inside `simulate_task_dcm` (adaptive-step underflow) — consistent with the pre-existing task-path `dt_sim` fragility (29-05 note). VLREC-04 requires these be SURFACED (report + CSV `status=error`), not dropped; they are. Re-running task variant after fixing the simulator dt is a follow-up, NOT a 30-03 classifier defect. Spectral + latent_circuit coverage is available for Phase 31; task coverage is missing pending the simulator fix.
- **[30-01-D1] Per-region R2 is consumed, not computed, by `assemble_cell_metrics`.** The VL runners emit no per-seed trajectories, so the assembler reads a driver-supplied `r2_per_region_list` (the 30-02 driver calls `compute_trajectory_r_squared(pooled=False)`) and median-aggregates; it NEVER re-pools (guards the pooled-R2 artifact R1). Spectral/task have no trajectory -> `r2_per_region=None` + note. Same pattern for `shrinkage_list`/`coverage_list`: median when present, None+note when absent, never fabricated.
- **[30-01-D2] Near-boundary-A exclusion band [-0.05,0] is inclusive-rejected, exposed as `NEAR_BOUNDARY_LO/HI`.** `exclude_near_boundary_A` returns True (acceptable) only when max Re eig < -0.05 or > 0; `resample_A_until_accepted` drives a seeded closure and raises RuntimeError (tries count) if exhausted. Keeps ground truth inside the eig_clamp-injective regime (pitfall N2, VLREC-03).
- **[30-01-D3] Spectral SNR diverges from task/latent: `{'noise_log_amplitude':-log(snr)}` vs `{'SNR':snr}`.** `snr_for_model` is the one place SNR semantics differ across the three forward models; the 30-02 driver expands the spectral scalar into the `noise_params` b/c observation-noise tensors. Keeps the matrix SNR axis comparable.
- **[30-02-D1] The grid driver INLINES the per-variant VL simulate→fit loop (importing the same simulate/forward symbols the Phase 29 runners use) to thread per-cell SNR**, rather than forking the runners or using env-var globals. Plan's preferred "no globals" seam; keeps the SNR axis comparable while reusing all fit logic. SNR injected via `snr_for_model`: spectral overrides the `noise_params['b']` global observation-noise log-amplitude (index [0,0]); task/latent pass the `SNR` kwarg into the simulator.
- **[30-02-D2] Seeds run INSIDE one array task** (`config.n_datasets=GRID_SEEDS=10`), so 10 cells = 10 SLURM array tasks = 120 fits. Seeds are NOT separate array tasks (mirrors the runner per-seed loop). sbatch `--array=0-9` must equal `len(enumerate_cells())`.
- **[30-02-D3] `latent_circuit` collapses the N axis to fixed N=4** (its ground truth is the fixed bilinear topology); the grid emits 10 cells (spectral 4 + task 4 + latent 2), never a fabricated N=2 latent-circuit cell. Recorded as `n_axis_note` on those cells.
- **[30-01-D4] `# type: ignore[import-untyped]` on the `masked_sign_recovery` import; pyproject mypy config left untouched.** `pyro_dcm` ships no `py.typed` (same condition affects every existing pyro_dcm-importing benchmark); scoped the fix to the plan's declared file rather than adding a repo-wide mypy override.
- **[29-05-D1] VL determinism is contracted within-machine at atol 1e-8, NOT enforced via `torch.use_deterministic_algorithms`.** That mode raises on the engine's linalg ops (solve/slogdet/cholesky/matrix_exp); reproducibility is achieved via fixed seeds + identical inputs. Cross-machine (laptop vs M3 BLAS) may differ below atol ~1e-6, so Phase 30 must compare within-machine, not bitwise across machines. Documented in `docs/03_methods_reference/vl_determinism_notes.md`.
- **[29-05-D2] Multi-restart stays a test-local helper, not an engine feature.** `_multistart_spectral` re-seeds + re-fits from the prior start and selects highest final free energy; pitfall N4 means the restart PATH is reproducible but the selected mode is basin-dependent (not guaranteed global). Engine multi-restart wrapping remains out of scope.
- **[29-02-D1] rank_connections is purely relative — absolute delta-F is never a pass/fail criterion.** VL Laplace overconfidence (job 55772525: truly-absent edge scored delta_F=-115.9, indistinguishable by sign) drives every reduction deeply negative. Only relative ordering of K single-prune costs + a separation gap (largest consecutive drop on sorted ascending costs) are reported. Avoids pitfall C1 by construction.
- **[29-02-D2] temper_vl_posterior is a primitive only; calibration deferred to Phase 31.** Temperature scale + symmetrize + loud Cholesky PD guard (ValueError with shape + factor). Default factor 1.0 = backwards-compatible identity; calibrated factor determined against Phase 30 coverage curves.
- **[20-01-D1] hemodynamic=False as keyword-only after stability_check_every.** No positional break for existing callers; bit-exact backward compat preserved.
- **[20-01-D2] simulate_latent_circuit reuses _normalize_B_list/_normalize_stimulus_to_input_fn from task_simulator.** DRY: bilinear path is identical; private helpers imported directly.
- **[20-01-D3] Initial state torch.zeros(N) not make_initial_state (5N).** make_initial_state returns wrong shape for hemodynamic=False mode.
- **[20-03-D1] pyro.deterministic() appears as type='sample' in this Pyro version.** Tests must check by site name, not by type. Pattern documented in 20-03-SUMMARY.md key-decisions.
- **[20-03-D2] AutoIAFNormal hidden_dim must exceed latent_dim.** For N=4, M=1 model: latent_dim=21 (N^2 + N*M + 1). Use hidden_dim=[32] not [20] to avoid AutoRegressiveNN ValueError.
- **[20-03-D3] LC_A_PRIOR_VARIANCE=1/16 confirmed as separate constant from BOLD A prior (1/64).** Addresses pitfall LC4. Stored in latent_circuit_dcm_model as module-level constant, re-exported from models/__init__.py.
- **[20-04-D1] importlib.import_module required to access LC_*_PRIOR_VARIANCE for monkey-patching.** pyro_dcm.models.__init__ re-exports latent_circuit_dcm_model function under the submodule name; import-as resolves to function not module. Fixed with importlib.import_module("pyro_dcm.models.latent_circuit_dcm_model").
- **[20-04-D2] 100s / dt=0.01 ODE = ~16s per SVI step on laptop.** Full acceptance runs (1000+ steps, 10+ restarts, 10 seeds) must go to M3 cluster. Smoke test uses _duration_override=2.0 for API verification only.
- **[20-04-D3] All LC acceptance thresholds provisional.** A-RMSE 0.15, B-RMSE 0.20, sign recovery 0.80, CI coverage 0.85, trajectory R2 0.95. Plan 20-05 recalibration pending.
- **[21-01-D1] alpha = dt/tau is a plain float attribute, not nn.Parameter.** Fixed for v0.6.0; avoids accidental gradient computation through it; learnable timescales deferred.
- **[21-01-D2] Euler integration chosen over torchdiffeq for CT-RNN training.** Matches Langdon & Engel (2025) trainRNNbrain exactly; faster and deterministic for fixed-dt neurogym observations.
- **[21-01-D3] Langdon & Engel (2025) formal REF-ID deferred to Phase 25 (PUB-03).** Cited by author/year in docstring as interim placeholder.
- **[21-02-D1] neurogym labels shape is (T, B) not (T*B,).** ngym.Dataset() v2.3.1 returns labels as (seq_len, batch_size); must .reshape(-1) before CrossEntropyLoss and accuracy computation.
- **[21-02-D2] neurogym imported inside train_rnn/eval_rnn_performance only.** Optional dependency guard: try/except ImportError with install hint. Callers without neurogym can still import ContinuousTimeRNN.
- **[21-02-D3] Early stopping checks every log_every steps; 3 consecutive checks >= criterion_acc trigger return.** Count resets if accuracy drops below threshold at any log checkpoint.
- **[21-03-D1] Module docstring must precede `from __future__ import annotations`.** ruff E402 treats any non-import statement (including docstrings) before imports as breaking import-block contiguity; PEP 257 module docstring is first statement.
- **[21-03-D2] output_r_squared_gate fail test uses orthogonal equal-variance embedding.** Correlated factor mixing allows 1 PC to capture >90% variance; disjoint H/3-block basis guarantees PC1 ~33% and gate fails with N=1.
- **[21-03-D3] classify_stability parameter named jacobian_matrix to avoid shadowing module import.** Module imports `jacobian` from `torch.autograd.functional`; parameter named `jacobian` would shadow it inside the function.
- **[21-03-D4] extract_trajectories metadata stored as Python dict under `__meta__` key.** np.ndarray cannot hold heterogeneous scalar types (dt_seconds, tau, alpha); dict is correct container.
- **[20-02] n_restarts=1 path is bit-exact with pre-Phase-20 single-run path.** Return dict has exactly {losses, final_loss, num_steps}; no extended keys. Backward compat verified by existing test suite.
- **[20-02] guide_factory required when n_restarts>1 (ValueError on None).** Prevents silent reuse of a pre-trained guide across restarts.
- **[20-02] Param store restored via get_state/set_state after all restarts.** Avoids performance cost of re-running the best restart.
- **[20-01-D4] make_stable_latent_circuit_A uses self_inhibition=1.0 Hz (vs 0.5 Hz SPM12 default).** RNN latent states evolve faster than BOLD; stronger self-inhibition appropriate.
- **[19-02] t_eval for task_dcm_model constructed from DURATION+DT_MODEL, not from simulate_task_dcm output times.** Model contract requires t_eval spacing == dt; using simulation times_fine (DT_SIM=0.01) would violate this and make SVI prohibitively slow.
- **[19-02] Single-edge B mask (0->1 only) used in task DCM demo for clarity.** Simpler than demo_bilinear_consumer's two-edge mask; makes recovery metric output less cluttered for demo purposes.
- **[19-01] Demo scripts use simulate_* CSD for SVI fitting, not MNE noise epochs.** epochs_to_csd is demonstrated for IO bridge visibility (shapes printed); recovery metrics require ground-truth CSD from the generative model. Comments in script explain the distinction.
- **v0.6.0 phase structure = 6 phases (20-25).** Derived from 10 requirement categories clustered into 6 delivery boundaries. Phase 20 (14 reqs) is the scientific core; Phases 20 and 21 can run in parallel.
- **C_obs fixed at identity for v0.6.0.** Addresses pitfall LC5 (rotation ambiguity). Learned C_obs deferred to v0.7.0+.
- **Multi-start SVI (>=10 restarts) non-optional.** Addresses pitfall LC11; L&E uses 100.
- **Prior recalibration mandatory.** LC_A_PRIOR_VARIANCE separate from BOLD priors. Addresses pitfall LC4.
- **[22-01-D1] eig_clamp=None disables clamping entirely; -1.0 recommended for MEG.** Default -1/32 preserves fMRI behavior. None relies on parameterize_A upstream.
- **[22-01-D2] prior_a_var uses variance (not std) to match SPM12 convention.** Standard deviation computed as prior_a_var**0.5.
- **[22-03-D1] OU process uses Euler-Maruyama with dt=1/sfreq, not adaptive ODE solver.** Simple, fast, matches plan specification; spectral DCM linear model doesn't need adaptive stepping.
- **[22-03-D2] CSD consistency test uses 100 samples x 20s at 50 Hz with correlation > 0.5 threshold.** Finite-sample OU noise requires many realizations and lenient threshold; 50 Hz keeps test fast (~4s).
- **[23-01-D1] BMR delta_F uses Laplace approximation, not VFE difference.** delta_F = log p(mu_f|m_r) - log p(mu_f|m_f) + 0.5*[log|Sigma_r| - log|Sigma_f|]. No trace term. Validated against exact conjugate Gaussian Bayes factor.
- **[23-01-D2] BMR antisymmetry holds only for equal-covariance prior pairs.** When full and reduced priors differ only in mean (same cov), delta_F(A->B) = -delta_F(B->A) exactly. Different covariances break antisymmetry due to distinct reduced posterior precisions.
- **[24-02-D1] TRIBE v2 import guarded with try/except ImportError, not added to pyproject.toml.** Optional GPU dependency; requires A100; install via git URL.
- **[24-02-D2] Pipeline scripts use lazy imports after argparse.** Heavy torch/pyro imports slow; --help should be fast.
- **[24-02-D3] compute_empirical_csd with fs=1.0 Hz for TRIBE v2.** TRIBE v2 outputs at 1 Hz (fMRI TR); Nyquist at 0.5 Hz.
- **[25-02-D1] SVI smoke test uses windowed average (first 5 vs last 5 finite losses).** Early SVI steps produce NaN losses from ODE divergence; NaN guard prevents gradient corruption but losses are NaN. Windowed comparison is more robust.
- **[25-02-D2] packer.total_dim used (not n_features) for LatentCircuitDCMPacker.** Sparse packing attribute name differs from TaskDCMPacker's n_features.
- **[25-04-D1] KL annealing uses poutine.scale with mutable beta container.** SVI created once with scaled_model closure; beta_container[0] updated per epoch. Avoids SVI recreation overhead.
- **[25-04-D2] Beta floor is 1e-3 (not 0.0) at epoch 0.** When scale=0.0, poutine.scale zeros all log-probs, causing degenerate ELBO (all NaN). 1e-3 floor ensures valid gradients from first epoch.
- **[25-04-D3] KL estimated analytically from encoder z_loc/z_scale vs N(0,I).** Avoids Trace_ELBO decomposition; 0.5*(scale^2 + loc^2 - 1 - 2*log(scale)).sum() is exact for diagonal Gaussian vs standard normal.
- **[27-02-D1] Generated figures are gitignored; script is source of truth.** figures/*.png and figures/*.pdf excluded by .gitignore. Regenerate via `python scripts/generate_publication_figures.py`.
- **[29-03-D1] _TASK_PRECISION_MAX_DIM=5000 caps the dense (T*N,T*N) task-DCM precision.** TaskDCMForward.build_precision fails loud (ValueError with expected-vs-actual size) above the cap; enforces the dt>=0.1 floor (VLROBUST-02, pitfall N1). Tractable path unchanged.
- **[29-03-D2] C-order CSD index contract (j fastest, i, w) locked by regression test.** tests/test_csd_corder_roundtrip.py guards the commit-64e326f fix against silent column-major/transpose regression (VLREC-05, pitfall S4). Registered the `vl` pytest marker (was unregistered).
- **[29-01-D1] All four new BenchmarkConfig VL fields default to None.** `max_iter`, `hyperprior_mean`, `hyperprior_precision`, `prior_mean_a_offset` appended after `fixtures_dir` (preserving positional order); zero behavior change for every existing caller / quick_config / full_config / test (VLINFRA-01). Consumed only by VL runners (Plan 29-04).
- **[29-04-D1] Spectral VL runner passes context={"freqs": freqs}, not the plan's {}.** `SpectralDCMForward.predict` reads `context["freqs"]`; the VL engine injects `a_mask` itself. Empty context raises `KeyError`. task_vl uses `t_eval` at TR resolution + `dt=0.1` internal RK4 step so predicted-BOLD rows match observed and `T*N` stays << the 5000 precision cap (guard never trips).
- **[29-04-D2] Fixed latent TaskDCMForward.predict bug: integrate_ode uses step_size=, not options=.** Task VL was never exercised through a runner before; the invalid `options={"step_size": ...}` kwarg raised `TypeError`. Matches `LatentCircuitForward._integrate`. Orthogonal to the pre-existing `test_vl_forward_model_protocol.py` `dt_sim` signature-drift failures (those remain, out of scope).
- **[29-04-D3] LC smoke test uses max_iter=4 (slowest fit) to keep the 3-runner suite under the 3-min laptop budget.** Full N×SNR multi-seed sweep is Phase 30/M3; the smoke proves plumbing (dict shape + finite A-RMSE without raising), not recovery quality. Full vl suite = 113s laptop CPU.
- **[29-01-D2] MATLAB_PATH centralized in root config.py, env-overridable.** Default matches the hardcoded literal in validation/run_validation.py:58 (`C:/Program Files/MATLAB/R2022a/bin/matlab`); that file deliberately untouched (consuming refactor is Phase 32). Single source of truth for the SPM12 bridge (VLINFRA-05).
- Prior v0.3.0/v0.4.0/v0.5.0 decisions: see earlier STATE.md history in git log.

## Blockers

**AUDIT COMPLETE 2026-06-10 (`.planning/v0.6.0-AUDIT.md`).** The 4 gaps below are triaged.
**Nothing hard-blocks v0.6.0 completion under the scope-cut** — real-data gaps (Phase 22/24/26)
are formally **deferred to v0.7.0**; the only in-scope open item is the **HVAE-02 masked-metric
re-eval** (a <5-min M3 job, or accept-with-caveat). Recommended next action: optionally close
HVAE-02, then `/gsd:complete-milestone`. Detail retained below.

**Triage summary:**
1. **[20-05]** ✅ closed (VL). 2. **[25/HVAE-02]** ✅ CONFIRMED 2026-06-10 (masked 0.7745, job
56331599) — Phase 25 now 4/4. 3. **[26/SBI-03]** → v0.7.0 Phase D (structural, not a v0.6.0
deliverable). 4. **[24-01 parcellation]** ✅ No-Placeholders violation resolved; runtime
validation → v0.7.0. Plus **[vl-overconfidence-for-bmr]** → v0.7.0 Phase C.

**No remaining in-scope blockers — v0.6.0 is clean for `/gsd:complete-milestone`.**

<details><summary>Original 4-gap analysis (pre-audit, 2026-06-09 — retained)</summary>

1. **[Phase 20-05] ✅ FULLY CLOSED via Variational Laplace 2026-06-09 — SYNTH-01/02/03 all pass.**
   - **SYNTH-01/02** (job 56268248, 10 seeds): A-RMSE 0.026, **B-RMSE 0.0048** (vs SVI ~0.31),
     sign 1.00, CI cov 1.00, **pooled trajectory-R² 0.961**. R² "failure" was a metric bug
     (recovered R² == oracle R² → 0.95 unachievable); fixed via variance-pooled R², gate 0.95→0.90.
   - **SYNTH-03** (job 56270544, 3 seeds): BMR evidence ranking recovers the true chain {4,9,14}
     3/3 (sep 14×/13×/1.8×). Caveat: VL Laplace overconfidence suppresses *absolute* BMR pruning;
     *relative* ranking is the robust signal (→ todo on tempering VL posterior for BMR).

   <details><summary>Original SVI failure analysis (resolved; retained for the record)</summary>

   A-RMSE passes; B-RMSE, trajectory R², and ELBO model-selection fail (SVI). Full analysis in
   `20-05-SUMMARY.md`. Three distinct causes: (a) **B under-identified by experiment design** —
   the 50s CPU-feasibility rework + 80/20 split leaves only ONE 8s modulator window in
   training, so B collapses to ~0.31 RMSE (same pathology as unresolved Phase 16.1 ~0.34);
   (b) **R² fail is downstream** — held-out window contains a modulator epoch a collapsed-B
   model can't reproduce; (c) **ELBO model selection is methodologically invalid** — candidates
   N∈{2..6} fit datasets of different observed dimensionality so −ELBO scales with N and
   min-loss always picks N=2 (BMR/Phase 23 is the correct tool). **Tier-A methodology fixes
   APPLIED 2026-06-09:** modulator epochs retimed to fractions of duration (all in training
   split) + `compute_elbo_model_selection` gained a fail-loud cross-dimensional guard; covered
   by `tests/test_latent_circuit_metrics.py` (8 tests pass). **Tier-B decision:** use
   Variational Laplace (already full-covariance → no structured SVI guide needed).
   **`LatentCircuitForward` adapter BUILT 2026-06-09** (`pyro_dcm.inference.forward_models`):
   direct-obs + bilinear B + time-domain residual for `_run_vl_generic`, validated by
   `tests/test_latent_circuit_vl.py` (VL recovers A/B signs, full covariance, R²>0.7).
   **DONE:** `cluster/scripts/lc_vl_acceptance_run.py` + `lc_vl_acceptance.sbatch` ran as job
   56268248 (10 seeds, all gates pass); `lc_vl_bmr_selection.py` (job 56270544) closed SYNTH-03.

   </details>
2. **[Phase 25 / HVAE-02] ~RESOLVED 2026-06-09 — metric artifact (like 20-05 R²).** Sign
   recovery was computed over ALL 16 A_free entries; with ~6 structural zeros per matrix and
   `sign(0)=0` never matching a non-zero prediction, each zero is a guaranteed miss. 0.4425 =
   7.08/16 → **~0.71 masked** (passes >0.6). Fixed: added `masked_sign_recovery` (|A_true|>0.1,
   unit-tested) used by the train script. Remaining: add an eval-only path to recompute the
   EXACT masked number on the existing checkpoint (job, no retraining) — see todo.
3. **[Phase 26 / SBI-03] SBC calibration fails — DIAGNOSED 2026-06-09, structural.** Job 55772094:
   2/9 pass. Failure mode = **parameter-specific bias** (not under-training: 50k sims; not
   overconfidence). Fixed a real plumbing bug (`--num-transforms/--hidden-features/--max-epochs`
   never reached `train_npe`). Retrain with a larger flow (job 56274446) **still 2/9 — the bias
   just redistributed**, so capacity is NOT the cause: the miscalibration is **structural**
   (likely `eig_clamp` non-injectivity near the stability boundary). Next: restrict the prior to
   the stable region / reparameterize, or accept that **VL/SVI is the calibrated path** and SBI
   is an optional speed-up, not a v0.6.0 blocker. See `2026-06-09-sbi-sbc-calibration-gap.md`.
4. **[Phase 24-01] Parcellation placeholder — violates "No Placeholders" critical rule.**
   `src/pyro_dcm/foundation/parcellation.py:146` assigns vertices to ROIs by naive equal-size
   contiguous blocks instead of the real Schaefer atlas vertex-to-parcel mapping. Fetches real
   atlas labels but averages the wrong vertices → scientifically invalid ROI timeseries for any
   real Phase 24 foundation-model analysis. Needs the nilearn surface-projection pipeline.
   *(2026-06-10: RESOLVED — rewrite confirmed real by audit; runtime validation → v0.7.0.)*

</details>

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 | 2026-04-19 | 6bade20 | [001](./quick/001-cluster-sbatch-phase-16-acceptance/) |
| 002 | Structure-audit migration waves 1-5 | 2026-04-29 | 33e03e1 | [002](./quick/002-structure-audit-migration-waves-1-5/) |

### Pending Todos

6 pending — see `.planning/todos/pending/`. (HVAE-02 sign-recovery todo → `done/` 2026-06-10.)
- **Mutagen `models/` ignore (HIGH, INFRA)** — unanchored ignore excludes `src/pyro_dcm/models/`
  from M3 sync; recreate session with anchored ignores before v0.7.0 M3 runs
- Neural ODE DCM (Approach 2) — separate milestone v0.7.0+ after v0.6.0 informs whether bilinear suffices
- ROI projection for latent circuit DCM — map PCA circuit nodes back to brain ROIs; blocked on behavioral + neuroimaging dataset
- **Parcellation runtime validation → v0.7.0** — placeholder REMOVED; only nilearn runtime check remains
- **Phase 26 SBI-03 SBC calibration → v0.7.0 Phase D** — structural; VL is the calibrated path
- **VL overconfidence for BMR → v0.7.0 Phase C** — absolute prune threshold suppressed; relative ranking works

## Key Risks

- **[INFRA, surfaced 2026-06-10] Mutagen silently excludes `src/pyro_dcm/models/` from M3
  sync.** The `dcm-pytorch` session's unanchored `models/` ignore matches the source package;
  M3 was frozen at May 29. Invisible (reports "Watching", no conflict). Past impact nil (only
  the June-9 metric helper was stale; VL engine is in the synced `inference/`). Future hazard:
  edits under `models/` won't deploy. Stopgap: `scp` (path is ignored → safe). Fix: recreate
  session with anchored ignores — todo `mutagen-models-ignore`, memory
  `reference-mutagen-models-ignore-footgun`.


- **Bilinear misspecification (LC1).** Bilinear DCM is a first-order approximation of nonlinear RNN dynamics. Mitigated by linearization quality diagnostic and L&E nonlinear comparison.
- **Prior scale mismatch (LC4).** BOLD-calibrated priors wrong for RNN hidden states. Mitigated by mandatory recalibration on 5+ synthetic RNNs in Phase 20.
- **Rotational degeneracy (LC2).** PCA basis is arbitrary. Mitigated by Procrustes alignment and perturbation validation.
- **PCA discards task-relevant dynamics (LC3).** Mitigated by output-R-squared gate (>= 0.90) in Phase 21.
- **Multi-start convergence (LC11).** ELBO landscape has local optima. Mitigated by >=10 random restarts.

## Session Continuity

Last session: 2026-06-11 (executed Phase 32 Plan 03 — code-complete; M3 run PENDING orchestrator)
Stopped at: Completed 32-03-PLAN.md (VLSPM-03, the Phase 32 deliverable) — wrote + committed all 5
  artifacts: `validation/run_vl_validation.py` (`run_vl_spectral_dcm_validation`, reciprocal-asymmetric
  N=2, SPM-matched priors, same-CSD injection, Ep 10% / matched-F 5% / relative ranking >=0.80),
  the `.m` SPM12_PATH parameterization, `tests/test_vl_spm_cross_validation.py` (SPM-gated, collects +
  skips 1/0-errors on this laptop), and the M3 harness `cluster/scripts/spm_cross_validation.py` +
  `cluster/sbatch/spm_cross_validation.sbatch` (record-don't-crash). ruff/mypy clean (pre-existing
  conventions only). DID NOT submit to M3 / ssh / run the VL fit locally. **NEXT (orchestrator): submit
  `cluster/sbatch/spm_cross_validation.sbatch`, harvest `cluster/results/spm_cross_validation_<jobid>.json`,
  populate the headline matched-F relative_error placeholder in 32-03-SUMMARY.md + STATE, then
  `/gsd:verify-phase 32`.**
Resume file: None (M3 submission is the orchestrator's step, not a continuation gate)

Last session: 2026-06-11 (executed Phase 32 Plan 01)
Stopped at: Completed 32-01-PLAN.md — the SAME-CSD injection bridge. `export_spectral_dcm_csd_for_spm()`
  in `validation/export_to_mat.py` writes a Python `(F,N,N)` complex128 CSD + float64 `Hz` into
  `DCM.Y.csd`/`DCM.Y.Hz` (C-order, NO transpose, S4-guarded; BOLD-only exporter untouched);
  `validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m` forks `run_spm_spectral_dcm.m` to
  verify+inject the CSD, conditionally cell-wrap, bypass the MAR `spm_dcm_fmri_csd_data` recompute,
  print the `Ep.A(1,2)`/`Ep.A(2,1)` asymmetry, and save the identical `Ep_A/Cp/F` block;
  `tests/test_csd_injection_roundtrip.py` (2 @pytest.mark.vl) proves `[w,i,j]` round-trips with the
  transpose guard `loaded[0,0,1] != loaded[0,1,0]`. Gate (`test_csd_corder_roundtrip`) confirmed GREEN
  before building; combined 4 vl tests pass (2.97s); ruff clean on both changed files. MATLAB script
  UNEXECUTED (license `-15,10032`; full estimation is 32-03) — verified by grep. Decisions
  32-01-D1..D4. Commits a4582a3, 158bc22, d72a514. Branch: gsd/phase-32-spm12-cross-validation.
  **Next: Plan 32-03 (strict-5%-matched-F cross-validation) wires this bridge + `compare_free_energies`
  under MATLAB+SPM12 — the LAST v0.7.0 plan.**
Prior session: 2026-06-11 (executed Phase 32 Plan 02)
Stopped at: Completed 32-02-PLAN.md — VLSPM-02 strict-5% matched free-energy comparator.
  Added `compare_free_energies(vl_free_energy, spm_F, rel_tolerance=0.05)` to
  `validation/compare_results.py` (single-matched-problem relative-tolerance comparator; returns
  `{vl_free_energy, spm_F, relative_error, within_tolerance, rel_tolerance}`; 5% HARD pass/fail gate
  per binding user decision; docstring forbids S3 cross-model absolute-F, ties 5% to same-CSD
  injection Plan 32-01; no existing function modified). `tests/test_compare_free_energies.py`
  (5 @pytest.mark.vl tests, 1.48s laptop: within/outside-tol, custom tolerance, zero-F guard,
  cross-model-ranking-is-separate-path pinning S3). Decisions 32-02-D1/D2. ruff clean both files;
  mypy delta 15→16 = same pre-existing bare-`dict` `[type-arg]` pattern. Commits 4e1ed26 (feat),
  1a2a096 (test). Branch: gsd/phase-32-spm12-cross-validation. **NOTE: Plan 32-01 (parallel wave,
  separate agent) edits `validation/export_to_mat.py` + MATLAB/tests — no file overlap. Plan 32-03
  consumes `compare_free_energies`'s `within_tolerance` contract.**
  **Next: remaining Phase 32 plans (32-01 parallel wave-1, 32-03 wave-2 run_vl_validation), then
  `/gsd:verify-phase 32` — the LAST v0.7.0 phase.**
Prior session: 2026-06-11 (executed Phase 31 Plan 03)
Stopped at: Completed 31-03-PLAN.md — VLBMR-03 EXPLORATORY posterior-tempering calibration.
  `benchmarks/bmr_recovery.py` + `select_tempering_factor` (smallest-T-in-band coverage matcher;
  closest-to-target surfaced if none in band, never raises) + `tempered_vs_untempered_ranking`
  (side-by-side rank_connections via temper_vl_posterior — ALL tempering routes through the PD
  guard, no hand-rolled Cholesky). `tests/test_bmr_tempering_calibration.py` (@pytest.mark.vl,
  5 tests, 4.90s laptop). M3 glue `cluster/scripts/bmr_tempering_calibration.py` + `.sbatch`
  (single task, no pip, dt>=0.1, rk4). M3 job 56397206 (432s, status=ok): task-N4 stress cell
  single re-fit untempered cov95 0.875 → tempered 1.0 at chosen T=2.0 (in_band=False — coarse
  ladder overshoots [0.90,0.98]); tempered top-K identical to untempered ([12,11,3,14,7,13],
  sep_gap 5.13e5 nats unchanged). Held-out task-N2 cross-condition: SAME T=2.0 BROKE PD on the
  N=2 posterior — the concrete C2c hazard, recorded `cross_condition_non_pd=true` (not aborted).
  Absolute delta-F NEVER gated (C1/C2). Decisions 31-03-D1/D2/D3. ruff+mypy clean (laptop);
  ruff/ast/bash clean (cluster). Commits 974fe58 (Task 1), 2598d0f (Task 2, incl. the C2c
  error-handling fix), 004a0ab (Task 3 harvest). Branch: gsd/phase-31-bmr-validation-tempering.
  **Next: Phase 31 complete (3/3); `/gsd:verify-phase 31`, then Phase 32 (SPM12, local/MATLAB).**
Prior session: 2026-06-11 (executed Phase 31 Plan 02)
Stopped at: Completed 31-02-PLAN.md — VLBMR-02 BMR-vs-brute-force-VL-refit agreement. New
  `tests/test_bmr_vs_vl_refit.py` (@pytest.mark.vl, 1 test, ~35-41s laptop, ruff+mypy clean): one
  full-model spectral VL fit → analytic BMR ΔF for 3 single-prune reduced models → brute-force VL
  refits (a_mask-zeroed) → RANK-only gate (present>absent on BOTH methods + worst single-prune-model
  agreement) + Spearman ρ=1.0 report. KEY FINDING / DEVIATION (31-02-D1): plan's sparse single-edge
  spectral ground truth is UNIDENTIFIABLE (hemodynamic forward insensitive to single off-diagonal A;
  VL collapses A_free→0; denser non-reciprocal A is rotated/overconfident → brute-force ranks
  absent>present, C1/S3). Fixed by adopting 31-01's RECIPROCAL-edge ground truth (0↔1+1↔2 present,
  0↔2 absent) → both gates pass. Worst-model gate on single-prune subset only (two-prune excluded,
  S3/C1 dim confound). test_bmr_vs_elbo.py (SVI) untouched. Commits bc0e33f (Task 1), ac69897 (Task 2).
  **Next: 31-03 (tempering calibration) reuses benchmarks/bmr_recovery.py + reciprocal ground truth.**
  Branch: gsd/phase-31-bmr-validation-tempering.
Prior session: 2026-06-11 (executed Phase 31 Plan 01)
  Completed 31-01-PLAN.md — VLBMR-01 recovery harness (the PRIMARY defensible BMR result).
  `benchmarks/bmr_recovery.py` (make_sparse_ground_truth_A + offdiag_indices + bmr_tensors_from_vl_result,
  full A_free cov slice sigma_post[:N*N,:N*N]) + `tests/test_bmr_vlbmr01_recovery.py` (@pytest.mark.vl,
  spectral N=2/N=4, 5 seeds, 5/5 recovery, 112.7s laptop). RELATIVE ranking + separation gap, never
  absolute delta-F (C1). KEY FINDING: feed-forward chain A is unidentifiable by spectral CSD (bit-identical
  to empty graph) → ground truth uses RECIPROCAL edges. ruff+mypy clean. Commits c081c5c, 766dd24.
Earlier session: 2026-06-10 (executed Phase 30 Plan 03)
Earlier-prior session: Completed 30-03-PLAN.md — POST-RESULTS harvest + classifier + report over the COMPLETE M3
  sweep (job 56346424). `benchmarks/recovery_matrix_thresholds.py` (classify_cell pass | identifiability_limit
  with evidence) + `cluster/scripts/recovery_matrix_aggregate.py` (matrix CSV/JSON + report + eig_clamp
  regime) + `tests/test_recovery_matrix_aggregate.py` (4 vl tests) + `30-RECOVERY-MATRIX-REPORT.md`.
  REAL VERDICT: 5 PASS / 1 identifiability-limit (latent cell 8, marginal RMSE 0.0501) / 4 ERRORED task
  cells surfaced (torchdiffeq underflow). Phase 30 CLOSED (3/3); VLREC-04 + VLROBUST-03 satisfied.
  Commits 29978dd, 32a93ed, 819201e, eb30643. ruff+mypy clean.
  **Next: `/gsd:plan-phase 31` (BMR tempering calibration) — consumes recovery_matrix.json coverage.
  CARRY-FORWARD: task-DCM coverage is MISSING (cells 4-7 errored); if Phase 31 needs task coverage, fix
  the simulate_task_dcm adaptive-step underflow and re-run those cells on M3 first.**
Prior session: 2026-06-10 (executed Phase 30 Plan 02)
Earlier-prior: Completed 30-02-PLAN.md — recovery-matrix sweep driver + M3 submission.
  `benchmarks/recovery_matrix_grid.py` (10-cell grid, `run_one_cell` reusing Phase 29 VL fit logic + SNR/
  boundary/metric wiring), `cluster/scripts/recovery_matrix_cell.py` (env-driven SLURM entrypoint),
  `cluster/sbatch/recovery_matrix_sweep.sbatch` (array 0-9, no-pip, dt≥0.1). LOCAL faithfulness pre-check
  PASSED (28.7s, valid JSON). SUBMITTED M3 array job **56346424** (120 fits, all 10 tasks RUNNING; latent
  cells confirmed synced via the Mutagen models/ fix). Results → cluster/results/recovery_matrix_56346424_
  <0..9>.json. ruff+mypy clean. Commits 5c7547a (grid), 10918ce (cluster).
  Next: 30-03 is the POST-RESULTS harvest+classifier — run AFTER job 56346424 completes (monitor via
  `ssh m3 "squeue -u aman0087 --name=recov_matrix"`; results sync back via Mutagen). The driver loop was
  NOT re-run for all 120 fits locally (multi-hour, routed to M3 per project rule); the per-cell path is
  proven by the 28.7s local pre-check + M3 in-env import sanity check.
Prior session: 2026-06-10 (executed Phase 30 Plan 01) — hardened per-cell recovery-metric assembler
  (`benchmarks/recovery_matrix_metrics.py` + tests, 7 vl tests). assemble_cell_metrics + near-boundary
  exclusion + snr_for_model. Commits 6ae82cd, 9fb55f4.
Earlier session: 2026-06-10 (executed Phase 29 Plan 05) — VL determinism regression suite
  (`tests/test_vl_determinism.py`, 5 `@pytest.mark.vl` tests, ~2m42s laptop): fixed-seed determinism
  across spectral/task/latent-circuit (same seed -> posterior means equal within atol 1e-8, bitwise
  preferred), seed-sensitivity guard, multi-restart reproducibility (pitfall N4). Methods note
  `docs/03_methods_reference/vl_determinism_notes.md` documents the within-machine determinism contract
  + non-determinism sources (BLAS order, float64 accumulation, rk4 ODE, FD step N5) + cross-machine
  caveat. VLROBUST-01. Commits ed71f9c (tests), b0bfd6e (docs). ruff+mypy clean on the new file.
  Prior 29-04 (commits 372e203, 6a09579, a731fd5): three `method="vl"` runners. 29-01/02/03 prior.
Next: Phase 29 is the final v0.7.0 infra phase before the Phase 30 recovery sweep; check ROADMAP for
  whether 29 has further plans, else `/gsd:plan-phase 30`.
  Note: pre-existing failures in tests/test_vl_forward_model_protocol.py task-DCM cases
  (make_block_stimulus/simulate_task_dcm `dt_sim` signature drift) predate 29-03/29-04/29-05 (confirmed
  on baseline a064e69) — worth a cleanup pass; NOT introduced by 29-05. INFRA reminder: fix the Mutagen
  `models/` ignore before any v0.7.0 M3 latent-circuit run.
Resume file: None
