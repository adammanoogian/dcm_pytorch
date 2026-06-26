---
phase: 33-cmc-core-dynamics-spm-int-l-integrator-single-source-parity
plan: 02
subsystem: validation
tags: [cmc, erp, spm12, matlab, spm_int_L, spm_fx_cmc, fixtures, m3, parity, eeg-meg]

# Dependency graph
requires:
  - phase: 33-01
    provides: "Frozen single-source reference P-struct shape + x_test/u_test/grid, the pure-torch cmc_f / spm_int_L port to assert against"
  - phase: 32-spm12-cross-validation
    provides: "The SPM12 MATLAB bridge pattern (env-var addpath, loud try/catch, subprocess -batch, record-don't-crash, int64->double spm_Ce fix) mirrored here"
provides:
  - "validation/export_to_mat.py::export_erp_dcm -- single-source CMC-ERP DCM .mat exporter (additive)"
  - "validation/matlab_scripts/spm_fx_cmc_nodelay.m -- nargout-aware 2-output D=I wrapper"
  - "validation/matlab_scripts/run_spm_erp_dcm.m -- the 5-array fixture generator + provenance meta"
  - "cluster/scripts/erp_cross_validation.py + cluster/sbatch/erp_cross_validation.sbatch -- M3 harness"
  - "validation/data/erp_single_source_fixtures.mat -- the FROZEN vs-SPM ground truth (5 arrays + meta, D=1, x0==0)"
affects: [33-03 parity-gate, 34-extrinsic-coupling, 35-lead-field, 36-erp-dcm-model]

# Tech tracking
tech-stack:
  added: []  # zero new deps (B4); pyproject.toml unchanged
  patterns:
    - "Force spm_int_L D=I via a 2-output wrapper (nargout(M.f)==2), NOT by stripping P.D"
    - "Wrapper must be nargout-aware: spm_fx_cmc differentiates M.f via spm_diff(M.f,...), so an unconditional 2-output body self-recurses"
    - "Frozen Gaussian drive written into DCM.U.u and integrated verbatim (spm_int_L does not regenerate the input)"
    - "Fixture .mat committed to git (validation/data/ is mutagen-ignored; byte-frozen ground truth needs a durable home)"

key-files:
  created:
    - validation/matlab_scripts/spm_fx_cmc_nodelay.m
    - validation/matlab_scripts/run_spm_erp_dcm.m
    - cluster/scripts/erp_cross_validation.py
    - cluster/sbatch/erp_cross_validation.sbatch
    - validation/data/erp_single_source_fixtures.mat
    - cluster/results/erp_cross_validation_57884677.json
  modified:
    - validation/export_to_mat.py  # append-only: export_erp_dcm + _erp_gaussian_u_grid

key-decisions:
  - "[33-02-D1] The D=I wrapper must guard on nargout to avoid spm_diff(M.f) self-recursion"
  - "[33-02-D2] The Gaussian drive is frozen into DCM.U.u by the exporter and integrated verbatim by spm_int_L"
  - "[33-02-D3] Commit the fixture .mat to git because validation/data/ is mutagen-ignored"

patterns-established:
  - "Pattern: per-stage MATLAB probe on M3 (single-quoted chars, absolute paths under run()) to localize a failing fixture line"
  - "Pattern: nargout-aware SPM fx wrapper for clean D=I forcing"

# Metrics
duration: ~70min (incl. M3 license-gated checkpoint + recursion diagnosis)
completed: 2026-06-26
---

# Phase 33 Plan 02: MATLAB Fixture Generation (D=1) + Export Bridge + M3 Harness Summary

**The byte-frozen single-source CMC-ERP SPM12 fixtures (f_field, J0, dtJ, Eexp, Q_update, y_states + provenance meta with D=1 / x0==0 / SPM $Id) generated on M3 via a nargout-aware 2-output wrapper that forces spm_int_L's delay operator to exact identity, plus the additive `export_erp_dcm` exporter and the M3 cluster harness — the vs-SPM ground truth the Wave-3 parity gate asserts against.**

## Performance

- **Duration:** ~70 min (including the M3 license-gated human-action checkpoint and the recursion diagnosis)
- **Started:** 2026-06-25T21:29Z
- **Completed:** 2026-06-26 (M3 job 57884677, fixtures synced back)
- **Tasks:** 3 (2 autonomous code-prep + 1 M3 fixture run via the gated checkpoint)
- **Files modified/created:** 7 (6 new, 1 append-only)

## Accomplishments
- Appended `export_erp_dcm` (+ `_erp_gaussian_u_grid`) to `validation/export_to_mat.py` (additive; existing task/spectral/rDCM exporters byte-untouched) — writes the frozen single-source DCM `.mat`: `P{T,G,C,S,R}` (no `D`/`A`), `M.x=zeros(1,8)`, `U.dt=0.004`, `ns=128`, `M.ons=60`, `M.dur=16` explicit, dims as float64 (int64->double `spm_Ce` footgun), with `x_test=0.1*ones(1,8)` / `u_test=32.0` in `DCM.meta`.
- Authored the MATLAB fixture generator `run_spm_erp_dcm.m` and the D=I wrapper `spm_fx_cmc_nodelay.m`, plus the M3 harness (`erp_cross_validation.py` + `.sbatch`) mirroring the Phase-32 bridge (env-var addpath, loud try/catch, `matlab -batch`, record-don't-crash, `--partition=comp`, no pip).
- Diagnosed and fixed a MATLAB infinite-recursion OOM on M3 (the first job, 57882745, failed): the wrapper had to be made nargout-aware because `spm_fx_cmc` builds its analytic Jacobian via `spm_diff(M.f,...)`.
- Generated the fixtures on M3 (job **57884677**, 16 s): all 5 arrays at the expected shapes (float64), `meta.D=1`, `nargout_Mf=2`, `x0==zeros(8)`, `dt=0.004`, `ns=128`, `ons=60`, `dur=16`, `u_test=32`; SPM `$Id` `spm_int_L.m 7143` and `spm_fx_cmc.m 7279`. `checks_pass=true`. Synced back via scp (mutagen ignores `validation/data/`) and re-verified locally.

## Frozen fixture provenance (for Plan 33-03 to assert against the SAME values)

- **M3 job:** 57884677 (comp partition, MATLAB R2022a + `/home/aman0087/fc37/Carrick/spm12`)
- **SPM `$Id`:** `spm_int_L.m 7143 2017-07-29` · `spm_fx_cmc.m 7279 2018-03-10` · `spm('Ver')=SPM12`
- **Arrays (all float64):** `f_field (8,)`, `J0 (8,8)`, `dtJ (8,8)`, `Eexp (8,8)`, `Q_update (8,8)`, `y_states (128,8)`
- **meta:** `D=1`, `nargout_Mf=2`, `x0=zeros(8)`, `dt=0.004`, `ns=128`, `ons=60`, `dur=16`, `sus=0`, `x_test=0.1*ones(8)`, `u_test=32.0`, `exp_shift=exp(-16)`, `u_grid` (the exact Gaussian drive integrated)
- **Cross-check:** `dtJ == dt*(J0 - I*exp(-16))` holds to 1e-15 (the regulariser-before-Q convention is in the fixture).
- **Wave-3 tolerances (from 33-RESEARCH):** `f_field`/`J0` ≤1e-10, `matrix_exp(dtJ)` vs `Eexp` MEASURED (do NOT assume; ~1e-12 expected), `Q_update` ≤1e-9, `y_states` ≤1e-8. Assert in ladder order; a `y_states`-only failure localizes to loop ordering, not algebra.

## Task Commits

1. **Task 1: Append export_erp_dcm to export_to_mat.py** - `e116440` (feat)
2. **Task 2: MATLAB fixture scripts + M3 cluster harness** - `ca2d6dc` (feat)
3. **Task 3 fix: nargout-aware D=I wrapper (break spm_diff recursion)** - `2a25e88` (fix)
4. **Task 3 artifact: frozen fixtures + provenance JSON (M3 job 57884677)** - `94b603d` (feat)

## Files Created/Modified
- `validation/export_to_mat.py` - append-only `export_erp_dcm` + `_erp_gaussian_u_grid` (numpy `spm_erp_u` port)
- `validation/matlab_scripts/spm_fx_cmc_nodelay.m` - nargout-aware 2-output D=I wrapper
- `validation/matlab_scripts/run_spm_erp_dcm.m` - fixture generator (5 arrays + meta, x0==0 assert, nargout==2 assert, $Id capture)
- `cluster/scripts/erp_cross_validation.py` - M3 entrypoint (export input, `matlab -batch`, round-trip + validate, record-don't-crash)
- `cluster/sbatch/erp_cross_validation.sbatch` - comp/16G/1h, exports MATLAB_PATH + SPM12_PATH, no pip
- `validation/data/erp_single_source_fixtures.mat` - the frozen ground truth (committed; mutagen-ignored dir)
- `cluster/results/erp_cross_validation_57884677.json` - provenance record (checks_pass=true)

## Decisions Made
- **[33-02-D1] The D=I wrapper must guard on `nargout`.** `spm_fx_cmc` computes its analytic Jacobian via `spm_diff(M.f,x,u,P,M,1)` (`spm_fx_cmc.m:208`) — it differentiates `M.f`, which we set to the wrapper. An unconditional `[f,J] = spm_fx_cmc(...)` body forces the 2-output (Jacobian) path on every call, so `spm_diff`'s 1-output probe never short-circuits `spm_fx_cmc`'s `if nargout<2,return` and it self-recurses (OOM). The fix: `if nargout<2, f=spm_fx_cmc(...); else [f,J]=spm_fx_cmc(...); end`. `nargout('spm_fx_cmc_nodelay')` still returns 2, so `spm_int_L` keeps `D=1` (verified on M3). This is the standard SPM single-output termination mechanism, preserved.
- **[33-02-D2] The Gaussian evoked drive is frozen into `DCM.U.u` by the exporter and integrated verbatim.** `spm_int_L(P,M,U)` uses `U.u` directly (it does NOT call `spm_erp_u`), so the numpy `_erp_gaussian_u_grid` port (`spm_erp_u.m:42-64`, 32x scaling, `sus=0`) is what both SPM and torch consume. The grid is also saved as `meta.u_grid` so Wave 3 is self-contained from the fixture. This makes the input identical on both sides regardless of any `erp_input.py`-vs-`spm_erp_u` micro-difference (which Wave 3 can additionally regression-check against `meta.u_grid`).
- **[33-02-D3] The fixture `.mat` is committed to git.** `validation/data/` matches the mutagen session's `data/` ignore (the same class as the documented `models/` footgun), so it never syncs back; I scp'd it and committed it (12K). It is byte-frozen SPM ground truth whose regeneration requires licensed MATLAB on M3, so a durable in-repo home is justified despite the repo's general "don't track .mat" convention.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Infinite-recursion OOM in the D=I wrapper (MATLAB)**
- **Found during:** Task 3 (M3 fixture run; first job 57882745 failed)
- **Issue:** `run_spm_erp_dcm.m` got past the `M.f`/x0 asserts then threw "Out of memory. The likely cause is an infinite recursion" inside `spm_int_L`'s `y_states` call. Root cause: `spm_fx_cmc` builds its Jacobian via `spm_diff(M.f,...)` (it differentiates `M.f`, = the wrapper); the original unconditional `[f,J]=spm_fx_cmc(...)` wrapper body forced the 2-output path on every call, so `spm_fx_cmc` re-entered `spm_diff(M.f) -> wrapper -> spm_fx_cmc(2 outputs) -> ...` forever (stack `spm_fx_cmc_nodelay:22 -> spm_diff:92 -> spm_fx_cmc:208`). `f_field`/`J0` passed because `J0` differentiates `@spm_fx_cmc` directly.
- **Fix:** Made the wrapper nargout-aware so a 1-output probe short-circuits `spm_fx_cmc`'s Jacobian block; `nargout(M.f)` still returns 2 → `D=1` preserved.
- **Files modified:** `validation/matlab_scripts/spm_fx_cmc_nodelay.m` (also fixed a cosmetic `N state(s)` fprintf in `run_spm_erp_dcm.m`)
- **Verification:** M3 per-stage probe then full job 57884677: `nargout(M.f)=2`, `y_states [128,8]`, all fixture checks pass.
- **Committed in:** `2a25e88` (the fix)

---

**Total deviations:** 1 auto-fixed (1 bug). The fix was a genuine MATLAB-semantics bug in the wrapper, surfaced only under a licensed SPM run (local FlexLM -15); not a design change — D=1 stays forced via the 2-output wrapper, parity stays vs-SPM.
**Impact on plan:** No scope creep, no architectural change. The fixtures match the plan's required shapes + meta exactly.

## Issues Encountered
- **mutagen `data/` ignore** silently dropped `validation/data/` from sync (same class as the `models/` footgun) — the produced `.mat` never came back via mutagen; worked around with `scp` + a git commit. Carry-forward: anchoring that ignore (e.g. `/data/` instead of `data/`) would let `validation/data/` sync; not changed here (out of scope, user's mutagen session config).
- **MATLAB string-vs-char during ad-hoc probing:** double-quoted literals create `string` objects that break SPM's `==` char comparisons. Probes must use single quotes (the committed scripts already do); `run()` cd's to the script dir, so probe scripts need absolute repo paths.

## Next Phase Readiness
- **Plan 33-03 (the parity gate) is unblocked:** `validation/data/erp_single_source_fixtures.mat` is in-repo with the 5 frozen arrays + full provenance meta; the SPM `$Id`s and frozen `x_test`/`u_test`/`dt`/`ns`/`dur` are recorded above for the test to assert the same provenance.
- The `matrix_exp(dtJ)` vs `Eexp` floor is now MEASURABLE (both arrays are in the fixture) — Wave 3 sets that tolerance empirically (do NOT assume) before gating `Q_update`/`y_states`.
- No blockers. Compute routed correctly: MATLAB (license-gated, >3 min incl. startup) ran on M3, never the laptop.

---
*Phase: 33-cmc-core-dynamics-spm-int-l-integrator-single-source-parity*
*Completed: 2026-06-26*
