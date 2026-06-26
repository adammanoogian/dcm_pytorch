---
phase: 34-extrinsic-coupling-condition-b-multi-source-evoked-integration
plan: 02
subsystem: validation
tags: [cmc, erp, spm12, multisource, mmn, spm_gen_Q, spm_gen_erp, spm_fx_cmc, matlab, m3, float64, fixtures]

# Dependency graph
requires:
  - phase: 34-01
    provides: "the locked free-param dict schema (T,G,C,S,R,A list[4],B list; x_design (Cnd,n_effects)); parameterize_cmc_network / apply_condition_modulation / cmc_network_f / simulate_erp_dcm the Wave-3 ladder asserts against these fixtures"
  - phase: 33-02
    provides: "the spm_fx_cmc_nodelay.m D=1 nargout-aware wrapper (REUSED UNCHANGED), the run_spm_erp_dcm.m + erp_cross_validation.py + sbatch single-source bridge this mirrors additively"
provides:
  - "export_erp_dcm_multisource: the 5-source MMN DCM input .mat writer (locked A/B/C masks + X, P.A/P.B as MATLAB cells, all dims float64)"
  - "run_spm_erp_dcm_multisource.m: per-condition spm_gen_Q QA/QG + per-condition frozen J0/Qupd + multi-source spm_gen_erp-loop trajectory y (D=1 + x0==0 asserted)"
  - "erp_cross_validation.py --mode multisource + erp_cross_validation_multisource.sbatch (M3 entrypoint, record-don't-crash)"
  - "validation/data/erp_multisource_fixtures.mat: the byte-frozen 5-source MATLAB ground truth (QA, QG, J0, Qupd, y, meta) committed to git"
affects:
  - "34-03 (Wave 3 multi-source parity ladder: asserts apply_condition_modulation QA/QG + cmc_network_f J0 + simulate_erp_dcm trajectory against this fixture, laptop, gated on the .mat)"
  - "35 (single-dipole lead-field reads the source-state trajectory shape (ns,8N))"
  - "36 (5-source MMN precision-sweep demo built on the diag(B)->Q.G(:,1) knob this fixture pins)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive sibling exporter: export_erp_dcm_multisource APPENDED (single-source export_erp_dcm byte-untouched, append-only diff)"
    - "Cluster --mode dispatch: argparse {single,multisource}, default single keeps the Phase-33 entrypoint byte-unchanged; main_multisource mirrors main"
    - "The spm_gen_erp loop body (spm_gen_Q -> spm_int_L per condition) gives SOURCE states (ns,8N) WITHOUT the lead field (Phase 35) -- the parity target for cmc_network_f"

key-files:
  created:
    - validation/matlab_scripts/run_spm_erp_dcm_multisource.m
    - cluster/sbatch/erp_cross_validation_multisource.sbatch
    - validation/data/erp_multisource_fixtures.mat
    - tests/test_export_erp_multisource.py
  modified:
    - validation/export_to_mat.py
    - cluster/scripts/erp_cross_validation.py

key-decisions:
  - "34-02-D1: store the per-condition SOURCE-state trajectory (the spm_gen_erp.m:69-86 loop body = spm_gen_Q -> spm_int_L per condition, D=1) NOT spm_gen_erp's channel-space L*x output -- the lead field is Phase 35, and the Wave-3 source-state parity target is cmc_network_f / simulate_erp_dcm which produce source states"
  - "34-02-D2: P.S is a SCALAR (1,1), not per-source (5,1) -- spm_cmc_priors.m:124 has E.S=0 and spm_fx_cmc.m:93 forms F=sigmoid(-R*x) where -R*x is a MATRIX multiply, so a (5,1) S makes (5,1)*(5,8) fail; (1,1) zero is numerically identical to the per-source-uniform S=0 the torch schema uses"

patterns-established:
  - "5-source MMN topology encoded as explicit [to,from] edge masks (forward/backward/lateral) + mask*32-32 free-log convention (live 0.0, dead -32.0); NO MNI coords (Phase 36)"
  - "B-folding teeth: B distinct from A on edges + non-zero diag at precision nodes, so the Wave-3 element-wise QA=A+X*B and QG=G(:,1)+X*diag(B) checks are non-trivial"

# Metrics
duration: 45min
completed: 2026-06-26
---

# Phase 34 Plan 02: Multi-Source (5-source MMN) MATLAB Fixtures + Export Bridge Summary

**The byte-frozen 5-source auditory-MMN SPM12 ground truth — per-condition `spm_gen_Q` `QA`/`QG`, per-condition frozen `J0`/`Qupd`, and the multi-source source-state trajectory `y` — generated on M3 under MATLAB R2022a + SPM12 with delays forced off (D=1, `nargout(M.f)==2`) and `x0==zeros(5,8)` asserted, harvested + committed to git for the Wave-3 parity ladder.**

## Performance

- **Duration:** ~45 min (incl. one M3 debug loop)
- **Completed:** 2026-06-26
- **Tasks:** 3 (2 laptop authoring + 1 M3 submit/harvest checkpoint)
- **Files modified:** 6 (4 created, 2 appended)
- **M3 jobs:** 57896254 (FAILED, the P.S bug) -> 57896525 (COMPLETED, 15.1s, exit 0, checks_pass=true)

## Accomplishments

- **Locked the canonical 5-source auditory-MMN reference TOPOLOGY** (sources A1L, A1R, STGL, STGR, rIFG; NO MNI coords — those are Phase 36) as explicit `(5,5)` `[to,from]` masks in `export_erp_dcm_multisource` + `DCM.meta`:
  - Forward `A{1}`/`A{2}`: A1L->STGL, A1R->STGR, STGL->rIFG, STGR->rIFG.
  - Backward `A{3}`/`A{4}`: rIFG->STGL, rIFG->STGR, STGL->A1L, STGR->A1R.
  - Lateral reciprocal STGL<->STGR (added to the forward blocks; triggers the `(1+4L)` reduction, `spm_fx_cmc.m:79-82`).
  - Input `C` drives A1L + A1R only; condition `B{1}` on every extrinsic edge + non-zero `diag(B)` at rIFG/A1L/A1R; `X=[[0],[1]]` (standard/deviant).
- **`export_erp_dcm_multisource`** (APPENDED, single-source `export_erp_dcm` byte-untouched): `P.A`/`P.B` as MATLAB cells (`np.empty((1,k),object)`), `U.X` `(2,1)` double, `M.x=zeros(5,8)`, `M.n=40`, `M.f='spm_fx_cmc_nodelay'`, ALL dims float64 (int64->`spm_Ce` footgun, Phase 32 a27828b). Laptop loadmat round-trip test green.
- **`run_spm_erp_dcm_multisource.m`** reuses the `spm_fx_cmc_nodelay.m` D=1 wrapper UNCHANGED (no recursion bug), ASSERTS `nargout(M.f)==2` + `isequal(M.x,zeros(N,8))`, and generates per condition: `QA{c}` (cell of 4 `(5,5)` `spm_gen_Q` blocks), `QG{c}` (`(5,)` `Q.G(:,1)` precision column), `J0{c}`/`Qupd{c}` `(40,40)`, `y{c}` `(128,40)`.
- **M3 GREEN:** job 57896525 `checks_pass=true` — `meta.D=1`, `nargout_Mf=2`, `N=5`, `x0==zeros(5,8)` all-zero; SPM `$Id` `spm_fx_cmc 7279` / `spm_gen_Q 7279` / `spm_int_L 7143` / `spm_gen_erp 6427`. Fixture scp'd back (validation/data/ mutagen-ignored) + committed.
- **Verified the EVOK-02 precision mechanism directly in the fixture:** `QG` standard `[0,0,0,0,0]` vs deviant `[0.5,0.5,0,0,0.5]` = exactly `diag(B)` at A1L/A1R/rIFG; `QA` standard != deviant (B folded into all four A blocks on the deviant).

## Task Commits

1. **Task 1: lock 5-source reference + export_erp_dcm_multisource** - `52e5748` (feat)
2. **Task 2: MATLAB generator + cluster multisource mode + sbatch** - `f8a6d08` (feat)
3. **Task 3: P.S scalar fix + harvested fixtures (M3)** - `751d6aa` (fix)

## Harvested fixture array shapes (`validation/data/erp_multisource_fixtures.mat`, N=5, Cnd=2)

| Array | Shape | Pins |
|-------|-------|------|
| `QA` | `{2}` x `{4}` x `(5,5)` | `spm_gen_Q` B->all-A folding (free log space) |
| `QG` | `{2}` x `(5,)` | `spm_gen_Q` `diag(B)->Q.G(:,1)` precision path |
| `J0` | `{2}` x `(40,40)` | per-condition network frozen Jacobian (spm_diff FD) |
| `Qupd` | `{2}` x `(40,40)` | right-division `(spm_expm(dt*dfdx)-I)/dfdx` |
| `y` | `{2}` x `(128,40)` | multi-source evoked SOURCE trajectory (spm_gen_erp loop) |

Provenance `meta`: `D=1`, `nargout_Mf=2`, `N=5`, `Cnd=2`, `x0=zeros(5,8)`, `X=[[0],[1]]`, `dt=0.004`, `ns=128`, `ons=60`, `dur=16`, the SPM `$Id` strings, the locked edge lists + source names.

## Decisions Made

- **34-02-D1 (source-state trajectory, not channel-space):** stored the per-condition `spm_int_L` source-state trajectory (the `spm_gen_erp.m:69-86` loop body) rather than `spm_gen_erp`'s `L*x` channel output. The lead field is Phase 35, and the Wave-3 parity target (`cmc_network_f` / `simulate_erp_dcm`) produces source states `(ns,8N)`. Documented in the `.m` header.
- **34-02-D2 (P.S scalar):** see Deviations (the M3 debug fix).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] P.S exported as per-source (5,1) broke the spm_fx_cmc sigmoid matmul**
- **Found during:** Task 3 (M3 job 57896254 FAILED, exit 1, at `spm_fx_cmc:93`)
- **Issue:** The exporter wrote `P.S` as `(5,1)` (following the 34-01 `S:(n,1)` schema), but `spm_cmc_priors.m:124` defines `E.S = 0` as a SCALAR and `spm_fx_cmc.m:92-93` computes `R=(2/3)*exp(P.S)` then `F=1./(1+exp(-R*x+B))` — the `-R*x` is a MATRIX multiply, so a `(5,1)` `R` against the `(5,8)` state is an invalid `(5,1)*(5,8)` product ("Incorrect dimensions for matrix multiplication").
- **Fix:** Set the `export_erp_dcm_multisource` default `P.S` to `(1,1)`. Numerically identical to the per-source-uniform `S=0` the torch `cmc_sigmoid` uses (`R=2/3` either way), so no Wave-3 parity impact.
- **Files modified:** `validation/export_to_mat.py`
- **Verification:** A granular M3 `matlab -batch` probe (per-step try/catch) localised the failure to `spm_fx_cmc:93`; after the fix the probe ran all four steps green (`spm_fx_cmc` f `(40,1)`, `spm_diff` J0 `(40,40)`, `spm_int_L` y `(128,40)`); re-run job 57896525 COMPLETED with `checks_pass=true`.
- **Committed in:** `751d6aa` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug, surfaced on the first M3 run as anticipated by the plan's debug-loop allowance).
**Impact on plan:** Necessary for correctness; the fix tightened the exporter to SPM's canonical prior shapes. No scope creep; `P.T` `(5,4)` zeros was left as-is (harmless — `spm_fx_cmc:148-149` broadcasts per-column and `exp(0)=1`).

## Issues Encountered

- **MATLAB available on the M3 login node** (verified) — used a granular per-step probe (`probe_ms.m`, scp'd + run + deleted, not committed) for fast root-cause localisation instead of a full sbatch round-trip per attempt.
- **`validation/data/` is mutagen-IGNORED** (the `data/` footgun, like `models/`): the fixture `.mat` does not sync automatically, so it was `scp`'d back and force-added to git (`git add -f`), the 33-02 precedent (commit 94b603d).

## Next Phase Readiness

- **Wave 3 (34-03 multi-source parity ladder) unblocked:** the provenance-pinned `erp_multisource_fixtures.mat` is committed to git, so the laptop ladder gates on FIXTURE availability (not MATLAB) — it runs + asserts on the laptop. The staged-ladder targets are frozen: `QA`/`QG` (spm_gen_Q algebra, ~1e-12), `J0` (network FD-matched, ~1e-10), `Qupd` (right-division, ~1e-9), `y` (scheme ~1e-13 / FD-Jac ~1e-8).
- **Carry-forward:** the trajectory is SOURCE-level (no lead field, Phase 35). The Wave-3 `y` rung must split scheme / FD-Jacobian / measured-jacrev rungs (33-03-D2; `spm_diff` FD vs exact `jacrev`).
- **No blockers.**

---
*Phase: 34-extrinsic-coupling-condition-b-multi-source-evoked-integration*
*Completed: 2026-06-26*
