---
phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward
plan: 02
subsystem: validation-bridge
tags: [erp, eeg, meg, cmc, lead-field, scalp-projection, lfp, spm12, matlab, m3, fixtures]

# Dependency graph
requires:
  - phase: 34-extrinsic-coupling-condition-b-multisource-evoked
    provides: export_erp_dcm_multisource (_MS_ 5-source locked net + topology constants), run_spm_erp_dcm_multisource.m scaffolding, spm_fx_cmc_nodelay.m D=1 wrapper, erp_cross_validation harness (--mode)
  - phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward
    plan: 01
    provides: build_lead_field (torch kron column-major), project_to_scalp, cmc_default_pj (P.J idx2), lfp_spatial, the locked (Cnd,ns,Nc) layout + P.J/P.L
provides:
  - "export_erp_dcm_leadfield: additive LFP spatial spec (P.L=ones(1,5), P.J one-hot idx2, dipfit.type='LFP') on the locked _MS_ net"
  - "run_spm_erp_dcm_leadfield.m: spm_lx_erp LFP L_full (Nc,8N) + per-condition scalp ERP y_scalp (ns,Nc) + diff_wave, self-contained spm_gen_Q->spm_int_L source trajectory"
  - "erp_cross_validation.py --mode leadfield (main_leadfield + _check_leadfield_fixtures) + erp_cross_validation_leadfield.sbatch"
  - "validation/data/erp_leadfield_fixtures.mat: the LEAD-05 parity target (L_full (5,40), y_scalp {2}x(128,5), diff_wave (128,5), provenance meta)"
affects: [phase-35-wave-3-scalp-parity-ladder, phase-36-erp-dcm-model-amortized-mmn-demo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LFP single-dipole lead field as the head-model-free parity target: P.L=ones -> diag identity (spm_erp_L.m:112 no exp), no spm_cond_units, no MNI coords (ECD deferred to Phase 36)"
    - "Self-contained scalp fixture: recompute ysrc=spm_int_L(spm_gen_Q(P,X(c,:)),M,U) per condition rather than coupling to the Phase-34 .mat (Open Q6 resolved -- one fixture, one provenance header)"
    - "Additive MATLAB bridge: NEW .m + appended export branch + new --mode, existing exporters/scripts byte-identical (Phase 33/34 precedent)"

key-files:
  created:
    - validation/matlab_scripts/run_spm_erp_dcm_leadfield.m
    - cluster/sbatch/erp_cross_validation_leadfield.sbatch
    - tests/test_export_erp_leadfield.py
    - validation/data/erp_leadfield_fixtures.mat
  modified:
    - validation/export_to_mat.py
    - cluster/scripts/erp_cross_validation.py

key-decisions:
  - "35-02-D1: P.L=ones(1,5) is the correct LFP identity because spm_erp_L.m:112 builds L=sparse(1:m,1:m,P.L,m,n) with NO exp -- ones map directly to the diagonal (verified bit-exact vs torch lfp_spatial)"
  - "35-02-D2 (Open Q6): the leadfield .m is self-contained -- recomputes the source trajectory via spm_gen_Q->spm_int_L rather than loading erp_multisource_fixtures.mat, so the fixture has one provenance header and no cross-fixture coupling"
  - "35-02-D3: dipfit is constructed in the .m (dipfit.type/Ns/Nc) AND mirrored into DCM.dipfit + meta for the round-trip check; spm_lx_erp handles both the spatial diag and the kron(P.J,.) internally"

patterns-established:
  - "LFP lead field bit-exact between SPM (spm_lx_erp) and torch (build_lead_field): max|diff|=0.0 with the identity block at column-major state idx2 -- confirms kron column-major == cmc_flatten at the lead-field boundary"

# Metrics
duration: 22min
completed: 2026-06-26
---

# Phase 35 Plan 02: spm_lx_erp LFP Lead-Field + Scalp-ERP MATLAB Fixtures Summary

**Byte-frozen SPM12 LFP lead-field + end-to-end scalp-ERP ground truth for the locked 5-source MMN reference (D=1), generated on M3 (job 57900055, first-try GREEN) -- `L_full` is bit-exact to the pure-torch `build_lead_field` (max|diff|=0.0), unblocking the Wave-3 scalp parity ladder.**

## Performance
- **Duration:** ~22 min active
- **Tasks:** 2 (3 atomic commits); M3 job 26s wall
- **Files:** 6 (4 new, 2 appended)

## Accomplishments
- **`export_erp_dcm_leadfield` (additive):** reuses the locked `_MS_` 5-source reference net (A1L/A1R/STGL/STGR/rIFG, Cnd=2) + the `export_erp_dcm_multisource` machinery and ADDS the LFP spatial spec to the DCM struct -- `P.L = ones(1,5)` (identity diagonal), `P.J = (1,8)` one-hot at index 2 (sp-voltage, `spm_L_priors.m:108`), `dipfit.type='LFP'`/`Ns=Nc=5`, all dims float64 (int64->`spm_Ce` footgun). The single- and multi-source exporters stay byte-identical (244 insertions, 0 deletions).
- **`run_spm_erp_dcm_leadfield.m` (NEW):** mirrors the multi-source scaffolding (env paths, loud try/catch, `M.f='spm_fx_cmc_nodelay'`, `assert(nargout(M.f)==2)`, `assert(x0==zeros(5,8))`), forms `L_full = spm_lx_erp(P, dipfit)` (Nc,8N), and per condition recomputes `ysrc = spm_int_L(spm_gen_Q(P,X(c,:)), M, U)` and projects `y_scalp{c} = ysrc * L_full'` (ns,Nc); `diff_wave = y_scalp{2} - y_scalp{1}` (deviant - standard). Records SPM `$Id` for `spm_lx_erp`/`spm_erp_L`/`spm_L_priors`/`spm_gen_Q`/`spm_int_L`/`spm_fx_cmc`/`spm_gen_erp`.
- **Cluster harness + sbatch:** `--mode leadfield` (`main_leadfield` + `_check_leadfield_fixtures`, record-don't-crash) + `erp_cross_validation_leadfield.sbatch` (comp/16G/1h, no pip). Default `single`/`multisource` modes byte-untouched.
- **M3 job 57900055 (first-try GREEN, 26s, exit 0):** `checks_pass=true`. `L_full (5,40)`, `y_scalp {2}x(128,5)`, `diff_wave (128,5)`; `meta.D=1`, `nargout_Mf=2`, `N=5`, `Nc=5`, `dipfit_type='LFP'`, `pj_index=2`, `x0_is_zero=True`, `dt=0.004/ns=128/dur=16`. SPM `$Id`: `spm_lx_erp.m 7256`, `spm_erp_L.m 7142`. Fixture scp'd back (validation/data mutagen-ignored) + committed.
- **Bonus parity confirmation:** SPM `L_full` is BIT-EXACT to torch `build_lead_field(cmc_default_pj, lfp_spatial(ones,5))` -- `max|L_spm - L_torch| = 0.0`; the 5x5 identity block sits exactly at columns `[10:15]` (state idx 2, column-major `state*n+source`), confirming `kron` column-major == `cmc_flatten` at the lead-field boundary; rest of `L_full` exactly zero; `diff_wave` nonzero.

## Task Commits
1. **Task 1: export_erp_dcm_leadfield + run_spm_erp_dcm_leadfield.m + unit test** - `f5b23b9` (feat)
2. **Task 2a/b: leadfield mode in harness + M3 sbatch** - `8016fe4` (feat)
3. **Task 2c: M3 fixtures (job 57900055) scp'd back + committed** - `170c11a` (test)

## Fixture array shapes (the LEAD-05 parity target)
| Array | Shape | Meaning |
|-------|-------|---------|
| `L_full` | (5, 40) = (Nc, 8N) | `kron(P.J, L_spatial)`, LFP identity, `spm_lx_erp.m:33` |
| `y_scalp{c}` | (128, 5) = (ns, Nc) | per-condition scalp ERP `ysrc * L_full'` |
| `diff_wave` | (128, 5) | deviant - standard difference wave |

## Provenance meta
- `D=1`, `nargout_Mf=2`, `x0==zeros(5,8)`, `N=5`, `Nc=5`, `dipfit_type='LFP'`, `P.J` one-hot at idx 2, `P.L=ones(1,5)`.
- SPM `$Id`: `spm_lx_erp.m 7256 2018-02-11`, `spm_erp_L.m 7142 2017-07-26` (+ spm_L_priors/spm_gen_Q/spm_int_L/spm_fx_cmc/spm_gen_erp captured), `spm_ver=SPM12`.
- Grid: `dt=0.004`, `ns=128`, `ons=60`, `dur=16`.

## Decisions Made
- **35-02-D1 (P.L=ones is identity):** `spm_erp_L.m:112` builds `L = sparse(1:m,1:m,P.L,m,n)` with NO `exp`, so `P.L=ones` maps directly to the diagonal identity (confirmed bit-exact vs torch `lfp_spatial`). The plan's `P.L=ones` is correct as written.
- **35-02-D2 (Open Q6, self-contained):** the `.m` recomputes the source trajectory per condition rather than loading the Phase-34 `.mat`, so this fixture carries one provenance header and no cross-fixture coupling.
- **35-02-D3 (dipfit construction):** `dipfit.type/Ns/Nc` built in the `.m` AND mirrored on `DCM.dipfit` + `meta`; `spm_lx_erp` internally does both the LFP diagonal and the `kron(P.J,.)`.

## Deviations from Plan
**None of substance.** The MATLAB job ran GREEN on the first submission -- no debug loop was needed (contrast Phase 33 recursion / Phase 34 P.S-shape). The plan labelled Task 2 a `checkpoint:human-action`, but per the objective's explicit M3-unlocked instruction the executor submitted + harvested the job itself (no user gate). All planned artifacts delivered.

## Quality Gates
- `pytest tests/test_export_erp_leadfield.py tests/test_export_erp_multisource.py` -> 5 passed (struct-shape, all-double, additive guards; no MATLAB).
- `ruff check` clean on `export_to_mat.py` + `erp_cross_validation.py` + the new test; mypy delta only the pre-existing numpy-stub `__init__.pyi:737` baseline.
- Additive-only: `export_erp_dcm` + `export_erp_dcm_multisource` byte-identical (0 deletions); harness `main`/`main_multisource` bodies byte-untouched (only argparse choices/help extended); sbatch has no pip install.

## Next Phase Readiness
- **Wave 3 (scalp parity ladder) is unblocked:** the frozen `erp_leadfield_fixtures.mat` is the LEAD-05 target. Wave 3 asserts `build_lead_field` element-wise vs `L_full` (already confirmed `max|diff|=0.0` <= 1e-12) and `project_to_scalp` vs per-condition `y_scalp` (carrying the inherited 3-way `spm_diff` Jacobian split; `diff_wave` <= 1e-7 AND non-zero).
- **Carry-forward (from 35-01):** the shipped-jacrev scalp floor (~4.7e-8) sits below the LEAD-05 <=1e-7 gate ONLY because the LFP lead field is the identity (no amplification); a non-identity `P.L != 1` would scale the floor by `max|P.L|`. This fixture uses `P.L=ones` (identity) so the gate holds.
- **No blockers.**

---
*Phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward*
*Completed: 2026-06-26*
