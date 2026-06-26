---
phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward
plan: 03
subsystem: testing
tags: [spm12, lead-field, scalp-erp, parity-gate, lfp, cmc, project-to-scalp, fixture]

# Dependency graph
requires:
  - phase: 35-01
    provides: build_lead_field / project_to_scalp / cmc_default_pj / lfp_spatial (pure-torch LFP lead field + ERPDCMForward)
  - phase: 35-02
    provides: frozen spm_lx_erp LFP fixtures (L_full, y_scalp, diff_wave, meta) in validation/data/erp_leadfield_fixtures.mat
  - phase: 34-02
    provides: erp_multisource_fixtures.mat (SPM's own frozen per-condition Qupd, byte-identical _MS_ topology + drive)
  - phase: 34-03
    provides: relative-error convention at N=5 + the _reference_p / _spm_diff_jacobian / fixture-loader ladder pattern
provides:
  - "tests/test_spm_erp_leadfield_validation.py: the 5-rung scalp parity ladder asserting build_lead_field/project_to_scalp element-wise vs the frozen LFP fixtures"
  - "LEAD-05 PARITY GATE green on laptop: production integrate_local_linearization -> project_to_scalp scalp ERP matches spm_gen_erp+spm_lx_erp <=1e-7 (measured floor 6.4e-11)"
  - "LEAD-02 green: L_full == frozen spm_lx_erp L_full max|diff|=0.0; LEAD-03 green: difference wave <=1e-7 AND non-zero"
  - "the measured Phase-35 finding: LFP-identity lead field does not amplify the source jacrev floor -> the production path can be GATED (not merely measured) at <=1e-7"
affects: [phase-36, erp_dcm_model, amortized-erp, mmn-precision-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scalp parity ladder (V5): L_full exact -> scalp scheme (SPM's Qupd) -> scalp FD-Jacobian -> production-jacrev GATE -> difference wave, each rung localizing one failure mode"
    - "Cross-fixture reuse: the byte-identical 34-02 multi-source Qupd drives the rung-2 scheme rung at machine-eps (same _MS_ topology + drive -> identical source trajectory)"
    - "Production-path gating where the lead field does not amplify: gate the shipped jacrev integrator directly at <=1e-7 instead of measure-not-gate"

key-files:
  created:
    - tests/test_spm_erp_leadfield_validation.py
  modified: []

key-decisions:
  - "35-03-D1: rung-2 scheme rung loads SPM's frozen Qupd from erp_multisource_fixtures.mat (byte-identical P+drive) since the leadfield fixture stores only y_scalp, not Qupd/J0 -- the cleanest machine-eps projection-isolation; skipif that fixture is absent (rungs 1,3,4,5 stay laptop-self-contained)"
  - "35-03-D2: LEAD-05 GATED DIRECTLY at <=1e-7 (not measure-not-gate). The measured scalp jacrev floor is 6.4e-11 -- BELOW both 1e-7 and the 4.7e-8 Phase-34 source floor, because the identity LFP lead field selects ONLY the sp-voltage state (P.J=e_2), which carries less of the propagated spm_diff FD truncation than the worst-case source state. Caveat baked into the test: a non-identity P.L scales the floor by max|P.L| and must be re-measured"
  - "35-03-D3: single atomic commit for the whole 5-rung ladder file (the established 34-03 precedent for validation-ladder files), not a per-task split, since both plan tasks edit one cohesive test file sharing helpers"

patterns-established:
  - "Pattern: scalp ERP is asserted ON TOP of the already-parity-verified source trajectory (Phase 34) -- any divergence localizes to the lead field / projection, never the network forward or integrator"
  - "Pattern: honor the locked (Cnd,ns,Nc) torch.stack + C-order reshape(-1) layout (ERPDCMForward.predict, gap 4) in every scalp comparison"

# Metrics
duration: ~30min
completed: 2026-06-26
---

# Phase 35 Plan 03: Scalp-ERP SPM12 Parity Ladder Summary

**The pure-torch LFP lead field + scalp projection match the frozen `spm_lx_erp` fixtures element-wise across a 5-rung ladder; the LEAD-05 production-path gate passes at <=1e-7 with a measured scalp jacrev floor of 6.4e-11 -- confirming the LFP-identity no-amplification finding.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Tasks:** 2 (authored as one cohesive test file)
- **Files modified:** 1 created, 0 source edits

## Accomplishments

- **LEAD-05 PARITY GATE green (laptop, fixture-keyed):** the full production `integrate_local_linearization(cmc_network_f) -> project_to_scalp` scalp ERP matches `spm_gen_erp + spm_lx_erp` (LFP mode) on the frozen fixtures with a **measured jacrev floor of 6.379e-11 < 1e-7** (rung 4), so the production integrator is GATED DIRECTLY at <=1e-7 -- the first Phase where the shipped path can be gated (not merely measured), because the identity LFP lead field does not amplify the source floor.
- **LEAD-02 green:** `build_lead_field(cmc_default_pj(), lfp_spatial(ones(5),5))` == frozen `L_full` **max|diff| = 0.000e+00** (rung 1), plus the distinct-valued kron column-major check (identity block at the sp-voltage state s=2, cols [10:15]) and the `P.J argmax==2, !=6` guard.
- **LEAD-03 green:** production-path `scalp[1]-scalp[0]` matches `diff_wave` (= deviant - standard) **max|diff| = 1.265e-11 <= 1e-7** AND is **non-zero (max|diff_wave| = 3.076e-2)**; the negative-going / frontal SIGN is deferred to Phase 36 (recorded as a non-gating diagnostic: peak channel sign = -1).
- Additive-only: one new test file, **zero source edits** (`git diff --stat HEAD -- src/ validation/` empty); Phase 33/34 ladders + `test_erp_leadfield.py` stay green; ruff + format clean.

## Per-rung measured floors

| Rung | What | Measured | Tolerance | Result |
| ---- | ---- | -------- | --------- | ------ |
| 1 | `L_full` build_lead_field vs SPM (element-wise) | **0.000e+00** | <=1e-12 | PASS (the 35-02 headline, bit-exact) |
| 2 | scalp scheme (SPM's own Qupd) relative | **1.578e-13** (abs 4.08e-14) | <=1e-12 rel | PASS |
| 3 | scalp FD-Jacobian (spm_diff J0) | **1.392e-13** | <=1e-8 | PASS |
| 4 | scalp PRODUCTION jacrev (LEAD-05 GATE) | **6.379e-11** | <=1e-7 | PASS (floor << 1e-7) |
| 5 | difference wave (production) | **1.265e-11** (non-zero mag 3.076e-2) | <=1e-7 | PASS |

**Key Phase-35 finding (35-03-D2):** the scalp jacrev floor (6.4e-11) is BELOW both the 1e-7 gate AND the 4.70e-8 Phase-34 source floor. The identity LFP lead field with `P.J = e_2` selects only the superficial-pyramidal voltage state, which carries less of the propagated `spm_diff` forward-difference truncation than the worst-case source state -- so projection not only fails to amplify, it tightens the floor. The gate therefore has ~1600x margin below 1e-7. The non-identity-`P.L` caveat (floor scales by `max|P.L|`, re-measure) is baked into the test body.

## Task Commits

1. **Task 1+2: scalp-ERP SPM12 parity ladder (5 rungs)** - `5ef52c8` (test)

(Single atomic commit per 35-03-D3 / the 34-03 validation-ladder precedent.)

## Files Created/Modified

- `tests/test_spm_erp_leadfield_validation.py` - the 5-rung scalp parity ladder: pre-asserts (D==1, LFP, N==Nc==5, x0==zeros, P.L==ones, P.J idx-2) + L_full-exact + scalp-scheme + scalp-FD-Jacobian + production-jacrev LEAD-05 gate + difference wave; reconstructs P from the imported `_MS_` constants (pitfall V1), honors the (Cnd,ns,Nc) C-order stacking.

## Decisions Made

- **35-03-D1:** rung-2 (scheme) loads SPM's frozen `Qupd` from `erp_multisource_fixtures.mat` (the leadfield fixture stores only `y_scalp`, not `Qupd`/`J0`; the 34-02 fixture is byte-identical in P+drive, verified: projecting its frozen source `y{c}` through `L_full` reproduces `y_scalp{c}` to ~5e-14 relative). `skipif` that fixture is absent, so rungs 1/3/4/5 stay laptop-self-contained on the leadfield fixture alone.
- **35-03-D2:** LEAD-05 gated DIRECTLY at <=1e-7 (measured floor 6.4e-11). See the Key Phase-35 finding above. Non-identity-`P.L` caveat documented in the test.
- **35-03-D3:** one atomic commit for the whole ladder file (34-03 precedent), not a forced per-task file split.

## Deviations from Plan

None - plan executed exactly as written. No source bug surfaced (the source trajectory was already parity-verified in Phase 34), so no Rule-1/3 fix was needed; the production-path gate was MEASURED and confirmed below 1e-7 as the plan anticipated (no Rule-4 "don't fudge" escalation required).

## Issues Encountered

- Initial `ruff` emitted 3 E501 (long docstring/print lines) + 1 D401 (non-imperative docstring); resolved by wrapping the print f-string, shortening the rung-3 docstring summary line, and rewording `_torch_l_full` to imperative mood. `ruff check` + `ruff format --check` then clean.
- mypy delta is only the documented baseline: `pyro_dcm.*` `[import-untyped]` (no py.typed marker, per 32-01-D3) + the numpy-stub `__init__.pyi:737` syntax baseline. Zero new error categories.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 35 is now 3/3 waves complete** (35-01 pure-torch lead field + ERPDCMForward + VL round-trip; 35-02 frozen LFP fixtures; 35-03 scalp parity gate). LEAD-02/03/05 all gated green. Ready for `/gsd:verify-phase 35` then Phase 36 (erp_dcm_model + amortized + 5-source MMN precision-sweep demo).
- **Phase 36 inherits a parity-verified scalp ERP forward end-to-end** (source trajectory + lead field + projection + difference wave all bit-exact / sub-1e-7 to SPM12) -- any MMN-sweep divergence cannot be blamed on the forward stack.
- **Deferred to Phase 36 (unchanged):** the difference-wave SIGN (negative-going / frontal direction, Fact 6) and the ECD spatial path (`ecd_spatial`, needs a sensor montage + MNI coords). The non-identity-`P.L` caveat (floor scales by `max|P.L|`) is recorded for any future non-LFP lead field.

---
*Phase: 35-single-dipole-lead-field-scalp-projection-erpdcmforward*
*Completed: 2026-06-26*
