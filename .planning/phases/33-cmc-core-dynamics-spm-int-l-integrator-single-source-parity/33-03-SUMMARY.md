---
phase: 33-cmc-core-dynamics-spm-int-l-integrator-single-source-parity
plan: 03
subsystem: testing
tags: [cmc, erp, spm12, parity, spm_int_L, spm_fx_cmc, spm_diff, matrix_exp, eeg-meg, validation]

# Dependency graph
requires:
  - phase: 33-01
    provides: "pure-torch cmc_f / parameterize_cmc forward + integrate_local_linearization (spm_int_L port) under test"
  - phase: 33-02
    provides: "the byte-frozen SPM12 fixtures (f_field, J0, dtJ, Eexp, Q_update, y_states + meta) the ladder asserts against"
provides:
  - "tests/test_spm_erp_dcm_validation.py -- the V5 staged single-source parity ladder vs the frozen .mat (CMC-06)"
  - "tests/test_local_linearization.py::test_matrix_exp_vs_spm_expm_floor -- the MEASURED matrix_exp<->spm_expm backend floor"
  - "cluster/sbatch/erp_parity_test.sbatch -- M3 entrypoint for the SPM-gated parity suite"
  - "PROVEN: cmc_f is bit-identical to spm_fx_cmc; the exp-Euler scheme + loop ordering are bit-identical to spm_int_L"
affects: [34-extrinsic-coupling, 35-lead-field, 36-erp-dcm-model]

# Tech tracking
tech-stack:
  added: []  # zero new deps; pure-torch vs the frozen arrays
  patterns:
    - "Parity gate keys on FIXTURE availability (skip iff .mat missing), NOT MATLAB availability -- the assertions are torch-vs-frozen-arrays, deterministic float64, so they run+pass on the laptop"
    - "MEASURE numerical-method floors, never assume them: matrix_exp<->spm_expm AND jacrev<->spm_diff are both recorded, not gated at an assumed 1e-12"
    - "Replicate SPM's spm_diff forward-difference Jacobian (dx=exp(-8)) to hold the Jacobian-construction method fixed when proving forward + integrator parity"

key-files:
  created:
    - tests/test_spm_erp_dcm_validation.py
    - cluster/sbatch/erp_parity_test.sbatch
  modified:
    - tests/test_local_linearization.py  # append-only: the measured matrix_exp floor test

key-decisions:
  - "[33-03-D1] The gate keys on fixture availability, not MATLAB -- the laptop run IS the authoritative deterministic assertion run"
  - "[33-03-D2] SPM freezes J0 via spm_diff forward differences (dx=exp(-8)), not analytically; cmc_f matches that bit-exact (0.0). The shipped integrator's exact jacrev is MORE accurate, so its J0/y_states differ from SPM by the FD truncation -- MEASURED, not a bug"
  - "[33-03-D3] The y_states bit-close gate is asserted on the scheme path (SPM's own Q -> 6.6e-14) and the FD-Jacobian integrator path (1.2e-10); the shipped-jacrev floor (4.7e-8) is recorded as a measured numerical floor"

patterns-established:
  - "Pattern: split a parity rung into (a) a forward/scheme gate with the numerical method held to SPM's and (b) a MEASURED floor for torch's more-accurate method"
  - "Pattern: prove integration-scheme/loop-ordering parity by driving the loop with SPM's OWN frozen update operator (isolates C1 from the Jacobian method)"

# Metrics
duration: ~45min
completed: 2026-06-26
---

# Phase 33 Plan 03: SPM12 Single-Source Parity Ladder + Measured matrix_exp Floor Summary

**The V5 staged parity ladder proves the pure-torch CMC forward `cmc_f` is bit-identical to `spm_fx_cmc` and the exp-Euler `spm_int_L` port's scheme + loop ordering are bit-identical to SPM12 (single source, D=1, x0=0) -- with the one orthogonal divergence (torch's exact `jacrev` Jacobian vs SPM's `spm_diff` finite-difference Jacobian) MEASURED, not assumed; all 9 rungs green on the laptop in 4.3s.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Tasks:** 3 (2 autonomous code + 1 M3 checkpoint satisfied-by-laptop, see D1)
- **Files modified/created:** 3 (2 new, 1 append-only)

## The 5+ MEASURED max|diff| per ladder rung (the deliverable numbers)

Asserted IN ORDER (a failure localises to one stage, V5); element-wise forward
agreement only (no absolute-F, no Cp -- V2). All values are deterministic
float64 (identical on laptop and M3):

| Rung | What it isolates | Reference | max&#124;diff&#124; | Gate |
|------|------------------|-----------|----------|------|
| 1. `f_field` | transforms / sigmoid / `J_PERM` / units (pre-integrator) | `spm_fx_cmc(x_test,u_test)` | **5.821e-11** | ≤1e-10 PASS |
| 2a. `J0` (spm_diff FD) | `cmc_f` == `spm_fx_cmc` (Jacobian method matched) | `spm_diff` `dx=exp(-8)` | **0.000e+00** | ≤1e-10 PASS (bit-exact) |
| 2b. `J0` (jacrev) **MEASURED** | exact-AD vs SPM finite-difference floor | jacrev − fixture `J0` | **5.556e-04** | recorded (FD truncation) |
| 3. `matrix_exp(dtJ)` **MEASURED** | Pade backend vs `spm_expm` | `torch.matrix_exp` − `Eexp` | **8.556e-11** | <1e-9 PASS |
| 4. `Q_update` | right-division `(E−I)@inv(dfdx)` (C2) | `solve(dfdx.T,(E−I).T).T` − `Q_update` | **2.745e-15** | ≤1e-9 PASS |
| 5a. `y_states` (SPM's Q) | exp-Euler loop ordering / scheme (C1) | loop w/ fixture `Q_update` − `y_states` | **6.573e-14** | ≤1e-8 PASS |
| 5b. `y_states` (FD-Jac integrator) | end-to-end integrator, Jacobian method matched | full operator path on `spm_diff` `J0` − `y_states` | **1.161e-10** | ≤1e-8 PASS |
| 5c. `y_states` (shipped jacrev) **MEASURED** | propagated `spm_diff` truncation | `integrate_local_linearization` − `y_states` | **4.692e-08** | recorded (more accurate) |

**Measured matrix_exp↔spm_expm floor:** `8.556e-11` (rung 3 + `test_matrix_exp_vs_spm_expm_floor`). This sets the small-multiple ceilings the `Q_update`/`y_states` tiers ride on.

## Accomplishments
- **The Phase-33 SPM12 parity gate (CMC-06) is GREEN on the laptop** (the assertions are torch-vs-frozen-arrays, MATLAB-independent, deterministic): `tests/test_spm_erp_dcm_validation.py`, 9 rungs, 4.3 s.
- **Proved `cmc_f` IS `spm_fx_cmc`** bit-exact: with the Jacobian-construction method held to SPM's `spm_diff` (forward differences, `dx=exp(-8)`), `J0` matches the fixture to **0.0** and `f_field` to 5.8e-11.
- **Proved the exp-Euler scheme + loop ordering are bit-identical to `spm_int_L`**: driving the loop with SPM's own frozen `Q_update` reproduces `y_states` to **6.6e-14** (isolates C1 from the Jacobian method); the full operator path on the FD Jacobian reproduces it to 1.2e-10.
- **MEASURED both numerical-method floors** (never assumed, V3): `matrix_exp`↔`spm_expm` = 8.6e-11; `jacrev`↔`spm_diff` `J0` = 5.6e-4; shipped-`jacrev` `y_states` = 4.7e-8.
- **Localised the only divergence** to an already-documented Wave-1 design choice (exact `jacrev` vs SPM's `spm_diff`) -- the torch integrator is MORE accurate than SPM, not buggy.
- Delivered `cluster/sbatch/erp_parity_test.sbatch` (the optional M3 re-run) and extended the Wave-1 integrator test additively; Wave-1 tests stay green (7 passed).

## Task Commits

1. **Task 1: V5 staged parity ladder** - `(test 33-03)` tests/test_spm_erp_dcm_validation.py
2. **Task 2: measured matrix_exp floor + M3 sbatch** - `(test 33-03)` test_local_linearization.py + erp_parity_test.sbatch
3. **Task 3: M3 checkpoint** - satisfied by the deterministic laptop run (see Deviation 1)

## Files Created/Modified
- `tests/test_spm_erp_dcm_validation.py` - the 9-rung staged parity ladder vs the frozen `.mat`, fixture-availability gated, D==1 + x0==0 + float64 pre-asserts
- `tests/test_local_linearization.py` - append-only `test_matrix_exp_vs_spm_expm_floor` (the measured backend floor); Wave-1 tests byte-untouched
- `cluster/sbatch/erp_parity_test.sbatch` - M3 SPM-gated parity suite runner (comp/16G/1h, no pip)

## Decisions Made
- **[33-03-D1] Gate on fixture availability, not MATLAB.** The parity assertions compare torch to the frozen fixture arrays (no live MATLAB), and the computation is deterministic float64. So the suite RUNS AND PASSES on the laptop, and the laptop run is the authoritative assertion run. The `@pytest.mark.spm`/`slow` markers + the M3 sbatch are retained so the suite can also be re-run on M3, but M3 is redundant (it would produce identical numbers). This overrides the PLAN's `skipif(not check_matlab_available())` pattern per the orchestrator's explicit instruction.
- **[33-03-D2] SPM's `J0` is a `spm_diff` forward-difference Jacobian (`dx=exp(-8)`), not analytic.** Discovered via the ladder: `cmc_f`'s exact `jacrev` Jacobian differs from the fixture `J0` by 5.6e-4, while a forward-difference Jacobian matching `spm_diff` matches to **0.0**. `spm_int_L` freezes its Jacobian the same way, which is why the shipped (jacrev) integrator's `y_states` differs from SPM's by 4.7e-8. This is NOT a bug -- the exact AD Jacobian is more accurate; the difference is the `spm_diff` truncation. The ladder therefore splits the `J0`/`y_states` rungs into a forward/scheme gate (method matched to SPM, bit-exact) and a MEASURED floor (torch's more-accurate method).
- **[33-03-D3] The bit-close `y_states` gate is asserted on the scheme + FD-Jacobian paths.** `≤1e-8` holds on the scheme path (SPM's own `Q` -> 6.6e-14) and the full integrator with `spm_diff` Jacobian (1.2e-10); the shipped-jacrev path (4.7e-8) is recorded as a measured floor with a loose ceiling, transparently above the bit-close threshold.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - design/measurement, per orchestrator override] Gate keyed on fixtures + ladder restructured for the AD-vs-FD Jacobian finding**
- **Found during:** Tasks 1 & 3
- **Issue:** (a) The PLAN mirrors `skipif(not check_matlab_available())`, which would SKIP on the laptop even though the fixtures are present; the orchestrator's critical constraints require the test to RUN AND PASS on the laptop. (b) The PLAN's rung 2 ("torch `jacrev` of `cmc_f` vs `J0` ≤1e-10") is infeasible AS WRITTEN: SPM's `J0` is a `spm_diff` finite-difference Jacobian, so exact `jacrev` differs by the FD truncation (5.6e-4), exactly the kind of numerical-method floor the orchestrator mandated MEASURING for `matrix_exp`.
- **Fix:** (a) Gate on fixture availability (`_FIXTURE_PATH.exists()`), keeping `@pytest.mark.spm`/`slow` for the M3 sbatch. (b) Split rung 2 into a forward-parity gate (FD Jacobian matching `spm_diff`, bit-exact 0.0) + a MEASURED `jacrev`-vs-`spm_diff` floor; split rung 5 into a scheme gate (SPM's `Q`, 6.6e-14) + an FD-Jacobian integrator gate (1.2e-10) + a MEASURED shipped-jacrev floor (4.7e-8). No Wave-1 source modified -- the forward and the scheme are bit-exact; the divergence is the already-documented `jacrev` choice.
- **Files modified:** tests/test_spm_erp_dcm_validation.py, tests/test_local_linearization.py
- **Verification:** 9/9 ladder rungs + 7/7 integrator tests green on laptop (4.3s + 4.7s); ruff clean; mypy only the accepted `pyro_dcm [import-untyped]` baseline.
- **Committed in:** `test(33-03)` Task 1 + Task 2 commits

---

**Total deviations:** 1 (restructure + gate change driven by the orchestrator override and the ladder's AD-vs-FD finding). No Wave-1 source edits; no architectural change; no scope creep. The forward + the integration scheme are bit-identical to SPM12 -- the parity gate is genuinely met, with the one numerical-method difference recorded transparently rather than masked.

## Issues Encountered
- **`spm_diff` finite-difference Jacobian.** The single non-trivial divergence (J0 5.6e-4, y_states 4.7e-8) traced cleanly to SPM building its frozen Jacobian by forward differences (`dx=exp(-8)`) while the torch integrator uses exact `jacrev`. Confirmed by reproducing the fixture `J0` to 0.0 with a matched forward-difference Jacobian, and reproducing `y_states` to 1.2e-10 by integrating with that Jacobian. The ladder's staged design localised it immediately.

## User Setup Required
None - no external service configuration required. The optional M3 re-run is `sbatch cluster/sbatch/erp_parity_test.sbatch` (requires the synced fixture; no MATLAB needed for the assertions).

## Next Phase Readiness
- **Phase 34 (extrinsic coupling) inherits a PROVEN single-source baseline:** `cmc_f` == `spm_fx_cmc` bit-exact, the exp-Euler scheme == `spm_int_L` bit-exact, with the tolerance floors recorded (matrix_exp 8.6e-11; Q_update 2.7e-15; scheme 6.6e-14). Any divergence introduced by the `A{1..4}` coupling blocks in Phase 34 cannot be blamed on the intrinsic forward or the integrator -- they are pinned here.
- **Carry-forward for Phase 34+:** SPM's `spm_int_L` freezes its Jacobian via `spm_diff` (forward differences, `dx=exp(-8)`). If multi-source/extrinsic parity must be tighter than ~5e-8 against an SPM fixture, hold the Jacobian-construction method fixed (replicate `spm_diff`) rather than comparing exact `jacrev` to SPM's FD Jacobian. The shipped integrator's exact AD is a feature for Phase 35 (amortized/gradient), not a parity liability.
- No blockers.

---
*Phase: 33-cmc-core-dynamics-spm-int-l-integrator-single-source-parity*
*Completed: 2026-06-26*
