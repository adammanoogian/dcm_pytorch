---
phase: 34-extrinsic-coupling-condition-b-multi-source-evoked-integration
plan: 01
subsystem: forward-models
tags: [cmc, erp, spm12, extrinsic-coupling, condition-b, mmn, spm_gen_Q, spm_fx_cmc, torch, float64]

# Dependency graph
requires:
  - phase: 33-cmc-core-spm-int-l-single-source-parity
    provides: "parity-verified cmc_f / parameterize_cmc / J_PERM / E0, the spm_int_L exp-Euler integrator (integrate_local_linearization), erp_gaussian_input, cmc_steady_state"
provides:
  - "parameterize_cmc_network: n>1 extrinsic blocks A[i]=exp(P.A[i])*E0[i] + lateral (1+4L) reciprocal reduction (spm_fx_cmc.m:68-82)"
  - "apply_condition_modulation: spm_gen_Q port (free/log space) folding B into all four A{1..4} (:47) AND diag(B)->G[:,0]->G[:,6] precision path (:65-67)"
  - "cmc_network_f: Phase-33 intrinsic EOM + four extrinsic A@S terms (fwd S[:,2] +; bwd S[:,6] -), bit-exact to cmc_f at n=1"
  - "simulate_erp_dcm: per-condition spm_gen_erp evoked loop returning source-LFP dict + deviant-minus-standard difference-wave hook"
  - "the locked free-param dict schema (T,G,C,S,R,A list[4], B list) for Wave 2/3 fixture encoding"
affects:
  - "34-02 (multi-source MATLAB fixtures: must encode the same P/A/B/X shapes)"
  - "34-03 (multi-source parity ladder: asserts apply_condition_modulation Q.A/Q.G + cmc_network_f J0 + trajectory)"
  - "35 (single-dipole lead-field reads simulate_erp_dcm source states)"
  - "36 (erp_dcm_model + MMN precision-sweep demo built on cmc_network_f + the diag(B)->G[:,6] knob)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "New-function path: cmc_network_f duplicates the frozen Phase-33 cmc_f intrinsic body + 4 extrinsic terms (Open Q1), pinned by an n=1 bit-exact guard, keeping the parity-gated cmc_f byte-frozen"
    - "RED-first C4 dual-B guard: precision-path POSITIVE test + omit-diag NEGATIVE control authored before the implementation"
    - "Free/log-space condition modulation BEFORE parameterisation (spm_gen_Q order of operations)"

key-files:
  created:
    - src/pyro_dcm/forward_models/erp_coupled_system.py
    - src/pyro_dcm/simulators/erp_simulator.py
    - tests/test_erp_coupled_system.py
  modified:
    - src/pyro_dcm/forward_models/__init__.py
    - src/pyro_dcm/simulators/__init__.py

key-decisions:
  - "34-01-D1: new cmc_network_f (not extend cmc_f in place) — zero edit risk to the parity-gated Phase-33 forward; ~12 intrinsic lines duplicated, pinned by the n=1==cmc_f bit-exact guard (max|diff|=0.0)"
  - "34-01-D2: parameterize_cmc_network returns zeros(4,n,n) when p['A'] is absent (mirrors parameterize_cmc) so the extrinsic terms vanish exactly at n=1; only builds exp*E + (1+4L) when 'A' is present"
  - "34-01-D3: -32 free-param convention for absent extrinsic edges in tests (exp(-32)*E0 << exp(-8), mirrors SPM mask*32-32) so lateral reduction and adjacency tests have clean live/dead edges"

patterns-established:
  - "Pattern: extrinsic A blocks stored as a (4,n,n) tensor; firing read via S[:, sp_V=2] (forward) / S[:, dp_V=6] (backward)"
  - "Pattern: difference_wave hook differences SOURCE states (sp voltage col 2); scalp projection deferred to Phase 35"

# Metrics
duration: 25min
completed: 2026-06-26
---

# Phase 34 Plan 01: Pure-Torch CMC Network Core + C4 Dual-B Guard Summary

**Hierarchical N-source CMC forward (`cmc_network_f`) composing the frozen Phase-33 intrinsic EOM with four extrinsic `A@S` terms, the `spm_gen_Q` condition-B port folding `diag(B)` into the `G[:,6]` precision knob, and a per-condition `spm_gen_erp` evoked simulator — all pure-torch, laptop, sub-second, with the C4 precision guard + omit-diag negative test driving the build.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-06-26
- **Tasks:** 3 (1 test-RED + 2 feat)
- **Files modified:** 5 (3 created, 2 appended)

## Accomplishments

- **C4 dual-B precision mechanism wired and guarded.** `apply_condition_modulation` ports `spm_gen_Q.m:24-67` in free/log space: one `B[i]` folds additively into all four `Q.A{1..4}` (`:47`) AND `diag(B[i])` into the free precision column `Q.G[:,0]` (`:65-67`), which `parameterize_cmc_network` then routes via `J_PERM[0]==6` to `G[:,6]` (sp self-inhibition / precision). The POSITIVE guard asserts `G[:,6]` moves; the NEGATIVE omit-diag control asserts it does NOT move when the precision line is skipped — proving the path is load-bearing (EVOK-02).
- **`cmc_network_f(n=1) == cmc_f` bit-exact (measured max|diff| = 0.0).** The four extrinsic terms vanish at `n=1` (absent `p["A"]` → `zeros(4,1,1)`), so the proven single-source forward is preserved with zero edit to `cmc_neural_mass.py`.
- **Extrinsic topology fully pinned** (`spm_fx_cmc.m:171-198`): forward `+A[0]@S[:,2]→ss`, `+A[1]@S[:,2]→dp`; backward `−A[2]@S[:,6]→sp`, `−A[3]@S[:,6]→ii`; the `exp(P.A)*E0` blocks (`E0=[200,100,200,100]`) with lateral `(1+4L)` reciprocal reduction (reciprocal pair /5, one-way unchanged); input `C` drives spiny-stellate only.
- **`simulate_erp_dcm`** loops conditions via the `spm_gen_erp` pattern, re-freezing the Jacobian per condition through the Phase-33 integrator, returning `{states (Cnd,ns,n,8), pst, inputs, difference_wave}`; the smoke test confirms finite trajectories and a non-zero source difference wave iff `B` is wired (exactly zero control when `B=0`).
- **34 tests green** (25 Phase-33 regression + 9 new), ruff + format clean, additive-only verified by `git diff` (frozen forward/integrator untouched).

## Task Commits

1. **Task 1: C4 guard tests FIRST (RED) + structural battery** - `4d67bbd` (test)
2. **Task 2: implement erp_coupled_system.py (GREEN)** - `4913725` (feat)
3. **Task 3: implement erp_simulator.py + smoke test** - `1db6bde` (feat)

## Locked function signatures (for Wave 2/3)

```python
parameterize_cmc_network(p: dict[str, Tensor], n: int) -> dict[str, Tensor]
    # -> {"T":(n,4), "G":(n,10), "C":(n,n_inp), "A":(4,n,n), "S":(n,1)}
apply_condition_modulation(p: dict[str, Tensor], x_design: Tensor) -> dict[str, Tensor]
    # x_design: (n_effects,) for ONE condition; returns Q (no "B" key), Q["A"] is (4,n,n)
cmc_network_f(x_flat: Tensor, u: Tensor, p: dict[str, Tensor], n: int) -> Tensor
    # x_flat (8n,) float64 column-major -> dx/dt (8n,)
simulate_erp_dcm(p, x_design: Tensor, n: int, ns=128, dt=0.004,
                 ons_ms=60.0, dur_ms=16.0, sus=0.0) -> dict
    # x_design: (Cnd, n_effects); row 0 = standard, row 1 = deviant
```

## Free-param dict schema (LOCKED — Wave 2/3 must encode identically)

| Key | Shape | Meaning |
|-----|-------|---------|
| `"T"` | `(n,4)` | synaptic time-constant log-params |
| `"G"` | `(n,4)` | free intrinsic log-params (col 0 → `G[:,6]` precision via `J_PERM`) |
| `"C"` | `(n,n_inp)` | input-gain log-params (single-page; C-effect skipped) |
| `"S"` | `(n,1)` | sigmoid-slope log-param |
| `"R"` | `(n_inp,2)` | Gaussian onset/dispersion log-params (B does NOT touch R) |
| `"A"` | length-4 list / `(4,n,n)` | extrinsic free log-params A1..A4 (absent → `zeros(4,n,n)`) |
| `"B"` | list of `(n,n)` | between-trial modulation matrices (dropped from `Q`) |

`x_design`: `(Cnd, n_effects)`; per-condition row passed to `apply_condition_modulation`.

## Decisions Made

- **34-01-D1 (new-function path, Open Q1 resolved):** built a NEW `cmc_network_f` rather than extending `cmc_f` in place. Keeps the Phase-33 parity-gated forward byte-frozen; the `n=1`-equals-`cmc_f` guard (max|diff| = 0.0) pins the duplicated intrinsic body.
- **34-01-D2 (absent-A → zeros):** `parameterize_cmc_network` only builds `exp*E + (1+4L)` when `p["A"]` is present, returning `parameterize_cmc`'s `zeros(4,n,n)` otherwise. This is what makes the n=1 extrinsic terms exactly zero (and `x + 0.0 == x` in IEEE-754 keeps the guard bit-exact).
- **34-01-D3 (sparse free-param convention in tests):** non-edges use free param `-32` (`exp(-32)*E0 ≈ 2.5e-12 < exp(-8)`), live edges use `0`, mirroring the SPM mask `mask*32-32`. Gives the lateral-reduction and adjacency guards clean live/dead edges.

## Deviations from Plan

None - plan executed exactly as written. All three tasks landed with their planned files and the C4 RED-first ordering; no bugs, blockers, or architectural decisions surfaced.

## Issues Encountered

- **ruff isort churn on `__init__.py` appends** (resolved): `--fix` re-sorted the new `erp_coupled_system` import after `csd_computation`; cosmetic, no semantic change.
- **mypy numpy-stub baseline** (pre-existing, not introduced): `numpy/__init__.pyi:737 "Type statement is only supported in Python 3.12+"` halts mypy identically on the frozen `cmc_neural_mass.py`, confirming no new type errors from this plan. This is the accepted environment baseline.

## Next Phase Readiness

- **Wave 2 (34-02, M3/MATLAB) unblocked:** the reference network forward + condition-B mechanism are isolated, green, and the `P`/`A`/`B`/`X` shapes are locked (schema table above). The B-wiring / coupling logic is proven in isolation before it compounds through `spm_int_L` (pitfall V5).
- **Carry-forward for Wave 3:** the multi-source trajectory gate must still split into scheme / FD-Jacobian / measured-jacrev rungs (Phase 33-03 D2; `spm_diff` FD vs exact `jacrev`). `cmc_network_f` is exact-AD-friendly (used inside the integrator's `jacrev`).
- **No blockers.** Difference-wave is source-level only by design (scalp lead-field is Phase 35).

---
*Phase: 34-extrinsic-coupling-condition-b-multi-source-evoked-integration*
*Completed: 2026-06-26*
