---
phase: 36-erp-dcm-pyro-model-amortized-wiring-mmn-precision-sweep-demo
plan: 01
subsystem: api
tags: [pyro, svi, amortized, normalizing-flow, dcm, erp, cmc, mmn, float64]

# Dependency graph
requires:
  - phase: 33-cmc-core
    provides: cmc_neural_mass + spm_int_L integrator + cmc_priors (prior scales)
  - phase: 34-extrinsic-coupling
    provides: cmc_network_f + apply_condition_modulation + simulate_erp_dcm
  - phase: 35-leadfield
    provides: build_lead_field + project_to_scalp + ERPDCMForward (frozen pack order)
provides:
  - "erp_dcm_model: Pyro generative ERP-DCM (samples A/B/C/T/G/S/R + scalp_noise_scale; reuses simulate_erp_dcm; Gaussian scalp likelihood)"
  - "ERPDCMPacker: identity-reshape packer mirroring ERPDCMForward pack order (B excluded)"
  - "ErpSummaryNet: MLP summary network over flattened (Cnd,ns,Nc) scalp"
  - "amortized_erp_dcm_model + _run_erp_forward_model: flow-guide path reusing ERPDCMForward.predict (B fixed)"
affects: [36-02-mmn-reference, 36-03-precision-sweep-demo, amortized-erp-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pyro generative DCM = priors + likelihood ONLY; deterministic forward delegated to the parity-gated simulator (pitfall V1)"
    - "Per-effect B_free_{j} sample loop with NO pyro.plate -> AutoGuide auto-discovery (MODEL-06)"
    - "CMC mask asymmetry: A/C dead -> free -32 (exp(P)*E0 dead), B dead -> 0 (additive log-offset)"
    - "Amortized ERP packer EXCLUDES B + noise (frozen ERPDCMForward order); fixed obs noise since the flow guide samples only _latent"

key-files:
  created:
    - src/pyro_dcm/models/erp_dcm_model.py
    - tests/test_erp_dcm_model.py
    - tests/test_amortized_erp.py
  modified:
    - src/pyro_dcm/models/__init__.py
    - src/pyro_dcm/guides/parameter_packing.py
    - src/pyro_dcm/guides/summary_networks.py
    - src/pyro_dcm/guides/__init__.py
    - src/pyro_dcm/models/amortized_wrappers.py
    - cluster/sbatch/erp_pytest.sbatch

key-decisions:
  - "B-prior variance stayed PROVISIONAL 1/8 (MUST-VERIFY vs spm_dcm_erp); not load-bearing for the fixed-B headline demo"
  - "Amortized ERP path holds B + observation noise FIXED (packer mirrors ERPDCMForward; flow guide samples only _latent)"
  - "Appended code made canonically ruff-format-clean; pre-existing sibling-packer/import-sort format debt left untouched to stay additive"

patterns-established:
  - "Pattern: new Pyro DCM model reuses the existing simulate_* forward verbatim (no re-assembly)"
  - "Pattern: amortized packer == frozen ForwardModel pack order for VL/amortized theta interchange"

# Metrics
duration: 34min
completed: 2026-06-26
---

# Phase 36 Plan 01: ERP-DCM Pyro Model + Amortized Wiring Summary

**Pyro generative ERP-DCM (`erp_dcm_model`) sampling CMC A/B/C/T/G/S/R + scalp noise over the parity-gated `simulate_erp_dcm` forward, plus the amortized flow path (`ERPDCMPacker` + `ErpSummaryNet` + `amortized_erp_dcm_model`) reusing `ERPDCMForward.predict` with B held fixed — all additive, AutoGuide auto-discovers the sites.**

## Performance

- **Duration:** 34 min
- **Started:** 2026-06-26T15:07:24Z
- **Completed:** 2026-06-26T15:42:12Z
- **Tasks:** 3
- **Files modified:** 12 (3 created, 7 modified, +1 sbatch; +1573/-2)

## Accomplishments
- `erp_dcm_model.py`: full SVI generative model — samples `A_free` (4,N,N), per-effect `B_free_{j}`, `C_free`, `T`, `G`, `S`, `R` from `spm_cmc_priors.m` log-space Normals (mean 0), masks A/C dead→`-32` and B dead→`0`, delegates the forward ENTIRELY to `simulate_erp_dcm(..., l_full=...)` (the parity-gated pipeline; grep-verified NO second `integrate_local_linearization`/`project_to_scalp`), conditions `obs_erp ~ Normal(pred.reshape(-1), scalp_noise_scale)`.
- `ERPDCMPacker`: identity-reshape pack/unpack in the FROZEN `ERPDCMForward` order `A_free(4NN)|C_free(NM)|T(4N)|G(4N)|S(N)|R(2M)` — round-trips bit-for-bit, `n_features == ERPDCMForward.param_count` (38 at N=2,M=1), pack vector == `ERPDCMForward.pack_params` element-for-element. No `.exp()` (CMC free params already unconstrained).
- `ErpSummaryNet`: MLP over flattened `(Cnd*ns*Nc,)` → `embed_dim` (float64); `AmortizedFlowGuide(ErpSummaryNet(...), packer.n_features, packer=packer)` needs ZERO changes.
- `amortized_erp_dcm_model` + `_run_erp_forward_model`: single `_latent` (B EXCLUDED), repack via the frozen order, reuse `ERPDCMForward.predict` (B FIXED inside `forward` — no amortized scope creep); fixed observation noise (the flow guide samples only `_latent`).
- `create_guide(auto_normal)` auto-discovers all latent sites incl. `B_free_0` with ZERO factory edits (MODEL-06).

## Task Commits

1. **Task 1: erp_dcm_model.py (ERPDCM-01)** — `7bd3b71` (feat)
2. **Task 2: ERPDCMPacker + ErpSummaryNet (ERPDCM-02 A)** — `d34caa6` (feat)
3. **Task 3: amortized_erp_dcm_model + tests + sbatch (ERPDCM-02 B)** — `e189011` (feat)

## Files Created/Modified
- `src/pyro_dcm/models/erp_dcm_model.py` — NEW Pyro generative ERP-DCM (priors + Gaussian likelihood; reuses `simulate_erp_dcm`).
- `src/pyro_dcm/models/__init__.py` — export `erp_dcm_model`, `B_PRIOR_VARIANCE`.
- `src/pyro_dcm/guides/parameter_packing.py` — appended `ERPDCMPacker`.
- `src/pyro_dcm/guides/summary_networks.py` — appended `ErpSummaryNet`.
- `src/pyro_dcm/guides/__init__.py` — export `ERPDCMPacker`, `ErpSummaryNet`.
- `src/pyro_dcm/models/amortized_wrappers.py` — appended `amortized_erp_dcm_model` + `_run_erp_forward_model` (TYPE_CHECKING imports).
- `cluster/sbatch/erp_pytest.sbatch` — default `TEST_TARGET` += the two new test files (append-only).
- `tests/test_erp_dcm_model.py` — NEW structural-trace + AutoNormal-discovery + SVI-smoke.
- `tests/test_amortized_erp.py` — NEW packer round-trip/parity (laptop) + amortized flow-trains-without-error (`@pytest.mark.slow` → M3).

## Verification

- **LAPTOP slice** (`pytest tests/test_erp_dcm_model.py tests/test_amortized_erp.py -m "not slow"`): **6 passed, 1 deselected in ~11s** (<30s).
- **M3 amortized fit** (the `slow` jacrev path + full new-file suite): `sbatch cluster/sbatch/erp_pytest.sbatch` with `TEST_TARGET="tests/test_amortized_erp.py tests/test_erp_dcm_model.py"` → **M3 job 57903632 COMPLETED, exit 0, 7 passed in 91.25s** (the amortized flow guide trained on `erp_simulator`-equivalent draws without error, B fixed).
- **MUTAGEN models/ FOOTGUN — CLEARED:** `src/pyro_dcm/models/erp_dcm_model.py` confirmed present on M3 at `~/fc37/adam/projects/dcm_pytorch/src/pyro_dcm/models/erp_dcm_model.py`; `md5sum` of all 6 changed/new files **byte-identical laptop↔M3** after `mutagen sync flush`. The unanchored `models/` ignore did NOT exclude the new file (no scp stopgap needed).
- **Additive-only:** `git diff --stat HEAD -- forward_models/ simulators/erp_simulator.py utils/local_linearization.py inference/forward_models.py` EMPTY — Phase 33/34/35 forward stack + the `ForwardModel` protocol byte-untouched.
- **ruff:** `ruff check` clean on all new/changed source + test files; appended code is canonically `ruff format`-clean (pre-existing sibling-packer + `amortized_wrappers` import-sort format debt left untouched to stay additive — pre-existed at HEAD, error count unchanged).
- **mypy:** delta only the documented numpy-stub baseline (`numpy/__init__.pyi:737`).

## Pack Order (locked)

`ERPDCMPacker` and `erp_dcm_model`/`amortized_erp_dcm_model` mirror `ERPDCMForward`:

```
A_free (4*N*N) | C_free (N*M) | T (4*N) | G (4*N) | S (N) | R (2*M)
unpack shapes: A_free (4,N,N) | C_free (N,M) | T (N,4) | G (N,4) | S (N,1) | R (M,2)
```

B is EXCLUDED from the amortized packer (fixed), but IS sampled per-effect in the full SVI `erp_dcm_model`.

## Decisions Made
- **B-prior variance stayed PROVISIONAL `1/8`** (`B_PRIOR_VARIANCE`): `cmc_prior_moments` does not carry B (it is Phase-34 condition modulation, fixed in the parity gate), and the exact `pC.B` could not be transcribed from a line-cited SPM source in this environment. Flagged MUST-VERIFY in the model docstring + the module constant; low-stakes for the fixed-B headline demo, matters only for B-modulation SVI recovery (D1/D2). **Carried into Plan 36-02/03 as a flagged item.**
- **Amortized path holds B + observation noise FIXED.** The packer mirrors the frozen `ERPDCMForward` order (no B, no noise scalar), and `AmortizedFlowGuide` samples only `_latent`, so an extra noise sample site would be unmatched by the guide — observation noise is a fixed `obs_noise_std=1.0` kwarg (VL/SVI-full path estimates noise instead).
- **Format-debt policy:** made the appended code canonically `ruff format`-clean and left the pre-existing sibling-packer (older trailing-comma style) + `amortized_wrappers.py` import-sort (`I001`, present at HEAD) untouched, to keep the change strictly additive (running a full reformat would have rewritten pre-existing lines).

## Deviations from Plan

None — plan executed exactly as written. (No bugs/blockers required auto-fix; the only flagged open item, the B-prior variance, was explicitly anticipated by the plan as MUST-VERIFY and left provisional per its guidance.)

## Issues Encountered
- `ruff format --check` initially reported the two appended source files as needing reformat; inspection showed all hunks fell in PRE-EXISTING sibling code except a few in the appended region, which were hand-aligned to ruff's canonical style so the residual format diff is purely pre-existing debt (additive-preserving).
- `amortized_wrappers.py` carries a pre-existing `I001` import-sort error at HEAD (torch/pyro ordering); adding the `from typing import TYPE_CHECKING` stdlib import did NOT introduce a new error (count unchanged) — left as-is to avoid rewriting pre-existing import lines.

## Zotero / References Flagged
- **`pC.B` (between-trial B prior variance):** needs verification against `spm_dcm_erp.m` / `spm_cmc_priors.m`. No `[REF-xxx]` key fabricated. Citations in the new code are SPM source + line refs (`spm_cmc_priors.m`, `spm_gen_erp.m`, `spm_gen_Q.m`) + author/year (David & Friston 2003; Bastos 2012) — both are already in scope for v0.8.0; confirm they are in the Zotero folder before any manuscript `\cite`.

## Next Phase Readiness
- **Ready:** the ERP-DCM is now a *DCM* (SVI/VL inference via `create_guide`) AND an *amortized* model (flow guide trains without error). Plan 36-02 (`mmn_reference.py`, runs in parallel, zero file overlap) and Plan 36-03 (the 5-source MMN precision-sweep demo) can consume `erp_dcm_model` / `amortized_erp_dcm_model` directly.
- **Concern (flagged, non-blocking):** B-prior variance provisional `1/8` — verify before any B-modulation *recovery* claim (fixed-B demo unaffected).

---
*Phase: 36-erp-dcm-pyro-model-amortized-wiring-mmn-precision-sweep-demo*
*Completed: 2026-06-26*
