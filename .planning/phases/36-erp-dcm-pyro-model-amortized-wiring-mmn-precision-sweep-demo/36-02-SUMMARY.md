---
phase: 36-erp-dcm-pyro-model-amortized-wiring-mmn-precision-sweep-demo
plan: 02
subsystem: api
tags: [cmc, mmn, erp, dcm, forward-model, actinf_physics, lead-field, precision-gain]

# Dependency graph
requires:
  - phase: 34
    provides: "_MS_* locked 5-source MMN topology + apply_condition_modulation / parameterize_cmc_network (the spm_gen_Q diag(B)->G[:,0] port)"
  - phase: 33
    provides: "J_PERM permutation (free P.G[:,0] -> parameterised G[:,6] sp self-inhibition)"
  - phase: 35
    provides: "build_lead_field / cmc_default_pj / lfp_spatial (identity LFP single-dipole lead field) + simulate_erp_dcm scalp keys"
provides:
  - "build_mmn_5source_network() — the public, importable canonical 5-source auditory-MMN CMC graph (one source of truth, byte-identical to the parity-gated _MS_* fixture)"
  - "mmn_cmc_params(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag) — the forward-only actinf_physics Phase-133 adapter returning a ready-to-simulate bundle {p, a_masks, b_masks, c_mask, x_design, l_full}"
affects: [36-03, actinf_physics-phase-133]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Promote-locked-constant: a public builder lazily reads the private _MS_* topology from validation.export_to_mat (single source of truth, no re-typed edge lists), keeping the package import scipy-free"
    - "Knob->free-param adapter: consumer-facing scalars map to FREE log-space P struct; precision knob targets free P.G[:,0] (never parameterised G[:,6] directly — the J_PERM trap)"

key-files:
  created:
    - "src/pyro_dcm/forward_models/mmn_reference.py — build_mmn_5source_network + mmn_cmc_params"
    - "tests/test_mmn_reference.py — topology-equality + permutation/precision + adapter-bundle + fwd/bwd negative control (7 tests)"
  modified:
    - "src/pyro_dcm/forward_models/__init__.py — append build_mmn_5source_network, mmn_cmc_params exports"

key-decisions:
  - "Lazy-import the _MS_* constants inside the builder (not module-level) so forward_models import stays scipy-free and the topology stays single-sourced"
  - "Bundle a_masks/b_masks/c_mask are the FREE-LOG-SPACE params actually used (= p[A]/p[B]/p[C]), so the fwd/bwd negative control inspects what drives the sim"
  - "No MNI coordinates emitted at all (not even placeholder fields) — LFP scope; a no-coord-key test guards it"

patterns-established:
  - "Promote-locked-constant single-source-of-truth pattern for shared topology"
  - "Forward-only consumer adapter bundle contract (p + x_design + l_full ready for simulate_erp_dcm)"

# Metrics
duration: 20min
completed: 2026-06-26
---

# Phase 36 Plan 02: Public 5-Source MMN Network + actinf_physics CMC Adapter Summary

**`build_mmn_5source_network()` promotes the locked `_MS_*` auditory-MMN topology (A1L/A1R/STGL/STGR/rIFG, fwd/bwd/lateral-reciprocal, C into bilateral A1, deviant precision-`diag(B)`) to a public builder, and `mmn_cmc_params(...)` maps the four precision-sweep knobs to a ready-to-simulate forward-only CMC bundle whose `sp_inhibition_gain` drives the FREE `P.G[:,0]`→parameterised `G[:,6]` via `J_PERM[0]=6`.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-06-26T15:13:47Z
- **Completed:** 2026-06-26T15:34:10Z
- **Tasks:** 2
- **Files modified:** 3 (1 new module, 1 new test, 1 `__init__` append)

## Accomplishments

- **ERPDCM-03** — `build_mmn_5source_network()`: the canonical 5-source auditory-MMN CMC graph as a public, importable builder. Returns `{a_masks (4×(5,5) presence: fwd,fwd,bwd,bwd), b_masks ([(5,5)] value matrix: 0.3 edges + 0.5 precision diag), c_mask ((5,1): bilateral A1), x_design ((2,1): standard/deviant), source_names, precision_nodes (4,0,1)}`. Topology is held byte-identical to the SPM12-parity-gated `_MS_*` fixture by **lazily reading those constants** from `validation.export_to_mat` — no re-typed edge lists.
- **ERPDCM-05** — `mmn_cmc_params(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag="both")`: the forward-only `actinf_physics` Phase-133 adapter. Builds the free-log-space `p` struct (`T(5,4)/G(5,4)/C(5,1)/S(5,1)/R(1,2)/A(list[4] (5,5))/B([(5,5)])`) via the `_MS_A_LIVE`(0)/`_MS_A_DEAD`(-32) convention, computes the identity LFP lead field, and returns `{p, a_masks, b_masks, c_mask, x_design, l_full}` ready for `simulate_erp_dcm(bundle["p"], bundle["x_design"], 5, l_full=bundle["l_full"])`.
- **Precision-permutation guard reused**: `sp_inhibition_gain` sets the FREE `P.G[node,0]` at `{rIFG,A1L,A1R}`; the test confirms this moves the **parameterised `G[:,6]`** (sp self-inhibition, `J_PERM[0]=6`) and leaves parameterised `G[:,0]` byte-unchanged — `G[:,6]` is NEVER indexed directly.
- **NO MNI coordinates** hard-coded (a no-coord-key test enforces it); LFP scope per Garrido 2009 / Ranlund 2016.

## Public API (for 36-03 to consume)

```python
from pyro_dcm.forward_models import build_mmn_5source_network, mmn_cmc_params

net = build_mmn_5source_network()
# -> {"a_masks": [4×(5,5)], "b_masks": [(5,5)], "c_mask": (5,1),
#     "x_design": (2,1), "source_names": ("A1L","A1R","STGL","STGR","rIFG"),
#     "precision_nodes": (4,0,1)}

bundle = mmn_cmc_params(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag="both")
# -> {"p": {T,G,C,S,R,A,B}, "a_masks": p["A"], "b_masks": p["B"],
#     "c_mask": p["C"], "x_design": (2,1), "l_full": (5, 40)}
# fwd_bwd_flag in {"forward","backward","both"}; invalid -> ValueError
```

## Task Commits

1. **Task 1: build_mmn_5source_network() public MMN net** — `937abf9` (feat)
2. **Task 2: mmn_cmc_params adapter + topology/permutation tests** — `4818a11` (test)

_Note: the adapter `mmn_cmc_params` function body landed in the Task-1 file write (single coherent module) and its export in the Task-1 `__init__` append; Task 2's commit is its test battery (the deviation-rule-aligned atomic split — see Deviations)._

## Files Created/Modified

- `src/pyro_dcm/forward_models/mmn_reference.py` (created) — `build_mmn_5source_network`, `mmn_cmc_params`, plus `_ms_topology` (lazy single-source reader) and `_edge_mask` helpers.
- `tests/test_mmn_reference.py` (created) — 7 LAPTOP tests: topology equality vs `_MS_*` element-wise, no-MNI-coords, permutation `G[:,6]`-not-`G[:,0]`, free-`P.G[:,0]`-at-precision-only, adapter-bundle shapes + finite sim + non-zero scalp diff wave, fwd/bwd negative control, invalid-flag `ValueError`.
- `src/pyro_dcm/forward_models/__init__.py` (modified, append-only) — exports both symbols.

## Decisions Made

- **Lazy-import the `_MS_*` constants** inside `_ms_topology()` rather than at module top: keeps `from pyro_dcm.forward_models import ...` scipy-free (the package no longer pulls `scipy.io` at import) while preserving the single-source-of-truth contract (the builder reads the locked, parity-gated edge tuples). A library→validation read-only dependency is the deliberate cost of the no-divergence guarantee the plan mandates.
- **Bundle `a_masks`/`b_masks`/`c_mask` are the free-log-space params actually used** (`= p["A"]/p["B"]/p["C"]`), not the binary presence masks — so the consumer inspects exactly what drives the simulation and the fwd/bwd negative control is meaningful.
- **Zero MNI coordinate fields** (not even MUST-VERIFY placeholders): the plan said placeholders are acceptable only if marked and not asserted; emitting none is simpler and a positive no-coord-key test is the cleaner guard.

## Deviations from Plan

**1. [Process — atomicity] `mmn_cmc_params` body committed in the Task-1 file write**
- **Found during:** Task 1 (module authoring)
- **Issue:** The plan splits Task 1 (builder) / Task 2 (adapter `mmn_cmc_params` APPEND + tests). The module was written coherently with both functions, so the adapter function + its `__init__` export landed in the Task-1 commit (`937abf9`); Task-2's commit (`4818a11`) is the test battery only.
- **Impact:** None on correctness or additive-only scope — both functions exist, are exported, and are fully tested. The atomic-commit boundary shifted by one function relative to the plan's task split. No scope creep.

**2. [Rule 3 — Blocking, transient] Task-2 `git commit` timed out once (concurrent parallel-agent commit)**
- **Found during:** Task 2 commit
- **Issue:** The first `git commit` returned exit 143 (2-min timeout) — coincident with the parallel 36-01 agent committing `d34caa6` on the same branch (index contention / wait).
- **Fix:** Re-ran `git add tests/test_mmn_reference.py && git commit`; succeeded as `4818a11` (exit 0). No hooks involved (no `.pre-commit-config.yaml`, no custom `.git/hooks`).
- **Verification:** `git log` shows `4818a11`; the file is tracked.

---

**Total deviations:** 2 (1 process/atomicity note, 1 transient blocking retry)
**Impact on plan:** None on deliverables. Additive-only preserved; the parallel 36-01 working-tree change (`models/amortized_wrappers.py`) was NEVER staged by this agent.

## Issues Encountered

- **Parallel-agent working-tree change present:** `src/pyro_dcm/models/amortized_wrappers.py` (the 36-01 agent's uncommitted edit) and `validation/data/erp_multisource_input.mat` (a fixture artifact) appeared in `git status`. Per the critical constraint, this agent staged ONLY `forward_models/mmn_reference.py`, `forward_models/__init__.py`, and `tests/test_mmn_reference.py` (file-by-file `git add`, never `-A`). The locked Phase-33/34/35 modules and `validation/export_to_mat.py` are byte-untouched (`git diff 8507ba8 -- ...` empty).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **36-03 (the MMN precision-sweep demo) is unblocked:** it imports `build_mmn_5source_network` / `mmn_cmc_params` and sweeps `sp_inhibition_gain` through `simulate_erp_dcm(bundle["p"], bundle["x_design"], 5, l_full=bundle["l_full"])`, asserting monotone gain → `|MMN|` on the non-zero scalp difference wave. The bundle contract (`{p, a_masks, b_masks, c_mask, x_design, l_full}`) is locked here.
- **actinf_physics Phase-133** consumes `mmn_cmc_params(...)` directly as the forward-only CMC-params map.
- **Quality gates:** 7 new tests green in ~8s (laptop); 31 ERP-sibling tests green (66s); ruff check + format clean; mypy delta only the documented `pyro_dcm.*` import-untyped + numpy-stub baseline; additive-only verified.
- **Carried forward (unchanged from Phase 35):** the scalp difference-wave SIGN (negative-going / frontal dominance) and the ECD spatial path still need MNI coords + dipole orientation (Fact 6) — out of scope here.

---
*Phase: 36-erp-dcm-pyro-model-amortized-wiring-mmn-precision-sweep-demo*
*Completed: 2026-06-26*
