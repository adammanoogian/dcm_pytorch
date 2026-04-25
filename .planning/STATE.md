# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.4.0 Circuit Explorer -- Phase 17 COMPLETE 2026-04-24 on branch `gsd/phase-17-circuit-visualization-module` (gsd-verifier passed 15/15 must-haves against codebase at ef2a8d8; VIZ-01..10 flipped to Complete in REQUIREMENTS.md). Single-phase v0.4.0 milestone is functionally shippable pending /gsd:audit-milestone + /gsd:complete-milestone. Pure-Python `dcm_circuit_explorer/v1` JSON serializer (`src/pyro_dcm/utils/circuit_viz.py` 659 LOC + `tests/test_circuit_viz.py` 506 LOC) with zero upstream edits; 15 fast tests + 2 slow Pyro-integration smokes green. v0.3.0 Phase 16 cluster-acceptance re-run remains orthogonal and pending. One pre-existing test failure surfaced during Phase 17 regression (`tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys` -- broke when Phase 16 commit f7c2ba9 added `simulation_diverged` without updating this test); NOT a Phase 17 blocker; routed as a todo for v0.3.0 Phase 16 follow-up.

## Current Position

**Milestone:** v0.4.0 Circuit Explorer (started 2026-04-24; first active milestone work)
**Phase:** Phase 17 -- Circuit Visualization Module (Plan 17-01 COMPLETE 2026-04-24).
**Plan:** 17-01 COMPLETE 2026-04-24 (CircuitViz serializer: CircuitVizConfig dataclass + CircuitViz class + flatten_posterior_for_viz helper + 17 tests). Phase 16 history: 16-01 / 16-03 COMPLETE 2026-04-18; 16-02 COMPLETE 2026-04-19; 16-FIX COMPLETE 2026-04-24 (step-0 NaN root cause + seed-pool filter; supersedes commit 9c50011).
**Status:** Phase 17 Plan 01 CircuitViz serializer shipped (commits `57c5922` feat + `b5af070` test on `gsd/phase-17-circuit-visualization-module`). `src/pyro_dcm/utils/circuit_viz.py` (NEW, 659 LOC) ships `CircuitVizConfig` dataclass with 13 first-class fields (`schema`, `status`, `meta`, `palette`, `regions`, `region_colors`, `matrices`, `mat_order`, `phenotypes`, `hypotheses`, `drugs`, `peb`, `fitted_params`) PLUS a pass-through `extras: dict` field (V1) that collects every top-level JSON key NOT in `_FIRST_CLASS_KEYS` frozenset, enabling byte-identical round-trip of `configs/heart2adapt_dcm_config.json` (`_study`, `_description`, `node_info` preserved). `CircuitViz.from_model_config` requires `regions` + `region_colors` (length-matched per V2, ValueError with expected-vs-actual message otherwise) + `A_prior_mean`; emits sort-deterministic `mat_order = ['A'] + sorted(B_keys) + (['C'] if C else [])` (V7); accepts torch.Tensor / numpy.ndarray / nested list matrix values via private `_to_list_of_list` dispatcher (V8). `CircuitViz.from_posterior` verbatim handoff semantics + V6 pre-deepcopy `_validate_no_nan` guard with `(key, row, col)` localization. `CircuitVizConfig.export` calls `_validate_no_nan(fitted_params)` before `json.dumps` (V6 scope-extension). Module-level helper `flatten_posterior_for_viz(posterior, mat_order, b_masks)` (V4; recommended, not auto-invoked) bridges `extract_posterior_params` output to the `from_posterior` input shape -- handles `'A'` (from deterministic A site or `parameterize_A(A_free)` fallback), `'C'`, and `'B{j}'` (from `'B'` stacked site or `B_free_{j} + parameterize_B(b_mask)` fallback). `src/pyro_dcm/utils/__init__.py` additive re-export of `CircuitViz`, `CircuitVizConfig`, `flatten_posterior_for_viz` with `# Phase 17` markers. `tests/test_circuit_viz.py` (NEW, 506 LOC) ships 10 A-series structural acceptance tests (HEART2ADAPT round-trip, status-flip, no-mutation, 13-key surface, deterministic mat_order, schema version, list-of-list vals, tensor coercion, export round-trip, empty-optional-collections), 5 regression gates (V2 missing, V2 length-mismatch, V6 NaN, V6 Inf, V1 extras round-trip), and 2 slow `TestPyroIntegration` smokes (B-01 end-to-end SVI -> flatten -> fitted with 20-step bare-SVI bilinear loop + `functools.partial`-bound Predictive; B-02 shape contract A (3,3) -> 3x3 nested list). Tests: 15 fast in 2.5s + 2 slow in 12.5s = 17 passed. Ruff + mypy clean on all 3 Phase-17 files. Zero upstream edits: `git diff --name-only 9979b7e..HEAD` shows exactly the 3 allowed files (VIZ-10 gate satisfied). Full repo regression (`pytest tests/ -m "not slow"`) = 483 passed / 1 failed / 37 deselected in 35:30; the single failure is `tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys` which pre-dates Phase 17 -- it broke when Phase 16 commit `f7c2ba9` added `simulation_diverged` to `simulate_task_dcm`'s return dict without updating this test (unrelated to this plan; orthogonal fix). Task 3 (REQUIREMENTS.md VIZ-01..10 append) was already applied by the planner in commit `9979b7e` before execution started -- grep sentinel verification confirmed every assertion byte-identical; no additional commit (git hygiene forbids empty commits). Closes VIZ-01..10 (pending `/gsd:verify-phase 17` gate flip).
**Last activity:** 2026-04-24 -- Plan 17-01 execution shipped the CircuitViz v1 serializer on `gsd/phase-17-circuit-visualization-module`. Branch now has 4 commits since `cf5bc69` merge base: `8543992` (docs: handoff spec + renderer template + HEART2ADAPT config), `9979b7e` (docs: plan + VIZ-01..10 requirements), `57c5922` (feat(17-01): CircuitViz core), `b5af070` (test(17-01): acceptance tests + utils re-exports). One Rule-3 deviation auto-fix: B-01 bilinear posterior-extraction path switched from the plan-spec `run_svi(..., model_kwargs=...)` + `extract_posterior_params(guide, model_args, model=task_dcm_model)` to a bare SVI loop + `functools.partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` bound before Predictive, because Predictive invokes `model(*args)` with no kwarg forwarding so the bilinear branch never activated during posterior sampling. Pattern mirrors `tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction::test_extract_posterior_includes_B_free_and_B`. No V-decision semantics changed. Full SUMMARY: `.planning/phases/17-circuit-visualization-module/17-01-SUMMARY.md`.

Progress: v0.1.0 [██████████] 100% | v0.2.0 [██████████] 100% | v0.3.0 [██████████] Phases 13 + 14 + 15 complete + Phase 16 IMPLEMENTATION complete (cluster acceptance-gate re-run pending for milestone closure) | v0.4.0 [██░░░░░░░░] Phase 17 Plan 17-01 CircuitViz serializer shipped (Phase 17 verification pending)

## Decisions

- **v0.3.0 scope: bilinear-only.** DCM.5 (PEB-lite group GLM) and DCM.V3 (4-node HEART2ADAPT
  circuit) are HEART2ADAPT-specific despite the YAML framing and are deferred. DCM.V2 (SPM12
  cross-validation) deferred pending MATLAB access. See PROJECT.md `Current Milestone`.
- **Research: ON.** User opted to run the full 4-agent project-research pass despite the YAML
  citing Friston 2003 + SPM12 directly -- value is in surfacing B-matrix-specific pitfalls
  (identifiability under sparse modulatory events, prior scale interactions, etc.) that are
  not in-codebase today.
- **D1 - B_free prior variance = 1.0** (SPM12 one-state match; required for future DCM.V2
  cross-validation). Corrects the factually wrong YAML claim of "1/16 SPM12 convention."
- **D2 - Variable-amplitude semantics = per-event piecewise-constant.** Reuses existing
  `PiecewiseConstantInput`; `LinearInterpolatedInput` deferred to v0.3.1.
- **D3 - Recovery sign metric = split by magnitude.** sign_recovery_nonzero >= 80% on
  |B_true|>0.1 AND coverage_of_zero >= 85% on |B_true|<0.5*prior_std.
- **D4 - Eigenvalue stability monitor = strict `max Re > 0`, log-warn only.** Never raises
  during SVI; divergent draws are expected and hard-stops would corrupt gradients.
- **D5 - Amortized-guide bilinear support deferred to v0.3.1.** `amortized_wrappers.py` and
  `TaskDCMPacker` remain linear-only in v0.3.0; DCM.V1 acceptance uses SVI paths only.
- **Roadmap phase structure = 4 phases (13-16), 1:1 with requirement categories.**
  Alternative splits (e.g., parameterize_B vs full Pyro model, runner vs acceptance analysis)
  considered and rejected: the 1:1 structure matches the research-identified critical path
  and produces four independently shippable/testable gates with no artificial boundaries.
- **Plan 16-01 L1 - `run_svi` gains keyword-only `model_kwargs` parameter.** 5-line additive
  change; default `None -> {}` is bit-exact for every pre-v0.3.0 caller. Unlocks
  `task_dcm_model`'s keyword-only bilinear kwargs (`b_masks`, `stim_mod`) via
  `svi.step(*model_args, **kw)` forwarding. Rejected Path B (duplicate bare SVI loop in
  runner) because it drifts from shared optimizer / ELBO / NaN-guard / LR-decay
  infrastructure.
- **Plan 16-01 L2 - acceptance benchmark uses `auto_normal` + `init_scale=0.005`**
  (Phase 15 L2 bilinear half-default) hardcoded in `benchmarks/runners/task_bilinear.py`.
  Surfacing `init_scale` through `BenchmarkConfig` would require dataclass schema changes;
  hardcoding preserves the shared `.npz` reproducibility path. Multi-guide sidebar sweep
  deferred to v0.3.1.
- **Plan 16-01 L3 - linear baseline runs INLINE within the bilinear runner** on the same
  seeds/fixtures via `b_masks=None` MODEL-04 short-circuit. Eliminates machine-variance
  in RECOV-03's 1.25x relative threshold; each seed emits both `a_rmse_bilinear` and
  `a_rmse_linear`.
- **Plan 16-01 L4 - `n_datasets=10` default for `full_config('task_bilinear', 'svi')`;
  `quick_config` default is 3.** RECOV floor is >=10 seeds; research estimated ~80 min
  runtime at 10 seeds × 500 steps (observed CPU runtime in 2-seed smoke suggests the
  estimate is conservative; 16-02 may need parallel-seed execution or `num_samples=100`
  trim in Predictive to hit acceptance-gate runtime budget).
- **Plan 16-01 (`task_bilinear`) is EXPLICIT-ONLY in the runner CLI (`--variant all`
  does NOT include it).** Fixture generation (`benchmarks/generate_fixtures.py --variant
  all`) DOES include `task_bilinear` because fixture generation is cheap (seconds per
  fixture). Research Section 9 Q10 asymmetry.
- **Plan 16-03 L9 - `stimulus_mod_factory` signature is `Callable[[int], dict[str, torch.Tensor]]`**
  returning `{'times': (K,) float64, 'values': (K, J) float64}`. Closure captures
  `duration`/`dt`/`n_inputs` at construction time (cleaner than threading them through
  the factory signature because different factory types — epoch, sinusoid, HGF
  trajectory — have different natural closure contexts). Factory returns a BREAKPOINT
  DICT (not a `PiecewiseConstantInput` instance) so the runner can wrap at the call
  site and so that .npz fixtures stay format-compatible.
- **Plan 16-03 L10 - non-None factory bypasses fixture cache for stim_mod.** When
  `stimulus_mod_factory is not None`, `run_task_bilinear_svi` skips `_load_or_make_fixture`
  entirely and always inlines via `_make_bilinear_ground_truth_with_factory`. A/B/C
  ground truth still seed-deterministic; only stim_mod is factory-driven. Rationale:
  custom factories are test/sweep artifacts not reproducible-run artifacts; mixing
  cached `.npz` `stim_mod_values` with a custom factory would produce a stim_mod /
  B_true inconsistency.
- **Plan 16-03 - factory NOT stored in BenchmarkConfig.** Per CONTEXT.md, the factory
  is a runner-level kwarg (test/sweep injection point), NOT a config-level field —
  keeps `.npz` reproducibility path clean and avoids dataclass schema churn.
  `metadata['stimulus_mod_factory']` records only `'default_epochs'` or `'custom'`
  (no factory-hash tracking; research Section 7.2 N5 option c).
- **Plan 16-03 - mock sinusoid factory is Phase 16 placeholder, not physiologically
  meaningful.** `make_sinusoid_mod_factory(0.05 Hz, amplitude 0.5)` exists exclusively
  to exercise factory plumbing per CONTEXT.md "the indirection is proven wired, not
  a theoretical API." v0.3.1 SIM-06 HGF factories share the L9 signature.
- **Plan 17-01 V1 - `CircuitVizConfig.extras: dict = field(default_factory=dict)`
  pass-through field.** Every top-level JSON key NOT in the 13-key first-class set
  (`_schema`, `_status`, `meta`, `palette`, `regions`, `region_colors`, `matrices`,
  `mat_order`, `phenotypes`, `hypotheses`, `drugs`, `peb`, `fitted_params`) is collected
  by `CircuitViz.load` into `cfg.extras`. `to_dict()` merges extras after first-class
  fields with first-class winning on collision. Enables byte-identical HEART2ADAPT
  round-trip without bloating the first-class surface with HEART2ADAPT-specific keys
  (`_study`, `_description`, `node_info`) the renderer treats as optional.
- **Plan 17-01 V2 - `from_model_config` REQUIRES `model_cfg['region_colors']` with
  length-matched assertion.** ValueError with expected-vs-actual message on absence
  or mismatch. No matplotlib/tab10 fallback: implicit color synthesis produces
  non-reproducible JSON across matplotlib versions and violates the CLAUDE.md
  "expected vs actual" convention.
- **Plan 17-01 V4 - `flatten_posterior_for_viz(posterior, mat_order, b_masks=None)`
  stays a sibling helper, NOT auto-invoked from `from_posterior`.** Caller retains
  control over the A vs A_free transform choice and the B (stacked) vs B_free_j + mask
  source choice. Auto-flattening would hide those decisions. The `from_posterior`
  signature matches the handoff verbatim (`dict[str, list[list[float]]]` pre-flattened
  input).
- **Plan 17-01 V6 - `_validate_no_nan` is called BEFORE `copy.deepcopy` in
  `from_posterior` AND BEFORE `json.dumps` in `export`.** Error message includes the
  offending `(key, row, col)` tuple. Catches post-SVI NaN (a known Phase 16 scenario)
  at the serializer boundary with a clear localization rather than deferring to
  `json.dumps(allow_nan=False)`'s less-localized error.
- **Plan 17-01 V7 - `mat_order = ['A'] + sorted(B_matrices.keys()) + (['C'] if C)`
  is always sort-deterministic.** Never relies on caller dict insertion order.
  Guarantees reproducible JSON across runs and test seeds even when callers construct
  `B_matrices` in arbitrary key order.
- **Plan 17-01 V8 - `_to_list_of_list` private helper dispatches on type.**
  `torch.Tensor` -> `.detach().cpu().tolist()`, `numpy.ndarray` -> `.tolist()` (numpy
  import guarded by try/except ImportError), `list`/`tuple` -> nested list pass-through.
  Other types raise `TypeError` with expected-vs-actual. Keeps the from_model_config
  API ergonomic for callers building `model_cfg` from live `task_dcm_model` state
  without sacrificing determinism (tolist is lossless for float64).
- **Plan 17-01 V3/V5 deferrals.** JSON Schema codification (`.schema.json` file) is
  DEFERRED to a later v0.4.x patch (avoids `jsonschema` runtime dep + schema-drift
  risk). Headless-browser SVG-fidelity rendering tests are DEFERRED (disproportionate
  infra for a ~150-LOC serializer; structural assertions + renderer-safe-fallback
  coverage per 17-RESEARCH.md "Renderer behavior on missing fields" table are the
  acceptance surface).
- **Plan 17-01 B-01 deviation (Rule 3 blocking, auto-fixed).** The plan-spec B-01
  integration smoke used `run_svi(..., model_kwargs=bilinear_kwargs)` + `extract_posterior_params(guide, model_args, model=task_dcm_model)`. This fails
  because `Predictive(model, guide, ...)` invokes `model(*args)` with no kwarg
  forwarding, so the bilinear branch never fires during posterior sampling
  (posterior has `['A', 'A_free', 'C', 'median', 'noise_prec', 'obs', 'predicted_bold']`
  -- no `B_free_0`). Fix: bare 20-step SVI loop + `bilinear_model = functools.partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` passed as `model=bilinear_model` to
  `extract_posterior_params`. Pattern mirrors `tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction::test_extract_posterior_includes_B_free_and_B`. No
  V-decision semantics changed.

See `.planning/milestones/v0.2.0-ROADMAP.md` and `.planning/milestones/v0.1.0-ROADMAP.md` for prior milestones.

## Blockers

None currently.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 acceptance-gate test (Monash M3; ds_env/rlwm_gpu fallback; autopush to results branch) | 2026-04-19 | 6bade20 | [001-cluster-sbatch-phase-16-acceptance](./quick/001-cluster-sbatch-phase-16-acceptance/) |

### Pending Todos

0 pending — see `.planning/todos/pending/`. The prior init-scale-retry todo was moved to `.planning/todos/done/2026-04-23-phase16-nan-seeds-rootcause-and-fix.md` on 2026-04-24 when the actual root cause (fixture corruption, not init-scale sensitivity) was identified and fixed via the seed-pool filter. See "Last activity" above.

## Key Risks

- **Identifiability** of B-matrix elements under sparse or low-amplitude modulatory inputs
  (Rowe 2015). Mitigated by Phase 16 RECOV-07 shrinkage metric (`std_post/std_prior <= 0.7`
  target).
- **Numerical stability:** `A_eff(t) = A + Σ u_j·B_j` can become unstable under sampled B
  tails + sustained u_mod. Mitigated by Phase 13 BILIN-05 eigenvalue monitor + BILIN-06
  worst-case 3-sigma test.
- **Runtime:** Bilinear forward model adds per-timestep cost proportional to J modulators.
  Benchmarked against ~235s/500-step linear baseline in Phase 16 RECOV-08; expected 3-6x
  slowdown (Pitfall B10), flagged if >10x.
- **Amortized packer shape drift:** v0.2.0 `TaskDCMPacker` hardcodes linear sample sites.
  Mitigated by Phase 15 MODEL-07 explicit refusal + clear v0.3.1 deferral message
  (Pitfall B3).
- **A-RMSE inflation under bilinear parameter pricing:** even with B_true=0, Bayesian
  parameter pricing inflates A RMSE 10-30% (Pitfall B13). Mitigated by Phase 16 RECOV-03
  relative acceptance (<= 1.25x linear baseline), not the YAML's too-strict <= 0.15.

## Accumulated Context

### Roadmap Evolution

- 2026-04-17: v0.3.0 milestone started. Bilinear DCM extension scoped from
  `C:\Users\aman0087\Downloads\GSD_pyro_dcm.yaml`. HEART2ADAPT-specific tasks
  (DCM.5, DCM.V3) deferred; SPM12 cross-val (DCM.V2) deferred pending MATLAB.
- 2026-04-17: 4-agent research pass completed (STACK, FEATURES, ARCHITECTURE,
  PITFALLS -> SUMMARY.md). Verified SPM one-state prior variance = 1.0 (not 1/16
  as YAML claimed); documented 14 bilinear-specific pitfalls.
- 2026-04-17: REQUIREMENTS.md finalized with D1-D5 decisions resolved; 27 v0.3.0
  requirements across BILIN (7), SIM (5), MODEL (7), RECOV (8).
- 2026-04-17: ROADMAP.md appended with Phases 13-16 (4 phases, 1:1 category
  mapping). Coverage 27/27. Execution order enforced by data dependency chain:
  13 (forward model) -> 14 (simulator produces ground truth) -> 15 (Pyro model
  needs both) -> 16 (benchmark integrates everything).
- 2026-04-24: Phase 17 (Circuit Visualization Module) appended to v0.3.0 via
  /gsd:add-phase. Implements `src/pyro_dcm/utils/circuit_viz.py` per
  `docs/HANDOFF_viz.md`: Python class that serialises a DCM model config
  (+ optional fitted posterior means) into the `dcm_circuit_explorer/v1`
  JSON schema consumed by `docs/dcm_circuit_explorer_template.html`.
  Reference config `configs/heart2adapt_dcm_config.json` with
  `fitted_params: null` slot for SVI posterior means. Only
  `CircuitViz.from_model_config()` remains to implement; `from_posterior`
  and `load` fully specified in handoff.
- 2026-04-24 (SAME DAY REORG): Phase 17 MOVED from v0.3.0 into new v0.4.0
  Circuit Explorer milestone. Rationale: Phase 17 acceptance is strictly
  serialization/schema-structural (round-trip equality, `_status` toggle
  semantics, renderer compat) per user directive "this phase should be
  distinct in its acceptances from the rest of the fitting work — it is
  just a visualizer." No dependency on Phase 16 RECOV; depends only on
  Phase 15 `extract_posterior_params` (MODEL-05, already shipped). v0.3.0
  reverts to its original 4-phase scope (13-16) and remains gated on
  Phase 16 RECOV cluster re-run. v0.4.0 "Circuit Explorer" milestone is
  defined but not yet started; may run in parallel with v0.3.0 Phase 16
  slow acceptance-gate re-run since the two are independent. Phase 17
  research doc (`.planning/phases/17-circuit-visualization-module/17-RESEARCH.md`,
  MEDIUM-HIGH confidence, committed as cf5bc69) remains valid under the
  reorg since it was milestone-agnostic. v0.4.0 next step: either
  /gsd:new-milestone for a full milestone kickoff (PROJECT.md update +
  REQUIREMENTS.md derivation + research pass) OR resume /gsd:plan-phase 17
  directly using the existing RESEARCH.md. User to decide.
- 2026-04-24: **Phase 16.1 INSERTED (URGENT)** after Phase 16 via
  /gsd:insert-phase. Trigger: cluster acceptance re-run (SLURM job 54933838
  on `origin/results/phase16-acceptance-20260424-214708`, commit `f7c2ba9`,
  147 min on m3e103) **FAILED RECOV-04**: B-RMSE = 0.3424 vs <= 0.20
  threshold, distribution 0.335-0.348 across all 10 seeds (systematic
  underfit, not outlier noise). RECOV-03 (0.9972 <= 1.25), RECOV-05 (0.85
  >= 0.80), RECOV-06 (1.00 >= 0.85), RECOV-08 (1.15x wall-time) all PASS.
  RECOV-07 shrinkage means ~0.008 on nonnull B entries -> SVI guide
  collapsing B posterior toward zero. Seed-pool fix (f7c2ba9) worked as
  designed: corrupt seeds 44/49/50/53 correctly skipped and replaced.
  v0.3.0 milestone closure BLOCKED on Phase 16.1 root-cause + re-run.
  Phase 16.1 scope hypotheses captured in ROADMAP (prior-variance /
  init-scale, guide family, B_true magnitude vs prior, stim_mod SNR, step
  count / LR schedule). Next: /gsd:plan-phase 16.1 to derive
  requirements + plans. Phase 17 (v0.4.0) remains complete and
  independently shippable once v0.3.0 closes.

## Session Continuity

Last session: 2026-04-24 (Plan 17-01 execution)
Stopped at: Plan 17-01 COMPLETE -- CircuitViz v1 serializer
shipped on `gsd/phase-17-circuit-visualization-module` as
the first (and currently only) plan in the v0.4.0 Circuit
Explorer milestone. `src/pyro_dcm/utils/circuit_viz.py`
(NEW, 659 LOC) ships `CircuitVizConfig` dataclass with 13
first-class JSON fields + extras pass-through (V1), the
`CircuitViz` class with `from_model_config` / `from_posterior`
/ `load` / `export` static methods, and the module-level
`flatten_posterior_for_viz` helper bridging
`extract_posterior_params` output to the `from_posterior`
input shape (V4; recommended, not auto-invoked).
`src/pyro_dcm/utils/__init__.py` gains additive re-exports
with `# Phase 17` markers. `tests/test_circuit_viz.py` (NEW,
506 LOC) ships 10 A-series structural acceptance tests + 5
V1/V2/V6 regression gates + 2 slow Pyro-integration smokes
(B-01 end-to-end SVI->flatten->fitted, B-02 shape contract).
Fast pytest: 15 passed in 2.5s. Slow pytest: 2 passed in
12.5s. Ruff + mypy clean on all 3 Phase-17 files (mypy's 61
pre-existing errors elsewhere unchanged). Full repo fast
regression: 483 passed / 1 failed / 37 deselected in 35:30;
the one failure (`tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys`)
pre-dates Phase 17 -- it broke when Phase 16 commit `f7c2ba9`
added `simulation_diverged` to `simulate_task_dcm`'s return
dict without updating this test. Task 3 (REQUIREMENTS.md
VIZ-01..10 append) was already landed by the planner in
commit `9979b7e` before execution started; grep sentinels
confirmed byte-identical state and no additional commit was
created (git hygiene forbids empty commits). Two task commits
on branch since plan: `57c5922` (feat(17-01): CircuitViz
core) + `b5af070` (test(17-01): acceptance tests + utils
re-exports). Plan metadata commit + STATE update + SUMMARY
lands next in this session. One Rule-3 deviation auto-fix:
B-01 bilinear posterior-extraction path switched from the
plan-spec `run_svi(..., model_kwargs=...)` +
`extract_posterior_params(guide, model_args, model=task_dcm_model)`
to bare SVI + `functools.partial(task_dcm_model, b_masks=b_masks, stim_mod=stim_mod)` bound as
`model=bilinear_model` in `extract_posterior_params`, because
Predictive invokes `model(*args)` with no kwarg forwarding so
the bilinear branch never fired during posterior sampling.
Pattern mirrors existing
`tests/test_posterior_extraction.py::TestBilinearPosteriorExtraction::test_extract_posterior_includes_B_free_and_B`.
No V1-V8 decision semantics changed. SUMMARY path:
`.planning/phases/17-circuit-visualization-module/17-01-SUMMARY.md`.
Resume file: None. Next: either (a) `/gsd:verify-phase 17` to
flip VIZ-01..10 to Complete in REQUIREMENTS.md Traceability,
or (b) return to v0.3.0 Phase 16 cluster acceptance-gate
re-run to close v0.3.0.

### 2026-04-19 -- Plan 16-02 complete (prior session)

Last session: 2026-04-19 (Plan 16-02 finalization)
Stopped at: Plan 16-02 COMPLETE -- all 4 Phase 16 plans
shipped on `gsd/phase-16-bilinear-recovery-benchmark`.
Phase 16 implementation is DONE; only the `@pytest.mark.slow`-
gated `TestTaskBilinearAcceptance::test_acceptance_gates_pass_at_10_seeds`
remains to be RUN (not re-implemented) to confirm the 4 RECOV
gates pass on real 10-seed runner output. This finalization
session verified Tasks 1-3 against the plan's `<verify>` + 
`<verification>` blocks (grep sentinels all pass their
thresholds; ruff clean on the 3 files 16-02 authored/extended:
bilinear_metrics.py / test_bilinear_metrics.py /
test_task_bilinear_benchmark.py; fast pytest regression 103
passed 3 deselected slow in 290.69s), then shipped Task 4
(`docs(16-02): complete bilinear metrics and acceptance plan`
metadata commit including `.planning/phases/16-bilinear-recovery-benchmark/16-02-SUMMARY.md`
+ this STATE update). `benchmarks/bilinear_metrics.py`
provides 5 pure metric helpers + `compute_acceptance_gates`
single-source RECOV-03..08 pass/fail entry point; L5 pooled
aggregation in RECOV-05/06; L7 95% CI; SIGMA_PRIOR=1.0 +
RECOV thresholds per REQUIREMENTS.md with ONE documented
Rule-1 auto-fix (`RECOV_06_NULL_MASK`: 0.5 -> 0.1 to prevent
mis-classifying the 0.3/0.4 non-null B elements as null;
see 16-02-SUMMARY.md "Deviations from Plan" section 1).
`benchmarks/plotting.py` adds `plot_bilinear_b_forest` (L6
per-seed-median scatter + cross-seed median+IQR + B_true
reference dot + inline shrinkage annotation; 9 rows for
3-region J=1) and `plot_acceptance_gates_table` (6-row
pass/fail table with row-tinted PASS/FAIL/INFO/10x-FLAG
cells); `generate_all_figures` dispatches to both when a
`('task_bilinear', 'svi')` entry is present (tuple +
str-tuple keys). `tests/test_bilinear_metrics.py` has 11
unit tests (5 TestMetricHelpers + 6 TestAcceptanceGates
including FIX-2 AND-combination regression gate and FIX-5
insufficient_data guard); all 11 green in 0.23s.
`tests/test_task_bilinear_benchmark.py::TestTaskBilinearAcceptance::test_acceptance_gates_pass_at_10_seeds`
(@pytest.mark.slow) enforces FIX-1 `n_success >= 10`
precondition before gate computation, runs
`BenchmarkConfig.full_config('task_bilinear', 'svi')` with
`n_svi_steps=500` override (L8 + FIX-4 convergence caveat),
writes both figures to `tmp_path`, and asserts all 4 RECOV
gates pass with descriptive failure messages pointing at
likely root causes (Pitfall B13 A-RMSE inflation for
RECOV-03; mean-field correlation underestimation per L9 for
RECOV-06). Slow acceptance test was NOT run in this
finalization session (~80 min research estimate). 4
plan-16-02 commits on branch: b9aae53 (feat: bilinear_metrics
module), 50a08fb (feat: forest plot + acceptance-gate
table), 4bddd8e (test: unit tests + acceptance gate at 10
seeds), `<metadata-commit-2026-04-19>` (docs: complete
plan). File-ownership contract with plan 16-03 respected.

### Deviations from Plan 16-02

- **Rule 1 (bug fix):** `RECOV_06_NULL_MASK` hardcoded to
  0.1 instead of planner-specified `0.5 * SIGMA_PRIOR = 0.5`.
  Plan's 0.5 threshold would mis-classify the two non-null
  B elements (0.3 and 0.4) as null since both have `|B| < 0.5`,
  contaminating RECOV-06 coverage-of-zero with non-null
  elements whose 95% CIs would sometimes cover zero, directly
  contradicting the R-topology documented intent ("selects
  the 7 nulls per seed"). Fix: 0.1 (complementary to the
  0.1 non-null mask). Inline comment in
  `benchmarks/bilinear_metrics.py` lines 40-47 explains the
  deviation. Verified by all 11 unit tests green and
  `test_coverage_of_zero_matches_ci_containment` passing on
  all-zero `B_true`. Fix committed in `b9aae53`.

- **Plan inconsistency (NOT a deviation):** plan's
  `<verification>` block references
  `tests/test_svi_runner.py` but the actual file is
  `tests/test_svi_integration.py` (same inconsistency
  hit + documented by plan 16-01). Finalization session
  dropped the nonexistent filename from the regression
  command; no code change.

- **Plan inconsistency (NOT a deviation):** plan's verify
  commands reference `pytest --timeout=N` but the
  `pytest-timeout` plugin is not installed (same issue
  documented in plans 16-01 and 16-03). Verification ran
  without `--timeout`; no code change.

- **Pre-existing ruff errors in plotting.py (NOT this
  plan's):** full-file `ruff check benchmarks/plotting.py`
  reports 3 B905/B007 errors at lines 239, 461, 1595
  introduced by commit 47e850e (2026-04-03 "fix(benchmarks):
  rewrite plotting.py...") long before Phase 16. Plan
  16-02's additions at lines 1618+ are ruff-clean when
  checked in isolation. Out of scope for this plan;
  documented in the SUMMARY and deferred to a dedicated
  follow-up commit.

Next: RUN the `@pytest.mark.slow`-gated Phase 16 acceptance
gate (`pytest tests/test_task_bilinear_benchmark.py -m
slow -k acceptance`, ~80 min research estimate) to confirm
RECOV-03..06 all pass on real 10-seed runner output.
Passing closes Phase 16 and v0.3.0 Bilinear DCM Extension.
Failing RECOV-06 is an EXPECTED failure path per L9 (mean-
field posterior-correlation underestimation under AutoNormal;
documented deferral to v0.3.1 AutoLowRankMVN fallback tier).
Failing RECOV-04 or RECOV-05 -> first recovery action per
L8/FIX-4 is to rerun with `n_svi_steps=1500` (the
`full_config` default) BEFORE concluding the implementation
is incorrect.
Resume file: None

---

### 2026-04-18 -- Plan 16-03 complete (prior session)

Last session: 2026-04-18T21:08-21:21Z (Plan 16-03 execution)
Stopped at: Plan 16-03 complete -- HGF factory hook +
mock sinusoid factory + wiring tests shipped on branch
`gsd/phase-16-bilinear-recovery-benchmark`. `run_task_bilinear_svi`
gains keyword-only `stimulus_mod_factory: StimulusModFactory | None
= None` parameter (L9: `Callable[[int], dict[str, torch.Tensor]]`
returning `{'times': (K,), 'values': (K, J)}`). Default `None`
preserves plan 16-01 behavior bit-exactly. Non-None factory
bypasses `_load_or_make_fixture` for stim_mod (L10) — A/B/C/b_mask/
driving-stim still seed-deterministic; only stim_mod is
factory-driven. Module-level `make_sinusoid_mod_factory(duration=
200.0, dt=0.01, frequency=0.05, amplitude=0.5)` ships as Phase 16
placeholder/mock; v0.3.1 SIM-06 HGF factories will share the L9
signature. `_make_bilinear_ground_truth_with_factory` mirrors plan
16-01's helper but substitutes the factory's stim_mod (with
TypeError type-guard on malformed returns).
`metadata['stimulus_mod_factory']` records `'default_epochs'` or
`'custom'` on every return dict (no factory-hash tracking).
`tests/test_task_bilinear_factory.py::TestFactoryHookWiring` ships
3 tests: 1 fast `test_factory_signature_contract` (4.5s; closure
surface only — shape, dtype, determinism, configurability) + 2
`@pytest.mark.slow` tests (default-path regression + custom-factory
differs + metadata flip). Fast test passes alongside full
plan 16-01 SVI integration suite (12 passed, 2 deselected slow,
194s total). Ruff clean on both modified files. File-ownership
contract with parallel plan 16-02 respected: 16-03 only touched
`benchmarks/runners/task_bilinear.py` + new
`tests/test_task_bilinear_factory.py`. Closes 16-CONTEXT.md HGF
trajectory forward-compatibility hook lock-in. 3 plan-16-03
commits on branch: 9fb391f (feat: factory hook + mock + helper),
7596aa8 (test: factory wiring tests), `<metadata-commit>` (plan
completion). Plan 16-02 commits interleaved in parallel:
b9aae53 (feat: bilinear_metrics module), 50a08fb (feat: forest
plot + acceptance-gate table); 16-02 remains in flight.

### Deviations from Plan 16-03

- Plan grep sentinel expected `grep -c "make_sinusoid_mod_factory"
  benchmarks/runners/task_bilinear.py >= 2` (def + docstring), but
  Python convention is for a docstring NOT to mention its own
  function name; only the `def` line matches verbatim. Substantive
  sentinels (`stimulus_mod_factory >= 4` got 10,
  `_make_bilinear_ground_truth_with_factory >= 2` got 2,
  `StimulusModFactory >= 2` got 7) all pass; functional
  verification (factory shape + determinism + closure-config)
  passes via Python smoke. Analogous to plan 16-01's
  `model_kwargs >= 5` plan-inconsistency note.
- Plan verify commands referenced `pytest --timeout=N`, but the
  `pytest-timeout` plugin is not installed in the project
  environment (pytest exits 4 on the flag). Verification ran
  without `--timeout`; fast `test_factory_signature_contract`
  completed in 4.5s. Tooling note for future plan-writers: the
  Phase 8 / Phase 16 plans systematically reference
  `--timeout=N` but the plugin is not in the dependency stack.

Next: Plan 16-02 (metrics + forest plot + acceptance-gate table)
finishes Phase 16, closing v0.3.0 Bilinear DCM Extension
milestone. Plan 16-02 is in flight in parallel and writes its own
SUMMARY + STATE entry on completion.
Resume file: None

---

### 2026-04-18 -- Plan 16-01 complete (prior session)

Last session: 2026-04-18 (Plan 16-01 execution)
Stopped at: Plan 16-01 complete -- Phase 16 runner
infrastructure shipped. `run_svi` gains keyword-only
`model_kwargs` parameter (L1) forwarding kwargs to `svi.step`
and `guide.laplace_approximation`; default `None -> {}` is
bit-exact backward compat for every pre-v0.3.0 caller.
`benchmarks/generate_fixtures.py::generate_task_bilinear_fixtures`
produces per-dataset `.npz` ground truth with `B[1,0]=0.4`,
`B[2,1]=0.3`, `C[0,0]=0.5`, 4x12s epoch modulator at
`[20, 65, 110, 155]s`, SNR=3, TR=2, duration=200s, dt_sim=0.01;
registered in `_GENERATORS['task_bilinear']`.
`benchmarks/runners/task_bilinear.py::run_task_bilinear_svi`
fits BOTH bilinear `task_dcm_model` (via L1 `model_kwargs={b_masks,
stim_mod}`) AND bit-exact linear baseline (`b_masks=None`,
MODEL-04) on the SAME per-seed fixture (L3), returning per-seed
`a_rmse_bilinear_list`, `a_rmse_linear_list`, `time_bilinear_list`,
`time_linear_list`, `posterior_list` (with raw `B_free_0` samples
shape `(100, N, N)` for RECOV-06 coverage_of_zero), `b_true_list`,
and metadata. Guide is `auto_normal` + `init_scale=0.005`
bilinear / `init_scale=0.01` linear (L2; hardcoded in runner).
Registry + `VALID_COMBOS` + `VARIANT_EXPANSION` + argparse
`--variant` wired; `task_bilinear` is EXPLICIT-ONLY in
`run_all_benchmarks.py --variant all` (research Section 9 Q10;
fixture `--variant all` DOES include it because cheap).
`BenchmarkConfig.quick_config/full_config` defaults
`n_datasets=3/10, n_svi_steps=500/1500` (L4).
`tests/test_svi_integration.py::TestRunSVIModelKwargs` class
appended with 2 tests: bilinear-kwargs-forward + bit-exact
None-default backward-compat. Both pass (2/2 in 31.55s). Full
`test_svi_integration.py` 11/11 in 338.76s. Phase 15 regression
74/74 green post-change. `@pytest.mark.slow`-gated smoke test
in `tests/test_task_bilinear_benchmark.py::TestTaskBilinearSmoke`
asserts return-dict contract + metadata + `B_free_0` (3,3) shape
+ `B_true[0,1,0]=0.4` / `B_true[0,2,1]=0.3`; collection passes,
`-m "not slow"` cleanly deselects. Full slow run was terminated
after ~59 min of in-progress SVI (output buffered; first-test-
in-progress phase never reached completion; expected behavior
per plan's explicit "pass OR pytest.skip on insufficient_data"
clause). `pyproject.toml` slow marker was already registered
(pre-existing). Branch `gsd/phase-16-bilinear-recovery-benchmark`
now carries 4 plan-16-01 feat/test commits + metadata: 48c0c3c,
e8d56bb, 38c09a2, 97bfaa9, `<metadata>`. Plan 16-01 closes
RECOV-01 + RECOV-02 structural. Plan 16-02 (metrics + forest
plot + acceptance-gate table) and 16-03 (HGF factory hook +
mock + wiring test) are pending.

### Deviations from Plan 16-01

- Plan referenced `tests/test_svi_runner.py`; the actual file
  in-repo is `tests/test_svi_integration.py` (hosts the existing
  `TestRunSvi` class). Applied Deviation Rule 3 (blocker fix):
  `TestRunSVIModelKwargs` class appended to the actual file.
  No functional change — implementation is line-identical to
  what the plan specified.
- Plan's global verification sentinel expected `grep -c
  "model_kwargs" guides.py >= 5`; Task 1 action body spec
  produces 4 occurrences (signature + docstring param +
  docstring body + `kw = model_kwargs or {}`). This is a
  plan inconsistency (Task 1's own verify expected `>= 4`);
  implementation matches Task 1's spec exactly.

Next: Plan 16-02 recovery-benchmark metrics + figures +
acceptance-gate table. Consumes `posterior_list`,
`a_rmse_bilinear_list`, `a_rmse_linear_list`,
`time_bilinear_list`, `b_true_list` from plan 16-01's runner
output contract.
Resume file: None

---

### 2026-04-18 -- Plan 15-03 complete (prior session)

Plan 15-03 complete -- Phase 15 COMPLETE. Defense-in-
depth bilinear refusal at BOTH user surfaces
(`TaskDCMPacker.pack` + `amortized_task_dcm_model`) raising
`NotImplementedError` with literal `v0.3.1` per D5.
`amortized_task_dcm_model` signature gains `*, b_masks=None,
stim_mod=None` kwargs with guard firing BEFORE
`_sample_latent_and_unpack`; linear None/[] pass through for API
symmetry with `task_dcm_model`. `extract_posterior_params`
docstring-only edit documents `B_free_0..B_free_{J-1}` raw keys +
`B` (J,N,N) deterministic key with explicit cross-Pyro-version
`return_sites` portability note; no code change (function was
already site-agnostic via `samples.items()`). 5 new tests (2
TestAmortizedRefusesBilinear + 2 TestTaskDCMPackerBilinearRefusal
+ 1 TestExtractPosteriorBilinear). TestExtractPosteriorBilinear
runs 20 SVI steps on bilinear task_dcm_model via bare SVI loop
(NOT run_svi -- run_svi's positional model_args cannot forward
task_dcm_model's keyword-only b_masks/stim_mod; F401 hygiene via
local imports omitting run_svi) then calls `Predictive` with
explicit `return_sites=['A_free','C','noise_prec','B_free_0','B']`
for portable assertion across Pyro 1.9+ patch versions;
supplementary `extract_posterior_params` default-return_sites
check asserts only always-present `B_free_0`. 81/81 Phase-15
suite green (19 test_task_dcm_model + 30 test_guide_factory + 14
test_posterior_extraction + 7 test_amortized_task_dcm non-slow +
11 test_parameter_packing); 51/51 Phase 13+14 regression green.
SpectralDCMPacker + amortized_spectral_dcm_model UNMODIFIED
(bilinear = task-only). Branch `gsd/phase-15-pyro-bilinear-model`
now carries 9 Phase-15 commits: 23a5591, cd405d2, 807fb46,
86cdb76 (15-01); 9b796c0, e1d986b (15-02); 6c68b10, 66cab62,
b9928c2 (15-03).
Next: Phase 16 recovery benchmark (RECOV-01..08).
Resume file: None

---

### 2026-04-18 -- Plan 15-03 complete

- `src/pyro_dcm/guides/parameter_packing.py`:
  - `TaskDCMPacker.pack` gains a 15-line bilinear refusal guard
    (3-line comment + 3-line scan + 8-line raise) at the very top
    of the method body, BEFORE the existing `a_flat = ...` line.
    `bilinear_keys = [k for k in params if k.startswith("B_free_")]`
    collects bilinear keys; non-empty list raises
    `NotImplementedError` with message literal including `v0.3.1`,
    `sorted(bilinear_keys)` for diagnostic, and path forward
    (`create_guide(task_dcm_model)`). Method body otherwise
    byte-identical; docstring unchanged (describes linear
    behavior, still what the method does when no bilinear keys).
  - `SpectralDCMPacker` UNMODIFIED (spectral DCM is not in
    MODEL-07 scope per plan).
- `src/pyro_dcm/models/amortized_wrappers.py`:
  - `amortized_task_dcm_model` signature extended with keyword-
    only `b_masks: list[torch.Tensor] | None = None` and
    `stim_mod: object | None = None` after new `*` sentinel. All
    8 pre-existing positional parameters untouched. Pre-existing
    pre-15-03 callers (which never pass these kwargs) continue to
    work unchanged.
  - 11-line refusal guard (3-line comment + 2-line gate + 8-line
    raise) at method entry, BEFORE
    `params = _sample_latent_and_unpack(packer)`. Guard fires ONLY
    when `b_masks is not None and len(b_masks) > 0` (None and []
    both pass through to linear body -- API symmetry with
    `task_dcm_model`). `del stim_mod` added to mark the kwarg
    intentionally unused in linear mode.
  - Docstring expanded: 2 new Parameters entries (`b_masks`,
    `stim_mod`), new "Bilinear support" Notes paragraph
    explaining v0.3.1 deferral per D5 with path forward
    (`create_guide(task_dcm_model) + run_svi`) and Pitfall B3
    citation. Docstring mentions `v0.3.1` 4 times (Parameters +
    Notes + existing placeholder sections); raise mentions once.
    Total 5 `v0.3.1` refs in the file.
  - `_run_task_forward_model`, `_sample_latent_and_unpack`, and
    `amortized_spectral_dcm_model` UNMODIFIED.
- `src/pyro_dcm/models/guides.py`:
  - `extract_posterior_params` docstring gains 17-line Notes
    paragraph ("Bilinear task DCM sites (v0.3.0+):") and 8-line
    Examples block showing `posterior['B_free_0']` (always
    available; shape `(N, N)`) and conditional
    `posterior['B']` access (shape `(J, N, N)`). Notes paragraph
    explicitly cites MODEL-05 and documents cross-Pyro-version
    `return_sites` portability for deterministic sites.
  - NO code change in function body (body was already
    site-agnostic via `samples.items()` iteration per 08-05
    design).
- `tests/test_amortized_task_dcm.py`:
  - New `TestAmortizedRefusesBilinear` class appended with 2
    tests:
    - `test_amortized_wrapper_refuses_bilinear_kwargs` -- non-empty
      `b_masks=[b_mask_0]` + `stim_mod=PiecewiseConstantInput(...)`
      kwargs trip the guard; `pytest.raises(NotImplementedError,
      match=r"v0\.3\.1")`. Uses `make_epoch_stimulus` (Pitfall B12
      boxcar primitive) for `stim_mod` fixture.
    - `test_amortized_wrapper_linear_mode_unchanged` -- MODEL-07
      regression gate: `b_masks=None` (default) AND `b_masks=[]`
      both pass through `pyro.poutine.trace(...).get_trace(...)`
      without raising. Uses `make_random_stable_A` +
      `simulate_task_dcm` to produce a valid 20s 3-region linear
      BOLD series; trivial single-item `fit_standardization`
      (emits `UserWarning: std(): degrees of freedom is <= 0` --
      benign, std clamps to 1e-6).
  - All 6 pre-existing tests (5 non-slow + 1 slow) UNCHANGED.
- `tests/test_parameter_packing.py`:
  - New `TestTaskDCMPackerBilinearRefusal` class appended with 2
    tests:
    - `test_packer_refuses_bilinear_keys` -- params dict with
      added `B_free_0` key raises `NotImplementedError` matching
      `r"v0\.3\.1"`.
    - `test_packer_accepts_linear_keys_after_bilinear_guard` --
      regression gate: after the guard is added, linear
      pack/unpack still produces correct 13-element packed vector
      and correct `(3, 3)` / `(3, 1)` unpacked shapes.
  - All 9 pre-existing tests UNCHANGED.
- `tests/test_posterior_extraction.py`:
  - New `TestExtractPosteriorBilinear` class appended with 1 test:
    - `test_extract_posterior_includes_bilinear_sites` -- 20 SVI
      steps on bilinear `task_dcm_model` via a bare SVI loop
      (`SVI(task_dcm_model, guide, ClippedAdam, Trace_ELBO())` +
      `svi.step(**model_kwargs)` to forward keyword-only
      `b_masks`/`stim_mod`). `create_guide(task_dcm_model,
      init_scale=0.005)` (L2 from 15-01 propagation). Then
      `Predictive(partial(task_dcm_model, b_masks=b_masks,
      stim_mod=stim_mod), guide=guide, num_samples=10,
      return_sites=['A_free','C','noise_prec','B_free_0','B'])`
      (explicit list for cross-Pyro-version portability -- Plan
      15-03 checker Blocker 2 resolution). Asserts
      `samples['B_free_0'].shape[-2:] == (N, N)` AND
      `samples['B'].shape[-3:] == (J, N, N)`. Supplementary
      assertion on `extract_posterior_params(guide, model_args,
      model=bilinear_model, num_samples=10)` requires only
      `B_free_0` (always a pyro.sample site; not Pyro-version
      coupled) with shape `(N, N)`; also asserts linear sites
      `A_free`, `C`, `noise_prec` present (regression check) and
      `B_free_0` in `posterior["median"]` (backward compat).
    - `run_svi` INTENTIONALLY NOT imported in the new test block
      (F401 hygiene -- Plan 15-03 checker Major 3 resolution).
      Pre-existing file-level `run_svi` import remains for
      `_train_guide` helper used by other tests.
    - `caplog.set_level(logging.ERROR, logger='pyro_dcm.stability')`
      autouse fixture silences stability monitor WARN spam (D4 +
      R6 pattern from 15-01).
  - All 13 pre-existing tests UNCHANGED.
- Commits: `6c68b10` `feat(15-03): refuse bilinear sites in
  TaskDCMPacker + amortized_task_dcm_model`; `66cab62`
  `docs(15-03): document bilinear posterior keys in
  extract_posterior_params`; `b9928c2` `test(15-03): bilinear
  refusal + posterior-extraction tests`.
- Verification: 5 new tests green individually;
  `tests/test_amortized_task_dcm.py + tests/test_parameter_packing.py +
  tests/test_posterior_extraction.py` non-slow 32/32 in 50.53s;
  `tests/test_task_dcm_model.py` 19/19 in 31.39s (Phase 15-01
  regression); Phase 13+14 regression
  (`test_linear_invariance + test_coupled_system_bilinear +
  test_bilinear_utils + test_bilinear_simulator + test_stimulus_utils`)
  51/51 green in 232.90s (0:03:52); full Phase-15 suite
  (`test_task_dcm_model + test_guide_factory +
  test_posterior_extraction + test_amortized_task_dcm +
  test_parameter_packing` non-slow) 81/81 green in 92.91s
  (0:01:32). Defense-in-depth verification passes
  (`OK defense in depth` inline script).
- **Grep sentinels** (all met):
  - `parameter_packing.py`: `v0\.3\.1` ×2 (comment + raise
    message), `NotImplementedError` ×1 (raise), `bilinear_keys`
    ×2 (scan + if-check).
  - `amortized_wrappers.py`: `v0\.3\.1` ×5 (raise message +
    docstring expansions), `NotImplementedError` ×3 (1 raise + 2
    docstring refs; plan target "exactly 1" was for the raise
    only -- docstring references are acceptable variance per
    15-01 precedent), `b_masks` ×8 (signature + gate + del +
    docstring entries + Notes).
  - `guides.py`: `Bilinear task DCM` ×1 (Notes paragraph header),
    `B_free_0` ×4 (docstring + examples), `(J, N, N)` ×3
    (docstring + examples), `MODEL-05` ×1 (requirement tag),
    `return_sites` ×2 (Notes + Examples).
  - `test_amortized_task_dcm.py`: `TestAmortizedRefusesBilinear`
    ×1, `v0\\.3\\.1` ×1, `B_free_` ×3 (fixture + match + comment).
  - `test_parameter_packing.py`: `test_packer_refuses_bilinear_keys`
    ×1, `v0\\.3\\.1` ×1, `B_free_` ×3.
  - `test_posterior_extraction.py`:
    `test_extract_posterior_includes_bilinear_sites` ×1,
    `B_free_0` ×23 (fixture + return_sites + shape asserts +
    posterior indexing + comments), `return_sites` ×14 (list def
    + Predictive kwarg + docstring + portability commentary),
    `run_svi` ×4 (3 pre-existing + 1 new comment block
    explaining why NOT imported; 0 new imports).
- **Ruff status:** 6 pre-existing errors in the 3 modified test
  files (I001 ×3, F401 ×1, B007 ×1, E741 ×1), 3 pre-existing in
  `guides.py` (I001, UP035, F401), 1 pre-existing in
  `amortized_wrappers.py` (I001). All verified pre-existing via
  `git stash` round-trip. Plan 15-03 adds ZERO new ruff errors.
  Pre-existing lint not fixed per Phase 14-02 / Plan 15-01
  additive-plan precedent.
- **D5 applied** (from STATE.md Decisions + plan frontmatter):
  Amortized bilinear inference deferred to v0.3.1 per D5. BOTH
  refusal surfaces (packer + wrapper) reference literal `v0.3.1`
  in error messages. DCM.V1 acceptance (Phase 16) uses SVI paths
  only.
- **L1 applied** (inherited from 15-01 via plan frontmatter):
  `posterior['B_free_0']['mean'].shape == (N, N)` assertion in
  `test_extract_posterior_includes_bilinear_sites` relies on 15-01
  sampling `B_free_j` as full `(N, N)` `Normal(0, 1.0).to_event(2)`
  (NOT a flat vector of free entries). Test passed, confirming
  L1 from 15-01 is stable.
- **L3 applied** (inherited from 15-01 via plan frontmatter):
  `samples['B'].shape[-3:] == (J, N, N)` assertion relies on
  15-01 emitting `pyro.deterministic('B', B_stacked)` in the
  bilinear branch. Test passed with explicit `return_sites`
  including `'B'`, confirming L3 from 15-01 is stable and
  `Predictive` honors `return_sites` for deterministic sites.
- **Deviations:** None -- plan executed exactly as written; no
  auto-fix rules triggered. Minor sentinel variances documented
  (all within acceptable ranges per 15-01 precedent; docstring
  references count toward literal-pattern greps).
- Requirements closed: MODEL-05 (docs + executable gate),
  MODEL-07 (defense-in-depth refusal at both packer and wrapper
  surfaces). Phase 15 now 7/7 requirements closed. Phase 15
  COMPLETE.

---

### 2026-04-18 -- Plan 15-02 complete

- `tests/test_guide_factory.py`:
  - Import block widened: `from pyro_dcm.models.task_dcm_model import
    task_dcm_model`; `from pyro_dcm.simulators.task_simulator import
    (make_block_stimulus, make_epoch_stimulus, make_random_stable_A,
    simulate_task_dcm)`; `from pyro_dcm.utils.ode_integrator import
    PiecewiseConstantInput` — alphabetized within each `from ... import
    (...)` block.
  - New module-scoped fixture `task_bilinear_guide_data()` (3-region,
    J=1, 30s simulation): mirrors the structure of
    `tests/test_task_dcm_model.py::task_bilinear_data` by
    duplication (plan truth-3 preference: test-local fixture over
    fragile cross-test-file import). `A = make_random_stable_A(3,
    density=0.5, seed=42)`; `C = [[0.25],[0.0],[0.0]]`;
    `b_masks = [b_mask_0]` with `b_mask_0[1,0] = 1.0` (zero diagonal
    per Pitfall B5); `stim_mod = PiecewiseConstantInput` from
    `make_epoch_stimulus(event_times=[10.0], event_durations=[10.0],
    event_amplitudes=[1.0], duration=30.0, dt=0.01, n_inputs=1)`
    (Pitfall B12 preferred boxcar primitive). Returns dict with 12
    keys including observed_bold, stimulus, a/c_mask, t_eval, TR,
    dt, N, M, J, b_masks, stim_mod.
  - New module-level constant `_BILINEAR_GUIDE_VARIANTS =
    ["auto_normal", "auto_lowrank_mvn", "auto_iaf"]` with a multi-
    paragraph docstring explaining per-variant AutoGuide discovery
    mechanics (auto_normal deep_setattr vs auto_lowrank_mvn /
    auto_iaf AutoContinuous _latent concatenation), init_scale
    portability (source-cited `guides.py:54-58` + `guides.py:171-172`
    guard), and the hidden_dim sizing rationale for auto_iaf.
  - New module-private helper `_guide_kwargs_for(guide_type: str) ->
    dict` returning `{"init_scale": 0.005}` for all variants AND
    `{"hidden_dim": 64}` when `guide_type == "auto_iaf"`. Called
    as `create_guide(task_dcm_model, guide_type=guide_type,
    **_guide_kwargs_for(guide_type))` in both test methods.
  - New `TestBilinearDiscovery` class with `autouse=True` fixture
    `_silence_stability_logger` using
    `caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")`
    per D4 + R6. Two parametrized methods across 3 variants (6
    tests total):
    - `test_b_free_sites_in_prototype_trace[auto_normal |
      auto_lowrank_mvn | auto_iaf]` — `pyro.clear_param_store()`;
      build guide via `create_guide`; call `guide(**model_kwargs)`
      under `torch.no_grad()` to trigger lazy `_setup_prototype`;
      assert `f"B_free_{j}" in guide.prototype_trace.nodes` for
      each j in range(J); also assert pre-Phase-15 sites A_free + C
      still present (sanity).
    - `test_b_free_sites_in_param_store_after_svi[auto_normal |
      auto_lowrank_mvn | auto_iaf]` — `pyro.clear_param_store()`;
      build `SVI(task_dcm_model, guide, ClippedAdam({lr: 0.01,
      clip_norm: 10.0}), loss=Trace_ELBO())`; run 20 steps asserting
      `math.isfinite(loss)` at every step; for auto_normal assert
      at least one `pyro.get_param_store()` key contains
      f"B_free_{j}" substring; for auto_lowrank_mvn / auto_iaf
      assert `B_free_{j}` still in `guide.prototype_trace.nodes`
      post-SVI AND `len(param_names) > 0`.
- Commits: `9b796c0` `test(15-02): bilinear guide auto-discovery
  across 3 autoguide variants`.
- Verification: `tests/test_guide_factory.py::TestBilinearDiscovery`
  6/6 in 25.55s (budget <90s — 71% under); full
  `tests/test_guide_factory.py` 30/30 in 26.41s (24 pre-existing +
  6 new; zero pre-existing regressed); Phase 15-01 regression
  `tests/test_task_dcm_model.py` 19/19 in 43.33s; Phase 13+14+15-01
  full subset (test_task_dcm_model + test_guide_factory +
  test_linear_invariance + test_coupled_system_bilinear +
  test_bilinear_utils + test_bilinear_simulator +
  test_stimulus_utils + test_neural_state + test_stability_monitor)
  113/113 in 392.97s (6:32). Ruff on new code clean; pre-existing
  I001 (line 9-32 import sort) + D403 (line 245
  `test_kwargs_passthrough_lowrank` docstring) confirmed via
  `git stash` round-trip, not touched per Phase 14 precedent.
- **Grep sentinels** (all met, all above plan thresholds):
  `class TestBilinearDiscovery` 1 (plan target exactly 1),
  `_BILINEAR_GUIDE_VARIANTS` 3 (plan target >=2),
  `prototype_trace` 11 (plan target >=3),
  `pyro_dcm.stability` 2 (plan target >=1),
  `B_free_` 19 (plan target >=4),
  `init_scale=0.005` 3 (plan target >=2),
  `task_bilinear_guide_data` 24 (plan target >=3),
  `auto_normal` 10, `auto_lowrank_mvn` 9, `auto_iaf` 16 (plan
  targets each >=2). `_guide_kwargs_for` 4 (def + 2 call sites +
  docstring).
- **L1 inherited from 15-01 (active):** tests rely on 15-01's full
  `(N, N) .to_event(2)` B-site shape — if 15-01 had used flat-vector
  alternative, `AutoGuide._setup_prototype` would still register
  the site but its event dim would differ and the prototype_trace
  assertion would still pass structurally (so L1 is NOT directly
  verified by this plan — but the SVI smoke IS sensitive to
  event-dim shape and passes cleanly, which is indirect evidence
  that L1 holds).
- **L2 inherited from 15-01 (applied):** `init_scale=0.005` passed
  explicitly to `create_guide` for all three variants via
  `_guide_kwargs_for`. For auto_normal + auto_lowrank_mvn it flows
  through to the AutoGuide ctor (per `_INIT_SCALE_GUIDES` guard at
  `guides.py:54-58`); for auto_iaf it is silently dropped inside
  `create_guide` at `guides.py:171-172` (no TypeError; verified by
  green test). Plan truth-6 portability claim confirmed in practice.
- **L3 inherited from 15-01 (indirect):** `pyro.deterministic("B",
  ...)` continues to be guarded to the bilinear branch — 15-01's
  `test_linear_reduction_when_b_masks_none` continues to assert
  this via exact set-equality. Not re-verified by 15-02 tests.
- **Deviations (Rule 3 blocking fix):**
  - `_guide_kwargs_for` helper added to supply `hidden_dim=64` for
    auto_iaf. `AutoIAFNormal` wraps `pyro.nn.AutoRegressiveNN` which
    raises `ValueError: Hidden dimension must not be less than input
    dimension.` at `auto_reg_nn.py:206` when `min(hidden_dims) <
    input_dim`. Bilinear `task_dcm_model` has latent dim 22 (A_free=9
    + C=3 + noise_prec=1 + B_free_0=9) which exceeds `create_guide`
    default `hidden_dim=[20]` (set at `guides.py:181`). The plan's
    truth-7 init_scale portability claim is CORRECT but did not
    anticipate this orthogonal constructor-arg failure. Fix is
    TEST-SIDE-ONLY: helper function + 2 call-site replacements +
    one docstring sentence. `create_guide` source unchanged; plan
    truth-5 ("Zero edits to `src/pyro_dcm/models/guides.py`")
    remains satisfied. `hidden_dim=64` gives 2.9x margin over the
    22-dim floor — no tuning sensitivity expected for Phase 16
    (3-8 regions; worst-case latent ~20 + J*N*N for moderate J).
- **Parallel-execution coordination:** Plan 15-03 committed
  `6c68b10 feat(15-03): refuse bilinear sites in TaskDCMPacker +
  amortized_task_dcm_model` on the same branch during this plan's
  execution window. Verified file-level non-overlap: 15-03 touches
  `src/pyro_dcm/guides/parameter_packing.py`,
  `src/pyro_dcm/models/amortized_wrappers.py`, and test files
  (test_amortized_task_dcm.py, test_parameter_packing.py,
  test_posterior_extraction.py). 15-02 touches only
  `tests/test_guide_factory.py`. Staged individually via `git add
  tests/test_guide_factory.py` (no `git add .` or `-A`).
- **Known follow-ups (out of scope for 15-02):**
  1. `auto_mvn` docstring update in `src/pyro_dcm/models/guides.py`
     noting bilinear J > 1 cost scaling (research R3) — deferred
     per plan truth-7.
  2. Consider auto-scaling `hidden_dim` default in `create_guide`
     for `auto_iaf` based on estimated latent dim (heuristic:
     `max(20, 2 * est_latent)`). Currently downstream callers for
     models with >20 latent dims must pass `hidden_dim` explicitly;
     `_guide_kwargs_for` is the reference pattern. Explicit
     `create_guide(..., bilinear_mode=True)` auto-switch REJECTED
     per 15-RESEARCH.md Section 14 Q1.
- Requirements closed: MODEL-06. Phase 15 now 5/7 (MODEL-01..04
  from 15-01, MODEL-06 from 15-02; MODEL-05 + MODEL-07 pending in
  Plan 15-03).

---

### 2026-04-18 -- Plan 15-01 complete

- `src/pyro_dcm/models/task_dcm_model.py`:
  - Import block widened: `parameterize_B` added to the
    `neural_state` import; `PiecewiseConstantInput` +
    `merge_piecewise_inputs` added to the `ode_integrator` import
    (alphabetized within the existing `from ... import ( ... )`
    blocks).
  - New module-level constant `B_PRIOR_VARIANCE: float = 1.0` with a
    NumPy-style docstring citing D1 (SPM12 `spm_dcm_fmri_priors.m`
    pC.B = B one-state match) and explicitly noting the YAML "1/16"
    claim was audited as factually wrong (v0.3.0 PITFALLS.md Section
    B8). Future one-state vs two-state alternatives tracked in
    REQUIREMENTS.md future-candidate BILIN-08 (v0.4.0+ scope).
  - New module-private helper `_validate_bilinear_args(b_masks,
    stim_mod, N) -> None` raising `ValueError` on None `stim_mod` /
    per-index `b_masks[j].shape != (N, N)` / `len(b_masks) !=
    stim_mod.values.shape[1]`, and `TypeError` if `stim_mod` lacks
    `.values` attr. Error messages include expected-vs-actual values
    per CLAUDE.md convention. Called ONCE from `task_dcm_model`
    inside `if b_masks is not None:` gate, BEFORE any `pyro.sample`.
  - `task_dcm_model` signature extended with two keyword-only args
    after `*` sentinel: `b_masks: list[torch.Tensor] | None = None`,
    `stim_mod: object | None = None`. All 7 pre-existing positional
    parameters untouched. All 10 pre-Phase-15 tests (which never
    pass these kwargs) continue to work unchanged.
  - Docstring expanded with two new Parameters entries, new
    "Linear short-circuit (MODEL-04)" + "Bilinear mode (MODEL-01)" +
    "NaN-safe BOLD guard" Notes sections, and SPM12
    `spm_dcm_fmri_priors.m` pC.B reference.
  - Body changes (five structural insertions):
    1. After `N = a_mask.shape[0]`: empty-list normalization
       `if b_masks is not None and len(b_masks) == 0: b_masks = None`
       (MODEL-04 edge case — J=0 takes the linear short-circuit as
       a single code path).
    2. After `M = c_mask.shape[1]`: validation gate
       `if b_masks is not None: _validate_bilinear_args(b_masks,
       stim_mod, N)`.
    3. After `C = C * c_mask`: bilinear B-sampling block (MODEL-01).
       Per-modulator `for j, b_mask_j in enumerate(b_masks): B_free_j
       = pyro.sample(f"B_free_{j}", dist.Normal(torch.zeros_like(
       b_mask_j), B_prior_std * torch.ones_like(b_mask_j))
       .to_event(2))`. L1 locked — full `(N, N)` shape, NOT a flat
       vector; NO `pyro.plate` around the loop. After the loop:
       `torch.stack(B_free_list, dim=0)` -> `(J, N, N)`,
       `torch.stack(list(b_masks), dim=0)` -> `(J, N, N)`, single
       `parameterize_B(B_free_stacked, b_mask_stacked)` call.
       `pyro.deterministic("B", B_stacked)` emitted ONLY in this
       branch (L3 guard preserves linear trace). Merge driving +
       modulator inputs via `merge_piecewise_inputs(drive_input,
       mod_input)` — accepts either `PiecewiseConstantInput` directly
       or a breakpoint dict `{"times", "values"}` (same contract as
       Phase 14 `_normalize_stimulus_to_input_fn`).
    4. Replaced `system = CoupledDCMSystem(A, C, stimulus)` with a
       branched construction:
       - Bilinear: `CoupledDCMSystem(A, C, merged_input_fn,
         B=B_stacked, n_driving_inputs=c_mask.shape[1])` — engages
         Phase 13 gate at `coupled_system.py:292-300`.
       - Linear short-circuit: `CoupledDCMSystem(A, C, stimulus)` —
         inherits Phase 13 literal-expression gate at
         `coupled_system.py:287-291` with NO `B=` kwarg (MODEL-04,
         L3).
    5. Before `pyro.deterministic("predicted_bold", predicted_bold)`:
       NaN/Inf guard `if torch.isnan(predicted_bold).any() or
       torch.isinf(predicted_bold).any(): predicted_bold =
       torch.zeros_like(predicted_bold).detach()`. Pattern ported
       from `amortized_wrappers.py:143-145`; extended to also catch
       `isinf` (defensive). Applied in BOTH branches.
- `tests/test_task_dcm_model.py`:
  - New `task_bilinear_data` fixture extending `task_data` with J=1
    bilinear coverage. `b_masks = [b_mask_0]` where
    `b_mask_0[1, 0] = 1.0` (off-diagonal 1<-0 connection modulated,
    zero diagonal per Pitfall B5). `stim_mod` from
    `make_epoch_stimulus(event_times=[10.0], event_durations=[10.0],
    event_amplitudes=[1.0], duration=30.0, dt=0.01, n_inputs=1)`
    wrapped in `PiecewiseConstantInput` (Pitfall B12 preferred
    boxcar primitive for dt-invariance under rk4 mid-step sampling).
    Returns superset of `task_data` with added `b_masks`,
    `stim_mod`, `J` keys.
  - New `TestBilinearStructure` class (8 tests, 5.66s):
    - `test_B_PRIOR_VARIANCE_constant` — imports constant and
      asserts `== 1.0` (MODEL-02).
    - `test_linear_reduction_when_b_masks_none` — passes `b_masks=
      None, stim_mod=None`, asserts `site_names == {A_free, C,
      noise_prec, obs, A, predicted_bold}` exactly and no
      `B`/`B_free_*` sites (MODEL-04, L3).
    - `test_linear_reduction_when_b_masks_empty_list` — passes
      `b_masks=[]`, asserts same linear site structure (MODEL-04
      []-to-None normalization).
    - `test_bilinear_trace_has_B_free_sites` — asserts `B_free_0`
      exists (NOT bare `B_free`), `B_free_0` value shape `(N, N)`,
      `B` deterministic site exists with shape `(J, N, N)`
      (MODEL-01 + L1).
    - `test_bilinear_masking_applied` — iterates all `(i, k)` where
      `b_mask == 0` and asserts `B[0, i, k] == 0` exactly (MODEL-03
      model-side).
    - `test_bilinear_stim_mod_required_error` —
      `pytest.raises(ValueError, match="stim_mod is required")` when
      bilinear with `stim_mod=None`.
    - `test_bilinear_stim_mod_shape_mismatch_error` — builds
      2-column `stim_mod` with 1-element `b_masks`, asserts
      `pytest.raises(ValueError, match=r"stim_mod\.values\.shape
      \[1\]=2")`.
    - `test_bilinear_deprecation_warning_on_stacked_nonzero_diag` —
      constructs non-zero-diagonal `bad_b_mask` (`[0, 0] = 1.0`),
      wraps `pyro.poutine.trace(conditioned).get_trace(...)` in
      `pytest.warns(DeprecationWarning, match="non-zero diagonal")`.
      Closes the SC-4 stacked-path coverage gap that Phase 13
      `test_bilinear_utils.py` (N, N)-test does not exercise
      (MODEL-03 stacked half).
  - New `TestBilinearSVI` class (1 test, 29.8s):
    - `_silence_stability_logger` autouse fixture sets
      `pyro_dcm.stability` logger level to ERROR via
      `caplog.set_level` (D4 + R6 — stability monitor WARN spam
      silenced; monitor fires frequently in bilinear early-SVI but
      does NOT raise).
    - `test_bilinear_svi_smoke_3region_converges` — 40 SVI steps
      with `AutoNormal(task_dcm_model, init_scale=0.005)` (L2) +
      `ClippedAdam(lr=0.01, clip_norm=10.0)` + `Trace_ELBO`.
      Asserts every loss finite; asserts `mean(last_10) <
      mean(first_10)`. Runtime budget 75s is SOFT — warns
      (`UserWarning`) but does not fail if exceeded (Pitfall B10
      3-6x slowdown estimate). Closes MODEL-04 bilinear SVI
      convergence acceptance gate.
- Commits: `23a5591` `feat(15-01): extend task_dcm_model with
  bilinear B path`; `cd405d2` `test(15-01): bilinear structure
  tests for task_dcm_model`; `807fb46` `test(15-01): bilinear SVI
  smoke test (3-region J=1, 40 steps, L2 init_scale)`.
- Verification: `tests/test_task_dcm_model.py` 19/19 in 74.16s (10
  pre-existing + 8 TestBilinearStructure + 1 TestBilinearSVI); Phase
  13+14 regression 51/51 in 298.50s
  (`test_linear_invariance.py`, `test_coupled_system_bilinear.py`,
  `test_bilinear_utils.py`, `test_bilinear_simulator.py`,
  `test_stimulus_utils.py`). Ruff clean on all new code
  (pre-existing I001 import-sort in both files + F401 unused
  `pyro.distributions` in test file confirmed via `git stash`
  round-trip; not touched per Phase 14 precedent). Mypy not run
  (not invoked by pre-existing CI in this environment; pytest-timeout
  plugin also unavailable, so `--timeout=...` flags silently ignored).
- **Grep sentinels** (all met within acceptable ranges):
  Source file: `B_PRIOR_VARIANCE` 5 (within 3-5 acceptable —
  1 const def + 3 docstring + 1 sqrt literal), `f"B_free_{j}"` 2
  (docstring + literal sample call), `merge_piecewise_inputs` 2
  (import + call), `pyro.deterministic("B"` 3 (2 docstring + 1
  call), `CoupledDCMSystem(A, C, stimulus)` 1 (linear short-circuit),
  `CoupledDCMSystem(` 2 (linear + bilinear), `_validate_bilinear_args`
  2 (def + call), `torch.isnan(predicted_bold)` 1.
  Test file: `B_PRIOR_VARIANCE` 7, `B_free_0` 7,
  `test_linear_reduction_when_b_masks` 2,
  `pytest.raises(ValueError, match="stim_mod is required")` 1,
  `test_bilinear_deprecation_warning_on_stacked_nonzero_diag` 2
  (docstring + def), `pytest.warns(DeprecationWarning` 1,
  `init_scale=0.005` 3 (1 literal call + 2 docstring),
  `_silence_stability_logger` 1,
  `test_bilinear_svi_smoke_3region_converges` 1,
  `class TestBilinearStructure` 1, `class TestBilinearSVI` 1,
  `pyro_dcm.stability` 2.
- **L1 applied** (from plan frontmatter): `B_free_j` sampled as
  full `(N, N)` `Normal(0, 1.0).to_event(2)` — NOT a flat vector of
  free entries. Required for AutoLowRankMultivariateNormal's
  single-vector concatenation compatibility in Plan 15-02
  (`AutoContinuous._unpack_latent` reshapes by site-event-dim).
- **L2 applied** (from plan frontmatter): `init_scale=0.005` for
  the bilinear SVI smoke test (half the linear default `0.01`),
  passed **explicitly** to `AutoNormal`. `create_guide` factory
  was NOT changed (additive-discipline). Downstream Phase 16
  recovery benchmark callers will pass `init_scale=0.005`
  explicitly.
- **L3 applied** (from plan frontmatter): `pyro.deterministic("B",
  B_stacked)` emitted ONLY inside the bilinear branch (guarded
  by `if b_masks is not None:`). Linear-mode trace has NO `"B"`
  site — verified by `test_linear_reduction_when_b_masks_none`
  via exact set-equality `site_names == {A_free, C, noise_prec,
  obs, A, predicted_bold}`. Preserves the pre-Phase-15
  `test_model_trace_has_expected_sites` assertion byte-for-byte.
- **Deviations:** None — plan executed exactly as written; no
  auto-fix rules triggered. Minor sentinel variances documented
  in 15-01-SUMMARY.md (all within acceptable ranges; docstring
  references count toward literal-pattern greps):
  `B_PRIOR_VARIANCE` count 5 (plan target 4, acceptable 3-5);
  `init_scale=0.005` count 3 (plan target 1; 1 literal call + 2
  docstring references). Pre-existing ruff I001 + F401 not fixed
  (Phase 14 precedent: don't touch pre-existing lint in additive
  plans).
- Requirements closed: MODEL-01, MODEL-02, MODEL-03 (BOTH halves:
  Phase 13 `(N, N)` source-side at `test_bilinear_utils.py` +
  Plan 15-01 stacked `(J, N, N)` call-site at
  `test_bilinear_deprecation_warning_on_stacked_nonzero_diag`),
  MODEL-04. Phase 15 now 4/7 requirements closed (MODEL-05,
  MODEL-06, MODEL-07 pending in Plans 15-02 and 15-03).

---

### 2026-04-18 -- Plan 14-02 complete

- `src/pyro_dcm/simulators/task_simulator.py`:
  - New module-private `_normalize_B_list(B_list, device, dtype) -> Tensor | None`
    accepts `None | list[Tensor] | tuple[Tensor, ...] | Tensor(J,N,N)`. Empty
    list and shape `(0,N,N)` collapse to `None` (linear mode). Raises
    `TypeError` on invalid input types and `ValueError` on non-3-D tensors.
  - New module-private `_normalize_stimulus_to_input_fn(stim, device, dtype)`
    accepts a dict `{'times','values'}` or a `PiecewiseConstantInput`; returns
    a `PiecewiseConstantInput`. Factored from the pre-existing inline
    normalization at old line 143-149.
  - `simulate_task_dcm` signature extended with three keyword-only args
    after `*`: `B_list: Tensor | list[Tensor] | None = None`,
    `stimulus_mod: dict | PiecewiseConstantInput | None = None`,
    `n_driving_inputs: int | None = None`. All pre-Phase-14 positional
    callers unchanged.
  - Body branches on `B_stacked is None`. **Linear short-circuit (SIM-03
    structural gate):** `CoupledDCMSystem(A_dev, C_dev, input_fn, hemo_params)`
    is called with NO `B=` or `n_driving_inputs=` kwarg, inheriting the Phase
    13 linear gate at `coupled_system.py:287-291`. **Bilinear branch:**
    merges driving + modulatory inputs via `merge_piecewise_inputs`, validates
    `stimulus_mod.values.shape[1] == B_list.shape[0]`, applies L3 default
    `n_driving_inputs = C.shape[1]`, then calls `CoupledDCMSystem(A_dev,
    C_dev, merged_input_fn, hemo_params, B=B_stacked, n_driving_inputs=n_driv)`.
  - Return dict extended with two new keys: `'B_list'` (`None` or stacked
    `(J,N,N)` tensor) and `'stimulus_mod'` (`None` or `PiecewiseConstantInput`).
  - Import widened to include `merge_piecewise_inputs`.
  - Docstring expanded with three Parameters entries, two Returns entries, a
    "Linear short-circuit" note, and a References section citing Friston 2003.
  - Validation gates: missing `stimulus_mod` in bilinear mode raises
    `ValueError("stimulus_mod is required when B_list is non-None; ...")`;
    `stimulus_mod` column-count mismatch with `B_list.shape[0]` raises
    `ValueError`; explicit `n_driving_inputs != C.shape[1]` raises `ValueError`.
- `tests/test_task_simulator.py`:
  - Single additive edit (one line) to
    `TestSimulatorOutputStructure::test_simulator_output_keys`: `expected_keys`
    set augmented with `"B_list"` and `"stimulus_mod"`. All 17 other tests
    untouched and green.
- `tests/test_bilinear_simulator.py` (NEW): 5 tests, all green.
  - `test_bilinear_arg_none_matches_no_kwarg` (SIM-03 primary): structural
    bit-exactness via `torch.equal` on `bold_clean`, `bold`, `neural` between
    no-kwarg and explicit-None-kwarg calls.
  - `test_bilinear_output_distinguishable_from_linear` (SIM-03 secondary):
    same A/C/stim/seed, bilinear BOLD differs from linear null by
    `max|diff| > 0.01` (ample margin).
  - `test_return_dict_has_bilinear_keys` (SIM-04): linear mode has both keys
    `None`; bilinear mode has stacked `(1,3,3)` tensor and
    `isinstance(..., PiecewiseConstantInput)`.
  - `test_dt_invariance_bilinear` (SIM-05 primary): rk4 `dt=0.01` vs
    `dt=0.005` on the same bilinear ground truth (A = parameterize_A(zeros),
    B off-diagonal = 0.1 per L4, stimulus_mod at finer grid), atol=1e-4.
  - `test_dt_invariance_linear` (SIM-05 L5 regression symmetry): mirror
    without bilinear kwargs; same atol=1e-4.
- Commits: `abeb5d8` `feat(14-02): extend simulate_task_dcm with bilinear
  mode`; `88cc1bb` `test(14-02): update expected_keys for new bilinear
  return-dict keys`; `22ee2f7` `test(14-02): bilinear simulator regression
  and dt-invariance tests`.
- Verification: `tests/test_bilinear_simulator.py` 5/5 in 306.11s;
  `tests/test_task_simulator.py` 18/18 in 36.64s; full Phase 13 + Phase 14
  + `test_task_dcm_model` regression subset 108/108 in ~5 minutes
  cumulative. Ruff clean on all new code (pre-existing E501 in
  `make_random_stable_A` and in 4 other `test_task_simulator.py` methods
  confirmed via `git stash` round-trip, unchanged by this plan).
- **Grep sentinels** (all met): `def simulate_task_dcm` ×1, `B_list` ×31
  in task_simulator.py, `stimulus_mod` ×14 in task_simulator.py,
  `merge_piecewise_inputs` ×2 in task_simulator.py (import + bilinear
  branch call), `set(result.keys())` ×1 in test_task_simulator.py,
  `torch.equal` ×4 in test_bilinear_simulator.py, `atol=1e-4` ×4 in
  test_bilinear_simulator.py.
- **L3 applied (from plan frontmatter):** `n_driving_inputs` defaults to
  `C.shape[1]` at the simulator level (simulator has `C` in scope). This
  diverges from `CoupledDCMSystem.__init__`'s raise-policy, which is
  defensible because `CoupledDCMSystem` does NOT have `C` in scope. Since
  `simulate_task_dcm` always passes an inferred-or-explicit value, the
  `CoupledDCMSystem` raise branch is never triggered by the simulator path.
- **L4 applied (from plan frontmatter):** SIM-05 dt-invariance fixture uses
  B off-diagonal magnitude = 0.1 (not 0.3) to keep the Phase 13 stability
  monitor silent.
- **L5 applied (from plan frontmatter):** linear-path dt-invariance
  regression test `test_dt_invariance_linear` added as symmetry with the
  bilinear primary test; guards against silent future breakage of linear
  rk4 reproducibility.
- **Deviation (Rule 1, testbed bug):** Test fixture `C` matrix in Tests 1-3
  changed from the plan's `torch.eye(3, 1)` (unit amplitude) to
  `torch.tensor([[0.25], [0.0], [0.0]])` (task-DCM suite convention).
  Discovered during execution: unit-amplitude drive + `make_random_stable_A(seed=42)`
  causes dopri5 (the `simulate_task_dcm` default solver) adaptive step to
  underflow to 0.0 at integration start (torchdiffeq assertion
  `underflow in dt 0.0` in `rk_common.py:284`). The fix keeps the tests
  faithful to their stated purpose while using fixtures consistent with the
  existing 40+ green task-DCM tests. Tests 4 and 5 use `parameterize_A(zeros)`
  + rk4 fixed-step so they were never at risk. No algorithmic change;
  documented in 14-02-SUMMARY.md.
- Requirements closed: SIM-03, SIM-04, SIM-05. Phase 14 now 5/5.

---

### 2026-04-18 -- Plan 14-01 complete

- `src/pyro_dcm/simulators/task_simulator.py`:
  - New `make_event_stimulus(event_times, event_amplitudes, duration, dt,
    n_inputs=None, dtype=torch.float64)` implements SIM-01: variable-amplitude
    stick-function stimuli via quantized breakpoint construction. Normalizes
    scalar / 1-D / 2-D amplitudes, sorts unsorted inputs, quantizes onsets
    via `round(t/dt)*dt`, raises `ValueError` on same-grid-index collisions
    (14-RESEARCH §3 R2), emits one-shot `UserWarning` on tail truncation.
    Docstring cites Pitfall B12 and steers modulator callers to
    `make_epoch_stimulus`.
  - New `make_epoch_stimulus(event_times, event_durations, event_amplitudes,
    duration, dt, n_inputs=None, dtype=torch.float64)` implements SIM-02:
    boxcar-shaped modulatory inputs via delta-amp sweep. Quantizes on/off
    times, clips at `duration` with one-shot `UserWarning`, SUMS overlapping
    epochs + emits one-shot `UserWarning("Overlapping epochs detected; ...")`
    per L1 locked decision.
  - `import warnings` added at module top.
- `src/pyro_dcm/utils/ode_integrator.py`:
  - New `merge_piecewise_inputs(drive, mod) -> PiecewiseConstantInput` at
    end of file. Takes `torch.unique(sorted=True)` of breakpoint times,
    evaluates `drive(t_k)` and `mod(t_k)` per-breakpoint, concatenates into
    `(K, M+J)` widened values. Raises `ValueError` on dtype/device mismatch
    (no silent cast) per 14-RESEARCH §10.2 R3. Public helper per L2 locked
    decision so Phase 15's Pyro model can import from `pyro_dcm.utils`
    without crossing a simulators/ boundary.
- `src/pyro_dcm/simulators/__init__.py`: `make_event_stimulus`,
  `make_epoch_stimulus` re-exported in the Phase-1 section of `__all__`
  (alphabetized).
- `src/pyro_dcm/utils/__init__.py`: `merge_piecewise_inputs` re-exported.
- `tests/test_stimulus_utils.py` (new): 25 passing tests across three classes
  -- `TestMakeEventStimulus` (13 incl. 4 parametrize), `TestMakeEpochStimulus`
  (8 incl. 3 parametrize), `TestMergePiecewiseInputs` (4). Explicit
  `pytest.warns(UserWarning, match="Overlapping epochs")` asserts the L1
  contract on overlapping epochs. `test_values_at_breakpoints_concat_correctly`
  verifies `merged(t*) = cat(drive(t*), mod(t*))` at 6 query points covering
  before-events, inside-block, in-rest, inside-mod, post-mod, and at an
  exact breakpoint.
- Commits: 5900146 `feat(14-01): add stimulus utilities (make_event_stimulus,
  make_epoch_stimulus, merge_piecewise_inputs)`; c82a961 `test(14-01): unit
  tests for stimulus utilities`.
- Verification subsets all green: `test_stimulus_utils + test_task_simulator
  + test_ode_integrator` 59/59 in 108s; Phase 13 regression subset
  (`test_linear_invariance + test_coupled_system_bilinear + test_bilinear_utils
  + test_neural_state + test_stability_monitor`) 34/34 in 18s;
  `test_task_dcm_model` 10/10 in 40s. Full-suite collection: 454 tests
  discoverable, no import errors. Ruff clean on all modified files.
- Grep sentinels: `def make_event_stimulus` ×1, `def make_epoch_stimulus` ×1,
  `def merge_piecewise_inputs` ×1, `Pitfall B12` ×3 in docstrings,
  `Overlapping epochs` ×2 (docstring + warning), 6 overlap refs in test file.
- **Decisions applied (both locked, no new decisions made):** L1 (overlap
  sum + UserWarning); L2 (`merge_piecewise_inputs` in `utils/ode_integrator.py`).
- Requirements closed: SIM-01, SIM-02. SIM-03..05 remain for Plan 14-02.
- Pre-existing ruff E501 at `task_simulator.py:847` (in `make_random_stable_A`)
  is unrelated to this plan, verified pre-existing via `git stash` round-trip,
  and left untouched. Candidate for a future chore commit.

### 2026-04-17 -- Plan 13-01 complete

- `src/pyro_dcm/forward_models/neural_state.py`:
  - New `parameterize_B(B_free, b_mask)` implements BILIN-01: masked (J,N,N)
    factory; elementwise mult only; DeprecationWarning on non-zero b_mask
    diagonal (Pitfall B5); ValueError on shape mismatch or non-3D inputs.
  - New `compute_effective_A(A, B, u_mod)` implements BILIN-02:
    `A + einsum('j,jnm->nm', u_mod, B)`; explicit J=0 short-circuit returns
    `A` bit-exactly (no einsum call, no allocation).
  - Module docstring rewritten to label `A+Cu` as **linear form** (BILIN-07
    source half); `NeuralStateEquation` class summary line rewritten likewise.
    Existing `parameterize_A` body and `NeuralStateEquation` method bodies
    untouched.
- `src/pyro_dcm/forward_models/__init__.py`: `compute_effective_A` +
  `parameterize_B` re-exported in the Phase 1 section of `__all__`.
- `tests/test_bilinear_utils.py`: new file, 9 passing tests across shape,
  mask semantics, default-diagonal pattern, DeprecationWarning path, J=0
  roundtrip, ValueError path, einsum correctness to 1e-12 tolerance, and
  J=0 short-circuit. Existing `test_neural_state.py` (8/8) untouched and green.
- Commits: 9e7f993 `feat(13-01): add parameterize_B + compute_effective_A
  utilities`; df1f15a `feat(13-01): export parameterize_B + compute_effective_A
  from forward_models`; fcedc56 `test(13-01): add tests/test_bilinear_utils.py
  with 9 coverage tests`.
- Regression subset (`test_ode_integrator.py` + `test_task_simulator.py`)
  green: 34/34 in 194s. No coupling added to `nn.Module` or `torchdiffeq` at
  this plan -- utilities are pure tensor ops consumable by later plans.

### 2026-04-17 -- Plan 13-04 complete

- CLAUDE.md: directory-tree `generative_models/` block rewritten to actual `models/`
  layout (5 files: task_dcm_model.py, spectral_dcm_model.py, rdcm_model.py, guides.py,
  amortized_wrappers.py). task_dcm_model.py annotated `[v0.3.0: + bilinear B path]`.
- .planning/PROJECT.md line 23: `- Bilinear neural state equation (dx/dt = Ax + Cu)...`
  rewritten to `- **Linear** neural state equation (dx/dt = Ax + Cu)...`. The v0.3.0
  true-bilinear entry will be added to Validated when Phase 16 passes.
- Closes BILIN-07 non-source half. Source half (neural_state.py module + class
  docstrings) is Plan 13-01 Task 1.
- Commit: f77560d `docs(13-04): correct stale doc drift (CLAUDE.md tree + PROJECT.md
  linear vs bilinear)`. Two .md files, zero source/test edits -- clean Wave 1
  parallelism with 13-01/13-02/13-03.

### 2026-04-17 -- Plan 13-03 complete

- `src/pyro_dcm/__init__.py`: attached `NullHandler` to the `pyro_dcm` root
  logger via underscore-prefixed `_logging` alias (PEP 282, stdlib library
  logging HOWTO). Propagates to `pyro_dcm.stability` child by hierarchical
  semantics. Not added to `__all__`.
- `src/pyro_dcm/forward_models/coupled_system.py`:
  - Module-level `_STABILITY_LOGGER = logging.getLogger("pyro_dcm.stability")`.
  - `CoupledDCMSystem.__init__` gained keyword-only `B: Tensor | None = None`
    (J,N,N), `n_driving_inputs: int | None = None`, and
    `stability_check_every: int = 10`. `B` registered as buffer when
    supplied, auto-aligned to `A.device` / `A.dtype` (mitigates device-drift
    risk, 13-RESEARCH Section 10.3). `ValueError` raised when `B` non-empty
    and `n_driving_inputs is None` (explicit-split policy).
  - `CoupledDCMSystem.forward` now branches: when `self.B is None` or empty-J,
    executes the literal `dx = self.A @ x + self.C @ u_all` expression
    (grep-verified: exactly one match on line 291); when non-empty, slices
    `u_drive = u_all[:n_driving_inputs]` + `u_mod = u_all[n_driving_inputs:]`,
    composes `A_eff = compute_effective_A(A, B, u_mod)`, routes
    `dx = A_eff @ x + C @ u_drive`, then invokes `_maybe_check_stability`.
  - New `_maybe_check_stability(t, A_eff, u_mod)` method: counter-modulo
    cadence on RHS evaluations; `stability_check_every=0` disables entirely
    (zero overhead). `torch.no_grad()` + `A_eff.detach()` avoids
    complex-gradient overhead. Logs WARNING on `max Re(eig(A_eff)) > 0`
    with format `"Stability warning at t=%.2fs: max Re(eig(A_eff))=%+.3f;
    ||B·u_mod||_F=%.3f"`. Never raises (D4).
- `tests/test_coupled_system_bilinear.py` (new): 5 passing tests in
  `TestCoupledDCMSystemBilinear` — `torch.equal` bit-exact no-kwarg-vs-B=None,
  buffer vs parameter assertion, `float32 -> float64` dtype auto-alignment,
  `pytest.raises(ValueError, match="n_driving_inputs")` on missing
  n_driving_inputs, and BOLD RMS > 1e-6 sanity check for the bilinear path.
- `tests/test_stability_monitor.py` (new): 5 passing tests across two
  classes. `TestStabilityMonitor` (BILIN-05, 4 tests): unstable fires
  WARNING, stable silent, `stability_check_every=0` disables, monitor
  never raises. `TestThreeSigmaWorstCase` (BILIN-06, 1 test): deterministic
  `N=3` fixture with `parameterize_A(zeros)` baseline, `B: (1,3,3)`
  off-diagonal `3.0` diagonal `0`, `C = zeros(3,1)`, `u_drive=0` +
  `u_mod=1` sustained, rk4 `dt=0.1` for 500 s — `torch.isfinite(sol).all()`.
- Commits: `3e2ffa9` `feat(13-03): add pyro_dcm.stability logger
  NullHandler`; `956e1de` `feat(13-03): extend CoupledDCMSystem with
  bilinear path + stability monitor`; `5988dbd` `test(13-03): add
  test_coupled_system_bilinear.py`; `ae9a265` `test(13-03): add
  test_stability_monitor.py with BILIN-06 3-sigma 500s test`.
- Verification: Phase 13 full suite + neural_state 34/34 green in 30.92s.
  BILIN-04 acceptance gate: `test_ode_integrator.py` +
  `test_task_simulator.py` + `test_task_dcm_model.py` 44/44 green in
  245.43s. Grep sentinels: `dx = self.A @ x + self.C @ u_all` exactly once
  (line 291), `_STABILITY_LOGGER = logging.getLogger` exactly once (line
  50), `NullHandler` twice in `__init__.py` (comment + attachment call,
  both expected). `pyro_dcm.stability` logger resolves to WARNING (level 30).
- BILIN-04 / BILIN-05 / BILIN-06 all closed. Phase 13 requirement coverage
  7/7 (BILIN-01 through BILIN-07). No deviations from plan.

### 2026-04-17 -- Plan 13-02 complete

- `src/pyro_dcm/forward_models/neural_state.py`:
  - `NeuralStateEquation.derivatives` signature extended to
    `(self, x, u, *, B=None, u_mod=None)`. Linear short-circuit guard at the
    top of the method body executes the literal expression
    `return self.A @ x + self.C @ u` when `B is None` or `B.shape[0] == 0`
    (grep-verified: the literal appears exactly once in the file). Bilinear
    branch routes through `compute_effective_A(self.A, B, u_mod)` and returns
    `A_eff @ x + self.C @ u`. `ValueError` raised when `B` is non-empty and
    `u_mod is None`. Class summary line from 13-01 + module docstring
    untouched.
- `tests/test_linear_invariance.py` (new): 7 passing tests across
  `TestLinearInvariance` (5) and `TestBilinearPathSanity` (2). Primary
  fixtures (`rtol=0, atol=1e-10`): hand-crafted 2-region; `make_random_stable_A(N=3,
  seed=42)`; `make_random_stable_A(N=5, seed=7)`; empty-J `(0, N, N)`. Strict
  `torch.equal` case: no-kwarg vs `B=None`. Bilinear sanity: hand-computed
  output + `ValueError` on missing `u_mod`.
- Commits: 55785de `feat(13-02): extend NeuralStateEquation.derivatives with
  bilinear path`; 7289ff9 `test(13-02): add test_linear_invariance.py with
  atol=1e-10 gate`.
- Verification: Phase 13 test suite 24/24 green in 3.89s; downstream
  regression 44/44 green in 221.68s. BILIN-07 non-regression grep confirms
  the misleading-label pattern is absent from neural_state.py.
- BILIN-03 acceptance criterion (bit-exact linear invariance) is now locked
  structurally (literal short-circuit) AND empirically (atol=1e-10 fixtures).

---
*Last updated: 2026-04-18 after Phase 15 verification passed (14/14 must-haves; MODEL-01..07 Complete; orchestrator gap-closure commit 75343a8 decouples test_amortized_wrapper_linear_mode_unchanged from forward-model RNG via monkeypatch sentinel)*

---

### 2026-04-18 -- Phase 15 verification passed

- Verifier (gsd-verifier, sonnet) initially reported `gaps_found` (13/14) on
  `test_amortized_wrapper_linear_mode_unchanged` failing in full-session pytest
  with NaN-scale ValueError.
- Verifier diagnosis ("missing `pyro.clear_param_store()` before Case 2") was
  incorrect: line 520 already had the call.
- True root cause: test ran `pyro.poutine.trace` on the full amortized forward
  model, which samples `_latent` from prior and runs the ODE. Global RNG state
  accumulated across the pytest session caused some draws to produce ODE
  divergence -> NaN predicted_bold -> NaN scale in `dist.Normal(predicted_bold,
  noise_std).to_event(2)` -> ValueError. Test's try/except caught only
  `NotImplementedError`, so `ValueError` propagated uncaught. Source code was
  fully correct; test was over-scoped for its MODEL-07 refusal purpose.
- Orchestrator auto-fix (Rule 1): refactored test to monkeypatch
  `_run_task_forward_model` with a `_Sentinel(RuntimeError)` raise. If guard
  wrongly rejects linear mode we see `NotImplementedError`; if guard allows
  linear through we see the sentinel. This is exactly what MODEL-07 requires.
  Test only (zero source changes). Decoupled from forward-model numerics /
  RNG state / standardization quirks.
- Commit: `75343a8` `fix(15-03): decouple amortized linear-mode regression
  test from forward-model RNG` (1 file, +38/-56).
- Re-verification: `pytest tests/test_amortized_task_dcm.py` 8/8 green in
  80.86s after fix. Full Phase-15 suite
  (test_task_dcm_model + test_guide_factory + test_amortized_task_dcm +
  test_parameter_packing + test_posterior_extraction) 82/82 green in 207.75s.
- VERIFICATION.md frontmatter updated: `status: passed`, `score: 14/14`, with
  gap_closures section documenting the misdiagnosis correction + fix commit
  + re-verification run.
- Phase 15 outcomes:
  - Total new commits on branch: 12
    (23a5591, cd405d2, 807fb46, 86cdb76, 9b796c0, 680e3f7, 49cc81b, 6c68b10,
    66cab62, b9928c2, 37755b2, 75343a8)
  - All 7 MODEL requirements closed (MODEL-01..07); REQUIREMENTS.md traceability
    table updated.
  - 27 v0.3.0 requirements: 19/27 Complete (BILIN-01..07 Phase 13, SIM-01..05
    Phase 14, MODEL-01..07 Phase 15); 8/27 Pending (RECOV-01..08 Phase 16).
  - Branch `gsd/phase-15-pyro-bilinear-model` ready for merge to main at user
    discretion; Phase 16 benchmark plan-phase workflow next.
