# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.3.0 Bilinear DCM Extension -- Phase 13 complete and verified; ready for Phase 14

## Current Position

**Milestone:** v0.3.0 Bilinear DCM Extension (started 2026-04-17)
**Phase:** Phase 14 -- Stimulus Utilities & Bilinear Simulator (IN PROGRESS, 1/2 plans complete)
**Plan:** Plan 14-01 complete (stimulus utilities); Plan 14-02 pending (bilinear `simulate_task_dcm` extension)
**Status:** Plan 14-01 shipped: `make_event_stimulus`, `make_epoch_stimulus`, `merge_piecewise_inputs` + 25-test file. SIM-01 and SIM-02 closed. Next: Plan 14-02 (SIM-03/04/05). Branch: `gsd/phase-14-stimulus-and-bilinear-simulator` carries 2 Phase-14 commits (5900146 feat, c82a961 test).
**Last activity:** 2026-04-18 -- Plan 14-01 complete; SUMMARY at `.planning/phases/14-stimulus-utilities-and-bilinear-simulator/14-01-SUMMARY.md`.

Progress: v0.1.0 [██████████] 100% | v0.2.0 [██████████] 100% | v0.3.0 [███▏░░░░░░] Phase 13 complete + 1/2 Phase 14 plans (2/4 phases in flight; 15-16 pending)

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

See `.planning/milestones/v0.2.0-ROADMAP.md` and `.planning/milestones/v0.1.0-ROADMAP.md` for prior milestones.

## Blockers

None currently.

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

## Session Continuity

Last session: 2026-04-18
Stopped at: Plan 14-01 complete. Stimulus utilities (`make_event_stimulus`,
`make_epoch_stimulus`, `merge_piecewise_inputs`) shipped with 25 new tests
in `tests/test_stimulus_utils.py` (all green). SIM-01 and SIM-02 closed;
SIM-03..05 remain for Plan 14-02. Branch
`gsd/phase-14-stimulus-and-bilinear-simulator` carries 2 Phase-14 commits
(5900146 `feat(14-01): add stimulus utilities`; c82a961
`test(14-01): unit tests for stimulus utilities`) on top of Phase 13 tip.
Next: Plan 14-02 (bilinear `simulate_task_dcm` extension).
Resume file: None

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
*Last updated: 2026-04-18 after Plan 14-01 complete (SIM-01, SIM-02 closed; 25 new tests green)*
