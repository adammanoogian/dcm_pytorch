# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.3.0 Bilinear DCM Extension -- roadmap defined, ready to plan Phase 13

## Current Position

**Milestone:** v0.3.0 Bilinear DCM Extension (started 2026-04-17)
**Phase:** Phase 13 -- Bilinear Neural State & Stability Monitor (in progress)
**Plan:** 13-01 + 13-02 + 13-04 complete; 13-03 still pending
**Status:** Plans 13-01, 13-02, 13-04 shipped. BILIN-01 (parameterize_B), BILIN-02 (compute_effective_A), and BILIN-03 (NeuralStateEquation bilinear extension + bit-exact linear-invariance test) live in `forward_models/neural_state.py`. BILIN-07 fully closed (source half via 13-01, non-source via 13-04). Phase 13 test totals: `test_neural_state.py` (8) + `test_bilinear_utils.py` (9) + `test_linear_invariance.py` (7) = 24 passing. Downstream regression sweep `test_ode_integrator.py`/`test_task_simulator.py`/`test_task_dcm_model.py` 44/44 green.
**Last activity:** 2026-04-17 -- 13-02-PLAN.md executed (2 files modified, 2 task commits + 1 metadata commit)

Progress: v0.1.0 [██████████] 100% | v0.2.0 [██████████] 100% | v0.3.0 [░░░░░░░░░░] 0/4 phases (Phase 13: 3/4 plans complete -- 13-01, 13-02, 13-04)

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

Last session: 2026-04-17
Stopped at: Plan 13-02 complete. `NeuralStateEquation.derivatives` now accepts
keyword-only `B` / `u_mod`; linear short-circuit is bit-exact (atol=1e-10 gate
at `tests/test_linear_invariance.py`). SUMMARY at
`.planning/phases/13-bilinear-neural-state/13-02-SUMMARY.md`.
Next: Plan 13-03 (stability monitor inside `CoupledDCMSystem` with
`stability_check_every` cadence + `pyro_dcm.stability` logger + BOLD-level
linear-invariance regression).
Resume file: None

---

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
*Last updated: 2026-04-17 after plan 13-02 completion (BILIN-03)*
