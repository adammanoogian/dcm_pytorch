# v0.3.0 Bilinear DCM -- Research Summary

**Project:** pyro_dcm
**Milestone:** v0.3.0 Bilinear DCM Extension
**Researched:** 2026-04-17
**Overall confidence:** MEDIUM-HIGH (HIGH on SPM mechanics and codebase; MEDIUM on numeric thresholds and DCM.V3 budget)

Sources synthesized:
- `.planning/research/v0.3.0/STACK.md`
- `.planning/research/v0.3.0/FEATURES.md`
- `.planning/research/v0.3.0/ARCHITECTURE.md`
- `.planning/research/v0.3.0/PITFALLS.md`

---

## Executive Summary

v0.3.0 extends an already-shipping linear DCM (`dx/dt = Ax + Cu`) to the Friston-2003 bilinear form
(`dx/dt = Ax + Σ_j u_j(t)·B_j·x + Cu`). All four researchers converge on the same structural
conclusion: **this is a narrow, well-bounded mathematical extension of the task-DCM chain**, not an
infrastructure project. No new runtime dependencies are needed; spectral DCM and rDCM are
architecturally untouched; existing `PiecewiseConstantInput`, Pyro `AutoGuide` auto-discovery, and
`torchdiffeq` time-varying RHS semantics all accommodate the new term with zero API churn. The entire
extension lands as optional None-default kwargs on five existing functions plus one new benchmark runner.

The scope YAML (DCM.1-4 core + DCM.V1 benchmark; DCM.5 PEB-lite, DCM.V2 SPM cross-val, DCM.V3 4-node
circuit DEFERRED) is broadly consistent with researcher findings, but contains **one verified factual
error and several unresolved specification choices** that must be decided before REQUIREMENTS.md. The
headline error: the YAML claims `B_free ~ N(0, 1/16)` is "SPM12 convention." Both Features and
Pitfalls researchers verified against SPM12 `spm_dcm_fmri_priors.m` that one-state SPM uses variance
**1.0** (two-state uses 1/4). The user must decide whether to match SPM (variance 1.0 -- needed for
future DCM.V2 cross-validation) or keep a tighter pyro_dcm-native prior (variance 1/16 or 0.25).

Main risks are well-characterized: bilinear `A_eff(t)` can lose stability under sampled B tails (no
exponential wrapper in one-state SPM; Pyro SVI differentiates through the full nonlinear ODE), B
parameters are non-identifiable under sparse/short-epoch designs (Rowe 2015), amortized-guide shape
contracts must be version-tagged (not warm-started from v0.2.0), and per-step ODE cost grows 3-6x at
moderate J/N. All are addressable with the stability monitor, shrinkage metrics, and explicit packer
versioning described below. The critical path is: DCM.1 neural state -> coupled system -> simulator
-> Pyro model -> DCM.V1 benchmark.

---

## Key Findings

### Recommended Stack (HIGH confidence)

**No new runtime dependencies.** Existing stack (PyTorch 2.9, Pyro 1.9.1, torchdiffeq 0.2.5, scipy,
matplotlib, pytest, ruff, mypy) covers every bilinear-DCM capability.

- **torchdiffeq 0.2.5** -- `odeint(func, y0, t_eval)` already calls `func(t, y)` per step; time-varying
  `A_eff(t)` is native. No version bump.
- **Pyro AutoGuides** -- `_setup_prototype` auto-discovers new `pyro.sample("B_free", ...)` sites on
  first trace. Zero `create_guide` / registry changes.
- **`torch.linalg.eigvals`** -- already in PyTorch core (used in `make_random_stable_A`); sufficient
  for A_eff stability monitor.
- **`PiecewiseConstantInput`** -- generalizes to any M via `values[idx]`; no shape assumption. Widening
  stimulus table from `(K, M_drive)` to `(K, M_drive + J_mod)` is the minimum-churn path.

**Explicitly rejected:** scipy.interpolate (breaks autograd), diffrax/torchode migration (months of
validation for zero new capability), Bambi/PyMC for PEB (wrong abstraction), `pyro.plate` around B_j
sampling (breaks some AutoGuides; use `.to_event(3)` or per-site loop instead).

### Expected Features (HIGH confidence on SPM mechanics)

**Table stakes (MUST ship in v0.3.0):**
- **TS-1** -- Bilinear neural state equation (DCM.1): `dx/dt = Ax + Σ u_j B_j x + Cu`.
- **TS-2** -- B-matrix priors `Normal(0, σ)` masked per modulator (DCM.2). **σ value is open Q1.**
- **TS-3** -- Per-modulator `b_masks[j]` list of (N,N) binary masks (DCM.2), parallel to existing
  `a_mask`/`c_mask`.
- **TS-4** -- `make_event_stimulus` + `make_epoch_stimulus` variable-amplitude utilities (DCM.3). HRF
  convolution deliberately NOT included (would double-count hemodynamics).
- **TS-5** -- Bilinear simulator extension to `simulate_task_dcm` (DCM.4).
- **TS-6** -- Eigenvalue stability monitor (soft warn when `max Re(eig(A_eff)) > 0`).
- **TS-7** -- 3-region recovery benchmark (DCM.V1), >=10 seeds, SNR=3, reuses v0.2.0 fixture infra.

**Differentiators (SHOULD ship if time):**
- **D-1** -- Per-modulator B posterior heatmap + 90% CI figures.
- **D-2** -- B-sparsity diagnostic (fraction of 90% CIs excluding 0).
- **D-3** -- Linear-vs-bilinear ELBO model comparison helper.

**Deferred / anti-features:** D-4 amplitude sensitivity sweep, D-5 two-state prior flag, DCM.5
PEB-lite, DCM.V2 SPM cross-val, DCM.V3 4-node circuit, nonlinear-DCM, time-varying-A(t), real-time.

### Architecture Approach (HIGH confidence)

**Extend in place, do not fork.** The bilinear math is a strict superset of linear (`B_list=None`
reduces exactly to `Ax + Cu`). All five modified files use None-default optional kwargs; zero existing
tests need edits.

**Blast radius is narrow:**
- **Modified in place:** `neural_state.py`, `coupled_system.py`, `task_dcm_model.py`, `task_simulator.py`,
  optionally `amortized_wrappers.py` (may defer -- see Open Q5).
- **Added:** `benchmarks/runners/task_bilinear.py` + 5 new test files.
- **Untouched:** Balloon model, BOLD signal, all spectral DCM, all rDCM, guide factory, SVI runner,
  model comparison.

**Build order (dependency-forced):** DCM.1 neural state -> CoupledDCMSystem extension -> DCM.4
simulator (needed for ground truth) -> DCM.3 stimulus utilities (fold into DCM.4) -> DCM.2 Pyro model
-> DCM.V1 benchmark runner -> (optional) amortized wrappers.

### Critical Pitfalls (top 5 of 14 bilinear-specific)

1. **B1 -- A_eff(t) stability loss (CRITICAL, DCM.1).** One-state SPM has no `exp` wrapper on the
   bilinear sum; sampled B tails + sustained u can push leading eigenvalue real part positive,
   producing NaN or biased ODE. Mitigate: stability monitor, worst-case-B unit test,
   stability-aware guide init (zeros for B_free).
2. **B2 -- B non-identifiability under sparse-event designs (CRITICAL, DCM.V1).** Rowe 2015 directly
   warns against HEART2ADAPT-style regimes (192 events, ~12.5s ISI). Mitigate: require posterior
   shrinkage metric (`std_post / std_prior <= 0.7`), document min-effective-events-per-B-parameter
   rule (~20).
3. **B3 -- Amortized guide shape failures (CRITICAL, DCM.2 or deferred).** Current `TaskDCMPacker`
   hardcodes `[A_free, C, noise_prec]`. Mitigate: new `TaskBilinearDCMPacker`, version-tag
   checkpoints, refuse warm-start from v0.2.0 weights.
4. **B5 -- B diagonal breaks stability (HIGH, DCM.2).** Positive sampled `B_jii` + u -> positive
   self-coupling. Mitigate: default `b_mask` diagonal to 0 with warning on override.
5. **B8 -- YAML B-prior variance is factually wrong (HIGH, DCM.2).** YAML says 1/16 "SPM12 convention";
   SPM one-state uses variance **1.0**, two-state 1/4. Mitigate: explicit decision (Open Q1);
   correct YAML/docs; unit-test the documented value.

Other notable: **B4** stale "bilinear" docstrings on current linear code (rename in DCM.1); **B6**
over-permissive b_masks overfit; **B7** sign-recovery metric meaningless when true B~=0; **B10** per-step
ODE cost 3-6x linear; **B12** `PiecewiseConstantInput` + rk4 mid-steps blur stick functions (prefer
boxcars); **B13** A RMSE degrades ~10-30% even when B_true=0 (relax acceptance to
`<= 1.25 x linear baseline`).

---

## Conflicts & Corrections

### YAML vs research -- verified errors

1. **`B_free ~ N(0, 1/16)` is NOT "SPM12 convention".** Verified against `spm_dcm_fmri_priors.m`:
   one-state `pC.B = B` -> **variance 1.0**; two-state -> variance 1/4. Zeidman et al. 2019 Table 3
   (PMC6711459) independently confirms. Likely YAML origin: confusion with PEB empirical-Bayes
   shrinkage priors. **Action:** correct YAML; user decides (Open Q1).

2. **YAML "A RMSE <= 0.15 (same as linear)" is too strict.** Per B13, bilinear DCM with `B_true=0`
   still inflates A posterior uncertainty ~10-30% (Bayesian parameter pricing). **Replacement:**
   `A RMSE <= 1.25 x linear-baseline`, OR compare against bilinear-with-`b_mask=0` null.

3. **YAML "B sign recovery >= 80%" has no literature backing.** Plausible but not paper-cited as a
   threshold. Ill-defined when `B_true ~= 0`. **Replacement:** split into
   `sign_recovery_nonzero >= 80%` (on `|B_true| > 0.1`) + `coverage_of_zero_null >= 85%`
   (on `|B_true| < 0.5·prior_std`).

### Researcher vs researcher -- minor

1. **B-prior site shape: `.to_event(3)` vs per-site loop.** STACK prefers `.to_event(3)` (simpler,
   contiguous latent). ARCHITECTURE prefers `for j: pyro.sample(f"B_free_{j}", ...)` (per-modulator
   site identity for model comparison; precedent in `rdcm_model.py:101-151`). **Resolution:** per-site
   loop -- aligns with existing pattern and preserves flexibility when `b_masks[j]` differ.

2. **`LinearInterpolatedInput` build-or-skip.** Depends on Open Q2 interpretation of "variable-amplitude."

3. **Amortized wrapper scope.** Depends on Open Q5.

4. **Eigenvalue threshold: strict vs preemptive.** Depends on Open Q4. Pitfalls' stance (log-warn
   on strict `max Re > 0`, never raise during SVI) is the safer default.

---

## Open Questions for User

The following decisions block REQUIREMENTS.md:

1. **B-prior variance** -- three credible options:
   - (a) SPM12 one-state match `variance=1.0` -- required for DCM.V2 cross-val; harder SVI.
   - (b) YAML `variance=1/16` -- tractability-friendly; must be documented as "pyro_dcm native, NOT SPM."
   - (c) Pyro_dcm native `variance=0.25` (= SPM two-state) -- compromise, citable.
   - **Recommendation:** (a) with DCM.V1 reporting both (a) and (b) for calibration.

2. **"Variable-amplitude" semantics in DCM.3:**
   - (a) Per-event piecewise-constant amplitudes -- no new primitive.
   - (b) Smooth-ramp (e.g., HGF belief updates) -- build `LinearInterpolatedInput` (~40 lines).
   - **Recommendation:** (a) for DCM.V1; (b) in v0.3.1 if continuous modulators needed.

3. **B sign-recovery threshold provenance:**
   - (a) Keep 80% documented as "pyro_dcm internal target."
   - (b) Calibrate from pilot DCM.V1 run.
   - (c) Split by magnitude per Pitfall B7.
   - **Recommendation:** (c) -- unambiguous, standard Bayesian practice.

4. **Eigenvalue-monitor threshold:** `max Re > 0` (strict) vs `> -0.05` (preemptive)?
   - **Recommendation:** Strict (`> 0`), log-warn only.

5. **Amortized-guide bilinear support in v0.3.0:**
   - (a) Include -- adds ~2 weeks, needs TaskBilinearDCMPacker + checkpoint versioning.
   - (b) Defer to v0.3.1 -- amortized wrappers stay linear-only.
   - (c) Defer to v0.4 -- bundle with PEB-lite / SPM cross-val.
   - **Recommendation:** (b). v0.3.0 acceptance (DCM.V1) doesn't need amortized; isolates risk.

---

## Implications for Roadmap

### Suggested Phase Structure

**Phase DCM.1 -- Bilinear Neural State + Stability Monitor.** Foundation; unblocks 4 downstream
modules. Delivers `compute_effective_A`, `parameterize_B`, `BilinearNeuralStateEquation` branch
(via None-default args), `CoupledDCMSystem` extension, eigenvalue monitor, docstring rename.
Addresses TS-1, TS-6. Avoids B1, B4, B9.

**Phase DCM.3+4 -- Stimulus Utilities + Bilinear Simulator.** Produces ground truth for model tests.
Delivers `make_event_stimulus`, `make_epoch_stimulus`, `simulate_task_dcm(..., B_list, stimulus_mod)`.
Addresses TS-4, TS-5. Avoids B12.

**Phase DCM.2 -- Pyro Generative Model with B Priors and Masks.** Requires Open Q1 resolved before
SVI runs. Delivers `task_dcm_model(..., b_masks, stim_mod)`, per-modulator sampling, `parameterize_B`,
packer versioning stub. Addresses TS-2, TS-3. Avoids B3, B5, B8, B14.

**Phase DCM.V1 -- 3-Region Recovery Benchmark.** Integration test; milestone acceptance. Delivers
`benchmarks/runners/task_bilinear.py`, split sign-recovery / coverage-of-zero / A-RMSE-relative
metrics. Addresses TS-7, optionally D-1, D-2, D-3. Avoids B2, B6, B7, B10, B11, B13.

**Deferred (not v0.3.0):** DCM.5 PEB-lite, DCM.V2 SPM cross-val, DCM.V3 HEART2ADAPT 4-node,
amortized-guide bilinear support (recommend v0.3.1).

### Critical Path

```
DCM.1 (neural state + stability)
    |
    v
DCM.3+4 (simulator)   <---- Open Q2 (stimulus semantics)
    |
    v
DCM.2 (Pyro model)    <---- Open Q1 (B-prior variance) BLOCKING
    |
    v
DCM.V1 (benchmark)    <---- Open Q3 (acceptance metrics)
```

### Research Flags

- **DCM.2** -- prior-variance choice has cross-val implications; MEDIUM confidence until Open Q1
  resolved.
- **DCM.V1** -- acceptance thresholds design-dependent; may need pilot calibration. MEDIUM confidence.
- **DCM.1, DCM.3, DCM.4** -- standard patterns; skip research-phase. HIGH confidence.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct source reads confirm all claims. No new deps. |
| Features | MEDIUM-HIGH | SPM mechanisms HIGH (verified); thresholds MEDIUM (plausible, not paper-cited). |
| Architecture | HIGH | File-by-file plan grounded in direct reads. Minor MEDIUM on buffer-adjoint compat. |
| Pitfalls | HIGH on mechanics; MEDIUM on thresholds and DCM.V3 budget | B1/B5/B8 verified against SPM source. |

**Overall:** MEDIUM-HIGH. Outstanding uncertainty concentrates in the 5 open questions
(user-resolvable) and in DCM.V3 compute predictions (deferred).

### Gaps to Address

- **Open Q1 (B-prior variance)** -- BLOCKING for DCM.2.
- **Open Q2 (variable-amplitude)** -- affects DCM.3 scope.
- **Open Q3 (sign-recovery threshold)** -- affects DCM.V1 acceptance.
- **Open Q4 (eigenvalue threshold)** -- low-stakes, decide in DCM.1 commit.
- **Open Q5 (amortized scope)** -- affects total milestone size.
- **Buffer-adjoint compatibility for stacked B** -- verify at DCM.2a implementation.

---

## Sources

### Primary (HIGH)
- SPM12 source (main, fetched 2026-04-17): `spm_fx_fmri.m`, `spm_dcm_fmri_priors.m`.
- Friston, Harrison & Penny (2003) NeuroImage -- DCM origin, Eq. 1 bilinear form (REF-001).
- Zeidman et al. (2019) NeuroImage PMC6711459 -- Table 3, BE prior variance = 1.
- Rowe et al. (2015) Frontiers Neurosci PMC4335185 -- identifiability under long-TR / sparse events.
- Direct reads of all affected source files and tests.
- torchdiffeq 0.2.5 docs (odeint RHS semantics).
- Pyro AutoGuide source (`_setup_prototype`).

### Secondary (MEDIUM)
- Baldy et al. (2025) J R Soc Interface -- ADVI under-dispersion.
- Razi et al. (2017) Network Neuroscience -- large-scale DCM.
- Frassle et al. (2017) NeuroImage -- B RMSE benchmarks.
- Penny et al. (2004), Friston et al. (2016), Stephan et al. (2010) -- model comparison / BMR.
- Nozari et al. (2024) -- bilinear-suffices justification.

### Tertiary (LOW / refuted)
- YAML claim `variance=1/16` is "SPM12 convention" -- refuted (see Conflicts §1).
- YAML threshold "B sign recovery >= 80%" -- not literature-cited (see Open Q3).

---

*Research completed: 2026-04-17*
*Ready for requirements: Pending user resolution of Open Questions 1-5.*
