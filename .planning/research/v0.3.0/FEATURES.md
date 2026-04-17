# Feature Landscape: Bilinear DCM Extension (v0.3.0)

**Domain:** Dynamic Causal Modeling — bilinear (B-matrix) extension to an existing
linear DCM codebase (pyro_dcm v0.2.0 shipped)
**Researched:** 2026-04-17
**Milestone:** v0.3.0 (branch: `gsd/phase-NN-bilinear-dcm`)
**Overall confidence:** MEDIUM-HIGH

---

## Summary

v0.3.0 is a **focused, scoped extension**: extend an already-shipping linear DCM
toolbox with the standard Friston-2003 bilinear term (`Σ_j u_j B_j x`). The
generative model, inference engine, simulator, and benchmark protocol already
exist for the linear case — v0.3.0 threads B-matrices through all five of them.

**Scope boundary (per `PROJECT.md` and YAML):**
- IN: DCM.1 (bilinear forward model), DCM.2 (Pyro model + B priors),
  DCM.3 (event/epoch stimulus utils), DCM.4 (bilinear simulator),
  DCM.V1 (bilinear recovery benchmark).
- OUT: DCM.5 PEB-lite, DCM.V2 SPM12 cross-val (needs MATLAB),
  DCM.V3 HEART2ADAPT 4-node benchmark.

**Critical prior finding.** The source YAML claims `B_free ~ N(0, 1/16)`. This
is **incorrect for SPM12**. Verified against `spm_dcm_fmri_priors.m` (one-state,
line ~66): `pC.B = B` (variance **1.0**, not 1/16) and `pE.B = B*0` (mean 0),
where `B` is the binary mask. The two-state model uses `pC.B = B/4` (variance
1/4). This finding is further confirmed by the Zeidman et al. group-DCM guide
(PMC6711459, Table 3): "BE prior variance = 1, 90% CI [-1.65, 1.65]". REQUIREMENTS.md
must correct the YAML claim — see Pitfalls doc for details.

**Key categorical decisions (evidence-based):**
1. Priors: Use SPM12 one-state convention `B ~ N(0, 1.0)` at active mask entries
   — match the reference implementation we're being compared against.
2. B diagonal: Allow but don't require. User-specifiable via `b_mask_j` (same
   binary-mask convention as `a_mask`). Common papers mask it to zero (e.g.,
   between-region-only modulators), but SPM permits diagonal B for
   self-modulation.
3. Stimulus primitives: Stick-functions + boxcars are the standard SPM pair.
   HRF convolution is out of scope — BOLD observation comes from the Balloon
   model downstream, not pre-convolved inputs.
4. Sign recovery ≥80% at moderate SNR is a reasonable target; literature
   reports B estimates are recoverable at SNR ≥ 3 but shrink toward zero at
   SNR ≤ 2 (Daunizeau/VBA simulations, Frässle 2017).

---

## Table Stakes

Features users expect from a credible bilinear DCM implementation. Missing any
of these makes the toolbox non-credible relative to SPM12.

### TS-1: Bilinear Neural State Equation (DCM.1)

| Aspect | Detail |
|--------|--------|
| Feature | `dx/dt = Ax + Σ_j u_j(t)·B_j·x + C·u_driving(t)` |
| Why expected | It IS bilinear DCM. The linear case is a degenerate bilinear model with `B_list = None`. |
| Complexity | Small |
| Confidence | HIGH |
| Dependencies | Existing `NeuralStateEquation`, `CoupledDCMSystem`, `parameterize_A` |
| Reference | REF-001 Eq. 1 (Friston, Harrison & Penny 2003) |

**Specification:**
- New utility: `compute_effective_A(A, B_list, u_mod_t) -> A_eff` returning
  `A + Σ_j u_j·B_j` at a given time.
- `CoupledDCMSystem.__init__` gains optional `B_list` (list of N×N tensors)
  and optional `stim_mod_fn` (modulatory-input interpolator).
- `CoupledDCMSystem.forward` evaluates `A_eff(t)` per ODE step, not just once.
- Backward compatibility: `B_list=None` → falls back to linear path (no branch
  cost in the common case).

**Dependency on v0.1.0/v0.2.0:** Extends existing `NeuralStateEquation` and
`CoupledDCMSystem` (both in `forward_models/`). `parameterize_A` stays
unchanged — the diagonal self-inhibition transform applies only to A, never B.

---

### TS-2: B-Matrix Priors Matching SPM12 (DCM.2)

| Aspect | Detail |
|--------|--------|
| Feature | `pyro.sample("B_free", Normal(0, 1.0))` masked by `b_mask_j`, per modulator |
| Why expected | Must match SPM12's reference prior to be cross-validatable. The reference implementation is what reviewers compare against. |
| Complexity | Small |
| Confidence | HIGH (verified against `spm_dcm_fmri_priors.m` and Zeidman 2019 Table 3) |
| Dependencies | Existing `task_dcm_model` Pyro model, `a_mask`/`c_mask` convention |

**Specification (SPM12-aligned):**
- Prior mean: 0 (like A).
- Prior std: **1.0** (variance 1.0, **not** 1/16 as the YAML claimed).
- Per-modulator masking: `B_free_j = B_free_sample_j * b_mask_j`.
- **No** special transform for the B diagonal (unlike A, which has
  `a_ii = -exp(A_free_ii)/2`). B enters additively; negative `A` already
  ensures baseline stability.
- Model signature:
  `task_dcm_model(bold, stim_driving, stim_mod, a_mask, b_masks, c_mask, t_eval, TR, dt)`
  where `b_masks: list[Tensor]` has J entries of shape (N, N).

**Dependency:** Builds directly on `models/task_dcm_model.py` which already has
A_free/C sampling patterns. Guide handling is automatic for `AutoNormal` /
`AutoLowRankMVN` / etc. because Pyro auto-guides pick up new `pyro.sample`
sites. `extract_posterior_params` needs a small update to return B posterior
medians.

---

### TS-3: Per-Modulator B-Masks (DCM.2)

| Aspect | Detail |
|--------|--------|
| Feature | List of J binary masks `b_masks[j]` — shape (N, N) each, specifying which connections modulator j can affect |
| Why expected | Users encode hypotheses via masks. Same mental model as `a_mask` and `c_mask` already in the codebase. |
| Complexity | Small |
| Confidence | HIGH |
| Dependencies | TS-2 |

**Specification:**
- Semantics: `b_masks[j][i, k] = 1` means "modulator j can change the
  i←k connection"; 0 means "fix this element at 0 (not a free parameter)."
- Typical convention: zero diagonals (between-region-only modulators), but
  diagonal entries are **allowed** and interpreted as "modulator changes
  region i's self-connection."
- Broadcast semantics: Element-wise multiplication after sampling, matching
  the existing `a_mask` pattern (`A_free = A_free * a_mask`).

**Dependency:** Parallel structure to `a_mask` at `task_dcm_model.py:131`.

---

### TS-4: Variable-Amplitude Event Stimulus Utility (DCM.3)

| Aspect | Detail |
|--------|--------|
| Feature | `make_event_stimulus(event_times, event_amplitudes, duration, dt) -> (T, J)` |
| Why expected | Modulatory inputs in task DCM are usually parametric: event-related with varying amplitude per trial (e.g., RT, prediction error, trial-wise HGF output). Without this, users cannot encode realistic hypotheses. |
| Complexity | Small |
| Confidence | HIGH |
| Dependencies | None (new standalone utility) |

**Specification:**
- Stick-function convention (SPM standard): single-timestep impulse at each
  `event_times[k]`, scaled by `event_amplitudes[k]`.
- Continuous time discretized at `dt`: output is `(T, J)` tensor where
  `T = ceil(duration/dt)`.
- Companion: `make_epoch_stimulus(event_times, event_durations,
  event_amplitudes, duration, dt)` for boxcar modulators (e.g., context held
  constant during an anticipation period).
- Both produce the same `(T, J)` tensor shape consumable by the ODE
  interpolator; the utility is a pre-processor, not part of the forward model.

**HRF convolution deliberately NOT included.** In DCM the BOLD response is
generated by the Balloon model from the neural state, not by pre-convolving
inputs with an HRF. Pre-convolution would double-count the hemodynamic
transform. Users who want gamma-convolved drives should do that in domain code
before calling `make_event_stimulus` — the utility is input-agnostic.

**Dependency:** Similar structure to the existing `make_block_stimulus` at
`simulators/task_simulator.py:237`. No dependency on any other v0.3.0 task.

---

### TS-5: Bilinear Simulator (DCM.4)

| Aspect | Detail |
|--------|--------|
| Feature | `simulate_task_dcm(..., B_list=None, stimulus_mod=None, ...)` — extends existing simulator |
| Why expected | Users need ground-truth data to validate the forward model and run recovery tests. Without a simulator, no recovery benchmark is possible. |
| Complexity | Small-to-Medium |
| Confidence | HIGH |
| Dependencies | DCM.1 (forward model), DCM.3 (stimulus utilities) |

**Specification:**
- Extends the existing `simulate_task_dcm(A, C, stimulus_driving, …)` call
  with optional `B_list` and `stimulus_mod` keyword args.
- `B_list is None` → current linear path (zero regression risk).
- Return dict gains `'B_list'` and `'stimulus_mod'` keys when bilinear.
- Numerical stability: log a warning (not error) if any `A + Σ_j u_j B_j` has
  an eigenvalue with positive real part during simulation — see TS-6.

**Dependency:** Builds on `simulate_task_dcm` in `simulators/task_simulator.py`.

---

### TS-6: Eigenvalue Stability Check for A_eff(t)

| Aspect | Detail |
|--------|--------|
| Feature | Soft monitoring: warn if `max Re(eig(A + Σ u_j B_j))` > 0 during simulation/inference |
| Why expected | Bilinear dynamics can destabilize even when A is stable. Users need a clear signal when their priors/data produce runaway dynamics. |
| Complexity | Small |
| Confidence | HIGH |
| Dependencies | DCM.1 |

**Specification:**
- Evaluate `A_eff` eigenvalues at simulation time points (sample-subset to
  avoid cost; e.g., every 10th step).
- Warn once per simulation; include max eigenvalue real part and the time.
- Don't hard-stop: in some studies, transient instability is recoverable and
  users may want to see the diverging trace.
- Document threshold: `max Re(eig(A_eff)) > 0` as the warn trigger.

**Dependency:** The YAML DCM.1 description explicitly calls this out as
REQUIRED for "debugging." I classify it as table-stakes because without it,
inference silently produces NaNs and users have no diagnostic path.

---

### TS-7: Bilinear Recovery Benchmark (DCM.V1)

| Aspect | Detail |
|--------|--------|
| Feature | `benchmarks/runners/task_bilinear.py`: 3-region, 1 driving + 1 modulatory, N-seed recovery suite |
| Why expected | Every DCM variant in the codebase has a recovery benchmark. Without one for bilinear, the feature is unverified. |
| Complexity | Medium |
| Confidence | HIGH |
| Dependencies | DCM.1–4 all required; reuses `benchmarks/runners/task_svi.py` pattern |

**Specification:**
- **Network:** 3-region, same graph topology as existing linear benchmark
  (enables direct linear-vs-bilinear comparison at fixed A).
- **Inputs:** 1 driving (block design, reuse `make_block_stimulus`) +
  1 modulatory (event-related with 20 events, amplitudes drawn
  iid from N(1, 0.25) — realistic HGF-quantity variability).
- **B-matrix:** 2 non-zero elements with known signs and magnitudes ~0.3
  (moderate effect size; typical of published DCM B values).
- **SNR:** 3 (literature "realistic" setting per Frässle 2017; above the
  shrinkage floor of SNR ≤ 2 per Daunizeau 2012).
- **Seeds:** Reuse existing fixture infrastructure (3 sizes × N seeds from
  v0.2.0). Target ≥ 10 seeds for stable RMSE std bars.
- **Metrics:**
  - A RMSE, A correlation (should match linear benchmark within noise).
  - B RMSE, B correlation.
  - B sign recovery: fraction of masked-active elements with `sign(B̂) == sign(B_true)`.
  - Wall-time overhead vs linear DCM.
- **Acceptance criteria (revised from YAML against literature):**
  - A RMSE ≤ 0.15 (match linear, from existing v0.1.0 benchmark).
  - B RMSE ≤ 0.20 (YAML value; consistent with Frässle 2017 reporting
    RMSE ~0.15–0.30 at SNR=3 for whole-brain regression DCM — smaller
    networks are easier).
  - B sign recovery ≥ 80% (YAML value; literature-plausible but not
    directly cited in a published paper as a threshold — flag as
    empirically set, not reference-backed).

**Dependency:** Reuses `benchmarks/runners/task_svi.py` infrastructure
(shared fixture generation from v0.2.0, `compute_rmse`, seed loop,
acceptance gates).

---

## Differentiators

Features that go beyond a faithful SPM-port and add scientific value. Not
required for "bilinear DCM works," but useful for a publishable research tool.

### D-1: Per-Modulator Posterior Diagnostic Plots

| Aspect | Detail |
|--------|--------|
| Feature | Auto-generated figure: posterior mean B_j heatmap + 90% CI per modulator |
| Value | Users running multi-modulator studies need to compare B_j posteriors visually. SPM's B matrix viewer is a MATLAB figure; a publication-quality Python version is differentiating. |
| Complexity | Small |
| Confidence | HIGH |
| Dependencies | TS-2, existing matplotlib infrastructure (benchmarks already produce 9 figure types per v0.2.0) |

**Specification:**
- One subfigure per modulator j: N×N heatmap of posterior means, annotated
  with 90% CI bars on a side panel.
- Color scale: diverging, centered at 0, shared across all B_j for
  cross-modulator comparison.
- Optional overlay: hatch on masked-off elements (b_mask == 0).
- Ties into v0.2.0's `benchmarks/figures/` infrastructure.

**Defer-eligibility:** MEDIUM. The benchmark runner must produce at least a
one-panel B comparison to report sign-recovery numbers; richer per-modulator
plots can land after DCM.V1's numerical results are in.

---

### D-2: B-Sparsity Diagnostic

| Aspect | Detail |
|--------|--------|
| Feature | Report fraction of B posterior CIs that exclude 0 per modulator |
| Value | Indicates which modulator-connection pairs the data actually constrains. Useful for users doing model reduction post-hoc (à la BMR). |
| Complexity | Small |
| Confidence | MEDIUM |
| Dependencies | TS-2, posterior-extraction utilities already in `models/guides.py` |

**Specification:**
- Per modulator j: count of `(i, k)` pairs where 90% credible interval of
  `B_j[i, k]` excludes 0.
- Report alongside RMSE/sign-recovery in benchmark output.
- Useful signal: BE prior variance is 1.0 (wide), so a tight posterior that
  excludes 0 is real evidence.

---

### D-3: Linear-vs-Bilinear Nested Model Comparison Utility

| Aspect | Detail |
|--------|--------|
| Feature | Run same data through `task_dcm_model` with `B_list=None` AND with `b_masks` non-trivial; report ELBO delta |
| Value | Users want to know whether adding modulators actually improves fit. The v0.1.0 codebase has ELBO-based model comparison infrastructure — reusing it for linear-vs-bilinear is a natural capability. |
| Complexity | Small-to-Medium |
| Confidence | MEDIUM |
| Dependencies | TS-1–5, existing ELBO comparison logic |

**Specification:**
- Helper: `compare_linear_vs_bilinear(data, a_mask, b_masks, c_mask, ...)`
  returns `{'elbo_linear': …, 'elbo_bilinear': …, 'delta_elbo': …}`.
- Caveat: ELBO is a lower-bound proxy for free energy; direct comparison
  across models with different numbers of parameters needs the complexity
  penalty (standard in SPM).
- Optional extension: Bayesian model reduction (BMR) to test every subset of
  B_j elements. Flag as STRETCH — BMR adds substantial complexity.

---

### D-4: Modulator-Amplitude Sensitivity Report

| Aspect | Detail |
|--------|--------|
| Feature | Sweep: rerun benchmark at stimulus amplitudes {0.5, 1.0, 2.0, 5.0}, report B RMSE vs amplitude |
| Value | Quantifies the identifiability cliff: B recovery requires strong modulators. Users need this to plan experiments. |
| Complexity | Medium (requires multiple benchmark runs) |
| Confidence | MEDIUM |
| Dependencies | TS-7 (as base runner) |

**Specification:**
- Rerun the bilinear benchmark with modulator amplitudes scaled by factors
  {0.5, 1.0, 2.0, 5.0}.
- Plot RMSE and sign-recovery as a function of amplitude.
- Expected shape: monotone decreasing RMSE with amplitude, plateau at high
  amplitudes (matching Daunizeau/VBA observations of modulator identifiability).
- Defer-eligible: NICE-TO-HAVE. The base benchmark answers "does it work";
  this answers "when does it work best."

---

### D-5: Two-State DCM B-Prior Mode

| Aspect | Detail |
|--------|--------|
| Feature | Config flag to use SPM two-state B prior (variance 1/4) instead of one-state (variance 1) |
| Value | Future-proofs for two-state DCM extension (Marreiros, Kiebel & Friston 2008). Minimal effort now pays off if/when two-state is implemented. |
| Complexity | Small |
| Confidence | LOW |
| Dependencies | TS-2 |

**Specification:**
- Optional `b_prior_variance: float = 1.0` kwarg to the Pyro model.
- Document: `1.0` matches SPM one-state; `0.25` matches SPM two-state.
- Defer-eligible: this is speculative until two-state work is scoped.

**Caution:** May be YAGNI. Include only if cost is ≤10 lines of code.

---

## Anti-Features

Features that seem reasonable but are **explicitly out of scope** for v0.3.0.
Each listed with the reason it's excluded.

### AF-1: Nonlinear DCM (Second-Order `x·x` Terms)

| Dimension | Detail |
|-----------|--------|
| Request | Add D-matrices (`Σ_k D_k x_k · x`) — state-dependent connectivity from Stephan et al. 2008 |
| Surface appeal | "Full DCM" as in SPM covers nonlinear too; bilinear alone seems incomplete. |
| Why NOT | Nozari et al. 2024 (REF-051) empirically shows linear/bilinear suffices for macroscopic BOLD; nonlinear terms are rarely supported by fMRI data. Project PROJECT.md constraint: "bilinear justified by Nozari 2024." |
| Alternative | If users need nonlinear, they can point-estimate with SPM; flag as "future contribution" in docs. |

---

### AF-2: Time-Varying A(t)

| Dimension | Detail |
|-----------|--------|
| Request | Allow A itself to vary with time (e.g., learning/adaptation) |
| Surface appeal | Modeling plasticity or state-dependent baselines |
| Why NOT | PROJECT.md line 74: "Non-stationary A(t) extensions — deferred to v0.2, requires separate contribution." Explicitly out of scope. This is a different (and deeper) contribution. |
| Alternative | Bilinear B-matrix covers many plasticity-like phenomena via input-dependent modulation. For true time-varying A, defer to a future v0.4+. |

---

### AF-3: Trial-by-Trial Bayesian Updating

| Dimension | Detail |
|-----------|--------|
| Request | Online Bayesian updates as trials arrive (e.g., Kalman-style) |
| Surface appeal | Real-time adaptation, closed-loop BCI applications |
| Why NOT | PROJECT.md line 76: "Clinical deployment or real-time processing — research tool only." Pyro SVI is batch-oriented by design. |
| Alternative | Re-run inference with growing data windows if trial-wise updates are needed. |

---

### AF-4: PEB-lite / Group-Level GLM on B Parameters (DCM.5)

| Dimension | Detail |
|-----------|--------|
| Request | Fit `B_i(subject) = X(subject) β + ε` across subjects |
| Surface appeal | HEART2ADAPT needs group-level B contrasts (e.g., "does anxiety group predict B_Δhr[0,1]?") |
| Why NOT | YAML explicitly scopes DCM.5 OUT of v0.3.0. PROJECT.md line 65: "Group-level PEB-lite GLM — not scoped to this single-subject toolbox." |
| Alternative | Extract per-subject B posterior medians, fit group GLM in external code (bambi/pymc). Leave PEB-lite to v0.4.0+. |

---

### AF-5: 4-Node HEART2ADAPT Circuit Benchmark (DCM.V3)

| Dimension | Detail |
|-----------|--------|
| Request | Validate on the specific 4-node AMY↔dACC↔vmPFC↔Insula network |
| Surface appeal | Direct applicability to HEART2ADAPT study |
| Why NOT | Study-specific; the toolbox benchmark should be domain-neutral. Deferred to v0.4+ per YAML. |
| Alternative | The 3-region generic benchmark (DCM.V1) validates the mechanism. HEART2ADAPT-specific circuit lands in v0.4 or in the study repo itself, not here. |

---

### AF-6: SPM12 Bilinear Cross-Validation (DCM.V2)

| Dimension | Detail |
|-----------|--------|
| Request | Run same simulated data through Pyro-DCM (SVI) and SPM12 (VL); compare posteriors |
| Surface appeal | Gold-standard validation against the reference implementation |
| Why NOT | Requires MATLAB + SPM12 + MATLAB-to-Python bridge. Deferred per YAML to v0.4+. Not blocking v0.3.0 because DCM.V1 (recovery on known ground truth) is itself a strong validation signal. |
| Alternative | Rely on DCM.V1 for v0.3.0. Plan DCM.V2 for v0.4 when MATLAB infrastructure is available. |

---

### AF-7: HRF-Convolved Input Pre-Processing Inside `make_event_stimulus`

| Dimension | Detail |
|-----------|--------|
| Request | Build gamma-HRF convolution into the stimulus utility |
| Surface appeal | Convenience for users coming from GLM fMRI workflows |
| Why NOT | Would double-count hemodynamics: DCM's Balloon model already transforms neural → BOLD. Pre-convolving the input to the neural state equation is scientifically wrong. |
| Alternative | Document clearly that `make_event_stimulus` produces **neural-level inputs**; HRF is handled downstream by `BalloonWindkessel`. If users want GLM-style convolved inputs for visualization only, provide a separate helper outside the DCM pipeline. |

---

### AF-8: Complex B-Matrix Parameterization (Log-Transform, Sign-Constrained, Etc.)

| Dimension | Detail |
|-----------|--------|
| Request | Apply a transform to B (e.g., `B = exp(B_free) - exp(-B_free)` for bounded effects) |
| Surface appeal | Prevents unphysical large modulators |
| Why NOT | SPM12 uses unconstrained Normal priors on B; transformation would diverge from the reference. Stability is ensured by (a) negative A diagonal and (b) small prior variance (1.0 at mask-active, 0 elsewhere). YAML DCM.2 explicitly notes "no special transform (B elements are unconstrained, unlike A diagonal)." |
| Alternative | If runaway B values are seen in practice, tighten the prior variance (e.g., 0.25) rather than add a nonlinear transform. |

---

## Feature Dependencies

```
TS-4 (event/epoch stimulus util)                [standalone]
       │
       ▼
TS-1 (bilinear forward model)
       │
       ├────► TS-2 (B-matrix priors) ──► TS-3 (b_masks)
       │
       ├────► TS-5 (bilinear simulator) ◄── TS-4
       │
       ├────► TS-6 (eigenvalue monitor)
       │
       └────► TS-7 (recovery benchmark) ◄── TS-2, TS-5

D-1 (per-modulator plots)     ──enhances──► TS-7
D-2 (sparsity diagnostic)     ──enhances──► TS-2 posterior extraction
D-3 (linear-vs-bilinear ELBO) ──enhances──► TS-1 + existing ELBO comparison
D-4 (amplitude sensitivity)   ──enhances──► TS-7
D-5 (two-state prior flag)    ──optional──► TS-2
```

### Dependency Notes

- **TS-1 is the keystone:** Nothing else ships without the bilinear forward
  model working. Do DCM.1 first.
- **TS-4 can ship in parallel with TS-1:** The stimulus utility is a pure
  function over times/amplitudes; no forward-model coupling.
- **TS-6 (stability check) should land with TS-1,** not later: without it,
  debugging TS-5/TS-7 becomes painful (silent NaNs).
- **TS-7 is the integration test:** It validates TS-1 + TS-2 + TS-3 + TS-4 +
  TS-5 end-to-end. If the benchmark doesn't pass its acceptance criteria,
  something upstream is broken.
- **All differentiators are additive:** They enhance but don't block the
  table-stakes features. Safe to defer any of them.

---

## MVP Definition

### Ship With v0.3.0 (Required)

- [x] **TS-1** — Bilinear neural state equation (DCM.1). No bilinear without this.
- [x] **TS-2** — B-matrix priors with SPM12-matching variance=1 (DCM.2).
- [x] **TS-3** — Per-modulator b_masks (DCM.2).
- [x] **TS-4** — Event + epoch stimulus utilities (DCM.3).
- [x] **TS-5** — Bilinear simulator (DCM.4).
- [x] **TS-6** — Eigenvalue stability monitor.
- [x] **TS-7** — Recovery benchmark with acceptance gates (DCM.V1).

### Add When Time Permits (v0.3.0 stretch)

- [ ] **D-1** — Per-modulator posterior plots. Enhances TS-7 output.
- [ ] **D-2** — B-sparsity diagnostic. Small addition, high interpretive value.
- [ ] **D-3** — Linear-vs-bilinear ELBO comparison. Reuses existing infrastructure.

### Defer to v0.4+ (out of scope)

- [ ] **D-4** — Amplitude sensitivity sweep. Useful but expensive (multi-run).
- [ ] **D-5** — Two-state B prior flag. Only valuable if/when two-state DCM is scoped.
- [ ] DCM.5 (PEB-lite), DCM.V2 (SPM cross-val), DCM.V3 (HEART2ADAPT circuit) — see YAML.

---

## Feature Complexity Matrix

| Feature | User Value | Implementation Cost | Priority | Depends On (existing) |
|---------|------------|---------------------|----------|----------------------|
| TS-1 bilinear state eq | HIGH | SMALL | P1 | `NeuralStateEquation`, `CoupledDCMSystem` |
| TS-2 B priors N(0,1) | HIGH | SMALL | P1 | `task_dcm_model.py` A/C sampling pattern |
| TS-3 b_masks | HIGH | SMALL | P1 | `a_mask` convention |
| TS-4 event/epoch utils | HIGH | SMALL | P1 | `make_block_stimulus` pattern |
| TS-5 bilinear simulator | HIGH | SMALL-MEDIUM | P1 | `simulate_task_dcm` |
| TS-6 eigenvalue monitor | MEDIUM | SMALL | P1 | torch.linalg.eigvals |
| TS-7 recovery benchmark | HIGH | MEDIUM | P1 | `benchmarks/runners/task_svi.py`, v0.2.0 fixture infra |
| D-1 per-modulator plots | MEDIUM | SMALL | P2 | `benchmarks/figures/` infrastructure |
| D-2 B-sparsity diagnostic | MEDIUM | SMALL | P2 | posterior extraction |
| D-3 lin-vs-bilin ELBO | MEDIUM | SMALL-MEDIUM | P2 | v0.1.0 ELBO model comparison |
| D-4 amplitude sweep | LOW-MEDIUM | MEDIUM | P3 | TS-7 |
| D-5 two-state B flag | LOW | SMALL | P3 | TS-2 |

**Priority key:**
- **P1** — Must ship in v0.3.0 for bilinear-DCM to count as "done."
- **P2** — Should ship if time; substantial user value at low cost.
- **P3** — Defer to v0.4+ unless trivially cheap.

---

## Confidence

| Claim | Confidence | Source |
|-------|------------|--------|
| SPM12 one-state B prior: variance 1.0 at mask-active | **HIGH** | `spm_dcm_fmri_priors.m` line ~66 verified via GitHub; Zeidman et al. 2019 Table 3 (PMC6711459) |
| SPM12 two-state B prior: variance 1/4 | **HIGH** | Same source, line 89 |
| YAML's "N(0, 1/16)" claim is INCORRECT | **HIGH** | Direct contradiction with SPM source |
| B diagonal is user-configurable (mask-controlled) | **HIGH** | Zeidman 2019 PMC6711459 explicit example; `spm_fx_fmri.m` doesn't force-zero |
| Stick + boxcar are the SPM stimulus primitives | **HIGH** | SPM docs, parametric modulation pages |
| SNR=3 is "realistic" benchmark standard | **HIGH** | Frässle 2017 (REF-020), Daunizeau/VBA |
| B sign-recovery ≥ 80% is plausible at SNR=3 | **MEDIUM** | Reasonable based on simulations but not directly cited as a threshold in any paper. The YAML value is empirically sensible; treat as a project-internal acceptance criterion, not a reference-backed one. |
| B RMSE ≤ 0.20 is achievable at SNR=3 for 3-region | **MEDIUM** | Consistent with Frässle 2017's RMSE 0.29 at SNR=3 for 66-region; 3-region should do considerably better, but exact threshold is empirical |
| "Nonlinear terms rarely improve BOLD fit" (justifies AF-1) | **HIGH** | Nozari et al. 2024 (REF-051) is the primary citation in PROJECT.md |
| HRF pre-convolution would double-count (justifies AF-7) | **HIGH** | Direct consequence of DCM's generative structure (Balloon model is the hemodynamic transform) |

### Open Questions for Requirements Phase

1. **B sign-recovery ≥ 80%** — verify or relax: no published paper cites this
   as a threshold. Suggest either (a) keep as internal target and document as
   such, or (b) calibrate against v0.3.0 benchmark results and set the
   threshold post-hoc from a pilot run.
2. **Modulator amplitude scaling** — The benchmark uses amplitudes ~N(1, 0.25).
   Real HGF-derived quantities can be on very different scales. Should the
   simulator standardize stimulus_mod to unit variance, or leave it to the
   user? Recommend: leave to user, document clearly.
3. **Eigenvalue warning threshold** — warn at `max Re > 0`, or at `max Re > -0.05`
   (preemptive)? Defer to implementation; both are defensible.

---

## Sources

- **Primary:** `spm_dcm_fmri_priors.m` (SPM12) — prior specifications:
  [github.com/spm/spm12/blob/main/spm_dcm_fmri_priors.m](https://github.com/spm/spm12/blob/main/spm_dcm_fmri_priors.m)
- **Primary:** `spm_fx_fmri.m` (SPM12) — bilinear dynamics reference implementation:
  [github.com/spm/spm12/blob/main/spm_fx_fmri.m](https://github.com/spm/spm12/blob/main/spm_fx_fmri.m)
- **Group-DCM guide Table 3:** Zeidman et al. (2019), "A guide to group
  effective connectivity analysis, part 1," PMC6711459 —
  [pmc.ncbi.nlm.nih.gov/articles/PMC6711459](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711459/)
- **DCM origin:** Friston, Harrison & Penny (2003), NeuroImage (REF-001) —
  [fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf](https://www.fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf)
- **Regression DCM recovery at SNR=3:** Frässle et al. (2017), NeuroImage
  (REF-020) — RMSE 0.29 at SNR=3 for 66-region; 0.09 at SNR=100.
- **VBA-toolbox DCM demo (3-node bilinear reference):**
  [mbb-team.github.io/VBA-toolbox/wiki/dcm/](https://mbb-team.github.io/VBA-toolbox/wiki/dcm/)
- **SPM parametric modulation / stick-function convention:** SPM docs —
  [fil.ion.ucl.ac.uk/spm/docs/manual/fmri_spec/fmri_spec/](https://www.fil.ion.ucl.ac.uk/spm/docs/manual/fmri_spec/fmri_spec/)
- **Nozari 2024 (linear suffices for macroscopic BOLD):** REF-051 in
  REFERENCES.md.
- **Existing codebase:**
  - `src/pyro_dcm/forward_models/neural_state.py` (linear baseline)
  - `src/pyro_dcm/forward_models/coupled_system.py` (ODE integration)
  - `src/pyro_dcm/models/task_dcm_model.py` (Pyro generative model — A_free
    prior pattern at line 126–131)
  - `src/pyro_dcm/simulators/task_simulator.py` (simulator + `make_block_stimulus`)
  - `benchmarks/runners/task_svi.py` (benchmark runner pattern to extend)

---

*Feature research for: bilinear DCM extension to pyro_dcm v0.3.0*
*Researched: 2026-04-17*
