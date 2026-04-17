---
type: "research"
scope: "pitfalls"
milestone: "v0.3.0"
branch: "bilinear"
domain: "bilinear DCM extension (B-matrix modulatory inputs) for existing Pyro-DCM codebase"
updated: "2026-04-17"
confidence: "HIGH for SPM-verified mechanics; MEDIUM for performance predictions"
scope_note: >
  Pitfalls specific to adding bilinear B-matrix support (DCM.1, DCM.2, DCM.V1,
  DCM.V3 in GSD_pyro_dcm.yaml). Does NOT repeat v0.2.0 pitfalls -- those still
  apply; see .planning/research/v0.2.0/PITFALLS.md. Covers what changes or
  appears newly when extending linear DCM to bilinear DCM.
---

# Pitfalls: Bilinear DCM Extension for Pyro-DCM v0.3.0

v0.1.0 linear task DCM (dx/dt = Ax + Cu) is shipped. v0.2.0 benchmark suite
shipped (commit 2f50fbc). v0.3.0 extends to:

    dx/dt = A x + Σ_j u_j(t) · B_j x + C u_driving(t)
    A_eff(t) = A + Σ_j u_j(t) · B_j

Ground truth for SPM claims: `spm12/spm_fx_fmri.m` and
`spm12/spm_dcm_fmri_priors.m` (verified via fetched source 2026-04-17).
Seven v0.2.0 pitfalls (P1, P2, P4, P6, P7, P10, P12) still apply verbatim;
four become worse under bilinear (see final section).

---

## Summary

| # | Pitfall | Severity | Primary Phase |
|---|---------|----------|----------------|
| B1 | A_eff(t) loses stability for sustained large u·B | Critical | DCM.1 |
| B2 | B non-identifiable under long-TR, short-epoch, sparse-event designs | Critical | DCM.V1, DCM.V3 |
| B3 | Amortized guide dimension mismatch (silent shape-ignore) | Critical | DCM.2 |
| B4 | Docstrings mislabel linear code as "bilinear" | High | DCM.1 rename |
| B5 | Free B diagonal produces A_eff positive self-coupling | High | DCM.2 |
| B6 | Overly permissive b_mask overfits B | High | DCM.2, DCM.V1 |
| B7 | Sign-recovery metric misleading when true B_ij ≈ 0 | High | DCM.V1 metrics |
| B8 | Wrong SPM prior variance in YAML (1/16 vs SPM's 1.0) | High | DCM.2 priors |
| B9 | v0.1.0 linear fixtures drift through bilinear code path | Medium | DCM.1 regression |
| B10 | Per-step ODE cost 3-6x linear; DCM.V3 needs compute budget | Medium | DCM.V3 |
| B11 | SVI convergence slower; no published DCM warmup trick | Medium | DCM.V1 |
| B12 | stim_mod interpolation at rk4 mid-steps changes B amplitude | Medium | DCM.1, DCM.3 |
| B13 | A RMSE degrades even when B truth is zero (Bayesian pricing) | Medium | DCM.V1 criteria |
| B14 | b_mask typing and None/empty-list API ambiguity | Low | DCM.2 API |

---

## Critical Pitfalls

### B1: A_eff(t) Loses Negative-Real-Part Eigenvalues Under Sustained Large u·B

**What goes wrong.** `parameterize_A` guarantees `a_ii = -exp(A_free_ii)/2 < 0`
for the intrinsic A. The bilinear sum A_eff(t) = A + Σ u_j B_j has no such
guarantee: a sampled B element near its prior tail combined with sustained
u_j produces positive-real-part leading eigenvalues. rk4 then either (a)
yields NaN after a few steps, or (b) produces finite-but-exponentially-wrong
trajectories that SVI backpropagates through, biasing the posterior.

**SPM reference.** `spm_fx_fmri.m` one-state form does
`P.A(:,:,1) = P.A(:,:,1) + u(i)*P.B(:,:,i)` with **no saturation, no clipping,
no exponential wrapper on the sum.** The two-state form applies `exp(P.A)/8`
after the sum — which we do not use. SPM tolerates this via (i) tight priors,
(ii) Variational Laplace's single linearization rather than full ODE
iteration, and (iii) Nelder-Mead posterior-mode search that stays near the
prior. Pyro SVI does MC gradient estimates over the full nonlinear ODE and
does not inherit these protections.

**Failure regime estimate.** SPM one-state prior: a_ii = -0.5 Hz nominal,
B_free ~ N(0, 0.5) per SPM (see B8). A sampled B at ~3σ combined with
u_mod = 1 sustained 2s pushes leading eigenvalue real part to ~+0.3 Hz;
states grow `exp(0.3·2) ≈ 1.8x` per 2s — blowup over 250s. Baldy et al.
(2025) reported divergent ODE trajectories for 10-param neural mass models
even under tight priors.

**Warning signs.**
- NaN in `predicted_bold` during SVI after step >10 (not step 1 — guide
  init near 0 keeps step 1 safe).
- `final_loss ≈ initial_loss` after 500 SVI steps (divergent sample loss
  being clipped silently).
- Guide `init_scale` must drop below 0.01 to avoid early divergence.
- A_eff eigenvalue monitor (if added) shows max real part > 0 for >5% of
  ODE time points.

**Prevention.**
1. **A_eff stability monitor in `coupled_system.py`:** compute
   `max(real(eig(A_eff(t))))` at a subsample of ODE steps; log when > 0.
   Do not raise during SVI — divergent draws are expected during sampling.
2. **Optional soft clamp on u_mod** at ODE RHS: `u_j' = sign(u_j) *
   min(|u_j|, u_max)` with `u_max = 2.0`. Document as a numerical
   safeguard, not a math change; unit-test clamp is inactive on valid data.
3. **Benchmark priors:** use `B_free ~ N(0, 1/16)` for simulation recovery
   (YAML spec) — 16x tighter than SPM default — to give margin. Document
   the choice in `docs/03_methods_reference/priors.md`.
4. **Standalone stability test** `tests/test_bilinear_stability.py`:
   worst-case B at prior 3σ, sustained u=1 for 500s, assert no NaN and
   max BOLD < 10x no-modulation baseline.
5. **Stability-aware guide init:** `init_loc_fn` returns zeros for B_free
   sites explicitly.

**Phase relevance:** DCM.1. Stability monitor + test must land with the
bilinear ODE code, before DCM.2.

**Confidence:** HIGH on mechanism (SPM source + linear-time-varying ODE
theory). MEDIUM on exact numeric threshold (network- and stimulus-dependent).

---

### B2: B Non-Identifiable Under Long-TR, Short-Epoch, or Sparse-Event Designs

**What goes wrong.** HEART2ADAPT-style designs (192 events at ~12.5s ISI,
brief event-related modulators) are exactly the regime Rowe et al. (2015)
identified as producing non-identifiable B. Posterior CIs span most of the
prior; the mean carries no information. The DCM.V1 acceptance criteria
(B RMSE ≤ 0.20, sign recovery ≥ 80%) cannot distinguish this from genuine
recovery.

**Published evidence.**
- **Rowe et al. (2015), PMC4335185.** At TR=6.44s (2x the 3.22s original),
  direct observation of "non-identifiabilities for the parameters of the
  connections leading to V5 and also for the modulating exogenous inputs."
  Halving session duration (180 volumes) loses identifiability on multiple
  parameters. Epochs ~1s make modulatory strengths worse: "the
  switching-on and -off of a stimulus cannot be recognized."
- **Zeidman et al. (2019), PMC6711459.** "Limiting modulatory effects to
  the self-connections, rather than including the between-region
  connections, adds biological interpretability and generally improves
  parameter identifiability." HEART2ADAPT puts B on between-region
  connections — the opposite choice.
- **Penny et al. (2004).** Bayesian model comparison is required (not
  optional) when B has many free elements.

**Warning signs.**
- B posterior std ≈ prior std (shrinkage < 20%).
- B RMSE depends strongly on event density; A RMSE does not.
- Sign recovery = 50% (chance), not target 80%.
- B coverage at 90% CI > 0.95 (prior-dominated over-coverage).
- Posterior B correlates with truth at r < 0.3 while A correlates at r > 0.9.

**Prevention.**
1. **Pre-experiment identifiability check** before DCM.V3: simulate the
   exact HEART2ADAPT design (192 events, 12.5s ISI, matching SNR) and
   measure B posterior shrinkage. Declare under-powered if posterior
   std > 0.7 × prior std for any free B element.
2. **"Effective events per B parameter" design metric:**
   `n_eff = Σ_t u_mod(t)² / max(u_mod)²` divided by free-B count.
   Require ≥ 20 effective events per element (based on Rowe's 180-volume
   threshold).
3. **Extend DCM.V1 acceptance criteria:**
   - posterior B std / prior B std ≤ 0.7 for all free elements.
   - For non-zero truth B, 90% CI excludes zero in ≥ 70% of datasets.
4. **Sign fallback for non-identifiable elements.** Per Rowe: report
   sign-only claims separately from magnitude claims; validate both.
5. **Document the minimum-data regime** in `docs/03_methods_reference/`:
   "B estimation requires modulator power ≥ X and duration ≥ Y."

**Phase relevance:** DCM.V1 (shrinkage metric), DCM.V3 (gate before full
simulation), docs.

**Confidence:** HIGH. Rowe 2015 and Zeidman 2019 directly warn against
these regimes.

---

### B3: Amortized Guide Dimension Mismatch (Silent Shape Failures)

**What goes wrong.** The v0.2.0 `AmortizedFlowGuide` + `TaskDCMPacker`
hardcode parameter vector `[A_free (N²), C (N·M), log(noise_prec)]` of
length `N² + N·M + 1`. At N=3, M=1: 13 features. Adding B_free_j sites
extends to `N² + N·M + J·N² + 1` (= 40 at N=3, M=1, J=3). Three
failure modes:

1. **Checkpoint shape mismatch (LOUD):** Loading v0.2.0 state dict raises
   "size mismatch for flow.transforms.0...". Good — visible.
2. **Silent wrong-parameter generation:** If packer dict lacks B sample
   sites, Pyro falls through to prior for those sites — effectively
   sampling B from prior, no amortization. ELBO shows large KL on B
   sites; user may not notice because A and C still converge.
3. **Stale standardization stats:** `fit_standardization` was called on
   v0.2.0 training (no B). NSF spline domain [-5, 5] (v0.2.0 P14) was
   calibrated for A_free range [-0.3, 0.3]. B_free at σ=0.25 occupies
   similar range, but at wider SPM default σ=1.0 may exceed [-5, 5].

**Warning signs.**
- Bilinear recovery reports A/C RMSE close to v0.1.0 but B RMSE ≈ prior std.
- `packer.n_features` does not match guide input dimension.
- `fit_standardization` not called on bilinear training data.
- Amortized-ELBO vs SVI-ELBO gap much larger than for linear.

**Prevention.**
1. **New `TaskBilinearDCMPacker` class** (do NOT extend `TaskDCMPacker`
   silently). Explicit `n_features = N²*(1+J) + N·M + 1` and
   `packer_version = "bilinear_v0.3.0"` tag.
2. **Version-tag guide checkpoints:** save `{"version": ..., "n_features":
   N, "J": J, ...}`; refuse to load on mismatch.
3. **`packer.assert_compatible_with(model)`:** inspect sample sites via
   `pyro.poutine.trace`; verify every site has a packer slot. Call from
   `AmortizedFlowGuide.__init__`.
4. **Re-fit standardization** on 50+ bilinear training examples. Document
   in DCM.2 plan.
5. **Do NOT warm-start bilinear guides from v0.2.0 weights.** Invalidates
   any reported amortization gap. Train from scratch; report v0.1.0/
   v0.2.0 guides as a separate "linear-only" datapoint.

**Phase relevance:** DCM.2. Packer and versioning must land with the
model change in a single commit.

**Confidence:** HIGH. Direct codebase inspection confirms current packer
hardcodes shape and lacks versioning.

---

## High-Severity Pitfalls

### B4: Docstrings Call Linear Model "Bilinear"

**What goes wrong.** Pre-v0.3.0, three places mislabel the linear code:
- `neural_state.py:3-4`: "Implements the bilinear neural state equation
  from [REF-001] Eq. 1" — but the code is only `Ax + Cu`.
- `neural_state.py:56`: `"""Bilinear neural state equation dx/dt = Ax + Cu.`
- `neural_state.py:58`: admits "restricted to the linear case (no
  modulatory B matrices)" but the class name and summary use "bilinear".

After v0.3.0, contributors grepping for "bilinear" hit both new and old
uses; confusion propagates.

**Warning signs.** Grep for "bilinear" returns both correct v0.3.0 uses
and the stale linear docstrings. Contributor writes B-dependent logic on
top of `NeuralStateEquation` assuming it is already bilinear.

**Prevention.**
1. **Rename as part of DCM.1:** `NeuralStateEquation` →
   `LinearNeuralStateEquation`; add new `BilinearNeuralStateEquation`.
   Keep alias `NeuralStateEquation = LinearNeuralStateEquation` with
   `DeprecationWarning`.
2. **Audit every "bilinear" mention** in DCM.1 commit; fix to match
   implementation.
3. **Update `CLAUDE.md`** directory-structure block to reflect the
   bilinear extension once DCM.1 ships.

**Phase relevance:** DCM.1. Cannot defer — each subsequent commit that
reads old docstrings propagates confusion.

**Confidence:** HIGH (direct code inspection).

---

### B5: Free B Diagonal Produces A_eff Positive Self-Coupling

**What goes wrong.** The v0.2.0 stability guarantee relies on
`a_ii = -exp(A_free_ii)/2 < 0`. But
`a_eff_ii(t) = a_ii + Σ_j u_j(t) · B_jii`. A sampled positive `B_jii`
under N(0, ·) prior (no sign constraint) and large u_j produces positive
effective self-coupling — direct stability violation.

The YAML DCM.2 notes "B diagonal: usually fixed at 0 ... unless explicitly
hypothesized" but leaves this as recommendation, not constraint. A user
setting `b_mask` diagonal to 1 and sampling freely would silently break
stability.

**SPM context.** Zeidman et al. (2019) recommend restricting B to
self-connections in the **two-state** model because `pE.B = 0` and the
exponential wrapper `exp(P.A)/8` enforces stability. In the **one-state**
model (which `pyro_dcm` uses) there is no wrapper — the bilinear sum is
added directly. So "B self-connections only", safe in SPM two-state, is
unsafe in our one-state model unless we add our own transform.

**Warning signs.** A_eff diagonal positivity fires during SVI;
monotonically growing BOLD on specific datasets (systematic, not noise);
posterior mass on B_jii concentrated at large positive values.

**Prevention.**
1. **Default `b_mask` diagonal to 0** in `parameterize_B`; explicit
   override emits warning.
2. **Negative-exp transform for B diagonal if free:**
   `b_jii_eff = -exp(B_free_jii) / 8`. Preserves negative-diagonal
   invariant under modulation.
3. **Assertion in bilinear forward pass:** non-zero B diagonal emits
   log warning once; hard error during benchmark runs.
4. **Unit test:** all B_jii at +2σ of prior, sustained u_j=1, assert no NaN.

**Phase relevance:** DCM.2 (`parameterize_B` + b_mask defaults).

**Confidence:** HIGH. Direct consequence of model equations; verified via
`spm_fx_fmri.m`.

---

### B6: Overly Permissive b_mask Overfits B

**What goes wrong.** Over-parameterized bilinear DCM (all-free b_mask,
multiple modulators) always fits simulated BOLD well: at N=3, J=3, there
are 27 B parameters on top of 12 A+C, vs ~450 BOLD observations. SVI
converges to a local ELBO minimum that overfits modulator-induced BOLD
fluctuations into B instead of locating them correctly.

**Published evidence.**
- **Razi et al. (2017), large-scale DCMs:** over-parameterization is the
  dominant failure mode at scale; proposed functional-connectivity priors
  as mode-space reduction.
- **Zeidman et al. (2019):** recommend the SPM workflow of fitting a
  "full" model and applying Bayesian Model Reduction (BMR) / PEB to
  prune. v0.3.0 does not ship BMR (DCM.5 is MEDIUM priority), so the
  full-model B posterior is unreliable on its own.

**Warning signs.**
- B RMSE better than A RMSE (suspicious — B should be harder).
- B posterior means on zero-truth elements are non-zero systematically.
- Predictive check `r² > 0.99` but parameter RMSE poor.
- `|corr(A_post, B_post)| > 0.8` (A and B compensating).

**Prevention.**
1. **API default:** b_mask allows only hypothesized B elements, not all.
   Zeidman: B sparser than A, roughly 1/3–1/2 A-mask density.
2. **DCM.V1 benchmarks both full and sparse b_mask.** Report separately;
   expect full-mask to be worse.
3. **Diagnostic flag:** `|corr(A_post_sample, B_post_sample)| > 0.8` →
   degenerate-identifiability warning.
4. **Document BMR plan:** even if DCM.5 deferred, note that real-data
   bilinear use should apply `spm_dcm_bmr`; add TODO for future
   Pyro-compatible BMR.

**Phase relevance:** DCM.2 defaults, DCM.V1 comparison, docs.

**Confidence:** HIGH for principle (Zeidman 2019, Razi 2017); MEDIUM
for overfitting threshold (design-dependent).

---

### B7: Sign-Recovery Metric Misleading When True B_ij ≈ 0

**What goes wrong.** YAML DCM.V1 acceptance is "B sign recovery ≥ 80%".
Ill-defined when true B_ij is near zero: inferred sign is a coin flip,
sign recovery = 50% in expectation, dragging aggregate below 80% even
when the method correctly reports "we don't know because truth is near
zero." Conversely, restricting to `|B_true| > threshold` invites
threshold-tuning cheating.

Rowe et al. (2015): sign is a fallback for magnitude non-identifiability
(implicitly assumes truth is non-zero). Coverage-of-zero is the correct
complement for the near-zero regime.

**Warning signs.** Sign recovery varies wildly with ground-truth
sparsity; method reports 100% sign recovery on dense synthetic truth
but fails on realistic sparse truth; sign recovery anti-correlates with
RMSE.

**Prevention.**
1. **Split metric by truth magnitude:**
   - `sign_recovery_nonzero`: sign match on `|B_true| > 0.1`.
   - `coverage_of_zero_null`: fraction of elements with `|B_true| < 0.05`
     whose posterior 90% CI includes 0.
2. **Separate acceptance thresholds:** `sign_recovery_nonzero ≥ 80%`;
   `coverage_of_zero_null ≥ 85%`.
3. **Define null region relative to prior, not absolute:**
   `|B_true| < 0.5 · prior_std`. Scale-invariant.
4. **Never aggregate sign recovery across zero and non-zero elements**
   without explicit split in the final table.

**Phase relevance:** DCM.V1 benchmark metrics. Must precede any
acceptance decision.

**Confidence:** HIGH (standard Bayesian practice).

---

### B8: SPM One-State B Prior Is Variance 1, Not 1/16

**What goes wrong.** The GSD YAML DCM.2 claims "B_free prior: N(0, 1/16)
— tighter than A prior (1/64)" with "Prior conventions (SPM12)". Direct
inspection of `spm_dcm_fmri_priors.m` (SPM12 main, fetched 2026-04-17)
shows the one-state model uses:

- `pC.A(:,:,i) = A/pA + eye(n,n)/pA;` with `pA = 64` — variance 1/64 ✓
- `pC.B = B;` — **variance 1.0** on masked elements, NOT 1/16.

The two-state model uses `pC.B = B/4;` (variance 1/4), still not 1/16.

Possible origin of 1/16: confusion with empirical-Bayes shrinkage applied
after BMR in the PEB cascade, where 1/16 is sometimes used as a "loose"
prior. But first-level task DCM uses the code values above.

**Why it matters.**
1. **Using 1/16 diverges from SPM** — DCM.V2 cross-validation against
   SPM will find systematic shrinkage bias in our B estimates.
2. **Using 1.0 (SPM default) is 16x looser** — SVI convergence is
   harder; search space grows.
3. **The YAML claim is auditable and wrong.** Any downstream reader
   will discover this.

**Prevention.**
1. **Decide explicitly with rationale:**
   - *SPM-matched (variance 1.0):* for DCM.V2 cross-validation. Expect
     slower SVI.
   - *Pyro-tight (variance 1/16):* for recovery-benchmark tractability.
     Document as `pyro_dcm` convention, NOT SPM.
2. **DCM.V1 reports both.** Under each prior, show B shrinkage + RMSE.
   A method that works under both is more trustworthy.
3. **Fix YAML/docs before DCM.2.** Replace "SPM12 convention" with
   correct citation or explicit deviation statement.
4. **Unit test:** assert prior variance matches the documented value
   in a reference YAML.

**Phase relevance:** DCM.2. Must be decided before any bilinear SVI runs.

**Confidence:** HIGH on SPM values (source verified); MEDIUM on origin
of 1/16 in YAML.

---

## Medium-Severity Pitfalls

### B9: v0.1.0 Linear Fixtures Through Bilinear Code Path

**What goes wrong.** DCM.4 says `B_list=None` falls back to linear. Risk:
DCM.1 bilinear `CoupledDCMSystem` branch may diverge numerically from
the linear path even when `B_list=None` (e.g., `A + 0*B = A` vs `A`
differs by float64 rounding of ~1e-16 per op, accumulating over 500
steps). v0.1.0 recovery fixtures might report slightly different RMSE
after refactor — not a bug, but noisier regression tests.

Worse: if bilinear is the new default and linear is only reached via
`B_list=None`, v0.1.0 tests that omit the argument dispatch through the
bilinear path silently.

**Warning signs.** v0.1.0 recovery RMSE shifts after v0.3.0 merge
(should be bit-exact unless intentional); `atol=1e-6` assertions now
fail at 1e-5; reference BOLD on disk mismatches regenerated output.

**Prevention.**
1. **Preserve linear path bit-exact.** `B_list=None` routes through
   original `LinearNeuralStateEquation.derivatives` unchanged. Do NOT
   re-implement `Ax+Cu` in the bilinear branch — call the existing
   function.
2. **Regression test `test_linear_invariance.py`:** v0.1.0 reference
   simulation with `B_list=None`, assert BOLD matches cached v0.1.0
   fixtures to `atol=1e-10`.
3. **Version-lock v0.1.0 reference fixtures** under
   `tests/fixtures/linear_reference_v0.1.0/*.pt`.
4. **Run full v0.1.0 test suite in v0.3.0 CI** before merging DCM.1.

**Phase relevance:** DCM.1.

**Confidence:** HIGH (standard SWE concern; touches hot path).

---

### B10: Per-Step ODE Cost 3–6x Linear; DCM.V3 Compute Budget

**What goes wrong.** Per-step FLOP count:
- Linear: `A@x + C@u` = `N² + N·M` multiplies.
- Bilinear: `(A + Σ u_j B_j)@x + C@u` = `N² + J·N² + N·M` if A_eff
  rebuilt each step, or `N² + N·M + J·N` if built incrementally. Every
  multiply is a torchdiffeq autograd graph node.

Concrete ratios: N=3, J=3: 12 → 39 multiplies (3.3x). N=4, J=3: 19 → 67
(3.5x). N=10, J=5: 105 → 605 (5.8x). Per-step memory adds `O(J·N²)` for
B_list plus graph nodes proportional to `steps · J·N²`.

**DCM.V3 extrapolation.** v0.1.0 linear 3-node 500-step SVI = 235s
(per CLAUDE.md). Bilinear 4-node J=3 at 500 steps: naïve 5.6x → ~1320s
≈ 22 min per seed. If SVI needs 2x more iterations (see B11): ~44 min.
Full DCM.V3 cell (5 seeds × 20 datasets) = ~73 hours wall time.

**Warning signs.** DCM.V3 never completes (silent timeout); Pyro param
store > 10 GB; GPU OOM at N=10 bilinear where linear fit.

**Prevention.**
1. **Pre-budget DCM.V3.** Compute total wall time with the formula
   above. If > 72 hours, reduce default to 3 seeds × 10 datasets.
2. **Per-step timing in bilinear forward:** log median step time; warn
   if > 2s at N=4, J=3.
3. **Try `step_size=1.0` for bilinear benchmarks** if DCM.V1 recovery
   quality holds. Test in DCM.V1 before DCM.V3.
4. **Do NOT use adjoint** — v0.2.0 flagged gradient reliability issues.
   Non-adjoint memory is fine at 500 steps, not at 1000+.

**Phase relevance:** DCM.V3.

**Confidence:** HIGH on FLOPs; MEDIUM on iteration-count factor.

---

### B11: SVI Convergence Slower; No Published DCM Anneal Trick

**What goes wrong.** v0.1.0 N=3 linear: 12 params. v0.3.0 N=3 J=3
bilinear: 39 params. ELBO landscape has more dimensions and stronger
A–B coupling (their product enters the likelihood). Baldy et al. (2025)
reported ADVI under-dispersion on 10-param neural mass models — already
unreliable at 10 params. At 39, B posterior is likely worse-calibrated
than A/C.

YAML asks for a published warmup/mask-anneal trick. None found in DCM
literature. Closest analogs:
- Stephan 2010 "Ten simple rules": hierarchical model fitting (not
  automatic annealing).
- SPM Variational Laplace: iterative prior updates between EM steps
  (not applicable to Pyro SVI).
- Mask annealing (`b_mask · α(step)` with α ramping 0→1): works in
  principle but is custom, not published for DCM.

**Warning signs.** ELBO plateaus by step 200 but RMSE still decreasing
at 3000; 3000-step ELBO better than 500-step but 6x wall time; B
posterior widths decrease monotonically (not converged at 500).

**Prevention.**
1. **Pre-specified convergence criterion:** ELBO relative change < 1e-4
   over last 100 steps. Report actual iteration count alongside wall
   time.
2. **Document warmup strategy in DCM.2.** Options: (a) existing
   ClippedAdam with `lrd` schedule; (b) custom `b_anneal` with `α`
   ramp over first 20% of SVI — lets A, C converge first. Caveat: (b)
   is not in published DCM literature.
3. **Increase `num_particles` to 4** for bilinear. 4x per-step cost
   but may halve required iterations.
4. **If ELBO does not converge in 3000 steps, document as negative
   result** and recommend NUTS for that configuration.

**Phase relevance:** DCM.V1 convergence analysis; DCM.V3 budget.

**Confidence:** MEDIUM. Slowness well-known (Baldy 2025); anneal trick
is LOW confidence.

---

### B12: stim_mod Interpolation at rk4 Mid-Steps

**What goes wrong.** Bilinear RHS calls `u_mod(t)` at every rk4 stage.
rk4 at `step_size=0.5` evaluates at `t, t+0.25, t+0.25, t+0.5`. For
modulatory inputs stored on a dt=0.5 grid, a stick function at `t=t_e`
with `u_mod[t_e]=amp, else 0`:

- If `PiecewiseConstantInput` uses `floor(t/dt)` indexing, then both
  `t=0` and `t=0.25` return the same `u_mod` value. A single-step
  impulse blurs into a 0.5s pulse — **effectively ~2x the discrete
  ground-truth modulator**. All recovered B values are biased by this
  factor.

YAML DCM.3 introduces `make_event_stimulus` (sticks) and
`make_epoch_stimulus` (boxcar). DCM.1 must specify how these interact
with rk4 mid-steps.

**Warning signs.** Bilinear sim BOLD has 2x higher modulator-driven
amplitude than SPM same-design; B recovery RMSE dominated by systematic
2x scaling; changing dt 0.5→0.25 changes recovered B by ~2x.

**Prevention.**
1. **Choose interpolation semantics explicitly.** For stick events, the
   continuous-time representation is `u(t) = amp · δ(t - t_e) · dt_event`
   — effectively an ODE with jump.
2. **Prefer boxcar (epoch) stimuli for modulators** via
   `make_epoch_stimulus`. Matches SPM internal behavior. Document sticks
   as compatibility mode only.
3. **dt-invariance unit test:** simulate with `dt=0.5, 0.25, 0.1` and
   assert BOLD matches to `atol=1e-3`. Passing confirms dt-invariance.
4. **If sticks remain user-facing,** warn in `make_event_stimulus`
   docstring about rk4 mid-step blur; recommend 1s boxcar as safer default.

**Phase relevance:** DCM.1, DCM.3.

**Confidence:** MEDIUM. Specific 2x factor depends on implementation;
principle is HIGH.

---

### B13: A RMSE Degrades Even When B Truth Is Zero (Bayesian Pricing)

**What goes wrong.** Adding B_free expands the posterior even when the
simulated truth has B=0. The extra dimensions absorb ELBO budget; the
posterior on A is slightly more diffuse. Example: v0.1.0 linear N=3
A RMSE ~0.08 → v0.3.0 bilinear N=3, J=1, B_true=0 A RMSE may rise to
0.09–0.12. This is correct Bayesian behavior, not a bug.

Posterior correlations between A_ii and B_jii (see B5) mean A picks up
variance from B even when data identifies A tightly.

If v0.3.0 is measured against v0.1.0 A-recovery, the bilinear code can
appear to regress even when correct.

**Warning signs.** "A RMSE same as linear" criterion fails by 10–30%;
`b_mask=0` restores v0.1.0 A RMSE; A posterior std grows ~20% when
adding zero-truth B.

**Prevention.**
1. **Relax DCM.V1 acceptance:** replace "A RMSE ≤ 0.15 (same as linear)"
   with "A RMSE ≤ 1.25 × linear-baseline RMSE", OR compare against
   bilinear-with-b_mask=0 as the null.
2. **Explicit regression:** linear model vs bilinear-with-zero-B must
   give identical A posteriors (bit-exact if code path preserved per B9).
3. **Document** as known limitation: "B pricing inflates A posterior
   uncertainty even when B_true=0; Bayesian behavior, not a bug."

**Phase relevance:** DCM.V1 acceptance design.

**Confidence:** MEDIUM. Direct Bayesian parameter-counting consequence;
magnitude depends on prior/data strength.

---

## Minor Pitfalls

### B14: b_mask Typing and None/Empty-List API

YAML DCM.2 specifies `b_masks: list of J (N,N) binary masks`. API risks:

- Caller passes `(J,N,N)` tensor vs list of tensors — runtime error if
  not accepted both.
- Boolean vs float64 `b_mask` → unexpected dtype on multiplication.
- `b_masks=None` must fall back to linear; DCM.4 handles this for
  simulator but DCM.2 model must handle it too.
- `b_masks=[]` (J=0) should behave identically to linear; distinguish
  from "missing argument".

**Prevention.**
1. **API contract in docstring:** `b_masks: list[Tensor] of J (N,N)
   float64 tensors, or None`.
2. **Normalize at model entry:** `None` → `[]`, list → `(J, N, N)` stack.
3. **Unit test edge cases:** `J=0`, `J=1 all-zero mask`, `dtype=float32
   input`.

**Phase relevance:** DCM.2 API.

---

## Phase-Specific Warning Summary

| Phase | Primary Pitfalls | Mitigation Order |
|-------|-------------------|------------------|
| DCM.1 (bilinear ODE) | B1, B4, B9, B12 | B1+B4 first; B9 before merge |
| DCM.2 (Pyro model) | B3, B5, B6, B8, B14 | B3+B8 before any SVI |
| DCM.3 (stimulus utility) | B12 | In DCM.1 test |
| DCM.4 (simulator) | B9, B14 | Low risk after DCM.1 |
| DCM.5 (group GLM) | None specific; inherits v0.2.0 | — |
| DCM.V1 (recovery benchmark) | B2, B7, B11, B13 | B7 in metrics; B13 in criteria |
| DCM.V2 (SPM cross-validation) | B8 | Before running |
| DCM.V3 (4-node HEART2ADAPT) | B2, B10, B11 | Compute budget first |

---

## v0.2.0 Pitfalls That Become WORSE Under Bilinear

Per instructions not to repeat v0.2.0 unless severity changes:

- **v0.2.0 P1 (mean-field coverage ceiling) — WORSE.** Bilinear has more
  posterior correlations (A ⊗ B product). Mean-field coverage on B will
  be strictly worse than on A. Expected ceiling drops ~0.80 → ~0.65 for
  B elements.
- **v0.2.0 P6 (full-rank memory) — WORSE.** At N=10, J=5: D = 100 (A) +
  50 (C) + 500 (B) + noise ≈ 651 dims vs 111 linear. Cholesky factor =
  213k entries vs 6k. Full-rank infeasible at this scale; low-rank
  becomes non-optional at N ≥ 5.
- **v0.2.0 P10 (amortized training distribution) — WORSE.** Training
  space for bilinear is `N² + N·M + J·N²`-dimensional. 50 training
  examples undersample by a factor `(1 + J·N²/(N²+N·M))`. Use 200+ for
  bilinear.
- **v0.2.0 P14 (spline domain truncation) — WORSE.** B prior σ wider
  than A (1.0 vs 1/64 SPM). Standardized B samples hit [-5, 5] boundary
  faster. Verify with P14's histogram check.

Severity remains HIGH/CRITICAL; prevention tightens (low-rank mandatory,
more training examples, re-fit standardization).

---

## Confidence Assessment

| Pitfall | Confidence | Basis |
|---------|-----------|-------|
| B1 | HIGH | SPM source; linear-time-varying ODE theory |
| B2 | HIGH | Rowe 2015; Zeidman 2019 |
| B3 | HIGH | Direct codebase inspection |
| B4 | HIGH | Direct codebase inspection |
| B5 | HIGH | Model equations; SPM comparison |
| B6 | HIGH principle; MEDIUM threshold | Razi 2017; Zeidman 2019 |
| B7 | HIGH | Standard Bayesian metrics |
| B8 | HIGH on SPM value; MEDIUM on YAML origin | SPM source |
| B9 | HIGH | Standard SWE |
| B10 | HIGH FLOPs; MEDIUM iterations | Direct analysis |
| B11 | MEDIUM | Baldy 2025; no DCM anneal literature |
| B12 | MEDIUM | Implementation-dependent |
| B13 | MEDIUM | Bayesian parameter counting |
| B14 | HIGH | API hygiene |

---

## Sources

### Primary (HIGH — verified source or codebase)

- **SPM12 source, main branch, 2026-04-17.**
  - `spm_fx_fmri.m`: bilinear sum, no saturation (one-state).
    [GitHub](https://github.com/spm/spm12/blob/main/spm_fx_fmri.m)
  - `spm_dcm_fmri_priors.m`: one-state `pC.B = B` (variance 1.0),
    two-state `pC.B = B/4`. Confirms B8.
    [GitHub](https://github.com/spm/spm12/blob/main/spm_dcm_fmri_priors.m)
- **Rowe et al. (2015) "Assessing parameter identifiability for dynamic
  causal modeling of fMRI data."** Frontiers Neurosci.
  [PMC4335185](https://pmc.ncbi.nlm.nih.gov/articles/PMC4335185/)
- **Zeidman et al. (2019) "A guide to group effective connectivity
  analysis, part 1."** NeuroImage.
  [PMC6711459](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711459/)
- **Friston, Harrison, Penny (2003) "Dynamic causal modelling."**
  NeuroImage 19(4), 1273-1302.
  [FIL PDF](https://www.fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf)
- **Codebase:** `src/pyro_dcm/forward_models/neural_state.py`,
  `src/pyro_dcm/guides/parameter_packing.py`,
  `src/pyro_dcm/models/task_dcm_model.py`.

### Secondary (MEDIUM — published but general, or extrapolated)

- **Baldy, Woodman, Jirsa, Hashemi (2025).** DCM inference in
  NumPyro/PyMC/Stan. J R Soc Interface 22(227):20240880.
  [PMC12133347](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133347/)
- **Razi et al. (2017) "Large-scale DCMs for resting-state fMRI."**
  Network Neuroscience.
  [MIT Press](https://direct.mit.edu/netn/article/1/3/222/2199/)
- **Penny, Stephan, Mechelli, Friston (2004) "Comparing dynamic causal
  models."** NeuroImage 22(3):1157-1172.
- **Friston et al. (2016) "Bayesian model reduction and empirical Bayes
  for group (DCM) studies."** NeuroImage.
  [PMC4767224](https://pmc.ncbi.nlm.nih.gov/articles/PMC4767224/)
- **Stephan et al. (2010) "Ten simple rules for dynamic causal
  modeling."** NeuroImage.
  [PMC2825373](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825373/)

### v0.2.0 Context (reference only)

- `.planning/research/v0.2.0/PITFALLS.md` — 17 pitfalls; seven (P1, P2,
  P4, P6, P7, P10, P12) still apply. Four become worse under bilinear
  (see section above).
