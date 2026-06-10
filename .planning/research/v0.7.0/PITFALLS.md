# Domain Pitfalls: v0.7.0 Variational Laplace Validation

**Domain:** Variational Laplace DCM engine validation + SPM12 cross-check
**Researched:** 2026-06-10
**Confidence:** HIGH (grounded in codebase, cluster logs, audit docs, and SPM12 source)

Sources: `src/pyro_dcm/inference/variational_laplace.py`, `src/pyro_dcm/inference/csd_precision.py`,
`src/pyro_dcm/inference/forward_models.py`, `src/pyro_dcm/model_selection/bmr.py`,
`cluster/logs/bmr_vs_elbo_55772525.out`, `cluster/logs/sbi_spectral_55772094.err`,
`.planning/milestones/v0.6.0-MILESTONE-AUDIT.md`, `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`,
`.planning/phases/06-validation-against-spm/06-RESEARCH.md`, SPM12 local source.

---

## Critical Pitfalls

### Pitfall C1: Laplace Overconfidence Breaks Absolute BMR Pruning

**What goes wrong:** The VL Laplace posterior is sharply overconfident at high SNR. Posterior
standard deviations are ~0.001–0.01 times the prior standard deviation. This makes the
`make_reduced_prior_zero_connection` shrinkage prior (variance 1e-8) only marginally tighter than
the posterior itself, so the BMR delta-F formula punishes *every* reduction by a massive negative
number regardless of whether the pruned connection is present or absent.

**Empirical evidence (job 55772525):** `prune_2to0` (truly absent connection) scores ΔF = -115.9;
`prune_0to1` (truly present) scores ΔF = -182,293. Both are wildly negative, but the ratio is
~1570x. The absolute threshold (e.g., "prune if ΔF > -3") never fires; only the relative
ordering survives.

**Root cause:** The Laplace Hessian at the posterior mode overestimates curvature when the forward
model is well-specified and data are informative. The posterior covariance `V @ Cp @ V.T` reflects
this compressed uncertainty, so the BMR log-evidence formula (which involves `log|Sigma_posterior| -
log|Sigma_reduced|`) yields a large negative offset that dominates the quadratic discrimination
term.

**Warning signs:**
- All reduced-model ΔF values are deeply negative (< -50), even for absent connections
- Absolute BMR pruning thresholds (e.g., ΔF > -3 nats, SPM convention) never activate
- Posterior std / prior std ratio is < 0.05 on every parameter

**Prevention:**
- Use RELATIVE evidence ranking only: rank reduced models by ΔF, do not threshold absolutely
- Report the separation gap: `ΔF(absent) - ΔF(present)` is the meaningful quantity, not the
  absolute values
- If absolute thresholding is needed, apply posterior tempering before BMR (see Pitfall C2)
- Do NOT set a pass/fail criterion of "absent connections get ΔF > threshold"

**Owning phase:** Phase C (BMR + model comparison)

---

### Pitfall C2: Posterior Tempering Introduces New Calibration Hazards

**What goes wrong:** Tempering the Laplace posterior (inflating `Sigma_post` by a factor T > 1)
before BMR recovers the absolute-threshold regime, but introduces new pitfalls:
(a) The optimal temperature is data- and model-dependent; a fixed T picked on one N/SNR
condition may overcorrect on others.
(b) Tempering makes the posterior covariance inconsistent with the free energy already computed —
you cannot interpret the BMR ΔF values as changes in log evidence anymore without re-deriving
the normalisation.
(c) At low SNR the posterior is already diffuse; tempering pushes it below the prior precision,
which can make the reduced posterior covariance non-positive-definite (BMR returns -inf).

**Warning signs:**
- BMR returns `delta_f = -inf` with a "non-positive definite" warning for some reductions after
  tempering
- ΔF values become sensitive to the choice of temperature across conditions in the recovery matrix
- Tempering-corrected and raw rankings disagree on present-vs-absent for low-SNR cells

**Prevention:**
- Validate the temperature schedule on held-out conditions at multiple SNR levels before using
  it as a reporting criterion
- Always check that `torch.linalg.cholesky(sigma_r_post)` succeeds after tempering;
  `bayesian_model_reduction` already guards this with a warning but it should be a hard assertion
  in the validation harness
- Treat tempered BMR as exploratory in Phase C; the untempered relative ranking is the primary
  evidence

**Owning phase:** Phase C

---

### Pitfall S1: SPM12 Parameter Parameterization Mismatch (A Matrix Diagonal)

**What goes wrong:** SPM12 stores `DCM.Ep.A` as free parameters (not the parameterised A matrix).
For the diagonal, SPM's actual self-connection is `-exp(Ep.A_ii) / 2`, matching
`parameterize_A()` in this codebase. When comparing posterior means, naively subtracting
`spm_A - our_A_free` compares apples to oranges if one side has applied the transform and the
other has not. This doubles or halves apparent errors on self-connections.

**Warning signs:**
- Self-connection errors are systematically 2x larger than off-diagonal errors
- Off-diagonal elements match within tolerance but diagonal elements do not
- Posterior mean comparison passes on off-diagonal, fails on diagonal

**Prevention:**
- Always compare in the SAME space. The cleaner choice is free-parameter space (compare
  `our A_free` vs `SPM Ep.A` directly), because the nonlinear transform amplifies small
  differences in a direction that is hard to normalise
- Document which space is used for every comparison table and assert the transform has been
  applied (or not) consistently before computing any error metric
- Add a unit test: simulate A_free, apply `parameterize_A`, verify the diagonal formula matches
  SPM's `-exp(x)/2` convention explicitly

**Owning phase:** Phase B (validation matrix)

---

### Pitfall S2: SPM12 Prior Conventions — Scale, Mean, and Hyperprior

**What goes wrong:** SPM12 spectral DCM uses several prior conventions that differ from the
defaults in `run_variational_laplace`:

| Parameter | SPM12 value | dcm_pytorch default |
|-----------|-------------|---------------------|
| A prior variance (pC.A) | `a_mask / 64` (= 1/64 where mask=1) | 1/64 (matches) |
| A prior mean offset | `a_mask / 128` | 0 (does NOT match) |
| Hyperprior mean (hE) | 8.0 for spectral DCM | computed from data: -log(var(y))+4 |
| Hyperprior precision (ihC) | 128.0 for spectral DCM | exp(4) ≈ 54.6 |
| Hemodynamic prior variance | 1/256 | 1/256 (matches) |

The prior mean offset (`a_mask / 128`) shifts A toward slightly positive values (weakly excitatory
network), which biases the posterior mean away from zero even when true connections are absent.
The hyperprior mismatch (8 vs ~data-dependent) changes which noise precision is preferred and
therefore the effective signal-to-noise weighting of the likelihood.

**Warning signs:**
- Posterior A means are consistently biased positive by ~0.008 relative to SPM (1/128 = 0.0078)
- Observation noise estimates differ by a constant offset in log space
- Free energy from our engine is lower than SPM's by ~10-50 nats even when posteriors agree

**Prevention:**
- When running SPM12 cross-validation, use `hyperprior_mean=8.0`, `hyperprior_precision=128.0`,
  and `prior_mean_a_offset=a_mask / 128` in `run_variational_laplace` / `_run_vl_generic`
  (all three parameters are already exposed in the API as of commit e1934e1)
- Run one condition with default parameters and one with SPM-matched parameters to quantify
  the effect; report both in the comparison table
- Never compare F values between SPM and dcm_pytorch runs that used different hyperpriors

**Owning phase:** Phase B

---

### Pitfall S3: Free Energy F Is Not Directly Comparable Across Implementations

**What goes wrong:** SPM's `DCM.F` and dcm_pytorch's `result.free_energy[-1]` are both lower
bounds on log model evidence, but they are computed with different approximations:
- SPM includes log-det terms from the full (non-diagonal) posterior covariance computed via
  the full parameter vector (including hyperparameters)
- dcm_pytorch's L_1 + L_2 + L_3 decomposition operates in the SVD-reduced subspace; the
  reduction changes the effective dimensionality and therefore the normalisation constant
- SPM's data precision includes the observation noise estimated via ReML jointly with the
  likelihood; dcm_pytorch's `compute_csd_precision` uses a fixed Wishart approximation
  initialised from the data

The absolute difference can easily be tens to hundreds of nats even when posterior means agree.

**Warning signs:**
- The F gap between SPM and dcm_pytorch is large (>50 nats) but posterior means agree
- Ranking of two models differs only when F values are very close (< 5 nats separation)
- One implementation consistently produces higher F across all conditions

**Prevention:**
- NEVER use absolute F values for the SPM cross-validation criterion
- The valid comparison is MODEL RANKING: on a set of synthetic conditions where the correct
  model is clearly better (has a true connection vs lacks it), both engines should agree on
  which model has higher evidence
- For ranking tests, require a separation gap of > 3 nats (equivalent to Bayes factor ~20)
  before counting a disagreement as meaningful

**Owning phase:** Phase B

---

### Pitfall S4: MATLAB/Python Data Layout — Column-Major vs Row-Major

**What goes wrong:** MATLAB stores matrices in column-major (Fortran) order; Python/NumPy/PyTorch
use row-major (C) order. `scipy.io.loadmat` partially handles this by transposing 2D arrays,
but complex CSD arrays and 3D tensors require explicit attention.

This project already encountered a C-order bug in `csd_precision.py` (fixed in commit 64e326f):
the index mapping `j = idx % N; i = (idx // N) % N; w = idx // (N*N)` must match the
PyTorch `.reshape(-1)` convention (C-order, fastest index last, which means j varies fastest).
The MATLAB `reshape(Q, [], 1)` equivalent uses Fortran order (column-major, i varies fastest).

**Warning signs:**
- CSD precision matrix Q has off-diagonal structure that is transposed relative to expected
- Posterior A elements appear in the wrong (i,j) position
- Off-diagonal A errors are asymmetric: A[1,0] matches but A[0,1] does not, or vice versa

**Prevention:**
- When loading any SPM result via `scipy.io.loadmat`, explicitly check the shape and apply
  `.T` where MATLAB column-major convention has transposed a matrix vs the expected layout
- For CSD arrays exported to/from MATLAB: export as (F, N, N) in Python; MATLAB will see
  this as a 3D array `(N, N, F)` in Fortran order. Explicitly permute on the MATLAB side with
  `permute(csd, [2, 3, 1])` to get `(F, N, N)` back
- After loading results: print `A_spm[0,1]` and `A_spm[1,0]` for a known asymmetric ground
  truth and confirm the sign pattern matches expectation before running any comparison
- Add a round-trip test: export a known asymmetric matrix to .mat, reload with `loadmat`, and
  assert element `[i,j]` matches the expected value

**Owning phase:** Phase B

---

### Pitfall S5: SPM12 BOLD Scaling Before VL Estimation

**What goes wrong:** `spm_dcm_estimate` (task DCM) scales BOLD internally to enforce maximum
change of 4% (line 151-153 in local SPM12). `spm_dcm_fmri_csd` (spectral DCM) scales to
target precision ~4 (line 124-127). Our Python simulator does not apply this normalisation.
The scaling affects the log-likelihood normalisation constant, which changes the absolute F
but should leave posterior A means approximately unchanged (the noise precision absorbs the
scale). However, the hemodynamic parameters (epsilon, transit time) interact with this scaling
non-trivially.

**Warning signs:**
- Posterior A means agree between SPM and dcm_pytorch but noise hyperparameter `h` estimates
  differ by a constant offset in log space (~log(scale_factor))
- F values differ by a systematic constant proportional to `log(scale_factor^2 * n_timepoints)`

**Prevention:**
- For cross-validation, apply SPM's scaling to the Python-generated BOLD before exporting:
  `bold = bold / (bold.abs().max() * 0.04)` for task DCM approximation
  OR accept the constant F offset and compare only parameter posteriors
- Do not apply the scaling inside the Python inference; apply it only to the data that is
  exported for SPM comparison, so both engines see the same (scaled) data

**Owning phase:** Phase B

---

### Pitfall S6: SPM12 Microtime Resolution for Task DCM Inputs

**What goes wrong:** `spm_dcm_estimate` expects `DCM.U.u` at microtime resolution (16× TR by
default), not at TR resolution. SPM internally discards the first 32 samples via
`Sess.U(i).u(33:end,j)`. Providing stimulus at TR resolution produces a mismatch in the
numerical integration of the hemodynamic convolution, leading to wrong posterior C parameters.

**Warning signs:**
- Posterior C estimates differ substantially (>20%) between SPM and dcm_pytorch
- A matrix estimates agree but C estimates do not
- SPM F value is much lower when stimulus is at TR resolution (poor data fit)

**Prevention:**
- Upsample stimulus to microtime grid before export: `u_dt = TR / 16`; prepend 32 zero samples
  to match SPM's internal convention (`Sess.U(i).u(33:end,j)` skip)
- Verify `DCM.U.dt` is set to `u_dt` (not TR) in the exported struct
- Add a test: check that `DCM.U.u.shape[0] == n_scans * 16 + 32` after export

**Owning phase:** Phase B

---

## Recovery Matrix Pitfalls

### Pitfall R1: Simpson's Paradox Across Forward Models and Scales

**What goes wrong:** Pooled metrics across forward models (spectral, task, latent-circuit) or
across scales (N=3 vs N=5, low vs high SNR) can show aggregate recovery that is misleading.
A 3-region model at SNR=10 recovers A with RMSE 0.03, but a 5-region model at SNR=2 gives RMSE
0.15. Averaging gives RMSE 0.09 — neither informative about either condition.

More subtly, if one forward model (e.g., latent-circuit) consistently recovers well across all
SNRs while another (spectral) recovers poorly at low SNR, an aggregate across-model metric makes
both look mediocre and hides the structural difference.

**Warning signs:**
- Aggregate RMSE or R² is "acceptable" but per-condition inspection shows bimodal distribution
- Performance on N=5 forward model dominates the aggregate (larger parameter count)
- A metric passes at the aggregate level but fails for every individual condition when split by
  forward model

**Prevention:**
- Report per-cell results in the recovery matrix: one row per (forward model × N × SNR) condition
- Only aggregate within the same forward model and the same dimensionality regime
- Include a visual heatmap of the matrix; do not report a single number as the headline result
- Flag any cell with RMSE > 2× the median as a reliability concern, not a pass/fail flag

**Owning phase:** Phase B

---

### Pitfall R2: Metric Artifacts — Pooled vs Per-Region R², Masked vs Unmasked Sign Recovery

**What goes wrong (specific to this codebase):** This project has already encountered two metric
artifact failures:

1. **Variance-pooled R²:** Computing `1 - SS_res / SS_tot` across all elements pools variance
   across regions. If one region has high variance, it dominates the R² numerator, masking poor
   recovery in low-variance regions. The correct metric is per-region R², then report
   mean ± std across regions.

2. **Sign recovery over structural zeros:** A connection absent from the ground truth (A[i,j]=0)
   has sign `sign(0) = 0` in PyTorch, but the posterior may place the mean at a small positive
   or negative value. Computing `(sign(A_inferred) == sign(A_true)).mean()` counts zero-entries
   as either all-correct (if the inferred value happens to round to zero) or all-wrong. The
   masked metric restricts to entries where `|A_true| > threshold` (0.01 used in v0.6.0, yielding
   masked sign recovery 0.7745 vs unmasked 0.4425 for HVAE-02).

**Warning signs:**
- R² appears good (>0.9) but some regions visually show poor tracking
- Sign recovery seems low (<0.5) despite qualitatively correct connectivity
- A result changes dramatically when the threshold for "structural zero" is varied

**Prevention:**
- Always compute per-region R², report as `mean ± std` across N regions
- Always use masked sign recovery with an explicit threshold (recommend |A_true| > 0.01)
- Report the threshold used as part of the result; include a sensitivity table showing recovery
  at thresholds 0.005, 0.01, 0.05
- Separate the sign-recovery rate for truly-absent vs truly-present connections

**Owning phase:** Phase B

---

### Pitfall R3: Too Few Seeds Mistaken for Reliable Recovery

**What goes wrong:** Running a recovery test with only 3 seeds at a given (N, SNR) condition
can give artificially good results if the random A matrices generated happen to be well-conditioned
or have strong signal. VL convergence and recovery quality are sensitive to the condition number
of A and the frequency overlap between signal and noise.

The project's existing test (`TestVLRecovery` in `test_variational_laplace_recovery.py`) uses
5 seeds (100–104) with a pass criterion of ≥3 converging. At moderate SNR (10.0) and N=3, this
is a loose gate.

**Warning signs:**
- All 5 seeds converge and succeed; variance across seeds is very low
- Changing to seeds 200–204 yields noticeably different results
- The reported RMSE is at the boundary of the acceptance criterion on 2 of 5 seeds

**Prevention:**
- Recovery matrix should use ≥10 seeds per condition; report median and 10th percentile RMSE
- Flag any condition where the 10th-percentile RMSE exceeds 1.5× the median (high variability)
- Use a fixed seed schedule (e.g., seeds 0–9 for all conditions) to enable comparison across
  conditions on the same circuits
- Do not report a single-seed result as a recovery number; always aggregate

**Owning phase:** Phase B

---

### Pitfall R4: Identifiability Limit Misidentified as Engine Bug

**What goes wrong:** At low SNR (≤2) or with near-unstable A matrices (max real eigenvalue
close to the `eig_clamp` boundary), VL may return a posterior that differs from the ground
truth by > 15% on several connections. This is not a bug — it is the fundamental identifiability
limit of the DCM likelihood at that SNR. Misdiagnosing it as a VL bug wastes time chasing
implementation errors.

The `eig_clamp` issue is particularly insidious: parameters that differ in A_free space can
produce near-identical predictions after clamping (non-injectivity), which makes the Jacobian
degenerate near the boundary and causes VL to return a posterior centred away from the true
value even with perfect data.

**Warning signs:**
- Poor recovery is concentrated in conditions where the true A has max real eigenvalue between
  -0.1 and 0 (near the stability boundary)
- Free energy does not improve across iterations even with tight convergence criteria
- The Jacobian condition number (computed via `torch.linalg.svdvals(J)`) is > 1e6

**Prevention:**
- Characterise the identifiability limit separately before testing the engine: compute the
  Fisher information matrix analytically or via the Jacobian and report its condition number
  for each test condition
- Exclude A matrices with max real eigenvalue > -0.05 from the recovery matrix (these are
  inherently near the boundary)
- When recovery fails, first check the eigenvalue spectrum of A_true; if near-unstable, report
  as "outside identifiability regime" not "engine failure"

**Owning phase:** Phase B

---

### Pitfall R5: Convergence Criterion Does Not Imply Parameter Convergence

**What goes wrong:** The VL convergence criterion (4 consecutive iterations with predicted ΔF
< 0.1 nats) tests free energy stabilisation, not parameter stability. VL can satisfy this
criterion while still having parameters that move by >0.05 in A_free space, especially when
the free energy landscape is flat near the prior boundary. In flat regions, small parameter
movements do not change F significantly but the posterior mean is not yet at the mode.

**Warning signs:**
- `result.converged = True` but running 50 more iterations changes `A_free` by >0.02
- Free energy trace shows a long flat plateau before the final criterion is satisfied
- Recovery RMSE improves substantially by increasing `max_iter` from 128 to 256

**Prevention:**
- Add a secondary convergence check: `||p_{k} - p_{k-1}||_2 / ||p_{k-1}||_2 < 1e-4` for
  3 consecutive steps, in addition to the free energy criterion
- For the validation matrix, run with `max_iter=256` on all conditions; report the iteration
  at which the free energy criterion first fires and whether extending iterations changes the
  result
- Include free energy traces in diagnostic output for any condition where RMSE > 0.1

**Owning phase:** Phase B

---

## Numerical Pitfalls

### Pitfall N1: Precision Matrix Intractability for Long Time-Domain Runs

**What goes wrong:** `TaskDCMForward.build_precision` returns a single identity matrix `Q` of
shape `(T*N, T*N)`. At fine dt (dt=0.01) and long duration (100s), this is 10^4 time points ×
N regions, yielding a `(4e4, 4e4)` dense matrix for N=4. Inverting or Cholesky-decomposing this
matrix in every M-step iteration of VL is O(n^3) ≈ 10^{12} operations per iteration — completely
intractable.

**Warning signs:**
- VL on task DCM with dt=0.01 and T>500 takes >10 minutes per iteration
- OOM errors on the precision matrix construction
- Memory usage spikes to >32 GB before the first Jacobian step

**Prevention:**
- For task DCM VL, always use dt=0.1 or coarser (confirmed working in v0.6.0 latent-circuit runs)
- Cap time series at ≤500 points in the VL forward model; if longer data is needed, subsample
  or use windowed estimation
- If fine-dt task DCM is required, implement a sparse or block-diagonal precision (e.g.,
  AR(1) structure) instead of identity; document this as a known limitation in the methods

**Owning phase:** Phase B (awareness), Phase A (documentation)

---

### Pitfall N2: eig_clamp Non-Injectivity at the Stability Boundary

**What goes wrong:** The stability clamp `eig_clamp = -1/32 ≈ -0.031` used in
`spectral_dcm_forward` (and passed through `SpectralDCMForward`) maps all A matrices whose
eigenvalues have real part > -0.031 to matrices with eigenvalue exactly at -0.031. Two
different A_free vectors can therefore produce identical predictions, making the Jacobian rank-
deficient near the boundary. This was confirmed as the structural cause of SBI SBC failure
(2/9 params pass, job 55772094).

For VL, the same non-injectivity means the Jacobian column corresponding to an A_free parameter
near the boundary has near-zero finite-difference derivative, which makes the Gauss-Newton
Hessian (J^H @ iS @ J) nearly singular. The SVD reduction via `_spm_svd` partially mitigates
this by removing low-variance dimensions, but it does not remove directions where the Jacobian
is degenerate due to clamping rather than due to zero prior variance.

**Warning signs:**
- Recovery fails consistently for A matrices generated near the stability boundary
- The Jacobian condition number is > 1e6 for parameters near eig_clamp
- The posterior standard deviation on clamped eigenvalue parameters is equal to the prior
  standard deviation (VL cannot update it)

**Prevention:**
- When generating synthetic A matrices for the recovery matrix, explicitly exclude those with
  max real eigenvalue in [-0.05, 0]; use a lower bound of -0.1 for testing
- Report the recovery matrix stratified by the maximum real eigenvalue of the true A: show
  that recovery degrades as eigenvalues approach the boundary
- For the SPM12 comparison, use A matrices well within the stable regime (max eig < -0.15);
  document this restriction
- Consider removing `eig_clamp` for well-conditioned recovery tests where the true A is
  strongly stable, and enabling it only when the prior distribution may generate near-unstable
  samples

**Owning phase:** Phase B

---

### Pitfall N3: Float64 Codebase vs Float32 NumPyro / External Tools

**What goes wrong:** The entire dcm_pytorch / VL codebase runs in float64 (`torch.float64`,
`complex128`). If any NumPyro or JAX code is used for validation (e.g., NUTS on a DCM
to produce "true" posterior samples for calibration comparison), JAX defaults to float32.
The mismatch means matrix inversions, log-determinants, and Cholesky decompositions that are
well-conditioned in float64 can lose 8 decimal places of precision in float32, leading to
different numerical answers on the same problem.

This is also relevant if `scipy.io.savemat` is used with default numpy dtypes: arrays created
as float32 are exported as single-precision to .mat, but SPM12 uses double precision internally.

**Warning signs:**
- NumPyro NUTS produces different free energy estimates than VL on the same synthetic data
- `scipy.io.loadmat` returns float32 arrays that differ from expected float64 values in the
  8th decimal place (not a problem for comparison, but can trigger `dtype` assertion failures)
- The VL convergence criterion fires differently depending on whether the CSD is loaded from
  a .mat file (which may be float32/float64 depending on MATLAB save options)

**Prevention:**
- All numpy arrays passed to `scipy.io.savemat` must be explicitly cast to `float64`:
  `arr.astype(np.float64)` on every array in the DCM struct
- When loading NumPyro or JAX results for comparison, cast to float64 before computing metrics
- If JAX is used for any validation computation, set `jax.config.update("jax_enable_x64", True)`
  at the top of the script and verify with `jnp.ones(1).dtype`

**Owning phase:** Phase B

---

### Pitfall N4: Local Optima and Multiple Restarts in VL

**What goes wrong:** The VL Gauss-Newton optimizer finds the local optimum nearest to the
prior mean (zero in reduced space). DCM likelihoods are non-convex, especially for spectral DCM
where the transfer function H(ω) = (iωI - A)^{-1} is highly nonlinear in A. At low SNR, there
can be multiple modes with comparable free energy, and VL consistently finds the prior-nearest
mode rather than the global optimum. A result can look like "recovery failure" when it is
actually "converged to a suboptimal local mode".

**Warning signs:**
- Running VL from a different starting point (using `initial_p` API) produces a different
  posterior mean with significantly higher free energy
- The free energy trace shows multiple discrete jumps (accept/reject cycles) without a smooth
  monotonic improvement
- Median RMSE across seeds is acceptable but variance is high; the worst seeds show RMSE 3-5×
  the median

**Prevention:**
- For the recovery matrix, run 3 restarts per condition: (a) default (zeros), (b) initialised
  from the ground truth A_free projected into reduced space, (c) a random perturbation
- Report the best-restart RMSE separately from the single-start RMSE; large differences indicate
  multi-modality
- Document the acceptance rate (fraction of iterations where F improves) in the diagnostic log;
  acceptance rate < 0.2 is a reliable indicator of local-optima issues

**Owning phase:** Phase B

---

### Pitfall N5: Finite-Difference Step Size Sensitivity

**What goes wrong:** The Jacobian in `_compute_jacobian_generic` uses step size `exp(-8) ≈ 3.35e-4`.
This matches SPM12's `spm_diff.m` default. However, the optimal step size depends on the scale
of the function: for parameters near the stability boundary where the transfer function changes
rapidly with A_free, `exp(-8)` may be too large (truncation error dominates); for parameters
with very small VL updates (posterior variance 1e-6), `exp(-8)` may be larger than the posterior
uncertainty, making the numerical Jacobian inaccurate relative to the local curvature.

Additionally, the analytical hemodynamic Jacobian (commit a064e69) uses SPM12's `1/64 ≈ 0.016`
FD step, which is ~50x larger than `exp(-8)`. If the generic VL engine uses different step sizes
for different sub-components, the Jacobian is inconsistent and the Hessian approximation is wrong.

**Warning signs:**
- Jacobian columns have values that vary by >100× in magnitude across parameters
- The Gauss-Newton Hessian `J^H @ iS @ J` has condition number > 1e10
- Posterior covariance has near-zero diagonal entries for some parameters even though the prior
  variance was non-zero

**Prevention:**
- Use a single step size `exp(-8)` throughout `_compute_jacobian_generic`; do not use
  different step sizes for different parameter groups
- The analytical hemodynamic Jacobian (if used) must be verified to agree with the FD Jacobian
  at the same step size to within 1% on test cases
- Add a pre-flight check in the recovery test: compute the Jacobian at the prior mean and
  assert that no column has L2-norm < 1e-12 (would indicate a stuck parameter)

**Owning phase:** Phase B

---

## Phase-Specific Warnings

| Phase | Topic | Pitfall | Mitigation |
|-------|-------|---------|-----------|
| Phase A | VL engine docs | Documenting SPM12 equations without checking current code | Re-read `variational_laplace.py` before writing equations; confirm 3-term F formula matches spm_nlsi_GN |
| Phase B | Recovery matrix | Pooled R² hiding per-region failures | Report per-region R² mean ± std; masked sign recovery with explicit threshold |
| Phase B | SPM12 comparison | Free energy absolute comparison | Never compare absolute F; use model ranking with ≥3 nats separation |
| Phase B | SPM12 comparison | A-matrix parameterization mismatch | Choose one comparison space (free-param or parameterized) and assert consistency throughout |
| Phase B | SPM12 comparison | Column-major vs row-major CSD indexing | Add round-trip test: export asymmetric matrix, reload, assert element [i,j] |
| Phase B | Recovery matrix | Too few seeds | ≥10 seeds; report median and 10th percentile; fixed seed schedule |
| Phase B | Numerical | eig_clamp degeneracy | Exclude near-boundary A matrices; stratify results by max eigenvalue |
| Phase B | Numerical | Precision matrix OOM (task DCM) | Enforce dt ≥ 0.1 and T ≤ 500 for task DCM VL |
| Phase C | BMR | Absolute ΔF thresholding | Relative ranking only; separation gap is the primary metric |
| Phase C | BMR tempering | Non-positive definite reduced posterior | Assert Cholesky succeeds post-tempering; validate T schedule on held-out conditions |
| All phases | Determinism | Float32 leaking from external tools | Explicit float64 cast everywhere; JAX `jax_enable_x64` if used |

---

## Open Questions (Gaps Not Fully Resolved)

1. **Optimal eig_clamp value for recovery matrix:** Should the recovery matrix use
   `eig_clamp=-1/32` (SPM default) or disable it for well-conditioned tests? The answer depends
   on whether the goal is "reproduce SPM behaviour" or "test VL engine in isolation". Both are
   useful but require separate tables.

2. **Tempering schedule for absolute BMR:** No validated temperature schedule exists for
   dcm_pytorch. This needs empirical calibration on the Phase B recovery matrix before Phase C
   can use it reliably.

3. **SPM12 CSD vs dcm_pytorch CSD computation:** SPM's `spm_dcm_fmri_csd_data` uses a MAR model
   internally. Our `SpectralDCMForward` takes CSD directly. For the cross-validation to be
   apples-to-apples on spectral DCM, either (a) export BOLD to SPM and let it compute CSD, or
   (b) export our CSD and use it as the input to `spm_dcm_fmri_csd`. Path (a) is cleaner but
   introduces an additional CSD estimation step that is a source of discrepancy.

4. **Tolerance for "acceptable" VL recovery:** The existing criterion is RMSE on A < 0.05.
   What is the right threshold for each (N, SNR, forward-model) cell? This should be derived
   from the Fisher information at the ground truth, not chosen arbitrarily.
