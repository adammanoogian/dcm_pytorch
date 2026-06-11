# Phase 31: BMR Validation & Posterior Tempering (Exploratory) - Research

**Researched:** 2026-06-11
**Domain:** Bayesian Model Reduction validation harnesses over an existing VL DCM engine
**Confidence:** HIGH (grounded entirely in committed codebase, cluster results, and v0.7.0 research docs)

## Summary

Phase 31 is **assembly + validation**, not new infrastructure. Every primitive already
exists and is unit-tested: `bayesian_model_reduction`, `rank_connections`,
`temper_vl_posterior` in `src/pyro_dcm/model_selection/bmr.py`; VL fit/extract in
`src/pyro_dcm/inference/variational_laplace.py`; per-cell ground-truth + VL-fit drivers in
`benchmarks/recovery_matrix_grid.py`; the proven cluster glue pattern in
`cluster/scripts/recovery_matrix_cell.py` + `cluster/sbatch/recovery_matrix_sweep.sbatch`.
There is a **near-complete reference implementation of VLBMR-01 already on disk**:
`cluster/scripts/lc_vl_bmr_selection.py` fits the full N=4 latent-circuit model via VL,
slices `result.sigma_post[:N*N, :N*N]` for the A_free posterior covariance, ranks off-diagonal
edges by single-prune ΔF, checks that the top-K essential edges == the true chain, and reports a
separation ratio. Phase 31 mostly **hardens that script into a tested harness, adds the
brute-force VL-refit agreement check (VLBMR-02), and adds the tempering calibration loop against
Phase 30 coverage output (VLBMR-03)**.

The single load-bearing fact (do NOT re-litigate): absolute BMR ΔF is broken by Laplace
overconfidence (cluster job 55772525 — a truly-absent edge scored ΔF=-115.9, a present edge
scored ΔF=-182,293; both wildly negative, only the *ratio*/ordering is meaningful). The defensible
primary result is **relative single-prune ranking + separation gap** (`rank_connections`).
Tempering is exploratory: it inflates `Cp` by a factor T>1 to partially restore an absolute-ΔF
regime, calibrated empirically against Phase 30 coverage curves, reported side-by-side with the
untempered ranking, NEVER used as pass/fail.

**Primary recommendation:** Three plans — (31-01) VLBMR-01 relative-ranking recovery harness on
**spectral** ground truth (laptop-fast, coverage=1.0 substrate); (31-02) VLBMR-02 BMR-vs-brute-force
**VL-refit** agreement on a small model set (route to M3 for task/latent; laptop OK for spectral);
(31-03) VLBMR-03 exploratory tempering calibration consuming `benchmarks/results/recovery_matrix.json`
coverage, asserting Cholesky PD post-temper, reporting tempered vs untempered side-by-side.

---

## Exact Function Signatures To Call (HIGH confidence — read from source)

All in `src/pyro_dcm/model_selection/bmr.py`, all re-exported from
`pyro_dcm.model_selection` (`__init__.py` confirmed exports `rank_connections`,
`temper_vl_posterior`, `bayesian_model_reduction`, `bmr_circuit_selection`,
`make_reduced_prior_zero_connection`, `enumerate_reduced_models`).

### `rank_connections` (bmr.py:431) — PRIMARY, the defensible result

```python
result = rank_connections(
    posterior_mean,   # torch.Tensor (D,)   -- A_free posterior mean, flattened
    posterior_cov,    # torch.Tensor (D, D) -- A_free posterior covariance (FULL block)
    prior_mean,       # torch.Tensor (D,)   -- zeros
    prior_cov,        # torch.Tensor (D, D) -- prior_variance * eye(D)
    prunable_indices, # list[int]           -- off-diagonal flat indices only
    shrinkage_variance=1e-8,
)
```
Returns dict:
- `ranked`: list of `{index, prune_delta_f, rank (1-based), gap_to_next}`, sorted ascending by
  `prune_delta_f` (most-negative = most-essential FIRST). `gap_to_next` = next entry's
  `prune_delta_f` minus this entry's (non-negative); `None` for last.
- `separation_gap`: float — the **maximum** `gap_to_next` across the list (largest consecutive
  essentiality drop). **This is the reported separation gap.**
- `separation_after_rank`: int — 1-based rank after which `separation_gap` occurs (the natural
  cut between essential and non-essential edges).
- `prunable_indices`: echo.

Raises `ValueError` if `prunable_indices` empty or any index out of range `[0, D)`.

**NOTE the units mismatch with the existing cluster script.** `lc_vl_bmr_selection.py` reports a
separation *ratio* (`kth_dF / next_dF`); `rank_connections` reports a separation *gap* (additive
difference in ΔF). The phase goal says "separation gap reported" — use `rank_connections`
(`separation_gap`), not the ratio. The planner should standardize on `rank_connections` as the
single source of truth and treat `lc_vl_bmr_selection.py` only as a fit-plumbing reference.

### `bayesian_model_reduction` (bmr.py:27) — used internally; needed for VLBMR-02 brute-force comparison setup

```python
delta_f, mu_r, sigma_r = bayesian_model_reduction(
    posterior_mean, posterior_cov,   # full-model VL posterior (Ep, Cp sub-block)
    prior_mean, prior_cov,           # full-model prior
    reduced_prior_mean, reduced_prior_cov,  # from make_reduced_prior_zero_connection
)
```
Returns `(delta_f: float, mu_r: (D,), sigma_r: (D,D))`. Casts to float64 internally. If the
reduced posterior covariance is not PD, returns `delta_f=-inf` and NaN tensors with a warning.

### `make_reduced_prior_zero_connection` (bmr.py:157)

```python
r_mean, r_cov = make_reduced_prior_zero_connection(
    prior_mean, prior_cov, indices,  # list[int]
    shrinkage_variance=1e-8,
)
```
Zeros the mean and shrinks variance (default 1e-8) at `indices`, zeroing cross-covariances.

### `temper_vl_posterior` (bmr.py:552) — EXPLORATORY only

```python
sigma_tempered = temper_vl_posterior(
    sigma_post,           # torch.Tensor (D, D) -- VL posterior covariance to inflate
    tempering_factor=1.0, # float > 0; >1 inflates uncertainty
)
```
Returns the symmetrized tempered covariance `0.5*(T+T.T)` with `T = factor * sigma_post`, in
float64. Raises `ValueError` if `factor <= 0` OR if Cholesky fails (message includes shape and
factor — assert on this in tests). **This is the only PD guard you need; do not hand-roll a second
one** — call `temper_vl_posterior` then feed its output into `rank_connections`/BMR.

### VL fit + posterior extraction (the substrate)

`pyro_dcm.inference` exports `run_variational_laplace_generic`, `extract_vl_posterior_generic`,
`SpectralDCMForward`, `TaskDCMForward`, `LatentCircuitForward`.

```python
result = run_variational_laplace_generic(
    forward, observed=..., a_mask=..., n_regions=N,
    max_iter=64, prior_variance=1.0/64.0, context={"freqs": freqs},  # spectral
)
```

**CRITICAL for the planner — where the full A_free covariance lives:**
`extract_vl_posterior_generic(result, forward, N)` returns only `{name: {mean, std, samples}}` —
**std is the diagonal sqrt only; there is NO full-covariance block in its output.** BMR needs the
**full** A_free covariance. Get it directly from the result object:

```python
A_free_mean = result.theta_post["A_free"].reshape(-1).double()  # (N*N,)
A_free_cov  = result.sigma_post[: N*N, : N*N].double()          # (N*N, N*N) FULL block
```
A_free is **always the first packed block** (verified: variational_laplace.py:846-852 and
:901; `_pack_params` packs `A_free` first; generic unpack iterates `params_template` in pack
order with A_free first). So slicing `[:N*N, :N*N]` is correct for all three forward models.
`lc_vl_bmr_selection.py:94-95` does exactly this — mirror it.

---

## What Is Already Tested vs What Phase 31 Must Add

### Already covered (do NOT re-test)
| File | Covers |
|------|--------|
| `tests/test_bmr.py` | `bayesian_model_reduction` analytic correctness (1-D closed form, multidimensional consistency, antisymmetry, PD guard), `make_reduced_prior_zero_connection` |
| `tests/test_bmr_circuit_selection.py` | `enumerate_reduced_models`, `bmr_circuit_selection` (sparse-truth recovery, full-model baseline, result structure) |
| `tests/test_bmr_rank_connections.py` | `rank_connections` on a **synthetic analytic D=4 circuit** (present {0,1} rank above absent {2,3}, separation_gap>0, cut after rank 2, empty-indices raises) AND `temper_vl_posterior` (inflation preserves PD, non-PD raises loud naming shape+factor) — **all laptop-fast, no VL fit** |
| `tests/test_bmr_vs_elbo.py` | `@pytest.mark.slow` — fits a **3-region latent circuit** full model via **SVI (`run_svi`)**, BMR-scores 3 reduced models, brute-force refits each via **SVI**, asserts BMR/ELBO agree on relative ordering and worst model. Uses SVI, NOT VL. |

### What Phase 31 MUST add (the gaps)
1. **VLBMR-01:** An *end-to-end* harness that does a real **VL fit on synthetic ground truth**,
   extracts the full A_free covariance, runs `rank_connections`, and asserts **top-K essential
   edges == true edges** with a reported `separation_gap`. The existing `rank_connections` test
   uses a *hand-built analytic* posterior, not a VL fit — VLBMR-01 closes that gap.
   `lc_vl_bmr_selection.py` already implements the fit logic but is a cluster script with a
   different (ratio) separation metric and no pytest harness.
2. **VLBMR-02:** A BMR-vs-brute-force agreement check using **VL refit** (not SVI). The existing
   `test_bmr_vs_elbo.py` uses SVI; the VL engine is the one under validation in v0.7.0. Either
   upgrade that file or add a new VL-based test.
3. **VLBMR-03:** A tempering calibration loop that **consumes Phase 30 coverage output**, picks a
   temperature, applies `temper_vl_posterior`, re-runs `rank_connections`, and reports tempered vs
   untempered side-by-side. **Nothing tests this consumption path today.**

---

## VLBMR-01: Synthetic Ground-Truth Recovery Harness (PRIMARY)

### Substrate choice: SPECTRAL (HIGH confidence)
Phase 30 coverage (`benchmarks/results/recovery_matrix.json`): spectral cells (indices 0-3) have
`coverage_95 = 1.0`, `sign_recovery_masked = 1.0`, `convergence_rate = 1.0`, and **per-seed VL
fit ≈ 1.4s (N=2) to 7.5s (N=4)** (derived from cell elapsed times 13.7s/21.1s/53.2s/75.6s over 10
seeds in `cluster/results/recovery_matrix_5634644*.json`). Spectral is the strongest, fastest, and
best-calibrated substrate — the right one for the **defensible primary** VLBMR-01 result.
Latent-circuit (the existing `lc_vl_bmr_selection.py` substrate) is also valid and has a known true
chain, but each fit is ~170-190s and `coverage_95` is null (not computed for latent_circuit), so it
is a weaker calibration story. **Recommend spectral as primary; optionally keep a latent-circuit
variant as a secondary confirmation reusing the existing script's chain definition.**

### Defining "true edges" and "top-K essential edges"
- **Prunable indices = off-diagonal flat indices only.** The diagonal is structural
  self-inhibition (`parameterize_A` maps A_ii to `-exp(A_free_ii)/2`); never prune it.
  `lc_vl_bmr_selection.py:64`: `OFFDIAG = [i*N+j for i in range(N) for j in range(N) if i != j]`.
- **A[i,j] is at flat index `i*N + j`** (row-major / C-order — matches PyTorch `.reshape(-1)`;
  see Pitfall S4). Connection j→i maps to A[i,j].
- **True edges = the off-diagonal indices where `|A_true[i,j]| > threshold`** (use the masked
  threshold 0.01 already standard in this project, Pitfall R2). For spectral, generate A_true via
  `make_stable_A_spectral(N, seed=...)` then read off which off-diagonals are non-negligible — OR
  (cleaner, recommended) **construct a sparse ground-truth A with a known set of present edges**
  (like `test_bmr_vs_elbo.py` does: chain 0→1, 1→2 present, others absent) so "true edges" is
  unambiguous and K is fixed by construction.
- **top-K essential edges** = the first K entries of `rank_connections(...)["ranked"]` (most-negative
  prune ΔF first), with K = number of true present off-diagonal edges.
- **Assertion (VLBMR-01):** `set(idx for top-K ranked entries) == set(true present off-diag indices)`.
- **Reported separation gap:** `result["separation_gap"]` (and `separation_after_rank`). Assert it
  is positive; for the primary result, also assert `separation_after_rank == K` (the cut lands
  exactly between essential and non-essential). The headline number to report is `separation_gap`.

### Tolerances / thresholds (justified)
- Top-K exact recovery: **hard pass** (set equality). Spectral N=2/N=4 at coverage 1.0 should
  recover cleanly; if it does not even at the best substrate, that is a real finding worth surfacing.
- Multi-seed: use **≥5 seeds** (Pitfall R3 says ≥10 for recovery matrices; for a structure-recovery
  assertion ≥5 is acceptable since each is a binary pass and fits are seconds). Report fraction of
  seeds recovering true structure (mirror `lc_vl_bmr_selection.py`'s `n_recovered/n_seeds`).
- `separation_gap > 0` required. Do NOT assert an absolute nat threshold on `separation_gap`
  (it inherits the overconfidence-driven scale; the *sign and ordering* are what matter — Pitfall C1).
  Report the value; do not gate on a magnitude.
- `prior_variance = 1.0/64.0` for spectral (matches runner `_A_PRIOR_VARIANCE_BOLD`,
  `recovery_matrix_grid.py:141`). `prior_cov = (1/64) * eye(N*N)`, `prior_mean = zeros(N*N)`.

### Compute routing: LAPTOP OK
A single spectral VL fit is ~1.4-7.5s; VLBMR-01 over 5 seeds at N=2 and N=4 = well under 3 min.
**Runs on laptop** as `@pytest.mark.vl` (the marker added in Phase 29). No cluster needed.

---

## VLBMR-02: BMR vs Brute-Force VL Model Comparison

### What "brute-force" means here (HIGH confidence decision)
The phase goal validates "the analytic approximation" of BMR. The honest brute-force baseline is
**re-fitting each reduced model independently and comparing free energy**, NOT an ELBO from SVI.
The v0.7.0 engine under validation is **VL**, and VL produces a 3-term free energy
(`result.free_energy[-1]`). So:
- **Analytic side:** one full-model VL fit → `bayesian_model_reduction` ΔF for each reduced model
  (milliseconds, no refit).
- **Brute-force side:** for each reduced model, **re-fit via VL** with the connection zeroed by
  the `a_mask` (set `a_mask[i,j]=0`), and read the resulting free energy. Brute-force ΔF =
  `F_reduced - F_full` (per model; full model ΔF ≡ 0).

The existing `test_bmr_vs_elbo.py` does this with **SVI** (`run_svi`, `final_loss`). Phase 31
should produce the **VL** analogue (the engine actually being validated). Reuse the `a_mask`-zeroing
pattern from `test_bmr_vs_elbo.py:241-245` and the VL-fit pattern from `recovery_matrix_grid.py`.

### "Small model set" (concrete)
Mirror `test_bmr_vs_elbo.py`: full model + ~3 reduced models on a 3-region circuit (one prune-absent,
one prune-present, one prune-two-absent). This keeps brute-force refits to ~4 VL fits.

### Agreement metric / tolerance (defensible)
Because of Pitfall C1/S3, **absolute ΔF is not comparable** between analytic BMR and brute-force
refit (different normalisation; refit re-estimates noise hyperparameters). The defensible check is
**RANK AGREEMENT**, not value agreement:
- **Primary assertion:** BMR and brute-force VL agree on the **relative ordering** of reduced
  models — specifically that pruning a present edge costs more evidence than pruning an absent one,
  and they agree on the **worst** reduced model (mirror `test_bmr_vs_elbo.py:345-375`: assert
  `bmr["prune_present"] < bmr["prune_absent"]` AND the worst-ranked model matches).
- **Quantitative (optional, report-only):** Spearman or Kendall-tau rank correlation between the
  BMR ΔF vector and the brute-force ΔF vector over the reduced set. `scipy.stats.spearmanr` /
  `kendalltau` are available (scipy 1.16.3 in env). With only ~3-4 models, require Spearman ρ ≥ 0.9
  OR (more robustly for tiny sets) exact agreement on the worst model + present>absent ordering.
  Do **not** make ρ on a 3-element set a hard gate — too few points; use the ordering assertions as
  the gate and report ρ as supporting evidence.

### Pitfall specific to VLBMR-02
When you zero a connection via `a_mask`, the VL SVD reduction (`_spm_svd`) changes the effective
parameter dimensionality, so the brute-force F includes a different normalisation than the analytic
BMR's Laplace-at-the-full-mean ΔF (Pitfall S3). This is expected — it is exactly why the test must
be **ranking-based, never value-based**. Document this in the plan.

### Compute routing
- **Spectral 3-region:** ~4 VL fits × ~2s = seconds → **laptop OK** (`@pytest.mark.vl`).
- **Task / latent_circuit:** a single task or latent VL fit is minutes (task cells ran 800-12,000s
  for 10 seeds → ~80-1200s per seed; latent ~170s per seed). A 4-refit brute-force on task/latent
  **exceeds 3 min → route to M3** per the project rule.
- **Recommendation:** make the *gated* VLBMR-02 test **spectral 3-region on laptop** (`@pytest.mark.vl`,
  <3 min), and offer an **optional cluster variant** (`@pytest.mark.slow`, latent_circuit) submitted
  via the existing sbatch pattern for a stronger cross-model confirmation. Keep `test_bmr_vs_elbo.py`'s
  SVI version as a separate historical/secondary check or upgrade it to VL.

---

## VLBMR-03: Exploratory Posterior Tempering Calibration

### Consuming Phase 30 coverage output (concrete path)
`benchmarks/results/recovery_matrix.json` → `rows[i]` has `variant`, `n_regions`, `snr`,
`coverage_95`, `shrinkage_median`. Verified values relevant to tempering:

| cell | variant | N | SNR | coverage_95 | shrinkage_median | regime |
|------|---------|---|-----|-------------|------------------|--------|
| 0-3 | spectral | 2/4 | 1/3 | **1.0** | 0.21 / 0.03 | well-calibrated |
| 4 | task | 2 | 1 | 0.75 | 0.68 | mild under-coverage |
| 5 | task | 2 | 3 | 0.875 | 0.59 | borderline |
| 6,7 | task | 4 | 1/3 | **0.0** | NaN | **strongest overconfidence — the stress case** |
| 8,9 | latent_circuit | 4 | 1/3 | null | 0.50 / 0.35 | coverage not computed |

**Calibration logic (the exploratory loop):** Under-coverage at 95% nominal means the posterior is
too tight by a factor; tempering inflates `Cp` by T>1 to push empirical coverage toward 0.95. The
natural calibration is a **coverage-matching sweep**: for a target cell (use task N=4, coverage=0.0
— the strongest Laplace-overconfidence regime, the natural stress case per the phase brief), sweep
`T ∈ {1, 2, 5, 10, 20, 50, ...}`, recompute empirical 95% coverage from tempered samples
(`compute_coverage_from_samples` in `benchmarks/metrics.py`, already used by the runners), and pick
the smallest T that brings coverage into a band around 0.95 (e.g. [0.90, 0.98]). Then apply that T
via `temper_vl_posterior` to the A_free `Cp`, re-run `rank_connections`, and report tempered vs
untempered ranking + separation_gap side-by-side.

Note: shrinkage_median ≈ std_post/std_prior. A first-order T estimate is `T ≈ (target_shrinkage /
observed_shrinkage)^2` if you prefer a shrinkage-target heuristic over coverage-matching, but
**coverage-matching against the Phase 30 `coverage_95` is the calibration the phase brief asks for**
— prefer it; report the shrinkage heuristic only as a cross-check.

### Hard requirements (from phase goal + Pitfall C2)
- **PD-safe:** every tempered covariance MUST pass `torch.linalg.cholesky`. Achieve this by routing
  ALL tempering through `temper_vl_posterior` (it raises loud on non-PD). Assert no `ValueError`
  for the chosen T; if low-SNR cells push the tempered reduced posterior non-PD inside BMR
  (`bayesian_model_reduction` returns -inf), that is the expected C2 failure mode — surface it,
  do not mask it.
- **Side-by-side reporting:** emit both untempered and tempered `rank_connections` output (ranked
  list + separation_gap + separation_after_rank) in the result JSON / report. The untempered
  ranking is PRIMARY; tempered is exploratory annotation.
- **NEVER a pass/fail gate on absolute ΔF.** The tempering test may assert mechanical properties
  (Cholesky PD holds; tempered separation_gap is finite; top-K under tempering still matches true
  edges when SNR is adequate) but MUST NOT assert "absolute ΔF > threshold". Document "exploratory,
  not a headline claim" in the test docstring and the report.
- **Validate the chosen T on a held-out condition** (Pitfall C2a): a T tuned on task-N4 must be
  checked on at least one other cell (e.g. task-N2 or spectral) to show it does not catastrophically
  over-inflate (which would destroy the relative ranking). Report the cross-condition behaviour;
  do not claim a universal schedule.

### Compute routing
The tempering calibration **consumes already-computed Phase 30 posteriors if they are persisted**.
Check whether `recovery_matrix.json` stores per-seed `Ep`/`Cp` — it does NOT (it stores aggregate
metrics + `a_true_list`/`a_inferred_list` only). Therefore VLBMR-03 must **re-fit one representative
seed** of the target cell to obtain `result.sigma_post`, then do the (analytic, fast) tempering
sweep on that single posterior.
- For the **task N=4** stress cell, a single VL fit is **minutes → route to M3** (or run the
  calibration on a persisted posterior if Phase 31 first dumps one).
- **Recommended decomposition:** a small **cluster script** (mirror
  `cluster/scripts/lc_vl_bmr_selection.py` + `recovery_matrix_cell.py`) does the single-seed task-N4
  (and a spectral hold-out) VL fit, runs the T-sweep + tempered/untempered ranking, and writes a
  JSON to `cluster/results/`. A **laptop-fast unit test** (`@pytest.mark.vl`) covers the pure
  tempering mechanics (PD guard, coverage recomputation function, T-selection logic) on a synthetic
  or spectral posterior so the logic is tested without the slow task fit.

---

## Don't Hand-Roll

| Problem | Don't build | Use instead |
|---------|-------------|-------------|
| Full A_free posterior covariance | Reconstruct from std/samples | `result.sigma_post[:N*N, :N*N]` (A_free packed first) |
| Single-prune ranking + separation gap | Loop BMR + sort manually | `rank_connections(...)` (does exactly K calls, returns `separation_gap`) |
| PD guard on tempered covariance | Manual Cholesky try/except | `temper_vl_posterior(...)` (raises loud, names shape+factor) |
| Reduced prior for a pruned edge | Manually edit cov | `make_reduced_prior_zero_connection(...)` |
| Coverage from samples | New CI computation | `benchmarks.metrics.compute_coverage_from_samples(A_true, samples, ci_level=0.95)` |
| Ground truth + VL fit per variant | New simulate→fit code | reuse `benchmarks/recovery_matrix_grid.py` `_run_spectral_cell` / `_build_ground_truth` patterns |
| Cluster SLURM glue | New sbatch from scratch | mirror `cluster/scripts/recovery_matrix_cell.py` + `cluster/sbatch/recovery_matrix_sweep.sbatch` |
| Rank correlation | Manual tau | `scipy.stats.spearmanr` / `kendalltau` (scipy 1.16.3 installed) |

---

## Common Pitfalls (Phase 31-specific — guard each)

### Pitfall C1: Laplace overconfidence breaks absolute ΔF (CONFIRMED, job 55772525)
- All single-prune ΔF are deeply negative (-100 to -200,000), present and absent alike.
- **Guard:** ALL plans use `rank_connections` (relative). NEVER assert `ΔF > threshold`. Report
  `separation_gap` but do not gate on its magnitude. (PITFALLS.md C1, lines 17-49.)

### Pitfall C2: Tempering introduces calibration hazards (PITFALLS.md C2, lines 52-79)
- (a) Optimal T is data/model-dependent — a T tuned on one cell over-/under-corrects elsewhere.
- (b) Tempered Cp is inconsistent with the already-computed free energy — tempered ΔF is no longer
  a clean log-evidence change. **Treat tempered BMR as exploratory only.**
- (c) At low SNR the posterior is already diffuse; tempering can push the reduced posterior
  non-PD → `bayesian_model_reduction` returns -inf.
- **Guard:** route through `temper_vl_posterior` (loud PD); validate T on held-out conditions;
  hard-assert Cholesky succeeds in the harness; report tempered+untempered side-by-side.

### Pitfall S4: Row-major vs column-major index mapping (PITFALLS.md S4, lines 175-202)
- A[i,j] → flat index `i*N + j` (C-order, j varies fastest), matching `.reshape(-1)`. The
  off-diagonal/prunable index list and "true edge" set MUST use this convention. A transposed
  mapping silently swaps A[i,j]↔A[j,i] and corrupts the recovery assertion.
- **Guard:** reuse `lc_vl_bmr_selection.py`'s `OFFDIAG`/index expression verbatim; add a tiny
  assertion that the constructed true-edge set round-trips through `i*N+j`.

### Pitfall R2 (masked structural zeros): when reading true edges off a generated A
- `sign(0)=0` and a near-zero posterior mean make naive present/absent classification fragile.
- **Guard:** define "true present edge" as `|A_true[i,j]| > 0.01` (the project-standard masked
  threshold); better, construct a sparse ground truth so present/absent is by construction.

### Pitfall N1 (precision OOM for task DCM): only relevant if VLBMR-02/03 touch task
- Task VL precision is `(T*N, T*N)` identity; at fine dt it is intractable.
- **Guard:** any task fit uses `dt >= 0.1` (the runners assert this, `recovery_matrix_grid.py:471`).

---

## Recommended Plan Decomposition

Three plans. 31-01 and 31-02 are independent (both depend only on existing BMR/VL code) → **Wave 1
(parallel)**. 31-03 depends on Phase 30 output existing (it does) and conceptually builds on the
ranking harness → **Wave 2** (can technically run parallel but is cleaner after 31-01 lands the
ranking-harness conventions).

### Plan 31-01 — VLBMR-01: Relative-ranking recovery harness (PRIMARY)
- **Creates:** `tests/test_bmr_vlbmr01_recovery.py` (or extend `test_bmr_rank_connections.py`) — a
  `@pytest.mark.vl` test that VL-fits a sparse-ground-truth **spectral** circuit (N=2 and N=4,
  ≥5 seeds), slices `result.sigma_post[:N*N,:N*N]`, runs `rank_connections` over off-diagonal
  indices, asserts top-K == true edges, asserts `separation_gap > 0` and reports it.
- **Optionally:** harden `cluster/scripts/lc_vl_bmr_selection.py` to use `rank_connections`
  (separation_gap, not ratio) for the latent-circuit secondary confirmation, and/or add a small
  reusable helper `benchmarks/bmr_recovery.py` building the prior/posterior tensors from a VL result.
- **Routing:** LAPTOP (`@pytest.mark.vl`, <3 min).
- **Guards:** C1 (relative only), S4 (index mapping), R2 (masked true edges).

### Plan 31-02 — VLBMR-02: BMR vs brute-force VL-refit agreement
- **Creates:** a VL-based agreement test (new `tests/test_bmr_vs_vl_refit.py`, or upgrade
  `tests/test_bmr_vs_elbo.py` from SVI to VL). Full-model VL fit + analytic BMR ΔF over ~3 reduced
  models vs brute-force VL refits (a_mask-zeroed). Assert relative ordering (present>absent) and
  worst-model agreement; report Spearman ρ.
- **Routing:** spectral 3-region gated test → LAPTOP (`@pytest.mark.vl`). Optional latent_circuit
  confirmation → M3 (`@pytest.mark.slow` + sbatch). Decision the planner must make explicit:
  laptop-spectral as the gate, cluster-latent as optional.
- **Guards:** S3 (ranking not absolute F), C1, N1 (dt≥0.1 if task).

### Plan 31-03 — VLBMR-03: Exploratory tempering calibration (NOT a headline claim)
- **Creates:** (a) a laptop `@pytest.mark.vl` unit test of tempering mechanics — coverage
  recomputation, T-selection from a coverage target, PD guard via `temper_vl_posterior`, tempered
  vs untempered ranking both finite; (b) a cluster script
  `cluster/scripts/bmr_tempering_calibration.py` (mirror `lc_vl_bmr_selection.py` +
  `recovery_matrix_cell.py`) + sbatch (mirror `recovery_matrix_sweep.sbatch`) that re-fits one
  representative seed of the **task N=4** stress cell (coverage=0.0) plus one held-out cell, runs the
  T-sweep against Phase 30 `coverage_95`, and writes a side-by-side JSON to `cluster/results/`.
- **Consumes:** `benchmarks/results/recovery_matrix.json` (coverage_95, shrinkage_median per cell).
- **Routing:** mechanics test LAPTOP; task-N4 re-fit + sweep → M3 (single fit is minutes).
- **Guards:** C2 (all sub-failure modes), PD-safe via `temper_vl_posterior`, held-out validation,
  "exploratory" documented, absolute ΔF NEVER a gate.

---

## Compute Routing Summary

| Work item | Per-fit time (measured) | Routing | Marker |
|-----------|------------------------|---------|--------|
| VLBMR-01 spectral N=2/N=4, ≥5 seeds | ~1.4-7.5s/fit | **Laptop** | `vl` |
| VLBMR-02 spectral 3-region, ~4 VL fits | ~2s/fit, <30s total | **Laptop** | `vl` |
| VLBMR-02 latent/task confirmation (optional) | 170-1200s/fit | **M3** | `slow` |
| VLBMR-03 tempering mechanics | analytic, ms | **Laptop** | `vl` |
| VLBMR-03 task-N4 stress re-fit + sweep | minutes/fit | **M3** | (cluster script) |

Cluster pattern (HIGH confidence — proven in repo): SLURM glue script under `cluster/scripts/`
mirroring `recovery_matrix_cell.py` (sys.path insert for repo root + `src`; env-driven knobs;
try/except → status="error" JSON; write to `cluster/results/`); sbatch mirroring
`recovery_matrix_sweep.sbatch` (`source cluster/lib/cluster_env.sh`; `activate_env
"actinf-py-scripts"`; `verify_torch`; **NEVER `pip install` inside an array job** — package found
via sys.path). Deploy via **Mutagen sync, never git** (project memory). Latent-circuit jobs require
the anchored `models/` Mutagen-ignore fix (per STATE.md it landed); spectral/task are unaffected.

---

## Open Questions

1. **Upgrade vs add for VLBMR-02:** `tests/test_bmr_vs_elbo.py` exists with SVI. The planner should
   decide: upgrade it in place to VL, or add a parallel `test_bmr_vs_vl_refit.py` and leave the SVI
   one as a secondary historical check. Recommend **add new VL test** (keeps the SVI cross-check)
   and mark the SVI one clearly as "SVI baseline, not the validated engine."
2. **VLBMR-01 substrate breadth:** spectral is the recommended primary. Whether to ALSO assert on
   latent_circuit (reusing `lc_vl_bmr_selection.py`'s known chain) as a second gated case, or leave
   it cluster-optional, is a planner call. Recommend spectral gated + latent cluster-optional.
3. **Tempering target band:** the phase brief says "calibrated against Phase 30 coverage output" but
   does not fix the acceptance band. Recommend coverage target 0.95 with an accepted band
   [0.90, 0.98]; the smallest T reaching the band is the reported temperature. Flag that this band
   is a choice, reported as exploratory, not a validated schedule (PITFALLS gap 2: no DCM precedent).
4. **Persisting posteriors:** Phase 30 did not save per-seed Ep/Cp. VLBMR-03 must re-fit. If the
   planner wants to avoid the task-N4 re-fit cost, a cheap option is to dump one task-N4
   `result.sigma_post` to disk in a tiny cluster job first, then run the (laptop-fast) tempering
   sweep on the saved matrix.

## Sources

### Primary (HIGH — all read this session)
- `src/pyro_dcm/model_selection/bmr.py` — full signatures (rank_connections:431, temper_vl_posterior:552, bayesian_model_reduction:27, make_reduced_prior_zero_connection:157)
- `src/pyro_dcm/inference/variational_laplace.py` — `extract_vl_posterior_generic` (no full-cov in output; sigma_post on result), A_free packed first (846-852, 901), result.sigma_post full block (771, 794, 1252, 1264)
- `cluster/scripts/lc_vl_bmr_selection.py` — near-complete VLBMR-01 reference (sigma_post[:N*N,:N*N], OFFDIAG, top-K==true chain)
- `benchmarks/recovery_matrix_grid.py` — per-variant VL fit drivers (spectral/task/latent), SNR injection, near-boundary exclusion, dt≥0.1
- `benchmarks/runners/spectral_vl.py` — spectral VL fit + coverage pattern
- `benchmarks/results/recovery_matrix.json` — Phase 30 coverage_95 / shrinkage per cell (consumed by VLBMR-03)
- `cluster/results/recovery_matrix_*.json` — per-cell elapsed_s (timing → routing)
- `cluster/scripts/recovery_matrix_cell.py`, `cluster/sbatch/recovery_matrix_sweep.sbatch`, `cluster/lib/cluster_env.sh` — proven cluster glue + no-pip rule
- `tests/test_bmr.py`, `test_bmr_circuit_selection.py`, `test_bmr_rank_connections.py`, `test_bmr_vs_elbo.py` — existing coverage (gaps identified)
- `.planning/research/v0.7.0/{PITFALLS,SUMMARY,FEATURES}.md` — C1/C2/S3/S4/R2/N1 pitfalls, Phase 31 scope, relative-only guidance

### Confidence
| Area | Level | Reason |
|------|-------|--------|
| Function signatures | HIGH | Read directly from source this session |
| Where full Cp lives | HIGH | sigma_post slicing confirmed in source + existing cluster script |
| Substrate/timing/routing | HIGH | Per-cell elapsed_s measured in cluster results |
| Tempering calibration approach | MEDIUM | No DCM precedent (PITFALLS gap 2); approach is sound but the band/schedule is a choice, reported exploratory |
| Pitfalls | HIGH | C1 has direct cluster-log evidence; all others from committed PITFALLS.md |

**Research date:** 2026-06-11
**Valid until:** stable (internal codebase; ~30 days)
