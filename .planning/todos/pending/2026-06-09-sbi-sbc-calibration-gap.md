---
created: 2026-06-09T23:30
title: Phase 26 SBI-03 SBC calibration fails (2/9 params)
area: sbi-spectral-dcm
priority: medium
files:
  - cluster/logs/sbi_spectral_55772094.out
  - results/sbi_spectral_55772094/sbc_ranks.pt
  - src/pyro_dcm/inference/sbi_diagnostics.py
  - .planning/phases/26-sbi-spectral-dcm/26-02-SUMMARY.md
---

## Problem

The SBI (NPE) estimator for spectral DCM trained fine and is fast (<1s amortized), but
**simulation-based calibration FAILS: only 2/9 parameters pass the KS rank-uniformity test**
(p>0.05) on cluster job 55772094. SBI-03 requires nominal coverage. Posteriors are therefore
not calibrated -- the amortized SBI inference is not yet trustworthy.

## Diagnosis 2026-06-09 (failure mode characterised)

NOT under-training: the run used **50,000 simulations** (not 1000 — that was the prior-sampling
progress bar) and converged at 285 epochs. NOT global overconfidence either. The SBC rank
histograms (`results/sbi_spectral_55772094/sbc_ranks.pt`, 200×9) show **parameter-specific
bias**:

| param | A entry | mean-rank | KS | shape |
|-------|---------|-----------|-----|-------|
| 0 | A[0,0] (diag) | 0.32 | 0.29 | **biased high** (posterior over-shrinks self-inhibition) |
| 2 | A[0,2] | 0.65 | 0.28 | **biased low** |
| 1 | A[0,1] | 0.49 | 0.14 | underconfident (too wide) |
| 3–8 | — | 0.43–0.58 | 0.06–0.16 | mild biases (strict KS at 200 trials flags them) |

The 9 params are the full A_free (N=3, all-ones mask); **noise is fixed**, so SBC is purely over
connectivity.

### Root cause #1 (FIXED): dead flow-capacity args

`scripts/train_sbi_spectral.py` parsed `--num-transforms`, `--hidden-features`, `--max-epochs`
but **never passed them to `train_npe`** — `posterior_nn(model="nsf")` used the small nsf
defaults (5 transforms, 50 hidden). The diagonal parameterization `a_ii = -exp(A_free_ii)/2`
makes diagonal-param posteriors skewed, so flow capacity matters. **Fixed** (commit): plumbed
through; sbatch defaults now `NUM_TRANSFORMS=8`, `HIDDEN_FEATURES=128`, `MAX_EPOCHS=500`.
**Retrain experiment job 56274446 in flight** to test whether this restores calibration.

### Experiment result (job 56274446): capacity fix FALSIFIED — bias is structural

Retrained with the now-plumbed larger flow (8 transforms × 128 hidden, max 500 epochs;
converged 172 epochs, 3546s). SBC **still 2/9**, but the bias **redistributed**:

| param | original KS | new KS | |
|-------|-------------|--------|---|
| 0 (A[0,0]) | 0.29 FAIL | **0.052 PASS** | fixed |
| 1 | 0.14 FAIL | **0.080 PASS** | fixed |
| 4 (A[1,1]) | 0.12 (ok-ish) | **0.272 FAIL** | worse |
| 5,7,8 | mild | 0.22–0.27 FAIL | worse |

More capacity just moved which parameters absorb the bias rather than removing it ⇒ **flow
capacity is NOT the bottleneck; the miscalibration is structural.** (The plumbing fix is still
a correct improvement and is kept.)

### Root cause #2 (now the leading cause): stability-boundary / degeneracy

Off-diagonals up to ±3σ≈±0.375 push `A` past the stability boundary, where `eig_clamp` (`-1/32`)
clamps → different `A_free` map to near-identical CSD → locally unidentifiable params that NPE
cannot calibrate (the bias has to land *somewhere*, so capacity reshuffles it). Fixed noise may
add further degeneracy.

**Recommended next experiments (pick one; each ~1h cluster):**
1. Restrict the prior to the provably-stable region (reject/clamp `A_free` so `max Re eig(A) <
   eig_clamp` never binds) and re-run SBC — the cleanest test of the eig_clamp hypothesis.
2. Reparameterize into a stability-guaranteed coordinate (as VL does via parameterize_A on the
   diagonal, but extend to a globally-stable A factorization).
3. Validate against the VL posterior on the same synthetic CSD as a calibrated oracle to confirm
   the data genuinely constrains the params (i.e. the issue is NPE, not identifiability).

**Pragmatic stance:** amortized SBI for this spectral DCM is not yet calibrated and should not
back a scientific claim. **VL/SVI is the trustworthy, calibrated path** (Phase 20-05 closed via
VL) — SBI is an optional speed-up to revisit, not a blocker for v0.6.0 results.

## Audit disposition (2026-06-10) — DEFER to v0.7.0 (Phase D)

The milestone audit confirms SBI-03 **acceptance-failed** (SBC 2/9, structural) and SBI-04 (real
M/EEG demo) was never planned. Under the v0.6.0 scope-cut, SBI is **not a v0.6.0 deliverable** —
the calibrated inference path is VL. Carry both the structural-calibration fix (stable-region
prior / reparam) and the real-M/EEG demo to **v0.7.0 Phase D (SBI reconciliation)** in
`.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`, where the framing decision is "fix SBC vs. scope
SBI as an optional benchmark against calibrated VL."


---

## ⚠️ 2026-09-08: root cause #2 may be an artifact -- the training data was corrupted

`sbi_spectral.py:156` calls `spectral_dcm_forward(A, freqs, a, b, c,
eig_clamp=eig_clamp)` **without** passing `hemodynamic`, so it defaults to
`True`. Every one of the 50,000 training simulations therefore went through
`compute_transfer_function_hemodynamic`.

That function was **defective until `007b686` (2026-09-06)**: it diagonalised a
Jacobian that is defective (repeated eigenvalues, too few independent
eigenvectors) whenever `A` is triangular or near-degenerate, and `pinv` silently
dropped the deficient modes -- collapsing `|H|` to ~1e-18. Job 55772094 (the
SBC run) predates the fix.

**Why this bears directly on root cause #2.** That root cause says off-diagonals
up to +/-3 sigma push `A` past the stability boundary, where "different `A_free`
map to near-identical CSD => locally unidentifiable". But the near-degenerate
regime is *precisely* where the eigendecomposition was broken. An unknown subset
of the training simulations had a **corrupted** CSD, not a
genuinely-unidentifiable one, and the corruption was selective (it hit exactly
the boundary draws) rather than a uniform rescale that NPE could absorb.

This is the same mistake already retracted once in this repo: the Phase 31
"feed-forward is unidentifiable" finding was caused by this identical bug and is
recorded as retracted in `STATE.md`. The SBI "structural miscalibration"
conclusion rests on the same broken forward and must be re-tested before it is
believed.

(The `mar2csd` 2/ns error also applied, but that is a uniform factor of 4 on
every simulation and would not by itself break calibration.)

**Also note:** the SBI path does NOT use `compute_csd_precision`, so the
2026-09-08 Q-norm fix is irrelevant here.

## Re-run submitted

Same configuration as the capacity-fixed run (50k sims, 8 transforms, 128
hidden, 500 epochs, 200 SBC) on the FIXED forward. This is the cheapest
decisive test: if SBC still fails at ~2/9, root cause #2 stands and the
stable-region-prior experiments are warranted. If it improves materially, the
"structural" conclusion was an artifact and this todo needs rewriting.

Do not run the reparameterisation experiments until this lands -- they were
designed against a diagnosis that may not survive.


## DECISION 2026-09-08 (user): this re-run is the last attempt

If job 55235108 does not materially improve SBC calibration, **all amortized
inference work goes on the back burner** -- SBI/NPE, the amortized guides, and
the hybrid VAE-DCM. Not deleted, not deprecated: parked, the way SVI was in
`f39357e`.

The rationale is the same one that retired SVI as a quality gate. Variational
Laplace is the supported, calibrated engine; it backs v0.6.0, the v0.7.0
recovery matrix and the v0.8.0 ERP stack, and as of `d77ab53` it agrees with
SPM12 to 1.90 nats. Amortized inference was always an optional speed-up, and
this todo's own "pragmatic stance" already said it should not back a scientific
claim.

**Decision rule:**

- SBC materially improves (well beyond 2/9) -> the old "structural" diagnosis
  was an artifact of the broken forward; rewrite this todo and continue.
- SBC still fails at ~2/9 -> the diagnosis survives the forward fix, the problem
  is real and deeper than a bug, and amortized work is parked. Record the
  negative result properly (it is a genuine finding: NPE does not calibrate on
  this problem even with a correct forward) and stop spending on it.

Either way this is the last SBI compute until something changes upstream.
