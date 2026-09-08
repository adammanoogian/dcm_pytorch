# Status of the amortized inference paths (SBI/NPE, amortized guides, hybrid VAE-DCM)

**Decision (2026-09-08): amortized inference is parked. Variational Laplace is
the supported path.**

This mirrors the earlier SVI decision in `svi_status.md`. Parked means
*not maintained and not gating anything* — it is not deleted, not deprecated,
and the code still runs.

## Why

Simulation-based inference (NPE) for spectral DCM trains fine and is genuinely
fast (<1 s amortized), but it has never been **calibrated**. Simulation-based
calibration has failed every time it has been measured:

| run | configuration | SBC pass rate |
|---|---|---|
| job 55772094 (2026-06) | 50k sims, small nsf flow | **2/9** |
| — (capacity fix) | larger flow actually wired through | still failing; bias merely moved between parameters |
| job 55235108 (2026-09-08) | 50k sims, 8 transforms, 128 hidden, **fixed forward** | **3/9** |

2/9 → 3/9 across a 200-trial KS test on 9 parameters is noise, not improvement.

## What the last run ruled out

This is the informative part, and it is a genuine negative result worth keeping.

Two real forward-model bugs were found and fixed in Sept 2026, both of which
had corrupted every SBI training simulation:

- `compute_transfer_function_hemodynamic` diagonalised a **defective** Jacobian
  for triangular / near-degenerate `A`, collapsing `|H|` to ~1e-18
  (fixed `007b686`). This bug had already forced the retraction of the Phase 31
  "feed-forward is unidentifiable" finding.
- `mar2csd` omitted SPM's `2/ns` normalisation, scaling every CSD by 1/4
  (fixed `bd2448d`).

The working hypothesis was that the SBC failure was an artifact of the first —
its "root cause #2" (stability-boundary degeneracy) pointed at exactly the
regime the eigendecomposition was breaking. **That hypothesis is refuted.**
With both bugs fixed and the flow capacity actually wired through, calibration
is essentially unchanged.

So the miscalibration is robust to flow capacity *and* to the forward-model
bugs. It is a property of NPE on this problem, not a downstream symptom.

## What this means in practice

| | Path |
|---|---|
| Supported inference | **`method="vl"`** — `run_variational_laplace`, `run_variational_laplace_generic` |
| Parked | `inference/sbi_*`, `guides/`, `models/hybrid_vae_dcm.py`, `models/amortized_wrappers.py` |

- **Do not use amortized posteriors to back a scientific claim.** They are not
  calibrated, and the SBI-03 acceptance gate has never been met.
- The code is kept: `scripts/compare_sbi_svi.py` depends on it, and the
  VL-vs-amortized contrast is itself a result.
- VL agrees with SPM12 to **1.90 nats** (0.37% relative) as of `d77ab53`, which
  is the standard amortized inference would have to meet.

## Known, unfixed, and recorded

A concrete defect found while running the final job, worth fixing if this is
ever picked up — and independent of the calibration question:

The SBI input features are **~2x redundant with ~100 dead dimensions**. `sbi`
warns that ~100 of ~576 inputs are structurally constant, all in the imaginary
block. Two causes: a CSD's diagonal is real by construction, so the imaginary
parts of the auto-spectra are identically zero; and the CSD is Hermitian
(`CSD[w,i,j] = conj(CSD[w,j,i])`), so feeding all `N^2` complex entries supplies
every off-diagonal twice. Only the real diagonal plus the upper triangle is
independent. This is *not* an explanation for the SBC failure — `sbi` zeroes the
constant dimensions — but it wastes embedding capacity.

## What would reopen this

- A calibrated amortized posterior (SBC passing at nominal coverage) on a
  problem where VL is too slow to be practical.
- A concrete diagnosis of *why* NPE miscalibrates here that survives testing —
  the stability-boundary/degeneracy story is the leading candidate but has now
  lost its supporting evidence, since the regime it blamed was also the regime
  the (now-fixed) eigendecomposition bug was corrupting.
