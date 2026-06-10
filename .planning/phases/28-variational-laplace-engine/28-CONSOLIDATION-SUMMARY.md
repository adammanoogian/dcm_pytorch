---
phase: 28
title: Variational Laplace Inference Engine (SPM12-grade) — retroactive consolidation
status: complete
kind: consolidation
created: 2026-06-10
milestone: v0.6.0
---

# Phase 28 — Variational Laplace Inference Engine (retroactive consolidation)

> **Why this document exists.** A substantial SPM12-grade Variational Laplace (VL)
> inference engine was built *after* the 2026-05-28 v0.6.0 consolidation but was never
> tracked in the ROADMAP — it landed under ad-hoc decimal commit prefixes (`21.1-*`,
> `21.2-*`, `21.3-02`, plus `vl`/`csd-precision`/`forward-model` fixes) that reference
> phases not in the roadmap. The 2026-06-10 milestone audit (`.planning/v0.6.0-AUDIT.md`)
> established that **this engine is what actually delivered v0.6.0's inference** (it closed
> Phase 20-05, which mean-field SVI could not). Per the audit decision, the already-built
> engine is documented here as a retroactive v0.6.0 phase; the forward-looking VL
> *validation matrix* + *real-data application* is reserved for v0.7.0.

## Goal (as-built)

Replace mean-field SVI with a Variational Laplace engine matching SPM12's `spm_nlsi_GN`,
returning a **full posterior covariance** (structured posterior) and a 3-term free energy,
generalized across forward models via a `ForwardModel` protocol, and make VL the default
DCM inference.

## What was built (commit inventory)

| Commit | What |
|--------|------|
| 1525b0a | Variational Laplace inference backend for spectral DCM (initial engine) |
| 42e9a69 | Replace SVI with VL in perturbation scripts and tests |
| ad00427 (21.1-01) | GPU-safe `a_mask` device alignment for CUDA |
| 6a0b531 (21.1-02) | Model/VL/simulator/tests updated for SPM12 noise dimensions |
| 7a5dc99 (21.1-03) | MAR round-trip integrated into `spectral_dcm_forward` |
| d9a717f (21.1-04) | Hemodynamic params wired into VL + Pyro model + tests |
| 0f154e4 (21.2-01) | CSD observation precision `Q` matching `spm_dcm_csd_Q` |
| 0b44326 (21.2-01) | Q-based observation precision into the VL E-step |
| f5ae127 (21.2-02) | ReML M-step (8 inner Fisher-scoring iterations) |
| 9dd3d2b (21.2-03) | SVD dimension reduction on parameter space (`spm_svd`) |
| 27b43fd (21.3-02) | `initial_p` + `n_reduced_params` VL API (multi-restart) |
| 079c1d9 | **Switch spectral DCM default inference SVI → VL** |
| 61e5c9d | Generalize VL via the `ForwardModel` protocol (model-agnostic engine) |
| e1934e1 | SPM12-compatible hyperprior + prior-mean parameters |
| e6059e2 | Remove `a_mask` from context before `build_result` |
| 64e326f | C-order CSD indexing matching PyTorch reshape |
| a064e69 | Analytical hemodynamic Jacobian + SPM12 finite-difference step size |
| e99f68c | `LatentCircuitForward` adapter (direct-obs + bilinear B, time-domain) |

## Deliverables (current state, `src/pyro_dcm/inference/`)

- **`variational_laplace.py`** — SPM12 `spm_nlsi_GN` engine: Gauss-Newton E-step, ReML
  M-step (8 Fisher-scoring iters), SVD parameter-space reduction, adaptive regularization
  (`spm_dx`), 3-term free energy (L1 data fit + L2 parameter KL + L3 hyperprior). Returns a
  **full posterior covariance**.
- **`forward_models.py`** — `ForwardModel` protocol + concrete impls:
  `SpectralDCMForward` (CSD/hemodynamic), `TaskDCMForward` (BOLD ODE),
  `LatentCircuitForward` (direct-obs bilinear, time-domain).
- **`csd_precision.py`** — `compute_csd_precision` (`spm_dcm_csd_Q`).

## Validation (what makes this "delivered")

- **`tests/test_variational_laplace_recovery.py`** — spectral DCM VL recovery.
- **`tests/test_vl_forward_model_protocol.py`** — protocol conformance + task-DCM VL recovery.
- **`tests/test_latent_circuit_vl.py`** — latent-circuit VL: recovers A/B signs, full
  (non-diagonal) covariance, R²>0.7.
- **Closed Phase 20-05** (jobs 56268248 / 56270544): A-RMSE 0.026, B-RMSE 0.0048 (vs SVI
  ~0.31), sign 1.00, CI coverage 1.00, pooled-R² 0.961, BMR structure recovery 3/3.

## Success criteria (retroactively verified TRUE)

1. ✅ VL engine returns a full posterior covariance and a 3-term free energy matching the
   SPM12 `spm_nlsi_GN` structure (E-step + ReML M-step + SVD reduction).
2. ✅ A single `ForwardModel`-generic engine drives spectral, task, and latent-circuit DCM
   (`_run_vl_generic` + three concrete forward models), each covered by a recovery test.
3. ✅ VL is the default inference for spectral DCM (079c1d9) and the path that closed the
   Phase 20-05 acceptance gate.
4. ✅ SPM12 fidelity details landed: `spm_dcm_csd_Q` observation precision, C-order CSD
   indexing, SPM12-compatible hyperpriors + prior means, analytical hemodynamic Jacobian +
   SPM12 finite-difference step size.

## Known limitations (→ deferred to v0.7.0)

- **Laplace overconfidence breaks *absolute* BMR pruning** (shrinkage ~0.001 → "prune if
  ΔF>0" never fires); *relative* evidence ranking is the robust signal. Todo:
  `vl-overconfidence-for-bmr` (posterior tempering for BMR).
- **No systematic VL validation matrix** (recovery × N × SNR, SPM12 cross-check,
  calibration from the full covariance) — reserved for v0.7.0 Phase B.
- **VL precision matrix is intractable at fine dt for long time-domain runs** (e.g. dt=0.01,
  100s → 10⁴ pts → 4×10⁴ dense inverse); latent-circuit runs use dt=0.1 over slow dynamics.

## Pointers

- Audit: `.planning/v0.6.0-AUDIT.md`
- Forward-looking scope: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md` (engine rows now
  consolidated here; that draft's Phases B–E remain the v0.7.0 proposal).
