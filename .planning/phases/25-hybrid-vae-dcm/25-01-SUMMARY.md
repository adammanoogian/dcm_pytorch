---
phase: 25-hybrid-vae-dcm
plan: 01
subsystem: guides
backfilled: 2026-06-09
tags: [vae, dcm, encoder, parameter-packing, cnn, primitives]
dependency-graph:
  requires: []
  provides: ["LatentCircuitDCMPacker", "DCMEncoderNet"]
  affects: ["25-02", "25-03", "25-04"]
tech-stack:
  added: []
  patterns: ["sparse-mask-packing", "dual-head-encoder", "near-zero-init"]
key-files:
  created:
    - src/pyro_dcm/guides/dcm_encoder_net.py
    - tests/test_hybrid_vae_dcm_primitives.py
  modified:
    - src/pyro_dcm/guides/parameter_packing.py
    - src/pyro_dcm/guides/__init__.py
decisions:
  - id: "25-01-D1"
    description: "LatentCircuitDCMPacker uses sparse packing (only non-zero a_mask/c_mask entries) and exposes total_dim as a property, not the dense n_features attribute used by TaskDCMPacker"
  - id: "25-01-D2"
    description: "pack() takes positional named args (a_free, c, x0, noise_prec) rather than a single params dict; unpack() returns a dict and operates on unbatched vectors only"
  - id: "25-01-D3"
    description: "DCMEncoderNet uses a single fc_out head producing 2*latent_dim split into loc/scale_raw, ReLU activation, kernel_size=3, and default hidden_channels [32, 64, 128]; no float64 cast"
metrics:
  duration: "~25 minutes"
  completed: "2026-05-27"
---

# Phase 25 Plan 01: Hybrid VAE-DCM Primitives Summary

Foundational building blocks for the hybrid VAE-DCM: a sparse-mask parameter
packer (`LatentCircuitDCMPacker`) that includes initial conditions `x0`, and a
1D-CNN recognition network (`DCMEncoderNet`) mapping observed timeseries to
approximate-posterior `(z_loc, z_scale)`.

## What Was Done

### Task 1: LatentCircuitDCMPacker + DCMEncoderNet

1. **`LatentCircuitDCMPacker`** (appended to `parameter_packing.py` as the third
   packer class after `TaskDCMPacker` and `SpectralDCMPacker`):
   - `__init__(n_regions, n_inputs, a_mask, c_mask)` — stores boolean masks and
     precomputes sparse counts `_n_a = a_mask.sum()`, `_n_c = c_mask.sum()`.
   - `total_dim` **property** = `n_a + n_c + n_regions + 1` (sparse: only
     non-zero mask entries are packed, plus `x0` and one noise term).
   - `pack(a_free, c, x0, noise_prec)` — packs
     `[a_free[a_mask], c[c_mask], x0.flatten(), log(noise_prec)]`.
     `noise_prec` stored in log-space (same contract as `TaskDCMPacker`).
   - `unpack(flat)` — reverse of pack; rebuilds dense `(N, N)` A and `(N, M)` C
     (zeros where mask is False), returns dict
     `{"A_free", "C", "x0", "noise_prec"}` with `noise_prec` left in log-space.
   - `fit_standardization(samples)`, `standardize(flat)`, `unstandardize(z_std)`
     — per-element mean/std with `std` clamped to `min=1e-6`.

2. **`DCMEncoderNet(nn.Module)`** (new file `dcm_encoder_net.py`):
   - `__init__(n_regions, latent_dim, hidden_channels=[32, 64, 128])` — dynamic
     Conv1d backbone (`N -> 32 -> 64 -> 128`, kernel_size=3, padding=1, ReLU),
     `AdaptiveAvgPool1d(1)`, single `fc_out: Linear(final_channels, 2*latent_dim)`.
   - Output FC weights initialized near zero (`std=1e-3`), bias zeroed, so the
     initial `z_loc` is near the prior mean (standard VAE init).
   - `forward(x)` — accepts `(T, N)` unbatched or `(batch, T, N)` batched;
     transposes to channels-first, runs backbone + pool, splits `fc_out` into
     `z_loc` and `z_scale_raw`, returns `z_scale = softplus(z_scale_raw) + 1e-5`;
     squeezes the batch dim back out for unbatched input.

3. Updated `guides/__init__.py` to re-export `LatentCircuitDCMPacker` and
   `DCMEncoderNet` (both added to imports and `__all__`).

### Task 2: Unit tests

Created `tests/test_hybrid_vae_dcm_primitives.py` (261 lines, 12 tests across two
`pytest` classes with shared `masks_4x1` / `full_masks_3x1` fixtures):

| # | Test | Status |
|---|------|--------|
| 1 | test_packer_total_dim | PASS |
| 2 | test_packer_total_dim_full | PASS |
| 3 | test_packer_round_trip | PASS |
| 4 | test_packer_round_trip_full | PASS |
| 5 | test_packer_standardization | PASS |
| 6 | test_packer_standardization_not_fitted | PASS |
| 7 | test_encoder_output_shapes | PASS |
| 8 | test_encoder_unbatched | PASS |
| 9 | test_encoder_initial_output_near_zero | PASS |
| 10 | test_encoder_scale_positive | PASS |
| 11 | test_encoder_custom_hidden_channels | PASS |
| 12 | test_encoder_variable_length | PASS |

All 12 pass (per commit message: "12/12 tests pass covering round-trip,
standardization, shapes, positivity").

## Decisions Made

- **[25-01-D1] Sparse packing + `total_dim` property.** Unlike the dense
  `TaskDCMPacker` (which packs the full `(N, N)` A and exposes `n_features`),
  `LatentCircuitDCMPacker` packs only the non-zero `a_mask`/`c_mask` entries and
  exposes a `total_dim` property. This is the attribute downstream code must use
  — confirmed load-bearing by [25-02-D2], where Plan 02 wires `packer.total_dim`
  (not `n_features`) into the encoder's `latent_dim`.

- **[25-01-D2] Named-arg `pack`, dict-returning `unpack`, unbatched only.**
  `pack(a_free, c, x0, noise_prec)` takes positional named tensors rather than a
  params dict, and `unpack` operates on a single `(total_dim,)` vector (no batch
  dimension handling). This diverges from the PLAN's dict-in / batch-unpack
  interface but is internally consistent and round-trips exactly.

- **[25-01-D3] Single-head encoder, ReLU, no float64 cast.** The encoder uses one
  `fc_out` producing `2*latent_dim` (split into loc/scale) rather than the PLAN's
  separate `fc_loc`/`fc_scale` heads, ReLU rather than ELU, kernel_size=3 rather
  than 5, no BatchNorm, and does not call `self.double()`. Near-zero output init
  (Pitfall-1 mitigation) is preserved.

## Deviations from Plan

The implementation diverged from the PLAN on several interface details, all
self-consistent and test-covered:

- **Packer is sparse, not dense.** PLAN specified `n_features = N*N + N*M + N + 1`
  (dense, full A); implementation uses `total_dim = n_a + n_c + N + 1` (sparse,
  masked entries only). For the PLAN's N=4, M=1 fully-connected example the dense
  count is 25; the sparse `total_dim` depends on the mask.
- **No bilinear `NotImplementedError` refusal** and **no `B_free_*` guard** —
  `pack` takes only the four named DCM tensors, so the bilinear-refusal test
  from the PLAN was dropped (the shipped test suite has no bilinear test).
- **No batch-dim unpack** in the packer (PLAN asked for `z.shape[:-1]` support);
  encoder `forward`, however, does handle both batched and unbatched input.
- **Encoder architecture differences** as in [25-01-D3] (single head, ReLU,
  kernel_size=3, hidden_channels `[32, 64, 128]`, no BatchNorm, no float64).
- **Test count/names differ:** 12 tests shipped (not the PLAN's enumerated 10),
  organized into `TestLatentCircuitDCMPacker` and encoder test functions.

## Next Phase Readiness

Both primitives are importable from `pyro_dcm.guides` and verified by the unit
suite. Plan 02 consumed them directly to build `hybrid_vae_dcm_model` and
`HybridVAEDCMGuide`, wiring `packer.total_dim` into the encoder `latent_dim`
(see [25-02-D2]).
