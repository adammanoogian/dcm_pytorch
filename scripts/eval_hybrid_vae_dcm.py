"""Eval-only recovery for a trained hybrid VAE-DCM encoder checkpoint.

Loads a saved ``encoder_checkpoint.pt``, regenerates the *identical* held-out
test set (same seed + ``n_samples`` offset as the original training run), runs
the encoder forward, and recomputes recovery metrics -- crucially the
**masked** A sign recovery (|A_true| > 0.1), which the original
``recovery_report.json`` predates.

This closes the HVAE-02 audit gap (`.planning/v0.6.0-AUDIT.md`): the May-31
cluster run stored the *unmasked* sign recovery (0.4425) under the
``A_sign_recovery`` key because the masked fix (commit fbddc0e) landed
afterward. No retraining -- the checkpoint weights are loaded as-is.

Usage::

    python scripts/eval_hybrid_vae_dcm.py \
        --checkpoint results/hybrid_vae_dcm/encoder_checkpoint.pt \
        --output-dir results/hybrid_vae_dcm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/hybrid_vae_dcm/encoder_checkpoint.pt",
        help="Path to encoder_checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hybrid_vae_dcm",
        help="Where to write eval_recovery_report.json.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=50,
        help="Held-out test examples to regenerate (default: 50, matches run).",
    )
    return parser.parse_args()


def _mean(vals: list[float]) -> float:
    """Mean ignoring an empty list."""
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(vals: list[float]) -> float:
    """Compute population standard deviation; NaN for fewer than 2 values."""
    if len(vals) < 2:
        return float("nan")
    mu = _mean(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))


def main() -> None:
    """Load checkpoint, regenerate held-out set, recompute masked recovery."""
    args = parse_args()

    import torch

    from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
    from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
    from pyro_dcm.models.hybrid_vae_dcm import (
        generate_synthetic_vae_dataset,
        masked_sign_recovery,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("eval_hvae")

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    n_regions = int(checkpoint["n_regions"])
    n_inputs = int(checkpoint["n_inputs"])
    latent_dim = int(checkpoint["latent_dim"])
    seed = int(checkpoint["seed"])
    n_samples = int(checkpoint["n_samples"])
    duration = float(checkpoint["duration"])
    dt = float(checkpoint["dt"])
    log.info(
        "Checkpoint: N=%d M=%d latent_dim=%d seed=%d n_samples=%d "
        "duration=%.2f dt=%.3f",
        n_regions, n_inputs, latent_dim, seed, n_samples, duration, dt,
    )

    # ------------------------------------------------------------------
    # Reconstruct the IDENTICAL held-out set: same seed + n_samples offset.
    # generate_synthetic_vae_dataset seeds once and draws sequentially, so the
    # test slice [n_samples : n_samples + n_test] requires regenerating the
    # leading n_samples training examples to reproduce the RNG stream exactly.
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    total_samples = n_samples + args.n_test
    log.info("Regenerating %d examples to recover held-out slice...",
             total_samples)
    t0 = time.time()
    all_data = generate_synthetic_vae_dataset(
        n_samples=total_samples,
        n_regions=n_regions,
        n_inputs=n_inputs,
        duration=duration,
        dt=dt,
        seed=seed,
    )
    test_data = all_data[n_samples:n_samples + args.n_test]
    log.info("Regenerated %d examples (%.1fs); held-out test = %d.",
             len(all_data), time.time() - t0, len(test_data))
    if not test_data:
        raise RuntimeError(
            f"Empty held-out set: regenerated {len(all_data)} examples but the "
            f"slice [{n_samples}:{n_samples + args.n_test}] is empty. "
            f"Expected >= {n_samples + 1} valid samples (ODE divergence?)."
        )

    # ------------------------------------------------------------------
    # Rebuild packer with the checkpoint's standardization (do NOT refit) and
    # the encoder with the saved weights.
    # ------------------------------------------------------------------
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.ones(n_regions, n_inputs, dtype=torch.float64)
    packer = LatentCircuitDCMPacker(n_regions, n_inputs, a_mask, c_mask)
    packer.mean_ = checkpoint["packer_mean"]
    packer.std_ = checkpoint["packer_std"]

    encoder_net = DCMEncoderNet(n_regions, latent_dim).double()
    encoder_net.load_state_dict(checkpoint["encoder_state_dict"])
    encoder_net.eval()

    # ------------------------------------------------------------------
    # Eval loop -- identical prediction logic to train_hybrid_vae_dcm.py:258+.
    # ------------------------------------------------------------------
    per_example: list[dict[str, float]] = []
    with torch.no_grad():
        for ex in test_data:
            z_loc, _z_scale = encoder_net(ex["observed"])
            z = packer.unstandardize(z_loc)
            params_pred = packer.unpack(z)

            A_true, A_pred = ex["A_free"], params_pred["A_free"]
            C_true, C_pred = ex["C"], params_pred["C"]
            x0_true, x0_pred = ex["x0"], params_pred["x0"]

            a_sign_masked = masked_sign_recovery(A_pred, A_true)
            a_sign_unmasked = float(
                (torch.sign(A_pred) == torch.sign(A_true)).float().mean().item()
            )
            per_example.append({
                "A_free_rmse": float((A_pred - A_true).pow(2).mean().sqrt()),
                "A_sign_recovery_masked": a_sign_masked,
                "A_sign_recovery_unmasked": a_sign_unmasked,
                "C_rmse": float((C_pred - C_true).pow(2).mean().sqrt()),
                "x0_rmse": float((x0_pred - x0_true).pow(2).mean().sqrt()),
            })

    masked = [e["A_sign_recovery_masked"] for e in per_example
              if not math.isnan(e["A_sign_recovery_masked"])]
    unmasked = [e["A_sign_recovery_unmasked"] for e in per_example]
    rmses = [e["A_free_rmse"] for e in per_example]

    report = {
        "checkpoint": str(ckpt_path),
        "n_test": len(per_example),
        "A_free_rmse_mean": _mean(rmses),
        "A_free_rmse_std": _std(rmses),
        "A_sign_recovery_masked_mean": _mean(masked),
        "A_sign_recovery_masked_std": _std(masked),
        "A_sign_recovery_unmasked_mean": _mean(unmasked),
        "A_sign_recovery_unmasked_std": _std(unmasked),
        "hvae_02_threshold": 0.6,
        "hvae_02_masked_pass": _mean(masked) > 0.6,
        "per_example": per_example,
    }
    out_path = Path(args.output_dir) / "eval_recovery_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print("HVAE-02 EVAL-ONLY RECOVERY (masked vs unmasked sign recovery)")
    print("=" * 60)
    print(f"  Test examples:            {report['n_test']}")
    rmse_m = report["A_free_rmse_mean"]
    rmse_s = report["A_free_rmse_std"]
    unmasked_m = report["A_sign_recovery_unmasked_mean"]
    masked_m = report["A_sign_recovery_masked_mean"]
    masked_s = report["A_sign_recovery_masked_std"]
    print(f"  A_free RMSE:               {rmse_m:.4f} +/- {rmse_s:.4f} "
          f"(sanity vs reported 0.076)")
    print(f"  A sign recovery (UNMASKED): {unmasked_m:.4f} "
          f"(should reproduce reported ~0.4425)")
    print(f"  A sign recovery (MASKED):   {masked_m:.4f} +/- {masked_s:.4f}")
    verdict = "PASS" if report["hvae_02_masked_pass"] else "FAIL"
    print(f"  HVAE-02 (masked > 0.6):     {verdict}")
    print(f"  Wrote: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
