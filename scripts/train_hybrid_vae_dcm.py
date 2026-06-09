r"""Train hybrid VAE-DCM on synthetic latent-circuit data.

Generates a synthetic dataset of DCM parameter configurations with
simulated trajectories, trains a hybrid VAE-DCM (physics-informed
encoder + ODE decoder), evaluates amortized parameter recovery on
held-out test examples, and saves checkpoints and recovery reports.

Intended for local smoke testing (small n_samples, n_epochs) and
full-scale cluster training via ``cluster/sbatch_hybrid_vae_dcm.sh``.

Usage
-----
Local smoke test::

    python scripts/train_hybrid_vae_dcm.py \
        --n_samples 30 --n_epochs 10 --duration 2.0 \
        --output_dir results/hybrid_vae_dcm_smoke

Full-scale cluster::

    sbatch cluster/sbatch_hybrid_vae_dcm.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train hybrid VAE-DCM on synthetic DCM data.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of training examples (default: 100).",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50,
        help="Total training epochs (default: 50).",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10,
        help="KL annealing warmup epochs (default: 10).",
    )
    parser.add_argument(
        "--n_regions", type=int, default=4,
        help="Number of brain regions / latent dims (default: 4).",
    )
    parser.add_argument(
        "--n_inputs", type=int, default=1,
        help="Number of driving inputs (default: 1).",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Simulation duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="ODE integration step size (default: 0.01).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for ClippedAdam (default: 1e-3).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--n_test", type=int, default=50,
        help="Number of held-out test examples (default: 50).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/hybrid_vae_dcm",
        help="Output directory (default: results/hybrid_vae_dcm).",
    )
    parser.add_argument(
        "--save_encoder", action="store_true", default=True,
        help="Save encoder state_dict and packer stats (default: True).",
    )
    parser.add_argument(
        "--no_save_encoder", dest="save_encoder", action="store_false",
        help="Disable saving encoder checkpoint.",
    )
    parser.add_argument(
        "--save_recovery_report", action="store_true", default=True,
        help="Evaluate on test set and save recovery report (default: True).",
    )
    parser.add_argument(
        "--no_save_recovery_report", dest="save_recovery_report",
        action="store_false",
        help="Disable recovery report generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for hybrid VAE-DCM training."""
    args = parse_args()

    # Lazy imports (heavy torch/pyro imports after argparse)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
    from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
    from pyro_dcm.models.hybrid_vae_dcm import (
        HybridVAEDCMGuide,
        generate_synthetic_vae_dataset,
        hybrid_vae_dcm_model,
        masked_sign_recovery,
        train_hybrid_vae_dcm,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N, M = args.n_regions, args.n_inputs

    # ------------------------------------------------------------------
    # 1. Generate synthetic dataset
    # ------------------------------------------------------------------
    log.info(
        "Generating %d training + %d test examples "
        "(N=%d, M=%d, duration=%.1f, dt=%.3f)...",
        args.n_samples, args.n_test, N, M, args.duration, args.dt,
    )
    total_samples = args.n_samples + args.n_test
    all_data = generate_synthetic_vae_dataset(
        n_samples=total_samples,
        n_regions=N,
        n_inputs=M,
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
    )
    if len(all_data) < args.n_samples + 1:
        log.error(
            "Only %d valid samples generated (need %d + %d). "
            "Check for ODE divergence.",
            len(all_data), args.n_samples, args.n_test,
        )
        sys.exit(1)

    train_data = all_data[:args.n_samples]
    test_data = all_data[args.n_samples:args.n_samples + args.n_test]
    log.info(
        "Dataset: %d train, %d test examples.",
        len(train_data), len(test_data),
    )

    # ------------------------------------------------------------------
    # 2. Create packer and fit standardization
    # ------------------------------------------------------------------
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.ones(N, M, dtype=torch.float64)
    packer = LatentCircuitDCMPacker(N, M, a_mask, c_mask)

    # Pack all training parameters for standardization
    packed_samples = torch.stack([
        packer.pack(
            ex["A_free"], ex["C"], ex["x0"], ex["noise_prec"],
        )
        for ex in train_data
    ])
    packer.fit_standardization(packed_samples)
    log.info("Packer: total_dim=%d, standardization fitted.", packer.total_dim)

    # ------------------------------------------------------------------
    # 3. Create encoder and guide
    # ------------------------------------------------------------------
    encoder_net = DCMEncoderNet(N, packer.total_dim).double()
    guide = HybridVAEDCMGuide(encoder_net, packer)
    n_params = sum(p.numel() for p in encoder_net.parameters())
    log.info("Encoder: %d parameters.", n_params)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    log.info(
        "Training: %d epochs, %d warmup, lr=%.1e...",
        args.n_epochs, args.warmup_epochs, args.lr,
    )
    t_train_start = time.time()
    result = train_hybrid_vae_dcm(
        model_fn=hybrid_vae_dcm_model,
        guide=guide,
        train_data=train_data,
        n_epochs=args.n_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        dt=args.dt,
        log_every=max(1, args.n_epochs // 10),
    )
    train_duration = time.time() - t_train_start
    log.info("Training complete in %.1f seconds.", train_duration)

    # ------------------------------------------------------------------
    # 5. Save training loss curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result["losses"], label="ELBO loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("Hybrid VAE-DCM Training Loss")
    ax.legend()
    fig.tight_layout()
    loss_png = output_dir / "training_loss.png"
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)
    log.info("Training loss curve saved: %s", loss_png)

    # ------------------------------------------------------------------
    # 6. Save encoder checkpoint
    # ------------------------------------------------------------------
    if args.save_encoder:
        checkpoint = {
            "encoder_state_dict": encoder_net.state_dict(),
            "packer_mean": packer.mean_,
            "packer_std": packer.std_,
            "n_regions": N,
            "n_inputs": M,
            "latent_dim": packer.total_dim,
            "training_losses": result["losses"],
            "training_betas": result["betas"],
            "n_epochs": args.n_epochs,
            "warmup_epochs": args.warmup_epochs,
            "lr": args.lr,
            "seed": args.seed,
            "duration": args.duration,
            "dt": args.dt,
            "n_samples": args.n_samples,
            "train_duration_seconds": train_duration,
        }
        ckpt_path = output_dir / "encoder_checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        log.info("Encoder checkpoint saved: %s", ckpt_path)

    # ------------------------------------------------------------------
    # 7. Evaluate on test set and save recovery report
    # ------------------------------------------------------------------
    if args.save_recovery_report and len(test_data) > 0:
        log.info("Evaluating recovery on %d test examples...", len(test_data))
        encoder_net.eval()
        per_example: list[dict[str, float]] = []
        inference_times: list[float] = []

        with torch.no_grad():
            for ex in test_data:
                t0 = time.time()
                z_loc, z_scale = encoder_net(ex["observed"])
                t_infer = time.time() - t0
                inference_times.append(t_infer)

                # Unstandardize and unpack
                z = packer.unstandardize(z_loc)
                params_pred = packer.unpack(z)

                # True parameters
                A_true = ex["A_free"]
                C_true = ex["C"]
                x0_true = ex["x0"]

                # Predicted parameters
                A_pred = params_pred["A_free"]
                C_pred = params_pred["C"]
                x0_pred = params_pred["x0"]

                # Metrics
                a_rmse = float(
                    (A_pred - A_true).pow(2).mean().sqrt().item()
                )
                # Sign recovery: masked to genuinely non-zero connections.
                # Unmasked over all entries is meaningless because sign(0)=0 can
                # never match a non-zero prediction, so every structural-zero
                # entry is a guaranteed mismatch (this produced the spurious
                # HVAE-02 0.44). masked_sign_recovery uses |A_true| > 0.1.
                a_sign_match = masked_sign_recovery(A_pred, A_true)
                a_sign_unmasked = float(
                    (torch.sign(A_pred) == torch.sign(A_true))
                    .float().mean().item()
                )
                c_rmse = float(
                    (C_pred - C_true).pow(2).mean().sqrt().item()
                )
                x0_rmse = float(
                    (x0_pred - x0_true).pow(2).mean().sqrt().item()
                )

                per_example.append({
                    "A_free_rmse": a_rmse,
                    "A_sign_recovery": a_sign_match,
                    "A_sign_recovery_unmasked": a_sign_unmasked,
                    "C_rmse": c_rmse,
                    "x0_rmse": x0_rmse,
                })

        # Aggregate metrics
        n_test_actual = len(per_example)
        a_rmses = [e["A_free_rmse"] for e in per_example]
        # Masked sign recovery (the gate metric); drop NaN (no eligible entry).
        a_signs = [
            e["A_sign_recovery"] for e in per_example
            if not math.isnan(e["A_sign_recovery"])
        ]
        a_signs_unmasked = [
            e["A_sign_recovery_unmasked"] for e in per_example
        ]
        c_rmses = [e["C_rmse"] for e in per_example]
        x0_rmses = [e["x0_rmse"] for e in per_example]

        def _mean(vals: list[float]) -> float:
            return sum(vals) / max(1, len(vals))

        def _std(vals: list[float]) -> float:
            m = _mean(vals)
            if len(vals) < 2:
                return 0.0
            return math.sqrt(
                sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            )

        # KL divergence from last epoch
        # Re-run one forward pass with guide to capture KL
        final_kl = float("nan")
        try:
            import pyro

            pyro.clear_param_store()
            # Re-register guide parameters
            for name, param in encoder_net.named_parameters():
                pyro.param(
                    f"hybrid_vae_dcm_encoder${name}",
                    param.detach().clone(),
                )

            # Estimate KL from z_loc, z_scale (analytic for diagonal
            # Gaussian vs standard normal)
            kl_vals: list[float] = []
            with torch.no_grad():
                for ex in test_data[:10]:  # Sample for speed
                    z_loc, z_scale = encoder_net(ex["observed"])
                    kl = 0.5 * (
                        z_scale.pow(2) + z_loc.pow(2)
                        - 1 - 2 * z_scale.log()
                    ).sum()
                    kl_vals.append(float(kl.item()))
            final_kl = _mean(kl_vals)
        except Exception as e:
            log.warning("KL estimation failed: %s", e)

        report = {
            "n_test": n_test_actual,
            "per_example": per_example,
            "aggregated": {
                "A_free_rmse_mean": _mean(a_rmses),
                "A_free_rmse_std": _std(a_rmses),
                "A_sign_recovery_mean": _mean(a_signs),
                "A_sign_recovery_std": _std(a_signs),
                "A_sign_recovery_unmasked_mean": _mean(a_signs_unmasked),
                "C_rmse_mean": _mean(c_rmses),
                "C_rmse_std": _std(c_rmses),
                "x0_rmse_mean": _mean(x0_rmses),
                "x0_rmse_std": _std(x0_rmses),
            },
            "inference_timing": {
                "mean_seconds": _mean(inference_times),
                "std_seconds": _std(inference_times),
                "max_seconds": max(inference_times)
                if inference_times else 0.0,
            },
            "kl_divergence": {
                "final_epoch_mean": final_kl,
            },
            "training": {
                "n_epochs": args.n_epochs,
                "warmup_epochs": args.warmup_epochs,
                "final_loss": result["losses"][-1]
                if result["losses"] else float("nan"),
                "train_duration_seconds": train_duration,
            },
        }

        report_path = output_dir / "recovery_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Recovery report saved: %s", report_path)

        # Print summary table
        print("\n" + "=" * 60)
        print("  Hybrid VAE-DCM Recovery Summary")
        print("=" * 60)
        print(f"  Test examples:      {n_test_actual}")
        print(f"  A_free RMSE:        "
              f"{_mean(a_rmses):.4f} +/- {_std(a_rmses):.4f}")
        print(f"  A sign recovery:    "
              f"{_mean(a_signs):.4f} +/- {_std(a_signs):.4f}")
        print(f"  C RMSE:             "
              f"{_mean(c_rmses):.4f} +/- {_std(c_rmses):.4f}")
        print(f"  x0 RMSE:            "
              f"{_mean(x0_rmses):.4f} +/- {_std(x0_rmses):.4f}")
        print(f"  Inference time:     "
              f"{_mean(inference_times)*1000:.1f} ms/example")
        print(f"  KL divergence:      {final_kl:.4f}")
        print(f"  Final train loss:   {result['losses'][-1]:.4f}")
        print(f"  Training duration:  {train_duration:.1f}s")
        print("=" * 60)

    log.info("Done. Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
