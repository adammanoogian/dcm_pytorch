r"""Train hybrid VAE-DCM on synthetic data (smoke test).

Self-contained training script for the hybrid VAE-DCM architecture.
Generates synthetic DCM parameter sets, simulates latent-state
trajectories, trains the encoder-decoder pair with KL annealing,
and evaluates amortized inference on held-out test examples.

This script is designed for **local smoke tests** with small
configurations (few samples, short duration, few epochs). Full-scale
training on the M3 cluster requires a separate sbatch script (Plan 04).

Usage
-----
Local smoke test::

    python scripts/train_hybrid_vae_dcm.py \
        --n_samples 20 --n_epochs 10 --duration 2.0

Full training (cluster)::

    See cluster/sbatch scripts for M3 submission.

References
----------
25-RESEARCH.md: Hybrid VAE-DCM architecture and training strategy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    """Run hybrid VAE-DCM training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train hybrid VAE-DCM on synthetic data.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of training samples (default: 100).",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="KL annealing warmup epochs (default: 10).",
    )
    parser.add_argument(
        "--n_regions",
        type=int,
        default=4,
        help="Number of latent dimensions (default: 4).",
    )
    parser.add_argument(
        "--n_inputs",
        type=int,
        default=1,
        help="Number of driving inputs (default: 1).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="ODE integration step size (default: 0.01).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/hybrid_vae_dcm",
        help="Output directory (default: results/hybrid_vae_dcm).",
    )
    args = parser.parse_args()

    # Lazy imports to keep --help fast
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyro
    import torch

    from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
    from pyro_dcm.guides.parameter_packing import (
        LatentCircuitDCMPacker,
    )
    from pyro_dcm.models.hybrid_vae_dcm import (
        HybridVAEDCMGuide,
        generate_synthetic_vae_dataset,
        hybrid_vae_dcm_model,
        train_hybrid_vae_dcm,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    pyro.clear_param_store()
    torch.manual_seed(args.seed)

    N = args.n_regions
    M = args.n_inputs

    # --- Generate training and test data ---
    n_test = max(10, args.n_samples // 10)
    n_train = args.n_samples
    log.info(
        "Generating %d train + %d test samples "
        "(N=%d, M=%d, duration=%.1fs)...",
        n_train,
        n_test,
        N,
        M,
        args.duration,
    )

    train_data = generate_synthetic_vae_dataset(
        n_train,
        n_regions=N,
        n_inputs=M,
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
    )
    test_data = generate_synthetic_vae_dataset(
        n_test,
        n_regions=N,
        n_inputs=M,
        duration=args.duration,
        dt=args.dt,
        seed=args.seed + 10_000,
    )
    log.info(
        "Generated %d train, %d test samples.",
        len(train_data),
        len(test_data),
    )

    # --- Create packer and fit standardization ---
    a_mask = train_data[0]["a_mask"]
    c_mask = train_data[0]["c_mask"]
    packer = LatentCircuitDCMPacker(N, M, a_mask, c_mask)

    packed_samples = []
    for ex in train_data:
        flat = packer.pack(
            ex["A"], ex["C"], ex["x0"], ex["noise_prec"],
        )
        packed_samples.append(flat)
    samples_tensor = torch.stack(packed_samples)
    packer.fit_standardization(samples_tensor)
    log.info(
        "Packer fitted: total_dim=%d, mean range=[%.3f, %.3f].",
        packer.total_dim,
        packer.mean_.min().item(),
        packer.mean_.max().item(),
    )

    # --- Create encoder and guide ---
    encoder = DCMEncoderNet(N, packer.total_dim).double()
    guide = HybridVAEDCMGuide(encoder, packer)

    n_params = sum(p.numel() for p in encoder.parameters())
    log.info("Encoder parameters: %d", n_params)

    # --- Train ---
    log.info(
        "Training for %d epochs (warmup=%d, lr=%g)...",
        args.n_epochs,
        args.warmup_epochs,
        args.lr,
    )
    result = train_hybrid_vae_dcm(
        hybrid_vae_dcm_model,
        guide,
        train_data,
        n_epochs=args.n_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        clip_norm=10.0,
        log_every=max(1, args.n_epochs // 10),
    )

    # --- Evaluate on test data ---
    log.info("Evaluating amortized inference on %d test examples...", len(test_data))
    guide.eval()
    sign_recoveries = []

    with torch.no_grad():
        for ex in test_data:
            z_loc, _ = encoder(ex["observed"])
            z = packer.unstandardize(z_loc)
            params = packer.unpack(z)
            a_pred = params["A_free"]
            a_true = ex["A"]

            # Sign recovery: fraction of elements where
            # sign(predicted) == sign(true)
            sign_match = (
                torch.sign(a_pred) == torch.sign(a_true)
            ).float().mean().item()
            sign_recoveries.append(sign_match)

    mean_sign_recovery = sum(sign_recoveries) / len(sign_recoveries)
    log.info(
        "Mean sign recovery on test set: %.3f "
        "(chance = 0.5 for binary sign)",
        mean_sign_recovery,
    )

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training losses plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    finite_losses = [
        (i, val)
        for i, val in enumerate(result["losses"])
        if torch.isfinite(torch.tensor(val))
    ]
    if finite_losses:
        epochs_f, losses_f = zip(*finite_losses, strict=True)
        ax1.plot(epochs_f, losses_f, "b-", linewidth=1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("ELBO Loss")
        ax1.set_title("Training Loss")
        ax1.set_yscale("symlog")

    ax2.plot(result["betas"], "r-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Beta")
    ax2.set_title("KL Annealing Schedule")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    log.info("Saved training curves to %s", output_dir / "training_curves.png")

    # Save encoder state dict
    torch.save(
        encoder.state_dict(),
        output_dir / "encoder_state_dict.pt",
    )
    log.info("Saved encoder to %s", output_dir / "encoder_state_dict.pt")

    # Save packer standardization stats
    torch.save(
        {"mean": packer.mean_, "std": packer.std_},
        output_dir / "packer_stats.pt",
    )

    # Save recovery summary
    summary_path = output_dir / "recovery_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"n_train: {n_train}\n")
        f.write(f"n_test: {n_test}\n")
        f.write(f"n_regions: {N}\n")
        f.write(f"n_inputs: {M}\n")
        f.write(f"n_epochs: {args.n_epochs}\n")
        f.write(f"warmup_epochs: {args.warmup_epochs}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"duration: {args.duration}\n")
        f.write(f"dt: {args.dt}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"mean_sign_recovery: {mean_sign_recovery:.4f}\n")
        f.write(
            f"final_loss: {result['losses'][-1]:.4f}\n",
        )
    log.info("Saved recovery summary to %s", summary_path)

    log.info("Done. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
