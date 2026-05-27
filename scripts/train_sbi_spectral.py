#!/usr/bin/env python
"""Train NPE density estimator for spectral DCM and validate via SBC.

Trains a Neural Posterior Estimation (NPE) model using the ``sbi`` library
on simulated cross-spectral densities from the spectral DCM forward model.
After training, runs Simulation-Based Calibration (SBC) to validate
posterior calibration, and benchmarks amortized inference speed.

Outputs:
    - ``estimator.pt``: Trained posterior object (pickled via torch.save)
    - ``sbc_ranks.pt``: SBC rank statistics tensor
    - ``training_metadata.pt``: Training configuration and timing

References
----------
Cranmer, Brehmer & Louppe (2020). The frontier of simulation-based
    inference. PNAS 117(48), 30055-30062.
Tejero-Cantero et al. (2020). sbi: A toolkit for simulation-based
    inference. JOSS 5(52), 2505.
Goncalves et al. (2025). VBI: Amortized Bayesian inference for
    neuroimaging. eLife.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

try:
    import sbi  # noqa: F401
except ImportError:
    print(
        "sbi package required. Install with: pip install 'sbi>=0.22'",
        file=sys.stderr,
    )
    sys.exit(1)

from scipy import stats

from pyro_dcm.forward_models.spectral_transfer import default_frequency_grid
from pyro_dcm.inference.sbi_diagnostics import run_sbc_validation
from pyro_dcm.inference.sbi_embedding import CSDEmbeddingNet
from pyro_dcm.inference.sbi_spectral import (
    make_spectral_dcm_prior,
    make_spectral_dcm_simulator,
    train_npe,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train NPE for spectral DCM and validate via SBC",
    )
    parser.add_argument(
        "--n-regions", type=int, default=3,
        help="Number of brain regions (default: 3)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=50_000,
        help="Number of training simulations (default: 50000)",
    )
    parser.add_argument(
        "--n-sbc", type=int, default=200,
        help="Number of SBC validation runs (default: 200)",
    )
    parser.add_argument(
        "--n-freqs", type=int, default=32,
        help="Number of frequency bins (default: 32)",
    )
    parser.add_argument(
        "--tr", type=float, default=2.0,
        help="Repetition time in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=64,
        help="Embedding network output dimension (default: 64)",
    )
    parser.add_argument(
        "--hidden-features", type=int, default=128,
        help="Hidden features in density estimator (default: 128)",
    )
    parser.add_argument(
        "--num-transforms", type=int, default=5,
        help="Number of NSF transforms (default: 5)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200,
        help="Maximum training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/sbi_spectral",
        help="Output directory (default: results/sbi_spectral)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run NPE training, SBC validation, and speed benchmark."""
    args = parse_args()
    N = args.n_regions
    F = args.n_freqs

    # Set seed
    torch.manual_seed(args.seed)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SBI Spectral DCM: NPE Training + SBC Validation")
    print("=" * 60)
    print(f"  Regions:       {N}")
    print(f"  Frequencies:   {F}")
    print(f"  TR:            {args.tr}s")
    print(f"  Simulations:   {args.n_sims}")
    print(f"  SBC trials:    {args.n_sbc}")
    print(f"  Max epochs:    {args.max_epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Seed:          {args.seed}")
    print(f"  Output:        {out_dir}")
    print()

    # --- Setup ---
    a_mask = torch.ones(N, N, dtype=torch.float64)
    freqs = default_frequency_grid(args.tr, F)
    obs_dim = 2 * F * N * N

    # Create simulator and prior
    simulator = make_spectral_dcm_simulator(N, freqs, a_mask)
    prior = make_spectral_dcm_prior(N, a_mask)

    # Create embedding network
    embedding_net = CSDEmbeddingNet(
        input_dim=obs_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_features,
    )

    # --- Train NPE ---
    print("Training NPE...")
    t0 = time.time()
    posterior = train_npe(
        simulator,
        prior,
        n_simulations=args.n_sims,
        embedding_net=embedding_net,
    )
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")
    print()

    # --- Save trained posterior ---
    estimator_path = out_dir / "estimator.pt"
    torch.save(posterior, estimator_path)
    print(f"  Saved estimator to {estimator_path}")

    # --- SBC Validation ---
    print(f"\nRunning SBC validation ({args.n_sbc} trials)...")
    t0_sbc = time.time()
    sbc_result = run_sbc_validation(
        posterior,
        simulator,
        prior,
        n_trials=args.n_sbc,
        n_posterior_samples=1000,
    )
    sbc_time = time.time() - t0_sbc
    print(f"  SBC complete in {sbc_time:.1f}s")

    # Save SBC ranks
    sbc_path = out_dir / "sbc_ranks.pt"
    torch.save(sbc_result["ranks"], sbc_path)
    print(f"  Saved SBC ranks to {sbc_path}")

    # KS test for uniformity on each parameter's ranks
    n_params = sbc_result["ranks"].shape[1]
    ranks_np = sbc_result["ranks"].numpy()
    n_post = sbc_result["n_posterior_samples"]

    print("\n  SBC Rank Uniformity (KS test):")
    print(f"  {'Param':<12} {'KS stat':>10} {'p-value':>10} {'Status':>10}")
    print("  " + "-" * 46)

    n_pass = 0
    for i in range(n_params):
        # Normalize ranks to [0, 1] for KS test against uniform
        normalized = ranks_np[:, i] / n_post
        ks_stat, p_val = stats.kstest(normalized, "uniform")
        status = "PASS" if p_val > 0.05 else "FAIL"
        if p_val > 0.05:
            n_pass += 1
        print(f"  param_{i:<5d} {ks_stat:>10.4f} {p_val:>10.4f} {status:>10}")

    print(f"\n  SBC summary: {n_pass}/{n_params} parameters pass"
          f" (KS p > 0.05)")

    # --- Amortized inference speed test ---
    print("\nAmortized inference speed test...")

    # Generate a single test observation
    theta_test = prior.sample()
    x_test = simulator(theta_test)

    # Condition posterior on test observation
    conditioned = posterior.set_default_x(x_test.to(torch.float32))

    # Time 100 posterior.sample() calls
    n_speed_trials = 100
    times_list: list[float] = []
    for _ in range(n_speed_trials):
        t_start = time.time()
        _ = conditioned.sample((1000,))
        t_end = time.time()
        times_list.append(t_end - t_start)

    times_tensor = torch.tensor(times_list)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()

    print(f"  Per-call time: {mean_time:.4f} +/- {std_time:.4f}s"
          f" (n={n_speed_trials})")
    speed_pass = mean_time < 1.0
    print(f"  Speed criterion (<1s): {'PASS' if speed_pass else 'FAIL'}")

    # --- Save training metadata ---
    metadata = {
        "n_regions": N,
        "n_freqs": F,
        "tr": args.tr,
        "n_sims": args.n_sims,
        "n_sbc": args.n_sbc,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "embed_dim": args.embed_dim,
        "hidden_features": args.hidden_features,
        "num_transforms": args.num_transforms,
        "seed": args.seed,
        "train_time_s": train_time,
        "sbc_time_s": sbc_time,
        "sbc_n_pass": n_pass,
        "sbc_n_params": n_params,
        "inference_mean_time_s": mean_time,
        "inference_std_time_s": std_time,
        "speed_pass": speed_pass,
    }
    metadata_path = out_dir / "training_metadata.pt"
    torch.save(metadata, metadata_path)
    print(f"\n  Saved metadata to {metadata_path}")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Training time:     {train_time:.1f}s")
    print(f"  SBC time:          {sbc_time:.1f}s")
    print(f"  SBC pass rate:     {n_pass}/{n_params}")
    print(f"  Inference speed:   {mean_time:.4f}s/call")
    print(f"  Speed criterion:   {'PASS' if speed_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
