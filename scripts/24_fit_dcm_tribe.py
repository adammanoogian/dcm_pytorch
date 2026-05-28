r"""Pipeline: ROI timeseries -> spectral DCM -> posterior A matrix.

Loads TRIBE v2 ROI timeseries from extraction script output, computes
empirical cross-spectral density, and fits a spectral DCM model using
multi-start SVI to recover the posterior effective connectivity matrix.

Usage
-----
    python scripts/24_fit_dcm_tribe.py \
        --input-npz results/phase24_tribe/tribe_roi_timeseries.npz \
        --output-dir results/phase24_tribe/ \
        --n-regions 6 \
        --num-steps 2000 \
        --n-restarts 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Run spectral DCM fitting on TRIBE v2 ROI timeseries."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit spectral DCM to TRIBE v2 ROI timeseries "
            "and extract posterior A matrix."
        ),
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to tribe_roi_timeseries.npz from extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase24_tribe/",
        help="Directory for output results.",
    )
    parser.add_argument(
        "--n-regions",
        type=int,
        default=6,
        help="Number of ROIs to include in DCM (subset).",
    )
    parser.add_argument(
        "--roi-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated ROI indices to select "
            "(e.g. '0,5,10,15,20,25'). "
            "If not specified, uses first N."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="Number of SVI optimization steps.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of multi-start SVI restarts.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for SVI optimizer.",
    )
    parser.add_argument(
        "--n-freqs",
        type=int,
        default=32,
        help="Number of frequency bins for CSD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # --- Import after argparse (heavy dependencies) ---
    try:
        from pyro_dcm import (
            create_guide,
            extract_posterior_params,
            parameterize_A,
            run_svi,
            spectral_dcm_model,
        )
        from pyro_dcm.forward_models.csd_computation import (
            compute_empirical_csd,
        )
    except ImportError:
        print(
            "pyro_dcm not installed. "
            "Run: pip install -e . from project root."
        )
        sys.exit(1)

    torch.manual_seed(args.seed)

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spectral DCM Fitting on TRIBE v2 ROI Timeseries")
    print("=" * 60)

    # --- Load data ---
    print(f"Loading: {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)
    roi_timeseries = data["roi_timeseries"]  # (T, n_rois)
    roi_names = list(data["roi_names"])

    print(
        f"  Full ROI timeseries: {roi_timeseries.shape} "
        f"({len(roi_names)} ROIs)"
    )

    # --- Select ROI subset ---
    if args.roi_indices is not None:
        indices = [
            int(i.strip())
            for i in args.roi_indices.split(",")
        ]
    else:
        indices = list(range(args.n_regions))

    n_regions = len(indices)
    roi_subset = roi_timeseries[:, indices]
    roi_names_selected = [roi_names[i] for i in indices]

    print(f"  Selected {n_regions} ROIs: {roi_names_selected}")
    print(f"  Subset shape: {roi_subset.shape}")

    # --- Compute empirical CSD ---
    # TRIBE v2 outputs at 1 Hz (TR = 1s), so fs = 1.0
    fs = 1.0
    fmax = fs / 2.0  # Nyquist
    freqs = np.linspace(
        0.01, fmax * 0.9, args.n_freqs
    )

    print(
        f"\nComputing empirical CSD "
        f"(fs={fs} Hz, {args.n_freqs} freqs)..."
    )
    csd_empirical = compute_empirical_csd(
        roi_subset, fs=fs, freqs=freqs
    )
    observed_csd = torch.from_numpy(csd_empirical)
    freqs_tensor = torch.from_numpy(freqs)

    print(f"  CSD shape: {observed_csd.shape}")

    # --- Set up spectral DCM ---
    # Full A mask: all connections allowed
    a_mask = torch.ones(
        n_regions, n_regions, dtype=torch.float64
    )
    model_args = (observed_csd, freqs_tensor, a_mask)

    # --- Run multi-start SVI ---
    print(
        f"\nRunning SVI "
        f"({args.num_steps} steps, "
        f"{args.n_restarts} restarts, "
        f"lr={args.lr})..."
    )

    def guide_factory() -> object:
        """Create fresh guide for each restart."""
        return create_guide(
            spectral_dcm_model,
            init_scale=0.01,
        )

    guide = guide_factory()
    svi_result = run_svi(
        spectral_dcm_model,
        guide,
        model_args,
        num_steps=args.num_steps,
        lr=args.lr,
        n_restarts=args.n_restarts,
        guide_factory=guide_factory,
    )

    final_loss = float(np.mean(svi_result["losses"][-10:]))
    print(f"  Final ELBO loss (last 10 mean): {final_loss:.2f}")

    # --- Extract posterior ---
    print("\nExtracting posterior A matrix...")
    posterior = extract_posterior_params(guide, model_args)
    a_free_mean = posterior["A_free"]["mean"]
    a_free_std = posterior["A_free"]["std"]

    a_mean = parameterize_A(a_free_mean)
    # Approximate std via parameterize_A on mean +/- std
    a_plus = parameterize_A(a_free_mean + a_free_std)
    a_minus = parameterize_A(a_free_mean - a_free_std)
    a_std = (a_plus - a_minus).abs() / 2.0

    print("\nPosterior A matrix (mean):")
    print(a_mean.detach().numpy().round(4))
    print("\nPosterior A matrix (std):")
    print(a_std.detach().numpy().round(4))

    # --- Save results ---
    output_path = output_dir / "tribe_dcm_results.npz"
    np.savez(
        output_path,
        A_mean=a_mean.detach().numpy(),
        A_std=a_std.detach().numpy(),
        A_free_mean=a_free_mean.detach().numpy(),
        A_free_std=a_free_std.detach().numpy(),
        final_loss=final_loss,
        roi_names_selected=np.array(roi_names_selected),
        roi_indices=np.array(indices),
        n_regions=n_regions,
        num_steps=args.num_steps,
        n_restarts=args.n_restarts,
        freqs=freqs,
    )

    print()
    print("=" * 60)
    print("DCM FITTING COMPLETE")
    print(f"  Regions:    {n_regions}")
    print(f"  Loss:       {final_loss:.2f}")
    print(f"  Output:     {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
