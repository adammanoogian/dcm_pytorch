r"""Pipeline: ROI timeseries -> spectral DCM -> posterior A matrix.

Loads TRIBE v2 ROI timeseries from extraction script output, computes
empirical cross-spectral density, and fits a spectral DCM model using
Variational Laplace to recover the posterior effective connectivity matrix.

Usage
-----
    python scripts/24_fit_dcm_tribe.py \
        --input-npz results/phase24_tribe/tribe_roi_timeseries.npz \
        --output-dir results/phase24_tribe/ \
        --n-regions 6
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
        "--max-iter",
        type=int,
        default=128,
        help="Maximum Gauss-Newton iterations for VL.",
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

    try:
        from pyro_dcm import run_variational_laplace
        from pyro_dcm.forward_models.csd_computation import (
            compute_empirical_csd,
        )
        from pyro_dcm.inference.variational_laplace import (
            extract_vl_posterior,
        )
    except ImportError:
        print(
            "pyro_dcm not installed. "
            "Run: pip install -e . from project root."
        )
        sys.exit(1)

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spectral DCM Fitting on TRIBE v2 ROI Timeseries")
    print("=" * 60)

    print(f"Loading: {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)
    roi_timeseries = data["roi_timeseries"]
    roi_names = list(data["roi_names"])

    print(
        f"  Full ROI timeseries: {roi_timeseries.shape} "
        f"({len(roi_names)} ROIs)"
    )

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

    fs = 1.0
    fmax = fs / 2.0
    freqs = np.linspace(0.01, fmax * 0.9, args.n_freqs)

    print(
        f"\nComputing empirical CSD "
        f"(fs={fs} Hz, {args.n_freqs} freqs)..."
    )
    csd_empirical = compute_empirical_csd(
        roi_subset, fs=fs, freqs=freqs
    )
    observed_csd = torch.from_numpy(csd_empirical)
    freqs_tensor = torch.from_numpy(freqs).to(torch.float64)

    print(f"  CSD shape: {observed_csd.shape}")

    a_mask = torch.ones(
        n_regions, n_regions, dtype=torch.float64
    )

    print(
        f"\nRunning Variational Laplace "
        f"(max_iter={args.max_iter})..."
    )

    vl_result = run_variational_laplace(
        observed_csd,
        freqs_tensor,
        a_mask,
        max_iter=args.max_iter,
        tolerance=1e-2,
    )

    fe = vl_result.free_energy[-1] if vl_result.free_energy else float("nan")
    print(f"  Free energy: {fe:.2f}")
    print(f"  Converged: {vl_result.converged}")
    print(f"  Iterations: {len(vl_result.free_energy)}")

    print("\nExtracting posterior A matrix...")
    posterior = extract_vl_posterior(vl_result, n_regions)

    a_mean = posterior["A"]["mean"]
    a_std = posterior["A"]["std"]

    print("\nPosterior A matrix (mean):")
    print(a_mean.detach().numpy().round(4))
    print("\nPosterior A matrix (std):")
    print(a_std.detach().numpy().round(4))

    output_path = output_dir / "tribe_dcm_results.npz"
    np.savez(
        output_path,
        A_mean=a_mean.detach().numpy(),
        A_std=a_std.detach().numpy(),
        free_energy=fe,
        converged=vl_result.converged,
        roi_names_selected=np.array(roi_names_selected),
        roi_indices=np.array(indices),
        n_regions=n_regions,
        max_iter=args.max_iter,
        freqs=freqs,
    )

    print()
    print("=" * 60)
    print("DCM FITTING COMPLETE")
    print(f"  Regions:      {n_regions}")
    print(f"  Free energy:  {fe:.2f}")
    print(f"  Converged:    {vl_result.converged}")
    print(f"  Output:       {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
