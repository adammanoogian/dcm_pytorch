r"""Pipeline: compare fMRI and M/EEG DCM effective connectivity.

Loads posterior A matrices from TRIBE v2 (fMRI) and M/EEG foundation
model DCM results, computes cross-modal agreement metrics, and
generates comparison figures.

Metrics
-------
- Pearson correlation of normalized A-matrix elements
- Cohen's kappa on sign patterns (excitatory/inhibitory agreement)
- Credible-interval overlap fraction (posterior consistency)

Usage
-----
::

    python scripts/24_compare_crossmodal.py \
        --fmri-results results/phase24_tribe/tribe_dcm_results.npz \
        --meeg-results results/phase24_meeg/meeg_dcm_results.npz \
        --output-dir results/phase24_comparison/

Requires
--------
- pyro_dcm (this project)
- scipy, sklearn, matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyro_dcm.foundation.comparison import (
    compute_credible_interval_overlap,
    compute_pearson_correlation,
    compute_sign_kappa,
    normalize_a_matrix,
)


def plot_comparison(
    a_fmri: np.ndarray,
    a_meeg: np.ndarray,
    roi_names: list[str],
    output_path: str,
) -> None:
    """Create 3-panel comparison figure for cross-modal A matrices.

    Panels:
    (a) fMRI A matrix heatmap with ROI labels.
    (b) M/EEG A matrix heatmap with ROI labels.
    (c) Scatter plot of fMRI vs M/EEG A elements with Pearson r.

    Parameters
    ----------
    a_fmri : np.ndarray, shape (N, N)
        Normalized fMRI A matrix.
    a_meeg : np.ndarray, shape (N, N)
        Normalized M/EEG A matrix.
    roi_names : list[str]
        ROI labels for axis ticks.
    output_path : str
        Base path for output (saves .png and .pdf).
    """
    n = a_fmri.shape[0]
    vmax = max(
        np.abs(a_fmri).max(), np.abs(a_meeg).max()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel (a): fMRI A matrix ---
    im0 = axes[0].imshow(
        a_fmri,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    axes[0].set_title("(a) fMRI A matrix", fontsize=12)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(roi_names, rotation=45, ha="right")
    axes[0].set_yticklabels(roi_names)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # --- Panel (b): M/EEG A matrix ---
    im1 = axes[1].imshow(
        a_meeg,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title("(b) M/EEG A matrix", fontsize=12)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(roi_names, rotation=45, ha="right")
    axes[1].set_yticklabels(roi_names)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # --- Panel (c): Scatter ---
    r = compute_pearson_correlation(a_fmri, a_meeg)
    axes[2].scatter(
        a_fmri.ravel(),
        a_meeg.ravel(),
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
        s=50,
    )
    # Identity line
    lim = max(np.abs(a_fmri).max(), np.abs(a_meeg).max()) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_xlabel("fMRI A elements")
    axes[2].set_ylabel("M/EEG A elements")
    axes[2].set_title(f"(c) Element scatter (r = {r:.3f})")
    axes[2].set_aspect("equal")

    fig.tight_layout()

    # Save PNG and PDF
    out = Path(output_path)
    fig.savefig(
        out.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        out.with_suffix(".pdf"), bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Saved figures: {out.with_suffix('.png')}")
    print(f"                 {out.with_suffix('.pdf')}")


def main() -> None:
    """Run cross-modal A-matrix comparison pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare effective connectivity (A matrices) from "
            "fMRI and M/EEG DCM results."
        ),
    )
    parser.add_argument(
        "--fmri-results",
        type=str,
        required=True,
        help="Path to tribe_dcm_results.npz from fMRI pipeline.",
    )
    parser.add_argument(
        "--meeg-results",
        type=str,
        required=True,
        help="Path to meeg_dcm_results.npz from M/EEG pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase24_comparison/",
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--roi-mapping",
        type=str,
        default=None,
        help=(
            "Optional JSON file mapping fMRI ROI names to M/EEG "
            "ROI names for alignment.  If not provided, assumes "
            "same ROI ordering."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Modal A-Matrix Comparison")
    print("=" * 60)

    # --- Load fMRI results ---
    print(f"\nLoading fMRI results: {args.fmri_results}")
    fmri = np.load(args.fmri_results, allow_pickle=True)
    a_fmri_mean = fmri["A_mean"]
    a_fmri_std = fmri["A_std"]
    print(f"  fMRI A shape: {a_fmri_mean.shape}")

    # --- Load M/EEG results ---
    print(f"Loading M/EEG results: {args.meeg_results}")
    meeg = np.load(args.meeg_results, allow_pickle=True)
    a_meeg_mean = meeg["A_mean"]
    a_meeg_std = meeg["A_std"]
    print(f"  M/EEG A shape: {a_meeg_mean.shape}")

    # --- Validate shapes ---
    if a_fmri_mean.shape != a_meeg_mean.shape:
        msg = (
            f"A matrix shape mismatch: "
            f"fMRI {a_fmri_mean.shape} vs "
            f"M/EEG {a_meeg_mean.shape}. "
            f"Ensure both pipelines use the same number of "
            f"regions/components."
        )
        raise ValueError(msg)

    n = a_fmri_mean.shape[0]

    # --- ROI names ---
    if "roi_names_selected" in fmri:
        roi_names = list(fmri["roi_names_selected"])
    else:
        roi_names = [f"ROI_{i}" for i in range(n)]

    # --- Apply ROI mapping if provided ---
    if args.roi_mapping is not None:
        print(f"Loading ROI mapping: {args.roi_mapping}")
        with open(args.roi_mapping) as f:
            mapping = json.load(f)
        # mapping: {"fmri_roi": "meeg_roi", ...}
        # Reorder M/EEG rows/columns to match fMRI ordering
        if "roi_names" in meeg.files:
            meeg_roi_names = list(meeg["roi_names"])
        elif "roi_names_selected" in meeg.files:
            meeg_roi_names = list(meeg["roi_names_selected"])
        else:
            meeg_roi_names = [f"ROI_{i}" for i in range(n)]

        reorder_idx = []
        for fmri_name in roi_names:
            meeg_name = mapping.get(fmri_name, fmri_name)
            if meeg_name in meeg_roi_names:
                reorder_idx.append(meeg_roi_names.index(meeg_name))
            else:
                msg = (
                    f"ROI '{meeg_name}' (mapped from fMRI "
                    f"'{fmri_name}') not found in M/EEG results."
                )
                raise ValueError(msg)
        reorder = np.array(reorder_idx)
        a_meeg_mean = a_meeg_mean[np.ix_(reorder, reorder)]
        a_meeg_std = a_meeg_std[np.ix_(reorder, reorder)]
        print("  Reordered M/EEG ROIs to match fMRI ordering")

    # --- Normalize ---
    a_fmri_norm = normalize_a_matrix(a_fmri_mean)
    a_meeg_norm = normalize_a_matrix(a_meeg_mean)

    # --- Compute metrics ---
    print("\n--- Agreement Metrics ---")

    pearson_r = compute_pearson_correlation(a_fmri_norm, a_meeg_norm)
    print(f"  Pearson r (normalized):      {pearson_r:+.4f}")

    sign_k = compute_sign_kappa(a_fmri_mean, a_meeg_mean)
    print(f"  Sign-pattern Cohen's kappa:  {sign_k:+.4f}")

    ci_overlap = compute_credible_interval_overlap(
        a_fmri_mean, a_fmri_std, a_meeg_mean, a_meeg_std
    )
    print(f"  CI overlap fraction (95%):   {ci_overlap:.4f}")

    # --- Generate figures ---
    print("\nGenerating comparison figures...")
    fig_path = output_dir / "crossmodal_comparison"
    plot_comparison(a_fmri_norm, a_meeg_norm, roi_names, str(fig_path))

    # --- Save metrics ---
    metrics = {
        "pearson_r": pearson_r,
        "sign_kappa": sign_k,
        "ci_overlap_fraction": ci_overlap,
        "n_regions": n,
        "fmri_results_path": str(args.fmri_results),
        "meeg_results_path": str(args.meeg_results),
    }
    metrics_path = output_dir / "crossmodal_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("CROSS-MODAL COMPARISON COMPLETE")
    print(f"  Regions:     {n}")
    print(f"  Pearson r:   {pearson_r:+.4f}")
    print(f"  Sign kappa:  {sign_k:+.4f}")
    print(f"  CI overlap:  {ci_overlap:.4f}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
