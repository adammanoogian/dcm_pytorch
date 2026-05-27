r"""Analysis and figure generation for perturbation experiment results.

Loads ``perturbation_results.npz`` produced by
``22_perturbation_experiment.py`` and generates diagnostic figures
quantifying whether the latent DCM pipeline detects known
connectivity perturbations.

Usage
-----
python scripts/22_analyze_perturbation.py \
    --results results/perturbation/perturbation_results.npz \
    --output-dir figures/perturbation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_results(path: Path) -> dict:
    """Load perturbation experiment results from npz file.

    Parameters
    ----------
    path : Path
        Path to ``perturbation_results.npz``.

    Returns
    -------
    dict
        Loaded numpy arrays.
    """
    data = np.load(path, allow_pickle=True)
    return dict(data)


def compute_zscore(
    delta_A: np.ndarray,
    perturbed_ij: np.ndarray,
) -> np.ndarray:
    """Compute z-score of perturbed element vs unperturbed elements.

    For each condition, computes: |delta_A[i,j]| / std(delta_A[others]).

    Parameters
    ----------
    delta_A : np.ndarray
        Shape ``(n_conditions, N, N)``, posterior delta A per condition.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``, (i, j) indices of perturbed element.

    Returns
    -------
    np.ndarray
        Z-scores, shape ``(n_conditions,)``.
    """
    n_conditions = delta_A.shape[0]
    N = delta_A.shape[1]
    z_scores = np.zeros(n_conditions)

    for c in range(n_conditions):
        i, j = perturbed_ij[c]
        delta_flat = np.abs(delta_A[c].ravel())
        perturbed_idx = i * N + j
        mask = np.ones(len(delta_flat), dtype=bool)
        mask[perturbed_idx] = False
        other_vals = delta_flat[mask]

        std_other = np.std(other_vals)
        if std_other > 0:
            z_scores[c] = np.abs(delta_A[c, i, j]) / std_other
        else:
            z_scores[c] = np.inf if np.abs(delta_A[c, i, j]) > 0 else 0.0

    return z_scores


def plot_detection_heatmap(
    delta_A: np.ndarray,
    condition_names: np.ndarray,
    perturbed_ij: np.ndarray,
    output_dir: Path,
    roi_names: np.ndarray | None = None,
) -> None:
    """Plot perturbation detection heatmap.

    Rows = perturbation conditions, columns = flattened A elements.
    Perturbed element annotated with a red star marker.

    Parameters
    ----------
    delta_A : np.ndarray
        Shape ``(n_conditions, N, N)``.
    condition_names : np.ndarray
        Condition name strings.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``.
    output_dir : Path
        Figure output directory.
    roi_names : np.ndarray or None
        ROI labels for axis ticks.
    """
    n_conditions, N, _ = delta_A.shape
    delta_flat = np.abs(delta_A.reshape(n_conditions, N * N))

    fig, ax = plt.subplots(figsize=(14, max(5, n_conditions * 0.6)))
    im = ax.imshow(delta_flat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="|delta_A posterior|")

    # Mark perturbed elements
    for c in range(n_conditions):
        i, j = perturbed_ij[c]
        col_idx = i * N + j
        ax.plot(col_idx, c, "r*", markersize=12, markeredgecolor="black")

    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels([str(n) for n in condition_names], fontsize=8)
    ax.set_xlabel("A matrix element (flattened)")
    ax.set_ylabel("Perturbation condition")
    ax.set_title("Perturbation Detection Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "detection_heatmap.png", dpi=150)
    plt.close(fig)


def plot_effect_size_bar(
    z_scores: np.ndarray,
    condition_names: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot effect size (z-score) bar chart per condition.

    Parameters
    ----------
    z_scores : np.ndarray
        Z-scores per condition.
    condition_names : np.ndarray
        Condition name strings.
    output_dir : Path
        Figure output directory.
    """
    n_conditions = len(z_scores)
    fig, ax = plt.subplots(figsize=(10, max(4, n_conditions * 0.5)))

    colors = ["green" if z > 2.0 else "orange" if z > 1.0 else "red"
              for z in z_scores]
    ax.barh(range(n_conditions), z_scores, color=colors, edgecolor="k")
    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels([str(n) for n in condition_names], fontsize=8)
    ax.set_xlabel("Z-score (|perturbed delta| / std(other delta))")
    ax.set_title("Perturbation Effect Size")
    ax.axvline(2.0, color="gray", linestyle="--", alpha=0.7, label="z=2")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "effect_size_bar.png", dpi=150)
    plt.close(fig)


def plot_sensitivity_by_strength(
    A_gt: np.ndarray,
    z_scores: np.ndarray,
    perturbed_ij: np.ndarray,
    condition_names: np.ndarray,
    output_dir: Path,
) -> None:
    """Scatter: baseline |A[i,j]| vs detection z-score.

    Parameters
    ----------
    A_gt : np.ndarray
        Ground-truth A matrix, shape ``(N, N)``.
    z_scores : np.ndarray
        Z-scores per condition.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``.
    condition_names : np.ndarray
        Condition name strings.
    output_dir : Path
        Figure output directory.
    """
    baseline_strengths = np.array([
        np.abs(A_gt[int(ij[0]), int(ij[1])]) for ij in perturbed_ij
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(baseline_strengths, z_scores, s=80, edgecolors="k")

    for k, name in enumerate(condition_names):
        ax.annotate(
            str(name), (baseline_strengths[k], z_scores[k]),
            textcoords="offset points", xytext=(5, 5), fontsize=7,
        )

    ax.set_xlabel("Baseline |A[i,j]| (ground truth)")
    ax.set_ylabel("Detection z-score")
    ax.set_title("Detection Sensitivity vs Connection Strength")
    ax.axhline(2.0, color="gray", linestyle="--", alpha=0.7, label="z=2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_vs_strength.png", dpi=150)
    plt.close(fig)


def plot_true_vs_recovered_delta(
    delta_A: np.ndarray,
    true_deltas: np.ndarray,
    perturbed_ij: np.ndarray,
    output_dir: Path,
) -> None:
    """Scatter: true delta_A[i,j] vs posterior delta_A[i,j].

    Only plots the perturbed element for each condition (not all NxN).

    Parameters
    ----------
    delta_A : np.ndarray
        Shape ``(n_conditions, N, N)``.
    true_deltas : np.ndarray
        Shape ``(n_conditions,)``.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``.
    output_dir : Path
        Figure output directory.
    """
    n_conditions = len(true_deltas)
    posterior_deltas = np.array([
        delta_A[c, int(perturbed_ij[c, 0]), int(perturbed_ij[c, 1])]
        for c in range(n_conditions)
    ])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_deltas, posterior_deltas, s=80, edgecolors="k")

    # Identity line
    all_vals = np.concatenate([true_deltas, posterior_deltas])
    lim = max(np.abs(all_vals).max() * 1.2, 0.01)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="Identity")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("True delta_A[i,j]")
    ax.set_ylabel("Posterior delta_A[i,j] (latent space)")
    ax.set_title("Ground Truth vs Recovered Perturbation")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_recovered_delta.png", dpi=150)
    plt.close(fig)


def plot_baseline_A(
    A_gt: np.ndarray,
    roi_names: np.ndarray | None,
    output_dir: Path,
) -> None:
    """Heatmap of the ground-truth baseline A matrix with ROI labels.

    Parameters
    ----------
    A_gt : np.ndarray
        Ground-truth A matrix, shape ``(N, N)``.
    roi_names : np.ndarray or None
        ROI labels.
    output_dir : Path
        Figure output directory.
    """
    N = A_gt.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.abs(A_gt).max()
    im = ax.imshow(A_gt, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Connection weight")

    if roi_names is not None and len(roi_names) == N:
        labels = [str(n) for n in roi_names]
        ax.set_xticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels, fontsize=8)

    ax.set_title("Baseline A Matrix (Ground Truth)")
    ax.set_xlabel("Source region")
    ax.set_ylabel("Target region")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_A_heatmap.png", dpi=150)
    plt.close(fig)


def print_summary_table(
    condition_names: np.ndarray,
    perturbed_ij: np.ndarray,
    true_deltas: np.ndarray,
    delta_A: np.ndarray,
    z_scores: np.ndarray,
    roi_names: np.ndarray | None = None,
) -> None:
    """Print summary table of perturbation detection results.

    Parameters
    ----------
    condition_names : np.ndarray
        Condition name strings.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``.
    true_deltas : np.ndarray
        Shape ``(n_conditions,)``.
    delta_A : np.ndarray
        Shape ``(n_conditions, N, N)``.
    z_scores : np.ndarray
        Shape ``(n_conditions,)``.
    roi_names : np.ndarray or None
        ROI labels.
    """
    n_conditions = len(condition_names)
    header = (
        f"{'Condition':<28} {'ij':>5} {'true_delta':>11} "
        f"{'post_delta':>11} {'z_score':>8} {'detected?':>10}"
    )
    print("\n" + "=" * len(header))
    print("PERTURBATION DETECTION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for c in range(n_conditions):
        i, j = int(perturbed_ij[c, 0]), int(perturbed_ij[c, 1])
        ij_str = f"({i},{j})"
        post_delta = delta_A[c, i, j]
        detected = "YES" if z_scores[c] > 2.0 else "no"
        print(
            f"{str(condition_names[c]):<28} {ij_str:>5} "
            f"{true_deltas[c]:>11.4f} {post_delta:>11.4f} "
            f"{z_scores[c]:>8.2f} {detected:>10}"
        )

    print("=" * len(header))
    n_detected = np.sum(z_scores > 2.0)
    print(
        f"Detected: {n_detected}/{n_conditions} conditions "
        f"(z > 2.0 threshold)"
    )
    print()


def main() -> None:
    """CLI entry point for perturbation analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze perturbation experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to perturbation_results.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/perturbation"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--roi-names",
        nargs="+",
        default=None,
        help="ROI names (overrides those stored in results)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results}")
    data = load_results(args.results)

    condition_names = data["condition_names"]
    perturbed_ij = data["perturbed_ij"]
    true_deltas = data["true_deltas"]
    delta_A = data["delta_A"]
    A_gt = data["A_ground_truth"]
    roi_names = args.roi_names or data.get("roi_names")

    # Compute z-scores
    z_scores = compute_zscore(delta_A, perturbed_ij)

    # Print summary table
    print_summary_table(
        condition_names, perturbed_ij, true_deltas,
        delta_A, z_scores, roi_names,
    )

    # Generate figures
    print(f"Generating figures in: {args.output_dir}")

    plot_detection_heatmap(
        delta_A, condition_names, perturbed_ij,
        args.output_dir, roi_names,
    )
    print("  [1/5] Detection heatmap")

    plot_effect_size_bar(z_scores, condition_names, args.output_dir)
    print("  [2/5] Effect size bar chart")

    plot_sensitivity_by_strength(
        A_gt, z_scores, perturbed_ij, condition_names, args.output_dir,
    )
    print("  [3/5] Sensitivity vs strength scatter")

    plot_true_vs_recovered_delta(
        delta_A, true_deltas, perturbed_ij, args.output_dir,
    )
    print("  [4/5] True vs recovered delta scatter")

    plot_baseline_A(A_gt, roi_names, args.output_dir)
    print("  [5/5] Baseline A heatmap")

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
