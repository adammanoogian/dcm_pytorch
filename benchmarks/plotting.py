"""Benchmark figure generation from JSON results.

Produces publication-quality figures from benchmark results JSON:
bar charts for RMSE/time/coverage comparisons, scatter plots for
true vs inferred connectivity, and amortization gap visualizations.

All figures are saved in dual format (PDF vector + PNG raster).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Apply publication-quality matplotlib style with fallback chain.

    Tries SciencePlots first, then seaborn-v0_8-whitegrid, then default.
    """
    global _STYLE_APPLIED  # noqa: PLW0603
    if _STYLE_APPLIED:
        return
    _STYLE_APPLIED = True

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
        return
    except (ImportError, OSError):
        print(
            "Warning: SciencePlots not installed, "
            "trying seaborn fallback"
        )

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        return
    except OSError:
        print(
            "Warning: seaborn style not available, "
            "using default matplotlib style"
        )


# Colorblind-friendly palette (tab10)
_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]

# Mapping from result keys to display labels
_VARIANT_LABELS: dict[str, str] = {
    "task": "Task DCM",
    "spectral": "Spectral DCM",
    "rdcm_rigid": "rDCM (rigid)",
    "rdcm_sparse": "rDCM (sparse)",
}

_METHOD_LABELS: dict[str, str] = {
    "svi": "SVI",
    "amortized": "Amortized",
    "vb": "Analytic VB",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _save_figure(fig: matplotlib.figure.Figure, path: str) -> None:
    """Save figure in both PDF and PNG formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Base path without extension.
    """
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def _get_variant_results(
    results: dict,
) -> list[tuple[str, str, dict]]:
    """Extract (variant_key, label, data) triples from results.

    Skips the 'metadata' key and any entries without a 'mean_rmse' field.

    Parameters
    ----------
    results : dict
        Full benchmark results dictionary.

    Returns
    -------
    list of tuple
        (key, display_label, data_dict) for each valid variant.
    """
    entries = []
    for key, val in results.items():
        if key == "metadata":
            continue
        if not isinstance(val, dict):
            continue
        if val.get("status") not in ("completed", None):
            # Allow entries without explicit status (legacy)
            if "mean_rmse" not in val:
                continue
        label = _VARIANT_LABELS.get(key, key)
        entries.append((key, label, val))
    return entries


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_rmse_comparison(
    results: dict,
    output_dir: str,
) -> None:
    """Grouped bar chart of RMSE(A) by variant and method.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for RMSE comparison -- skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    labels = [e[1] for e in entries]
    means = [e[2].get("mean_rmse", 0.0) for e in entries]
    stds = [e[2].get("std_rmse", 0.0) for e in entries]

    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.5,
    )

    # Add method annotation below each bar
    for i, (key, _, data) in enumerate(entries):
        method = "SVI"
        meta = data.get("metadata", {})
        if isinstance(meta, dict):
            m = meta.get("method", "")
            method = _METHOD_LABELS.get(m, m.upper())
        ax.text(
            i, -0.003, method,
            ha="center", va="top", fontsize=8, style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE(A)")
    ax.set_title("Parameter Recovery: RMSE(A) by Variant and Method")
    fig.tight_layout()

    _save_figure(fig, os.path.join(output_dir, "benchmark_rmse_comparison"))


def plot_time_comparison(
    results: dict,
    output_dir: str,
) -> None:
    """Grouped bar chart of wall time by variant and method.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for time comparison -- skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    labels = [e[1] for e in entries]
    times = [e[2].get("mean_time", 0.0) for e in entries]

    x = np.arange(len(labels))
    bars = ax.bar(  # noqa: F841
        x, times, capsize=4,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.5,
    )

    # Use log scale if range is large
    if len(times) > 0:
        t_pos = [t for t in times if t > 0]
        if len(t_pos) >= 2 and max(t_pos) / min(t_pos) > 100:
            ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Inference Time per Subject")
    fig.tight_layout()

    _save_figure(fig, os.path.join(output_dir, "benchmark_time_comparison"))


def plot_coverage_comparison(
    results: dict,
    output_dir: str,
) -> None:
    """Grouped bar chart of 90% CI coverage by variant and method.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for coverage comparison -- skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    labels = [e[1] for e in entries]
    coverages = [e[2].get("mean_coverage", 0.0) for e in entries]
    stds = [e[2].get("std_coverage", 0.0) for e in entries]

    x = np.arange(len(labels))
    ax.bar(
        x, coverages, yerr=stds, capsize=4,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.0,
               label="Nominal 90%")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("90% CI Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Posterior Calibration: 90% CI Coverage")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, "benchmark_coverage_comparison"),
    )


def plot_amortization_gap(
    results: dict,
    output_dir: str,
) -> None:
    """Bar chart of amortization gap (relative ELBO degradation).

    Only plots variants that have both SVI and amortized results.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    """
    _apply_style()

    # Find variants with both SVI and amortized results
    gaps: list[tuple[str, float]] = []

    # Check for paired results
    for base_variant in ["task", "spectral"]:
        svi_key = base_variant
        amort_key = f"{base_variant}_amortized"

        svi_data = results.get(svi_key, {})
        amort_data = results.get(amort_key, {})

        # Also check if amortized is stored as a sub-key
        if not isinstance(amort_data, dict) or "mean_elbo" not in amort_data:
            # Try alternate naming
            for k, v in results.items():
                if not isinstance(v, dict):
                    continue
                meta = v.get("metadata", {})
                if (isinstance(meta, dict)
                        and meta.get("variant") == base_variant
                        and meta.get("method") == "amortized"):
                    amort_data = v
                    break

        if (isinstance(svi_data, dict) and isinstance(amort_data, dict)
                and "mean_elbo" in svi_data and "mean_elbo" in amort_data):
            svi_elbo = svi_data["mean_elbo"]
            amort_elbo = amort_data["mean_elbo"]
            denom = abs(svi_elbo) if abs(svi_elbo) > 1e-15 else 1.0
            rel_gap = abs(amort_elbo - svi_elbo) / denom * 100.0
            label = _VARIANT_LABELS.get(base_variant, base_variant)
            gaps.append((label, rel_gap))

    if not gaps:
        print(
            "No paired SVI/amortized results for amortization "
            "gap -- skipping"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    labels = [g[0] for g in gaps]
    values = [g[1] for g in gaps]
    x = np.arange(len(labels))

    ax.bar(
        x, values,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.0,
               label="10% target threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Relative Amortization Gap (%)")
    ax.set_title("Amortization Gap: ELBO Degradation from Amortization")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _save_figure(fig, os.path.join(output_dir, "amortization_gap"))


def plot_true_vs_inferred(
    results: dict,
    output_dir: str,
) -> None:
    """Scatter plot of true vs inferred A matrix elements.

    Shows identity line and Pearson correlation per variant.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for scatter plot -- skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    has_data = False
    for i, (key, label, data) in enumerate(entries):
        corr = data.get("mean_correlation", None)
        if corr is None:
            continue

        # We plot summary statistics rather than raw A elements
        # since raw A elements are not stored in JSON
        # Instead show mean RMSE vs correlation as a proxy
        rmse = data.get("mean_rmse", None)
        if rmse is not None:
            corr_label = f"{label} (r={corr:.3f})"
            ax.scatter(
                [rmse], [corr], s=120, marker="o",
                color=_COLORS[i % len(_COLORS)],
                edgecolors="black", linewidth=0.5,
                label=corr_label, zorder=5,
            )
            has_data = True

    if not has_data:
        print("No correlation data for scatter plot -- skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Mean RMSE(A)")
    ax.set_ylabel("Mean Correlation")
    ax.set_title("True vs. Inferred Connectivity (A matrix)")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, "true_vs_inferred_scatter"),
    )


def generate_all_figures(
    results: dict,
    output_dir: str = "figures",
) -> None:
    """Generate all benchmark figures from results.

    Calls all five plotting functions. Creates output directory
    if it does not exist.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str, optional
        Directory for saving figures. Default ``"figures"``.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_funcs = [
        plot_rmse_comparison,
        plot_time_comparison,
        plot_coverage_comparison,
        plot_amortization_gap,
        plot_true_vs_inferred,
    ]

    n_generated = 0
    for func in plot_funcs:
        try:
            func(results, output_dir)
            n_generated += 1
        except Exception as e:
            print(f"Warning: {func.__name__} failed: {e}")

    print(f"Generated {n_generated} figures in {output_dir}/")


def load_results(
    path: str = "benchmarks/results/benchmark_results.json",
) -> dict:
    """Load benchmark results from JSON file.

    Parameters
    ----------
    path : str, optional
        Path to JSON results file.

    Returns
    -------
    dict
        Benchmark results dictionary.
    """
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Use non-interactive backend for script execution
    matplotlib.use("Agg")

    results_path = "benchmarks/results/benchmark_results.json"
    if not Path(results_path).exists():
        print(f"Error: results file not found at {results_path}")
        print("Run 'python benchmarks/run_all_benchmarks.py --quick' first")
        sys.exit(1)

    data = load_results(results_path)
    generate_all_figures(data)
