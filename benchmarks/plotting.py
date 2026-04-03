"""Benchmark figure generation from JSON results.

Produces publication-quality figures from benchmark results JSON:
element-wise true vs inferred scatter plots, metric distribution
strip plots, and amortization gap visualizations.

Figures default to PNG. Pass ``formats=("png", "pdf")`` for vector output.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.figure
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


def _save_figure(
    fig: matplotlib.figure.Figure,
    path: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Save figure in the specified formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Base path without extension.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
        Common choices: ``"png"``, ``"pdf"``, ``"svg"``.
    """
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=150)
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


def _get_label_with_method(
    key: str, data: dict,
) -> str:
    """Build a display label including variant name and method.

    Parameters
    ----------
    key : str
        Result dictionary key (e.g. ``"spectral"``).
    data : dict
        Result data dict containing ``metadata.method``.

    Returns
    -------
    str
        Label like ``"Spectral DCM (SVI)"``.
    """
    base = _VARIANT_LABELS.get(key, key)
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        m = meta.get("method", "")
        method_label = _METHOD_LABELS.get(m, m.upper())
        if method_label:
            return f"{base} ({method_label})"
    return base


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_true_vs_inferred(
    results: dict,
    output_dir: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Scatter plot of ALL per-element true vs inferred A values.

    For each variant that has ``a_true_list`` and ``a_inferred_list``,
    pools all elements from all datasets and plots them. Identity line
    y=x shown as black dashed. Legend includes Pearson r per variant.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for scatter plot -- skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    has_data = False
    all_vals: list[float] = []

    for i, (key, _label, data) in enumerate(entries):
        a_true = data.get("a_true_list")
        a_inf = data.get("a_inferred_list")
        if not a_true or not a_inf:
            continue

        # Pool all elements from all datasets
        true_flat: list[float] = []
        inf_flat: list[float] = []
        for t_row, i_row in zip(a_true, a_inf):
            true_flat.extend(t_row)
            inf_flat.extend(i_row)

        if len(true_flat) == 0:
            continue

        true_arr = np.array(true_flat)
        inf_arr = np.array(inf_flat)
        all_vals.extend(true_flat)
        all_vals.extend(inf_flat)

        # Get correlation from summary or compute
        corr = data.get("mean_correlation", None)
        if corr is None and len(true_arr) > 1:
            num = np.sum(
                (true_arr - true_arr.mean())
                * (inf_arr - inf_arr.mean()),
            )
            den = (
                np.sqrt(np.sum((true_arr - true_arr.mean()) ** 2))
                * np.sqrt(np.sum((inf_arr - inf_arr.mean()) ** 2))
            )
            corr = float(num / den) if den > 1e-15 else 0.0

        label = _get_label_with_method(key, data)
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"

        ax.scatter(
            true_arr, inf_arr, s=18, alpha=0.6,
            color=_COLORS[i % len(_COLORS)],
            edgecolors="none",
            label=f"{label} (r={corr_str})",
            zorder=3,
        )
        has_data = True

    if not has_data:
        print(
            "No a_true_list/a_inferred_list data for scatter "
            "plot -- skipping"
        )
        plt.close(fig)
        return

    # Identity line
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        margin = (hi - lo) * 0.05 if hi > lo else 0.1
        lims = [lo - margin, hi + margin]
        ax.plot(
            lims, lims, "k--", linewidth=1.0,
            alpha=0.7, zorder=2, label="y = x",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_xlabel("True A element")
    ax.set_ylabel("Inferred A element")
    ax.set_title("Element-wise Parameter Recovery (A matrix)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, "true_vs_inferred_scatter"),
        formats,
    )


def plot_metric_strips(
    results: dict,
    output_dir: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Strip/jitter plot of per-dataset metrics in a 2x2 grid.

    Subplots: RMSE, Coverage, Correlation, Wall Time.
    Each variant is shown as a jittered strip of per-dataset values
    with a horizontal line at the mean. Coverage subplot includes a
    horizontal reference at 0.90 (nominal).

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
    """
    _apply_style()
    entries = _get_variant_results(results)
    if not entries:
        print("No valid results for metric strips -- skipping")
        return

    metric_specs = [
        ("RMSE(A)", "rmse_list", "mean_rmse"),
        ("Coverage", "coverage_list", "mean_coverage"),
        ("Correlation", "correlation_list", "mean_correlation"),
        ("Wall Time (s)", "time_list", "mean_time"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes_flat = axes.flatten()

    rng = np.random.default_rng(42)

    for ax_idx, (title, list_key, mean_key) in enumerate(
        metric_specs,
    ):
        ax = axes_flat[ax_idx]

        x_positions: list[float] = []
        x_labels: list[str] = []

        for v_idx, (key, _label, data) in enumerate(entries):
            values = data.get(list_key)
            # Amortized runners store time as amort_time_list
            if values is None and list_key == "time_list":
                values = data.get("amort_time_list")
            if values is None:
                continue

            label = _get_label_with_method(key, data)
            x_pos = float(v_idx)
            x_positions.append(x_pos)
            x_labels.append(label)

            # Jitter x
            n_pts = len(values)
            jitter = rng.uniform(-0.15, 0.15, size=n_pts)
            xs = x_pos + jitter

            color = _COLORS[v_idx % len(_COLORS)]
            ax.scatter(
                xs, values, s=30, alpha=0.7,
                color=color, edgecolors="none",
                zorder=3,
            )

            # Mean line
            mean_val = data.get(mean_key)
            if mean_val is None and len(values) > 0:
                mean_val = float(np.mean(values))
            if mean_val is not None:
                ax.hlines(
                    mean_val,
                    x_pos - 0.3, x_pos + 0.3,
                    colors=color, linewidth=2.0,
                    zorder=4,
                )

        # Coverage nominal line
        if list_key == "coverage_list":
            ax.axhline(
                0.90, color="red", linestyle="--",
                linewidth=1.0, alpha=0.7,
                label="Nominal 90%",
            )
            ax.legend(fontsize=8, loc="lower right")

        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8, rotation=15)

        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Benchmark Metrics (per-dataset distributions)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, "benchmark_metric_strips"),
        formats,
    )


def plot_amortization_gap(
    results: dict,
    output_dir: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Bar chart of amortization gap (relative ELBO degradation).

    Only plots variants that have both SVI and amortized results.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
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
        if (
            not isinstance(amort_data, dict)
            or "mean_elbo" not in amort_data
        ):
            # Try alternate naming
            for k, v in results.items():
                if not isinstance(v, dict):
                    continue
                meta = v.get("metadata", {})
                if (
                    isinstance(meta, dict)
                    and meta.get("variant") == base_variant
                    and meta.get("method") == "amortized"
                ):
                    amort_data = v
                    break

        if (
            isinstance(svi_data, dict)
            and isinstance(amort_data, dict)
            and "mean_elbo" in svi_data
            and "mean_elbo" in amort_data
        ):
            svi_elbo = svi_data["mean_elbo"]
            amort_elbo = amort_data["mean_elbo"]
            denom = (
                abs(svi_elbo) if abs(svi_elbo) > 1e-15 else 1.0
            )
            rel_gap = (
                abs(amort_elbo - svi_elbo) / denom * 100.0
            )
            label = _VARIANT_LABELS.get(
                base_variant, base_variant,
            )
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
        color=[
            _COLORS[i % len(_COLORS)]
            for i in range(len(labels))
        ],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(
        10.0, color="red", linestyle="--", linewidth=1.0,
        label="10% target threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Relative Amortization Gap (%)")
    ax.set_title(
        "Amortization Gap: ELBO Degradation from Amortization",
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, "amortization_gap"),
        formats,
    )


def generate_all_figures(
    results: dict,
    output_dir: str = "figures",
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Generate all benchmark figures from results.

    Calls all three plotting functions. Creates output directory
    if it does not exist.

    Parameters
    ----------
    results : dict
        Benchmark results loaded from JSON.
    output_dir : str, optional
        Directory for saving figures. Default ``"figures"``.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
        Use ``("png", "pdf")`` for publication-quality output.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_funcs = [
        plot_true_vs_inferred,
        plot_metric_strips,
        plot_amortization_gap,
    ]

    n_generated = 0
    for func in plot_funcs:
        try:
            func(results, output_dir, formats)
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
        print(
            "Run 'python benchmarks/run_all_benchmarks.py "
            "--quick' first"
        )
        sys.exit(1)

    data = load_results(results_path)
    generate_all_figures(data)
