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

# Guide-level color and label mappings for calibration plots
GUIDE_COLORS: dict[str, str] = {
    "auto_delta": "#1f77b4",
    "auto_normal": "#ff7f0e",
    "auto_lowrank_mvn": "#2ca02c",
    "auto_mvn": "#d62728",
    "auto_iaf": "#9467bd",
    "auto_laplace": "#8c564b",
    "rdcm_rigid": "#e377c2",
    "rdcm_sparse": "#7f7f7f",
}

GUIDE_LABELS: dict[str, str] = {
    "auto_delta": "AutoDelta",
    "auto_normal": "AutoNormal",
    "auto_lowrank_mvn": "AutoLowRankMVN",
    "auto_mvn": "AutoMVN",
    "auto_iaf": "AutoIAF",
    "auto_laplace": "AutoLaplace",
    "rdcm_rigid": "rDCM (rigid)",
    "rdcm_sparse": "rDCM (sparse)",
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


# ---------------------------------------------------------------------------
# Calibration analysis helpers
# ---------------------------------------------------------------------------

# Canonical CI levels for calibration curves
_CI_LEVELS = [0.50, 0.75, 0.90, 0.95]
_CI_LEVEL_STRS = ["0.5", "0.75", "0.9", "0.95"]

# Variant detection patterns for rDCM keys
_RDCM_PREFIXES = ("rdcm_rigid", "rdcm_sparse")

# SVI variant prefixes
_SVI_VARIANTS = ("task", "spectral")


def _parse_calibration_key(
    key: str,
) -> dict[str, str] | None:
    """Parse a calibration result key into components.

    Handles both SVI keys (e.g., ``spectral_auto_normal_trace_elbo_3``)
    and rDCM keys (e.g., ``rdcm_rigid_vb_na_3``).

    Parameters
    ----------
    key : str
        Result dictionary key.

    Returns
    -------
    dict[str, str] or None
        Parsed components with keys ``"variant"``, ``"guide_type"``,
        ``"elbo_type"``, ``"n_regions"``. Returns ``None`` for
        metadata or unrecognized keys.
    """
    if key == "metadata":
        return None

    parts = key.split("_")
    if len(parts) < 3:
        return None

    # Last element should be the n_regions integer
    try:
        int(parts[-1])
    except ValueError:
        return None
    n_regions = parts[-1]

    # rDCM keys: rdcm_{rigid|sparse}_vb_na_{N}
    for prefix in _RDCM_PREFIXES:
        prefix_parts = prefix.split("_")
        n_prefix = len(prefix_parts)
        if parts[:n_prefix] == prefix_parts:
            remaining = parts[n_prefix:-1]
            if len(remaining) >= 2:
                return {
                    "variant": prefix,
                    "guide_type": prefix,
                    "elbo_type": "_".join(remaining),
                    "n_regions": n_regions,
                }
            return None

    # SVI keys: {variant}_{guide_type}_{elbo_type}_{N}
    for variant in _SVI_VARIANTS:
        variant_parts = variant.split("_")
        n_var = len(variant_parts)
        if parts[:n_var] == variant_parts:
            remaining = parts[n_var:-1]
            # Find split between guide_type and elbo_type
            # Guide types: auto_delta, auto_normal,
            # auto_lowrank_mvn, auto_mvn, auto_iaf, auto_laplace
            # ELBO types: trace_elbo, tracemeanfield_elbo,
            # renyi_elbo
            guide_type, elbo_type = _split_guide_elbo(
                remaining,
            )
            if guide_type is not None:
                return {
                    "variant": variant,
                    "guide_type": guide_type,
                    "elbo_type": elbo_type,
                    "n_regions": n_regions,
                }
    return None


def _split_guide_elbo(
    parts: list[str],
) -> tuple[str | None, str | None]:
    """Split remaining key parts into guide_type and elbo_type.

    Parameters
    ----------
    parts : list[str]
        Key parts between variant prefix and N suffix.

    Returns
    -------
    tuple[str or None, str or None]
        ``(guide_type, elbo_type)`` or ``(None, None)`` if
        unrecognized.
    """
    joined = "_".join(parts)

    # Known guide types (longest first to avoid prefix ambiguity)
    guide_types = [
        "auto_lowrank_mvn",
        "auto_laplace",
        "auto_normal",
        "auto_delta",
        "auto_mvn",
        "auto_iaf",
    ]
    for gt in guide_types:
        if joined.startswith(gt + "_"):
            elbo = joined[len(gt) + 1 :]
            if elbo:
                return gt, elbo
    return None, None


def _group_by_variant_and_guide(
    results: dict,
    n_regions: int,
    param_type: str = "all",
) -> dict[str, dict[str, dict]]:
    """Group calibration results by variant and guide type.

    Parameters
    ----------
    results : dict
        Full calibration results.
    n_regions : int
        Network size to filter on.
    param_type : str
        ``"all"``, ``"diagonal"``, or ``"off_diagonal"``.

    Returns
    -------
    dict[str, dict[str, dict]]
        ``{variant: {guide_type: result_data}}``
    """
    coverage_key_map = {
        "all": "coverage_multi",
        "diagonal": "coverage_diag_multi",
        "off_diagonal": "coverage_offdiag_multi",
    }
    cov_key = coverage_key_map.get(param_type, "coverage_multi")

    grouped: dict[str, dict[str, dict]] = {}
    for key, val in results.items():
        parsed = _parse_calibration_key(key)
        if parsed is None:
            continue
        if parsed["n_regions"] != str(n_regions):
            continue
        if not isinstance(val, dict):
            continue
        if val.get("status") not in ("completed", None):
            continue
        if cov_key not in val:
            continue

        variant = parsed["variant"]
        guide = parsed["guide_type"]
        grouped.setdefault(variant, {})[guide] = val
    return grouped


# ---------------------------------------------------------------------------
# Calibration curve plotting (CAL-01)
# ---------------------------------------------------------------------------


def plot_calibration_curves(
    results: dict,
    output_dir: str,
    n_regions: int = 3,
    param_type: str = "all",
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Plot expected-vs-observed coverage calibration curves.

    Creates one subplot per DCM variant with lines for each guide
    type showing median coverage across datasets. IQR bands show
    variability. A y=x diagonal is the reference for perfect
    calibration.

    Never aggregates across DCM variants (STATE.md risk P9).

    Parameters
    ----------
    results : dict
        Calibration results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    n_regions : int, optional
        Network size to plot. Default 3.
    param_type : str, optional
        Parameter subset: ``"all"``, ``"diagonal"``, or
        ``"off_diagonal"``. Default ``"all"``.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
    """
    _apply_style()
    grouped = _group_by_variant_and_guide(
        results, n_regions, param_type,
    )
    if not grouped:
        print(
            f"No calibration data for N={n_regions}, "
            f"param_type={param_type} -- skipping"
        )
        return

    coverage_key_map = {
        "all": "coverage_multi",
        "diagonal": "coverage_diag_multi",
        "off_diagonal": "coverage_offdiag_multi",
    }
    cov_key = coverage_key_map.get(param_type, "coverage_multi")

    variant_order = [
        v for v in ["task", "spectral", "rdcm_rigid", "rdcm_sparse"]
        if v in grouped
    ]
    n_variants = len(variant_order)

    fig, axes = plt.subplots(
        1, n_variants,
        figsize=(5 * n_variants + 1, 5),
        dpi=150,
        squeeze=False,
    )

    for ax_idx, variant in enumerate(variant_order):
        ax = axes[0, ax_idx]
        guide_data = grouped[variant]

        # y=x diagonal reference
        ax.plot(
            [0, 1], [0, 1], "k--",
            linewidth=1.0, alpha=0.7,
            label="y = x",
            zorder=1,
        )

        for guide_type in sorted(guide_data.keys()):
            data = guide_data[guide_type]
            cov_multi = data.get(cov_key, {})

            medians = []
            q25s = []
            q75s = []
            x_levels = []

            for lv, lv_str in zip(
                _CI_LEVELS, _CI_LEVEL_STRS, strict=True,
            ):
                values = cov_multi.get(lv_str, [])
                if not values:
                    continue
                arr = np.array(values)
                medians.append(float(np.median(arr)))
                q25s.append(float(np.percentile(arr, 25)))
                q75s.append(float(np.percentile(arr, 75)))
                x_levels.append(lv)

            if not x_levels:
                continue

            color = GUIDE_COLORS.get(guide_type, "#333333")
            label = GUIDE_LABELS.get(guide_type, guide_type)

            ax.plot(
                x_levels, medians,
                color=color, marker="o", markersize=5,
                linewidth=1.5, label=label, zorder=3,
            )
            ax.fill_between(
                x_levels, q25s, q75s,
                color=color, alpha=0.2, zorder=2,
            )

        variant_label = _VARIANT_LABELS.get(variant, variant)
        param_label = param_type.replace("_", "-")
        ax.set_title(
            f"{variant_label} (N={n_regions}, {param_label})",
            fontsize=11,
        )
        ax.set_xlabel("Nominal CI level")
        ax.set_ylabel("Observed coverage")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(_CI_LEVELS)
        ax.grid(alpha=0.3)

    # Shared legend below subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        f"Coverage Calibration Curves ({param_label}, N={n_regions})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    fname = f"calibration_curves_{param_type}_N{n_regions}"
    _save_figure(fig, os.path.join(output_dir, fname), formats)


# ---------------------------------------------------------------------------
# Cross-method comparison table (CAL-03)
# ---------------------------------------------------------------------------


def generate_comparison_table(
    results: dict,
    output_dir: str,
    n_regions: int = 3,
) -> dict[str, str]:
    """Generate cross-method comparison tables with median (IQR).

    Produces Markdown, LaTeX, and JSON tables grouped by DCM variant
    with one row per guide type. All cells use median (q25-q75)
    format per STATE.md risk P12.

    Never aggregates across variants (STATE.md risk P9).

    Parameters
    ----------
    results : dict
        Calibration results loaded from JSON.
    output_dir : str
        Directory for saving table files.
    n_regions : int, optional
        Network size to tabulate. Default 3.

    Returns
    -------
    dict[str, str]
        Keys ``"markdown"``, ``"latex"``, ``"json"`` with string
        content for each format.
    """
    grouped = _group_by_variant_and_guide(
        results, n_regions, param_type="all",
    )
    if not grouped:
        print(
            f"No results for comparison table at N={n_regions} "
            "-- skipping"
        )
        return {"markdown": "", "latex": "", "json": ""}

    os.makedirs(output_dir, exist_ok=True)

    # Collect table data
    table_data: dict[str, list[dict[str, str]]] = {}

    for variant in [
        "task", "spectral", "rdcm_rigid", "rdcm_sparse",
    ]:
        if variant not in grouped:
            continue
        variant_label = _VARIANT_LABELS.get(variant, variant)
        rows = []
        for guide_type in sorted(grouped[variant].keys()):
            data = grouped[variant][guide_type]
            row = _build_table_row(data, guide_type)
            rows.append(row)
        table_data[variant_label] = rows

    # Generate Markdown
    md = _format_table_markdown(table_data, n_regions)
    md_path = os.path.join(
        output_dir,
        f"comparison_table_N{n_regions}.md",
    )
    with open(md_path, "w") as f:
        f.write(md)

    # Generate LaTeX
    tex = _format_table_latex(table_data, n_regions)
    tex_path = os.path.join(
        output_dir,
        f"comparison_table_N{n_regions}.tex",
    )
    with open(tex_path, "w") as f:
        f.write(tex)

    # Generate JSON
    json_path = os.path.join(
        output_dir,
        f"comparison_table_N{n_regions}.json",
    )
    json_str = json.dumps(table_data, indent=2)
    with open(json_path, "w") as f:
        f.write(json_str)

    print(
        f"Comparison table N={n_regions}: "
        f"{md_path}, {tex_path}, {json_path}"
    )
    return {"markdown": md, "latex": tex, "json": json_str}


def _build_table_row(
    data: dict, guide_type: str,
) -> dict[str, str]:
    """Build one row of the comparison table.

    Parameters
    ----------
    data : dict
        Result data for one configuration.
    guide_type : str
        Guide type string.

    Returns
    -------
    dict[str, str]
        Row dict with keys ``"method"``, ``"rmse"``,
        ``"coverage_90"``, ``"correlation"``, ``"wall_time"``.
    """
    method = GUIDE_LABELS.get(guide_type, guide_type)

    # RMSE
    rmse_str = _format_median_iqr_from_list(
        data.get("rmse_list", []),
    )

    # Coverage@90%
    cov_multi = data.get("coverage_multi", {})
    cov90_list = cov_multi.get("0.9", [])
    cov90_str = _format_median_iqr_from_list(cov90_list)

    # Correlation
    corr_str = _format_median_iqr_from_list(
        data.get("correlation_list", []),
    )

    # Wall time
    time_list = data.get("time_list", [])
    if not time_list:
        time_list = data.get("amort_time_list", [])
    time_str = _format_median_iqr_from_list(time_list, fmt=".1f")

    return {
        "method": method,
        "rmse": rmse_str,
        "coverage_90": cov90_str,
        "correlation": corr_str,
        "wall_time": time_str,
    }


def _format_median_iqr_from_list(
    values: list[float],
    fmt: str = ".3f",
) -> str:
    """Format a list as 'median (q25-q75)'.

    Parameters
    ----------
    values : list[float]
        Raw values.
    fmt : str, optional
        Format specifier. Default ``".3f"``.

    Returns
    -------
    str
        Formatted string like ``"0.123 (0.110-0.140)"``.
    """
    if not values:
        return "N/A"
    arr = np.array(values)
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return f"{med:{fmt}} ({q25:{fmt}}-{q75:{fmt}})"


def _format_table_markdown(
    table_data: dict[str, list[dict[str, str]]],
    n_regions: int,
) -> str:
    """Format comparison table as Markdown.

    Parameters
    ----------
    table_data : dict
        ``{variant_label: [row_dicts]}``.
    n_regions : int
        Network size.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [f"# Cross-Method Comparison (N={n_regions})\n"]

    for variant_label, rows in table_data.items():
        lines.append(f"\n## {variant_label}\n")
        header = (
            "| Method | RMSE | Coverage@90% "
            "| Pearson r | Wall Time (s) |"
        )
        sep = (
            "|--------|------|----------"
            "----|-----------|---------------|"
        )
        lines.append(header)
        lines.append(sep)
        for row in rows:
            lines.append(
                f"| {row['method']} "
                f"| {row['rmse']} "
                f"| {row['coverage_90']} "
                f"| {row['correlation']} "
                f"| {row['wall_time']} |"
            )

    return "\n".join(lines) + "\n"


def _format_table_latex(
    table_data: dict[str, list[dict[str, str]]],
    n_regions: int,
) -> str:
    """Format comparison table as LaTeX.

    Parameters
    ----------
    table_data : dict
        ``{variant_label: [row_dicts]}``.
    n_regions : int
        Network size.

    Returns
    -------
    str
        LaTeX table string.
    """
    lines = [
        f"% Cross-Method Comparison (N={n_regions})",
    ]

    for variant_label, rows in table_data.items():
        lines.append(f"\n% {variant_label}")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(
            f"\\caption{{{variant_label} (N={n_regions})}}"
        )
        lines.append(
            "\\begin{tabular}{l c c c c}"
        )
        lines.append("\\toprule")
        lines.append(
            "Method & RMSE & Coverage@90\\% "
            "& Pearson $r$ & Wall Time (s) \\\\"
        )
        lines.append("\\midrule")
        for row in rows:
            lines.append(
                f"{row['method']} & {row['rmse']} "
                f"& {row['coverage_90']} "
                f"& {row['correlation']} "
                f"& {row['wall_time']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scaling study (CAL-02)
# ---------------------------------------------------------------------------


def plot_scaling_study(
    results: dict,
    output_dir: str,
    metric: str = "rmse",
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Plot metric vs network size scaling study.

    X-axis is network size (3, 5, 10). Y-axis is the chosen metric.
    One line per guide type with IQR error bars. Separate subplots
    per DCM variant.

    Parameters
    ----------
    results : dict
        Calibration results loaded from JSON.
    output_dir : str
        Directory for saving the figure.
    metric : str, optional
        Metric to plot: ``"rmse"``, ``"coverage"``, or ``"time"``.
        Default ``"rmse"``.
    formats : tuple of str, optional
        File formats to save. Default ``("png",)``.
    """
    _apply_style()

    # Discover all (variant, guide_type, n_regions) combos
    entries: dict[
        str, dict[str, dict[int, list[float]]]
    ] = {}

    for key, val in results.items():
        parsed = _parse_calibration_key(key)
        if parsed is None:
            continue
        if not isinstance(val, dict):
            continue
        if val.get("status") not in ("completed", None):
            continue

        variant = parsed["variant"]
        guide = parsed["guide_type"]
        n_reg = int(parsed["n_regions"])
        values = _extract_metric_values(val, metric)
        if values is None:
            continue

        entries.setdefault(variant, {}).setdefault(
            guide, {},
        )[n_reg] = values

    if not entries:
        print(
            f"No scaling data for metric={metric} -- skipping"
        )
        return

    variant_order = [
        v for v in [
            "task", "spectral", "rdcm_rigid", "rdcm_sparse",
        ]
        if v in entries
    ]
    n_variants = len(variant_order)

    fig, axes = plt.subplots(
        1, n_variants,
        figsize=(5 * n_variants + 1, 5),
        dpi=150,
        squeeze=False,
    )

    metric_labels = {
        "rmse": "RMSE(A)",
        "coverage": "Coverage@90%",
        "time": "Wall Time (s)",
    }

    for ax_idx, variant in enumerate(variant_order):
        ax = axes[0, ax_idx]
        guide_data = entries[variant]

        for guide_type in sorted(guide_data.keys()):
            size_data = guide_data[guide_type]
            sizes = sorted(size_data.keys())
            medians = []
            q25s = []
            q75s = []

            for n in sizes:
                arr = np.array(size_data[n])
                medians.append(float(np.median(arr)))
                q25s.append(float(np.percentile(arr, 25)))
                q75s.append(float(np.percentile(arr, 75)))

            color = GUIDE_COLORS.get(guide_type, "#333333")
            label = GUIDE_LABELS.get(guide_type, guide_type)
            err_lo = [
                m - q for m, q in zip(medians, q25s, strict=True)
            ]
            err_hi = [
                q - m for m, q in zip(medians, q75s, strict=True)
            ]

            ax.errorbar(
                sizes, medians,
                yerr=[err_lo, err_hi],
                color=color, marker="o", markersize=5,
                linewidth=1.5, capsize=3,
                label=label, zorder=3,
            )

        variant_label = _VARIANT_LABELS.get(variant, variant)
        ax.set_title(variant_label, fontsize=11)
        ax.set_xlabel("Network size (N)")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_xticks(
            sorted(
                {
                    n
                    for gd in guide_data.values()
                    for n in gd.keys()
                },
            ),
        )
        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        f"Scaling Study: {metric_labels.get(metric, metric)}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    _save_figure(
        fig,
        os.path.join(output_dir, f"scaling_{metric}"),
        formats,
    )


def _extract_metric_values(
    data: dict, metric: str,
) -> list[float] | None:
    """Extract per-dataset metric values from result data.

    Parameters
    ----------
    data : dict
        Single result entry.
    metric : str
        ``"rmse"``, ``"coverage"``, or ``"time"``.

    Returns
    -------
    list[float] or None
        Per-dataset values, or ``None`` if unavailable.
    """
    if metric == "rmse":
        return data.get("rmse_list")
    if metric == "coverage":
        cov_multi = data.get("coverage_multi", {})
        return cov_multi.get("0.9")
    if metric == "time":
        t = data.get("time_list")
        if t is None:
            t = data.get("amort_time_list")
        return t
    return None


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

    # Calibration figures (when coverage_multi data is present)
    has_calibration = any(
        isinstance(v, dict) and "coverage_multi" in v
        for v in results.values()
    )
    if has_calibration:
        # Discover available n_regions values
        n_regions_set: set[int] = set()
        for key in results:
            parsed = _parse_calibration_key(key)
            if parsed is not None:
                n_regions_set.add(int(parsed["n_regions"]))

        for n_reg in sorted(n_regions_set):
            for ptype in ("all", "diagonal", "off_diagonal"):
                try:
                    plot_calibration_curves(
                        results, output_dir, n_reg, ptype,
                        formats,
                    )
                    n_generated += 1
                except Exception as e:
                    print(
                        f"Warning: plot_calibration_curves"
                        f"({ptype}, N={n_reg}) failed: {e}"
                    )
            try:
                generate_comparison_table(
                    results, output_dir, n_reg,
                )
                n_generated += 1
            except Exception as e:
                print(
                    f"Warning: generate_comparison_table"
                    f"(N={n_reg}) failed: {e}"
                )
        for m in ("rmse", "coverage", "time"):
            try:
                plot_scaling_study(
                    results, output_dir, m, formats,
                )
                n_generated += 1
            except Exception as e:
                print(
                    f"Warning: plot_scaling_study"
                    f"({m}) failed: {e}"
                )

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
