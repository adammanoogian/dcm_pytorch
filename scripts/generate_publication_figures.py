"""Generate publication-quality figures for the v0.6.0 DCM paper.

Modular script with one function per figure. Produces both PDF (vector)
and PNG (raster) outputs in the figures/ directory.

Usage
-----
    python scripts/generate_publication_figures.py
    python scripts/generate_publication_figures.py --figures pipeline_schematic
    python scripts/generate_publication_figures.py --figures all --formats png,pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Use non-interactive backend before any plt calls
matplotlib.use("Agg")

# Append project root so benchmarks package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from benchmarks.plotting import _apply_style  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Journal column widths (inches)
_SINGLE_COL = 3.5
_DOUBLE_COL = 7.0

# Font sizes for publication figures
_LABEL_SIZE = 9
_TICK_SIZE = 7
_PANEL_LABEL_SIZE = 11
_ANNOT_SIZE = 7

# Pipeline stage colors
_CLR_DATA = "#4C72B0"  # steel blue -- data processing
_CLR_TRAIN = "#55A868"  # sage green -- model training
_CLR_DCM = "#DD8452"  # muted orange -- DCM inference


def _require_file(
    path: Path,
    phase: str,
    description: str,
) -> Path:
    """Raise ``FileNotFoundError`` if *path* does not exist.

    Parameters
    ----------
    path : Path
        Expected result file.
    phase : str
        Upstream phase identifier (e.g. ``"Phase 20"``).
    description : str
        Human-readable description of the file.

    Returns
    -------
    Path
        The validated path.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            f"Complete {phase} first to generate this artifact."
        )
    return path


# ===================================================================
# Figure 1: Pipeline schematic (no data dependencies)
# ===================================================================


def fig_pipeline_schematic(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """DCM interpretability pipeline schematic diagram.

    Produces a schematic showing the full v0.6.0 pipeline:
    [Neural Data] -> [Train Model] -> [Extract Latents] -> [PCA] ->
    [Fit DCM] -> [Posterior A, B_j].

    No data files required -- pure matplotlib drawing.

    Parameters
    ----------
    results_dir : Path
        Unused (included for API consistency).

    Returns
    -------
    matplotlib.figure.Figure
        The schematic figure.
    """
    _apply_style()
    fig, ax = plt.subplots(
        figsize=(_DOUBLE_COL, 2.4),
        constrained_layout=True,
    )
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.2, 1.6)
    ax.axis("off")

    # Stage definitions: (x, label, annotation, color)
    stages = [
        (0.0, "Neural\nData", "Cam-CAN MEG /\nsynthetic RNN", _CLR_DATA),
        (1.2, "Train\nModel", "CT-RNN /\nTransformer", _CLR_TRAIN),
        (2.4, "Extract\nLatents", "Hidden\nactivations", _CLR_TRAIN),
        (3.6, "PCA +\nR² Gate", "Dim. reduction\n(R² ≥ 0.90)", _CLR_DATA),
        (4.8, "Fit DCM", "spDCM /\nbilinear DCM", _CLR_DCM),
        (6.0, "Posterior\nA, Bⱼ", "A, Bⱼ + 95% CI", _CLR_DCM),
    ]

    box_w = 0.85
    box_h = 0.75

    for x, label, annot, color in stages:
        # Rounded rectangle for each stage
        bbox = FancyBboxPatch(
            (x - box_w / 2, -box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.add_patch(bbox)

        # Stage label (white text inside box)
        ax.text(
            x,
            0.0,
            label,
            ha="center",
            va="center",
            fontsize=_LABEL_SIZE,
            fontweight="bold",
            color="white",
        )

        # Annotation below box
        ax.text(
            x,
            -box_h / 2 - 0.18,
            annot,
            ha="center",
            va="top",
            fontsize=_ANNOT_SIZE,
            color="#444444",
            style="italic",
        )

    # Arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i][0] + box_w / 2 + 0.02
        x_end = stages[i + 1][0] - box_w / 2 - 0.02
        arrow = FancyArrowPatch(
            (x_start, 0.0),
            (x_end, 0.0),
            arrowstyle="->,head_width=5,head_length=4",
            color="#333333",
            linewidth=1.5,
            mutation_scale=12,
        )
        ax.add_patch(arrow)

    # Title
    ax.text(
        3.0,
        1.1,
        "DCM Interpretability Pipeline (v0.6.0)",
        ha="center",
        va="bottom",
        fontsize=_PANEL_LABEL_SIZE,
        fontweight="bold",
    )

    # Legend for color coding
    legend_items = [
        mpatches.Patch(
            color=_CLR_DATA, label="Data processing", alpha=0.85,
        ),
        mpatches.Patch(
            color=_CLR_TRAIN, label="Model training", alpha=0.85,
        ),
        mpatches.Patch(
            color=_CLR_DCM, label="DCM inference", alpha=0.85,
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=_ANNOT_SIZE,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    return fig


# ===================================================================
# Figure 2: Synthetic parameter recovery (Phase 20)
# ===================================================================


def fig_synthetic_recovery(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """Parameter recovery on synthetic ground truth (Phase 20).

    2x2 subplot grid:
    - Top-left: True vs inferred A scatter (identity line)
    - Top-right: True vs inferred B scatter (identity line)
    - Bottom-left: A RMSE distribution across seeds (violin)
    - Bottom-right: B RMSE distribution across seeds (violin)

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase20_recovery.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The recovery figure.

    Raises
    ------
    FileNotFoundError
        If ``phase20_recovery.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase20_recovery.npz",
        "Phase 20 (latent circuit DCM acceptance)",
        "Synthetic recovery results",
    )
    data = np.load(path, allow_pickle=True)

    a_true = data["a_true"]
    a_inferred = data["a_inferred"]
    b_true = data["b_true"]
    b_inferred = data["b_inferred"]
    a_rmse_seeds = data["a_rmse_seeds"]
    b_rmse_seeds = data["b_rmse_seeds"]

    fig, axes = plt.subplots(
        2, 2, figsize=(_DOUBLE_COL, 6.0), constrained_layout=True,
    )

    # --- (a) True vs inferred A ---
    ax = axes[0, 0]
    ax.scatter(
        a_true.ravel(), a_inferred.ravel(),
        s=12, alpha=0.5, color=_CLR_DCM, edgecolors="none",
    )
    lims = _scatter_lims(a_true, a_inferred)
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True A", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Inferred A", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_aspect("equal")
    ax.text(
        0.03, 0.95, "(a)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (b) True vs inferred B ---
    ax = axes[0, 1]
    ax.scatter(
        b_true.ravel(), b_inferred.ravel(),
        s=12, alpha=0.5, color=_CLR_TRAIN, edgecolors="none",
    )
    lims = _scatter_lims(b_true, b_inferred)
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True B", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Inferred B", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_aspect("equal")
    ax.text(
        0.03, 0.95, "(b)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (c) A RMSE distribution ---
    ax = axes[1, 0]
    parts = ax.violinplot(
        a_rmse_seeds, showmeans=True, showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(_CLR_DCM)
        pc.set_alpha(0.6)
    ax.set_ylabel("A RMSE", fontsize=_LABEL_SIZE)
    ax.set_xticks([1])
    ax.set_xticklabels(["Seeds"], fontsize=_TICK_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.text(
        0.03, 0.95, "(c)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (d) B RMSE distribution ---
    ax = axes[1, 1]
    parts = ax.violinplot(
        b_rmse_seeds, showmeans=True, showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(_CLR_TRAIN)
        pc.set_alpha(0.6)
    ax.set_ylabel("B RMSE", fontsize=_LABEL_SIZE)
    ax.set_xticks([1])
    ax.set_xticklabels(["Seeds"], fontsize=_TICK_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.text(
        0.03, 0.95, "(d)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    return fig


# ===================================================================
# Figure 3: Connectivity matrices (Phase 22)
# ===================================================================


def fig_connectivity_matrices(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """DCM-derived effective connectivity from M/EEG latents (Phase 22).

    1x3 subplot grid:
    - Left: Posterior mean A matrix (RdBu_r, centered at zero)
    - Center: Posterior std A matrix (sequential colormap)
    - Right: Significant connections (95% CI excluding zero)

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase22_connectivity.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The connectivity figure.

    Raises
    ------
    FileNotFoundError
        If ``phase22_connectivity.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase22_connectivity.npz",
        "Phase 22 (spectral DCM for latent circuits)",
        "Connectivity results",
    )
    data = np.load(path, allow_pickle=True)

    a_mean = data["a_mean"]
    a_std = data["a_std"]
    a_significant = data["a_significant"]
    region_names = list(data["region_names"])

    n = a_mean.shape[0]
    fig, axes = plt.subplots(
        1, 3, figsize=(_DOUBLE_COL, 2.8), constrained_layout=True,
    )

    # --- (a) Posterior mean A ---
    ax = axes[0]
    vmax = max(abs(a_mean.min()), abs(a_mean.max()), 0.1)
    im = ax.imshow(
        a_mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, fontsize=_TICK_SIZE, rotation=45)
    ax.set_yticklabels(region_names, fontsize=_TICK_SIZE)
    ax.set_title("Mean A", fontsize=_LABEL_SIZE)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.text(
        -0.15, 1.05, "(a)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
    )

    # --- (b) Posterior std A ---
    ax = axes[1]
    im = ax.imshow(
        a_std, cmap="YlOrRd", vmin=0, aspect="equal",
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, fontsize=_TICK_SIZE, rotation=45)
    ax.set_yticklabels(region_names, fontsize=_TICK_SIZE)
    ax.set_title("Std A", fontsize=_LABEL_SIZE)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.text(
        -0.15, 1.05, "(b)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
    )

    # --- (c) Significant connections ---
    ax = axes[2]
    # Overlay: mean where significant, gray elsewhere
    masked = np.where(a_significant, a_mean, 0.0)
    vmax_sig = max(abs(masked.min()), abs(masked.max()), 0.1)
    im = ax.imshow(
        masked, cmap="RdBu_r", vmin=-vmax_sig, vmax=vmax_sig,
        aspect="equal",
    )
    # Add hatching for non-significant entries
    for i in range(n):
        for j in range(n):
            if not a_significant[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="#cccccc",
                    linewidth=0.5,
                ))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, fontsize=_TICK_SIZE, rotation=45)
    ax.set_yticklabels(region_names, fontsize=_TICK_SIZE)
    ax.set_title("Significant (95% CI)", fontsize=_LABEL_SIZE)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.text(
        -0.15, 1.05, "(c)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
    )

    return fig


# ===================================================================
# Figure 4: BMR model comparison (Phase 23)
# ===================================================================


def fig_bmr_model_comparison(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """BMR circuit selection results (Phase 23).

    Two-panel figure:
    - Left: Horizontal bar chart of log model evidence
    - Right: BMR evidence vs brute-force ELBO scatter

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase23_bmr.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The BMR comparison figure.

    Raises
    ------
    FileNotFoundError
        If ``phase23_bmr.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase23_bmr.npz",
        "Phase 23 (Bayesian model reduction)",
        "BMR results",
    )
    data = np.load(path, allow_pickle=True)

    model_names = list(data["model_names"])
    log_evidence = data["log_evidence"].astype(float)
    brute_force_elbo = data["brute_force_elbo"].astype(float)

    # Relative evidence (subtract worst)
    rel_evidence = log_evidence - log_evidence.min()
    best_idx = int(np.argmax(rel_evidence))

    fig, axes = plt.subplots(
        1, 2, figsize=(_DOUBLE_COL, 3.2), constrained_layout=True,
    )

    # --- (a) Horizontal bar chart ---
    ax = axes[0]
    n_models = len(model_names)
    y_pos = np.arange(n_models)
    colors = [
        _CLR_DCM if i == best_idx else "#aaaaaa"
        for i in range(n_models)
    ]
    ax.barh(y_pos, rel_evidence, color=colors, edgecolor="black",
            linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=_TICK_SIZE)
    ax.set_xlabel(
        "Relative log evidence", fontsize=_LABEL_SIZE,
    )
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.text(
        0.03, 0.95, "(a)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (b) BMR vs brute-force scatter ---
    ax = axes[1]
    ax.scatter(
        brute_force_elbo, log_evidence,
        s=25, color=_CLR_DCM, edgecolors="black",
        linewidths=0.5, zorder=3,
    )
    lims = _scatter_lims_1d(
        np.concatenate([brute_force_elbo, log_evidence]),
    )
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Brute-force ELBO", fontsize=_LABEL_SIZE)
    ax.set_ylabel("BMR log evidence", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_aspect("equal")
    ax.text(
        0.03, 0.95, "(b)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    return fig


# ===================================================================
# Figure 5: Foundation model comparison (Phase 24)
# ===================================================================


def fig_foundation_model_comparison(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """Foundation model comparison across modalities (Phase 24).

    Grouped bar chart comparing DCM-derived connectivity metrics
    across model types.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase24_foundation.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The foundation model comparison figure.

    Raises
    ------
    FileNotFoundError
        If ``phase24_foundation.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase24_foundation.npz",
        "Phase 24 (foundation model use cases)",
        "Foundation model comparison results",
    )
    data = np.load(path, allow_pickle=True)

    metrics = data["metrics"]  # shape (n_models, n_metrics)
    model_names = list(data["model_names"])
    metric_names = list(data["metric_names"])

    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.7 / n_models

    fig, ax = plt.subplots(
        figsize=(_DOUBLE_COL, 3.0), constrained_layout=True,
    )

    bar_colors = [_CLR_DATA, _CLR_TRAIN, _CLR_DCM, "#9467bd"]
    for i, name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        color = bar_colors[i % len(bar_colors)]
        ax.bar(
            x + offset, metrics[i, :], width,
            label=name, color=color, edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        metric_names, fontsize=_TICK_SIZE, rotation=15,
    )
    ax.set_ylabel("Metric value", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_ANNOT_SIZE, loc="upper right")
    ax.set_title(
        "Foundation Model Comparison", fontsize=_LABEL_SIZE,
    )

    return fig


# ===================================================================
# Figure 6: Hybrid VAE-DCM (Phase 25)
# ===================================================================


def fig_hybrid_vae_dcm(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """Hybrid VAE-DCM results (Phase 25).

    2x1 subplot grid:
    - Top: Reconstruction loss curve (VAE ELBO over training epochs)
    - Bottom: VAE-recovered A vs SVI-recovered A scatter

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase25_vae_dcm.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The VAE-DCM figure.

    Raises
    ------
    FileNotFoundError
        If ``phase25_vae_dcm.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase25_vae_dcm.npz",
        "Phase 25 (hybrid VAE-DCM)",
        "VAE-DCM results",
    )
    data = np.load(path, allow_pickle=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(_SINGLE_COL, 5.0), constrained_layout=True,
    )

    # --- (a) ELBO training curve ---
    ax = axes[0]
    epochs = data["epochs"]
    elbo_values = data["elbo_values"]
    ax.plot(epochs, elbo_values, color=_CLR_DCM, linewidth=1.2)
    ax.set_xlabel("Epoch", fontsize=_LABEL_SIZE)
    ax.set_ylabel("VAE ELBO", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.text(
        0.03, 0.95, "(a)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (b) VAE A vs SVI A scatter ---
    ax = axes[1]
    a_vae = data["a_vae"]
    a_svi = data["a_svi"]
    ax.scatter(
        a_svi.ravel(), a_vae.ravel(),
        s=12, alpha=0.5, color=_CLR_TRAIN, edgecolors="none",
    )
    lims = _scatter_lims(a_svi, a_vae)
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SVI-recovered A", fontsize=_LABEL_SIZE)
    ax.set_ylabel("VAE-recovered A", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_aspect("equal")
    ax.text(
        0.03, 0.95, "(b)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    return fig


# ===================================================================
# Figure 7: SBI calibration (Phase 26)
# ===================================================================


def fig_sbi_calibration(
    results_dir: Path,
) -> matplotlib.figure.Figure:
    """SBI calibration for spectral DCM (Phase 26).

    2x1 subplot grid:
    - Top: SBC rank histogram (uniform = well calibrated)
    - Bottom: Coverage plot (expected vs observed CI coverage)

    Parameters
    ----------
    results_dir : Path
        Directory containing ``phase26_sbi.npz``.

    Returns
    -------
    matplotlib.figure.Figure
        The SBI calibration figure.

    Raises
    ------
    FileNotFoundError
        If ``phase26_sbi.npz`` is missing.
    """
    _apply_style()
    path = _require_file(
        results_dir / "phase26_sbi.npz",
        "Phase 26 (SBI/NPE for spectral DCM)",
        "SBI calibration results",
    )
    data = np.load(path, allow_pickle=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(_SINGLE_COL, 5.0), constrained_layout=True,
    )

    # --- (a) SBC rank histogram ---
    ax = axes[0]
    ranks = data["sbc_ranks"]
    n_bins = min(20, len(ranks) // 5) if len(ranks) > 20 else 10
    ax.hist(
        ranks, bins=n_bins, color=_CLR_DATA,
        edgecolor="black", linewidth=0.5, density=True,
    )
    # Uniform reference line
    ax.axhline(
        1.0 / n_bins * len(ranks) / (ranks.max() - ranks.min())
        if ranks.max() > ranks.min()
        else 1.0,
        color="red", linestyle="--", linewidth=0.8, alpha=0.7,
        label="Uniform",
    )
    ax.set_xlabel("Rank", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_ANNOT_SIZE)
    ax.text(
        0.03, 0.95, "(a)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    # --- (b) Coverage plot ---
    ax = axes[1]
    nominal_levels = data["nominal_levels"]
    observed_coverage = data["observed_coverage"]
    ax.plot(
        nominal_levels, observed_coverage,
        "o-", color=_CLR_DCM, markersize=5, linewidth=1.2,
        label="Observed",
    )
    ax.plot(
        [0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6,
        label="Ideal",
    )
    ax.set_xlabel("Nominal CI level", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Observed coverage", fontsize=_LABEL_SIZE)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_ANNOT_SIZE)
    ax.text(
        0.03, 0.95, "(b)", transform=ax.transAxes,
        fontsize=_PANEL_LABEL_SIZE, fontweight="bold",
        va="top",
    )

    return fig


# ===================================================================
# Helpers
# ===================================================================


def _scatter_lims(
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> list[float]:
    """Compute symmetric axis limits for a scatter plot.

    Parameters
    ----------
    arr1 : np.ndarray
        First array (e.g. true values).
    arr2 : np.ndarray
        Second array (e.g. inferred values).

    Returns
    -------
    list[float]
        ``[lo, hi]`` limits with 5% margin.
    """
    lo = min(float(arr1.min()), float(arr2.min()))
    hi = max(float(arr1.max()), float(arr2.max()))
    margin = (hi - lo) * 0.05 if hi > lo else 0.1
    return [lo - margin, hi + margin]


def _scatter_lims_1d(arr: np.ndarray) -> list[float]:
    """Compute symmetric axis limits from a single array.

    Parameters
    ----------
    arr : np.ndarray
        Combined array of values.

    Returns
    -------
    list[float]
        ``[lo, hi]`` limits with 5% margin.
    """
    lo = float(arr.min())
    hi = float(arr.max())
    margin = (hi - lo) * 0.05 if hi > lo else 0.1
    return [lo - margin, hi + margin]


# ===================================================================
# Registry and CLI
# ===================================================================

# Map figure name -> function
_FIGURE_REGISTRY: dict[str, tuple[str, object]] = {
    "pipeline_schematic": (
        "Pipeline schematic (no data needed)",
        fig_pipeline_schematic,
    ),
    "synthetic_recovery": (
        "Synthetic parameter recovery (Phase 20)",
        fig_synthetic_recovery,
    ),
    "connectivity_matrices": (
        "Connectivity matrices (Phase 22)",
        fig_connectivity_matrices,
    ),
    "bmr_model_comparison": (
        "BMR model comparison (Phase 23)",
        fig_bmr_model_comparison,
    ),
    "foundation_model_comparison": (
        "Foundation model comparison (Phase 24)",
        fig_foundation_model_comparison,
    ),
    "hybrid_vae_dcm": (
        "Hybrid VAE-DCM (Phase 25)",
        fig_hybrid_vae_dcm,
    ),
    "sbi_calibration": (
        "SBI calibration (Phase 26)",
        fig_sbi_calibration,
    ),
}


def _save_fig(
    fig: matplotlib.figure.Figure,
    output_dir: Path,
    name: str,
    formats: list[str],
) -> list[Path]:
    """Save figure in the requested formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    output_dir : Path
        Output directory.
    name : str
        Base filename without extension.
    formats : list[str]
        Output formats (e.g. ``["png", "pdf"]``).

    Returns
    -------
    list[Path]
        Paths of saved files.
    """
    saved = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    """CLI entry point for publication figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate v0.6.0 publication figures",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.BENCHMARK_RESULTS_DIR,
        help="Directory with phase result NPZ files "
        "(default: benchmarks/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.FIGURES_DIR,
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated figure names, or 'all' "
        "(default: all). Available: "
        + ", ".join(_FIGURE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats "
        "(default: png,pdf)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir

    formats = [f.strip() for f in args.formats.split(",")]

    if args.figures == "all":
        figure_names = list(_FIGURE_REGISTRY.keys())
    else:
        figure_names = [
            f.strip() for f in args.figures.split(",")
        ]
        for name in figure_names:
            if name not in _FIGURE_REGISTRY:
                print(
                    f"Error: unknown figure '{name}'. "
                    f"Available: {list(_FIGURE_REGISTRY.keys())}"
                )
                sys.exit(1)

    generated = []
    skipped = []

    for name in figure_names:
        desc, func = _FIGURE_REGISTRY[name]
        try:
            fig = func(results_dir)
            paths = _save_fig(fig, output_dir, name, formats)
            generated.append(name)
            for p in paths:
                print(f"  Saved: {p}")
        except FileNotFoundError as exc:
            skipped.append((name, str(exc)))
            print(f"  Skipped {name}: {exc}")

    # Summary
    print("\nFigure generation complete:")
    print(f"  Generated: {len(generated)}/{len(figure_names)}")
    if generated:
        print(f"    {', '.join(generated)}")
    if skipped:
        print(f"  Skipped: {len(skipped)}/{len(figure_names)}")
        for name, reason in skipped:
            print(f"    {name}: {reason}")


if __name__ == "__main__":
    main()
