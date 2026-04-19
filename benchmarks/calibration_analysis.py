#!/usr/bin/env python
"""Calibration analysis figure and table generation.

Loads calibration sweep results from JSON and generates all Phase 11
figures and tables: calibration curves (CAL-01), scaling studies
(CAL-02), and cross-method comparison tables (CAL-03).

Usage
-----
Generate all figures from default results location::

    python benchmarks/calibration_analysis.py

Specify custom paths::

    python benchmarks/calibration_analysis.py \
        --results-path benchmarks/results/calibration_results.json \
        --output-dir benchmarks/figures \
        --formats png,pdf

Filter to specific network sizes::

    python benchmarks/calibration_analysis.py --n-regions 3,5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

# Non-interactive backend for script execution
matplotlib.use("Agg")

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.plotting import (  # noqa: E402
    _parse_calibration_key,
    generate_comparison_table,
    plot_calibration_curves,
    plot_pareto_frontier,
    plot_scaling_study,
    plot_posterior_violins,
    plot_timing_breakdown,
)
from benchmarks.timing_profiler import (  # noqa: E402
    profile_all_guides,
)

# ---------------------------------------------------------------------------
# Result key parsing (public API for external use)
# ---------------------------------------------------------------------------


def _parse_result_key(key: str) -> dict[str, str] | None:
    """Parse a calibration result key into components.

    Public wrapper around the plotting module's parser for use by
    analysis scripts and tests.

    Parameters
    ----------
    key : str
        Result dictionary key (e.g.,
        ``"spectral_auto_normal_trace_elbo_3"`` or
        ``"rdcm_rigid_vb_na_3"``).

    Returns
    -------
    dict[str, str] or None
        Parsed components with keys ``"variant"``, ``"guide_type"``,
        ``"elbo_type"``, ``"n_regions"``. Returns ``None`` for
        metadata or unrecognized keys.

    Examples
    --------
    >>> _parse_result_key("spectral_auto_normal_trace_elbo_3")
    {'variant': 'spectral', 'guide_type': 'auto_normal',
     'elbo_type': 'trace_elbo', 'n_regions': '3'}
    >>> _parse_result_key("rdcm_rigid_vb_na_3")
    {'variant': 'rdcm_rigid', 'guide_type': 'rdcm_rigid',
     'elbo_type': 'vb_na', 'n_regions': '3'}
    >>> _parse_result_key("metadata") is None
    True
    """
    return _parse_calibration_key(key)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def generate_calibration_figures(
    results_path: str = "benchmarks/results/calibration_results.json",
    output_dir: str = "benchmarks/figures",
    formats: tuple[str, ...] = ("png",),
    n_regions_filter: list[int] | None = None,
) -> None:
    """Generate all calibration analysis figures and tables.

    Loads calibration sweep results and produces:
    - Calibration curves for all/diagonal/off-diagonal parameters
      at each network size (CAL-01)
    - Comparison tables in Markdown, LaTeX, JSON with median (IQR)
      format (CAL-03)
    - Scaling study plots for RMSE, coverage, and wall time vs
      network size (CAL-02)

    Parameters
    ----------
    results_path : str, optional
        Path to calibration results JSON.
    output_dir : str, optional
        Directory for output figures and tables.
    formats : tuple of str, optional
        Figure formats. Default ``("png",)``.
    n_regions_filter : list[int] or None, optional
        If provided, only generate for these network sizes.
    """
    if not Path(results_path).exists():
        print(f"Error: results file not found at {results_path}")
        print(
            "Run 'python benchmarks/calibration_sweep.py "
            "--tier 1 --quick' first"
        )
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Discover n_regions values present in results
    n_regions_found: set[int] = set()
    for key in results:
        parsed = _parse_result_key(key)
        if parsed is not None:
            n_regions_found.add(int(parsed["n_regions"]))

    if n_regions_filter is not None:
        n_regions_found = n_regions_found & set(n_regions_filter)

    if not n_regions_found:
        print("No valid result keys found in results JSON")
        return

    n_regions_sorted = sorted(n_regions_found)
    print(
        f"Generating figures for N={n_regions_sorted} "
        f"from {results_path}"
    )
    print(f"Output directory: {output_dir}")
    print(f"Formats: {formats}")
    print()

    generated: list[str] = []

    # Per-network-size figures
    for n_reg in n_regions_sorted:
        # CAL-01: Calibration curves
        for ptype in ("all", "diagonal", "off_diagonal"):
            try:
                plot_calibration_curves(
                    results, output_dir, n_reg, ptype, formats,
                )
                fname = (
                    f"calibration_curves_{ptype}_N{n_reg}"
                )
                generated.append(fname)
            except Exception as e:
                print(
                    f"Warning: calibration curves "
                    f"({ptype}, N={n_reg}) failed: {e}"
                )

        # CAL-03: Comparison tables
        try:
            generate_comparison_table(
                results, output_dir, n_reg,
            )
            generated.append(f"comparison_table_N{n_reg}")
        except Exception as e:
            print(
                f"Warning: comparison table "
                f"(N={n_reg}) failed: {e}"
            )

    # CAL-02: Scaling studies (across all N)
    for metric in ("rmse", "coverage", "time"):
        try:
            plot_scaling_study(
                results, output_dir, metric, formats,
            )
            generated.append(f"scaling_{metric}")
        except Exception as e:
            print(
                f"Warning: scaling study "
                f"({metric}) failed: {e}"
            )

    # CAL-05: Pareto frontier (wall-time vs RMSE)
    for n_reg in n_regions_sorted:
        try:
            plot_pareto_frontier(
                results, output_dir, n_reg, formats,
            )
            generated.append(
                f"pareto_frontier_N{n_reg}",
            )
        except Exception as e:
            print(
                f"Warning: Pareto frontier "
                f"(N={n_reg}) failed: {e}"
            )

    # Summary
    print(f"\nGenerated {len(generated)} outputs:")
    for name in generated:
        print(f"  - {output_dir}/{name}")


# ---------------------------------------------------------------------------
# Supplementary analysis (CAL-04, CAL-05)
# ---------------------------------------------------------------------------


def generate_violin_plots(
    variant: str = "spectral",
    n_regions: int = 3,
    fixtures_dir: str | None = None,
    output_dir: str = "benchmarks/figures",
    seed: int = 42,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Generate per-A_ij posterior violin plots across guide types.

    Re-runs SVI inference on dataset index 0 for each guide type,
    extracts posterior A samples, applies ``parameterize_A``, and
    produces an NxN grid of violin plots overlaying all guide types.

    Parameters
    ----------
    variant : str, optional
        DCM variant. Default ``"spectral"``.
    n_regions : int, optional
        Network size. Default 3.
    fixtures_dir : str or None, optional
        Path to fixtures directory. If provided, must exist and
        contain at least one dataset. If None, generates inline.
    output_dir : str, optional
        Directory for output figures. Default ``"benchmarks/figures"``.
    seed : int, optional
        Random seed. Default 42.
    formats : tuple of str, optional
        Figure formats. Default ``("png",)``.

    Raises
    ------
    FileNotFoundError
        If ``fixtures_dir`` is provided but does not exist or is
        empty.
    """
    import numpy as np
    import pyro
    import torch

    from pyro_dcm.forward_models.neural_state import (
        parameterize_A,
    )
    from pyro_dcm.models import (
        create_guide,
        extract_posterior_params,
        run_svi,
        spectral_dcm_model,
    )

    # Guard fixture dir
    if fixtures_dir is not None:
        fx_path = Path(fixtures_dir)
        if not fx_path.exists() or not any(fx_path.iterdir()):
            msg = (
                f"Fixtures directory not found or empty: "
                f"{fixtures_dir}. Run "
                f"'python benchmarks/generate_fixtures.py' first."
            )
            raise FileNotFoundError(msg)

    if variant != "spectral":
        print(
            f"Warning: violin plots only support "
            f"variant='spectral', got {variant!r}. Skipping."
        )
        return

    from benchmarks.timing_profiler import (
        _build_spectral_model_args,
    )

    N = n_regions
    os.makedirs(output_dir, exist_ok=True)

    # Build model args for dataset 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)

    model_args, A_true = _build_spectral_model_args(
        N, seed, fixtures_dir,
    )

    # Guide types to include (skip auto_delta: point estimate)
    guide_types = [
        "auto_normal",
        "auto_lowrank_mvn",
        "auto_mvn",
        "auto_iaf",
        "auto_laplace",
    ]

    posterior_samples: dict[str, torch.Tensor] = {}

    for guide_type in guide_types:
        # Skip auto_mvn for large networks
        if guide_type == "auto_mvn" and N > 7:
            print(
                f"  Skipping {guide_type} (N={N} > 7, "
                f"memory limit)"
            )
            continue

        print(f"  Running {guide_type} for violin plots...")
        try:
            pyro.clear_param_store()
            torch.manual_seed(seed)

            guide = create_guide(
                spectral_dcm_model,
                init_scale=0.01,
                guide_type=guide_type,
                n_regions=N,
            )

            svi_result = run_svi(
                spectral_dcm_model, guide, model_args,
                num_steps=500, lr=0.01,
                clip_norm=10.0, lr_decay_factor=0.1,
                guide_type=guide_type,
            )

            extract_guide = svi_result.get("guide", guide)
            posterior = extract_posterior_params(
                extract_guide, model_args,
            )

            A_free_samples = posterior["A_free"]["samples"]
            A_param_samples = torch.stack(
                [parameterize_A(s) for s in A_free_samples],
            )
            posterior_samples[guide_type] = A_param_samples
            print(
                f"    Collected {A_param_samples.shape[0]} "
                f"samples"
            )

        except (RuntimeError, ValueError) as e:
            print(f"    FAILED: {e}")

    if posterior_samples:
        plot_posterior_violins(
            posterior_samples, A_true, output_dir, formats,
        )
        print(
            f"  Saved: {output_dir}/"
            f"posterior_violins_N{N}"
        )
    else:
        print("  No posterior samples collected -- skipping")


def generate_timing_figures(
    variant: str = "spectral",
    n_regions: int = 3,
    fixtures_dir: str | None = None,
    output_dir: str = "benchmarks/figures",
    seed: int = 42,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Generate timing breakdown figures from SVI profiling.

    Profiles all guide types on a single representative dataset
    and produces a stacked bar chart of timing components. Saves
    raw timing data as JSON.

    Parameters
    ----------
    variant : str, optional
        DCM variant. Default ``"spectral"``.
    n_regions : int, optional
        Network size. Default 3.
    fixtures_dir : str or None, optional
        Path to fixtures directory. If None, generates inline.
    output_dir : str, optional
        Directory for output figures. Default ``"benchmarks/figures"``.
    seed : int, optional
        Random seed. Default 42.
    formats : tuple of str, optional
        Figure formats. Default ``("png",)``.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Profiling all guides ({variant}, N={n_regions})...")
    timing_data = profile_all_guides(
        variant=variant,
        n_regions=n_regions,
        fixtures_dir=fixtures_dir,
        seed=seed,
    )

    if not timing_data:
        print("  No timing data collected -- skipping")
        return

    # Plot timing breakdown
    plot_timing_breakdown(timing_data, output_dir, formats)
    print(f"  Saved: {output_dir}/timing_breakdown")

    # Save raw timing data as JSON
    json_path = os.path.join(
        output_dir,
        f"timing_data_N{n_regions}.json",
    )
    serializable = {}
    for gt, profile in timing_data.items():
        serializable[gt] = {
            k: v for k, v in profile.items()
            if not isinstance(v, list) or all(
                isinstance(x, (int, float)) for x in v
            )
        }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run calibration analysis from CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate calibration analysis figures and tables "
            "from sweep results. Default mode generates calibration "
            "curves, comparison tables, scaling plots, and Pareto "
            "frontiers from the JSON results file. Use --violin "
            "and --timing flags for supplementary analysis that "
            "re-runs inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/calibration_analysis.py\n"
            "  python benchmarks/calibration_analysis.py "
            "--formats png,pdf\n"
            "  python benchmarks/calibration_analysis.py "
            "--n-regions 3,5\n"
            "  python benchmarks/calibration_analysis.py "
            "--violin --variant spectral\n"
            "  python benchmarks/calibration_analysis.py "
            "--timing --variant spectral\n"
        ),
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="benchmarks/results/calibration_results.json",
        help=(
            "Path to calibration results JSON "
            "(default: benchmarks/results/calibration_results.json)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/figures",
        help=(
            "Output directory for figures and tables "
            "(default: benchmarks/figures)"
        ),
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help=(
            "Comma-separated output formats "
            "(default: png). Use 'png,pdf' for publication."
        ),
    )
    parser.add_argument(
        "--n-regions",
        type=str,
        default=None,
        help=(
            "Comma-separated network sizes to filter "
            "(default: all found in results)"
        ),
    )
    parser.add_argument(
        "--violin",
        action="store_true",
        help=(
            "Generate posterior violin plots (CAL-04). "
            "Re-runs SVI on 1 dataset per guide type."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Generate timing breakdown figures (CAL-05). "
            "Profiles SVI step decomposition."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="spectral",
        help=(
            "DCM variant for --violin/--timing analysis "
            "(default: spectral). Only 'spectral' currently "
            "supported."
        ),
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=None,
        help=(
            "Path to fixtures directory for --violin/--timing. "
            "If not provided, generates data inline."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --violin/--timing (default: 42).",
    )

    args = parser.parse_args()

    fmt = tuple(f.strip() for f in args.formats.split(","))

    n_filter = None
    if args.n_regions is not None:
        n_filter = [
            int(x.strip()) for x in args.n_regions.split(",")
        ]

    # Default n_regions for supplementary analysis
    supp_n = 3
    if n_filter and len(n_filter) > 0:
        supp_n = n_filter[0]

    all_artifacts: list[str] = []

    # Default mode: figures from JSON results
    generate_calibration_figures(
        results_path=args.results_path,
        output_dir=args.output_dir,
        formats=fmt,
        n_regions_filter=n_filter,
    )
    all_artifacts.append("calibration_curves, tables, scaling, pareto")

    # Supplementary: violin plots (CAL-04)
    if args.violin:
        print("\n--- Generating violin plots (CAL-04) ---\n")
        generate_violin_plots(
            variant=args.variant,
            n_regions=supp_n,
            fixtures_dir=args.fixtures_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            formats=fmt,
        )
        all_artifacts.append(
            f"posterior_violins_N{supp_n}",
        )

    # Supplementary: timing breakdown (CAL-05)
    if args.timing:
        print("\n--- Generating timing figures (CAL-05) ---\n")
        generate_timing_figures(
            variant=args.variant,
            n_regions=supp_n,
            fixtures_dir=args.fixtures_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            formats=fmt,
        )
        all_artifacts.append("timing_breakdown")
        all_artifacts.append(
            f"timing_data_N{supp_n}.json",
        )

    # Final summary
    out = os.path.abspath(args.output_dir)
    print(f"\n{'=' * 60}")
    print("All generated artifacts:")
    for art in all_artifacts:
        print(f"  {out}/{art}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
