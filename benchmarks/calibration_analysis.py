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
    plot_scaling_study,
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

    # Summary
    print(f"\nGenerated {len(generated)} outputs:")
    for name in generated:
        print(f"  - {name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run calibration analysis from CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate calibration analysis figures and tables "
            "from sweep results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/calibration_analysis.py\n"
            "  python benchmarks/calibration_analysis.py "
            "--formats png,pdf\n"
            "  python benchmarks/calibration_analysis.py "
            "--n-regions 3,5\n"
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

    args = parser.parse_args()

    fmt = tuple(f.strip() for f in args.formats.split(","))

    n_filter = None
    if args.n_regions is not None:
        n_filter = [
            int(x.strip()) for x in args.n_regions.split(",")
        ]

    generate_calibration_figures(
        results_path=args.results_path,
        output_dir=args.output_dir,
        formats=fmt,
        n_regions_filter=n_filter,
    )


if __name__ == "__main__":
    main()
