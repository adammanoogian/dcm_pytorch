#!/usr/bin/env python
"""Run DCM parameter recovery benchmarks.

CLI entry point for executing benchmark runners across task, spectral,
and regression DCM variants. Dispatches to registered runners via the
``RUNNER_REGISTRY`` and collects results into a JSON file.

Usage
-----
Quick CI run (reduced parameters)::

    python benchmarks/run_all_benchmarks.py --quick

Full paper-quality run for all variants::

    python benchmarks/run_all_benchmarks.py

Specific variant and method::

    python benchmarks/run_all_benchmarks.py --variant spectral --method svi

All rDCM benchmarks (rigid + sparse)::

    python benchmarks/run_all_benchmarks.py --variant rdcm

With shared fixtures::

    python benchmarks/run_all_benchmarks.py --fixtures-dir benchmarks/fixtures

Multi-region benchmark::

    python benchmarks/run_all_benchmarks.py --n-regions 3,5,10 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.runners import RUNNER_REGISTRY


# Valid (variant, method) combinations
VALID_COMBOS: list[tuple[str, str]] = [
    ("task", "svi"),
    ("task", "amortized"),
    ("spectral", "svi"),
    ("spectral", "amortized"),
    ("rdcm_rigid", "vb"),
    ("rdcm_sparse", "vb"),
    ("spm", "reference"),
]

# Mapping from user-facing variant to registry variants
VARIANT_EXPANSION: dict[str, list[str]] = {
    "task": ["task"],
    "spectral": ["spectral"],
    "rdcm": ["rdcm_rigid", "rdcm_sparse"],
    "spm": ["spm"],
    "all": ["task", "spectral", "rdcm_rigid", "rdcm_sparse", "spm"],
}

METHOD_EXPANSION: dict[str, list[str]] = {
    "svi": ["svi"],
    "amortized": ["amortized"],
    "vb": ["vb"],
    "all": ["svi", "amortized", "vb", "reference"],
}


def _get_git_hash() -> str:
    """Get current git commit hash.

    Returns
    -------
    str
        Short git commit hash, or ``"unknown"`` if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _get_combos(
    variant_arg: str,
    method_arg: str,
) -> list[tuple[str, str]]:
    """Expand user arguments into valid (variant, method) pairs.

    Parameters
    ----------
    variant_arg : str
        User-specified variant (may be ``"all"`` or ``"rdcm"``).
    method_arg : str
        User-specified method (may be ``"all"``).

    Returns
    -------
    list of tuple[str, str]
        Valid (variant, method) pairs to execute.
    """
    variants = VARIANT_EXPANSION.get(variant_arg, [variant_arg])
    methods = METHOD_EXPANSION.get(method_arg, [method_arg])

    combos = []
    for v in variants:
        for m in methods:
            if (v, m) in VALID_COMBOS:
                combos.append((v, m))

    return combos


def _convert_for_json(obj: object) -> object:
    """Convert torch tensors and other non-serializable objects.

    Parameters
    ----------
    obj : object
        Value to convert for JSON serialization.

    Returns
    -------
    object
        JSON-serializable version of the input.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    return str(obj)


def _format_results_table(results: dict[str, object]) -> str:
    """Format results as a text table.

    Parameters
    ----------
    results : dict
        Benchmark results dictionary.

    Returns
    -------
    str
        Formatted table string.
    """
    try:
        from tabulate import tabulate

        rows = []
        for key, val in results.items():
            if key == "metadata":
                continue
            if isinstance(val, dict):
                status = val.get("status", "unknown")
                rmse = val.get("mean_rmse", "N/A")
                coverage = val.get("mean_coverage", "N/A")
                corr = val.get("mean_correlation", "N/A")
                if isinstance(rmse, float):
                    rmse = f"{rmse:.4f}"
                if isinstance(coverage, float):
                    coverage = f"{coverage:.4f}"
                if isinstance(corr, float):
                    corr = f"{corr:.4f}"
                rows.append([key, status, rmse, coverage, corr])

        if rows:
            return tabulate(
                rows,
                headers=["Variant", "Status", "RMSE", "Coverage", "Corr"],
                tablefmt="simple",
            )
    except ImportError:
        pass

    # Fallback: manual formatting
    lines = [
        f"{'Variant':<20} {'Status':<15} {'RMSE':<10} "
        f"{'Coverage':<10} {'Corr':<10}",
        "-" * 65,
    ]
    for key, val in results.items():
        if key == "metadata":
            continue
        if isinstance(val, dict):
            status = str(val.get("status", "unknown"))
            rmse = val.get("mean_rmse", "N/A")
            coverage = val.get("mean_coverage", "N/A")
            corr = val.get("mean_correlation", "N/A")
            if isinstance(rmse, float):
                rmse = f"{rmse:.4f}"
            if isinstance(coverage, float):
                coverage = f"{coverage:.4f}"
            if isinstance(corr, float):
                corr = f"{corr:.4f}"
            lines.append(
                f"{key:<20} {status:<15} {str(rmse):<10} "
                f"{str(coverage):<10} {str(corr):<10}"
            )

    return "\n".join(lines)


def main() -> None:
    """Run benchmarks based on CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run DCM parameter recovery benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/run_all_benchmarks.py --quick\n"
            "  python benchmarks/run_all_benchmarks.py "
            "--variant spectral --method svi\n"
            "  python benchmarks/run_all_benchmarks.py "
            "--variant rdcm --quick\n"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["task", "spectral", "rdcm", "all"],
        default="all",
        help="DCM variant to benchmark (default: all)",
    )
    parser.add_argument(
        "--method",
        choices=["svi", "amortized", "vb", "all"],
        default="all",
        help="Inference method to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced dataset counts and SVI steps for CI/dev",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for saving results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=None,
        help=(
            "Load data from pre-generated fixtures instead "
            "of inline generation"
        ),
    )
    parser.add_argument(
        "--guide-type",
        type=str,
        default="mean_field",
        help="Guide type for inference (default: mean_field)",
    )
    parser.add_argument(
        "--n-regions",
        type=str,
        default="3",
        help=(
            "Comma-separated region sizes (default: 3)"
        ),
    )

    args = parser.parse_args()

    # Determine which combos to run
    combos = _get_combos(args.variant, args.method)

    if not combos:
        print(
            f"No valid combinations for --variant={args.variant} "
            f"--method={args.method}"
        )
        print(f"Valid combinations: {VALID_COMBOS}")
        sys.exit(1)

    print(f"Benchmarks to run: {combos}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print()

    # Run each combo
    results: dict[str, object] = {}

    for variant, method in combos:
        label = f"{variant}/{method}"
        runner = RUNNER_REGISTRY.get((variant, method))

        if runner is None:
            print(f"[SKIP] {label}: no runner registered")
            results[variant] = {"status": "not_registered"}
            continue

        # Build config
        if args.quick:
            config = BenchmarkConfig.quick_config(variant, method)
        else:
            config = BenchmarkConfig.full_config(variant, method)

        config.seed = args.seed
        config.output_dir = args.output_dir
        config.save_figures = not args.no_figures
        config.fixtures_dir = args.fixtures_dir
        config.guide_type = args.guide_type
        n_regions_list = [
            int(x) for x in args.n_regions.split(",")
        ]
        config.n_regions_list = n_regions_list
        config.n_regions = n_regions_list[0]

        print(f"[RUN]  {label} (n_datasets={config.n_datasets}, "
              f"n_svi_steps={config.n_svi_steps})")

        try:
            result = runner(config)
            results[variant] = result
            results[variant]["status"] = "completed"  # type: ignore[index]
            print(f"[DONE] {label}")
        except NotImplementedError as e:
            print(f"[SKIP] {label}: {e}")
            results[variant] = {"status": "not_implemented", "message": str(e)}
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            results[variant] = {"status": "failed", "error": str(e)}

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _get_git_hash(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "args": {
            "variant": args.variant,
            "method": args.method,
            "quick": args.quick,
            "seed": args.seed,
        },
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(_convert_for_json(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== Benchmark Summary ===\n")
    print(_format_results_table(results))

    # Generate figures
    if not args.no_figures:
        try:
            from benchmarks.plotting import generate_all_figures

            figures_dir = os.path.join(
                os.path.dirname(args.output_dir), "figures",
            )
            generate_all_figures(results, output_dir=figures_dir)
        except ImportError:
            print(
                "\n[WARN] matplotlib not available, "
                "skipping figure generation"
            )
        except Exception as e:
            print(f"\n[WARN] Figure generation failed: {e}")


if __name__ == "__main__":
    main()
