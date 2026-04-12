#!/usr/bin/env python
"""Calibration analysis sweep orchestrator for Phase 11.

Runs tiered combinatorial benchmarks across DCM variants, guide types,
ELBO objectives, and network sizes. Produces structured JSON results
with multi-level coverage data for downstream analysis and plotting.

Usage
-----
Tier 1 (quick smoke test -- all guides, spectral, N=3)::

    python benchmarks/calibration_sweep.py --tier 1 --quick

Tier 2 (ELBO comparison -- mean-field guides, spectral, N=3,5)::

    python benchmarks/calibration_sweep.py --tier 2

Tier 3 (full cross-method -- all variants, all sizes)::

    python benchmarks/calibration_sweep.py --tier 3

Resume interrupted run::

    python benchmarks/calibration_sweep.py --tier 3 --resume
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
from typing import Any

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.runners import RUNNER_REGISTRY

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# All SVI guide types supported by create_guide
GUIDE_TYPES: list[str] = [
    "auto_delta",
    "auto_normal",
    "auto_lowrank_mvn",
    "auto_mvn",
    "auto_iaf",
    "auto_laplace",
]

# Mean-field-only guide types (for N=10 where structured guides OOM)
MEAN_FIELD_GUIDE_TYPES: list[str] = [
    "auto_delta",
    "auto_normal",
]

# Valid (guide_type, elbo_type) combinations.
# TraceMeanField_ELBO only valid with mean-field guides (research S2.3).
VALID_GUIDE_ELBO: list[tuple[str, str]] = [
    ("auto_delta", "trace_elbo"),
    ("auto_delta", "tracemeanfield_elbo"),
    ("auto_delta", "renyi_elbo"),
    ("auto_normal", "trace_elbo"),
    ("auto_normal", "tracemeanfield_elbo"),
    ("auto_normal", "renyi_elbo"),
    ("auto_lowrank_mvn", "trace_elbo"),
    ("auto_lowrank_mvn", "renyi_elbo"),
    ("auto_mvn", "trace_elbo"),
    ("auto_mvn", "renyi_elbo"),
    ("auto_iaf", "trace_elbo"),
    ("auto_iaf", "renyi_elbo"),
    ("auto_laplace", "trace_elbo"),
    ("auto_laplace", "renyi_elbo"),
]

# Maximum N for auto_mvn (memory constraint, STATE.md risk P6)
_MAX_N_AUTO_MVN = 7

# Tiered configuration groups
TIER_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "1": [
        # All guides x trace_elbo x spectral x N=3
        {
            "variants": ["spectral"],
            "n_regions_list": [3],
            "guide_elbo_pairs": [
                (g, "trace_elbo") for g in GUIDE_TYPES
            ],
        },
    ],
    "2": [
        # Mean-field guides x 3 ELBOs x spectral x N=3,5
        {
            "variants": ["spectral"],
            "n_regions_list": [3, 5],
            "guide_elbo_pairs": [
                (g, e)
                for g in MEAN_FIELD_GUIDE_TYPES
                for e in [
                    "trace_elbo",
                    "tracemeanfield_elbo",
                    "renyi_elbo",
                ]
            ],
        },
    ],
    "3": [
        # All guides x trace_elbo x SVI variants x N=3,5
        {
            "variants": ["task", "spectral"],
            "n_regions_list": [3, 5],
            "guide_elbo_pairs": [
                (g, "trace_elbo") for g in GUIDE_TYPES
            ],
        },
        # Mean-field + SVI at N=10 (structured guides excluded)
        {
            "variants": ["task", "spectral"],
            "n_regions_list": [10],
            "guide_elbo_pairs": [
                (g, "trace_elbo")
                for g in MEAN_FIELD_GUIDE_TYPES
            ],
        },
        # rDCM at all sizes
        {
            "variants": ["rdcm_rigid", "rdcm_sparse"],
            "n_regions_list": [3, 5, 10],
            "guide_elbo_pairs": [("vb", "na")],
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_hash() -> str:
    """Get current git commit hash.

    Returns
    -------
    str
        Short git commit hash, or ``"unknown"`` if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _convert_for_json(obj: object) -> object:
    """Convert non-serializable objects for JSON output.

    Parameters
    ----------
    obj : object
        Value to convert.

    Returns
    -------
    object
        JSON-serializable version.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {
            str(k): _convert_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    return str(obj)


def _make_result_key(
    variant: str,
    guide_type: str,
    elbo_type: str,
    n_regions: int,
) -> str:
    """Build canonical result key.

    Parameters
    ----------
    variant : str
        DCM variant (e.g., ``"spectral"``, ``"rdcm_rigid"``).
    guide_type : str
        Guide type string.
    elbo_type : str
        ELBO type string.
    n_regions : int
        Number of regions.

    Returns
    -------
    str
        Key like ``"spectral_auto_normal_trace_elbo_3"``.
    """
    return f"{variant}_{guide_type}_{elbo_type}_{n_regions}"


def _expand_tier(tier: str) -> list[tuple[str, str, str, int]]:
    """Expand a tier into flat (variant, guide, elbo, N) tuples.

    Parameters
    ----------
    tier : str
        Tier identifier: ``"1"``, ``"2"``, ``"3"``, or ``"all"``.

    Returns
    -------
    list of tuple[str, str, str, int]
        Flat list of configuration tuples.
    """
    if tier == "all":
        tiers_to_expand = ["1", "2", "3"]
    else:
        tiers_to_expand = [tier]

    seen: set[tuple[str, str, str, int]] = set()
    configs: list[tuple[str, str, str, int]] = []

    for t in tiers_to_expand:
        for group in TIER_CONFIGS[t]:
            for variant in group["variants"]:
                for n_reg in group["n_regions_list"]:
                    for g, e in group["guide_elbo_pairs"]:
                        key = (variant, g, e, n_reg)
                        if key not in seen:
                            seen.add(key)
                            configs.append(key)

    return configs


def _should_skip(
    variant: str,
    guide_type: str,
    elbo_type: str,
    n_regions: int,
) -> str | None:
    """Check if a config should be skipped.

    Parameters
    ----------
    variant : str
        DCM variant.
    guide_type : str
        Guide type.
    elbo_type : str
        ELBO type.
    n_regions : int
        Number of regions.

    Returns
    -------
    str or None
        Reason string if should skip, ``None`` if valid.
    """
    # rDCM uses its own VB; skip SVI guide combos
    if variant.startswith("rdcm") and guide_type != "vb":
        return "rDCM uses VB, not SVI guides"

    # SVI variants should not use rDCM's "vb"/"na"
    if not variant.startswith("rdcm") and guide_type == "vb":
        return "SVI variants require SVI guide types"

    # auto_mvn blocked at large N (memory)
    if guide_type == "auto_mvn" and n_regions > _MAX_N_AUTO_MVN:
        return f"auto_mvn blocked at N>{_MAX_N_AUTO_MVN} (OOM)"

    # Validate guide-ELBO combination for SVI
    if not variant.startswith("rdcm"):
        if (guide_type, elbo_type) not in VALID_GUIDE_ELBO:
            return (
                f"Invalid guide-ELBO pair: "
                f"({guide_type}, {elbo_type})"
            )

    return None


def _get_runner_key(
    variant: str,
) -> tuple[str, str]:
    """Map variant to RUNNER_REGISTRY key.

    Parameters
    ----------
    variant : str
        DCM variant.

    Returns
    -------
    tuple[str, str]
        ``(variant, method)`` key for RUNNER_REGISTRY.
    """
    if variant in ("task", "spectral"):
        return (variant, "svi")
    if variant in ("rdcm_rigid", "rdcm_sparse"):
        return (variant, "vb")
    msg = f"Unknown variant: {variant}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------


def run_calibration_sweep(
    tier: str,
    quick: bool = False,
    fixtures_dir: str | None = None,
    output_dir: str = "benchmarks/results",
    resume: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Run tiered calibration sweep.

    Parameters
    ----------
    tier : str
        Tier level: ``"1"``, ``"2"``, ``"3"``, or ``"all"``.
    quick : bool, optional
        Use reduced datasets/steps for development.
    fixtures_dir : str or None, optional
        Path to shared fixtures directory.
    output_dir : str, optional
        Output directory for results JSON.
    resume : bool, optional
        Skip result keys already in existing output.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict[str, Any]
        Results dictionary keyed by result key.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, "calibration_results.json",
    )

    # Load existing results for resume
    results: dict[str, Any] = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(
            f"Loaded {len(results) - 1} existing results "
            f"(resume mode)"
        )

    # Expand tier to flat config list
    all_configs = _expand_tier(tier)
    total = len(all_configs)
    print(
        f"Calibration sweep: tier={tier}, "
        f"{total} configurations, "
        f"mode={'quick' if quick else 'full'}"
    )
    print()

    completed = 0
    skipped = 0

    for idx, (variant, guide_type, elbo_type, n_reg) in enumerate(
        all_configs, 1,
    ):
        result_key = _make_result_key(
            variant, guide_type, elbo_type, n_reg,
        )

        # Skip if already completed (resume)
        if resume and result_key in results:
            print(
                f"[{idx}/{total}] {result_key}: "
                f"SKIP (resume)"
            )
            skipped += 1
            continue

        # Check if config should be skipped
        skip_reason = _should_skip(
            variant, guide_type, elbo_type, n_reg,
        )
        if skip_reason is not None:
            print(
                f"[{idx}/{total}] {result_key}: "
                f"SKIP ({skip_reason})"
            )
            skipped += 1
            continue

        # Build config
        runner_key = _get_runner_key(variant)
        runner = RUNNER_REGISTRY.get(runner_key)
        if runner is None:
            print(
                f"[{idx}/{total}] {result_key}: "
                f"SKIP (no runner for {runner_key})"
            )
            skipped += 1
            continue

        if quick:
            config = BenchmarkConfig.quick_config(
                variant, runner_key[1],
            )
        else:
            config = BenchmarkConfig.full_config(
                variant, runner_key[1],
            )

        config.seed = seed
        config.n_regions = n_reg
        config.guide_type = guide_type
        config.elbo_type = elbo_type
        config.output_dir = output_dir
        if fixtures_dir is not None:
            config.fixtures_dir = fixtures_dir

        print(
            f"[{idx}/{total}] {result_key}: "
            f"running ({config.n_datasets} datasets, "
            f"{config.n_svi_steps} steps)..."
        )

        try:
            result = runner(config)
            result["status"] = "completed"
            results[result_key] = result
            completed += 1

            # Extract summary for progress line
            rmse_med = "N/A"
            cov90_med = "N/A"
            if "rmse_stats" in result:
                rmse_med = f"{result['rmse_stats']['median']:.4f}"
            elif "mean_rmse" in result:
                rmse_med = f"{result['mean_rmse']:.4f}"
            if "coverage_multi" in result:
                cov90_list = result["coverage_multi"].get(
                    "0.9", [],
                )
                if cov90_list:
                    import statistics

                    cov90_med = (
                        f"{statistics.median(cov90_list):.3f}"
                    )
            elif "mean_coverage" in result:
                cov90_med = f"{result['mean_coverage']:.3f}"

            print(
                f"  -> RMSE={rmse_med}, "
                f"cov@90={cov90_med}"
            )

        except Exception as e:
            print(f"  -> FAILED: {e}")
            results[result_key] = {
                "status": "failed",
                "error": str(e),
            }

        # Save intermediate (resume safety)
        _save_results(results, output_path, tier, quick, seed)

    # Final save with metadata
    _save_results(results, output_path, tier, quick, seed)

    # Print summary table
    _print_summary_table(results)

    print(
        f"\nSweep complete: {completed} completed, "
        f"{skipped} skipped, "
        f"{total - completed - skipped} failed"
    )
    print(f"Results saved to {output_path}")

    return results


def _save_results(
    results: dict[str, Any],
    output_path: str,
    tier: str,
    quick: bool,
    seed: int,
) -> None:
    """Save results with metadata to JSON.

    Parameters
    ----------
    results : dict
        Results dictionary.
    output_path : str
        Output file path.
    tier : str
        Tier used.
    quick : bool
        Whether quick mode was used.
    seed : int
        Random seed.
    """
    output = dict(results)
    output["metadata"] = {
        "timestamp": datetime.now(
            tz=timezone.utc,
        ).isoformat(),
        "git_commit": _get_git_hash(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "tier": tier,
        "quick": quick,
        "seed": seed,
    }
    with open(output_path, "w") as f:
        json.dump(_convert_for_json(output), f, indent=2)


def _print_summary_table(results: dict[str, Any]) -> None:
    """Print summary table of sweep results.

    Parameters
    ----------
    results : dict
        Results dictionary.
    """
    print("\n=== Calibration Sweep Summary ===\n")

    header = (
        f"{'Result Key':<45} {'Status':<12} "
        f"{'RMSE med':<10} {'Cov@90 med':<12} "
        f"{'Corr med':<10} {'Time med':<10}"
    )
    print(header)
    print("-" * len(header))

    for key, val in sorted(results.items()):
        if key == "metadata":
            continue
        if not isinstance(val, dict):
            continue

        status = val.get("status", "unknown")
        if status != "completed":
            print(f"{key:<45} {status:<12}")
            continue

        rmse = _extract_median(val, "rmse_stats", "mean_rmse")
        cov = _extract_cov90_median(val)
        corr = _extract_median(
            val, "correlation_stats", "mean_correlation",
        )
        t = _extract_median(val, "time_stats", "mean_time")

        print(
            f"{key:<45} {status:<12} "
            f"{rmse:<10} {cov:<12} {corr:<10} {t:<10}"
        )


def _extract_median(
    val: dict,
    stats_key: str,
    fallback_key: str,
) -> str:
    """Extract median from stats dict or fall back to mean.

    Parameters
    ----------
    val : dict
        Result dictionary.
    stats_key : str
        Key for stats sub-dict.
    fallback_key : str
        Fallback key for plain float.

    Returns
    -------
    str
        Formatted value string.
    """
    if stats_key in val:
        return f"{val[stats_key]['median']:.4f}"
    if fallback_key in val:
        v = val[fallback_key]
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
    return "N/A"


def _extract_cov90_median(val: dict) -> str:
    """Extract median coverage at 90% CI.

    Parameters
    ----------
    val : dict
        Result dictionary.

    Returns
    -------
    str
        Formatted value string.
    """
    if "coverage_multi" in val:
        cov90_list = val["coverage_multi"].get("0.9", [])
        if cov90_list:
            import statistics

            return f"{statistics.median(cov90_list):.4f}"
    if "coverage_stats" in val:
        return f"{val['coverage_stats']['median']:.4f}"
    if "mean_coverage" in val:
        return f"{val['mean_coverage']:.4f}"
    return "N/A"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run calibration sweep from CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 11 calibration analysis sweep orchestrator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/calibration_sweep.py "
            "--tier 1 --quick\n"
            "  python benchmarks/calibration_sweep.py "
            "--tier 3 --resume\n"
            "  python benchmarks/calibration_sweep.py "
            "--tier all\n"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="1",
        help="Tier level (default: 1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced datasets/steps for development",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing result keys in output JSON",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=None,
        help="Path to shared fixtures directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    run_calibration_sweep(
        tier=args.tier,
        quick=args.quick,
        fixtures_dir=args.fixtures_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
