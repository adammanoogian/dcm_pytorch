"""Repo-wide path constants (single source of truth).

Central path config for `scripts/`, `benchmarks/`, `validation/`, and
`cluster/scripts/`. All directory constants derive from :data:`PROJECT_ROOT`,
so they resolve correctly wherever the repo is checked out (laptop or the
Monash M3 cluster) without per-machine hardcoding.

Naming note (CCDS v2 alignment)
-------------------------------
The constant names follow the CCDS v2 ``{PARENT}_{CHILD}_DIR`` style, but point
at this repo's ESTABLISHED directory layout rather than the CCDS canonical
``reports/`` / ``models/`` tree: this project uses ``results/`` (not
``reports/``), a top-level ``figures/`` (not ``reports/figures/``), and
``checkpoints/`` (not ``models/``). The physical layout is intentionally left
unchanged; these constants name what already exists so call sites can stop
hardcoding literals. Absent/unused CCDS dirs (``data/raw`` etc.) are omitted
until something needs them.

Constants
---------
PROJECT_ROOT : pathlib.Path
    Absolute path to the repository root (parent of this file).
DATA_DIR : pathlib.Path
    Parent of input/generated data (``data/``).
DATA_TRAINING_DIR : pathlib.Path
    Generated amortized-guide training tensors (``data/training/``; created on
    demand by ``scripts/generate_training_data.py``).
RESULTS_DIR : pathlib.Path
    Analysis/fitting outputs (``results/``).
FIGURES_DIR : pathlib.Path
    Top-level publication/diagnostic figures (``figures/``).
CHECKPOINTS_DIR : pathlib.Path
    Fitted model artifacts (``checkpoints/``).
CLUSTER_DIR : pathlib.Path
    SLURM + Snakemake infrastructure (``cluster/``).
CLUSTER_RESULTS_DIR : pathlib.Path
    Cluster job outputs (``cluster/results/``).
CLUSTER_LOGS_DIR : pathlib.Path
    SLURM ``.out`` / ``.err`` logs (``cluster/logs/``).
BENCHMARK_RESULTS_DIR : pathlib.Path
    Default directory for benchmark JSON / CSV outputs.
BENCHMARK_FIGURES_DIR : pathlib.Path
    Default directory for benchmark figure outputs.
BENCHMARK_FIXTURES_DIR : pathlib.Path
    Default directory for benchmark ``.npz`` fixture caches.
TAPAS_RDCM_PATH : pathlib.Path
    Path to the local clone of ``tapas/rDCM`` (MATLAB rDCM toolbox), used by
    ``validation/run_rdcm_validation.py`` and ``validation/run_validation.py``.
    Override via the ``TAPAS_RDCM_PATH`` environment variable when running on a
    different machine. Not currently installed on this workstation.
MATLAB_PATH : pathlib.Path
    Path to the MATLAB binary used by the SPM12 cross-validation bridge
    (Phase 32, ``validation/run_vl_validation.py``). Override via the
    ``MATLAB_PATH`` environment variable when running on a different machine.
SPM12_PATH : pathlib.Path
    Path to the local SPM12 installation used by every script in
    ``validation/matlab_scripts/``. Exported into the MATLAB subprocess
    environment, where the ``.m`` files read it via ``getenv('SPM12_PATH')``.
    Override via the ``SPM12_PATH`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# --- Data ---
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_TRAINING_DIR: Path = DATA_DIR / "training"

# --- Outputs ---
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"

# --- Cluster (SLURM + Snakemake) ---
CLUSTER_DIR: Path = PROJECT_ROOT / "cluster"
CLUSTER_RESULTS_DIR: Path = CLUSTER_DIR / "results"
CLUSTER_LOGS_DIR: Path = CLUSTER_DIR / "logs"

# --- Benchmarks (isolated benchmark harness outputs) ---
BENCHMARK_RESULTS_DIR: Path = PROJECT_ROOT / "benchmarks" / "results"
BENCHMARK_FIGURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "figures"
BENCHMARK_FIXTURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "fixtures"

# --- External MATLAB toolchain (per-machine; env-overridable) ---
# Defaults describe the DCCN workstation (verified 2026-09-05: MATLAB R2025b
# holds a valid licence and SPM12 is complete, so the SPM bridge runs LOCALLY).
# The DCCN cluster has MATLAB modules but NO system SPM12 -- set SPM12_PATH
# explicitly if a SLURM job ever needs it.
TAPAS_RDCM_PATH: Path = Path(
    os.environ.get(
        "TAPAS_RDCM_PATH",
        "C:/Users/adaman/Documents/external/tapas/rDCM",
    )
)

MATLAB_PATH: Path = Path(
    os.environ.get(
        "MATLAB_PATH",
        "C:/Program Files/MATLAB/R2025b/bin/matlab",
    )
)

SPM12_PATH: Path = Path(
    os.environ.get(
        "SPM12_PATH",
        "C:/Users/adaman/Documents/external/spm12",
    )
)
