"""Fixture loading helpers for benchmark runners.

Provides ``load_fixture`` and ``get_fixture_count`` for loading
pre-generated .npz fixture files produced by ``generate_fixtures.py``.

Fixture directory layout::

    benchmarks/fixtures/{variant}_{N}region/dataset_{NNN}.npz

Where ``variant`` is one of ``task``, ``spectral``, ``rdcm``; ``N``
is the number of regions (3, 5, or 10); and ``NNN`` is a zero-padded
dataset index.

Complex arrays (CSD, noisy CSD) are stored as separate real/imag
parts in .npz files and automatically reconstructed on load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Pairs of (real_key, imag_key) -> reconstructed_key
_COMPLEX_PAIRS: list[tuple[str, str, str]] = [
    ("csd_real", "csd_imag", "csd"),
    ("noisy_csd_real", "noisy_csd_imag", "noisy_csd"),
    ("X_real", "X_imag", "X"),
    ("Y_real", "Y_imag", "Y"),
]


def load_fixture(
    variant: str,
    n_regions: int,
    index: int,
    fixtures_dir: str = "benchmarks/fixtures",
) -> dict[str, torch.Tensor]:
    """Load a single .npz fixture as a dict of torch tensors.

    Parameters
    ----------
    variant : str
        One of ``"task"``, ``"spectral"``, ``"rdcm"``.
    n_regions : int
        Number of regions (3, 5, or 10).
    index : int
        Dataset index (0-based).
    fixtures_dir : str
        Root fixtures directory.

    Returns
    -------
    dict[str, torch.Tensor]
        Tensors keyed by field name. Complex arrays (CSD, rDCM
        regressors) are reconstructed from real/imag parts.

    Raises
    ------
    FileNotFoundError
        If the fixture file does not exist.
    """
    subdir = f"{variant}_{n_regions}region"
    filename = f"dataset_{index:03d}.npz"
    path = Path(fixtures_dir) / subdir / filename

    if not path.exists():
        msg = (
            f"Fixture not found: {path}. "
            f"Run generate_fixtures.py to create fixtures."
        )
        raise FileNotFoundError(msg)

    data = np.load(str(path), allow_pickle=False)
    result: dict[str, torch.Tensor] = {}

    for key in data.files:
        arr = data[key]
        result[key] = torch.from_numpy(arr)

    # Reconstruct complex tensors from real/imag pairs
    for real_key, imag_key, complex_key in _COMPLEX_PAIRS:
        if real_key in result and imag_key in result:
            real_part = result[real_key].to(torch.float64)
            imag_part = result[imag_key].to(torch.float64)
            result[complex_key] = torch.complex(real_part, imag_part)

    return result


def get_fixture_count(
    variant: str,
    n_regions: int,
    fixtures_dir: str = "benchmarks/fixtures",
) -> int:
    """Get the number of fixture datasets for a variant and size.

    Reads the ``manifest.json`` in the fixture subdirectory. Falls
    back to counting ``.npz`` files if manifest is missing.

    Parameters
    ----------
    variant : str
        One of ``"task"``, ``"spectral"``, ``"rdcm"``.
    n_regions : int
        Number of regions (3, 5, or 10).
    fixtures_dir : str
        Root fixtures directory.

    Returns
    -------
    int
        Number of available fixture datasets.

    Raises
    ------
    FileNotFoundError
        If the fixture subdirectory does not exist.
    """
    subdir = Path(fixtures_dir) / f"{variant}_{n_regions}region"

    if not subdir.exists():
        msg = (
            f"Fixture directory not found: {subdir}. "
            f"Run generate_fixtures.py to create fixtures."
        )
        raise FileNotFoundError(msg)

    manifest_path = subdir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return int(manifest["n_datasets"])

    # Fallback: count .npz files
    return len(list(subdir.glob("dataset_*.npz")))
