"""Injected-CSD round-trip regression tests (VLSPM, pitfall S4).

These are the S4 teeth for the INJECTED CSD path specifically. They prove the
``(F, N, N)`` complex array written by
``validation.export_to_mat.export_spectral_dcm_csd_for_spm`` reloads through
``scipy.io.loadmat`` with element ``[w, i, j]`` preserved and its asymmetric
off-diagonal structure intact -- catching a transpose bug at export time, before
any MATLAB ``spm_dcm_fmri_csd`` run (Plan 32-03).

The asymmetric ground truth here ties to the matched-problem reciprocal-but-
ASYMMETRIC edge strengths used in Plan 32-03 (A[0, 1] = 0.15 vs A[1, 0] = 0.10):
spectral DCM cannot identify a lone off-diagonal A entry, so the cross-validation
uses a reciprocal-asymmetric ground truth, and a silent transpose of the CSD
would swap those two edges. ``loaded[0, 0, 1] != loaded[0, 1, 0]`` is the guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from validation.export_to_mat import export_spectral_dcm_csd_for_spm

pytestmark = pytest.mark.vl


def _asymmetric_csd(num_freqs: int, num_regions: int) -> np.ndarray:
    """Build a deterministic asymmetric complex CSD of shape (F, N, N).

    ``csd[w, i, j] = (w + 1) * 100 + i * 10 + j + 1j * (i - j)``. The real part
    is asymmetric (``csd[w, i, j] != csd[w, j, i]``) and the imaginary part is
    antisymmetric -- exactly the structure a transposition bug would corrupt.
    Mirrors ``tests/test_csd_corder_roundtrip.py::_asymmetric_csd``.

    Parameters
    ----------
    num_freqs : int
        Number of frequency bins F.
    num_regions : int
        Number of regions N.

    Returns
    -------
    np.ndarray
        Complex128 CSD of shape ``(num_freqs, num_regions, num_regions)``.
    """
    w = np.arange(num_freqs).reshape(num_freqs, 1, 1)
    i = np.arange(num_regions).reshape(1, num_regions, 1)
    j = np.arange(num_regions).reshape(1, 1, num_regions)
    real = (w + 1) * 100 + i * 10 + j
    imag = i - j
    return (real + 1j * imag).astype(np.complex128)


def _load_csd(mat_path: Path) -> np.ndarray:
    """Reload the injected ``DCM.Y.csd`` array from a saved .mat file.

    Parameters
    ----------
    mat_path : Path
        Path to the .mat file written by ``export_spectral_dcm_csd_for_spm``.

    Returns
    -------
    np.ndarray
        The reloaded CSD array of shape ``(F, N, N)``.
    """
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=False)
    dcm = mat["DCM"]
    y = dcm["Y"][0, 0]
    return np.asarray(y["csd"][0, 0])


def test_injected_csd_roundtrips_asymmetric(tmp_path: Path) -> None:
    """Exported asymmetric CSD reloads element-identical, no transpose.

    Asserts ``loaded[w, i, j] == csd[w, i, j]`` for every element AND that the
    ``(0, 0, 1)`` / ``(0, 1, 0)`` off-diagonal slots stay distinct, so a
    transpose bug fails the test (pitfall S4).
    """
    num_freqs, num_regions = 4, 2
    csd = _asymmetric_csd(num_freqs, num_regions)
    freqs = np.linspace(0.01, 0.25, num_freqs)
    a_mask = np.ones((num_regions, num_regions))
    c_mask = np.ones((num_regions, 1))

    out = tmp_path / "csd_injection.mat"
    export_spectral_dcm_csd_for_spm(
        observed_csd=csd,
        freqs=freqs,
        a_mask=a_mask,
        c_mask=c_mask,
        TR=2.0,
        output_path=str(out),
    )

    loaded = _load_csd(out)

    assert loaded.shape == csd.shape, (
        f"shape mismatch: expected {csd.shape}, got {loaded.shape}"
    )
    assert np.allclose(loaded.real, csd.real), "real part not preserved"
    assert np.allclose(loaded.imag, csd.imag), "imag part not preserved"

    # Transpose guard: the (i=0, j=1) and (i=1, j=0) slots must stay distinct.
    assert loaded[0, 0, 1] != loaded[0, 1, 0], (
        "asymmetry lost: a transpose bug collapsed off-diagonal CSD elements"
    )


def test_injected_freqs_roundtrip(tmp_path: Path) -> None:
    """Exported ``DCM.Y.Hz`` reloads equal to the injected freqs (float64)."""
    num_freqs, num_regions = 4, 2
    csd = _asymmetric_csd(num_freqs, num_regions)
    freqs = np.linspace(0.01, 0.25, num_freqs)
    a_mask = np.ones((num_regions, num_regions))
    c_mask = np.ones((num_regions, 1))

    out = tmp_path / "csd_injection_freqs.mat"
    export_spectral_dcm_csd_for_spm(
        observed_csd=csd,
        freqs=freqs,
        a_mask=a_mask,
        c_mask=c_mask,
        TR=2.0,
        output_path=str(out),
    )

    mat = scipy.io.loadmat(str(out), squeeze_me=False)
    y = mat["DCM"]["Y"][0, 0]
    loaded_hz = np.asarray(y["Hz"][0, 0]).reshape(-1)

    assert loaded_hz.dtype == np.float64, (
        f"Hz dtype: expected float64, got {loaded_hz.dtype}"
    )
    assert np.allclose(loaded_hz, freqs), "frequency grid not preserved"
