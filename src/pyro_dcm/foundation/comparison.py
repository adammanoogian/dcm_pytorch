"""Cross-modal comparison metrics for DCM effective connectivity.

Provides functions to compare A matrices estimated from different
modalities (e.g., fMRI via TRIBE v2 vs M/EEG via LaBraM).  Metrics
include Pearson correlation, sign-pattern Cohen's kappa, and
credible-interval overlap fraction.

All functions accept raw (unnormalized) A matrices.  Use
:func:`normalize_a_matrix` to make comparisons scale-independent
across modalities with different temporal resolutions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def normalize_a_matrix(a_matrix: np.ndarray) -> np.ndarray:
    """Normalize an A matrix by its Frobenius norm.

    This makes cross-modal comparison scale-independent.  fMRI
    operates at ~1 Hz while M/EEG operates at ~200 Hz, producing
    A matrices with very different magnitudes.

    Parameters
    ----------
    a_matrix : np.ndarray, shape (N, N)
        Effective connectivity matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Frobenius-normalized A matrix (unit norm).

    Raises
    ------
    ValueError
        If the matrix has zero Frobenius norm.
    """
    norm = np.linalg.norm(a_matrix, "fro")
    if norm == 0.0:
        msg = (
            "Cannot normalize A matrix with zero Frobenius norm. "
            f"Got all-zero matrix of shape {a_matrix.shape}."
        )
        raise ValueError(msg)
    return a_matrix / norm


def compute_pearson_correlation(
    a1: np.ndarray,
    a2: np.ndarray,
) -> float:
    """Compute Pearson correlation between two A matrices.

    Flattens both matrices and computes the Pearson r coefficient.
    Matrices should be normalized first via :func:`normalize_a_matrix`
    for cross-modal comparison.

    Parameters
    ----------
    a1 : np.ndarray, shape (N, N)
        First A matrix.
    a2 : np.ndarray, shape (N, N)
        Second A matrix (same shape as ``a1``).

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].

    Raises
    ------
    ValueError
        If matrices have different shapes.
    """
    if a1.shape != a2.shape:
        msg = (
            f"Shape mismatch: a1 {a1.shape} vs a2 {a2.shape}. "
            "Both matrices must have the same dimensions."
        )
        raise ValueError(msg)
    r, _ = pearsonr(a1.ravel(), a2.ravel())
    return float(r)


def compute_sign_kappa(
    a1: np.ndarray,
    a2: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Compute Cohen's kappa on sign patterns of two A matrices.

    Binarizes each element into {-1, 0, +1} based on sign, then
    computes Cohen's kappa to measure agreement beyond chance.

    Parameters
    ----------
    a1 : np.ndarray, shape (N, N)
        First A matrix.
    a2 : np.ndarray, shape (N, N)
        Second A matrix (same shape as ``a1``).
    threshold : float, optional
        Elements with absolute value below this are set to 0.
        Default is 0.0 (only exact zeros become 0).

    Returns
    -------
    float
        Cohen's kappa in [-1, 1].  1.0 = perfect agreement,
        0.0 = chance, negative = systematic disagreement.

    Raises
    ------
    ValueError
        If matrices have different shapes.
    """
    from sklearn.metrics import cohen_kappa_score

    if a1.shape != a2.shape:
        msg = (
            f"Shape mismatch: a1 {a1.shape} vs a2 {a2.shape}. "
            "Both matrices must have the same dimensions."
        )
        raise ValueError(msg)

    def _sign_pattern(a: np.ndarray) -> np.ndarray:
        signs = np.sign(a).astype(int)
        if threshold > 0.0:
            signs[np.abs(a) < threshold] = 0
        return signs.ravel()

    s1 = _sign_pattern(a1)
    s2 = _sign_pattern(a2)
    return float(cohen_kappa_score(s1, s2))


def compute_credible_interval_overlap(
    a1_mean: np.ndarray,
    a1_std: np.ndarray,
    a2_mean: np.ndarray,
    a2_std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Compute fraction of A-matrix elements with overlapping CIs.

    For each (i, j) element, constructs symmetric credible intervals
    ``[mean - z*std, mean + z*std]`` for both modalities and checks
    whether the intervals overlap.

    Parameters
    ----------
    a1_mean : np.ndarray, shape (N, N)
        Posterior mean of first A matrix.
    a1_std : np.ndarray, shape (N, N)
        Posterior std of first A matrix.
    a2_mean : np.ndarray, shape (N, N)
        Posterior mean of second A matrix.
    a2_std : np.ndarray, shape (N, N)
        Posterior std of second A matrix.
    z : float, optional
        Number of standard deviations for interval width.
        Default 1.96 (95% CI).

    Returns
    -------
    float
        Fraction of elements in [0, 1] with overlapping CIs.

    Raises
    ------
    ValueError
        If any input arrays have mismatched shapes.
    """
    shapes = {a1_mean.shape, a1_std.shape, a2_mean.shape, a2_std.shape}
    if len(shapes) > 1:
        msg = (
            "All inputs must have the same shape. "
            f"Got: {a1_mean.shape}, {a1_std.shape}, "
            f"{a2_mean.shape}, {a2_std.shape}."
        )
        raise ValueError(msg)

    lo1 = a1_mean - z * a1_std
    hi1 = a1_mean + z * a1_std
    lo2 = a2_mean - z * a2_std
    hi2 = a2_mean + z * a2_std

    # Two intervals overlap iff max(lo) <= min(hi)
    overlaps = (np.maximum(lo1, lo2) <= np.minimum(hi1, hi2)).ravel()
    return float(np.mean(overlaps))
