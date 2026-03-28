"""Result loading and comparison utilities for SPM12/tapas validation.

Provides functions to load estimation results from SPM12 and tapas .mat
files, compare posterior means element-wise using a hybrid relative/absolute
error metric, and validate model comparison rankings.

The hybrid metric uses relative error for parameters with |value| > 0.01
and absolute error for near-zero parameters, avoiding infinite relative
error for absent connections.

References
----------
SPM12 output: spm_dcm_estimate.m (Ep.A, Cp, F fields).
tapas output: tapas_rdcm_estimate.m (Ep, logF, Ip fields).
"""

from __future__ import annotations

import numpy as np
import scipy.io


def load_spm_results(mat_path: str) -> dict:
    """Load SPM12 estimation results from .mat file.

    Handles nested struct access from scipy.io.loadmat output.
    SPM12 results contain ``Ep_A``, ``Ep_C``, ``Cp``, ``F``, and
    optionally spectral-specific fields.

    Parameters
    ----------
    mat_path : str
        Path to the results .mat file saved by MATLAB batch script.

    Returns
    -------
    dict
        Dictionary with available fields:
        - ``'Ep_A'``: np.ndarray, posterior mean A matrix (free params).
        - ``'F'``: float, free energy (log evidence bound).
        - ``'Ep_C'``: np.ndarray, posterior mean C (if present).
        - ``'Cp'``: np.ndarray, full posterior covariance (if present).
        - ``'y_predicted'``: np.ndarray, predicted BOLD (if present).
        - Spectral fields: ``'Ep_transit'``, ``'Ep_decay'``, ``'Hc'``,
          ``'Hz'`` (if present).

    Notes
    -----
    scipy.io.loadmat with ``squeeze_me=False`` returns structured
    numpy arrays. Fields are accessed via ``data['results'][field][0, 0]``.
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=False)
    results_raw = data["results"]

    out: dict = {}

    # Required fields
    out["Ep_A"] = results_raw["Ep_A"][0, 0].astype(np.float64)
    out["F"] = float(results_raw["F"][0, 0].item())

    # Optional fields
    optional_fields = [
        "Ep_C", "Cp", "y_predicted", "R",
        "Ep_transit", "Ep_decay", "Hc", "Hz",
    ]
    for field in optional_fields:
        try:
            val = results_raw[field][0, 0]
            if val.size > 0:
                out[field] = val.astype(np.float64)
        except (ValueError, IndexError, KeyError):
            pass

    return out


def load_tapas_results(mat_path: str) -> dict:
    """Load tapas rDCM estimation results from .mat file.

    Extracts rigid and sparse sub-structs with posterior expectations
    (``Ep``), free energy (``logF``), and binary indicators (``Ip``
    for sparse only).

    Parameters
    ----------
    mat_path : str
        Path to the results .mat file saved by MATLAB batch script.

    Returns
    -------
    dict
        Dictionary with ``'rigid'`` and ``'sparse'`` sub-dicts, each
        containing:
        - ``'Ep'``: np.ndarray, posterior expectations.
        - ``'logF'``: float, negative free energy.
        - ``'Ip'``: np.ndarray, binary indicators (sparse only).
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=False)
    results_raw = data["results"]

    out: dict = {"rigid": {}, "sparse": {}}

    # Rigid results
    try:
        rigid = results_raw["rigid"][0, 0]
        out["rigid"]["Ep"] = rigid["Ep"][0, 0].astype(np.float64)
        out["rigid"]["logF"] = float(rigid["logF"][0, 0].item())
    except (ValueError, IndexError, KeyError):
        pass

    # Sparse results
    try:
        sparse = results_raw["sparse"][0, 0]
        out["sparse"]["Ep"] = sparse["Ep"][0, 0].astype(np.float64)
        out["sparse"]["logF"] = float(sparse["logF"][0, 0].item())
        try:
            out["sparse"]["Ip"] = sparse["Ip"][0, 0].astype(np.float64)
        except (ValueError, IndexError, KeyError):
            pass
    except (ValueError, IndexError, KeyError):
        pass

    return out


def compare_posterior_means(
    pyro_A: np.ndarray,
    ref_A: np.ndarray,
    tolerance: float = 0.10,
    near_zero_threshold: float = 0.01,
    near_zero_atol: float = 0.02,
) -> dict:
    """Element-wise comparison of posterior mean A matrices.

    Uses a hybrid metric: relative error for elements where
    ``|ref_A| > near_zero_threshold``, absolute error for
    elements where ``|ref_A| <= near_zero_threshold``.

    Parameters
    ----------
    pyro_A : np.ndarray
        Posterior mean A from Pyro inference, shape ``(N, N)``.
    ref_A : np.ndarray
        Reference posterior mean A (from SPM12 or tapas), shape ``(N, N)``.
    tolerance : float, optional
        Relative error tolerance for large-value elements. Default 0.10.
    near_zero_threshold : float, optional
        Threshold below which absolute error is used. Default 0.01.
    near_zero_atol : float, optional
        Absolute error tolerance for near-zero elements. Default 0.02.

    Returns
    -------
    dict
        - ``'max_relative_error'``: float, maximum hybrid error.
        - ``'mean_relative_error'``: float, mean hybrid error.
        - ``'within_tolerance'``: bool, True if all errors within
          their respective tolerances.
        - ``'element_errors'``: np.ndarray, hybrid error per element.
    """
    abs_diff = np.abs(pyro_A - ref_A)
    abs_ref = np.abs(ref_A)

    # Build hybrid error: relative where |ref| > threshold, absolute otherwise
    large_mask = abs_ref > near_zero_threshold
    # Use safe division to avoid RuntimeWarning for zero-valued refs
    safe_ref = np.where(large_mask, abs_ref, 1.0)
    element_errors = np.where(
        large_mask,
        abs_diff / safe_ref,  # relative error
        abs_diff,             # absolute error
    )

    # Check tolerance: relative for large, absolute for near-zero
    within_large = np.where(large_mask, element_errors < tolerance, True)
    within_small = np.where(~large_mask, abs_diff < near_zero_atol, True)
    all_within = bool(np.all(within_large & within_small))

    return {
        "max_relative_error": float(element_errors.max()),
        "mean_relative_error": float(element_errors.mean()),
        "within_tolerance": all_within,
        "element_errors": element_errors,
    }


def compare_model_ranking(
    scenarios: list[dict],
) -> dict:
    """Compare model ranking between SPM free energy and Pyro ELBO.

    For all pairs of scenarios, checks whether SPM's ranking
    (higher F = better) agrees with Pyro's ranking (higher ELBO =
    better, i.e., lower loss).

    Parameters
    ----------
    scenarios : list of dict
        Each dict must have:
        - ``'spm_F'``: float, SPM free energy (higher = better).
        - ``'pyro_elbo'``: float, Pyro ELBO (higher = better).
          This is the negative of the SVI loss.

    Returns
    -------
    dict
        - ``'agreement_rate'``: float, fraction of pairs with matching
          ranking. Agreement >= 0.80 is the pass criterion.
        - ``'pairwise_results'``: list of dicts with details for each
          pair comparison.
        - ``'total_pairs'``: int, total number of pairwise comparisons.
        - ``'agreements'``: int, number of agreeing pairs.
    """
    n = len(scenarios)
    pairwise: list[dict] = []
    agreements = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            spm_prefers_i = scenarios[i]["spm_F"] > scenarios[j]["spm_F"]
            pyro_prefers_i = (
                scenarios[i]["pyro_elbo"] > scenarios[j]["pyro_elbo"]
            )
            match = spm_prefers_i == pyro_prefers_i

            pairwise.append({
                "i": i,
                "j": j,
                "spm_prefers_i": bool(spm_prefers_i),
                "pyro_prefers_i": bool(pyro_prefers_i),
                "agree": bool(match),
            })

            if match:
                agreements += 1
            total_pairs += 1

    rate = agreements / total_pairs if total_pairs > 0 else 0.0

    return {
        "agreement_rate": rate,
        "pairwise_results": pairwise,
        "total_pairs": total_pairs,
        "agreements": agreements,
    }


def compute_free_param_comparison(
    pyro_A_free: np.ndarray,
    spm_Ep_A: np.ndarray,
    tolerance: float = 0.10,
) -> dict:
    """Compare A matrices in free parameter space.

    SPM12's ``Ep.A`` stores free parameters (not parameterized A).
    Off-diagonal: direct comparison (same space). Diagonal: direct
    comparison (both are free params; parameterized = ``-exp(x)/2``).

    Parameters
    ----------
    pyro_A_free : np.ndarray
        Pyro posterior mean A in free parameter space, shape ``(N, N)``.
    spm_Ep_A : np.ndarray
        SPM12 posterior mean Ep.A (free parameters), shape ``(N, N)``.
    tolerance : float, optional
        Error tolerance. Default 0.10.

    Returns
    -------
    dict
        Same structure as ``compare_posterior_means``.
    """
    return compare_posterior_means(pyro_A_free, spm_Ep_A, tolerance)
