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


def _densify(value: object) -> np.ndarray:
    """Return a dense float64 array, converting scipy sparse if needed.

    ``spm_dcm_fmri_priors`` returns several posterior fields as MATLAB sparse
    matrices, so ``scipy.io.loadmat`` hands them back as ``csc_array``.
    ``.astype(np.float64)`` on a sparse array stays sparse, and the first
    downstream ``np.asarray(..., dtype=float)`` then raises "setting an array
    element with a sequence" (DCCN job 55242695 lost all five seeds this way).
    """
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


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
    out["Ep_A"] = _densify(results_raw["Ep_A"][0, 0])
    out["F"] = float(results_raw["F"][0, 0].item())

    # Optional fields
    optional_fields = [
        "Ep_C", "Cp", "y_predicted", "R",
        "Ep_transit", "Ep_decay", "Ep_epsilon", "Hc", "Hz",
        # Noise parameters, needed to re-evaluate SPM's Ep through another
        # forward model on the same footing as the VL estimate.
        "Ep_a", "Ep_b", "Ep_c",
    ]
    for field in optional_fields:
        try:
            val = results_raw[field][0, 0]
            if val.size > 0:
                out[field] = _densify(val)
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
    true_A: np.ndarray | None = None,
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
    true_A : np.ndarray or None, optional
        Ground-truth A, shape ``(N, N)``. When given, the near-zero decision
        is keyed on the TRUE value -- the correct criterion, since whether a
        parameter is near zero is a property of the parameter, not of either
        engine's estimate of it. When None, an element takes the absolute
        branch if EITHER estimate is near zero.

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

    # Which elements are "large enough" for a relative error to mean anything.
    #
    # Keying this on |ref_A| ALONE is wrong: relative error blows up whenever
    # the denominator can be near zero, and a parameter whose true value is ~0
    # can still draw a sizeable estimate from one engine. On the N=2 parity
    # problem the A_free diagonal is a log-deviation from a fixed -0.5
    # self-connection (correct value ~0); SPM reported -0.0674, the guard
    # stayed off, and a 0.068 absolute disagreement was reported as a relative
    # error of 1.0085 (job 55234501).
    #
    # With ground truth available, whether a parameter is near zero is a
    # property of the PARAMETER. Without it, fall back to requiring that
    # neither estimate is near zero.
    if true_A is not None:
        large_mask = np.abs(true_A) > near_zero_threshold
    else:
        large_mask = np.minimum(abs_ref, np.abs(pyro_A)) > near_zero_threshold
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


def flag_free_energy_fit_inconsistency(
    rows: list[dict],
    f_tolerance: float = 1.0,
    residual_tolerance: float = 1e-9,
) -> dict:
    """Flag scenarios where the higher free energy comes with the worse fit.

    Free energy trades data fit against complexity, so a higher ``F`` with a
    larger residual is normally legitimate -- the simpler model is being
    rewarded. That defence does not apply here: both engines fit the **same
    model under the same mask**, so the complexity term is identical and only
    the fit and the approximation to it can differ. An engine reporting the
    higher ``F`` while achieving the larger residual is therefore reporting a
    free energy that does not track its own fit quality.

    Residuals must be computed with the SAME forward model for both engines,
    or this comparison measures forward differences instead.

    Parameters
    ----------
    rows : list of dict
        One entry per scenario with ``name``, ``vl_F``, ``spm_F``,
        ``vl_residual`` and ``spm_residual``. A ``None`` residual marks the
        scenario unknown rather than consistent.
    f_tolerance : float, optional
        Free-energy difference below which the two engines are treated as
        tied and the scenario is not flagged. Default 1.0 nat.
    residual_tolerance : float, optional
        Residual difference below which the fits are treated as tied.
        Default 1e-9.

    Returns
    -------
    dict
        - ``'per_scenario'``: per-scenario dicts with ``higher_f_engine``,
          ``lower_residual_engine`` and ``inconsistent`` (``None`` when a
          residual is missing).
        - ``'n_inconsistent'``, ``'n_unknown'``: counts.
        - ``'inconsistent_scenarios'``: names, for a quick readout.
    """
    per: list[dict] = []
    n_inconsistent = 0
    n_unknown = 0
    names: list[str] = []

    for row in rows:
        vl_f, spm_f = float(row["vl_F"]), float(row["spm_F"])
        vl_r, spm_r = row.get("vl_residual"), row.get("spm_residual")
        name = str(row.get("name", len(per)))

        higher_f = None
        if abs(vl_f - spm_f) > f_tolerance:
            higher_f = "vl" if vl_f > spm_f else "spm"

        lower_r = None
        if vl_r is not None and spm_r is not None:
            if abs(float(vl_r) - float(spm_r)) > residual_tolerance:
                lower_r = "vl" if float(vl_r) < float(spm_r) else "spm"

        if vl_r is None or spm_r is None:
            inconsistent: bool | None = None
            n_unknown += 1
        elif higher_f is None or lower_r is None:
            inconsistent = False
        else:
            inconsistent = higher_f != lower_r
            if inconsistent:
                n_inconsistent += 1
                names.append(name)

        per.append({
            "name": name,
            "vl_F": vl_f,
            "spm_F": spm_f,
            "vl_residual": None if vl_r is None else float(vl_r),
            "spm_residual": None if spm_r is None else float(spm_r),
            "higher_f_engine": higher_f,
            "lower_residual_engine": lower_r,
            "inconsistent": inconsistent,
        })

    return {
        "per_scenario": per,
        "n_inconsistent": n_inconsistent,
        "n_unknown": n_unknown,
        "inconsistent_scenarios": names,
        "f_tolerance": f_tolerance,
        "residual_tolerance": residual_tolerance,
    }


def compare_scenario_free_energies(
    scenarios: list[dict],
    rel_tolerance: float = 0.05,
) -> dict:
    """Compare VL and SPM free energy on EVERY model scenario.

    The matched-F gate historically checked only the full model. Reduced-model
    free energies were computed for the ranking sweep and then discarded, so a
    large divergence on a reduced model passed every gate: job 55234938 seed 44
    had a full-model relative error of 0.0162 (inside the 5% gate) alongside a
    ~54-nat divergence on a reduced model. Bayesian Model Reduction and every
    structure-selection claim operate on reduced models, so that is the regime
    that matters most and it was the least tested.

    Parameters
    ----------
    scenarios : list of dict
        Each dict must have:

        - ``'spm_F'``: float, SPM free energy for this ``a_mask``.
        - ``'pyro_elbo'``: float, VL free energy for the same ``a_mask``.
        - ``'name'``: str, optional scenario label. Defaults to the index.
    rel_tolerance : float, optional
        Relative tolerance ``|vl - spm| / |spm|``. Default 0.05, matching the
        strict matched-F gate.

    Returns
    -------
    dict
        - ``'per_scenario'``: list of dicts with ``name``, ``spm_F``,
          ``vl_F``, ``absolute_difference``, ``relative_error`` and
          ``within_tolerance``.
        - ``'all_within_tolerance'``: bool, True only if every scenario passes.
        - ``'worst_scenario'``: str, the scenario with the largest relative
          error (not merely the first failure).
        - ``'worst_relative_error'``: float.
        - ``'n_scenarios_failing'``: int.
        - ``'rel_tolerance'``: float, echoed for provenance.
    """
    per: list[dict] = []
    for idx, sc in enumerate(scenarios):
        spm_f = float(sc["spm_F"])
        vl_f = float(sc["pyro_elbo"])
        abs_diff = abs(vl_f - spm_f)
        rel = abs_diff / max(abs(spm_f), 1e-12)
        per.append({
            "name": str(sc.get("name", idx)),
            "spm_F": spm_f,
            "vl_F": vl_f,
            "absolute_difference": vl_f - spm_f,
            "relative_error": rel,
            "within_tolerance": bool(rel <= rel_tolerance),
        })

    failing = [r for r in per if not r["within_tolerance"]]
    worst = max(per, key=lambda r: r["relative_error"]) if per else None

    return {
        "per_scenario": per,
        "all_within_tolerance": len(failing) == 0,
        "worst_scenario": worst["name"] if worst else None,
        "worst_relative_error": worst["relative_error"] if worst else 0.0,
        "n_scenarios_failing": len(failing),
        "rel_tolerance": rel_tolerance,
    }


def compare_model_ranking(
    scenarios: list[dict],
    close_call_nats: float = 1.0,
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
    close_call_nats : float, optional
        Evidence separation below which a disagreement is a near-tie
        rather than a conflict. A pair is a close call when BOTH engines
        separate the two models by less than this. Default 1.0.

    Returns
    -------
    dict
        - ``'agreement_rate'``: float, fraction of pairs with matching
          ranking. Agreement >= 0.80 is the pass criterion.
        - ``'pairwise_results'``: list of dicts with details for each
          pair comparison, including ``spm_delta``, ``pyro_delta``,
          ``min_abs_separation`` and ``is_close_call``.
        - ``'total_pairs'``: int, total number of pairwise comparisons.
        - ``'agreements'``: int, number of agreeing pairs.
        - ``'disagreements_close'``: int, disagreements on near-ties.
        - ``'disagreements_separated'``: int, disagreements where at
          least one engine was confident. These are the serious ones.

    Notes
    -----
    The bare agreement rate is not interpretable on its own. A
    disagreement on models separated by 0.2 nats is a coin-flip; one on
    models separated by 40 nats is a real conflict. Job 55234748 seed 44
    returned 0.667 with no way to tell which it was.
    """
    n = len(scenarios)
    pairwise: list[dict] = []
    agreements = 0
    total_pairs = 0
    disagreements_close = 0
    disagreements_separated = 0

    for i in range(n):
        for j in range(i + 1, n):
            spm_prefers_i = scenarios[i]["spm_F"] > scenarios[j]["spm_F"]
            pyro_prefers_i = (
                scenarios[i]["pyro_elbo"] > scenarios[j]["pyro_elbo"]
            )
            match = spm_prefers_i == pyro_prefers_i

            spm_delta = float(scenarios[i]["spm_F"] - scenarios[j]["spm_F"])
            pyro_delta = float(
                scenarios[i]["pyro_elbo"] - scenarios[j]["pyro_elbo"]
            )
            # Conservative read of how close the call was: a pair is only a
            # near-tie if BOTH engines found the two models hard to separate.
            min_sep = min(abs(spm_delta), abs(pyro_delta))
            is_close = bool((not match) and min_sep < close_call_nats)

            pairwise.append({
                "i": i,
                "j": j,
                "spm_prefers_i": bool(spm_prefers_i),
                "pyro_prefers_i": bool(pyro_prefers_i),
                "agree": bool(match),
                "spm_delta": spm_delta,
                "pyro_delta": pyro_delta,
                "min_abs_separation": min_sep,
                "is_close_call": is_close,
            })

            if match:
                agreements += 1
            elif is_close:
                disagreements_close += 1
            else:
                disagreements_separated += 1
            total_pairs += 1

    rate = agreements / total_pairs if total_pairs > 0 else 0.0

    return {
        "agreement_rate": rate,
        "pairwise_results": pairwise,
        "total_pairs": total_pairs,
        "agreements": agreements,
        "disagreements_close": disagreements_close,
        "disagreements_separated": disagreements_separated,
        "close_call_nats": close_call_nats,
    }


def compute_free_param_comparison(
    pyro_A_free: np.ndarray,
    spm_Ep_A: np.ndarray,
    tolerance: float = 0.10,
    true_A: np.ndarray | None = None,
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
    true_A : np.ndarray or None, optional
        Ground-truth A. Forwarded to ``compare_posterior_means`` so the
        near-zero decision is keyed on the parameter rather than on either
        engine's estimate.

    Returns
    -------
    dict
        Same structure as ``compare_posterior_means``.
    """
    return compare_posterior_means(
        pyro_A_free, spm_Ep_A, tolerance, true_A=true_A,
    )


def compare_free_energies(
    vl_free_energy: float,
    spm_F: float,
    rel_tolerance: float = 0.05,
) -> dict:
    """Compare VL and SPM free energy on the SAME matched problem.

    Computes the relative error between the Variational Laplace free energy
    (``free_energy[-1]``) and the SPM12 free energy (``DCM.F``) and gates it
    against a relative tolerance. The 5% default is a HARD pass/fail criterion.

    VLSPM-02: relative tolerance on the SAME matched problem ONLY (same priors,
    same data, same model). The 5% gate is a HARD pass/fail criterion (user
    decision: strict-5%-F is pass/fail, not merely descriptive).

    This function must NEVER be used to compare absolute F across DIFFERENT
    models (pitfall S3). Cross-model comparison goes through
    ``compare_model_ranking`` (relative ranking of free energies / ELBOs), not
    through this element-wise absolute-F comparator.

    Both VL ``free_energy[-1]`` and SPM ``DCM.F`` are lower bounds on the
    log-evidence; the 5% target is only meaningful when both are evaluated on
    the IDENTICAL CSD (same-CSD injection, Plan 32-01). Comparing F values
    derived from different data or different models is not defensible here.

    Parameters
    ----------
    vl_free_energy : float
        Variational Laplace free energy on the matched problem, i.e. the final
        value ``free_energy[-1]`` from the VL engine.
    spm_F : float
        SPM12 free energy ``DCM.F`` on the IDENTICAL matched problem.
    rel_tolerance : float, optional
        Relative-error pass/fail threshold. Default 0.05 (the strict 5% gate).

    Returns
    -------
    dict
        - ``'vl_free_energy'``: float, the VL free energy as passed in.
        - ``'spm_F'``: float, the SPM free energy as passed in.
        - ``'relative_error'``: float, ``|vl - spm| / max(|spm|, 1e-12)``.
        - ``'within_tolerance'``: bool, True iff ``relative_error <
          rel_tolerance``.
        - ``'rel_tolerance'``: float, the tolerance used for the gate.
    """
    rel_err = abs(vl_free_energy - spm_F) / max(abs(spm_F), 1e-12)
    return {
        "vl_free_energy": float(vl_free_energy),
        "spm_F": float(spm_F),
        "relative_error": float(rel_err),
        "within_tolerance": bool(rel_err < rel_tolerance),
        "rel_tolerance": float(rel_tolerance),
    }
