"""rDCM cross-validation and model ranking validation orchestrators.

Provides:
- ``run_rdcm_validation``: Cross-validate rDCM analytic VB against tapas
  (or internal consistency fallback if tapas unavailable).
- ``run_model_ranking_validation``: Validate ELBO ranking vs SPM free
  energy ranking for task or spectral DCM.
- ``run_model_ranking_validation_rdcm``: Model ranking via analytic free
  energy (no MATLAB dependency).
- ``check_tapas_available``: Check whether tapas rDCM is accessible.
- ``check_matlab_available``: Check whether MATLAB + SPM12 is accessible.

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
[REF-021] Frassle et al. (2018), NeuroImage 155, 406-421.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from pyro_dcm.forward_models.rdcm_forward import (
    create_regressors,
    generate_bold,
    get_hrf,
)
from pyro_dcm.forward_models.rdcm_posterior import (
    rigid_inversion,
    sparse_inversion,
)
from pyro_dcm.simulators.rdcm_simulator import (
    make_block_stimulus_rdcm,
    make_stable_A_rdcm,
)
from validation.export_to_mat import (
    export_rdcm_for_tapas,
    upsample_stimulus,
)
from validation.compare_results import (
    compare_posterior_means,
    load_tapas_results,
)


# -----------------------------------------------------------------------
# Availability checks
# -----------------------------------------------------------------------


def check_tapas_available() -> bool:
    """Check whether tapas rDCM is accessible via MATLAB.

    Returns
    -------
    bool
        True if tapas rDCM directory exists at the expected path.
    """
    tapas_path = "C:/Users/aman0087/Documents/Github/tapas/rDCM"
    return os.path.isdir(tapas_path)


def check_matlab_available() -> bool:
    """Check whether MATLAB is accessible via subprocess.

    Returns
    -------
    bool
        True if ``matlab -batch "disp('ok')"`` returns exit code 0.
    """
    matlab_exe = "C:/Program Files/MATLAB/R2022a/bin/matlab.exe"
    if not os.path.isfile(matlab_exe):
        return False
    try:
        result = subprocess.run(
            [matlab_exe, "-batch", "disp('ok')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# -----------------------------------------------------------------------
# rDCM cross-validation
# -----------------------------------------------------------------------


def _generate_rdcm_data(
    seed: int = 42,
    n_regions: int = 3,
    n_time: int = 4000,
    u_dt: float = 0.5,
    y_dt: float = 2.0,
    SNR: float = 5.0,
) -> dict:
    """Generate synthetic rDCM data for validation.

    Parameters
    ----------
    seed : int
        Random seed.
    n_regions : int
        Number of brain regions.
    n_time : int
        Number of stimulus time steps at ``u_dt`` resolution.
    u_dt : float
        Stimulus sampling interval in seconds.
    y_dt : float
        BOLD repetition time (TR) in seconds.
    SNR : float
        Signal-to-noise ratio.

    Returns
    -------
    dict
        Keys: A, C, a_mask, c_mask, u, bold_result, hrf, X, Y, N_eff.
    """
    nr = n_regions
    nu = 1

    A, a_mask = make_stable_A_rdcm(nr, density=0.5, seed=seed)
    C = torch.zeros(nr, nu, dtype=torch.float64)
    C[0, 0] = 0.5
    c_mask = torch.zeros(nr, nu, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    torch.manual_seed(seed)
    u = make_block_stimulus_rdcm(n_time, nu, u_dt, seed=seed)
    bold_result = generate_bold(A, C, u, u_dt, y_dt, SNR=SNR)
    hrf = get_hrf(n_time, u_dt)
    X, Y, N_eff = create_regressors(
        hrf, bold_result["y"], u, u_dt, y_dt,
    )

    return {
        "A": A,
        "C": C,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "u": u,
        "bold_result": bold_result,
        "hrf": hrf,
        "X": X,
        "Y": Y,
        "N_eff": N_eff,
        "u_dt": u_dt,
        "y_dt": y_dt,
    }


def run_rdcm_validation(
    seed: int = 42,
    n_regions: int = 3,
    n_time: int = 4000,
    output_dir: str = "validation/data",
) -> dict | None:
    """Cross-validate rDCM analytic VB against tapas rDCM.

    If tapas is unavailable, returns internal consistency results
    comparing rigid vs sparse analytic VB instead.

    Parameters
    ----------
    seed : int
        Random seed for data generation.
    n_regions : int
        Number of brain regions.
    n_time : int
        Number of stimulus time steps.
    output_dir : str
        Directory for intermediate .mat files.

    Returns
    -------
    dict or None
        Comparison results. Keys depend on tapas availability:
        - If tapas available: ``'tapas_available'``, ``'rigid_comparison'``,
          ``'sparse_comparison'``, ``'rigid_result'``, ``'tapas_result'``.
        - If tapas unavailable: ``'tapas_available'``, ``'reason'``,
          ``'internal_rigid'``, ``'internal_sparse'``,
          ``'rigid_vs_sparse_means_close'``.
    """
    data = _generate_rdcm_data(
        seed=seed,
        n_regions=n_regions,
        n_time=n_time,
    )

    # Run our rigid analytic VB
    rigid_result = rigid_inversion(
        data["X"], data["Y"],
        data["a_mask"], data["c_mask"],
    )

    # Run our sparse analytic VB
    sparse_result = sparse_inversion(
        data["X"], data["Y"],
        data["a_mask"], data["c_mask"],
        n_reruns=10,
    )

    tapas_avail = check_tapas_available()

    if tapas_avail:
        # Export and run tapas
        os.makedirs(output_dir, exist_ok=True)

        bold_np = data["bold_result"]["y"].numpy()
        u_np = data["u"].numpy()
        a_np = data["a_mask"].numpy()
        c_np = data["c_mask"].numpy()

        # Upsample stimulus for tapas
        n_scans = bold_np.shape[0]
        u_tr = u_np[:: int(data["y_dt"] / data["u_dt"])][:n_scans]
        u_micro, u_dt_micro = upsample_stimulus(u_tr, data["y_dt"])

        input_path = os.path.join(output_dir, "rdcm_input.mat")
        output_path = os.path.join(output_dir, "rdcm_tapas_results.mat")
        export_rdcm_for_tapas(
            bold_np, u_micro, a_np, c_np,
            data["y_dt"], u_dt_micro, input_path,
        )

        # Run tapas via MATLAB
        matlab_exe = (
            "C:/Program Files/MATLAB/R2022a/bin/matlab.exe"
        )
        script_path = os.path.abspath(
            "validation/matlab_scripts/run_tapas_rdcm.m"
        )
        env = os.environ.copy()
        env["DCM_INPUT_PATH"] = os.path.abspath(input_path)
        env["DCM_OUTPUT_PATH"] = os.path.abspath(output_path)

        try:
            proc = subprocess.run(
                [matlab_exe, "-batch", f"run('{script_path}')"],
                capture_output=True, text=True, timeout=600,
                env=env,
            )
            if proc.returncode != 0 or not os.path.isfile(output_path):
                return {
                    "tapas_available": True,
                    "tapas_error": proc.stderr or proc.stdout,
                    "internal_rigid": rigid_result,
                    "internal_sparse": sparse_result,
                }

            tapas_results = load_tapas_results(output_path)

            # Compare rigid posteriors
            our_A = rigid_result["A_mu"].numpy()
            tapas_A = tapas_results["rigid"]["Ep"][:n_regions, :n_regions]

            rigid_comparison = compare_posterior_means(
                our_A, tapas_A, tolerance=0.10,
            )

            return {
                "tapas_available": True,
                "rigid_comparison": rigid_comparison,
                "rigid_result": rigid_result,
                "sparse_result": sparse_result,
                "tapas_result": tapas_results,
            }
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
        ) as exc:
            return {
                "tapas_available": True,
                "tapas_error": str(exc),
                "internal_rigid": rigid_result,
                "internal_sparse": sparse_result,
            }

    # Fallback: internal consistency check
    rigid_A = rigid_result["A_mu"].numpy()
    sparse_A = sparse_result["A_mu"].numpy()

    # Compare rigid vs sparse means (should be broadly consistent)
    internal_comparison = compare_posterior_means(
        rigid_A, sparse_A, tolerance=0.30,
    )

    return {
        "tapas_available": False,
        "reason": (
            "tapas rDCM not found at "
            "C:/Users/aman0087/Documents/Github/tapas/rDCM. "
            "Clone from "
            "https://github.com/translationalneuromodeling/tapas"
        ),
        "internal_rigid": rigid_result,
        "internal_sparse": sparse_result,
        "internal_comparison": internal_comparison,
        "rigid_vs_sparse_means_close": (
            internal_comparison["mean_relative_error"] < 0.30
        ),
    }


# -----------------------------------------------------------------------
# Model ranking validation (rDCM -- pure Python, no MATLAB)
# -----------------------------------------------------------------------


def run_model_ranking_validation_rdcm(
    seeds: list[int] | None = None,
) -> dict:
    """Validate rDCM model ranking via analytic free energy.

    For each seed, generates 3-region data and compares free energy
    across 3 model masks:
    - Model A (correct): true A mask.
    - Model B (missing connection): one true connection removed.
    - Model C (wrong structure): diagonal-only mask.

    No MATLAB dependency. Pure Python analytic VB.

    Parameters
    ----------
    seeds : list of int or None
        Random seeds. Default ``[42, 123, 456]``.

    Returns
    -------
    dict
        Keys: ``'agreement_rate'``, ``'per_seed_results'``,
        ``'correct_wins_count'``, ``'total_scenarios'``.
    """
    if seeds is None:
        seeds = [42, 123, 789]

    per_seed: list[dict] = []
    correct_wins = 0
    total_comparisons = 0

    for seed in seeds:
        data = _generate_rdcm_data(seed=seed, n_time=4000)
        nr = data["a_mask"].shape[0]
        a_mask_correct = data["a_mask"]
        c_mask = data["c_mask"]

        # Model A: correct mask
        result_correct = rigid_inversion(
            data["X"], data["Y"], a_mask_correct, c_mask,
        )

        # Model B: missing one true off-diagonal connection
        a_mask_missing = a_mask_correct.clone()
        removed_conn = None
        for i in range(nr):
            for j in range(nr):
                if (
                    i != j
                    and a_mask_correct[i, j] > 0.5
                ):
                    a_mask_missing[i, j] = 0.0
                    removed_conn = (i, j)
                    break
            if removed_conn is not None:
                break

        if removed_conn is None:
            # No off-diagonal connections to remove; skip
            continue

        result_missing = rigid_inversion(
            data["X"], data["Y"], a_mask_missing, c_mask,
        )

        # Model C: diagonal-only (wrong structure)
        a_mask_diag = torch.eye(nr, dtype=torch.float64)
        result_diag = rigid_inversion(
            data["X"], data["Y"], a_mask_diag, c_mask,
        )

        F_correct = float(result_correct["F_total"])
        F_missing = float(result_missing["F_total"])
        F_diag = float(result_diag["F_total"])

        # Correct vs missing-connection
        correct_vs_missing = F_correct > F_missing
        # Correct vs wrong-structure
        correct_vs_diag = F_correct > F_diag

        if correct_vs_missing:
            correct_wins += 1
        total_comparisons += 1

        if correct_vs_diag:
            correct_wins += 1
        total_comparisons += 1

        per_seed.append({
            "seed": seed,
            "F_correct": F_correct,
            "F_missing": F_missing,
            "F_diag": F_diag,
            "correct_vs_missing": correct_vs_missing,
            "correct_vs_diag": correct_vs_diag,
            "removed_connection": removed_conn,
            "A_true": data["A"].numpy(),
            "a_mask_correct": a_mask_correct.numpy(),
        })

    agreement_rate = (
        correct_wins / total_comparisons
        if total_comparisons > 0
        else 0.0
    )

    return {
        "agreement_rate": agreement_rate,
        "per_seed_results": per_seed,
        "correct_wins_count": correct_wins,
        "total_comparisons": total_comparisons,
    }
