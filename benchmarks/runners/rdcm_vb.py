"""rDCM analytic VB benchmark runners (rigid + sparse).

Implements the simulate -> infer -> measure loop for regression DCM
with analytic Variational Bayes inversion. Provides two separate
runner functions for rigid (fixed architecture) and sparse (ARD with
learned sparsity) modes.

Reuses patterns from tests/test_rdcm_recovery.py.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.fixtures import load_fixture
from benchmarks.metrics import (
    compute_coverage_from_ci,
    compute_rmse,
    compute_summary_stats,
    pearson_corr,
)
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


# rDCM-specific constants matching test_rdcm_recovery.py
_N_TIME = 4000
_U_DT = 0.5
_Y_DT = 2.0
_SNR = 3.0
_N_INPUTS = 2

# z-scores for multi-level CI computation (analytic Gaussian)
_Z_SCORES: dict[float, float] = {
    0.50: 0.6745,
    0.75: 1.1503,
    0.90: 1.6449,
    0.95: 1.9600,
}


def _extract_A_ci(
    result: dict,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    nr: int,
    nc: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 95% credible intervals for A from VB posterior.

    Parameters
    ----------
    result : dict
        Output from rigid/sparse inversion with mu_per_region
        and Sigma_per_region.
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(nr, nu)``.
    nr : int
        Number of regions.
    nc : int, optional
        Number of confound columns. Default 1.

    Returns
    -------
    tuple of torch.Tensor
        (A_lo, A_hi) each shape ``(nr, nr)``.
    """
    dtype = torch.float64
    A_lo = torch.zeros(nr, nr, dtype=dtype)
    A_hi = torch.zeros(nr, nr, dtype=dtype)

    for r in range(nr):
        mu_r = result["mu_per_region"][r]
        Sigma_r = result["Sigma_per_region"][r]
        std_r = torch.sqrt(torch.diag(Sigma_r))

        pos = 0
        for j in range(nr):
            if a_mask[r, j] > 0:
                A_lo[r, j] = mu_r[pos] - 1.96 * std_r[pos]
                A_hi[r, j] = mu_r[pos] + 1.96 * std_r[pos]
                pos += 1

    return A_lo, A_hi


def _extract_sparse_A_ci(
    result: dict,
    nr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 95% CI for A from sparse VB posterior (with z).

    Parameters
    ----------
    result : dict
        Output from sparse_inversion with mu_per_region,
        Sigma_per_region, and z_per_region.
    nr : int
        Number of regions.

    Returns
    -------
    tuple of torch.Tensor
        (A_lo, A_hi) each shape ``(nr, nr)``.
    """
    dtype = torch.float64
    A_lo = torch.zeros(nr, nr, dtype=dtype)
    A_hi = torch.zeros(nr, nr, dtype=dtype)

    for r in range(nr):
        mu_r = result["mu_per_region"][r]
        Sigma_r = result["Sigma_per_region"][r]
        std_r = torch.sqrt(torch.diag(Sigma_r))
        z_r = result["z_per_region"][r]

        for j in range(nr):
            val = mu_r[j] * z_r[j]
            std_val = std_r[j]
            A_lo[r, j] = val - 1.96 * std_val
            A_hi[r, j] = val + 1.96 * std_val

    return A_lo, A_hi


def _extract_A_std_rigid(
    result: dict,
    a_mask: torch.Tensor,
    nr: int,
) -> torch.Tensor:
    """Extract standard deviations of A from rigid VB posterior.

    Parameters
    ----------
    result : dict
        Output from rigid_inversion with Sigma_per_region.
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``.
    nr : int
        Number of regions.

    Returns
    -------
    torch.Tensor
        Standard deviations, shape ``(nr, nr)``.
    """
    dtype = torch.float64
    A_std = torch.zeros(nr, nr, dtype=dtype)
    for r in range(nr):
        Sigma_r = result["Sigma_per_region"][r]
        std_r = torch.sqrt(torch.diag(Sigma_r))
        pos = 0
        for j in range(nr):
            if a_mask[r, j] > 0:
                A_std[r, j] = std_r[pos]
                pos += 1
    return A_std


def _extract_A_std_sparse(
    result: dict,
    nr: int,
) -> torch.Tensor:
    """Extract standard deviations of A from sparse VB posterior.

    Parameters
    ----------
    result : dict
        Output from sparse_inversion with Sigma_per_region.
    nr : int
        Number of regions.

    Returns
    -------
    torch.Tensor
        Standard deviations, shape ``(nr, nr)``.
    """
    dtype = torch.float64
    A_std = torch.zeros(nr, nr, dtype=dtype)
    for r in range(nr):
        Sigma_r = result["Sigma_per_region"][r]
        std_r = torch.sqrt(torch.diag(Sigma_r))
        for j in range(nr):
            A_std[r, j] = std_r[j]
    return A_std


def _compute_f1(
    A_true: torch.Tensor,
    z_matrix: torch.Tensor,
    nr: int,
) -> float:
    """Compute F1 score for sparsity pattern recovery.

    Parameters
    ----------
    A_true : torch.Tensor
        Ground truth A, shape ``(nr, nr)``.
    z_matrix : torch.Tensor
        Learned binary indicators, shape ``(nr, nr)``.
    nr : int
        Number of regions.

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    true_mask = torch.zeros(nr, nr, dtype=torch.float64)
    for i in range(nr):
        for j in range(nr):
            if i != j and A_true[i, j].abs() > 1e-10:
                true_mask[i, j] = 1.0
            elif i == j:
                true_mask[i, j] = 1.0

    pred_mask = (z_matrix > 0.5).to(torch.float64)
    tp = ((true_mask == 1) & (pred_mask == 1)).sum().float()
    fp = ((true_mask == 0) & (pred_mask == 1)).sum().float()
    fn = ((true_mask == 1) & (pred_mask == 0)).sum().float()

    if (tp + fp) > 0 and (tp + fn) > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) > 1e-10:
            return (
                2.0 * precision * recall / (precision + recall)
            ).item()
    return 0.0


def _generate_rdcm_data(
    seed_i: int,
    nr: int,
) -> dict:
    """Generate rDCM data for a single trial.

    Parameters
    ----------
    seed_i : int
        Random seed.
    nr : int
        Number of regions.

    Returns
    -------
    dict
        Keys: A, a_mask, C, c_mask, X, Y, N_eff.
    """
    A, a_mask = make_stable_A_rdcm(nr, density=0.5, seed=seed_i)

    C = torch.zeros(nr, _N_INPUTS, dtype=torch.float64)
    if nr >= 2:
        C[0, 0] = 1.0
        C[1, 1] = 1.0
    else:
        C[0, 0] = 1.0
        C[0, 1] = 1.0

    c_mask = (C != 0).to(torch.float64)

    u = make_block_stimulus_rdcm(
        n_time=_N_TIME, n_inputs=_N_INPUTS,
        u_dt=_U_DT, seed=seed_i,
    )

    torch.manual_seed(seed_i + 10000)
    bold_result = generate_bold(
        A, C, u, u_dt=_U_DT, y_dt=_Y_DT, SNR=_SNR,
    )

    hrf = get_hrf(_N_TIME, _U_DT)
    X, Y, N_eff = create_regressors(
        hrf, bold_result["y"], u, _U_DT, _Y_DT,
    )

    return {
        "A": A, "a_mask": a_mask,
        "C": C, "c_mask": c_mask,
        "X": X, "Y": Y, "N_eff": N_eff,
    }


def run_rdcm_rigid_vb(config: BenchmarkConfig) -> dict[str, Any]:
    """Run rDCM rigid VB benchmark.

    For each dataset: generate synthetic BOLD, construct
    frequency-domain regressors, run rigid VB inversion, compute
    recovery metrics.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    dict
        Results dict with keys: rmse_list, coverage_list,
        correlation_list, free_energy_list, time_list, summary,
        metadata.
    """
    nr = config.n_regions

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    free_energy_list: list[float] = []
    time_list: list[float] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    coverage_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    coverage_diag_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    coverage_offdiag_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"Running dataset {i + 1}/{config.n_datasets}...")

        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            if config.fixtures_dir is not None:
                fdata = load_fixture(
                    "rdcm", nr, i, config.fixtures_dir,
                )
                hrf = get_hrf(_N_TIME, _U_DT)
                X, Y, N_eff = create_regressors(
                    hrf, fdata["y"], fdata["u"],
                    _U_DT, _Y_DT,
                )
                data = {
                    "A": fdata["A_true"],
                    "a_mask": fdata["a_mask"],
                    "C": fdata["C_true"],
                    "c_mask": fdata["c_mask"],
                    "X": X, "Y": Y, "N_eff": N_eff,
                }
            else:
                data = _generate_rdcm_data(seed_i, nr)

            t0 = time.time()
            inv_result = rigid_inversion(
                data["X"], data["Y"],
                data["a_mask"], data["c_mask"],
                confound_cols=1,
            )
            elapsed = time.time() - t0

            A_true = data["A"]
            A_inferred = inv_result["A_mu"]

            A_lo, A_hi = _extract_A_ci(
                inv_result, data["a_mask"], data["c_mask"], nr,
            )

            # Metrics
            rmse = compute_rmse(A_true, A_inferred)
            coverage = compute_coverage_from_ci(A_true, A_lo, A_hi)
            corr = pearson_corr(
                A_true[data["a_mask"].bool()],
                A_inferred[data["a_mask"].bool()],
            )

            # Multi-level coverage via z-score CIs
            A_mu = inv_result["A_mu"]
            A_std = _extract_A_std_rigid(
                inv_result, data["a_mask"], nr,
            )
            diag_mask = torch.eye(nr, dtype=torch.bool)
            offdiag_mask = ~diag_mask
            for lv, z in _Z_SCORES.items():
                lo = A_mu - z * A_std
                hi = A_mu + z * A_std
                cov_all = compute_coverage_from_ci(
                    A_true, lo, hi,
                )
                cov_diag = compute_coverage_from_ci(
                    A_true[diag_mask],
                    lo[diag_mask],
                    hi[diag_mask],
                )
                cov_offdiag = compute_coverage_from_ci(
                    A_true[offdiag_mask],
                    lo[offdiag_mask],
                    hi[offdiag_mask],
                )
                coverage_multi[lv].append(cov_all)
                coverage_diag_multi[lv].append(cov_diag)
                coverage_offdiag_multi[lv].append(
                    cov_offdiag,
                )

            rmse_list.append(rmse)
            coverage_list.append(coverage)
            correlation_list.append(corr)
            free_energy_list.append(float(inv_result["F_total"]))
            time_list.append(elapsed)
            a_true_list.append(A_true.flatten().tolist())
            a_inferred_list.append(
                A_inferred.flatten().tolist(),
            )

            print(
                f"  RMSE={rmse:.4f}, coverage={coverage:.3f}, "
                f"corr={corr:.3f}, F={inv_result['F_total']:.1f}, "
                f"time={elapsed:.2f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1

    n_success = len(rmse_list)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
        }

    summary: dict[str, Any] = {
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_correlation": float(np.mean(correlation_list)),
        "std_correlation": float(np.std(correlation_list)),
        "mean_time": float(np.mean(time_list)),
        "mean_free_energy": float(np.mean(free_energy_list)),
        "rmse_stats": compute_summary_stats(rmse_list),
        "coverage_stats": compute_summary_stats(coverage_list),
        "correlation_stats": compute_summary_stats(
            correlation_list,
        ),
        "time_stats": compute_summary_stats(time_list),
    }

    return {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "correlation_list": correlation_list,
        "free_energy_list": free_energy_list,
        "time_list": time_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "coverage_multi": {
            str(k): v for k, v in coverage_multi.items()
        },
        "coverage_diag_multi": {
            str(k): v for k, v in coverage_diag_multi.items()
        },
        "coverage_offdiag_multi": {
            str(k): v
            for k, v in coverage_offdiag_multi.items()
        },
        "n_success": n_success,
        "n_failed": n_failed,
        **summary,
        "metadata": {
            "variant": "rdcm_rigid",
            "method": "vb",
            "n_regions": nr,
            "n_time": _N_TIME,
            "u_dt": _U_DT,
            "y_dt": _Y_DT,
            "snr": _SNR,
            "quick": config.quick,
        },
    }


def run_rdcm_sparse_vb(config: BenchmarkConfig) -> dict[str, Any]:
    """Run rDCM sparse VB benchmark.

    For each dataset: generate synthetic BOLD, run sparse VB
    inversion with ARD, compute recovery metrics including F1 score.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    dict
        Results dict with keys: rmse_list, coverage_list,
        correlation_list, f1_list, free_energy_list, time_list,
        summary, metadata.
    """
    nr = config.n_regions
    n_reruns = 10 if config.quick else 20

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    f1_list: list[float] = []
    free_energy_list: list[float] = []
    time_list: list[float] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    coverage_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    coverage_diag_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    coverage_offdiag_multi: dict[float, list[float]] = {
        lv: [] for lv in _Z_SCORES
    }
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"Running dataset {i + 1}/{config.n_datasets}...")

        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            if config.fixtures_dir is not None:
                fdata = load_fixture(
                    "rdcm", nr, i, config.fixtures_dir,
                )
                hrf = get_hrf(_N_TIME, _U_DT)
                X, Y, N_eff = create_regressors(
                    hrf, fdata["y"], fdata["u"],
                    _U_DT, _Y_DT,
                )
                data = {
                    "A": fdata["A_true"],
                    "a_mask": fdata["a_mask"],
                    "C": fdata["C_true"],
                    "c_mask": fdata["c_mask"],
                    "X": X, "Y": Y, "N_eff": N_eff,
                }
            else:
                data = _generate_rdcm_data(seed_i, nr)

            t0 = time.time()
            inv_result = sparse_inversion(
                data["X"], data["Y"],
                data["a_mask"], data["c_mask"],
                confound_cols=1, n_reruns=n_reruns,
            )
            elapsed = time.time() - t0

            A_true = data["A"]
            A_inferred = inv_result["A_mu"]

            # Extract z_matrix
            z_matrix = torch.zeros(nr, nr, dtype=torch.float64)
            for r in range(nr):
                z_r = inv_result["z_per_region"][r]
                z_matrix[r, :nr] = z_r[:nr]

            # CI from sparse posterior
            A_lo, A_hi = _extract_sparse_A_ci(inv_result, nr)

            # Metrics
            rmse = compute_rmse(A_true, A_inferred)
            coverage = compute_coverage_from_ci(A_true, A_lo, A_hi)

            # Correlation on active connections (z > 0.5)
            active = z_matrix > 0.5
            if active.sum() > 0:
                corr = pearson_corr(
                    A_true[active], A_inferred[active],
                )
            else:
                corr = 0.0

            f1 = _compute_f1(A_true, z_matrix, nr)

            # Multi-level coverage via z-score CIs
            A_mu = inv_result["A_mu"]
            A_std = _extract_A_std_sparse(inv_result, nr)
            diag_mask = torch.eye(nr, dtype=torch.bool)
            offdiag_mask = ~diag_mask
            for lv, z in _Z_SCORES.items():
                lo = A_mu - z * A_std
                hi = A_mu + z * A_std
                cov_all = compute_coverage_from_ci(
                    A_true, lo, hi,
                )
                cov_diag = compute_coverage_from_ci(
                    A_true[diag_mask],
                    lo[diag_mask],
                    hi[diag_mask],
                )
                cov_offdiag = compute_coverage_from_ci(
                    A_true[offdiag_mask],
                    lo[offdiag_mask],
                    hi[offdiag_mask],
                )
                coverage_multi[lv].append(cov_all)
                coverage_diag_multi[lv].append(cov_diag)
                coverage_offdiag_multi[lv].append(
                    cov_offdiag,
                )

            rmse_list.append(rmse)
            coverage_list.append(coverage)
            correlation_list.append(corr)
            f1_list.append(f1)
            free_energy_list.append(float(inv_result["F_total"]))
            time_list.append(elapsed)
            a_true_list.append(A_true.flatten().tolist())
            a_inferred_list.append(
                A_inferred.flatten().tolist(),
            )

            print(
                f"  RMSE={rmse:.4f}, F1={f1:.3f}, "
                f"coverage={coverage:.3f}, corr={corr:.3f}, "
                f"time={elapsed:.2f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1

    n_success = len(rmse_list)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
        }

    summary: dict[str, Any] = {
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_correlation": float(np.mean(correlation_list)),
        "std_correlation": float(np.std(correlation_list)),
        "mean_f1": float(np.mean(f1_list)),
        "std_f1": float(np.std(f1_list)),
        "mean_time": float(np.mean(time_list)),
        "mean_free_energy": float(np.mean(free_energy_list)),
        "rmse_stats": compute_summary_stats(rmse_list),
        "coverage_stats": compute_summary_stats(coverage_list),
        "correlation_stats": compute_summary_stats(
            correlation_list,
        ),
        "f1_stats": compute_summary_stats(f1_list),
        "time_stats": compute_summary_stats(time_list),
    }

    return {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "correlation_list": correlation_list,
        "f1_list": f1_list,
        "free_energy_list": free_energy_list,
        "time_list": time_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "coverage_multi": {
            str(k): v for k, v in coverage_multi.items()
        },
        "coverage_diag_multi": {
            str(k): v for k, v in coverage_diag_multi.items()
        },
        "coverage_offdiag_multi": {
            str(k): v
            for k, v in coverage_offdiag_multi.items()
        },
        "n_success": n_success,
        "n_failed": n_failed,
        **summary,
        "metadata": {
            "variant": "rdcm_sparse",
            "method": "vb",
            "n_regions": nr,
            "n_time": _N_TIME,
            "u_dt": _U_DT,
            "y_dt": _Y_DT,
            "snr": _SNR,
            "n_reruns": n_reruns,
            "quick": config.quick,
        },
    }
