"""Latent-circuit DCM Variational Laplace benchmark runner (Plan 29-04).

Thin wrapper around the proven VL fit block in
``cluster/scripts/lc_vl_acceptance_run.py``: reuses ``_build_ground_truth`` from
``benchmarks.runners.latent_circuit_recovery`` and fits the N=4 bilinear
ground truth with ``LatentCircuitForward`` + ``run_variational_laplace_generic``.
Follows the ``(BenchmarkConfig) -> dict`` runner contract (VLINFRA-02) so it slots
into ``recovery_validation.py`` and the figure pipeline unchanged.

.. note::

    This module imports from ``pyro_dcm.models.*`` (``LC_A_PRIOR_VARIANCE`` /
    ``LC_B_PRIOR_VARIANCE``). The current ``dcm-pytorch`` Mutagen session uses an
    UNANCHORED ``models/`` ignore that also matches ``src/pyro_dcm/models/``, so
    that package is NOT synced to the M3 cluster. Any M3 latent-circuit VL run
    therefore requires the anchored-ignore fix first (todo
    ``mutagen-models-ignore``; see STATE.md Key Risks). Spectral/task VL runners
    are unaffected (they live under the synced ``inference/``). This is a
    deployment caveat only -- no code action here.

References
----------
.planning/phases/29-vl-validation-infra-bmr-rank/29-04-PLAN.md
    Runner contract (VLINFRA-02).
cluster/scripts/lc_vl_acceptance_run.py
    The verbatim VL fit block (lines ~85-159) reused here.
benchmarks/runners/latent_circuit_recovery.py
    ``_build_ground_truth`` (shared N=4 bilinear topology).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import compute_rmse, compute_summary_stats
from benchmarks.runners.latent_circuit_recovery import _build_ground_truth
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.inference import (
    LatentCircuitForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
)
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# Coarse latent grid: VL time-domain precision is a dense (T*N, T*N) matrix.
_DT: float = 0.1
_SNR: float = 10.0
_TRAIN_FRACTION: float = 0.80


def run_latent_circuit_vl(config: BenchmarkConfig) -> dict[str, Any]:
    """Run latent-circuit DCM Variational Laplace benchmark.

    For each seed: build the shared N=4 bilinear ground truth via
    ``_build_ground_truth``, simulate latent trajectories, train/test split, fit
    ``LatentCircuitForward`` via VL, extract the Laplace posterior, and compute
    A-RMSE, magnitude-masked B-RMSE (on ``|B_true| > 0.1``), and convergence.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses ``n_datasets`` (number of seeds),
        ``seed`` (base seed), ``quick`` (short 30s duration for smoke speed,
        else 50s), and ``max_iter`` (VL Gauss-Newton cap; defaults to 128 when
        ``None``).

    Returns
    -------
    dict
        Results dict with keys ``variant`` (``"latent_circuit"``), ``method``
        (``"vl"``), ``n_regions``, per-seed lists (``a_rmse_list``,
        ``b_rmse_list``, ``converged_list``, ``n_iterations_list``,
        ``time_list``), ``n_success``, ``n_failed``, an ``errors`` list, a
        nested ``summary`` block, and a ``metadata`` block. Per-seed failures
        append an ``"error"`` record and are skipped.
    """
    max_iter = config.max_iter if config.max_iter is not None else 128
    duration = 30.0 if config.quick else 50.0

    a_rmse_list: list[float] = []
    b_rmse_list: list[float] = []
    converged_list: list[bool] = []
    n_iterations_list: list[int] = []
    time_list: list[float] = []
    errors: list[dict[str, Any]] = []
    n_failed = 0
    n_regions = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"LC VL: seed {i + 1}/{config.n_datasets} (seed {seed_i})...")

        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            # Shared ground truth (duration-aware modulator retiming).
            gt = _build_ground_truth(seed=0, duration=duration)
            A_true = gt["A_true"]
            B_true = gt["B_true"]  # (1, N, N)
            C_true = gt["C"]
            b_mask_0 = gt["b_mask_0"]
            stim = gt["stim"]
            stim_mod = gt["stim_mod"]
            a_mask = gt["a_mask"]
            c_mask = gt["c_mask"]
            N = A_true.shape[0]
            n_regions = N

            # Simulate at the coarse VL grid.
            sim = simulate_latent_circuit(
                A_true, C_true, stim,
                duration=duration, dt=_DT, SNR=_SNR,
                solver="rk4", seed=seed_i,
                B_list=[B_true[0]], stimulus_mod=stim_mod,
            )
            trajs = sim["trajectories"].to(torch.float64)  # (T, N)
            t_all = sim["times"].to(torch.float64)
            if torch.isnan(trajs).any() or torch.isinf(trajs).any():
                raise ValueError("Simulated trajectories contain NaN/Inf.")

            t_train = int(trajs.shape[0] * _TRAIN_FRACTION)
            trajs_train = trajs[:t_train]
            t_eval_train = t_all[:t_train]
            driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

            # Variational Laplace fit.
            forward = LatentCircuitForward(
                stimulus=driving_stim,
                c_mask=c_mask,
                t_eval=t_eval_train,
                dt=_DT,
                b_masks=[b_mask_0],
                stim_mod=stim_mod,
                c_prior_variance=1.0,
                b_prior_variance=LC_B_PRIOR_VARIANCE,
            )
            t0 = time.time()
            result = run_variational_laplace_generic(
                forward,
                observed=trajs_train,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=LC_A_PRIOR_VARIANCE,
                context={},
            )
            elapsed = time.time() - t0
            posterior = extract_vl_posterior_generic(result, forward, N)

            # A-RMSE.
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)
            a_rmse = float(compute_rmse(A_true.to(torch.float64), A_inferred))

            # B-RMSE (magnitude-masked on |B_true| > 0.1; verbatim formula).
            B_free_mean = posterior["B_free"]["mean"].to(torch.float64)  # (J,N,N)
            B_inferred = parameterize_B(B_free_mean, b_mask_0.unsqueeze(0))
            b_eligible = (B_true.to(torch.float64).abs() > 0.1).float()
            b_rmse = float(
                (
                    ((B_true.to(torch.float64) - B_inferred) ** 2 * b_eligible).sum()
                    / b_eligible.sum().clamp(min=1.0)
                )
                ** 0.5
            )

            a_rmse_list.append(a_rmse)
            b_rmse_list.append(b_rmse)
            converged_list.append(bool(result.converged))
            n_iterations_list.append(int(result.n_iterations))
            time_list.append(elapsed)

            print(
                f"  A-RMSE={a_rmse:.4f}, B-RMSE={b_rmse:.4f}, "
                f"iters={result.n_iterations}, converged={result.converged}, "
                f"time={elapsed:.1f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1
            errors.append({"seed": seed_i, "error": str(e)})

    n_success = len(a_rmse_list)
    summary: dict[str, Any] = {}
    if a_rmse_list:
        summary = {
            "a_rmse_stats": compute_summary_stats(a_rmse_list),
            "b_rmse_stats": compute_summary_stats(b_rmse_list),
            "convergence_rate": float(np.mean(converged_list)),
        }

    return {
        "variant": "latent_circuit",
        "method": "vl",
        "n_regions": n_regions,
        "a_rmse_list": a_rmse_list,
        "b_rmse_list": b_rmse_list,
        "converged_list": converged_list,
        "n_iterations_list": n_iterations_list,
        "time_list": time_list,
        "n_success": n_success,
        "n_failed": n_failed,
        "errors": errors,
        "summary": summary,
        "metadata": {
            "variant": "latent_circuit",
            "method": "vl",
            "n_regions": n_regions,
            "duration": duration,
            "dt": _DT,
            "snr": _SNR,
            "max_iter": max_iter,
            "quick": config.quick,
        },
    }
