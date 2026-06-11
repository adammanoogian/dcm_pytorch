"""Recovery-matrix sweep grid driver for the Phase 30 validation matrix.

Defines the USER-CONFIRMED small validation grid (N x SNR x forward-model x
seed) and the single-cell driver ``run_one_cell`` that runs ONE
``(variant, N, SNR)`` cell over ``GRID_SEEDS`` seeds end-to-end:

1. build near-stability-boundary-EXCLUDED ground truth (spectral/task) or the
   fixed N=4 bilinear topology (latent_circuit),
2. fit each seed via the SAME Variational Laplace forward/simulate symbols the
   Phase 29 runners use (this driver REUSES that fit logic; it does NOT
   re-derive fits), injecting the per-cell SNR through ``snr_for_model``,
3. collect the per-seed lists the hardened assembler expects, and
4. assemble them via ``assemble_cell_metrics`` (Plan 30-01).

The latent-circuit N-axis asymmetry
-----------------------------------
``latent_circuit``'s ground truth is the FIXED N=4 bilinear topology
(``_build_ground_truth``); there is no genuine N=2 latent-circuit model. The grid
therefore IGNORES the N axis for ``latent_circuit`` and emits exactly ONE N-entry
(``n_regions=4`` with an ``"n_axis_note"``) so it never fabricates a non-existent
N=2 latent-circuit cell. ``spectral`` and ``task`` use both ``N in (2, 4)``.

Cell count
----------
With the asymmetry: spectral ``2 N x 2 SNR = 4`` + task ``2 N x 2 SNR = 4`` +
latent_circuit ``1 N x 2 SNR = 2`` = ``10`` cells. Each cell runs
``GRID_SEEDS = 10`` seeds inside ONE task (mirroring the runner per-seed loop via
``config.n_datasets``), so a SLURM array of 10 tasks = ``120`` fits. Seeds are
NOT separate array tasks.

Requirements
------------
VLREC-01 (cluster execution, >=10 seeds/cell, per-cell JSON)
    ``GRID_SEEDS = 10``; one array task per cell emits one per-cell metric block.
VLREC-03 (near-boundary exclusion + dt>=0.1)
    ``run_one_cell`` calls ``resample_A_until_accepted`` /
    ``exclude_near_boundary_A`` on each spectral/task ground-truth ``A`` and
    enforces ``dt >= 0.1`` for task/latent (asserts loud on a smaller dt).

References
----------
benchmarks.runners.spectral_vl / task_vl / latent_circuit_vl
    The Phase 29 VL fit blocks whose simulate/forward symbols this driver reuses.
benchmarks.recovery_matrix_metrics
    ``assemble_cell_metrics`` (Plan 30-01) + ``exclude_near_boundary_A`` /
    ``resample_A_until_accepted`` + ``snr_for_model``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.latent_circuit_metrics import compute_trajectory_r_squared
from benchmarks.metrics import (
    compute_coverage_from_samples,
    compute_rmse,
)
from benchmarks.recovery_matrix_metrics import (
    assemble_cell_metrics,
    compute_shrinkage_ratio,
    resample_A_until_accepted,
    snr_for_model,
)
from benchmarks.runners import RUNNER_REGISTRY
from benchmarks.runners.latent_circuit_recovery import (
    _build_ground_truth,
    _predict_trajectories,
)
from pyro_dcm.forward_models.neural_state import (  # type: ignore[import-untyped]
    parameterize_A,
)
from pyro_dcm.inference import (  # type: ignore[import-untyped]
    LatentCircuitForward,
    SpectralDCMForward,
    TaskDCMForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.models.latent_circuit_dcm_model import (  # type: ignore[import-untyped]
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
)
from pyro_dcm.simulators.latent_circuit_simulator import (  # type: ignore[import-untyped]
    simulate_latent_circuit,
)
from pyro_dcm.simulators.spectral_simulator import (  # type: ignore[import-untyped]
    default_noise_priors,
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (  # type: ignore[import-untyped]
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import (  # type: ignore[import-untyped]
    PiecewiseConstantInput,
)

__all__ = [
    "GRID_VARIANTS",
    "GRID_N",
    "GRID_SNR",
    "GRID_SEEDS",
    "LATENT_CIRCUIT_N",
    "enumerate_cells",
    "cell_for_index",
    "run_one_cell",
]

# ---------------------------------------------------------------------------
# USER-CONFIRMED small validation grid
# ---------------------------------------------------------------------------

GRID_VARIANTS: tuple[str, ...] = ("spectral", "task", "latent_circuit")
"""Forward-model variants swept. All three are VL (``method="vl"``)."""

GRID_N: tuple[int, ...] = (2, 4)
"""Region counts swept for spectral/task. ``latent_circuit`` ignores this axis."""

GRID_SNR: tuple[float, ...] = (1.0, 3.0)
"""Signal-to-noise levels swept (mapped per model by ``snr_for_model``)."""

GRID_SEEDS: int = 10
"""Seeds per cell (>=10 satisfies VLREC-01). All run inside ONE array task."""

LATENT_CIRCUIT_N: int = 4
"""Fixed region count for ``latent_circuit`` (intrinsic N=4 bilinear topology)."""

# dt floor enforced for task/latent VL (VLREC-03, pitfall N1/N2).
_DT_TASK: float = 0.1
_DT_LATENT: float = 0.1
_TR_TASK: float = 2.0
_LATENT_TRAIN_FRACTION: float = 0.80
_A_PRIOR_VARIANCE_BOLD: float = 1.0 / 64.0


def _seeded_A_factory(
    make_A: Callable[[int], torch.Tensor], seed_base: int,
) -> Callable[[], torch.Tensor]:
    """Build a zero-arg ``A`` maker advancing a fresh seed on each call.

    ``resample_A_until_accepted`` calls the returned closure repeatedly; each
    call must draw a DIFFERENT ``A`` (otherwise a rejected near-boundary draw
    regenerates identically). The per-call seed is ``seed_base + try_index``.

    Parameters
    ----------
    make_A : callable
        One-arg ground-truth maker taking an integer seed.
    seed_base : int
        Base seed; call ``k`` (1-indexed) uses ``seed_base + k``.

    Returns
    -------
    callable
        Zero-argument closure returning a fresh ``A`` per call.
    """
    counter = {"n": 0}

    def _draw() -> torch.Tensor:
        counter["n"] += 1
        return make_A(seed_base + counter["n"])

    return _draw


# ---------------------------------------------------------------------------
# Cell enumeration + index mapping
# ---------------------------------------------------------------------------


def enumerate_cells() -> list[dict[str, Any]]:
    """Enumerate the canonical, stably ordered list of sweep cells.

    The ordering is deterministic so a SLURM array index maps to a fixed cell.
    Iteration order is ``variant`` (outer, in ``GRID_VARIANTS`` order) then
    ``n_regions`` then ``snr``. For ``latent_circuit`` the ``N`` axis collapses
    to the single fixed ``LATENT_CIRCUIT_N`` (see module docstring), so it emits
    ``len(GRID_SNR)`` cells instead of ``len(GRID_N) * len(GRID_SNR)``.

    Returns
    -------
    list of dict
        Each cell is ``{"cell_index": int, "variant": str, "n_regions": int,
        "snr": float}`` plus, for ``latent_circuit``, an ``"n_axis_note"``
        explaining the collapsed N axis. With the default grid this returns
        exactly 10 cells (spectral 4 + task 4 + latent_circuit 2).
    """
    cells: list[dict[str, Any]] = []
    index = 0
    for variant in GRID_VARIANTS:
        if variant == "latent_circuit":
            n_values: tuple[int, ...] = (LATENT_CIRCUIT_N,)
            note = (
                "latent_circuit is intrinsically N=4 (fixed bilinear topology); "
                "the grid N axis does not apply and is collapsed to one entry."
            )
        else:
            n_values = GRID_N
            note = None
        for n_regions in n_values:
            for snr in GRID_SNR:
                cell: dict[str, Any] = {
                    "cell_index": index,
                    "variant": variant,
                    "n_regions": int(n_regions),
                    "snr": float(snr),
                }
                if note is not None:
                    cell["n_axis_note"] = note
                cells.append(cell)
                index += 1
    return cells


def cell_for_index(cell_index: int) -> dict[str, Any]:
    """Inverse lookup: return the cell at ``cell_index``.

    Parameters
    ----------
    cell_index : int
        Zero-based SLURM-array index into ``enumerate_cells()``.

    Returns
    -------
    dict
        The cell dict (see ``enumerate_cells``).

    Raises
    ------
    IndexError
        If ``cell_index`` is out of range, with the expected range in the
        message.
    """
    cells = enumerate_cells()
    if not 0 <= cell_index < len(cells):
        raise IndexError(
            f"cell_index out of range (expected 0..{len(cells) - 1} for "
            f"{len(cells)} cells); got {cell_index}."
        )
    return cells[cell_index]


# ---------------------------------------------------------------------------
# Per-cell driver (reuses the Phase 29 VL fit logic, adds SNR/boundary/metric)
# ---------------------------------------------------------------------------


def run_one_cell(
    cell: dict[str, Any],
    *,
    base_seed: int = 42,
    max_iter: int = 64,
    quick: bool = False,
) -> dict[str, Any]:
    """Run ONE ``(variant, N, SNR)`` cell over ``GRID_SEEDS`` seeds.

    Dispatches to the per-variant inline VL loop (which mirrors the matching
    Phase 29 runner's simulate -> VL fit -> posterior path) while injecting the
    cell SNR via ``snr_for_model`` and excluding near-boundary ground-truth
    ``A`` (spectral/task) via ``resample_A_until_accepted``. The resulting
    per-seed lists are assembled by ``assemble_cell_metrics`` (Plan 30-01).

    Parameters
    ----------
    cell : dict
        A cell from ``enumerate_cells`` / ``cell_for_index``.
    base_seed : int, optional
        Base seed; seed ``i`` is ``base_seed + i``. Default 42.
    max_iter : int, optional
        VL Gauss-Newton iteration cap. Default 64.
    quick : bool, optional
        If True, shorten the per-variant duration/blocks for a fast local
        faithfulness pre-check (not recovery-quality). Default False.

    Returns
    -------
    dict
        ``{**cell, "metrics": <assembled block>, "raw": <per-seed lists>,
        "config": {...}, "n_seeds": GRID_SEEDS}``.

    Raises
    ------
    ValueError
        On an unknown variant.
    """
    variant = cell["variant"]
    n_regions = int(cell["n_regions"])
    snr = float(cell["snr"])

    # Confirm the (variant, "vl") runner exists so the grid stays in lockstep
    # with the Phase 29 registry even though we inline the loop for SNR control.
    if (variant, "vl") not in RUNNER_REGISTRY:
        raise ValueError(
            f"no VL runner registered for variant {variant!r} (expected one of "
            f"{[v for (v, m) in RUNNER_REGISTRY if m == 'vl']})."
        )

    config = BenchmarkConfig(
        variant=variant,
        method="vl",
        n_datasets=GRID_SEEDS,
        n_regions=n_regions,
        seed=base_seed,
        max_iter=max_iter,
        quick=quick,
    )

    if variant == "spectral":
        cell_result, raw = _run_spectral_cell(config, snr)
    elif variant == "task":
        cell_result, raw = _run_task_cell(config, snr)
    elif variant == "latent_circuit":
        cell_result, raw = _run_latent_circuit_cell(config, snr)
    else:
        raise ValueError(
            f"unknown variant for run_one_cell (expected one of {GRID_VARIANTS}); "
            f"got {variant!r}."
        )

    metrics = assemble_cell_metrics(cell_result)
    return {
        **{k: v for k, v in cell.items()},
        "metrics": metrics,
        "raw": raw,
        "config": {
            "base_seed": base_seed,
            "max_iter": max_iter,
            "quick": quick,
            "snr": snr,
            "n_regions": n_regions,
        },
        "n_seeds": GRID_SEEDS,
    }


def _run_spectral_cell(
    config: BenchmarkConfig, snr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spectral VL per-seed loop with SNR + near-boundary exclusion.

    Mirrors ``run_spectral_vl`` but (a) rejects ground-truth ``A`` in the
    near-boundary band and (b) sets the observation-noise log-amplitude from
    ``snr_for_model("spectral", snr)`` instead of the default zero log-noise.
    """
    N = config.n_regions
    max_iter = config.max_iter if config.max_iter is not None else 64
    knob = snr_for_model("spectral", snr)
    log_amp = knob["noise_log_amplitude"]

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    converged_list: list[bool] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    shrinkage_list: list[float] = []
    max_real_eig_list: list[float] = []
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            # Near-boundary-excluded ground truth (fresh per-try seed).
            A_true = resample_A_until_accepted(
                _seeded_A_factory(
                    lambda s: make_stable_A_spectral(N, seed=s),
                    seed_base=seed_i * 1000,
                )
            )
            max_real_eig_list.append(
                float(torch.linalg.eigvals(A_true.to(torch.complex128)).real.max())
            )

            # Observation noise: override global log-amplitude from SNR.
            priors = default_noise_priors(N)
            b = priors["b_prior_mean"].clone()
            b[0, 0] = log_amp
            noise_params = {
                "a": priors["a_prior_mean"],
                "b": b,
                "c": priors["c_prior_mean"],
            }
            sim = simulate_spectral_dcm(
                A_true, noise_params=noise_params, TR=2.0, n_freqs=32, seed=seed_i,
            )
            csd_obs = sim["csd"].to(torch.complex128)
            freqs = sim["freqs"].to(torch.float64)
            a_mask = torch.ones(N, N, dtype=torch.float64)

            forward = SpectralDCMForward()
            result = run_variational_laplace_generic(
                forward,
                observed=csd_obs,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=_A_PRIOR_VARIANCE_BOLD,
                context={"freqs": freqs},
            )
            posterior = extract_vl_posterior_generic(result, forward, N)
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)
            A_free_samples = posterior["A_free"]["samples"].to(torch.float64)
            A_param_samples = torch.stack(
                [parameterize_A(s * a_mask) for s in A_free_samples],
            )
            rmse = compute_rmse(A_true.to(torch.float64), A_inferred)
            coverage = compute_coverage_from_samples(
                A_true.to(torch.float64), A_param_samples, ci_level=0.95,
            )
            a_std = posterior["A_free"]["std"].to(torch.float64)
            shrinkage = float(
                compute_shrinkage_ratio(a_std, _A_PRIOR_VARIANCE_BOLD ** 0.5)
                .mean()
                .item()
            )

            rmse_list.append(float(rmse))
            coverage_list.append(float(coverage))
            converged_list.append(bool(result.converged))
            a_true_list.append(A_true.to(torch.float64).flatten().tolist())
            a_inferred_list.append(A_inferred.flatten().tolist())
            shrinkage_list.append(shrinkage)

        except (RuntimeError, ValueError) as e:
            print(f"  spectral seed {seed_i} FAILED: {e}")
            n_failed += 1

    cell_result = {
        "variant": "spectral",
        "method": "vl",
        "n_regions": N,
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "converged_list": converged_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "shrinkage_list": shrinkage_list,
        "n_success": len(rmse_list),
        "n_failed": n_failed,
    }
    raw = {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "shrinkage_list": shrinkage_list,
        "converged_list": converged_list,
        "max_real_eig_list": max_real_eig_list,
        "r2_per_region_list": None,
    }
    return cell_result, raw


def _run_task_cell(
    config: BenchmarkConfig, snr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Task VL per-seed loop with SNR + near-boundary exclusion (dt>=0.1).

    Mirrors ``run_task_vl`` but rejects near-boundary ``A`` and passes
    ``snr_for_model("task", snr)["SNR"]`` into ``simulate_task_dcm``.
    """
    if not _DT_TASK >= 0.1:
        raise AssertionError(
            f"task VL dt must be >= 0.1 (VLREC-03 precision floor); got {_DT_TASK}."
        )
    N = config.n_regions
    M = 1
    max_iter = config.max_iter if config.max_iter is not None else 64
    duration = 40.0 if config.quick else 120.0
    n_blocks = 2 if config.quick else 3
    task_snr = snr_for_model("task", snr)["SNR"]

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    converged_list: list[bool] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    shrinkage_list: list[float] = []
    max_real_eig_list: list[float] = []
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            A_true = resample_A_until_accepted(
                _seeded_A_factory(
                    lambda s: make_random_stable_A(N, density=0.5, seed=s),
                    seed_base=seed_i * 1000,
                )
            )
            max_real_eig_list.append(
                float(torch.linalg.eigvals(A_true.to(torch.complex128)).real.max())
            )
            C_true = torch.zeros(N, M, dtype=torch.float64)
            C_true[0, 0] = 1.0
            stim = make_block_stimulus(
                n_blocks=n_blocks,
                block_duration=15.0,
                rest_duration=15.0,
                n_inputs=M,
            )
            # Fixed-step rk4 (not the default adaptive dopri5): on the M3 stack
            # (torchdiffeq 0.2.5 / torch 2.10) dopri5 underflows ("dt 0.0") on
            # this neural+hemodynamic ODE, killing every task cell. rk4 at
            # dt=0.01 is deterministic, platform-independent, and matches the
            # solver the VL forward (TaskDCMForward) already uses for the fit.
            sim = simulate_task_dcm(
                A_true, C_true, stim,
                duration=duration, dt=0.01, TR=_TR_TASK, SNR=task_snr,
                seed=seed_i, solver="rk4",
            )
            bold = sim["bold"].to(torch.float64)
            a_mask = torch.ones(N, N, dtype=torch.float64)
            c_mask = torch.zeros(N, M, dtype=torch.float64)
            c_mask[0, 0] = 1.0
            t_eval = torch.arange(
                0.0, bold.shape[0] * _TR_TASK, _TR_TASK, dtype=torch.float64,
            )[: bold.shape[0]]

            forward = TaskDCMForward(
                stimulus_fn=sim["stimulus"],
                c_mask=c_mask,
                t_eval=t_eval,
                dt=_DT_TASK,
            )
            result = run_variational_laplace_generic(
                forward,
                observed=bold,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=_A_PRIOR_VARIANCE_BOLD,
                context={"a_mask": a_mask},
            )
            posterior = extract_vl_posterior_generic(result, forward, N)
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)
            A_free_samples = posterior["A_free"]["samples"].to(torch.float64)
            A_param_samples = torch.stack(
                [parameterize_A(s * a_mask) for s in A_free_samples],
            )
            rmse = compute_rmse(A_true.to(torch.float64), A_inferred)
            coverage = compute_coverage_from_samples(
                A_true.to(torch.float64), A_param_samples, ci_level=0.95,
            )
            a_std = posterior["A_free"]["std"].to(torch.float64)
            shrinkage = float(
                compute_shrinkage_ratio(a_std, _A_PRIOR_VARIANCE_BOLD ** 0.5)
                .mean()
                .item()
            )

            rmse_list.append(float(rmse))
            coverage_list.append(float(coverage))
            converged_list.append(bool(result.converged))
            a_true_list.append(A_true.to(torch.float64).flatten().tolist())
            a_inferred_list.append(A_inferred.flatten().tolist())
            shrinkage_list.append(shrinkage)

        except (RuntimeError, ValueError, AssertionError) as e:
            # AssertionError covers torchdiffeq adaptive-solver underflow
            # ("underflow in dt 0.0") so a single bad seed degrades to a
            # skipped seed rather than aborting the whole cell.
            print(f"  task seed {seed_i} FAILED: {e}")
            n_failed += 1

    cell_result = {
        "variant": "task",
        "method": "vl",
        "n_regions": N,
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "converged_list": converged_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "shrinkage_list": shrinkage_list,
        "n_success": len(rmse_list),
        "n_failed": n_failed,
    }
    raw = {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "shrinkage_list": shrinkage_list,
        "converged_list": converged_list,
        "max_real_eig_list": max_real_eig_list,
        "r2_per_region_list": None,
    }
    return cell_result, raw


def _run_latent_circuit_cell(
    config: BenchmarkConfig, snr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Latent-circuit VL per-seed loop with SNR + per-region R2 (dt>=0.1).

    Mirrors ``run_latent_circuit_vl`` but threads ``snr_for_model`` and adds the
    per-region (NOT pooled) trajectory R2 and per-seed shrinkage the assembler
    consumes. The fixed N=4 bilinear ground truth is exempt from near-boundary
    resampling, but its max real eigenvalue is logged for characterization.
    """
    if not _DT_LATENT >= 0.1:
        raise AssertionError(
            f"latent VL dt must be >= 0.1 (VLREC-03 floor); got {_DT_LATENT}."
        )
    max_iter = config.max_iter if config.max_iter is not None else 64
    duration = 30.0 if config.quick else 50.0
    latent_snr = snr_for_model("latent_circuit", snr)["SNR"]

    a_rmse_list: list[float] = []
    converged_list: list[bool] = []
    r2_per_region_list: list[float] = []
    shrinkage_list: list[float] = []
    max_real_eig_list: list[float] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    n_failed = 0
    n_regions = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            gt = _build_ground_truth(seed=0, duration=duration)
            A_true = gt["A_true"]
            B_true = gt["B_true"]
            C_true = gt["C"]
            b_mask_0 = gt["b_mask_0"]
            stim = gt["stim"]
            stim_mod = gt["stim_mod"]
            a_mask = gt["a_mask"]
            c_mask = gt["c_mask"]
            N = A_true.shape[0]
            n_regions = N
            max_real_eig_list.append(
                float(torch.linalg.eigvals(A_true.to(torch.complex128)).real.max())
            )

            sim = simulate_latent_circuit(
                A_true, C_true, stim,
                duration=duration, dt=_DT_LATENT, SNR=latent_snr,
                solver="rk4", seed=seed_i,
                B_list=[B_true[0]], stimulus_mod=stim_mod,
            )
            trajs = sim["trajectories"].to(torch.float64)
            t_all = sim["times"].to(torch.float64)
            if torch.isnan(trajs).any() or torch.isinf(trajs).any():
                raise ValueError("Simulated trajectories contain NaN/Inf.")

            t_train = int(trajs.shape[0] * _LATENT_TRAIN_FRACTION)
            trajs_train = trajs[:t_train]
            trajs_test = trajs[t_train:]
            t_eval_train = t_all[:t_train]
            driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

            forward = LatentCircuitForward(
                stimulus=driving_stim,
                c_mask=c_mask,
                t_eval=t_eval_train,
                dt=_DT_LATENT,
                b_masks=[b_mask_0],
                stim_mod=stim_mod,
                c_prior_variance=1.0,
                b_prior_variance=LC_B_PRIOR_VARIANCE,
            )
            result = run_variational_laplace_generic(
                forward,
                observed=trajs_train,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=LC_A_PRIOR_VARIANCE,
                context={},
            )
            posterior = extract_vl_posterior_generic(result, forward, N)
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            C_mean = posterior["C_free"]["mean"].to(torch.float64) * c_mask
            B_free_mean = posterior["B_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)
            a_rmse = float(compute_rmse(A_true.to(torch.float64), A_inferred))

            # Held-out per-region (NOT pooled) trajectory R2.
            predicted_test = _predict_trajectories(
                A_free_mean, C_mean, B_free_mean[0], b_mask_0,
                driving_stim, stim_mod, t_all, _DT_LATENT, t_train,
            )
            r2_mean = float(
                compute_trajectory_r_squared(
                    predicted_test, trajs_test, pooled=False,
                )
            )

            a_std = posterior["A_free"]["std"].to(torch.float64)
            shrinkage = float(
                compute_shrinkage_ratio(a_std, LC_A_PRIOR_VARIANCE ** 0.5)
                .mean()
                .item()
            )

            a_rmse_list.append(a_rmse)
            converged_list.append(bool(result.converged))
            r2_per_region_list.append(r2_mean)
            shrinkage_list.append(shrinkage)
            a_true_list.append(A_true.to(torch.float64).flatten().tolist())
            a_inferred_list.append(A_inferred.flatten().tolist())

        except (RuntimeError, ValueError) as e:
            print(f"  latent_circuit seed {seed_i} FAILED: {e}")
            n_failed += 1

    cell_result = {
        "variant": "latent_circuit",
        "method": "vl",
        "n_regions": n_regions,
        "a_rmse_list": a_rmse_list,
        "converged_list": converged_list,
        "r2_per_region_list": r2_per_region_list,
        "shrinkage_list": shrinkage_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "n_success": len(a_rmse_list),
        "n_failed": n_failed,
    }
    raw = {
        "a_rmse_list": a_rmse_list,
        "r2_per_region_list": r2_per_region_list,
        "shrinkage_list": shrinkage_list,
        "converged_list": converged_list,
        "max_real_eig_list": max_real_eig_list,
        "coverage_list": None,
    }
    return cell_result, raw


def _selftest() -> None:
    """Print the enumerated grid for a quick manual sanity check."""
    cells = enumerate_cells()
    print(f"{len(cells)} cells:")
    for c in cells:
        print(c)
    print("first via cell_for_index:", cell_for_index(0))
    print("last via cell_for_index:", cell_for_index(len(cells) - 1))
    elapsed = time.time()
    print("module import + enumerate ok", round(time.time() - elapsed, 4))


if __name__ == "__main__":
    _selftest()
