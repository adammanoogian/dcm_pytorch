"""Full validation orchestrator: raw CSD vs latent CSD perturbation comparison.

Runs the perturbation experiment twice to compare DCM sensitivity:

Path A (raw CSD): Compute CSD directly from 10-region simulated timeseries,
    fit spectral DCM (N=10), measure perturbation detection z-scores.
Path B (latent CSD): Pass timeseries through trained LSTM autoencoder,
    extract latent CSD (N_latent=20), fit spectral DCM, measure z-scores.

Comparing detection sensitivity across paths reveals what the autoencoder
representation adds or loses relative to the raw observation space.

Usage
-----
python scripts/22_run_full_validation.py --output-dir results/full_validation

Cluster
-------
python scripts/22_run_full_validation.py --submit-cluster

See Also
--------
scripts/22_perturbation_experiment.py : Path B implementation.
scripts/22_analyze_perturbation.py : Post-hoc figure generation.
"""
from __future__ import annotations

import argparse
import logging
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.forward_models.csd_computation import compute_empirical_csd
from pyro_dcm.inference.variational_laplace import (
    extract_vl_posterior,
    run_variational_laplace,
)
from pyro_dcm.neural_data_models.latent_csd import (
    compute_latent_csd,
    extract_latent_trajectories,
    prepare_for_spectral_dcm,
)
from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder
from pyro_dcm.neural_data_models.trainer import AutoencoderTrainer
from pyro_dcm.simulators.meg_simulator import (
    SENSORIMOTOR_ROI_NAMES,
    make_sensorimotor_A,
    simulate_meg_timeseries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Perturbation conditions (same as 22_perturbation_experiment.py)
PERTURBATION_CONDITIONS: list[dict] = [
    {"name": "M1_S1_strengthen", "i": 0, "j": 2, "factor": 1.5},
    {"name": "M1_S1_weaken", "i": 0, "j": 2, "factor": 0.5},
    {"name": "M1_S1_remove", "i": 0, "j": 2, "factor": 0.0},
    {"name": "PMC_M1_strengthen", "i": 0, "j": 4, "factor": 1.5},
    {"name": "PMC_M1_weaken", "i": 0, "j": 4, "factor": 0.5},
    {"name": "PMC_M1_remove", "i": 0, "j": 4, "factor": 0.0},
    {"name": "A1_M1_strengthen", "i": 0, "j": 8, "factor": 1.5},
    {"name": "A1_M1_remove", "i": 0, "j": 8, "factor": 0.0},
    {"name": "bilateral_M1_strengthen", "i": 0, "j": 1, "factor": 2.0},
]


# -- Spectral DCM fitting (shared by both paths) -------------------------


def _fit_spectral_dcm(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    *,
    max_iter: int,
    tolerance: float,
    prior_a_var: float = 1.0 / 16.0,
    eig_clamp: float = -1.0,
) -> dict:
    """Fit spectral DCM via Variational Laplace and return posterior A.

    Parameters
    ----------
    observed_csd : torch.Tensor
        Observed CSD, shape ``(F, N, N)``, complex128.
    freqs : torch.Tensor
        Frequency grid, shape ``(F,)``, float64.
    a_mask : torch.Tensor
        Connectivity mask, shape ``(N, N)``, float64.
    max_iter : int
        Maximum Gauss-Newton iterations.
    tolerance : float
        Convergence criterion on free energy change.
    prior_a_var : float
        Prior variance for A matrix elements.
    eig_clamp : float
        Maximum eigenvalue real part for A.

    Returns
    -------
    dict
        Keys: ``'A_mean'``, ``'A_std'``, ``'free_energy'``,
        ``'converged'``, ``'n_iterations'``.
    """
    N = a_mask.shape[0]
    result = run_variational_laplace(
        observed_csd,
        freqs,
        a_mask,
        max_iter=max_iter,
        tolerance=tolerance,
        prior_variance=prior_a_var,
        eig_clamp=eig_clamp,
    )

    posterior = extract_vl_posterior(result, N, num_samples=500)

    a_mean = result.theta_post["A"].detach().cpu().numpy()
    a_std = posterior["A"]["std"].detach().cpu().numpy()

    return {
        "A_mean": a_mean,
        "A_std": a_std,
        "free_energy": result.free_energy[-1] if result.free_energy else 0.0,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
    }


# -- Raw CSD computation (Path A) ----------------------------------------


def _compute_raw_csd(
    timeseries: torch.Tensor,
    *,
    sfreq: float = 250.0,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_freqs: int = 64,
) -> dict[str, np.ndarray]:
    """Compute CSD directly from raw timeseries, averaged over samples.

    Parameters
    ----------
    timeseries : torch.Tensor
        Raw timeseries, shape ``(n_samples, T, N)``.
    sfreq : float
        Sampling frequency in Hz.
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    n_freqs : int
        Number of frequency bins.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``'csd'`` (F, N, N), ``'freqs'`` (F,),
        ``'sfreq'``, ``'n_latent'`` (= N, for compatibility).
    """
    freqs = np.linspace(fmin, fmax, n_freqs)
    data = timeseries.numpy()
    n_samples, _T, n_roi = data.shape

    csd_sum = np.zeros((n_freqs, n_roi, n_roi), dtype=np.complex128)
    for k in range(n_samples):
        csd_sum += compute_empirical_csd(data[k], fs=sfreq, freqs=freqs)
    csd = csd_sum / n_samples

    return {
        "csd": csd,
        "freqs": freqs,
        "sfreq": sfreq,
        "n_latent": n_roi,
    }


# -- Z-score computation -------------------------------------------------


def _compute_zscores(
    delta_A: np.ndarray,
    perturbed_ij: np.ndarray,
) -> np.ndarray:
    """Compute z-score of perturbed element vs unperturbed elements.

    Parameters
    ----------
    delta_A : np.ndarray
        Shape ``(n_conditions, N, N)``, posterior delta per condition.
    perturbed_ij : np.ndarray
        Shape ``(n_conditions, 2)``, (i, j) of perturbed element.

    Returns
    -------
    np.ndarray
        Z-scores, shape ``(n_conditions,)``.
    """
    n_conditions = delta_A.shape[0]
    N = delta_A.shape[1]
    z_scores = np.zeros(n_conditions)

    for c in range(n_conditions):
        i, j = perturbed_ij[c]
        delta_flat = np.abs(delta_A[c].ravel())
        perturbed_idx = i * N + j
        mask = np.ones(len(delta_flat), dtype=bool)
        mask[perturbed_idx] = False
        other_vals = delta_flat[mask]

        std_other = np.std(other_vals)
        if std_other > 0:
            z_scores[c] = np.abs(delta_A[c, i, j]) / std_other
        else:
            z_scores[c] = (
                np.inf if np.abs(delta_A[c, i, j]) > 0 else 0.0
            )

    return z_scores


# -- Path A: Raw CSD DCM -------------------------------------------------


def run_raw_csd_path(
    *,
    A_base: torch.Tensor,
    n_eval: int,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> dict:
    """Run perturbation sweep with raw CSD (no autoencoder).

    Parameters
    ----------
    A_base : torch.Tensor
        Baseline connectivity, shape ``(N, N)``.
    n_eval : int
        Number of evaluation samples per condition.
    max_iter : int
        Maximum Gauss-Newton iterations per DCM fit.
    tolerance : float
        Convergence criterion on free energy change.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``'baseline'``, ``'conditions'``, ``'delta_A'``,
        ``'perturbed_ij'``, ``'z_scores'``.
    """
    n_roi = A_base.shape[0]
    logger.info("=== PATH A: Raw CSD (N=%d) ===", n_roi)

    # Baseline raw CSD
    logger.info("  Generating baseline eval data...")
    eval_result = simulate_meg_timeseries(
        A_base, n_samples=n_eval, seed=seed + 1000
    )
    eval_data = eval_result["timeseries"]

    logger.info("  Computing baseline raw CSD...")
    baseline_csd_result = _compute_raw_csd(eval_data)
    baseline_dcm_input = prepare_for_spectral_dcm(baseline_csd_result)

    logger.info("  Fitting baseline raw DCM...")
    t0 = time.time()
    baseline_post = _fit_spectral_dcm(
        baseline_dcm_input["csd"],
        baseline_dcm_input["freqs"],
        baseline_dcm_input["a_mask"],
        max_iter=max_iter,
        tolerance=tolerance,
    )
    logger.info(
        "  Baseline raw DCM F=%.2f (%.1f s)",
        baseline_post["free_energy"],
        time.time() - t0,
    )

    # Perturbation sweep
    results_list = []
    for cond_idx, cond in enumerate(PERTURBATION_CONDITIONS):
        t0 = time.time()
        name = cond["name"]
        i, j, factor = cond["i"], cond["j"], cond["factor"]

        logger.info(
            "  Raw condition %d/%d: %s",
            cond_idx + 1,
            len(PERTURBATION_CONDITIONS),
            name,
        )

        A_perturbed = A_base.clone()
        true_delta = A_base[i, j].item() * (factor - 1.0)
        A_perturbed[i, j] = A_base[i, j] * factor

        # Stability check
        eigvals = torch.linalg.eigvals(
            A_perturbed.to(torch.complex128)
        )
        if eigvals.real.max().item() >= 0:
            logger.warning("    Unstable, skipping %s", name)
            continue

        # Generate perturbed data
        perturbed_result = simulate_meg_timeseries(
            A_perturbed,
            n_samples=n_eval,
            seed=seed + 2000 + cond_idx,
        )

        # Compute raw CSD and fit DCM
        perturbed_csd_result = _compute_raw_csd(
            perturbed_result["timeseries"],
        )
        perturbed_dcm_input = prepare_for_spectral_dcm(
            perturbed_csd_result,
        )

        perturbed_post = _fit_spectral_dcm(
            perturbed_dcm_input["csd"],
            perturbed_dcm_input["freqs"],
            perturbed_dcm_input["a_mask"],
            max_iter=max_iter,
            tolerance=tolerance,
        )

        delta_a = perturbed_post["A_mean"] - baseline_post["A_mean"]
        logger.info(
            "    delta_A[%d,%d]=%.4f (true=%.4f), took %.1f s",
            i,
            j,
            delta_a[i, j],
            true_delta,
            time.time() - t0,
        )

        results_list.append({
            "name": name,
            "i": i,
            "j": j,
            "factor": factor,
            "true_delta": true_delta,
            "delta_A": delta_a,
            "A_mean": perturbed_post["A_mean"],
            "A_std": perturbed_post["A_std"],
            "free_energy": perturbed_post["free_energy"],
        })

    # Assemble arrays
    condition_names = [r["name"] for r in results_list]
    perturbed_ij = np.array([[r["i"], r["j"]] for r in results_list])
    delta_A_stack = np.stack([r["delta_A"] for r in results_list])
    z_scores = _compute_zscores(delta_A_stack, perturbed_ij)

    return {
        "baseline_A_mean": baseline_post["A_mean"],
        "baseline_A_std": baseline_post["A_std"],
        "condition_names": np.array(condition_names),
        "perturbed_ij": perturbed_ij,
        "true_deltas": np.array(
            [r["true_delta"] for r in results_list]
        ),
        "delta_A": delta_A_stack,
        "z_scores": z_scores,
        "n_roi": n_roi,
    }


# -- Path B: Latent CSD DCM (via autoencoder) ----------------------------


def run_latent_csd_path(
    *,
    A_base: torch.Tensor,
    n_train: int,
    n_eval: int,
    n_latent_multiplier: int,
    hidden_size: int,
    ae_epochs: int,
    max_iter: int,
    tolerance: float,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run perturbation sweep with latent CSD (autoencoder path).

    Parameters
    ----------
    A_base : torch.Tensor
        Baseline connectivity, shape ``(N, N)``.
    n_train : int
        Number of training samples for autoencoder.
    n_eval : int
        Number of evaluation samples per condition.
    n_latent_multiplier : int
        Latent dimension = multiplier * n_roi.
    hidden_size : int
        LSTM hidden size.
    ae_epochs : int
        Autoencoder training epochs.
    max_iter : int
        Maximum Gauss-Newton iterations per DCM fit.
    tolerance : float
        Convergence criterion on free energy change.
    seed : int
        Random seed.
    output_dir : Path
        Directory for saving autoencoder checkpoint.

    Returns
    -------
    dict
        Keys: ``'baseline_A_mean'``, ``'condition_names'``,
        ``'perturbed_ij'``, ``'delta_A'``, ``'z_scores'``, etc.
    """
    n_roi = A_base.shape[0]
    n_latent = n_latent_multiplier * n_roi
    logger.info("=== PATH B: Latent CSD (N_latent=%d) ===", n_latent)

    # Train autoencoder on baseline data
    logger.info("  Generating training data (%d samples)...", n_train)
    train_result = simulate_meg_timeseries(
        A_base, n_samples=n_train, seed=seed
    )
    train_data = train_result["timeseries"]

    logger.info("  Training autoencoder...")
    t0 = time.time()
    torch.manual_seed(seed)
    ae_model = MEGAutoencoder(
        n_roi=n_roi, n_latent=n_latent, hidden_size=hidden_size
    )
    trainer = AutoencoderTrainer(ae_model, lr=1e-3)
    train_dataset = TensorDataset(train_data.float())
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    history = trainer.train(
        train_loader, n_epochs=ae_epochs, log_every=10
    )
    logger.info(
        "  AE trained: %d epochs, loss=%.6f (%.1f s)",
        history["final_epoch"] + 1,
        history["train_losses"][-1],
        time.time() - t0,
    )

    ckpt_path = output_dir / "ae_checkpoint.pt"
    trainer.save_checkpoint(ckpt_path)

    # Baseline latent CSD
    logger.info("  Generating baseline eval data...")
    eval_result = simulate_meg_timeseries(
        A_base, n_samples=n_eval, seed=seed + 1000
    )
    eval_data = eval_result["timeseries"]

    baseline_latents = extract_latent_trajectories(
        ae_model, eval_data.float()
    )
    baseline_csd_result = compute_latent_csd(
        baseline_latents, sfreq=250.0, average_over_samples=True
    )
    baseline_dcm_input = prepare_for_spectral_dcm(baseline_csd_result)

    logger.info("  Fitting baseline latent DCM...")
    t0 = time.time()
    baseline_post = _fit_spectral_dcm(
        baseline_dcm_input["csd"],
        baseline_dcm_input["freqs"],
        baseline_dcm_input["a_mask"],
        max_iter=max_iter,
        tolerance=tolerance,
    )
    logger.info(
        "  Baseline latent DCM F=%.2f (%.1f s)",
        baseline_post["free_energy"],
        time.time() - t0,
    )

    # Perturbation sweep
    results_list = []
    for cond_idx, cond in enumerate(PERTURBATION_CONDITIONS):
        t0 = time.time()
        name = cond["name"]
        i, j, factor = cond["i"], cond["j"], cond["factor"]

        logger.info(
            "  Latent condition %d/%d: %s",
            cond_idx + 1,
            len(PERTURBATION_CONDITIONS),
            name,
        )

        A_perturbed = A_base.clone()
        true_delta = A_base[i, j].item() * (factor - 1.0)
        A_perturbed[i, j] = A_base[i, j] * factor

        eigvals = torch.linalg.eigvals(
            A_perturbed.to(torch.complex128)
        )
        if eigvals.real.max().item() >= 0:
            logger.warning("    Unstable, skipping %s", name)
            continue

        perturbed_result = simulate_meg_timeseries(
            A_perturbed,
            n_samples=n_eval,
            seed=seed + 2000 + cond_idx,
        )

        # Through SAME autoencoder (no retraining)
        perturbed_latents = extract_latent_trajectories(
            ae_model, perturbed_result["timeseries"].float()
        )
        perturbed_csd_result = compute_latent_csd(
            perturbed_latents, sfreq=250.0, average_over_samples=True
        )
        perturbed_dcm_input = prepare_for_spectral_dcm(
            perturbed_csd_result,
        )

        perturbed_post = _fit_spectral_dcm(
            perturbed_dcm_input["csd"],
            perturbed_dcm_input["freqs"],
            perturbed_dcm_input["a_mask"],
            max_iter=max_iter,
            tolerance=tolerance,
        )

        delta_a = perturbed_post["A_mean"] - baseline_post["A_mean"]
        logger.info(
            "    delta_A[%d,%d]=%.4f (true=%.4f), took %.1f s",
            i,
            j,
            delta_a[i, j],
            true_delta,
            time.time() - t0,
        )

        results_list.append({
            "name": name,
            "i": i,
            "j": j,
            "factor": factor,
            "true_delta": true_delta,
            "delta_A": delta_a,
            "A_mean": perturbed_post["A_mean"],
            "A_std": perturbed_post["A_std"],
            "free_energy": perturbed_post["free_energy"],
        })

    # Assemble arrays
    condition_names = [r["name"] for r in results_list]
    perturbed_ij = np.array([[r["i"], r["j"]] for r in results_list])
    delta_A_stack = np.stack([r["delta_A"] for r in results_list])
    z_scores = _compute_zscores(delta_A_stack, perturbed_ij)

    return {
        "baseline_A_mean": baseline_post["A_mean"],
        "baseline_A_std": baseline_post["A_std"],
        "condition_names": np.array(condition_names),
        "perturbed_ij": perturbed_ij,
        "true_deltas": np.array(
            [r["true_delta"] for r in results_list]
        ),
        "delta_A": delta_A_stack,
        "z_scores": z_scores,
        "n_latent": n_latent,
    }


# -- Comparison -----------------------------------------------------------


def compare_paths(
    raw_results: dict,
    latent_results: dict,
) -> dict:
    """Compare raw vs latent DCM perturbation detection.

    Parameters
    ----------
    raw_results : dict
        Output of :func:`run_raw_csd_path`.
    latent_results : dict
        Output of :func:`run_latent_csd_path`.

    Returns
    -------
    dict
        Comparison statistics for each condition.
    """
    raw_names = list(raw_results["condition_names"])
    latent_names = list(latent_results["condition_names"])

    # Match conditions by name (both should have same set)
    common_names = [n for n in raw_names if n in latent_names]

    comparison = {
        "condition_names": [],
        "raw_z_scores": [],
        "latent_z_scores": [],
        "raw_detected": [],
        "latent_detected": [],
        "true_deltas": [],
    }

    for name in common_names:
        raw_idx = raw_names.index(name)
        latent_idx = latent_names.index(name)

        raw_z = raw_results["z_scores"][raw_idx]
        latent_z = latent_results["z_scores"][latent_idx]

        comparison["condition_names"].append(name)
        comparison["raw_z_scores"].append(raw_z)
        comparison["latent_z_scores"].append(latent_z)
        comparison["raw_detected"].append(raw_z > 2.0)
        comparison["latent_detected"].append(latent_z > 2.0)
        comparison["true_deltas"].append(
            raw_results["true_deltas"][raw_idx]
        )

    # Convert to arrays
    for key in comparison:
        comparison[key] = np.array(comparison[key])

    return comparison


def print_comparison_table(comparison: dict) -> None:
    """Print comparison table to logger.

    Parameters
    ----------
    comparison : dict
        Output of :func:`compare_paths`.
    """
    logger.info("")
    logger.info("=" * 78)
    logger.info("RAW CSD vs LATENT CSD COMPARISON")
    logger.info("=" * 78)
    header = (
        f"{'Condition':<28} {'true_dA':>8} "
        f"{'raw_z':>8} {'raw_det':>8} "
        f"{'lat_z':>8} {'lat_det':>8}"
    )
    logger.info(header)
    logger.info("-" * 78)

    for k in range(len(comparison["condition_names"])):
        name = comparison["condition_names"][k]
        true_d = comparison["true_deltas"][k]
        raw_z = comparison["raw_z_scores"][k]
        lat_z = comparison["latent_z_scores"][k]
        raw_det = "YES" if comparison["raw_detected"][k] else "no"
        lat_det = "YES" if comparison["latent_detected"][k] else "no"

        logger.info(
            f"{name:<28} {true_d:>8.4f} "
            f"{raw_z:>8.2f} {raw_det:>8} "
            f"{lat_z:>8.2f} {lat_det:>8}"
        )

    logger.info("-" * 78)
    n_raw = int(np.sum(comparison["raw_detected"]))
    n_lat = int(np.sum(comparison["latent_detected"]))
    n_total = len(comparison["condition_names"])
    logger.info(
        "Detected: raw=%d/%d, latent=%d/%d", n_raw, n_total,
        n_lat, n_total,
    )
    logger.info("=" * 78)


# -- Main orchestrator ----------------------------------------------------


def run_full_validation(
    *,
    output_dir: Path,
    n_train: int = 200,
    n_eval: int = 50,
    n_latent_multiplier: int = 2,
    hidden_size: int = 64,
    ae_epochs: int = 100,
    max_iter: int = 128,
    tolerance: float = 1e-2,
    seed: int = 42,
) -> dict:
    """Execute full raw-vs-latent validation experiment.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    n_train : int
        Training samples for autoencoder.
    n_eval : int
        Evaluation samples per condition.
    n_latent_multiplier : int
        Latent dimension = multiplier * n_roi.
    hidden_size : int
        LSTM hidden size.
    ae_epochs : int
        Autoencoder training epochs.
    max_iter : int
        Maximum Gauss-Newton iterations per DCM fit.
    tolerance : float
        Convergence criterion on free energy change.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``'comparison'``, ``'raw_results'``,
        ``'latent_results'``, ``'save_path'``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()

    # Shared baseline A matrix
    A_base = make_sensorimotor_A(seed=seed)
    logger.info(
        "Baseline A: shape=%s, max_eigval_real=%.4f",
        A_base.shape,
        torch.linalg.eigvals(
            A_base.to(torch.complex128)
        ).real.max().item(),
    )

    # Path A: raw CSD
    raw_results = run_raw_csd_path(
        A_base=A_base,
        n_eval=n_eval,
        max_iter=max_iter,
        tolerance=tolerance,
        seed=seed,
    )

    # Path B: latent CSD
    latent_results = run_latent_csd_path(
        A_base=A_base,
        n_train=n_train,
        n_eval=n_eval,
        n_latent_multiplier=n_latent_multiplier,
        hidden_size=hidden_size,
        ae_epochs=ae_epochs,
        max_iter=max_iter,
        tolerance=tolerance,
        seed=seed,
        output_dir=output_dir,
    )

    # Compare
    comparison = compare_paths(raw_results, latent_results)
    print_comparison_table(comparison)

    # Save comparison
    save_path = output_dir / "raw_vs_latent_comparison.npz"
    np.savez(
        save_path,
        # Comparison
        condition_names=comparison["condition_names"],
        raw_z_scores=comparison["raw_z_scores"],
        latent_z_scores=comparison["latent_z_scores"],
        raw_detected=comparison["raw_detected"],
        latent_detected=comparison["latent_detected"],
        true_deltas=comparison["true_deltas"],
        # Raw path details
        raw_baseline_A_mean=raw_results["baseline_A_mean"],
        raw_baseline_A_std=raw_results["baseline_A_std"],
        raw_delta_A=raw_results["delta_A"],
        raw_perturbed_ij=raw_results["perturbed_ij"],
        raw_n_roi=raw_results["n_roi"],
        # Latent path details
        latent_baseline_A_mean=latent_results["baseline_A_mean"],
        latent_baseline_A_std=latent_results["baseline_A_std"],
        latent_delta_A=latent_results["delta_A"],
        latent_perturbed_ij=latent_results["perturbed_ij"],
        latent_n_latent=latent_results["n_latent"],
        # Metadata
        A_ground_truth=A_base.numpy(),
        roi_names=np.array(SENSORIMOTOR_ROI_NAMES),
        seed=seed,
        n_train=n_train,
        n_eval=n_eval,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    logger.info("Saved comparison: %s", save_path)

    total_elapsed = time.time() - total_t0
    logger.info("Total validation time: %.1f s", total_elapsed)

    return {
        "comparison": comparison,
        "raw_results": raw_results,
        "latent_results": latent_results,
        "save_path": str(save_path),
    }


def _print_sbatch_command(args: argparse.Namespace) -> None:
    """Print sbatch submission command for M3 cluster.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    sbatch_script = textwrap.dedent("""\
        #!/bin/bash
        #SBATCH --job-name=val_22_full
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=32G
        #SBATCH --time=04:00:00
        #SBATCH --output=cluster/logs/val_22_full_%j.out
        #SBATCH --error=cluster/logs/val_22_full_%j.err
        #SBATCH --partition=comp

        source cluster/lib/cluster_env.sh
        crlf_guard

        ENV_NAME="${{ENV_NAME:-actinf-py-scripts}}"

        setup_torch_threads 4
        activate_env "$ENV_NAME"
        verify_torch
        print_job_header "Phase 22 Full Validation (raw vs latent)"

        mkdir -p cluster/logs "{output_dir}"

        python scripts/22_run_full_validation.py \\
            --output-dir "{output_dir}" \\
            --n-train {n_train} \\
            --n-eval {n_eval} \\
            --max-iter {max_iter} \\
            --tolerance {tolerance} \\
            --ae-epochs {ae_epochs} \\
            --seed {seed}

        EXIT_CODE=$?
        echo ""
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "FULL VALIDATION COMPLETE (exit 0)"
            echo "Results: {output_dir}/raw_vs_latent_comparison.npz"
        else
            echo "FULL VALIDATION FAILED (exit ${{EXIT_CODE}})"
        fi
        exit $EXIT_CODE
    """).format(
        output_dir=str(args.output_dir).replace("\\", "/"),
        n_train=args.n_train,
        n_eval=args.n_eval,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        ae_epochs=args.ae_epochs,
        seed=args.seed,
    )

    sbatch_path = Path("cluster/sbatch/22_full_validation.sbatch")
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(sbatch_script)

    print(f"Wrote sbatch script: {sbatch_path}")
    print(f"Submit with:  sbatch {sbatch_path}")


def main() -> None:
    """CLI entry point for full validation orchestrator."""
    parser = argparse.ArgumentParser(
        description=(
            "Full validation: compare raw CSD vs latent CSD DCM "
            "perturbation sensitivity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_validation"),
        help="Root output directory",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=200,
        help="Training samples for autoencoder",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Evaluation samples per condition",
    )
    parser.add_argument(
        "--n-latent-multiplier",
        type=int,
        default=2,
        help="Latent dimension = multiplier * n_roi",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=100,
        help="Autoencoder training epochs",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=128,
        help="Maximum Gauss-Newton iterations per DCM fit",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="VL convergence criterion on free energy change",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--submit-cluster",
        action="store_true",
        help=(
            "Generate sbatch script and print submission command "
            "instead of running locally"
        ),
    )
    args = parser.parse_args()

    if args.submit_cluster:
        _print_sbatch_command(args)
        return

    run_full_validation(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_latent_multiplier=args.n_latent_multiplier,
        hidden_size=args.hidden_size,
        ae_epochs=args.ae_epochs,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
