"""Perturbation experiment for DCM interpretability validation.

Generates timeseries from a known baseline A matrix, trains an LSTM
autoencoder, fits spectral DCM to latent CSD, then perturbs specific
connections in A, regenerates timeseries through the SAME autoencoder,
fits DCM to the new latent CSD, and checks whether perturbations
appear in the DCM posterior. If the pipeline detects known
perturbations, it validates DCM as an interpretability tool.

Usage
-----
python scripts/22_perturbation_experiment.py --output-dir results/perturbation

Cluster
-------
sbatch cluster/sbatch/22_perturbation.sbatch
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.inference.variational_laplace import (
    extract_vl_posterior,
    run_variational_laplace,
)
from pyro_dcm.neural_data_models.latent_extraction import (
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

# ---------------------------------------------------------------
# Perturbation conditions
# ---------------------------------------------------------------
# Region index mapping (from meg_simulator.py):
#   M1_lh=0, M1_rh=1, S1_lh=2, S1_rh=3
#   PMC_lh=4, PMC_rh=5, SMA_lh=6, SMA_rh=7
#   A1_lh=8, A1_rh=9

PERTURBATION_CONDITIONS: list[dict] = [
    # M1_lh <- S1_lh (strong intra: 0.15)
    {"name": "M1_S1_strengthen", "i": 0, "j": 2, "factor": 1.5},
    {"name": "M1_S1_weaken", "i": 0, "j": 2, "factor": 0.5},
    {"name": "M1_S1_remove", "i": 0, "j": 2, "factor": 0.0},
    # M1_lh <- PMC_lh (feedforward: 0.10)
    {"name": "PMC_M1_strengthen", "i": 0, "j": 4, "factor": 1.5},
    {"name": "PMC_M1_weaken", "i": 0, "j": 4, "factor": 0.5},
    {"name": "PMC_M1_remove", "i": 0, "j": 4, "factor": 0.0},
    # M1_lh <- A1_lh (weak auditory-motor: 0.05)
    {"name": "A1_M1_strengthen", "i": 0, "j": 8, "factor": 1.5},
    {"name": "A1_M1_remove", "i": 0, "j": 8, "factor": 0.0},
    # M1_lh <- M1_rh (bilateral: 0.08)
    {"name": "bilateral_M1_strengthen", "i": 0, "j": 1, "factor": 2.0},
]


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


def run_perturbation_experiment(
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
    """Execute the full perturbation experiment.

    Parameters
    ----------
    output_dir : Path
        Directory for saving results.
    n_train : int
        Number of training samples for autoencoder.
    n_eval : int
        Number of evaluation samples per condition.
    n_latent_multiplier : int
        Latent dimension multiplier (n_latent = multiplier * n_roi).
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
        Experiment results suitable for np.savez.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_roi = 10
    n_latent = n_latent_multiplier * n_roi

    # ------------------------------------------------------------------
    # Step 1: Generate baseline dataset
    # ------------------------------------------------------------------
    logger.info("Step 1: Generating baseline dataset...")
    t0 = time.time()

    A_base = make_sensorimotor_A(seed=seed)
    logger.info("  A_base shape: %s", A_base.shape)
    logger.info(
        "  A_base max eigval real part: %.4f",
        torch.linalg.eigvals(A_base.to(torch.complex128)).real.max().item(),
    )

    train_result = simulate_meg_timeseries(
        A_base, n_samples=n_train, seed=seed
    )
    train_data = train_result["timeseries"]
    logger.info("  Training data shape: %s", train_data.shape)

    eval_baseline_result = simulate_meg_timeseries(
        A_base, n_samples=n_eval, seed=seed + 1000
    )
    eval_baseline_data = eval_baseline_result["timeseries"]
    logger.info("  Baseline eval data shape: %s", eval_baseline_data.shape)
    logger.info("  Step 1 took %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 2: Train autoencoder on baseline data
    # ------------------------------------------------------------------
    logger.info("Step 2: Training autoencoder...")
    t0 = time.time()

    torch.manual_seed(seed)
    ae_model = MEGAutoencoder(
        n_roi=n_roi, n_latent=n_latent, hidden_size=hidden_size
    )
    trainer = AutoencoderTrainer(ae_model, lr=1e-3)

    train_dataset = TensorDataset(train_data.float())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    history = trainer.train(train_loader, n_epochs=ae_epochs, log_every=10)
    logger.info(
        "  AE training: %d epochs, final loss=%.6f",
        history["final_epoch"] + 1,
        history["train_losses"][-1],
    )

    # Save checkpoint
    ckpt_path = output_dir / "ae_checkpoint.pt"
    trainer.save_checkpoint(ckpt_path)
    logger.info("  Saved AE checkpoint: %s", ckpt_path)
    logger.info("  Step 2 took %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 3: Extract baseline latent CSD and fit baseline DCM
    # ------------------------------------------------------------------
    logger.info("Step 3: Fitting baseline DCM...")
    t0 = time.time()

    baseline_latents = extract_latent_trajectories(
        ae_model, eval_baseline_data.float()
    )
    logger.info("  Baseline latents shape: %s", baseline_latents.shape)

    baseline_csd_result = compute_latent_csd(
        baseline_latents, sfreq=250.0, average_over_samples=True
    )
    baseline_dcm_input = prepare_for_spectral_dcm(baseline_csd_result)

    baseline_posterior = _fit_spectral_dcm(
        baseline_dcm_input["csd"],
        baseline_dcm_input["freqs"],
        baseline_dcm_input["a_mask"],
        max_iter=max_iter,
        tolerance=tolerance,
    )
    logger.info(
        "  Baseline DCM free energy: %.2f (converged=%s, %d iters)",
        baseline_posterior["free_energy"],
        baseline_posterior["converged"],
        baseline_posterior["n_iterations"],
    )
    logger.info("  Step 3 took %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 4: Perturbation sweep
    # ------------------------------------------------------------------
    logger.info(
        "Step 4: Running perturbation sweep (%d conditions)...",
        len(PERTURBATION_CONDITIONS),
    )

    results_list = []
    for cond_idx, cond in enumerate(PERTURBATION_CONDITIONS):
        t0 = time.time()
        name = cond["name"]
        i, j, factor = cond["i"], cond["j"], cond["factor"]

        logger.info(
            "  Condition %d/%d: %s (A[%d,%d] *= %.2f)",
            cond_idx + 1,
            len(PERTURBATION_CONDITIONS),
            name,
            i,
            j,
            factor,
        )

        # Create perturbed A
        A_perturbed = A_base.clone()
        true_delta = A_base[i, j].item() * (factor - 1.0)
        A_perturbed[i, j] = A_base[i, j] * factor

        # Check stability of perturbed A
        eigvals = torch.linalg.eigvals(
            A_perturbed.to(torch.complex128)
        )
        max_real = eigvals.real.max().item()
        if max_real >= 0:
            logger.warning(
                "    Perturbed A is unstable (max eigval real=%.4f), "
                "skipping condition %s",
                max_real,
                name,
            )
            continue

        # Generate perturbed eval data through SAME autoencoder
        perturbed_result = simulate_meg_timeseries(
            A_perturbed, n_samples=n_eval, seed=seed + 2000 + cond_idx
        )
        perturbed_data = perturbed_result["timeseries"]

        # Extract latent trajectories (no retraining!)
        perturbed_latents = extract_latent_trajectories(
            ae_model, perturbed_data.float()
        )

        # Compute latent CSD
        perturbed_csd_result = compute_latent_csd(
            perturbed_latents, sfreq=250.0, average_over_samples=True
        )
        perturbed_dcm_input = prepare_for_spectral_dcm(perturbed_csd_result)

        # Fit spectral DCM
        perturbed_posterior = _fit_spectral_dcm(
            perturbed_dcm_input["csd"],
            perturbed_dcm_input["freqs"],
            perturbed_dcm_input["a_mask"],
            max_iter=max_iter,
            tolerance=tolerance,
        )

        # Compute delta_A
        delta_a = (
            perturbed_posterior["A_mean"] - baseline_posterior["A_mean"]
        )

        elapsed = time.time() - t0
        logger.info(
            "    delta_A[%d,%d]=%.4f (true=%.4f), F=%.2f, "
            "took %.1f s",
            i,
            j,
            delta_a[i, j],
            true_delta,
            perturbed_posterior["free_energy"],
            elapsed,
        )

        results_list.append({
            "name": name,
            "i": i,
            "j": j,
            "factor": factor,
            "true_delta": true_delta,
            "delta_A": delta_a,
            "A_mean_perturbed": perturbed_posterior["A_mean"],
            "A_std_perturbed": perturbed_posterior["A_std"],
            "free_energy": perturbed_posterior["free_energy"],
        })

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    logger.info("Step 5: Saving results...")

    # Build arrays for npz
    condition_names = [r["name"] for r in results_list]
    perturbed_ij = np.array([[r["i"], r["j"]] for r in results_list])
    factors = np.array([r["factor"] for r in results_list])
    true_deltas = np.array([r["true_delta"] for r in results_list])
    delta_A_stack = np.stack([r["delta_A"] for r in results_list])
    A_mean_perturbed_stack = np.stack(
        [r["A_mean_perturbed"] for r in results_list]
    )
    A_std_perturbed_stack = np.stack(
        [r["A_std_perturbed"] for r in results_list]
    )
    free_energies = np.array([r["free_energy"] for r in results_list])

    save_path = output_dir / "perturbation_results.npz"
    np.savez(
        save_path,
        condition_names=np.array(condition_names),
        perturbed_ij=perturbed_ij,
        factors=factors,
        true_deltas=true_deltas,
        delta_A=delta_A_stack,
        A_mean_baseline=baseline_posterior["A_mean"],
        A_std_baseline=baseline_posterior["A_std"],
        A_mean_perturbed=A_mean_perturbed_stack,
        A_std_perturbed=A_std_perturbed_stack,
        A_ground_truth=A_base.numpy(),
        free_energies=free_energies,
        roi_names=np.array(SENSORIMOTOR_ROI_NAMES),
        seed=seed,
        n_train=n_train,
        n_eval=n_eval,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    logger.info("  Saved results: %s", save_path)

    return {
        "save_path": str(save_path),
        "n_conditions": len(results_list),
        "baseline_free_energy": baseline_posterior["free_energy"],
    }


def main() -> None:
    """CLI entry point for perturbation experiment."""
    parser = argparse.ArgumentParser(
        description="Perturbation experiment for DCM interpretability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/perturbation"),
        help="Directory for saving results",
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
        help="Evaluation samples per perturbation condition",
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
    args = parser.parse_args()

    result = run_perturbation_experiment(
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

    logger.info("Experiment complete!")
    logger.info("  Conditions run: %d", result["n_conditions"])
    logger.info("  Results saved to: %s", result["save_path"])


if __name__ == "__main__":
    main()
