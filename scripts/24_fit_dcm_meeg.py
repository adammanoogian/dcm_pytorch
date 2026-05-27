r"""Pipeline: latent dynamics -> CSD -> spectral DCM -> posterior A.

Loads latent dynamics from M/EEG foundation model extraction (Phase 24),
computes cross-spectral density using Welch periodogram, and fits
spectral or latent-circuit DCM to recover effective connectivity
(posterior A matrix).

Usage
-----
::

    python scripts/24_fit_dcm_meeg.py \
        --input-npz results/phase24_meeg/meeg_latent_dynamics.npz \
        --output-dir results/phase24_meeg/ \
        --dcm-type spectral \
        --num-steps 2000 \
        --n-restarts 10

Requires
--------
- pyro_dcm (this project)
- scipy (for CSD computation)
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
import pyro
import torch


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for DCM fitting on M/EEG latents.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit DCM (spectral or latent-circuit) to latent dynamics "
            "extracted from M/EEG foundation models."
        ),
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help=(
            "Path to .npz file from 24_extract_meeg_latents.py "
            "containing latent_dynamics array."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase24_meeg/",
        help="Output directory (default: results/phase24_meeg/).",
    )
    parser.add_argument(
        "--dcm-type",
        choices=["spectral", "latent_circuit"],
        default="spectral",
        help="DCM variant to fit (default: spectral).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="SVI optimization steps (default: 2000).",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Multi-start restarts (default: 10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01).",
    )
    return parser


def _fit_spectral(
    latent_dynamics: np.ndarray,
    num_steps: int,
    n_restarts: int,
    lr: float,
) -> dict[str, np.ndarray | float]:
    """Fit spectral DCM to latent dynamics via CSD.

    Parameters
    ----------
    latent_dynamics : np.ndarray
        Shape ``(n_epochs, n_patches, n_components)``.
    num_steps : int
        SVI steps per restart.
    n_restarts : int
        Number of random restarts.
    lr : float
        Learning rate.

    Returns
    -------
    dict[str, np.ndarray | float]
        Keys: A_mean, A_std, final_loss.
    """
    from pyro_dcm.forward_models.csd_computation import (
        compute_empirical_csd,
    )
    from pyro_dcm.models import (
        create_guide,
        extract_posterior_params,
        run_svi,
        spectral_dcm_model,
    )

    # Treat concatenated epochs as a single timeseries
    n_epochs, n_patches, n_comp = latent_dynamics.shape
    timeseries = latent_dynamics.reshape(
        n_epochs * n_patches, n_comp
    )

    # Compute CSD (Welch) -- assume 1 Hz sample rate for patch dynamics
    fs = 1.0
    n_freqs = min(32, timeseries.shape[0] // 2)
    freqs = np.linspace(0.01, fs / 2, n_freqs)
    csd = compute_empirical_csd(timeseries, fs=fs, freqs=freqs)

    # Convert to torch
    csd_tensor = torch.tensor(csd, dtype=torch.complex128)
    freqs_tensor = torch.tensor(freqs, dtype=torch.float64)
    a_mask = torch.ones(
        n_comp, n_comp, dtype=torch.float64
    )

    # Fit
    pyro.clear_param_store()

    model_args = (csd_tensor, freqs_tensor, a_mask)

    if n_restarts > 1:
        guide_factory = functools.partial(
            create_guide,
            spectral_dcm_model,
            guide_type="auto_lowrank_mvn",
        )
        guide = guide_factory()
        result = run_svi(
            spectral_dcm_model,
            guide,
            model_args,
            num_steps=num_steps,
            lr=lr,
            n_restarts=n_restarts,
            guide_factory=guide_factory,
        )
    else:
        guide = create_guide(
            spectral_dcm_model,
            guide_type="auto_lowrank_mvn",
        )
        result = run_svi(
            spectral_dcm_model,
            guide,
            model_args,
            num_steps=num_steps,
            lr=lr,
        )

    # Extract posterior
    posterior = extract_posterior_params(guide, model_args)

    a_mean = posterior["A_free"]["mean"].numpy()
    a_std = posterior["A_free"]["std"].numpy()
    return {
        "A_mean": a_mean,
        "A_std": a_std,
        "final_loss": result["final_loss"],
    }


def _fit_latent_circuit(
    latent_dynamics: np.ndarray,
    num_steps: int,
    n_restarts: int,
    lr: float,
) -> dict[str, np.ndarray | float]:
    """Fit latent-circuit DCM to latent dynamics.

    Parameters
    ----------
    latent_dynamics : np.ndarray
        Shape ``(n_epochs, n_patches, n_components)``.
    num_steps : int
        SVI steps per restart.
    n_restarts : int
        Number of random restarts.
    lr : float
        Learning rate.

    Returns
    -------
    dict[str, np.ndarray | float]
        Keys: A_mean, A_std, final_loss.
    """
    from pyro_dcm.models import (
        create_guide,
        extract_posterior_params,
        latent_circuit_dcm_model,
        run_svi,
    )

    # Use first epoch as representative trajectory
    trajectory = latent_dynamics[0]  # (n_patches, n_comp)
    n_patches, n_comp = trajectory.shape

    observed = torch.tensor(trajectory, dtype=torch.float64)
    # Create dummy stimulus (constant driving input)
    stimulus = torch.ones(n_patches, 1, dtype=torch.float64)
    a_mask = torch.ones(n_comp, n_comp, dtype=torch.float64)
    c_mask = torch.ones(n_comp, 1, dtype=torch.float64)
    t_eval = torch.linspace(0, n_patches - 1, n_patches).double()
    dt = 1.0  # patch-level time step

    pyro.clear_param_store()

    model_args = (observed, stimulus, a_mask, c_mask, t_eval, dt)

    if n_restarts > 1:
        guide_factory = functools.partial(
            create_guide,
            latent_circuit_dcm_model,
            guide_type="auto_lowrank_mvn",
        )
        guide = guide_factory()
        result = run_svi(
            latent_circuit_dcm_model,
            guide,
            model_args,
            num_steps=num_steps,
            lr=lr,
            n_restarts=n_restarts,
            guide_factory=guide_factory,
        )
    else:
        guide = create_guide(
            latent_circuit_dcm_model,
            guide_type="auto_lowrank_mvn",
        )
        result = run_svi(
            latent_circuit_dcm_model,
            guide,
            model_args,
            num_steps=num_steps,
            lr=lr,
        )

    posterior = extract_posterior_params(guide, model_args)

    a_mean = posterior["A_free"]["mean"].numpy()
    a_std = posterior["A_free"]["std"].numpy()
    return {
        "A_mean": a_mean,
        "A_std": a_std,
        "final_loss": result["final_loss"],
    }


def main() -> None:
    """Run DCM fitting on M/EEG foundation model latents."""
    args = build_parser().parse_args()

    # --- Load latent dynamics ---
    print(f"Loading latent dynamics from {args.input_npz} ...")
    data = np.load(args.input_npz, allow_pickle=True)
    latent_dynamics = data["latent_dynamics"]
    model_name = str(data["model_name"])
    n_components = int(data["n_components"])

    print(
        f"  Model: {model_name}, "
        f"shape: {latent_dynamics.shape}, "
        f"n_components: {n_components}"
    )

    # --- Fit DCM ---
    print(
        f"\nFitting {args.dcm_type} DCM "
        f"({args.num_steps} steps, "
        f"{args.n_restarts} restarts) ..."
    )

    if args.dcm_type == "spectral":
        results = _fit_spectral(
            latent_dynamics,
            args.num_steps,
            args.n_restarts,
            args.lr,
        )
    elif args.dcm_type == "latent_circuit":
        results = _fit_latent_circuit(
            latent_dynamics,
            args.num_steps,
            args.n_restarts,
            args.lr,
        )
    else:
        raise ValueError(f"Unknown DCM type: {args.dcm_type}")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "meeg_dcm_results.npz"

    np.savez(
        output_path,
        A_mean=results["A_mean"],
        A_std=results["A_std"],
        final_loss=results["final_loss"],
        dcm_type=args.dcm_type,
        model_name=model_name,
    )

    # --- Summary ---
    a_mean = results["A_mean"]
    a_std = results["A_std"]
    n = a_mean.shape[0]

    print("\n--- DCM Fitting Summary ---")
    print(f"  DCM type:      {args.dcm_type}")
    print(f"  Final loss:    {results['final_loss']:.2f}")
    print(f"  A matrix ({n}x{n}) posterior mean (std):")
    for i in range(n):
        row = "    " + "  ".join(
            f"{a_mean[i, j]:+.4f}({a_std[i, j]:.3f})"
            for j in range(n)
        )
        print(row)
    print(f"  Saved to:      {output_path}")


if __name__ == "__main__":
    main()
