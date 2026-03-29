"""Generate training data for amortized inference guides.

CLI script that wraps existing simulators to produce cached
(data, params) .pt files for training normalizing flow guides.
Supports both task DCM and spectral DCM variants.

Usage
-----
Task DCM (3 regions, 100 simulations):
    python scripts/generate_training_data.py \
        --variant task --n-simulations 100 --n-regions 3

Spectral DCM (3 regions, 100 simulations):
    python scripts/generate_training_data.py \
        --variant spectral --n-simulations 100 --n-regions 3

Output Structure
----------------
Each .pt file contains::

    {
        "data": [tensor, ...],      # list of BOLD/CSD tensors
        "params": [dict, ...],       # list of param dicts
        "metadata": {...}            # stimulus, masks, etc.
    }

References
----------
07-RESEARCH.md: Training data generation requirements.
07-RESEARCH.md Pitfall 5: NaN filtering for ODE divergence.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_noise import default_noise_priors
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)


def invert_A_to_A_free(A: torch.Tensor) -> torch.Tensor:
    """Invert parameterize_A to recover A_free from A.

    For diagonal elements: ``A_free_ii = log(-2 * A_ii)``,
    inverting ``A_ii = -exp(A_free_ii) / 2``.
    For off-diagonal elements: ``A_free_ij = A_ij`` (identity).

    Parameters
    ----------
    A : torch.Tensor
        Parameterized connectivity matrix, shape ``(N, N)``.
        Diagonal must be negative.

    Returns
    -------
    torch.Tensor
        Free parameters A_free, shape ``(N, N)``.

    See Also
    --------
    pyro_dcm.forward_models.neural_state.parameterize_A :
        Forward transform this function inverts.
    """
    N = A.shape[0]
    A_free = A.clone()
    diag_idx = torch.arange(N)
    # Invert: A_ii = -exp(x)/2 => x = log(-2*A_ii)
    A_free[diag_idx, diag_idx] = torch.log(-2.0 * A[diag_idx, diag_idx])
    return A_free


def generate_task_data(
    n_simulations: int,
    n_regions: int,
    n_inputs: int,
    seed: int,
    output_dir: str,
) -> str:
    """Generate task DCM training data.

    Parameters
    ----------
    n_simulations : int
        Number of simulations to generate.
    n_regions : int
        Number of brain regions.
    n_inputs : int
        Number of experimental inputs.
    seed : int
        Base random seed.
    output_dir : str
        Output directory for .pt files.

    Returns
    -------
    str
        Path to the saved .pt file.
    """
    data_list: list[torch.Tensor] = []
    params_list: list[dict[str, torch.Tensor]] = []

    # Simulation parameters
    TR = 2.0
    dt = 0.01
    SNR = 5.0
    n_blocks = 5
    block_duration = 30.0
    rest_duration = 30.0
    duration = n_blocks * (block_duration + rest_duration)

    # Create stimulus (shared across all simulations)
    stimulus = make_block_stimulus(
        n_blocks=n_blocks,
        block_duration=block_duration,
        rest_duration=rest_duration,
        n_inputs=n_inputs,
    )

    # Masks (all-ones for fully connected)
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.ones(n_regions, n_inputs, dtype=torch.float64)

    # Fine time grid matching simulator
    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)

    n_skipped = 0
    for i in range(n_simulations):
        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"  Task DCM: simulation {i + 1}/{n_simulations}"
            )

        # Generate random stable A matrix
        A = make_random_stable_A(n_regions, seed=seed + i)

        # Random C from N(0, 1) masked
        torch.manual_seed(seed + n_simulations + i)
        C = torch.randn(n_regions, n_inputs, dtype=torch.float64)
        C = C * c_mask

        # Simulate
        try:
            result = simulate_task_dcm(
                A, C, stimulus,
                duration=duration, dt=dt, TR=TR, SNR=SNR,
                seed=seed + 2 * n_simulations + i,
            )
        except Exception:
            n_skipped += 1
            continue

        bold = result["bold"]

        # Filter NaN/Inf (07-RESEARCH.md Pitfall 5)
        if torch.isnan(bold).any() or torch.isinf(bold).any():
            n_skipped += 1
            continue

        # Invert A to A_free
        A_free = invert_A_to_A_free(A)

        # Compute noise_prec from SNR and signal variance
        signal_std = result["bold_clean"].std(dim=0)
        noise_std = signal_std / SNR
        # noise_prec = 1 / noise_var = SNR^2 / signal_var
        signal_var = signal_std.pow(2).mean()
        noise_prec = torch.tensor(
            SNR**2 / signal_var.item() if signal_var.item() > 0
            else 1.0,
            dtype=torch.float64,
        )

        data_list.append(bold)
        params_list.append({
            "A_free": A_free,
            "C": C,
            "noise_prec": noise_prec,
        })

    if n_skipped > 0:
        print(
            f"  Skipped {n_skipped}/{n_simulations} simulations "
            f"(NaN/Inf/error)"
        )

    print(f"  Valid simulations: {len(data_list)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"task_{n_regions}regions_{n_simulations}sims.pt"
    filepath = os.path.join(output_dir, filename)

    torch.save({
        "data": data_list,
        "params": params_list,
        "metadata": {
            "stimulus": stimulus,
            "a_mask": a_mask,
            "c_mask": c_mask,
            "t_eval": t_eval,
            "TR": TR,
            "dt": dt,
            "n_regions": n_regions,
            "n_inputs": n_inputs,
        },
    }, filepath)

    print(f"  Saved to: {filepath}")
    return filepath


def generate_spectral_data(
    n_simulations: int,
    n_regions: int,
    seed: int,
    output_dir: str,
) -> str:
    """Generate spectral DCM training data.

    Parameters
    ----------
    n_simulations : int
        Number of simulations to generate.
    n_regions : int
        Number of brain regions.
    seed : int
        Base random seed.
    output_dir : str
        Output directory for .pt files.

    Returns
    -------
    str
        Path to the saved .pt file.
    """
    data_list: list[torch.Tensor] = []
    params_list: list[dict[str, torch.Tensor]] = []

    TR = 2.0
    n_freqs = 32
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)

    freqs_ref = None
    n_skipped = 0

    for i in range(n_simulations):
        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"  Spectral DCM: simulation {i + 1}/{n_simulations}"
            )

        # Generate random stable A matrix
        A = make_stable_A_spectral(n_regions, seed=seed + i)

        # Use default noise priors (vary slightly for diversity)
        torch.manual_seed(seed + n_simulations + i)
        priors = default_noise_priors(n_regions)
        noise_a = priors["a_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )
        noise_b = priors["b_prior_mean"] + 0.1 * torch.randn(
            2, 1, dtype=torch.float64,
        )
        noise_c = priors["c_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )

        noise_params = {"a": noise_a, "b": noise_b, "c": noise_c}

        # Simulate
        try:
            result = simulate_spectral_dcm(
                A, noise_params=noise_params,
                TR=TR, n_freqs=n_freqs, seed=seed + i,
            )
        except Exception:
            n_skipped += 1
            continue

        csd = result["csd"]

        # Filter NaN/Inf
        if torch.isnan(csd.real).any() or torch.isinf(csd.real).any():
            n_skipped += 1
            continue
        if torch.isnan(csd.imag).any() or torch.isinf(csd.imag).any():
            n_skipped += 1
            continue

        # Store reference frequency grid
        if freqs_ref is None:
            freqs_ref = result["freqs"]

        # Extract A_free from A
        A_free = invert_A_to_A_free(A)

        # Extract noise params using EXACT key paths from simulator
        # result["params"]["noise_params"]["a"/"b"/"c"]
        sim_noise_a = result["params"]["noise_params"]["a"]
        sim_noise_b = result["params"]["noise_params"]["b"]
        sim_noise_c = result["params"]["noise_params"]["c"]

        # csd_noise_scale: absent from simulator (noiseless CSD)
        # Use fixed default: HalfCauchy(1.0) prior mode
        csd_noise_scale = torch.tensor(1.0, dtype=torch.float64)

        data_list.append(csd)
        params_list.append({
            "A_free": A_free,
            "noise_a": sim_noise_a,
            "noise_b": sim_noise_b,
            "noise_c": sim_noise_c,
            "csd_noise_scale": csd_noise_scale,
        })

    if n_skipped > 0:
        print(
            f"  Skipped {n_skipped}/{n_simulations} simulations "
            f"(NaN/Inf/error)"
        )

    print(f"  Valid simulations: {len(data_list)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"spectral_{n_regions}regions_{n_simulations}sims.pt"
    )
    filepath = os.path.join(output_dir, filename)

    torch.save({
        "data": data_list,
        "params": params_list,
        "metadata": {
            "a_mask": a_mask,
            "freqs": freqs_ref,
            "n_regions": n_regions,
            "TR": TR,
            "n_freqs": n_freqs,
        },
    }, filepath)

    print(f"  Saved to: {filepath}")
    return filepath


def main() -> None:
    """CLI entry point for training data generation."""
    parser = argparse.ArgumentParser(
        description="Generate training data for amortized DCM guides",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["task", "spectral"],
        help="DCM variant: task or spectral",
    )
    parser.add_argument(
        "--n-simulations", type=int, default=10000,
        help="Number of simulations (default: 10000)",
    )
    parser.add_argument(
        "--n-regions", type=int, default=3,
        help="Number of brain regions (default: 3)",
    )
    parser.add_argument(
        "--n-inputs", type=int, default=1,
        help="Number of inputs, task DCM only (default: 1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/training/",
        help="Output directory (default: data/training/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )

    args = parser.parse_args()

    print(
        f"Generating {args.variant} DCM training data: "
        f"{args.n_simulations} simulations, "
        f"{args.n_regions} regions"
    )

    if args.variant == "task":
        generate_task_data(
            n_simulations=args.n_simulations,
            n_regions=args.n_regions,
            n_inputs=args.n_inputs,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.variant == "spectral":
        generate_spectral_data(
            n_simulations=args.n_simulations,
            n_regions=args.n_regions,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
