"""Train amortized normalizing flow guides for DCM inference.

End-to-end training script that loads pre-generated simulation data,
creates summary networks and NSF flow guides, and runs SVI training
to learn the amortized inference mapping.

All simulation metadata (stimulus, masks, t_eval, TR, dt, n_regions,
n_inputs) is loaded from the .pt data file -- no hardcoded values
needed.

Usage
-----
Task DCM::

    python scripts/train_amortized_guide.py \
        --variant task \
        --data-path data/training/task_3regions_100sims.pt \
        --n-train-steps 1000 \
        --output-dir models/task/

Spectral DCM::

    python scripts/train_amortized_guide.py \
        --variant spectral \
        --data-path data/training/spectral_3regions_100sims.pt \
        --n-train-steps 5000 \
        --output-dir models/spectral/

Notes
-----
For task DCM, each SVI step runs the full ODE integration (~1-2s on
CPU). Training with 50,000 steps takes ~14-28 hours. Consider using
a coarser dt (loaded from metadata) or shorter simulations for
faster iteration.

For spectral DCM, the forward model is algebraic (no ODE), so steps
are much faster (~0.01s each).

References
----------
[REF-042] Radev et al. (2020). BayesFlow.
[REF-043] Cranmer, Brehmer & Louppe (2020). SBI frontier.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_dcm.guides import (
    AmortizedFlowGuide,
    BoldSummaryNet,
    CsdSummaryNet,
    SpectralDCMPacker,
    TaskDCMPacker,
)
from pyro_dcm.models.amortized_wrappers import (
    amortized_spectral_dcm_model,
    amortized_task_dcm_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train amortized DCM inference guides",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["task", "spectral"],
        help="DCM variant: task or spectral",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to .pt training data file",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=128,
        help="Summary network embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--n-transforms", type=int, default=5,
        help="Number of NSF transforms (default: 5)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=8,
        help="Number of spline bins (default: 8)",
    )
    parser.add_argument(
        "--hidden-features", type=str, default="256,256",
        help="Hidden features per transform, comma-separated (default: 256,256)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=10.0,
        help="Gradient clip norm (default: 10.0)",
    )
    parser.add_argument(
        "--num-particles", type=int, default=16,
        help="ELBO particles (default: 16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128, not yet used for batching)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=200,
        help="Number of epochs (default: 200, spectral DCM only)",
    )
    parser.add_argument(
        "--n-train-steps", type=int, default=50000,
        help="Total SVI steps (default: 50000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/",
        help="Output directory (default: models/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    return parser.parse_args()


def load_dataset(data_path: str) -> dict:
    """Load training dataset from .pt file.

    Parameters
    ----------
    data_path : str
        Path to .pt file produced by generate_training_data.py.

    Returns
    -------
    dict
        Dataset with keys ``data``, ``params``, ``metadata``.
    """
    return torch.load(data_path, weights_only=False)


def create_task_components(
    metadata: dict,
    train_params: list[dict],
    embed_dim: int,
    n_transforms: int,
    n_bins: int,
    hidden_features: list[int],
) -> tuple[AmortizedFlowGuide, TaskDCMPacker]:
    """Create summary net, packer, and guide for task DCM.

    Parameters
    ----------
    metadata : dict
        Dataset metadata from .pt file.
    train_params : list of dict
        Training parameter dicts for standardization fitting.
    embed_dim : int
        Summary network embedding dimension.
    n_transforms : int
        Number of NSF transforms.
    n_bins : int
        Number of spline bins.
    hidden_features : list of int
        Hidden layer sizes per transform.

    Returns
    -------
    tuple of (AmortizedFlowGuide, TaskDCMPacker)
    """
    n_regions = metadata["n_regions"]
    n_inputs = metadata["n_inputs"]
    a_mask = metadata["a_mask"]
    c_mask = metadata["c_mask"]

    packer = TaskDCMPacker(n_regions, n_inputs, a_mask, c_mask)
    packer.fit_standardization(train_params)

    summary_net = BoldSummaryNet(n_regions, embed_dim).double()
    guide = AmortizedFlowGuide(
        summary_net, packer.n_features,
        embed_dim=embed_dim,
        n_transforms=n_transforms,
        n_bins=n_bins,
        hidden_features=hidden_features,
        packer=packer,
    ).double()

    return guide, packer


def create_spectral_components(
    metadata: dict,
    train_params: list[dict],
    embed_dim: int,
    n_transforms: int,
    n_bins: int,
    hidden_features: list[int],
) -> tuple[AmortizedFlowGuide, SpectralDCMPacker]:
    """Create summary net, packer, and guide for spectral DCM.

    Parameters
    ----------
    metadata : dict
        Dataset metadata from .pt file.
    train_params : list of dict
        Training parameter dicts for standardization fitting.
    embed_dim : int
        Summary network embedding dimension.
    n_transforms : int
        Number of NSF transforms.
    n_bins : int
        Number of spline bins.
    hidden_features : list of int
        Hidden layer sizes per transform.

    Returns
    -------
    tuple of (AmortizedFlowGuide, SpectralDCMPacker)
    """
    n_regions = metadata["n_regions"]
    n_freqs = metadata["n_freqs"]

    packer = SpectralDCMPacker(n_regions)
    packer.fit_standardization(train_params)

    summary_net = CsdSummaryNet(n_regions, n_freqs, embed_dim).double()
    guide = AmortizedFlowGuide(
        summary_net, packer.n_features,
        embed_dim=embed_dim,
        n_transforms=n_transforms,
        n_bins=n_bins,
        hidden_features=hidden_features,
        packer=packer,
    ).double()

    return guide, packer


def train_loop(
    svi: SVI,
    data_list: list[torch.Tensor],
    n_steps: int,
    svi_args_fn: callable,
    output_dir: str,
    variant: str,
) -> list[float]:
    """Run the SVI training loop.

    Samples random items from the dataset for each step, prints
    progress every 100 steps, and saves checkpoints every 5000 steps.

    Parameters
    ----------
    svi : SVI
        Configured Pyro SVI instance.
    data_list : list of torch.Tensor
        Training data tensors.
    n_steps : int
        Total number of SVI steps.
    svi_args_fn : callable
        Function that takes a data tensor and returns SVI step args.
    output_dir : str
        Directory for saving checkpoints.
    variant : str
        DCM variant name for checkpoint filenames.

    Returns
    -------
    list of float
        ELBO losses at each step.
    """
    losses: list[float] = []
    n_data = len(data_list)
    t0 = time.time()

    for step in range(n_steps):
        i = torch.randint(n_data, (1,)).item()
        args = svi_args_fn(data_list[i])
        loss = svi.step(*args)
        losses.append(loss)

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"Step {step}/{n_steps}: loss={loss:.2f} "
                f"({elapsed:.1f}s elapsed)"
            )

        if step % 5000 == 0 and step > 0:
            ckpt_path = os.path.join(
                output_dir, f"{variant}_checkpoint_{step}.pt",
            )
            torch.save(
                {"step": step, "losses": losses},
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    return losses


def main() -> None:
    """Main entry point for amortized guide training."""
    args = parse_args()

    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    # Disable validation: ODE-based models produce NaN for some
    # parameter samples from the untrained flow. The wrapper model
    # handles NaN via detach-and-replace (see amortized_wrappers.py).
    pyro.enable_validation(False)

    hidden_features = [int(x) for x in args.hidden_features.split(",")]

    print(f"Loading data from: {args.data_path}")
    dataset = load_dataset(args.data_path)
    data_list = dataset["data"]
    params_list = dataset["params"]
    metadata = dataset["metadata"]

    # Train/val split
    n_val = max(1, int(len(data_list) * args.val_fraction))
    train_data = data_list[n_val:]
    train_params = params_list[n_val:]

    print(
        f"Dataset: {len(data_list)} total, {len(train_data)} train, "
        f"{n_val} val"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.variant == "task":
        guide, packer = create_task_components(
            metadata, train_params, args.embed_dim,
            args.n_transforms, args.n_bins, hidden_features,
        )
        stimulus = metadata["stimulus"]
        a_mask = metadata["a_mask"]
        c_mask = metadata["c_mask"]
        TR = metadata["TR"]

        # Use coarser dt for SVI (metadata dt is fine-grained for simulation)
        # dt=0.5 balances speed and numerical stability for ODE integration
        dt = 0.5
        duration = metadata["t_eval"][-1].item() + metadata["dt"]
        t_eval = torch.arange(0, duration, dt, dtype=torch.float64)

        # Note: vectorize_particles=False because parameterize_A
        # does not support batch dims from vectorized particles.
        # num_particles > 1 uses sequential evaluation.
        svi = SVI(
            amortized_task_dcm_model, guide,
            ClippedAdam({"lr": args.lr, "clip_norm": args.clip_norm}),
            loss=Trace_ELBO(
                num_particles=args.num_particles,
                vectorize_particles=False,
            ),
        )

        def task_svi_args(bold):
            return (bold, stimulus, a_mask, c_mask, t_eval, TR, dt, packer)

        losses = train_loop(
            svi, train_data, args.n_train_steps,
            task_svi_args, args.output_dir, "task",
        )

    elif args.variant == "spectral":
        guide, packer = create_spectral_components(
            metadata, train_params, args.embed_dim,
            args.n_transforms, args.n_bins, hidden_features,
        )
        a_mask = metadata["a_mask"]
        freqs = metadata["freqs"]

        svi = SVI(
            amortized_spectral_dcm_model, guide,
            ClippedAdam({"lr": args.lr, "clip_norm": args.clip_norm}),
            loss=Trace_ELBO(
                num_particles=args.num_particles,
                vectorize_particles=False,
            ),
        )

        def spectral_svi_args(csd):
            return (csd, freqs, a_mask, packer)

        losses = train_loop(
            svi, train_data, args.n_train_steps,
            spectral_svi_args, args.output_dir, "spectral",
        )

    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.variant}_final.pt")
    torch.save({
        "guide_state_dict": guide.state_dict(),
        "packer_mean": packer.mean_,
        "packer_std": packer.std_,
        "losses": losses,
        "args": vars(args),
        "metadata": metadata,
    }, final_path)
    print(f"Final model saved: {final_path}")
    print(f"Final loss: {losses[-1]:.2f}")


if __name__ == "__main__":
    main()
