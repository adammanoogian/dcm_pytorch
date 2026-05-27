r"""Pipeline: M/EEG data -> foundation model -> latent dynamics (.npz).

Loads epoched M/EEG data from an MNE .fif file, passes it through a
pretrained foundation model (LaBraM or BrainOmni), extracts patch-level
or layer-level latent dynamics, applies PCA to reduce to
DCM-compatible dimensions, and saves the result as a NumPy archive.

Usage
-----
::

    python scripts/24_extract_meeg_latents.py \
        --model labram \
        --input-fif data/epochs-epo.fif \
        --output-dir results/phase24_meeg/ \
        --n-components 4

Requires
--------
- braindecode >= 1.3.0 (for LaBraM)
- BrainOmni (https://github.com/OpenTSLab/BrainOmni)
- mne >= 1.6
- scikit-learn (for PCA reduction)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for M/EEG latent extraction.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract latent dynamics from M/EEG data using a "
            "pretrained foundation model (LaBraM or BrainOmni)."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["labram", "brainomni"],
        default="labram",
        help="Foundation model to use (default: labram).",
    )
    parser.add_argument(
        "--input-fif",
        type=str,
        required=True,
        help="Path to MNE .fif file with epoched M/EEG data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase24_meeg/",
        help="Output directory (default: results/phase24_meeg/).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="PCA components for DCM dimensions (default: 4).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local pretrained weights (optional).",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=200.0,
        help="Sampling frequency in Hz (default: 200.0).",
    )
    parser.add_argument(
        "--n-chans",
        type=int,
        default=64,
        help="Number of EEG/MEG channels (default: 64).",
    )
    return parser


def main() -> None:
    """Run M/EEG latent extraction pipeline."""
    args = build_parser().parse_args()

    # --- Load M/EEG data ---
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "MNE-Python is required. "
            "Install with: pip install mne>=1.6"
        ) from exc

    print(f"Loading epochs from {args.input_fif} ...")
    epochs = mne.read_epochs(args.input_fif, preload=True)
    data = epochs.get_data()  # (n_epochs, n_chans, n_times)
    print(
        f"  Loaded {data.shape[0]} epochs, "
        f"{data.shape[1]} channels, "
        f"{data.shape[2]} time points"
    )

    input_tensor = torch.from_numpy(data).float()

    # --- Extract latent dynamics ---
    if args.model == "labram":
        from pyro_dcm.foundation.labram_extractor import (
            LaBraMExtractor,
        )

        n_times = data.shape[2]
        extractor = LaBraMExtractor(
            n_times=n_times,
            n_chans=args.n_chans,
            sfreq=args.sfreq,
        )
        print("Loading LaBraM model ...")
        extractor.load_model(checkpoint_path=args.checkpoint)

        print("Extracting features ...")
        result = extractor.extract_features(input_tensor)
        features = result["features"]

        print("Reducing to DCM space via PCA ...")
        reduced, pca = extractor.reduce_to_dcm_space(
            features, n_components=args.n_components
        )
        explained_var = pca.explained_variance_ratio_

    elif args.model == "brainomni":
        from pyro_dcm.foundation.brainomni_extractor import (
            BrainOmniExtractor,
        )

        extractor = BrainOmniExtractor(modality="eeg")
        print("Loading BrainOmni model ...")
        extractor.load_model(checkpoint_path=args.checkpoint)

        print("Extracting latents via forward hooks ...")
        activations = extractor.extract_latents(input_tensor)

        # Use the last encoder block by default
        layer_names = list(activations.keys())
        if not layer_names:
            raise RuntimeError(
                "No activations captured. Check model architecture."
            )
        target_layer = layer_names[-1]
        print(f"  Using layer: {target_layer}")

        print("Reducing to DCM space via PCA ...")
        reduced, pca = extractor.reduce_to_dcm_space(
            activations,
            target_layer,
            n_components=args.n_components,
        )
        explained_var = pca.explained_variance_ratio_

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "meeg_latent_dynamics.npz"

    np.savez(
        output_path,
        latent_dynamics=reduced,
        model_name=args.model,
        n_components=args.n_components,
        pca_explained_variance=explained_var,
    )

    # --- Summary ---
    print("\n--- Extraction Summary ---")
    print(f"  Model:           {args.model}")
    print(f"  Input shape:     {data.shape}")
    print(f"  Latent shape:    {reduced.shape}")
    print(
        f"  Variance explained: "
        f"{explained_var.sum():.3f} "
        f"({', '.join(f'{v:.3f}' for v in explained_var)})"
    )
    print(f"  Saved to:        {output_path}")


if __name__ == "__main__":
    main()
