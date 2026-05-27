r"""Pipeline: stimulus -> TRIBE v2 -> ROI timeseries (.npz).

Extracts vertex-wise fMRI predictions from Meta TRIBE v2 for a given
stimulus video, parcellates to ROI timeseries using the Schaefer atlas,
and saves the result as a NumPy archive.

Requires GPU (A100 40 GB recommended) and tribev2 package.

Usage
-----
    python scripts/24_extract_tribe_latents.py \
        --video-path stimulus.mp4 \
        --output-dir results/phase24_tribe/ \
        --n-rois 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Run TRIBE v2 extraction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract ROI timeseries from TRIBE v2 vertex "
            "predictions for a stimulus video."
        ),
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to stimulus video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase24_tribe/",
        help="Directory for output .npz file.",
    )
    parser.add_argument(
        "--n-rois",
        type=int,
        default=100,
        help="Number of ROIs for Schaefer parcellation.",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="./tribe_cache",
        help="Cache directory for TRIBE v2 model weights.",
    )

    args = parser.parse_args()

    # --- Import after argparse (heavy dependencies) ---
    try:
        from pyro_dcm.foundation.tribe_extractor import (
            TRIBEExtractor,
        )
    except ImportError:
        print(
            "pyro_dcm not installed. "
            "Run: pip install -e . from project root."
        )
        sys.exit(1)

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRIBE v2 ROI Timeseries Extraction")
    print("=" * 60)
    print(f"  Video:       {args.video_path}")
    print(f"  N ROIs:      {args.n_rois}")
    print(f"  Cache:       {args.cache_folder}")
    print(f"  Output dir:  {output_dir}")
    print()

    # --- Load model ---
    print("Loading TRIBE v2 model...")
    extractor = TRIBEExtractor(
        cache_folder=args.cache_folder,
        n_rois=args.n_rois,
    )
    extractor.load_model()
    print("  Model loaded.")

    # --- Predict vertex timeseries ---
    print("Predicting vertex-wise fMRI timeseries...")
    vertex_timeseries = extractor.predict_vertex_timeseries(
        video_path=args.video_path,
    )
    print(
        f"  Vertex timeseries shape: {vertex_timeseries.shape}"
    )

    # --- Parcellate to ROIs ---
    print("Parcellating to ROI timeseries...")
    roi_timeseries, roi_names = extractor.extract_roi_timeseries(
        vertex_timeseries,
    )
    print(f"  ROI timeseries shape: {roi_timeseries.shape}")

    # --- Save ---
    output_path = output_dir / "tribe_roi_timeseries.npz"
    np.savez(
        output_path,
        roi_timeseries=roi_timeseries,
        roi_names=np.array(roi_names),
        n_rois=args.n_rois,
        vertex_shape=np.array(vertex_timeseries.shape),
        stimulus_path=args.video_path,
    )
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"  Shape:      {roi_timeseries.shape}")
    print(f"  N ROIs:     {args.n_rois}")
    print(f"  Timepoints: {roi_timeseries.shape[0]}")
    print(f"  Output:     {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
