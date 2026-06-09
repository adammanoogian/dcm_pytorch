"""Parcellation utilities for vertex-to-ROI aggregation.

Aggregates fsaverage5 surface timeseries (20 484 vertices) into Schaefer-2018
parcels. The vertex-to-parcel assignment is obtained by projecting the
volumetric Schaefer atlas onto the fsaverage5 surface with nilearn's
nearest-neighbour ``vol_to_surf`` -- NOT by any contiguous-block heuristic.

The label loader requires ``nilearn`` (and downloads the atlas + fsaverage5
meshes on first use; nilearn caches them under ``~/nilearn_data``). If the data
cannot be obtained the loader raises -- it never silently returns an
approximate parcellation.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

# fsaverage5 vertex counts (per hemisphere and total).
_FSAVERAGE5_VERTICES_PER_HEMI = 10242
_FSAVERAGE5_TOTAL_VERTICES = 2 * _FSAVERAGE5_VERTICES_PER_HEMI


def aggregate_vertices_by_labels(
    vertex_timeseries: np.ndarray,
    vertex_labels: np.ndarray,
    n_rois: int,
) -> np.ndarray:
    """Average vertex timeseries within each integer parcel label.

    Pure-numpy aggregation: column ``r-1`` of the output is the mean over all
    vertices whose label equals ``r`` (for ``r`` in ``1..n_rois``). Label ``0``
    denotes unassigned vertices (e.g. the medial wall) and is excluded.

    Parameters
    ----------
    vertex_timeseries : np.ndarray, shape (T, V)
        Vertex-level timeseries.
    vertex_labels : np.ndarray, shape (V,)
        Integer parcel label per vertex, in ``0..n_rois`` (0 = unassigned).
    n_rois : int
        Number of parcels. Output has ``n_rois`` columns.

    Returns
    -------
    np.ndarray, shape (T, n_rois), dtype float64
        Mean timeseries per parcel. A parcel with no assigned vertices yields
        a column of ``nan`` (and a warning) -- never silently fabricated data.

    Raises
    ------
    ValueError
        If shapes are inconsistent or labels fall outside ``0..n_rois``.
    """
    if vertex_timeseries.ndim != 2:
        raise ValueError(
            f"vertex_timeseries must be 2-D (T, V); got "
            f"{vertex_timeseries.ndim}-D."
        )
    if vertex_labels.shape != (vertex_timeseries.shape[1],):
        raise ValueError(
            f"vertex_labels shape {vertex_labels.shape} must be "
            f"(V,) = ({vertex_timeseries.shape[1]},)."
        )
    lab = np.asarray(vertex_labels).astype(np.int64)
    if lab.min() < 0 or lab.max() > n_rois:
        raise ValueError(
            f"vertex_labels must lie in 0..n_rois ({n_rois}); got range "
            f"[{lab.min()}, {lab.max()}]."
        )

    n_timepoints = vertex_timeseries.shape[0]
    out = np.full((n_timepoints, n_rois), np.nan, dtype=np.float64)
    empty: list[int] = []
    for r in range(1, n_rois + 1):
        cols = lab == r
        if not cols.any():
            empty.append(r)
            continue
        out[:, r - 1] = vertex_timeseries[:, cols].mean(axis=1)

    if empty:
        warnings.warn(
            f"{len(empty)} of {n_rois} parcels had no fsaverage5 vertices "
            f"assigned (labels {empty[:8]}{'...' if len(empty) > 8 else ''}); "
            f"their columns are NaN. Consider a finer atlas resolution_mm.",
            stacklevel=2,
        )
    return out


@functools.lru_cache(maxsize=4)
def load_schaefer_fsaverage5_labels(
    n_rois: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Load the real Schaefer-2018 per-vertex labels for fsaverage5.

    Projects the volumetric Schaefer atlas onto the fsaverage5 surface using
    nilearn's nearest-neighbour ``vol_to_surf`` (sampling between the white and
    pial surfaces), giving an integer parcel label for each of the 20 484
    vertices. Cached per ``n_rois``.

    Parameters
    ----------
    n_rois : int
        Schaefer resolution (100, 200, 400, 600, 800, or 1000).

    Returns
    -------
    vertex_labels : tuple[int, ...], length 20484
        Integer parcel label per vertex (0 = medial wall / unassigned).
        Returned as a tuple so the result is hashable/cacheable; wrap with
        ``np.asarray`` at the call site.
    roi_names : tuple[str, ...], length n_rois
        Parcel name for label ``r`` at index ``r-1``.

    Raises
    ------
    ImportError
        If nilearn is not installed.
    RuntimeError
        If the atlas/surface cannot be fetched or projected (e.g. no network
        on first use and nothing cached).
    """
    try:
        from nilearn.datasets import (
            fetch_atlas_schaefer_2018,
            fetch_surf_fsaverage,
        )
        from nilearn.surface import vol_to_surf
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for Schaefer surface parcellation. "
            "Install with: pip install 'nilearn>=0.10.3' (the [foundation] "
            "extra)."
        ) from exc

    try:
        atlas = fetch_atlas_schaefer_2018(
            n_rois=n_rois, resolution_mm=1, verbose=0,
        )
        fsavg = fetch_surf_fsaverage(mesh="fsaverage5")

        def _project(pial: str, white: str) -> np.ndarray:
            surf = vol_to_surf(
                atlas["maps"], pial,
                interpolation="nearest", inner_mesh=white,
            )
            return np.rint(np.nan_to_num(surf)).astype(np.int64)

        lh = _project(fsavg["pial_left"], fsavg["white_left"])
        rh = _project(fsavg["pial_right"], fsavg["white_right"])
    except Exception as exc:  # noqa: BLE001 -- surface fetch/projection failure
        raise RuntimeError(
            "Failed to build the Schaefer fsaverage5 surface parcellation via "
            f"nilearn vol_to_surf: {exc}. This needs network access on first "
            "use (nilearn caches to ~/nilearn_data thereafter)."
        ) from exc

    vertex_labels = np.concatenate([lh, rh])
    if vertex_labels.shape[0] != _FSAVERAGE5_TOTAL_VERTICES:
        raise RuntimeError(
            f"Projected label vector has {vertex_labels.shape[0]} vertices, "
            f"expected {_FSAVERAGE5_TOTAL_VERTICES} (fsaverage5)."
        )
    vertex_labels = np.clip(vertex_labels, 0, n_rois)

    names_raw = [
        lbl.decode() if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas["labels"]
    ]
    roi_names = [n for n in names_raw if n != "Background"]
    if len(roi_names) != n_rois:
        # Older nilearn includes a Background entry; newer does not. Pad/trim
        # defensively so names align with labels 1..n_rois.
        roi_names = (roi_names + [f"ROI_{i}" for i in range(n_rois)])[:n_rois]

    return tuple(int(x) for x in vertex_labels), tuple(roi_names)


def parcellate_vertices_to_rois(
    vertex_timeseries: np.ndarray,
    n_rois: int = 100,
    atlas_name: str = "schaefer",
) -> tuple[np.ndarray, list[str]]:
    """Aggregate fsaverage5 vertex timeseries into Schaefer ROI signals.

    Loads the real Schaefer-2018 vertex-to-parcel assignment for fsaverage5
    (via :func:`load_schaefer_fsaverage5_labels`) and averages vertices within
    each parcel (:func:`aggregate_vertices_by_labels`).

    Parameters
    ----------
    vertex_timeseries : np.ndarray, shape (T, 20484)
        Vertex-level timeseries on fsaverage5.
    n_rois : int
        Number of parcels. Must be a valid Schaefer resolution
        (100, 200, 400, 600, 800, 1000).
    atlas_name : str
        Atlas to use. Currently only ``"schaefer"`` is supported.

    Returns
    -------
    roi_timeseries : np.ndarray, shape (T, n_rois)
        Mean timeseries per parcel (NaN for any parcel with no vertices).
    roi_names : list[str]
        Human-readable name for each ROI column.

    Raises
    ------
    ValueError
        If ``vertex_timeseries`` is not ``(T, 20484)`` or ``atlas_name`` is
        unsupported.
    ImportError
        If nilearn is not installed.
    RuntimeError
        If the Schaefer surface parcellation cannot be obtained.
    """
    if atlas_name != "schaefer":
        raise ValueError(
            f"Unsupported atlas: '{atlas_name}'. "
            "Currently only 'schaefer' is supported."
        )
    if vertex_timeseries.ndim != 2:
        raise ValueError(
            f"Expected 2-D array (T, V), got {vertex_timeseries.ndim}-D."
        )
    n_vertices = vertex_timeseries.shape[1]
    if n_vertices != _FSAVERAGE5_TOTAL_VERTICES:
        raise ValueError(
            f"Expected {_FSAVERAGE5_TOTAL_VERTICES} vertices "
            f"(fsaverage5), got {n_vertices}."
        )

    vertex_labels, roi_names = load_schaefer_fsaverage5_labels(n_rois)
    roi_timeseries = aggregate_vertices_by_labels(
        vertex_timeseries, np.asarray(vertex_labels), n_rois,
    )
    return roi_timeseries, list(roi_names)
