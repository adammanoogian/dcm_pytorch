"""TRIBE v2 fMRI brain encoding model extractor.

Wraps Meta's TRIBE v2 model to produce vertex-wise fMRI predictions
from multimodal stimuli, then parcellates to ROI timeseries suitable
for spectral DCM fitting.

TRIBE v2 outputs vertex-wise predictions at 1 Hz on fsaverage5
(~20 484 vertices).  The extractor provides two levels of latent
extraction:

1. **Vertex-to-ROI path** (default): parcellate vertex output into ROI
   timeseries via Schaefer atlas, then fit spectral DCM directly.
2. **Hook path** (optional): extract intermediate transformer layer
   activations via PyTorch forward hooks for richer dynamics.

References
----------
TRIBE v2: Meta AI (2026). Trimodal Brain Encoding v2.
    https://github.com/facebookresearch/tribev2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pyro_dcm.foundation.base_extractor import BaseExtractor
from pyro_dcm.foundation.parcellation import parcellate_vertices_to_rois

if TYPE_CHECKING:
    pass

# Number of fsaverage5 vertices (both hemispheres)
_FSAVERAGE5_TOTAL_VERTICES = 20484


class TRIBEExtractor(BaseExtractor):
    """Extractor for Meta TRIBE v2 fMRI brain encoding model.

    Wraps ``TribeModel`` from the ``tribev2`` package to produce
    vertex-wise fMRI predictions from multimodal stimuli (video,
    audio, text), then parcellates to ROI timeseries using a standard
    cortical atlas.

    Parameters
    ----------
    cache_folder : str
        Directory for caching downloaded model weights.
    n_rois : int
        Number of ROIs for parcellation (e.g. 100, 200, 400).
    device : torch.device | None
        Device for inference.  Defaults to CPU.

    Attributes
    ----------
    model_ : object | None
        Loaded ``TribeModel`` instance.  ``None`` until
        :meth:`load_model` is called.
    """

    def __init__(
        self,
        cache_folder: str = "./tribe_cache",
        n_rois: int = 100,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model_name="tribe_v2", device=device)
        self.cache_folder = cache_folder
        self.n_rois = n_rois
        self.model_: Any = None

    def load_model(
        self, checkpoint_path: str | None = None
    ) -> None:
        """Load TRIBE v2 model weights from HuggingFace hub.

        Parameters
        ----------
        checkpoint_path : str | None
            Ignored for TRIBE v2 (uses HuggingFace hub).  Kept for
            API compatibility with :class:`BaseExtractor`.

        Raises
        ------
        ImportError
            If the ``tribev2`` package is not installed.
        """
        try:
            from tribev2.demo_utils import TribeModel
        except ImportError as exc:
            raise ImportError(
                "Install tribev2: "
                "pip install 'tribev2 @ "
                "git+https://github.com/"
                "facebookresearch/tribev2.git'"
            ) from exc

        self.model_ = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=self.cache_folder,
        )

    def predict_vertex_timeseries(
        self,
        video_path: str | None = None,
        audio_path: str | None = None,
        events_df: Any = None,
    ) -> np.ndarray:
        """Predict vertex-wise fMRI timeseries from stimulus.

        Uses TRIBE v2 to generate fMRI predictions on fsaverage5
        surface (20 484 vertices) at 1 Hz temporal resolution.

        Parameters
        ----------
        video_path : str | None
            Path to stimulus video file.  Used to generate events
            dataframe if ``events_df`` is not provided.
        audio_path : str | None
            Path to stimulus audio file.  Reserved for future use.
        events_df : Any
            Pre-built events dataframe for TRIBE v2.  If ``None``,
            generated from ``video_path`` via
            ``model.get_events_dataframe()``.

        Returns
        -------
        np.ndarray, shape (T, 20484)
            Vertex-wise fMRI predictions on fsaverage5 at 1 Hz.

        Raises
        ------
        ValueError
            If ``model_`` is ``None`` (model not loaded).
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )

        if events_df is None:
            events_df = self.model_.get_events_dataframe(
                video_path=video_path,
            )

        preds, _segments = self.model_.predict(events=events_df)
        return np.asarray(preds)

    def extract_roi_timeseries(
        self,
        vertex_timeseries: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Parcellate vertex-level timeseries to ROI signals.

        Averages vertex predictions within each parcel defined by
        the Schaefer atlas on fsaverage5.

        Parameters
        ----------
        vertex_timeseries : np.ndarray, shape (T, 20484)
            Vertex-level timeseries on fsaverage5.

        Returns
        -------
        roi_timeseries : np.ndarray, shape (T, n_rois)
            Mean timeseries per parcel.
        roi_names : list[str]
            Human-readable name for each ROI column.
        """
        return parcellate_vertices_to_rois(
            vertex_timeseries, n_rois=self.n_rois
        )

    def extract_latents(
        self,
        input_data: torch.Tensor,
        layer_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract latent representations from TRIBE v2.

        When ``layer_names`` is ``None``, returns the default
        vertex-to-ROI path: predict vertex timeseries from input,
        parcellate to ROIs, and return as a torch tensor.

        When ``layer_names`` is provided, uses PyTorch forward hooks
        to extract intermediate transformer layer activations.

        Parameters
        ----------
        input_data : torch.Tensor
            For the default path, a dummy tensor (unused -- stimulus
            is provided via :meth:`predict_vertex_timeseries`).
            For hook extraction, the model input tensor.
        layer_names : list[str] | None
            Names of internal layers to hook.  If ``None``, returns
            the ROI timeseries from the vertex-to-ROI path.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer/output name to activation tensor.
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )

        if layer_names is not None:
            # Hook path: extract internal transformer activations
            from torch import nn

            if not isinstance(self.model_, nn.Module):
                raise TypeError(
                    "Hook extraction requires model_ to be an "
                    "nn.Module. TRIBE v2's TribeModel may not "
                    "support this path directly."
                )
            return self.extract_layer_activations(
                self.model_, input_data, layer_names
            )

        # Default path: vertex-to-ROI parcellation
        vertex_ts = self.predict_vertex_timeseries()
        roi_ts, _roi_names = self.extract_roi_timeseries(vertex_ts)
        return {"roi_timeseries": torch.from_numpy(roi_ts)}
