"""LaBraM EEG foundation model extractor for DCM interpretability.

Wraps braindecode's ``Labram`` model (ICLR 2024 spotlight) for
patch-level temporal feature extraction and PCA reduction to
DCM-compatible state spaces.  LaBraM segments EEG into fixed-length
patches and produces a 200-dim embedding per patch via a 12-layer
vision transformer, yielding a temporal sequence of latent states
suitable for spectral or latent-circuit DCM.

References
----------
Jiang et al. (2024). Large Brain Model for Learning Generic
Representations with Tremendous EEG Data in BCI. ICLR 2024 Spotlight.

Braindecode Labram API:
https://braindecode.org/dev/generated/braindecode.models.Labram.html
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from pyro_dcm.foundation.base_extractor import BaseExtractor


class LaBraMExtractor(BaseExtractor):
    """Extract patch-level temporal features from pretrained LaBraM.

    LaBraM (Large Brain Model) is a vision-transformer-based EEG model
    pretrained on ~2,500 hours of EEG data.  It segments the input EEG
    into fixed-length patches (default 200 samples = 1 s at 200 Hz),
    tokenises each patch via a neural tokeniser, and produces per-patch
    embeddings of size ``embed_dim``.

    The extractor provides two paths for latent extraction:

    1. **Built-in feature API** (``extract_features``): calls the model
       with ``return_features=True`` to obtain patch-level embeddings
       and the [CLS] token in a single pass.
    2. **Hook-based API** (``extract_latents`` with ``layer_names``):
       registers forward hooks on specific transformer blocks via the
       inherited ``extract_layer_activations`` method.

    After extraction, ``reduce_to_dcm_space`` applies PCA to map from
    ``embed_dim`` to a small number of components suitable for DCM
    fitting.

    Parameters
    ----------
    n_times : int
        Number of time samples per EEG segment (default 1600 = 8 s at
        200 Hz).
    n_chans : int
        Number of EEG channels (default 64).
    n_outputs : int
        Number of classification outputs (ignored during feature
        extraction; default 4).
    sfreq : float
        Sampling frequency in Hz (default 200.0).
    patch_size : int
        Samples per patch (default 200 = 1 s at 200 Hz).
    embed_dim : int
        Transformer embedding dimension (default 200).
    n_layers : int
        Number of transformer encoder layers (default 12).
    n_heads : int
        Number of attention heads (default 10).
    device : torch.device or None
        Inference device.  Defaults to CPU.
    chs_info : list[dict] or None
        Channel information dicts (e.g. from ``epochs.info['chs']``).
        Required for correct spatial embeddings when channel ordering
        differs from pretraining montage (see Pitfall 5 in
        24-RESEARCH.md).  Currently stored for future use.

    Attributes
    ----------
    model_ : nn.Module or None
        The loaded LaBraM model.  ``None`` until ``load_model`` is
        called.
    """

    def __init__(
        self,
        n_times: int = 1600,
        n_chans: int = 64,
        n_outputs: int = 4,
        sfreq: float = 200.0,
        patch_size: int = 200,
        embed_dim: int = 200,
        n_layers: int = 12,
        n_heads: int = 10,
        device: torch.device | None = None,
        chs_info: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__("labram", device)
        self.n_times = n_times
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.sfreq = sfreq
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.chs_info = chs_info
        self.model_: nn.Module | None = None

    def load_model(
        self, checkpoint_path: str | None = None
    ) -> None:
        """Load pretrained LaBraM weights.

        If ``checkpoint_path`` is ``None``, downloads the pretrained
        model via ``Labram.from_pretrained("braindecode/labram-pretrained")``.
        Otherwise instantiates a ``Labram`` with the stored
        hyperparameters and loads weights from the given path.

        Parameters
        ----------
        checkpoint_path : str or None
            Path to a local checkpoint file.  ``None`` triggers
            HuggingFace download.

        Raises
        ------
        ImportError
            If ``braindecode`` is not installed (>= 1.3.0 required).
        """
        try:
            from braindecode.models import Labram
        except ImportError as exc:
            raise ImportError(
                "braindecode is required for LaBraMExtractor. "
                "Install with: pip install braindecode>=1.3.0"
            ) from exc

        if checkpoint_path is None:
            self.model_ = Labram.from_pretrained(
                "braindecode/labram-pretrained"
            )
        else:
            model = Labram(
                n_times=self.n_times,
                n_chans=self.n_chans,
                n_outputs=self.n_outputs,
                sfreq=self.sfreq,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                num_layers=self.n_layers,
                num_heads=self.n_heads,
            )
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            self.model_ = model

        self.model_.to(self.device)
        self.model_.eval()

    def extract_features(
        self,
        eeg_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract patch-level features using LaBraM built-in API.

        Calls the model with ``return_features=True`` to obtain
        patch-level embeddings and the [CLS] token.

        Parameters
        ----------
        eeg_input : torch.Tensor, shape (batch, n_chans, n_times)
            Raw EEG segments.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"features"`` : shape ``(batch, n_patches, embed_dim)``
            ``"cls_token"`` : shape ``(batch, embed_dim)``

        Raises
        ------
        ValueError
            If ``load_model`` has not been called.
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )
        eeg_input = eeg_input.to(self.device)
        with torch.no_grad():
            out = self.model_(eeg_input, return_features=True)
        return out

    def extract_latents(
        self,
        input_data: torch.Tensor,
        layer_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract latent representations from LaBraM.

        When ``layer_names`` is ``None``, delegates to
        ``extract_features`` (built-in return_features API).  When
        ``layer_names`` is provided, uses PyTorch forward hooks via
        the inherited ``extract_layer_activations`` method to capture
        activations from specific transformer blocks.

        Parameters
        ----------
        input_data : torch.Tensor, shape (batch, n_chans, n_times)
            Raw EEG segments.
        layer_names : list[str] or None
            Dot-separated layer names to hook (e.g.
            ``["encoder.blocks.5"]``).  ``None`` uses the built-in
            feature extraction path.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer/feature name to activation tensor.

        Raises
        ------
        ValueError
            If ``load_model`` has not been called.
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )
        if layer_names is None:
            return self.extract_features(input_data)

        input_data = input_data.to(self.device)
        return self.extract_layer_activations(
            self.model_, input_data, layer_names
        )

    def reduce_to_dcm_space(
        self,
        features: torch.Tensor | np.ndarray,
        n_components: int = 4,
    ) -> tuple[np.ndarray, Any]:
        """Reduce patch embeddings to DCM-compatible dimensions via PCA.

        Reshapes ``(batch, n_patches, embed_dim)`` to
        ``(batch * n_patches, embed_dim)``, fits PCA, then reshapes
        back to ``(batch, n_patches, n_components)``.

        Parameters
        ----------
        features : torch.Tensor or np.ndarray
            Patch-level features, shape ``(batch, n_patches, embed_dim)``.
        n_components : int
            Number of PCA components (= DCM regions).  Default 4.

        Returns
        -------
        reduced : np.ndarray, shape (batch, n_patches, n_components)
            PCA-reduced latent dynamics.
        pca_model : sklearn.decomposition.PCA
            Fitted PCA model (for variance explained, inverse transform).

        Raises
        ------
        ImportError
            If ``scikit-learn`` is not installed.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for PCA reduction. "
                "Install with: pip install scikit-learn"
            ) from exc

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        batch, n_patches, embed_dim = features.shape
        flat = features.reshape(batch * n_patches, embed_dim)

        pca = PCA(n_components=n_components)
        reduced_flat = pca.fit_transform(flat)
        reduced = reduced_flat.reshape(batch, n_patches, n_components)
        return reduced, pca
