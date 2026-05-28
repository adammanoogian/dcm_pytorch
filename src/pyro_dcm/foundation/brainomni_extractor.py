"""BrainOmni EEG+MEG foundation model extractor for DCM interpretability.

Wraps the BrainOmni model (NeurIPS 2025) for latent activation
extraction via PyTorch forward hooks.  BrainOmni is a unified
foundation model pretrained on ~2,000 hours of EEG and ~660 hours of
MEG data using a BrainTokenizer architecture.  Unlike LaBraM, BrainOmni
does not expose a ``return_features`` API, so extraction relies
exclusively on forward hooks registered on transformer encoder blocks.

References
----------
Chen et al. (2025). BrainOmni: A Unified Brain Foundation Model for
EEG and MEG. NeurIPS 2025. arXiv:2505.18185.

Repository: https://github.com/OpenTSLab/BrainOmni
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from pyro_dcm.foundation.base_extractor import BaseExtractor


class BrainOmniExtractor(BaseExtractor):
    """Extract latent activations from pretrained BrainOmni via hooks.

    BrainOmni handles both EEG and MEG modalities within a single
    architecture.  Since BrainOmni does not provide a built-in
    feature-extraction API, this extractor uses PyTorch forward hooks
    (via :meth:`BaseExtractor.extract_layer_activations`) to capture
    intermediate transformer layer outputs during inference.

    The extraction pipeline is designed defensively: BrainOmni's API
    is less mature than braindecode's, and the model requires cloning
    the repository and downloading HuggingFace checkpoints.  All import
    paths are guarded with clear error messages.

    Parameters
    ----------
    modality : str
        Input modality: ``"eeg"`` or ``"meg"``.  Used for model
        configuration.  Default ``"eeg"``.
    device : torch.device or None
        Inference device.  Defaults to CPU.

    Attributes
    ----------
    model_ : nn.Module or None
        The loaded BrainOmni model.  ``None`` until ``load_model``
        is called.
    """

    def __init__(
        self,
        modality: str = "eeg",
        device: torch.device | None = None,
    ) -> None:
        if modality not in ("eeg", "meg"):
            raise ValueError(
                f"modality must be 'eeg' or 'meg', got '{modality}'"
            )
        super().__init__("brainomni", device)
        self.modality = modality
        self.model_: nn.Module | None = None

    def load_model(
        self, checkpoint_path: str | None = None
    ) -> None:
        """Load pretrained BrainOmni weights.

        BrainOmni requires cloning the GitHub repository and
        downloading HuggingFace checkpoints.  If ``checkpoint_path``
        is provided, loads weights directly from the file.  Otherwise
        attempts to download via ``huggingface_hub.hf_hub_download``.

        Parameters
        ----------
        checkpoint_path : str or None
            Path to a local checkpoint file.  ``None`` triggers
            HuggingFace download.

        Raises
        ------
        ImportError
            If BrainOmni or huggingface_hub is not installed.
        """
        try:
            from brainomni.model import BrainOmniModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "BrainOmni is required for BrainOmniExtractor. "
                "Clone and install from: "
                "https://github.com/OpenTSLab/BrainOmni"
            ) from exc

        if checkpoint_path is not None:
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            model = BrainOmniModel(modality=self.modality)
            model.load_state_dict(state_dict)
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required for automatic "
                    "checkpoint download. Install with: "
                    "pip install huggingface_hub>=0.20"
                ) from exc

            ckpt_path = hf_hub_download(
                repo_id="OpenTSLab/BrainOmni",
                filename="brainomni_base.ckpt",
            )
            model = BrainOmniModel(modality=self.modality)
            state_dict = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model_ = model

    def extract_latents(
        self,
        input_data: torch.Tensor,
        layer_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract latent activations from BrainOmni via forward hooks.

        Registers PyTorch forward hooks on specified transformer layers
        and captures their outputs during a single forward pass.  If
        ``layer_names`` is ``None``, hooks all encoder blocks found via
        ``model.named_modules()``.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor appropriate for BrainOmni (shape depends on
            modality and preprocessing).
        layer_names : list[str] or None
            Dot-separated names of layers to hook (e.g.
            ``["encoder.blocks.3", "encoder.blocks.5"]``).  ``None``
            auto-discovers all modules whose name contains
            ``"encoder"`` and ``"block"``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer name to activation tensor.

        Raises
        ------
        ValueError
            If ``load_model`` has not been called.
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )
        input_data = input_data.to(self.device)

        if layer_names is None:
            layer_names = [
                name
                for name, _ in self.model_.named_modules()
                if "encoder" in name and "block" in name
            ]
            if not layer_names:
                # Fallback: hook all non-trivial named children
                layer_names = [
                    name
                    for name, mod in self.model_.named_modules()
                    if name and not list(mod.children())
                ]

        return self.extract_layer_activations(
            self.model_, input_data, layer_names
        )

    def reduce_to_dcm_space(
        self,
        activations: dict[str, torch.Tensor],
        layer_name: str,
        n_components: int = 4,
    ) -> tuple[np.ndarray, Any]:
        """Reduce activations from a single layer to DCM dimensions.

        Selects the activation tensor from the specified layer,
        reshapes to 2-D ``(samples, features)``, and applies PCA.

        Parameters
        ----------
        activations : dict[str, torch.Tensor]
            Output from ``extract_latents``.
        layer_name : str
            Key into ``activations`` to select the layer for reduction.
        n_components : int
            Number of PCA components (= DCM regions).  Default 4.

        Returns
        -------
        reduced : np.ndarray
            PCA-reduced representation.  If the input activation has
            shape ``(batch, seq_len, hidden_dim)``, output is
            ``(batch, seq_len, n_components)``.  For 2-D activations
            ``(batch, hidden_dim)`` the output is
            ``(batch, n_components)``.
        pca_model : sklearn.decomposition.PCA
            Fitted PCA model.

        Raises
        ------
        KeyError
            If ``layer_name`` not in ``activations``.
        ImportError
            If ``scikit-learn`` is not installed.
        """
        if layer_name not in activations:
            available = list(activations.keys())
            raise KeyError(
                f"Layer '{layer_name}' not in activations. "
                f"Available: {available}"
            )

        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for PCA reduction. "
                "Install with: pip install scikit-learn"
            ) from exc

        tensor = activations[layer_name]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        original_shape = tensor.shape

        if tensor.ndim == 3:
            batch, seq_len, hidden_dim = original_shape
            flat = tensor.reshape(batch * seq_len, hidden_dim)
        elif tensor.ndim == 2:
            flat = tensor
        else:
            raise ValueError(
                f"Expected 2-D or 3-D activation, got shape "
                f"{original_shape}"
            )

        pca = PCA(n_components=n_components)
        reduced_flat = pca.fit_transform(flat)

        if tensor.ndim == 3:
            reduced = reduced_flat.reshape(
                batch, seq_len, n_components
            )
        else:
            reduced = reduced_flat

        return reduced, pca
