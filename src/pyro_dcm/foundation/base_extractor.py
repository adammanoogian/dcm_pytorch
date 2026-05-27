"""Base class for foundation model latent extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    pass


class BaseExtractor(ABC):
    """Abstract base for foundation model feature extractors.

    Subclasses implement ``load_model`` and ``extract_latents`` for a
    specific foundation model (e.g. BrainLM, LaBraM, EEGNet).  The
    concrete ``extract_layer_activations`` method uses PyTorch forward
    hooks to capture intermediate activations from arbitrary named
    layers.

    Parameters
    ----------
    model_name : str
        Human-readable identifier for the foundation model.
    device : torch.device | None
        Device for inference. Defaults to CPU.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cpu")

    @abstractmethod
    def load_model(
        self, checkpoint_path: str | None = None
    ) -> None:
        """Load model weights from checkpoint or hub.

        Parameters
        ----------
        checkpoint_path : str | None
            Path to local checkpoint file.  If ``None``, implementations
            should attempt to fetch from a model hub.
        """

    @abstractmethod
    def extract_latents(
        self,
        input_data: torch.Tensor,
        layer_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract latent representations from the loaded model.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor appropriate for the foundation model.
        layer_names : list[str] | None
            Subset of layer names to extract.  If ``None``,
            implementations should return a default set.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer name to activation tensor.
        """

    def extract_layer_activations(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer_names: list[str],
    ) -> dict[str, torch.Tensor]:
        """Extract activations from named layers via forward hooks.

        Registers temporary forward hooks on the requested named
        modules, runs a single forward pass with ``torch.no_grad()``,
        then removes all hooks.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to extract from.
        input_tensor : torch.Tensor
            Input for the forward pass.
        layer_names : list[str]
            Dot-separated names matching ``model.named_modules()``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer name to the layer's output tensor.

        Raises
        ------
        ValueError
            If a requested layer name is not found in the model.
        """
        activations: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHook] = []

        # Validate layer names and register hooks
        module_dict = dict(model.named_modules())
        for name in layer_names:
            if name not in module_dict:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(module_dict.keys())}"
                )

            def _make_hook(
                layer_name: str,
            ) -> callable:
                def _hook(
                    _module: nn.Module,
                    _input: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                ) -> None:
                    activations[layer_name] = output.detach()

                return _hook

            handle = module_dict[name].register_forward_hook(
                _make_hook(name)
            )
            handles.append(handle)

        try:
            with torch.no_grad():
                model(input_tensor)
        finally:
            for handle in handles:
                handle.remove()

        return activations
