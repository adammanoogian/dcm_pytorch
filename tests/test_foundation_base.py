"""Tests for foundation model extractor base infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from pyro_dcm.foundation.base_extractor import BaseExtractor
from pyro_dcm.foundation.parcellation import (
    _FSAVERAGE5_TOTAL_VERTICES,
    parcellate_vertices_to_rois,
)


class TestBaseExtractor:
    """Tests for BaseExtractor abstract base class."""

    def test_base_extractor_is_abstract(self) -> None:
        """BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseExtractor(model_name="test")  # type: ignore[abstract]

    def test_extract_layer_activations_hook_pattern(self) -> None:
        """Forward hooks capture activations from requested layers."""
        model = nn.Sequential()
        model.add_module("layer_0", nn.Linear(4, 8))
        model.add_module("layer_1", nn.Linear(8, 16))
        model.add_module("layer_2", nn.Linear(16, 2))

        # Concrete subclass for testing
        class _DummyExtractor(BaseExtractor):
            def load_model(
                self, checkpoint_path: str | None = None
            ) -> None:
                pass

            def extract_latents(
                self,
                input_data: torch.Tensor,
                layer_names: list[str] | None = None,
            ) -> dict[str, torch.Tensor]:
                return {}

        extractor = _DummyExtractor(model_name="test")
        x = torch.randn(2, 4)
        result = extractor.extract_layer_activations(
            model, x, ["layer_0", "layer_2"]
        )

        assert set(result.keys()) == {"layer_0", "layer_2"}
        assert result["layer_0"].shape == (2, 8)
        assert result["layer_2"].shape == (2, 2)

    def test_extract_layer_activations_removes_hooks(self) -> None:
        """All forward hooks are removed after extraction."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 8))
        model.add_module("fc2", nn.Linear(8, 2))

        class _DummyExtractor(BaseExtractor):
            def load_model(
                self, checkpoint_path: str | None = None
            ) -> None:
                pass

            def extract_latents(
                self,
                input_data: torch.Tensor,
                layer_names: list[str] | None = None,
            ) -> dict[str, torch.Tensor]:
                return {}

        extractor = _DummyExtractor(model_name="test")
        x = torch.randn(1, 4)

        # Count hooks before
        hooks_before = sum(
            len(m._forward_hooks) for m in model.modules()
        )

        extractor.extract_layer_activations(model, x, ["fc1", "fc2"])

        # Count hooks after
        hooks_after = sum(
            len(m._forward_hooks) for m in model.modules()
        )

        assert hooks_after == hooks_before


class TestParcellation:
    """Tests for vertex-to-ROI parcellation."""

    def test_parcellate_vertices_mock(self) -> None:
        """Parcellation reduces (T, 20484) to (T, n_rois)."""
        n_rois = 100
        mock_atlas = {
            "labels": [b"Background"]
            + [f"Network_{i}".encode() for i in range(1, n_rois + 1)],
            "maps": "dummy_path",
        }

        # Mock nilearn.datasets so the function-level import resolves
        mock_datasets = MagicMock()
        mock_datasets.fetch_atlas_schaefer_2018.return_value = mock_atlas
        mock_nilearn = MagicMock()
        mock_nilearn.datasets = mock_datasets

        with patch.dict(
            "sys.modules",
            {
                "nilearn": mock_nilearn,
                "nilearn.datasets": mock_datasets,
            },
        ):
            rng = np.random.default_rng(42)
            vertex_ts = rng.standard_normal(
                (10, _FSAVERAGE5_TOTAL_VERTICES)
            )

            roi_ts, roi_names = parcellate_vertices_to_rois(
                vertex_ts, n_rois=n_rois
            )

        assert roi_ts.shape == (10, n_rois)
        assert len(roi_names) == n_rois

    def test_parcellate_vertices_wrong_shape(self) -> None:
        """ValueError raised when vertex count is not 20484."""
        wrong_data = np.zeros((10, 100))
        with pytest.raises(ValueError, match="20484"):
            parcellate_vertices_to_rois(wrong_data)
