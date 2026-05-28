"""Unit tests for LaBraM and BrainOmni M/EEG extractors.

All tests mock external dependencies (braindecode, BrainOmni) so that
no pretrained model downloads or GPU access are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from pyro_dcm.foundation.labram_extractor import LaBraMExtractor
from pyro_dcm.foundation.brainomni_extractor import BrainOmniExtractor


# -------------------------------------------------------------------
# LaBraM tests
# -------------------------------------------------------------------


class TestLaBraMExtractorContract:
    """Verify LaBraMExtractor is a concrete BaseExtractor subclass."""

    def test_instantiates(self) -> None:
        """LaBraMExtractor can be instantiated (not abstract)."""
        ext = LaBraMExtractor()
        assert ext.model_name == "labram"

    def test_has_required_methods(self) -> None:
        """LaBraMExtractor exposes load_model, extract_features, etc."""
        ext = LaBraMExtractor()
        assert hasattr(ext, "load_model")
        assert hasattr(ext, "extract_features")
        assert hasattr(ext, "extract_latents")
        assert hasattr(ext, "reduce_to_dcm_space")

    def test_chs_info_stored(self) -> None:
        """Channel info is stored for future use."""
        chs = [{"ch_name": "Fp1"}, {"ch_name": "Fp2"}]
        ext = LaBraMExtractor(chs_info=chs)
        assert ext.chs_info == chs


class TestLaBraMExtractFeatures:
    """Test extract_features with mocked braindecode model."""

    def test_shape(self) -> None:
        """extract_features returns correct shape dict."""
        ext = LaBraMExtractor(
            n_times=1600,
            n_chans=64,
            embed_dim=200,
            patch_size=200,
        )
        # n_patches = n_times // patch_size = 1600 // 200 = 8
        n_patches = 8
        embed_dim = 200

        # Mock model that returns features dict
        mock_model = MagicMock(spec=nn.Module)
        mock_model.return_value = {
            "features": torch.randn(1, n_patches, embed_dim),
            "cls_token": torch.randn(1, embed_dim),
        }
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        ext.model_ = mock_model

        eeg_input = torch.randn(1, 64, 1600)
        result = ext.extract_features(eeg_input)

        assert "features" in result
        assert "cls_token" in result
        assert result["features"].shape == (1, n_patches, embed_dim)
        assert result["cls_token"].shape == (1, embed_dim)


class TestLaBraMReduceToDCMSpace:
    """Test PCA reduction of patch embeddings."""

    def test_output_shape(self) -> None:
        """reduce_to_dcm_space produces correct shape."""
        ext = LaBraMExtractor()
        features = torch.randn(2, 8, 200)
        reduced, pca = ext.reduce_to_dcm_space(
            features, n_components=4
        )
        assert reduced.shape == (2, 8, 4)
        assert hasattr(pca, "explained_variance_ratio_")

    def test_accepts_numpy(self) -> None:
        """reduce_to_dcm_space works with numpy input."""
        ext = LaBraMExtractor()
        features = np.random.randn(3, 8, 200)
        reduced, pca = ext.reduce_to_dcm_space(
            features, n_components=2
        )
        assert reduced.shape == (3, 8, 2)


class TestLaBraMNotLoadedError:
    """Test error when model is not loaded."""

    def test_extract_features_raises(self) -> None:
        """extract_features raises ValueError without load_model."""
        ext = LaBraMExtractor()
        with pytest.raises(ValueError, match="Model not loaded"):
            ext.extract_features(torch.randn(1, 64, 1600))

    def test_extract_latents_raises(self) -> None:
        """extract_latents raises ValueError without load_model."""
        ext = LaBraMExtractor()
        with pytest.raises(ValueError, match="Model not loaded"):
            ext.extract_latents(torch.randn(1, 64, 1600))


# -------------------------------------------------------------------
# BrainOmni tests
# -------------------------------------------------------------------


class TestBrainOmniExtractorContract:
    """Verify BrainOmniExtractor is a concrete BaseExtractor subclass."""

    def test_instantiates(self) -> None:
        """BrainOmniExtractor can be instantiated (not abstract)."""
        ext = BrainOmniExtractor()
        assert ext.model_name == "brainomni"

    def test_has_required_methods(self) -> None:
        """BrainOmniExtractor exposes the expected API."""
        ext = BrainOmniExtractor()
        assert hasattr(ext, "load_model")
        assert hasattr(ext, "extract_latents")
        assert hasattr(ext, "reduce_to_dcm_space")

    def test_modality_stored(self) -> None:
        """Modality parameter is stored."""
        ext = BrainOmniExtractor(modality="meg")
        assert ext.modality == "meg"

    def test_invalid_modality_raises(self) -> None:
        """Invalid modality raises ValueError."""
        with pytest.raises(ValueError, match="modality must be"):
            BrainOmniExtractor(modality="ecog")


class TestBrainOmniExtractLatents:
    """Test extract_latents with a mock model using hooks."""

    def test_uses_hooks(self) -> None:
        """extract_latents captures activations via forward hooks."""
        ext = BrainOmniExtractor()

        # Build a simple nn.Sequential as mock BrainOmni model
        mock_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        mock_model.eval()
        ext.model_ = mock_model

        input_data = torch.randn(2, 10)
        activations = ext.extract_latents(
            input_data, layer_names=["0", "2"]
        )
        assert "0" in activations
        assert "2" in activations
        assert activations["0"].shape == (2, 20)
        assert activations["2"].shape == (2, 5)

    def test_not_loaded_raises(self) -> None:
        """extract_latents raises ValueError without load_model."""
        ext = BrainOmniExtractor()
        with pytest.raises(ValueError, match="Model not loaded"):
            ext.extract_latents(torch.randn(1, 10))


class TestBrainOmniReduceToDCMSpace:
    """Test PCA reduction of BrainOmni activations."""

    def test_3d_activation(self) -> None:
        """reduce_to_dcm_space handles 3-D activations."""
        ext = BrainOmniExtractor()
        activations = {
            "encoder.block.3": torch.randn(4, 16, 128),
        }
        reduced, pca = ext.reduce_to_dcm_space(
            activations, "encoder.block.3", n_components=4
        )
        assert reduced.shape == (4, 16, 4)

    def test_2d_activation(self) -> None:
        """reduce_to_dcm_space handles 2-D activations."""
        ext = BrainOmniExtractor()
        activations = {
            "pool": torch.randn(8, 64),
        }
        reduced, pca = ext.reduce_to_dcm_space(
            activations, "pool", n_components=3
        )
        assert reduced.shape == (8, 3)

    def test_missing_layer_raises(self) -> None:
        """reduce_to_dcm_space raises KeyError for missing layer."""
        ext = BrainOmniExtractor()
        activations = {"layer_a": torch.randn(2, 64)}
        with pytest.raises(KeyError, match="layer_b"):
            ext.reduce_to_dcm_space(activations, "layer_b")
