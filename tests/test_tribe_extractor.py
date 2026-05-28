"""Unit tests for TRIBEExtractor with mocked tribev2 dependency."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyro_dcm.foundation.tribe_extractor import TRIBEExtractor


class TestTRIBEExtractor:
    """Tests for TRIBEExtractor using mocked tribev2."""

    def test_tribe_extractor_load_model_import_error(self) -> None:
        """Verify load_model raises ImportError with install hint."""
        extractor = TRIBEExtractor()

        with patch.dict("sys.modules", {"tribev2": None, "tribev2.demo_utils": None}):
            with pytest.raises(ImportError, match="Install tribev2"):
                extractor.load_model()

    def test_tribe_extractor_predict_vertex_shape(self) -> None:
        """Predict returns (T, 20484) vertex timeseries from mocked model."""
        extractor = TRIBEExtractor()
        n_timepoints = 50
        n_vertices = 20484

        # Create mock TribeModel
        mock_model = MagicMock()
        mock_model.get_events_dataframe.return_value = MagicMock()
        mock_preds = np.random.randn(n_timepoints, n_vertices)
        mock_model.predict.return_value = (mock_preds, None)

        # Inject mock model directly (bypass load_model import)
        extractor.model_ = mock_model

        result = extractor.predict_vertex_timeseries(
            video_path="dummy.mp4"
        )

        assert result.shape == (n_timepoints, n_vertices)
        assert isinstance(result, np.ndarray)
        mock_model.get_events_dataframe.assert_called_once_with(
            video_path="dummy.mp4"
        )
        mock_model.predict.assert_called_once()

    def test_tribe_extractor_extract_roi_shape(self) -> None:
        """Extract ROI timeseries returns (T, n_rois) with ROI names."""
        n_rois = 100
        extractor = TRIBEExtractor(n_rois=n_rois)
        n_timepoints = 50
        n_vertices = 20484

        vertex_data = np.random.randn(n_timepoints, n_vertices)

        # Mock parcellate_vertices_to_rois to avoid nilearn dependency
        mock_roi_ts = np.random.randn(n_timepoints, n_rois)
        mock_names = [f"ROI_{i}" for i in range(n_rois)]

        with patch(
            "pyro_dcm.foundation.tribe_extractor"
            ".parcellate_vertices_to_rois",
            return_value=(mock_roi_ts, mock_names),
        ) as mock_parcellate:
            roi_ts, roi_names = extractor.extract_roi_timeseries(
                vertex_data
            )

            assert roi_ts.shape == (n_timepoints, n_rois)
            assert len(roi_names) == n_rois
            mock_parcellate.assert_called_once_with(
                vertex_data, n_rois=n_rois
            )

    def test_tribe_extractor_not_loaded_error(self) -> None:
        """Calling predict without load_model raises ValueError."""
        extractor = TRIBEExtractor()

        assert extractor.model_ is None

        with pytest.raises(ValueError, match="Model not loaded"):
            extractor.predict_vertex_timeseries(
                video_path="dummy.mp4"
            )
