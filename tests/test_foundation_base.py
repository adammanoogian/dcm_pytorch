"""Tests for foundation model extractor base infrastructure."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from pyro_dcm.foundation.base_extractor import BaseExtractor
from pyro_dcm.foundation.parcellation import (
    _FSAVERAGE5_TOTAL_VERTICES,
    aggregate_vertices_by_labels,
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


class TestAggregateVerticesByLabels:
    """Tests for the real label-based aggregation (pure numpy, no nilearn)."""

    def test_averages_within_label(self) -> None:
        """Each ROI column is the mean of its assigned vertices."""
        # 5 vertices: labels [1, 2, 1, 2, 0]; vertex 4 (label 0) excluded.
        ts = np.array(
            [[1.0, 10.0, 3.0, 20.0, 999.0],
             [2.0, 30.0, 4.0, 50.0, 999.0]],
        )
        labels = np.array([1, 2, 1, 2, 0])
        out = aggregate_vertices_by_labels(ts, labels, n_rois=2)
        assert out.shape == (2, 2)
        # ROI 1 = mean(vertices 0,2); ROI 2 = mean(vertices 1,3).
        np.testing.assert_allclose(out[:, 0], [(1 + 3) / 2, (2 + 4) / 2])
        np.testing.assert_allclose(out[:, 1], [(10 + 20) / 2, (30 + 50) / 2])

    def test_respects_labels_not_contiguity(self) -> None:
        """Non-contiguous labels are honoured (regression vs block placeholder).

        The old placeholder averaged contiguous vertex blocks; the correct
        behaviour follows the atlas labels even when interleaved.
        """
        ts = np.arange(6, dtype=np.float64).reshape(1, 6)  # [0,1,2,3,4,5]
        labels = np.array([1, 2, 1, 2, 1, 2])  # interleaved, NOT blocks
        out = aggregate_vertices_by_labels(ts, labels, n_rois=2)
        # ROI 1 = mean(0,2,4)=2 ; ROI 2 = mean(1,3,5)=3. A contiguous-block
        # split would give 1 and 4 -- assert we do NOT get that.
        np.testing.assert_allclose(out[0], [2.0, 3.0])

    def test_empty_parcel_is_nan_with_warning(self) -> None:
        """A parcel with no vertices yields NaN (never fabricated) + warns."""
        ts = np.ones((2, 3))
        labels = np.array([1, 1, 1])  # label 2 has no vertices
        with pytest.warns(UserWarning, match="no fsaverage5 vertices"):
            out = aggregate_vertices_by_labels(ts, labels, n_rois=2)
        assert np.isnan(out[:, 1]).all()
        assert not np.isnan(out[:, 0]).any()

    def test_label_out_of_range_raises(self) -> None:
        """Labels outside 0..n_rois are an error."""
        with pytest.raises(ValueError, match="0..n_rois"):
            aggregate_vertices_by_labels(
                np.ones((1, 3)), np.array([1, 2, 5]), n_rois=2,
            )


class TestParcellation:
    """Tests for vertex-to-ROI parcellation (loader patched; nilearn absent)."""

    def test_parcellate_uses_real_labels(self) -> None:
        """parcellate_vertices_to_rois aggregates by the loaded atlas labels."""
        n_rois = 4
        rng = np.random.default_rng(0)
        # Synthetic but realistic per-vertex label vector + names.
        labels = rng.integers(0, n_rois + 1, size=_FSAVERAGE5_TOTAL_VERTICES)
        names = tuple(f"Net_{i}" for i in range(1, n_rois + 1))
        with patch(
            "pyro_dcm.foundation.parcellation.load_schaefer_fsaverage5_labels",
            return_value=(tuple(int(x) for x in labels), names),
        ):
            vertex_ts = rng.standard_normal((10, _FSAVERAGE5_TOTAL_VERTICES))
            roi_ts, roi_names = parcellate_vertices_to_rois(
                vertex_ts, n_rois=n_rois,
            )
        assert roi_ts.shape == (10, n_rois)
        assert roi_names == list(names)
        # Column 0 must equal the mean of vertices labelled 1 (real math).
        expected = vertex_ts[:, labels == 1].mean(axis=1)
        np.testing.assert_allclose(roi_ts[:, 0], expected)

    def test_parcellate_no_silent_fallback_on_missing_nilearn(self) -> None:
        """If labels cannot be loaded, the error propagates (no block fake)."""
        with patch(
            "pyro_dcm.foundation.parcellation.load_schaefer_fsaverage5_labels",
            side_effect=ImportError("nilearn required"),
        ):
            with pytest.raises(ImportError, match="nilearn"):
                parcellate_vertices_to_rois(
                    np.zeros((4, _FSAVERAGE5_TOTAL_VERTICES)), n_rois=4,
                )

    def test_parcellate_vertices_wrong_shape(self) -> None:
        """ValueError raised when vertex count is not 20484."""
        wrong_data = np.zeros((10, 100))
        with pytest.raises(ValueError, match="20484"):
            parcellate_vertices_to_rois(wrong_data)

    @pytest.mark.slow
    def test_real_schaefer_labels_nonuniform(self) -> None:
        """Real nilearn loader gives a genuine (non-uniform) parcellation.

        Skips when nilearn is absent. Downloads the atlas + fsaverage5 meshes on
        first run (nilearn caches). The contiguous-block placeholder produced
        exactly-uniform parcel sizes; the real atlas does not -- this is the
        regression guard.
        """
        pytest.importorskip("nilearn")
        from pyro_dcm.foundation.parcellation import (
            load_schaefer_fsaverage5_labels,
        )

        labels, names = load_schaefer_fsaverage5_labels(100)
        labels_arr = np.asarray(labels)
        assert labels_arr.shape == (_FSAVERAGE5_TOTAL_VERTICES,)
        assert len(names) == 100
        counts = np.bincount(
            labels_arr[labels_arr > 0], minlength=101,
        )[1:]
        # A real surface atlas has varying parcel sizes; the block fake did not.
        assert counts.std() > 1.0
