"""Round-trip tests for .mat export/import and comparison utilities.

Tests verify that:
1. Task DCM export produces correct SPM12-compatible DCM struct
2. Spectral DCM export produces CSD-mode struct with induced=1
3. rDCM export produces tapas-compatible struct
4. Stimulus upsampling handles microtime resolution and SPM padding
5. Hybrid error metric works correctly for large and near-zero values
6. Model ranking comparison detects agreement and disagreement

All tests use tmp_path for .mat file I/O. No MATLAB dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from validation.compare_results import (
    compare_model_ranking,
    compare_posterior_means,
)
from validation.export_to_mat import (
    export_rdcm_for_tapas,
    export_spectral_dcm_for_spm,
    export_task_dcm_for_spm,
    upsample_stimulus,
)


class TestTaskDCMExportRoundtrip:
    """Test task DCM .mat export produces correct struct format."""

    def test_task_dcm_export_roundtrip(self, tmp_path):
        """Export and reload task DCM struct, verify all fields."""
        # Generate synthetic data
        v, N, M = 100, 3, 1
        bold = np.random.randn(v, N).astype(np.float64)
        TR = 2.0

        # Stimulus at microtime resolution
        stim_tr = np.zeros((v, M), dtype=np.float64)
        stim_tr[10:30, 0] = 1.0
        stim_tr[50:70, 0] = 1.0
        stimulus, u_dt = upsample_stimulus(stim_tr, TR)

        a_mask = np.ones((N, N), dtype=np.float64)
        c_mask = np.zeros((N, M), dtype=np.float64)
        c_mask[0, 0] = 1.0

        # Export
        mat_path = str(tmp_path / "task_dcm.mat")
        export_task_dcm_for_spm(bold, stimulus, a_mask, c_mask, TR, u_dt, mat_path)

        # Load and verify
        data = scipy.io.loadmat(mat_path, squeeze_me=False)
        DCM = data["DCM"]

        # Y fields
        Y_y = DCM["Y"][0, 0]["y"][0, 0]
        assert Y_y.shape == (v, N), f"Y.y shape: {Y_y.shape}"
        np.testing.assert_array_almost_equal(Y_y, bold)

        Y_dt = DCM["Y"][0, 0]["dt"][0, 0]
        assert Y_dt.shape == (1, 1), "Y.dt should be 2D scalar"
        assert Y_dt[0, 0] == TR

        Y_X0 = DCM["Y"][0, 0]["X0"][0, 0]
        assert Y_X0.shape == (v, 1), f"Y.X0 shape: {Y_X0.shape}"

        Y_name = DCM["Y"][0, 0]["name"][0, 0]
        assert Y_name.dtype == object, "Y.name should be object array"

        # U fields
        U_u = DCM["U"][0, 0]["u"][0, 0]
        assert U_u.shape == stimulus.shape, f"U.u shape: {U_u.shape}"

        U_dt = DCM["U"][0, 0]["dt"][0, 0]
        assert U_dt.shape == (1, 1), "U.dt should be 2D scalar"
        assert abs(U_dt[0, 0] - u_dt) < 1e-10

        # Connectivity masks
        a = DCM["a"][0, 0]
        assert a.shape == (N, N)
        np.testing.assert_array_equal(a, a_mask)

        c = DCM["c"][0, 0]
        assert c.shape == (N, M)
        np.testing.assert_array_equal(c, c_mask)

        # b and d should be empty 3D
        b = DCM["b"][0, 0]
        assert b.shape == (N, N, 0), f"b shape: {b.shape}"

        d = DCM["d"][0, 0]
        assert d.shape == (N, N, 0), f"d shape: {d.shape}"

        # Dimensions -- 2D scalar arrays
        n_val = DCM["n"][0, 0]
        assert n_val.shape == (1, 1), "n should be 2D"
        assert n_val[0, 0] == N

        v_val = DCM["v"][0, 0]
        assert v_val.shape == (1, 1), "v should be 2D"
        assert v_val[0, 0] == v

        # Timing
        TE = DCM["TE"][0, 0]
        assert TE.shape == (1, 1), "TE should be 2D"
        assert TE[0, 0] == 0.04

        delays = DCM["delays"][0, 0]
        assert delays.shape == (1, N)

        # Options
        options = DCM["options"][0, 0]
        assert options["nonlinear"][0, 0][0, 0] == 0
        assert options["induced"][0, 0][0, 0] == 0
        assert options["nograph"][0, 0][0, 0] == 1
        assert options["maxit"][0, 0][0, 0] == 128


class TestSpectralDCMExportRoundtrip:
    """Test spectral DCM .mat export produces CSD-mode struct."""

    def test_spectral_dcm_export_roundtrip(self, tmp_path):
        """Export and reload spectral DCM struct, verify CSD fields."""
        v, N, M = 200, 3, 1
        bold = np.random.randn(v, N).astype(np.float64)
        TR = 2.0
        a_mask = np.ones((N, N), dtype=np.float64)
        c_mask = np.eye(N, M, dtype=np.float64)

        mat_path = str(tmp_path / "spectral_dcm.mat")
        export_spectral_dcm_for_spm(bold, a_mask, c_mask, TR, mat_path)

        # Load and verify
        data = scipy.io.loadmat(mat_path, squeeze_me=False)
        DCM = data["DCM"]

        # Y.y should be the BOLD data (not CSD)
        Y_y = DCM["Y"][0, 0]["y"][0, 0]
        assert Y_y.shape == (v, N)
        np.testing.assert_array_almost_equal(Y_y, bold)

        # Options: induced=1, analysis=CSD
        options = DCM["options"][0, 0]
        assert options["induced"][0, 0][0, 0] == 1
        analysis = options["analysis"][0, 0]
        assert analysis[0, 0] == "CSD", f"analysis = {analysis[0, 0]}"
        assert options["order"][0, 0][0, 0] == 8

        # U should have constant stimulus
        U_u = DCM["U"][0, 0]["u"][0, 0]
        microtime_bins = v * 16 + 32
        assert U_u.shape == (microtime_bins, M)
        # All ones (constant input)
        assert np.all(U_u == 1.0)


class TestRDCMExportRoundtrip:
    """Test rDCM .mat export for tapas compatibility."""

    def test_rdcm_export_roundtrip(self, tmp_path):
        """Export and reload rDCM struct, verify basic fields."""
        v, N, M = 500, 3, 2
        bold = np.random.randn(v, N).astype(np.float64)
        TR = 0.5
        u_dt = TR  # rDCM uses TR-resolution inputs

        # Stimulus at TR resolution (no microtime upsampling for rDCM)
        stimulus = np.zeros((v, M), dtype=np.float64)
        stimulus[10:30, 0] = 1.0
        stimulus[50:70, 1] = 1.0

        a_mask = np.ones((N, N), dtype=np.float64)
        c_mask = np.ones((N, M), dtype=np.float64)

        mat_path = str(tmp_path / "rdcm.mat")
        export_rdcm_for_tapas(bold, stimulus, a_mask, c_mask, TR, u_dt, mat_path)

        # Load and verify
        data = scipy.io.loadmat(mat_path, squeeze_me=False)
        DCM = data["DCM"]

        # Basic fields
        Y_y = DCM["Y"][0, 0]["y"][0, 0]
        assert Y_y.shape == (v, N)
        np.testing.assert_array_almost_equal(Y_y, bold)

        U_u = DCM["U"][0, 0]["u"][0, 0]
        assert U_u.shape == (v, M)

        a = DCM["a"][0, 0]
        assert a.shape == (N, N)
        np.testing.assert_array_equal(a, a_mask)

        # Dimensions
        n_val = DCM["n"][0, 0]
        assert n_val[0, 0] == N

        # Options: minimal (just nograph)
        options = DCM["options"][0, 0]
        assert options["nograph"][0, 0][0, 0] == 1


class TestUpsampleStimulus:
    """Test stimulus upsampling to microtime resolution."""

    def test_upsample_shape_and_padding(self):
        """Verify output shape and zero-padding convention."""
        T, M = 100, 2
        TR = 2.0
        microtime_factor = 16

        stim_tr = np.ones((T, M), dtype=np.float64)
        upsampled, u_dt = upsample_stimulus(stim_tr, TR, microtime_factor)

        # Expected shape: T * factor + 32 padding
        expected_rows = T * microtime_factor + 32
        assert upsampled.shape == (expected_rows, M), (
            f"Expected {(expected_rows, M)}, got {upsampled.shape}"
        )

        # u_dt should be TR / microtime_factor
        assert abs(u_dt - TR / microtime_factor) < 1e-12

        # First 32 rows should be zeros (padding)
        np.testing.assert_array_equal(upsampled[:32, :], 0.0)

        # Remaining rows should be the upsampled stimulus (all ones)
        assert np.all(upsampled[32:, :] == 1.0)

    def test_upsample_block_pattern(self):
        """Verify nearest-neighbor upsampling preserves block structure."""
        T, M = 10, 1
        TR = 2.0
        factor = 4

        # Block: off-on-on-off-...
        stim = np.array(
            [[0], [1], [1], [0], [0], [1], [0], [0], [0], [0]],
            dtype=np.float64,
        )
        upsampled, _ = upsample_stimulus(stim, TR, factor)

        # Skip 32 padding, check repeated pattern
        body = upsampled[32:, 0]
        assert len(body) == T * factor

        # Each TR sample repeated factor times
        for i in range(T):
            block = body[i * factor : (i + 1) * factor]
            expected = stim[i, 0]
            np.testing.assert_array_equal(
                block, expected, err_msg=f"Block {i} mismatch"
            )


class TestComparePosteriorMeans:
    """Test hybrid relative/absolute error metric."""

    def test_within_tolerance_large_values(self):
        """Relative error for large values, should pass at 10%."""
        ref_A = np.array([[0.5, 0.3], [0.2, 0.4]], dtype=np.float64)
        # Pyro values within 5% of reference
        pyro_A = ref_A * 1.05

        result = compare_posterior_means(pyro_A, ref_A, tolerance=0.10)
        assert result["within_tolerance"] is True
        assert result["max_relative_error"] < 0.10

    def test_within_tolerance_near_zero(self):
        """Absolute error for near-zero values."""
        ref_A = np.array([[0.0, 0.005], [0.002, 0.0]], dtype=np.float64)
        # Small absolute differences
        pyro_A = np.array([[0.01, 0.015], [0.012, 0.005]], dtype=np.float64)

        result = compare_posterior_means(
            pyro_A, ref_A, tolerance=0.10,
            near_zero_threshold=0.01, near_zero_atol=0.02,
        )
        assert result["within_tolerance"] is True

    def test_failure_large_values(self):
        """Values clearly outside tolerance should fail."""
        ref_A = np.array([[0.5, 0.3], [0.2, 0.4]], dtype=np.float64)
        # 30% error on first element
        pyro_A = ref_A.copy()
        pyro_A[0, 0] = 0.65  # 30% error

        result = compare_posterior_means(pyro_A, ref_A, tolerance=0.10)
        assert result["within_tolerance"] is False
        assert result["max_relative_error"] > 0.10

    def test_error_matrix_shape(self):
        """Element errors matrix should match input shape."""
        N = 4
        ref_A = np.random.randn(N, N).astype(np.float64)
        pyro_A = ref_A + 0.01 * np.random.randn(N, N)

        result = compare_posterior_means(pyro_A, ref_A)
        assert result["element_errors"].shape == (N, N)

    def test_mixed_large_and_near_zero(self):
        """Hybrid metric correctly switches between relative and absolute."""
        ref_A = np.array([[0.5, 0.0], [0.001, 0.3]], dtype=np.float64)
        pyro_A = np.array([[0.55, 0.01], [0.009, 0.33]], dtype=np.float64)

        result = compare_posterior_means(
            pyro_A, ref_A, tolerance=0.15,
            near_zero_threshold=0.01, near_zero_atol=0.02,
        )
        # (0,0): rel error = |0.55-0.5|/0.5 = 0.10 < 0.15 OK
        # (0,1): abs error = |0.01-0.0| = 0.01 < 0.02 OK
        # (1,0): abs error = |0.009-0.001| = 0.008 < 0.02 OK
        # (1,1): rel error = |0.33-0.3|/0.3 = 0.10 < 0.15 OK
        assert result["within_tolerance"] is True


class TestCompareModelRanking:
    """Test model ranking comparison utility."""

    def test_perfect_agreement(self):
        """All rankings agree: agreement rate = 1.0."""
        scenarios = [
            {"spm_F": 100.0, "pyro_elbo": -50.0},
            {"spm_F": 80.0, "pyro_elbo": -70.0},
            {"spm_F": 60.0, "pyro_elbo": -90.0},
        ]
        result = compare_model_ranking(scenarios)
        assert result["agreement_rate"] == 1.0
        assert result["total_pairs"] == 3
        assert result["agreements"] == 3

    def test_one_disagreement(self):
        """One pair disagrees: rate = 2/3."""
        scenarios = [
            {"spm_F": 100.0, "pyro_elbo": -50.0},   # SPM best, Pyro best
            {"spm_F": 80.0, "pyro_elbo": -90.0},    # SPM mid, Pyro worst
            {"spm_F": 60.0, "pyro_elbo": -70.0},    # SPM worst, Pyro mid
        ]
        # Pair (0,1): SPM: 0>1, Pyro: 0>1 -> agree
        # Pair (0,2): SPM: 0>2, Pyro: 0>2 -> agree
        # Pair (1,2): SPM: 1>2, Pyro: 1<2 -> disagree
        result = compare_model_ranking(scenarios)
        assert result["total_pairs"] == 3
        assert result["agreements"] == 2
        assert abs(result["agreement_rate"] - 2.0 / 3.0) < 1e-10

    def test_two_scenarios(self):
        """Minimal case: two scenarios, one pair."""
        scenarios = [
            {"spm_F": 100.0, "pyro_elbo": -50.0},
            {"spm_F": 80.0, "pyro_elbo": -70.0},
        ]
        result = compare_model_ranking(scenarios)
        assert result["total_pairs"] == 1
        assert result["agreements"] == 1
        assert result["agreement_rate"] == 1.0

    def test_pairwise_details(self):
        """Check pairwise results contain expected structure."""
        scenarios = [
            {"spm_F": 100.0, "pyro_elbo": -50.0},
            {"spm_F": 80.0, "pyro_elbo": -70.0},
        ]
        result = compare_model_ranking(scenarios)
        assert len(result["pairwise_results"]) == 1
        pair = result["pairwise_results"][0]
        assert "i" in pair
        assert "j" in pair
        assert "spm_prefers_i" in pair
        assert "pyro_prefers_i" in pair
        assert "agree" in pair
