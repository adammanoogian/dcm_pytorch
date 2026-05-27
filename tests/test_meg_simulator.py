"""Tests for MEG-like timeseries simulator (OU process).

Tests cover:
- Output shapes, dtypes, and reproducibility
- Numerical stability (no NaN/Inf in long simulations)
- Variance bounds (output neither explodes nor collapses)
- Sensorimotor A matrix structure and eigenvalue stability
- CSD consistency between OU timeseries and analytical spectral DCM
- Dataset generation convenience function
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.csd_computation import compute_empirical_csd
from pyro_dcm.simulators.meg_simulator import (
    SENSORIMOTOR_ROI_NAMES,
    generate_meg_dataset,
    make_sensorimotor_A,
    simulate_meg_timeseries,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sensorimotor_A() -> torch.Tensor:
    """Stable 10x10 sensorimotor A matrix."""
    return make_sensorimotor_A(seed=42)


@pytest.fixture()
def small_A() -> torch.Tensor:
    """Stable 3x3 A matrix for fast tests."""
    return make_stable_A_spectral(3, seed=42)


# ---------------------------------------------------------------------------
# (a) test_simulate_meg_timeseries_shapes
# ---------------------------------------------------------------------------


class TestSimulateMEGTimeseriesShapes:
    """Verify (n_samples, T, N) output shape for various configs."""

    def test_default_params(self, small_A: torch.Tensor) -> None:
        """Default sfreq=250, duration=4.0 -> T=1000."""
        result = simulate_meg_timeseries(small_A, n_samples=3)
        ts = result["timeseries"]
        assert ts.shape == (3, 1000, 3)
        assert ts.dtype == torch.float64

    def test_custom_sfreq_duration(self, small_A: torch.Tensor) -> None:
        """Custom sfreq=100, duration=2.0 -> T=200."""
        result = simulate_meg_timeseries(
            small_A, sfreq=100.0, duration=2.0, n_samples=10
        )
        assert result["timeseries"].shape == (10, 200, 3)

    def test_sensorimotor_shape(
        self, sensorimotor_A: torch.Tensor
    ) -> None:
        """10-region sensorimotor A -> N=10."""
        result = simulate_meg_timeseries(
            sensorimotor_A, n_samples=2, duration=1.0
        )
        assert result["timeseries"].shape == (2, 250, 10)

    def test_output_keys(self, small_A: torch.Tensor) -> None:
        """All expected keys present in output dict."""
        result = simulate_meg_timeseries(small_A, n_samples=1)
        expected_keys = {
            "timeseries",
            "A",
            "sfreq",
            "duration",
            "roi_names",
            "sigma",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# (b) test_simulate_meg_timeseries_reproducible
# ---------------------------------------------------------------------------


class TestSimulateMEGTimeseriesReproducible:
    """Same seed produces identical output."""

    def test_same_seed_identical(self, small_A: torch.Tensor) -> None:
        """Two runs with same seed yield bit-identical timeseries."""
        r1 = simulate_meg_timeseries(small_A, n_samples=5, seed=123)
        r2 = simulate_meg_timeseries(small_A, n_samples=5, seed=123)
        assert torch.equal(r1["timeseries"], r2["timeseries"])

    def test_different_seed_different(
        self, small_A: torch.Tensor
    ) -> None:
        """Two runs with different seeds yield different timeseries."""
        r1 = simulate_meg_timeseries(small_A, n_samples=5, seed=10)
        r2 = simulate_meg_timeseries(small_A, n_samples=5, seed=20)
        assert not torch.equal(r1["timeseries"], r2["timeseries"])


# ---------------------------------------------------------------------------
# (c) test_simulate_meg_timeseries_stable
# ---------------------------------------------------------------------------


class TestSimulateMEGTimeseriesStable:
    """No NaN or Inf in 10s simulation."""

    def test_no_nan_inf_long_sim(self, small_A: torch.Tensor) -> None:
        """10s simulation at 250 Hz produces no NaN/Inf."""
        result = simulate_meg_timeseries(
            small_A, duration=10.0, n_samples=3, seed=42
        )
        ts = result["timeseries"]
        assert not torch.isnan(ts).any(), "NaN found in timeseries"
        assert not torch.isinf(ts).any(), "Inf found in timeseries"


# ---------------------------------------------------------------------------
# (d) test_simulate_meg_timeseries_variance_bounded
# ---------------------------------------------------------------------------


class TestSimulateMEGTimeseriesVarianceBounded:
    """Variance is finite, positive, and reasonable (not exploding)."""

    def test_variance_bounded(self, small_A: torch.Tensor) -> None:
        """Variance per region is in a reasonable range."""
        result = simulate_meg_timeseries(
            small_A, n_samples=20, duration=4.0, seed=42
        )
        ts = result["timeseries"]
        # Variance per region across all samples and time
        var_per_region = ts.var(dim=(0, 1))
        # Each region should have positive, finite variance
        assert (var_per_region > 0).all(), (
            f"Zero variance in some regions: {var_per_region}"
        )
        assert (var_per_region < 1e6).all(), (
            f"Exploding variance: {var_per_region}"
        )
        assert not torch.isnan(var_per_region).any()


# ---------------------------------------------------------------------------
# (e) test_make_sensorimotor_A_shape
# ---------------------------------------------------------------------------


class TestMakeSensorimotorAShape:
    """Returns (10, 10) float64 tensor."""

    def test_shape_and_dtype(self) -> None:
        """Shape is (10, 10), dtype is float64."""
        A = make_sensorimotor_A(seed=42)
        assert A.shape == (10, 10)
        assert A.dtype == torch.float64


# ---------------------------------------------------------------------------
# (f) test_make_sensorimotor_A_stable
# ---------------------------------------------------------------------------


class TestMakeSensorimotorAStable:
    """All eigenvalues have negative real parts."""

    def test_eigenvalues_negative(self) -> None:
        """Max Re(lambda) < 0 across several seeds."""
        for seed in [0, 42, 123, 999]:
            A = make_sensorimotor_A(seed=seed)
            eigvals = torch.linalg.eigvals(A.to(torch.complex128))
            assert eigvals.real.max().item() < 0, (
                f"Unstable A for seed={seed}: "
                f"max Re(lambda) = {eigvals.real.max().item():.6f}"
            )


# ---------------------------------------------------------------------------
# (g) test_make_sensorimotor_A_structure
# ---------------------------------------------------------------------------


class TestMakeSensorimotorAStructure:
    """Structured connections exist in the sensorimotor A."""

    def test_diagonal_negative(self) -> None:
        """All diagonal elements (self-connections) are negative."""
        A = make_sensorimotor_A(seed=42)
        assert (A.diagonal() < 0).all()

    def test_m1_s1_connections_exist(self) -> None:
        """M1 <-> S1 ipsilateral connections are nonzero.

        M1_lh=0, S1_lh=2, M1_rh=1, S1_rh=3.
        """
        A = make_sensorimotor_A(seed=42)
        # Left hemisphere
        assert A[0, 2].abs() > 0.01, "M1_lh -> S1_lh connection missing"
        assert A[2, 0].abs() > 0.01, "S1_lh -> M1_lh connection missing"
        # Right hemisphere
        assert A[1, 3].abs() > 0.01, "M1_rh -> S1_rh connection missing"
        assert A[3, 1].abs() > 0.01, "S1_rh -> M1_rh connection missing"

    def test_bilateral_connections_symmetric(self) -> None:
        """Homotopic bilateral connections are bidirectional.

        Pairs: (0,1), (2,3), (4,5), (6,7), (8,9).
        """
        A = make_sensorimotor_A(seed=42)
        for lh, rh in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
            assert abs(A[lh, rh].item() - A[rh, lh].item()) < 1e-12, (
                f"Bilateral connection ({lh},{rh}) not symmetric: "
                f"A[{lh},{rh}]={A[lh, rh].item():.6f}, "
                f"A[{rh},{lh}]={A[rh, lh].item():.6f}"
            )

    def test_roi_names_match(self) -> None:
        """SENSORIMOTOR_ROI_NAMES has exactly 10 entries."""
        assert len(SENSORIMOTOR_ROI_NAMES) == 10
        assert SENSORIMOTOR_ROI_NAMES[0] == "M1_lh"
        assert SENSORIMOTOR_ROI_NAMES[9] == "A1_rh"


# ---------------------------------------------------------------------------
# (h) test_csd_consistency
# ---------------------------------------------------------------------------


class TestCSDConsistency:
    """Empirical CSD from OU timeseries matches analytical CSD qualitatively.

    The Ornstein-Uhlenbeck process is the time-domain realization of the
    linear stochastic model underlying spectral DCM [REF-010]. Therefore,
    the empirical CSD computed from OU-generated timeseries should be
    qualitatively consistent with the analytical CSD from
    ``simulate_spectral_dcm``.

    We test that diagonal (auto-spectra) correlation > 0.5 across
    frequencies. Exact match is not expected due to finite-sample noise
    in the OU simulation.
    """

    def test_autospectra_correlation(self) -> None:
        """Auto-spectra from OU timeseries correlate with analytical CSD."""
        # Use a small A for speed
        A = make_stable_A_spectral(3, seed=42)
        N = A.shape[0]

        # Analytical CSD from spectral DCM
        # Use TR matching OU sfreq: sfreq=250 -> Nyquist=125 Hz
        # But spectral DCM uses fMRI frequencies (1/128 to 0.25 Hz).
        # For comparison, use a TR that matches our simulation rate:
        # sfreq=50 Hz (small for speed), TR = 1/50 = 0.02s
        sfreq = 50.0
        duration = 20.0  # Long enough for stable CSD estimate
        n_samples = 100  # Many samples to reduce variance

        # Generate OU timeseries
        result = simulate_meg_timeseries(
            A,
            sfreq=sfreq,
            duration=duration,
            n_samples=n_samples,
            sigma=1.0,
            seed=42,
        )
        ts = result["timeseries"]  # (n_samples, T, N)

        # Average empirical CSD across samples for stability
        # compute_empirical_csd expects (T, N)
        # Use frequency grid matching spectral DCM conventions
        T = ts.shape[1]
        nperseg = min(256, T)
        # Target frequencies: skip DC, go up to Nyquist/4 for stability
        freqs_target = np.linspace(0.5, sfreq / 4, 32)

        # Average CSD across all samples
        csd_sum = np.zeros((32, N, N), dtype=np.complex128)
        for s in range(n_samples):
            sample = ts[s].numpy()  # (T, N)
            csd_s = compute_empirical_csd(
                sample, fs=sfreq, freqs=freqs_target, nperseg=nperseg
            )
            csd_sum += csd_s
        csd_empirical = csd_sum / n_samples

        # Analytical CSD from spectral DCM
        # spectral_dcm_forward uses TR-based frequency grid
        # We pass TR=1/sfreq to match our sampling
        analytical_result = simulate_spectral_dcm(
            A, TR=1.0 / sfreq, n_freqs=128
        )
        analytical_csd = analytical_result["csd"]  # (F, N, N)
        analytical_freqs = analytical_result["freqs"].numpy()  # (F,)

        # Interpolate analytical CSD onto our frequency grid
        for i in range(N):
            analytical_auto_full = analytical_csd[:, i, i].real.numpy()
            empirical_auto = csd_empirical[:, i, i].real

            # Interpolate analytical onto empirical frequency grid
            analytical_auto = np.interp(
                freqs_target, analytical_freqs, analytical_auto_full
            )

            # Both should be positive (auto-spectra)
            assert (empirical_auto > 0).sum() > len(empirical_auto) // 2, (
                f"Region {i}: empirical auto-spectrum mostly non-positive"
            )

            # Pearson correlation between empirical and analytical
            emp_c = empirical_auto - empirical_auto.mean()
            ana_c = analytical_auto - analytical_auto.mean()
            emp_std = float(np.sqrt(np.sum(emp_c**2)))
            ana_std = float(np.sqrt(np.sum(ana_c**2)))

            if emp_std < 1e-30 or ana_std < 1e-30:
                pytest.skip(
                    f"Region {i}: degenerate auto-spectrum (zero variance)"
                )

            corr = float(np.sum(emp_c * ana_c) / (emp_std * ana_std))

            assert corr > 0.5, (
                f"Region {i}: auto-spectra correlation = {corr:.4f}, "
                f"expected > 0.5 (empirical vs analytical CSD)"
            )


# ---------------------------------------------------------------------------
# (i) test_generate_meg_dataset_shapes
# ---------------------------------------------------------------------------


class TestGenerateMEGDatasetShapes:
    """Train and val have correct shapes."""

    def test_shapes(self) -> None:
        """Default 10 regions, 250 Hz, 4s -> T=1000."""
        ds = generate_meg_dataset(
            n_train=10, n_val=5, duration=2.0, seed=42
        )
        # T = 250 * 2.0 = 500
        assert ds["train"].shape == (10, 500, 10)
        assert ds["val"].shape == (5, 500, 10)
        assert ds["A"].shape == (10, 10)

    def test_custom_n_roi(self) -> None:
        """Non-10 n_roi uses make_stable_A_spectral."""
        ds = generate_meg_dataset(
            n_roi=4, n_train=5, n_val=2, duration=1.0, seed=42
        )
        assert ds["train"].shape == (5, 250, 4)
        assert ds["val"].shape == (2, 250, 4)
        assert ds["A"].shape == (4, 4)


# ---------------------------------------------------------------------------
# (j) test_generate_meg_dataset_default_A
# ---------------------------------------------------------------------------


class TestGenerateMEGDatasetDefaultA:
    """When A=None and n_roi=10, uses make_sensorimotor_A."""

    def test_default_uses_sensorimotor(self) -> None:
        """roi_names are set when using default 10-region A."""
        ds = generate_meg_dataset(
            n_train=3, n_val=2, duration=1.0, seed=42
        )
        assert ds["roi_names"] is not None
        assert len(ds["roi_names"]) == 10
        assert ds["roi_names"][0] == "M1_lh"
        assert ds["roi_names"][-1] == "A1_rh"

    def test_custom_A_no_roi_names(self) -> None:
        """When A is provided, roi_names is None."""
        A = make_stable_A_spectral(5, seed=42)
        ds = generate_meg_dataset(
            A=A, n_train=3, n_val=2, duration=1.0, seed=42
        )
        assert ds["roi_names"] is None
        assert ds["A"].shape == (5, 5)
