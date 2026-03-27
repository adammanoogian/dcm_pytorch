"""Unit tests for the spectral DCM Pyro generative model.

Tests cover:
- CSD decomposition (shape, roundtrip, dtype)
- Model trace structure (sample sites, deterministic sites, shapes)
- A matrix properties (negative diagonal, masking)
- Numerical stability (finite samples, finite log_prob)
- SVI smoke tests (no NaN, loss decreases)
"""

from __future__ import annotations

import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

from pyro_dcm.models.spectral_dcm_model import (
    decompose_csd_for_likelihood,
    spectral_dcm_model,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def spectral_data() -> dict:
    """Generate synthetic spectral DCM data for 3 regions.

    Uses ``simulate_spectral_dcm`` to produce a small CSD dataset
    with known parameters.
    """
    A = make_stable_A_spectral(3, seed=42)
    result = simulate_spectral_dcm(A, TR=2.0, n_freqs=32, seed=42)
    # Build a_mask: all connections present (full connectivity)
    N = 3
    a_mask = torch.ones(N, N, dtype=torch.float64)
    return {
        "observed_csd": result["csd"],
        "freqs": result["freqs"],
        "a_mask": a_mask,
        "N": N,
    }


@pytest.fixture()
def sparse_mask() -> torch.Tensor:
    """Sparse 3x3 structural mask with some absent connections."""
    return torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# CSD decomposition tests
# ---------------------------------------------------------------------------


class TestDecomposeCsd:
    """Tests for decompose_csd_for_likelihood."""

    def test_decompose_csd_shape(self) -> None:
        """Output shape should be (2 * F * N * N,) for (F, N, N) input."""
        F, N = 32, 3
        csd = torch.randn(F, N, N, dtype=torch.complex128)
        vec = decompose_csd_for_likelihood(csd)
        assert vec.shape == (2 * F * N * N,)

    def test_decompose_csd_roundtrip(self) -> None:
        """Decomposition preserves values: roundtrip within 1e-15."""
        F, N = 32, 3
        csd = torch.randn(F, N, N, dtype=torch.complex128)
        vec = decompose_csd_for_likelihood(csd)

        # Reconstruct
        half = F * N * N
        real_part = vec[:half].reshape(F, N, N)
        imag_part = vec[half:].reshape(F, N, N)
        csd_reconstructed = torch.complex(real_part, imag_part)

        assert torch.allclose(csd, csd_reconstructed, atol=1e-15)

    def test_decompose_csd_dtype(self) -> None:
        """Output must be float64 (not complex128)."""
        csd = torch.randn(32, 3, 3, dtype=torch.complex128)
        vec = decompose_csd_for_likelihood(csd)
        assert vec.dtype == torch.float64


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


class TestModelStructure:
    """Tests for model trace sites and shapes."""

    def _run_trace(
        self, spectral_data: dict
    ) -> pyro.poutine.trace_struct.Trace:
        """Run model under trace poutine and return the trace."""
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=spectral_data["observed_csd"],
            freqs=spectral_data["freqs"],
            a_mask=spectral_data["a_mask"],
            N=spectral_data["N"],
        )
        return trace

    def test_model_trace_has_expected_sites(
        self, spectral_data: dict
    ) -> None:
        """Trace must contain all expected sample and deterministic sites."""
        trace = self._run_trace(spectral_data)
        site_names = set(trace.nodes.keys()) - {"_INPUT", "_RETURN"}

        # Sample sites
        expected_samples = {
            "A_free",
            "noise_a",
            "noise_b",
            "noise_c",
            "csd_noise_scale",
            "obs_csd",
        }
        for name in expected_samples:
            assert name in site_names, f"Missing sample site: {name}"
            assert trace.nodes[name]["type"] == "sample"

        # Deterministic sites
        expected_det = {"A", "predicted_csd"}
        for name in expected_det:
            assert name in site_names, f"Missing deterministic site: {name}"

    def test_model_samples_correct_shapes(
        self, spectral_data: dict
    ) -> None:
        """Sample site values must have correct shapes."""
        trace = self._run_trace(spectral_data)
        N = spectral_data["N"]

        assert trace.nodes["A_free"]["value"].shape == (N, N)
        assert trace.nodes["noise_a"]["value"].shape == (2, N)
        assert trace.nodes["noise_b"]["value"].shape == (2, 1)
        assert trace.nodes["noise_c"]["value"].shape == (2, N)

    def test_model_a_diagonal_negative(
        self, spectral_data: dict
    ) -> None:
        """A matrix diagonal elements must all be negative."""
        trace = self._run_trace(spectral_data)
        A = trace.nodes["A"]["value"]
        diag = A.diagonal()
        assert (diag < 0).all(), f"Non-negative diagonal found: {diag}"

    def test_model_masking_works(
        self, spectral_data: dict, sparse_mask: torch.Tensor
    ) -> None:
        """Masked off-diagonal A positions must be zero."""
        data = dict(spectral_data)
        data["a_mask"] = sparse_mask

        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=data["observed_csd"],
            freqs=data["freqs"],
            a_mask=data["a_mask"],
            N=data["N"],
        )
        A = trace.nodes["A"]["value"]

        # Check off-diagonal zeros where mask is 0
        N = data["N"]
        for i in range(N):
            for j in range(N):
                if i != j and sparse_mask[i, j] == 0:
                    assert A[i, j].item() == 0.0, (
                        f"A[{i},{j}] should be 0 but is {A[i, j].item()}"
                    )

    def test_model_no_c_sampled(self, spectral_data: dict) -> None:
        """No sample site named 'C' should exist in spectral DCM."""
        trace = self._run_trace(spectral_data)
        site_names = set(trace.nodes.keys())
        assert "C" not in site_names, "Spectral DCM should not sample C"

    def test_model_no_complex_in_obs(
        self, spectral_data: dict
    ) -> None:
        """The obs_csd sample site distribution must use float64."""
        trace = self._run_trace(spectral_data)
        obs_site = trace.nodes["obs_csd"]
        obs_val = obs_site["value"]
        assert obs_val.dtype == torch.float64, (
            f"obs_csd value dtype is {obs_val.dtype}, expected float64"
        )


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for finite values and stable computation."""

    def test_model_prior_samples_finite(
        self, spectral_data: dict
    ) -> None:
        """Predicted CSD from prior samples must be finite."""
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=spectral_data["observed_csd"],
            freqs=spectral_data["freqs"],
            a_mask=spectral_data["a_mask"],
            N=spectral_data["N"],
        )
        pred_csd = trace.nodes["predicted_csd"]["value"]
        assert torch.isfinite(pred_csd.real).all(), "NaN/Inf in real CSD"
        assert torch.isfinite(pred_csd.imag).all(), "NaN/Inf in imag CSD"

    def test_model_log_prob_finite(self, spectral_data: dict) -> None:
        """Total log probability from trace must be finite."""
        trace = pyro.poutine.trace(spectral_dcm_model).get_trace(
            observed_csd=spectral_data["observed_csd"],
            freqs=spectral_data["freqs"],
            a_mask=spectral_data["a_mask"],
            N=spectral_data["N"],
        )
        lp = trace.log_prob_sum()
        assert torch.isfinite(lp), f"log_prob_sum is {lp}, expected finite"


# ---------------------------------------------------------------------------
# SVI smoke tests
# ---------------------------------------------------------------------------


class TestSVI:
    """Smoke tests for SVI training with AutoNormal guide."""

    def test_svi_runs_without_nan(self, spectral_data: dict) -> None:
        """10 SVI steps with AutoNormal must produce no NaN losses."""
        pyro.clear_param_store()

        guide = AutoNormal(spectral_dcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            spectral_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        losses = []
        for _ in range(10):
            loss = svi.step(
                observed_csd=spectral_data["observed_csd"],
                freqs=spectral_data["freqs"],
                a_mask=spectral_data["a_mask"],
                N=spectral_data["N"],
            )
            losses.append(loss)

        for i, loss in enumerate(losses):
            assert not torch.isnan(torch.tensor(loss)), (
                f"NaN loss at step {i}"
            )

    def test_svi_loss_decreases(self, spectral_data: dict) -> None:
        """ELBO should decrease over 50 SVI steps (mean comparison)."""
        pyro.clear_param_store()

        guide = AutoNormal(spectral_dcm_model, init_scale=0.01)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0}
        )
        svi = SVI(
            spectral_dcm_model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        losses = []
        for _ in range(50):
            loss = svi.step(
                observed_csd=spectral_data["observed_csd"],
                freqs=spectral_data["freqs"],
                a_mask=spectral_data["a_mask"],
                N=spectral_data["N"],
            )
            losses.append(loss)

        # Mean of first 10 should be greater than mean of last 10
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Loss did not decrease: first 10 mean={first_10:.4f}, "
            f"last 10 mean={last_10:.4f}"
        )
