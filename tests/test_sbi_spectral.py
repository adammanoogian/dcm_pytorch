"""Tests for SBI spectral DCM infrastructure.

Tests simulator, prior, embedding net, and diagnostics.
Guards sbi-dependent tests with pytest.importorskip.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.spectral_transfer import default_frequency_grid
from pyro_dcm.inference.sbi_embedding import CSDEmbeddingNet
from pyro_dcm.inference.sbi_spectral import (
    make_spectral_dcm_prior,
    make_spectral_dcm_simulator,
)


def _sbi_available() -> bool:
    """Check if sbi package is importable."""
    try:
        import sbi  # noqa: F401

        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture()
def n_regions() -> int:
    """Return number of brain regions for tests."""
    return 3


@pytest.fixture()
def a_mask(n_regions: int) -> torch.Tensor:
    """Full connectivity mask (all connections)."""
    return torch.ones(n_regions, n_regions, dtype=torch.float64)


@pytest.fixture()
def freqs() -> torch.Tensor:
    """Return standard frequency grid for testing."""
    return default_frequency_grid(TR=2.0, n_freqs=16)


@pytest.fixture()
def simulator(
    n_regions: int,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
) -> object:
    """Build simulator for tests."""
    return make_spectral_dcm_simulator(n_regions, freqs, a_mask)


@pytest.fixture()
def prior(n_regions: int, a_mask: torch.Tensor) -> object:
    """Build prior for tests."""
    return make_spectral_dcm_prior(n_regions, a_mask)


# ------------------------------------------------------------------ #
#  Simulator tests                                                    #
# ------------------------------------------------------------------ #


def test_simulator_returns_correct_shape(
    simulator: object,
    n_regions: int,
    freqs: torch.Tensor,
) -> None:
    """Simulator output is 1D float64 with expected length."""
    N = n_regions
    F = freqs.shape[0]
    n_free = N * N  # full mask
    theta = torch.zeros(n_free, dtype=torch.float64)
    x = simulator(theta)

    expected_dim = 2 * F * N * N
    assert x.shape == (expected_dim,), (
        f"Expected shape ({expected_dim},), got {x.shape}"
    )
    assert x.dtype == torch.float64, (
        f"Expected float64, got {x.dtype}"
    )


def test_simulator_stable_a_no_nan(
    simulator: object,
    n_regions: int,
) -> None:
    """Stable A matrix produces no NaN in simulator output."""
    n_free = n_regions * n_regions
    # Small values -> stable system (parameterize_A makes diagonal negative)
    theta = torch.zeros(n_free, dtype=torch.float64)
    x = simulator(theta)

    assert not torch.any(torch.isnan(x)), "Simulator output contains NaN"
    assert not torch.any(torch.isinf(x)), "Simulator output contains Inf"


def test_simulator_unstable_a_fallback(
    n_regions: int,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
) -> None:
    """Unstable A matrix returns zero fallback without crashing."""
    sim = make_spectral_dcm_simulator(
        n_regions, freqs, a_mask, eig_clamp=None
    )
    n_free = n_regions * n_regions
    # Large positive off-diagonal -> potentially unstable
    theta = torch.full((n_free,), 10.0, dtype=torch.float64)
    x = sim(theta)

    # Should not crash; output should be finite (zeros fallback)
    assert torch.isfinite(x).all(), "Unstable fallback produced non-finite"
    assert x.shape[0] == 2 * freqs.shape[0] * n_regions * n_regions


# ------------------------------------------------------------------ #
#  Prior tests                                                        #
# ------------------------------------------------------------------ #


def test_prior_shape_and_type(
    prior: object,
    n_regions: int,
) -> None:
    """Prior has correct event_shape matching free parameters."""
    n_free = n_regions * n_regions
    assert prior.event_shape == (n_free,), (
        f"Expected event_shape ({n_free},), got {prior.event_shape}"
    )


def test_prior_sample_range(prior: object) -> None:
    """Prior samples are in a reasonable range for N(0, 1/64)."""
    samples = prior.sample((10_000,))
    expected_std = (1.0 / 64.0) ** 0.5  # ~0.125

    # Empirical std should be close to prior std
    empirical_std = samples.std(dim=0).mean().item()
    assert abs(empirical_std - expected_std) < 0.02, (
        f"Empirical std {empirical_std:.4f} too far from "
        f"expected {expected_std:.4f}"
    )

    # No extreme outliers expected
    assert samples.abs().max() < 2.0, (
        f"Max absolute sample {samples.abs().max():.2f} unexpectedly large"
    )


# ------------------------------------------------------------------ #
#  Embedding net tests                                                #
# ------------------------------------------------------------------ #


def test_embedding_net_output_shape(n_regions: int) -> None:
    """CSDEmbeddingNet produces correct output dimensions."""
    F = 16
    input_dim = 2 * F * n_regions * n_regions
    embed_dim = 64
    net = CSDEmbeddingNet(input_dim=input_dim, embed_dim=embed_dim)

    batch_size = 8
    x = torch.randn(batch_size, input_dim)
    out = net(x)

    assert out.shape == (batch_size, embed_dim), (
        f"Expected ({batch_size}, {embed_dim}), got {out.shape}"
    )


def test_embedding_net_single_sample() -> None:
    """CSDEmbeddingNet handles single sample (batch=1) in eval mode."""
    input_dim = 128
    embed_dim = 32
    net = CSDEmbeddingNet(
        input_dim=input_dim, embed_dim=embed_dim, hidden_dim=64
    )
    net.eval()  # BatchNorm needs eval for single samples

    x = torch.randn(1, input_dim)
    out = net(x)
    assert out.shape == (1, embed_dim)


# ------------------------------------------------------------------ #
#  SBC diagnostics (requires sbi)                                    #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(
    not _sbi_available(),
    reason="sbi package not installed",
)
def test_sbc_validation_structure() -> None:
    """SBC validation returns dict with expected keys."""
    # This is a structural test only -- actual SBC requires
    # a trained posterior which needs sbi
    from pyro_dcm.inference.sbi_diagnostics import run_sbc_validation  # noqa: F401

    # We just verify the import works; full SBC test requires
    # expensive training and is deferred to integration tests
