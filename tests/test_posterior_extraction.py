"""Tests for Predictive-based posterior extraction from all guide types.

Verifies that ``extract_posterior_params`` works identically for all six
supported guide types, returns the backward-compatible ``'median'`` key,
produces ``std=0`` for ``AutoDelta``, and respects ``num_samples``.
"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import pytest
import torch

from pyro_dcm.models.guides import (
    GUIDE_REGISTRY,
    create_guide,
    extract_posterior_params,
    run_svi,
)


# ------------------------------------------------------------------
# Minimal Pyro model for extraction tests
# ------------------------------------------------------------------


def _simple_model(obs: torch.Tensor) -> None:
    """Two-site model: a, b -> obs = Normal(a + b, 1).

    Parameters
    ----------
    obs : torch.Tensor
        Observed value.
    """
    a = pyro.sample("a", dist.Normal(0.0, 1.0))
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    pyro.sample("obs", dist.Normal(a + b, 1.0), obs=obs)


_OBS = torch.tensor(3.0, dtype=torch.float64)
_MODEL_ARGS = (_OBS,)


# ------------------------------------------------------------------
# Helper: train a guide for a given type
# ------------------------------------------------------------------


def _train_guide(guide_type: str, num_steps: int = 30) -> tuple:
    """Train a guide and return (guide_or_post_guide, model_args).

    For ``auto_laplace``, returns the post-Laplace guide.

    Parameters
    ----------
    guide_type : str
        One of the keys in ``GUIDE_REGISTRY``.
    num_steps : int
        SVI steps to run.

    Returns
    -------
    tuple
        (trained guide, model_args tuple)
    """
    pyro.clear_param_store()
    guide = create_guide(
        _simple_model, guide_type=guide_type, init_scale=0.1,
    )
    result = run_svi(
        _simple_model,
        guide,
        _MODEL_ARGS,
        num_steps=num_steps,
        lr=0.01,
        elbo_type="trace_elbo",
        guide_type=guide_type,
    )
    # For auto_laplace, use the post-Laplace guide
    extract_guide = result.get("guide", guide)
    return extract_guide, _MODEL_ARGS


# ------------------------------------------------------------------
# 1. Parametrized extraction for all 6 guide types
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    sorted(GUIDE_REGISTRY.keys()),
    ids=sorted(GUIDE_REGISTRY.keys()),
)
def test_extract_posterior_all_guide_types(
    guide_type: str,
) -> None:
    """extract_posterior_params returns per-site mean/std/samples."""
    guide, model_args = _train_guide(guide_type)
    posterior = extract_posterior_params(
        guide, model_args, num_samples=50,
    )

    # Must have latent sites a and b
    for site in ("a", "b"):
        assert site in posterior, (
            f"Site {site!r} missing for guide_type={guide_type!r}"
        )
        info = posterior[site]
        assert "mean" in info
        assert "std" in info
        assert "samples" in info
        assert info["samples"].shape[0] == 50, (
            f"Expected 50 samples, got {info['samples'].shape[0]}"
        )
        assert torch.isfinite(info["mean"]).all()
        assert torch.isfinite(info["std"]).all()


# ------------------------------------------------------------------
# 2. AutoDelta std=0 test
# ------------------------------------------------------------------


def test_extract_posterior_auto_delta_std_zero() -> None:
    """AutoDelta extraction produces std=0 for all sites."""
    guide, model_args = _train_guide("auto_delta")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=100,
    )

    for site in ("a", "b"):
        std = posterior[site]["std"]
        assert torch.allclose(
            std, torch.zeros_like(std), atol=1e-6,
        ), (
            f"AutoDelta site {site!r} std should be 0, "
            f"got {std.item():.6f}"
        )


# ------------------------------------------------------------------
# 3. Backward-compatible median key test
# ------------------------------------------------------------------


def test_extract_posterior_median_key_backward_compat() -> None:
    """Extraction result has 'median' key mapping site names to means."""
    guide, model_args = _train_guide("auto_normal")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=50,
    )

    assert "median" in posterior
    median = posterior["median"]
    assert isinstance(median, dict)

    for site in ("a", "b"):
        assert site in median, (
            f"Site {site!r} missing from median dict"
        )
        # median[site] should match the per-site mean
        assert torch.allclose(
            median[site], posterior[site]["mean"], atol=1e-6,
        )


# ------------------------------------------------------------------
# 4. Sample-based quantiles test
# ------------------------------------------------------------------


def test_sample_based_quantiles() -> None:
    """Quantiles computed from samples bracket the posterior mean."""
    guide, model_args = _train_guide("auto_normal", num_steps=50)
    posterior = extract_posterior_params(
        guide, model_args, num_samples=500,
    )

    for site in ("a", "b"):
        samples = posterior[site]["samples"].float()
        q025 = torch.quantile(samples, 0.025, dim=0)
        q975 = torch.quantile(samples, 0.975, dim=0)
        mean = posterior[site]["mean"]

        # Mean must be within the 95% CI
        assert q025 <= mean <= q975, (
            f"Site {site!r}: mean={mean.item():.4f} not in "
            f"[{q025.item():.4f}, {q975.item():.4f}]"
        )

        # CI width should be positive (non-degenerate)
        assert (q975 - q025) > 0, (
            f"Site {site!r}: CI width is zero"
        )


# ------------------------------------------------------------------
# 5. num_samples parameter test
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [10, 100, 500])
def test_num_samples_controls_sample_count(
    n_samples: int,
) -> None:
    """num_samples parameter controls number of posterior samples."""
    guide, model_args = _train_guide("auto_normal")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=n_samples,
    )

    for site in ("a", "b"):
        assert posterior[site]["samples"].shape[0] == n_samples, (
            f"Expected {n_samples} samples for {site!r}, "
            f"got {posterior[site]['samples'].shape[0]}"
        )


# ------------------------------------------------------------------
# 6. AutoLaplace extraction test
# ------------------------------------------------------------------


def test_extract_posterior_auto_laplace() -> None:
    """AutoLaplace post-guide produces non-zero std."""
    guide, model_args = _train_guide("auto_laplace")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=200,
    )

    for site in ("a", "b"):
        assert site in posterior
        std = posterior[site]["std"]
        # Post-Laplace approximation should have non-zero uncertainty
        assert std.item() > 0, (
            f"AutoLaplace site {site!r} std should be >0, "
            f"got {std.item():.6f}"
        )
