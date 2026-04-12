"""Tests for the create_guide factory in guides.py.

Verifies that all six supported Pyro AutoGuide types can be
instantiated via ``create_guide``, that init_scale asymmetry is
handled correctly, that the N-based blocklist raises for dangerous
configurations, and that each guide can complete a single SVI step.
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from pyro.optim import Adam

from pyro_dcm.models.guides import (
    GUIDE_REGISTRY,
    MEAN_FIELD_GUIDES,
    create_guide,
)


# ------------------------------------------------------------------
# Minimal Pyro model used across all tests
# ------------------------------------------------------------------


def _toy_model() -> None:
    """Three-site model: a, b -> obs = Normal(a + b, 1)."""
    a = pyro.sample("a", dist.Normal(0.0, 1.0))
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    pyro.sample("obs", dist.Normal(a + b, 1.0), obs=torch.tensor(3.0))


# ------------------------------------------------------------------
# 1. Parametrized instantiation
# ------------------------------------------------------------------

_GUIDE_TYPE_CLASS = [
    ("auto_delta", AutoDelta),
    ("auto_normal", AutoNormal),
    ("auto_lowrank_mvn", AutoLowRankMultivariateNormal),
    ("auto_mvn", AutoMultivariateNormal),
    ("auto_iaf", AutoIAFNormal),
    ("auto_laplace", AutoLaplaceApproximation),
]


@pytest.mark.parametrize(
    ("guide_type", "expected_cls"),
    _GUIDE_TYPE_CLASS,
    ids=[g for g, _ in _GUIDE_TYPE_CLASS],
)
def test_create_guide_returns_correct_type(
    guide_type: str,
    expected_cls: type,
) -> None:
    """create_guide returns the correct AutoGuide subclass."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    assert isinstance(guide, expected_cls), (
        f"Expected {expected_cls.__name__}, "
        f"got {type(guide).__name__}"
    )


# ------------------------------------------------------------------
# 2. Default backward compatibility
# ------------------------------------------------------------------


def test_create_guide_default_is_auto_normal() -> None:
    """create_guide with no guide_type returns AutoNormal."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model)
    assert isinstance(guide, AutoNormal)


def test_create_guide_with_init_scale_only() -> None:
    """create_guide(model, init_scale=0.01) returns AutoNormal."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, init_scale=0.01)
    assert isinstance(guide, AutoNormal)


# ------------------------------------------------------------------
# 3. init_scale asymmetry
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    ["auto_delta", "auto_laplace", "auto_iaf"],
)
def test_init_scale_not_passed_to_unsupported_guides(
    guide_type: str,
) -> None:
    """init_scale is silently ignored for guides that don't accept it."""
    pyro.clear_param_store()
    # Should NOT raise even though init_scale is provided
    guide = create_guide(
        _toy_model, guide_type=guide_type, init_scale=0.01,
    )
    assert guide is not None


# ------------------------------------------------------------------
# 4. Blocklist enforcement
# ------------------------------------------------------------------


def test_auto_mvn_blocked_at_high_n_regions() -> None:
    """auto_mvn raises ValueError when n_regions >= 8."""
    pyro.clear_param_store()
    with pytest.raises(ValueError, match="auto_lowrank_mvn"):
        create_guide(
            _toy_model, guide_type="auto_mvn", n_regions=8,
        )


def test_auto_mvn_allowed_at_max_n_regions() -> None:
    """auto_mvn is allowed when n_regions == 7 (the max)."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model, guide_type="auto_mvn", n_regions=7,
    )
    assert isinstance(guide, AutoMultivariateNormal)


# ------------------------------------------------------------------
# 5. Invalid guide_type
# ------------------------------------------------------------------


def test_invalid_guide_type_raises_value_error() -> None:
    """Unknown guide_type raises ValueError with valid keys."""
    pyro.clear_param_store()
    with pytest.raises(ValueError, match="auto_normal"):
        create_guide(_toy_model, guide_type="nonexistent")


# ------------------------------------------------------------------
# 6. kwargs passthrough
# ------------------------------------------------------------------


def test_kwargs_passthrough_lowrank() -> None:
    """rank kwarg overrides default for auto_lowrank_mvn."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model, guide_type="auto_lowrank_mvn", rank=3,
    )
    assert isinstance(guide, AutoLowRankMultivariateNormal)


def test_kwargs_passthrough_iaf() -> None:
    """hidden_dim and num_transforms kwargs for auto_iaf."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model,
        guide_type="auto_iaf",
        hidden_dim=32,
        num_transforms=3,
    )
    assert isinstance(guide, AutoIAFNormal)


# ------------------------------------------------------------------
# 7. SVI smoke test per guide type
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    list(GUIDE_REGISTRY.keys()),
    ids=list(GUIDE_REGISTRY.keys()),
)
def test_svi_step_with_each_guide_type(guide_type: str) -> None:
    """Each guide type can complete 1 SVI step with finite loss."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    optimizer = Adam({"lr": 0.01})
    elbo = Trace_ELBO()
    svi = SVI(_toy_model, guide, optimizer, loss=elbo)

    loss = svi.step()
    assert math.isfinite(loss), (
        f"guide_type={guide_type!r} produced non-finite loss: {loss}"
    )


# ------------------------------------------------------------------
# Registry sanity checks
# ------------------------------------------------------------------


def test_guide_registry_has_six_entries() -> None:
    """GUIDE_REGISTRY contains exactly 6 guide types."""
    assert len(GUIDE_REGISTRY) == 6


def test_mean_field_guides_subset() -> None:
    """MEAN_FIELD_GUIDES is a subset of GUIDE_REGISTRY keys."""
    assert MEAN_FIELD_GUIDES <= set(GUIDE_REGISTRY.keys())
    assert MEAN_FIELD_GUIDES == {"auto_delta", "auto_normal"}
