"""Tests for ELBO variant support in run_svi.

Verifies all 14 valid (guide, ELBO) combinations produce finite loss,
4 rejected combinations raise ``ValueError``, RenyiELBO smoke tests,
AutoLaplace post-processing, and backward compatibility.
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from pyro_dcm.models.guides import (
    ELBO_REGISTRY,
    GUIDE_REGISTRY,
    MEAN_FIELD_GUIDES,
    create_guide,
    run_svi,
)


# ------------------------------------------------------------------
# Minimal Pyro model for ELBO tests
# ------------------------------------------------------------------


def _toy_model() -> None:
    """Three-site model: a, b -> obs = Normal(a + b, 1)."""
    a = pyro.sample("a", dist.Normal(0.0, 1.0))
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    pyro.sample(
        "obs", dist.Normal(a + b, 1.0), obs=torch.tensor(3.0),
    )


# ------------------------------------------------------------------
# 1. All 14 valid (guide, ELBO) combinations
# ------------------------------------------------------------------

_ALL_GUIDE_TYPES = sorted(GUIDE_REGISTRY.keys())
_ALL_ELBO_TYPES = sorted(ELBO_REGISTRY.keys())

_VALID_COMBOS: list[tuple[str, str]] = [
    (g, e)
    for g in _ALL_GUIDE_TYPES
    for e in _ALL_ELBO_TYPES
    if not (e == "tracemeanfield_elbo" and g not in MEAN_FIELD_GUIDES)
]

_VALID_IDS = [f"{g}+{e}" for g, e in _VALID_COMBOS]


@pytest.mark.parametrize(
    ("guide_type", "elbo_type"),
    _VALID_COMBOS,
    ids=_VALID_IDS,
)
def test_valid_guide_elbo_combinations(
    guide_type: str,
    elbo_type: str,
) -> None:
    """Each valid (guide, ELBO) pair produces finite loss in 5 steps."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=5,
        lr=0.01,
        elbo_type=elbo_type,
        guide_type=guide_type,
    )
    assert math.isfinite(result["final_loss"]), (
        f"{guide_type}+{elbo_type} produced non-finite loss: "
        f"{result['final_loss']}"
    )
    assert len(result["losses"]) == 5


def test_valid_combos_count() -> None:
    """Exactly 14 valid (guide, ELBO) combinations exist."""
    assert len(_VALID_COMBOS) == 14, (
        f"Expected 14 valid combos, got {len(_VALID_COMBOS)}"
    )


# ------------------------------------------------------------------
# 2. All 4 rejected (guide, ELBO) combinations
# ------------------------------------------------------------------

_REJECTED_COMBOS: list[tuple[str, str]] = [
    (g, "tracemeanfield_elbo")
    for g in _ALL_GUIDE_TYPES
    if g not in MEAN_FIELD_GUIDES
]

_REJECTED_IDS = [f"{g}+tracemeanfield_elbo" for g, _ in _REJECTED_COMBOS]


@pytest.mark.parametrize(
    ("guide_type", "elbo_type"),
    _REJECTED_COMBOS,
    ids=_REJECTED_IDS,
)
def test_rejected_guide_elbo_combinations(
    guide_type: str,
    elbo_type: str,
) -> None:
    """Non-mean-field guide + TraceMeanField_ELBO raises ValueError."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    with pytest.raises(ValueError, match="TraceMeanField_ELBO"):
        run_svi(
            _toy_model,
            guide,
            model_args=(),
            num_steps=5,
            elbo_type=elbo_type,
            guide_type=guide_type,
        )


def test_rejected_combos_count() -> None:
    """Exactly 4 rejected (guide, ELBO) combinations exist."""
    assert len(_REJECTED_COMBOS) == 4, (
        f"Expected 4 rejected combos, got {len(_REJECTED_COMBOS)}"
    )


# ------------------------------------------------------------------
# 3. RenyiELBO smoke tests
# ------------------------------------------------------------------


def test_renyi_elbo_alpha_smoke() -> None:
    """RenyiELBO with alpha=0.5 completes without error."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_normal")
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=10,
        elbo_type="renyi_elbo",
        guide_type="auto_normal",
    )
    assert math.isfinite(result["final_loss"])
    assert len(result["losses"]) == 10


def test_renyi_elbo_min_particles() -> None:
    """RenyiELBO forces num_particles >= 2 even if 1 is requested."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_normal")
    # Pass num_particles=1; run_svi should bump to 2 internally
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=5,
        num_particles=1,
        elbo_type="renyi_elbo",
        guide_type="auto_normal",
    )
    assert math.isfinite(result["final_loss"])


# ------------------------------------------------------------------
# 4. AutoLaplace post-processing
# ------------------------------------------------------------------


def test_auto_laplace_returns_post_guide() -> None:
    """AutoLaplace run returns post-Laplace AutoMVN guide in result."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_laplace")
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=20,
        lr=0.01,
        elbo_type="trace_elbo",
        guide_type="auto_laplace",
    )
    assert "guide" in result, (
        "AutoLaplace result must contain 'guide' key"
    )
    assert isinstance(result["guide"], AutoMultivariateNormal), (
        f"Expected AutoMultivariateNormal, "
        f"got {type(result['guide']).__name__}"
    )


def test_non_laplace_guide_no_post_guide() -> None:
    """Non-Laplace guides do not include 'guide' in result dict."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_normal")
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=5,
        elbo_type="trace_elbo",
        guide_type="auto_normal",
    )
    assert "guide" not in result


# ------------------------------------------------------------------
# 5. Default elbo_type backward compatibility
# ------------------------------------------------------------------


def test_default_elbo_type_backward_compat() -> None:
    """run_svi with no elbo_type uses Trace_ELBO (backward compat)."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_normal")
    result = run_svi(
        _toy_model,
        guide,
        model_args=(),
        num_steps=5,
    )
    assert math.isfinite(result["final_loss"])
    assert len(result["losses"]) == 5
    # Default should not include a post-guide
    assert "guide" not in result


# ------------------------------------------------------------------
# 6. Invalid elbo_type
# ------------------------------------------------------------------


def test_invalid_elbo_type_raises_value_error() -> None:
    """Unknown elbo_type raises ValueError with valid keys."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type="auto_normal")
    with pytest.raises(ValueError, match="Unknown elbo_type"):
        run_svi(
            _toy_model,
            guide,
            model_args=(),
            num_steps=5,
            elbo_type="nonexistent",
        )
