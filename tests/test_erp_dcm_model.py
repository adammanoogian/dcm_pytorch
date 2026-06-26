"""Structural + SVI-smoke tests for ``erp_dcm_model`` (ERPDCM-01).

Verifies the Pyro generative ERP-DCM model:

1. Structural trace -- a ``pyro.poutine.trace`` over a small 2-source net
   discovers the expected sample sites ``A_free``, ``B_free_0``, ``C_free``,
   ``T``, ``G``, ``S``, ``R``, ``scalp_noise_scale``, ``obs_erp`` (the
   per-effect ``B_free_{j}`` loop with NO ``pyro.plate`` is what makes the
   AutoGuide auto-discover ``B``, MODEL-06).
2. ``create_guide`` auto-discovery -- ``AutoNormal`` traces the model with ZERO
   factory edits.
3. SVI-smoke -- a few steps on a tiny simulated draw yield finite ELBO.

All tiny + LAPTOP (<30s slice). The real (jacrev x many-step) amortized fit
lives in ``test_amortized_erp.py`` and is marked ``slow`` (routes to M3).
"""

from __future__ import annotations

import math

import pyro
import pyro.poutine
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from pyro_dcm.forward_models import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
)
from pyro_dcm.models import create_guide, erp_dcm_model

_F64 = torch.float64
_NS = 16  # tiny peristimulus length keeps the SVI-smoke fast (<30s)


def _small_net() -> dict[str, object]:
    """Build a tiny 2-source / 1-input / 2-condition ERP-DCM problem."""
    n, m = 2, 1
    # One live forward edge sp->ss at (to=1, from=0); other blocks empty.
    block0 = torch.zeros(n, n, dtype=_F64)
    block0[1, 0] = 1.0
    a_masks = [
        block0,
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
    ]
    # Between-trial B: live lower-triangle positions (one effect).
    b_masks = [torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=_F64)]
    c_mask = torch.ones(n, m, dtype=_F64)
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))
    return {
        "n": n,
        "a_masks": a_masks,
        "b_masks": b_masks,
        "c_mask": c_mask,
        "x_design": x_design,
        "l_full": l_full,
    }


def _model_args(net: dict[str, object], observed: torch.Tensor) -> tuple:
    """Positional args for ``erp_dcm_model`` from a net dict + observed scalp."""
    return (
        observed,
        net["a_masks"],
        net["b_masks"],
        net["c_mask"],
        net["x_design"],
        net["l_full"],
        net["n"],
    )


def _planted_scalp(net: dict[str, object], seed: int = 0) -> torch.Tensor:
    """Simulate an observed scalp ERP from a single seeded prior draw.

    Runs the model forward once (conditioned on a zero placeholder) and reads
    the ``predicted_scalp`` deterministic -- which depends only on the sampled
    latents, giving a realistic non-trivial target for the SVI-smoke.
    """
    nc = net["l_full"].shape[0]
    placeholder = torch.zeros(2, _NS, nc, dtype=_F64)
    seeded = pyro.poutine.seed(erp_dcm_model, rng_seed=seed)
    trace = pyro.poutine.trace(seeded).get_trace(
        *_model_args(net, placeholder),
        ns=_NS,
    )
    return trace.nodes["predicted_scalp"]["value"].detach()


def test_structural_trace_discovers_all_sites() -> None:
    """The model trace exposes every named sample site (incl. B_free_0)."""
    net = _small_net()
    observed = _planted_scalp(net)
    trace = pyro.poutine.trace(erp_dcm_model).get_trace(
        *_model_args(net, observed),
        ns=_NS,
    )
    expected = {
        "A_free",
        "B_free_0",
        "C_free",
        "T",
        "G",
        "S",
        "R",
        "scalp_noise_scale",
        "obs_erp",
    }
    sample_sites = {
        name for name, node in trace.nodes.items() if node["type"] == "sample"
    }
    assert expected <= sample_sites, expected - sample_sites
    # The deterministic forward output is also exposed.
    assert "predicted_scalp" in trace.nodes


def test_autonormal_auto_discovers_sites() -> None:
    """create_guide(auto_normal) traces the model with ZERO factory edits."""
    net = _small_net()
    observed = _planted_scalp(net)
    pyro.clear_param_store()
    guide = create_guide(erp_dcm_model, guide_type="auto_normal")
    # Calling the guide traces the model and registers the latent sites.
    guide(*_model_args(net, observed), ns=_NS)
    guide_trace = pyro.poutine.trace(guide).get_trace(
        *_model_args(net, observed),
        ns=_NS,
    )
    guide_sites = {
        name
        for name, node in guide_trace.nodes.items()
        if node["type"] == "sample" and not node["is_observed"]
    }
    # AutoNormal auto-discovered the latent (non-observed) sites, including the
    # loop-sampled B_free_0 (MODEL-06).
    for site in ("A_free", "B_free_0", "C_free", "T", "G", "S", "R"):
        assert site in guide_sites, site


def test_svi_smoke_finite_elbo() -> None:
    """A few SVI steps on a tiny simulated draw keep the ELBO finite."""
    net = _small_net()
    observed = _planted_scalp(net)
    pyro.clear_param_store()
    guide = create_guide(erp_dcm_model, guide_type="auto_normal")
    svi = SVI(erp_dcm_model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())
    losses = [svi.step(*_model_args(net, observed), ns=_NS) for _ in range(3)]
    assert all(math.isfinite(val) for val in losses), losses


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
