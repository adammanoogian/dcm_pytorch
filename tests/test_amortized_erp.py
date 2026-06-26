"""ERPDCMPacker round-trip + amortized-flow smoke tests (ERPDCM-02).

1. ``ERPDCMPacker`` round-trips ``unpack(pack(x)) == x`` bit-for-bit as identity
   reshapes (no exp/log) and matches ``ERPDCMForward.param_count`` -- LAPTOP.
2. The amortized flow guide trains on a handful of simulated draws WITHOUT
   error (``B`` held FIXED inside ``ERPDCMForward``) -- the jacrev forward x
   several SVI steps is the >30s path, so it is marked ``slow`` and routes to
   M3 (``cluster/sbatch/erp_pytest.sbatch`` with
   ``TEST_TARGET=tests/test_amortized_erp.py``).
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.forward_models import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
)
from pyro_dcm.guides import AmortizedFlowGuide, ERPDCMPacker, ErpSummaryNet
from pyro_dcm.inference.forward_models import ERPDCMForward
from pyro_dcm.models.amortized_wrappers import amortized_erp_dcm_model

_F64 = torch.float64
_NS = 16


def _net() -> dict[str, object]:
    """Build a tiny 2-source / 1-input / 2-condition ERP problem + forward."""
    n, m = 2, 1
    block0 = torch.zeros(n, n, dtype=_F64)
    block0[1, 0] = 1.0
    a_masks = [
        block0,
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
    ]
    # FIXED between-trial B value matrix (NOT a free param in the amortized path).
    b_masks = [torch.tensor([[0.3, 0.0], [0.2, 0.5]], dtype=_F64)]
    c_mask = torch.ones(n, m, dtype=_F64)
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))
    forward = ERPDCMForward(
        l_full=l_full,
        x_design=x_design,
        a_masks=a_masks,
        b_masks=b_masks,
        c_mask=c_mask,
        dt=0.004,
        ns=_NS,
    )
    return {
        "n": n,
        "m": m,
        "a_masks": a_masks,
        "b_masks": b_masks,
        "c_mask": c_mask,
        "x_design": x_design,
        "l_full": l_full,
        "forward": forward,
    }


def _random_param_dict(n: int, m: int, seed: int) -> dict[str, torch.Tensor]:
    """Build a random CMC free-param dict in the ERPDCMPacker layout."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape: int) -> torch.Tensor:
        return 0.1 * torch.randn(*shape, generator=g, dtype=_F64)

    return {
        "A_free": rn(4, n, n),
        "C_free": rn(n, m),
        "T": rn(n, 4),
        "G": rn(n, 4),
        "S": rn(n, 1),
        "R": rn(m, 2),
    }


def test_packer_round_trip_identity() -> None:
    """unpack(pack(x)) == x bit-for-bit (identity reshapes, no exp/log)."""
    n, m = 2, 1
    packer = ERPDCMPacker(N=n, M=m)
    d = _random_param_dict(n, m, seed=0)
    z = packer.pack(d)
    d2 = packer.unpack(z)
    for key in d:
        assert torch.equal(d[key], d2[key]), key
    # Vector round-trip too: pack(unpack(z)) == z.
    z_rand = torch.randn(packer.n_features, dtype=_F64)
    assert torch.equal(packer.pack(packer.unpack(z_rand)), z_rand)


def test_packer_n_features_matches_forward() -> None:
    """ERPDCMPacker.n_features == ERPDCMForward.param_count (frozen order)."""
    net = _net()
    packer = ERPDCMPacker(N=net["n"], M=net["m"])
    assert packer.n_features == net["forward"].param_count(net["n"])
    # Explicit arithmetic guard (4NN + NM + 4N + 4N + N + 2M at N=2, M=1).
    assert packer.n_features == 4 * 4 + 2 + 8 + 8 + 2 + 2


def test_packer_order_matches_forward_pack_params() -> None:
    """The packed vector equals ERPDCMForward.pack_params element-for-element."""
    net = _net()
    packer = ERPDCMPacker(N=net["n"], M=net["m"])
    d = _random_param_dict(net["n"], net["m"], seed=1)
    assert torch.equal(packer.pack(d), net["forward"].pack_params(**d))


@pytest.mark.slow
def test_amortized_flow_trains_without_error() -> None:
    """The amortized flow guide runs a few SVI steps without error (B FIXED).

    M3-routed (jacrev forward x several steps > 30s). Mirrors the
    ``amortized_task_dcm_model`` smoke: fit packer standardization on a handful
    of prior draws, build ``AmortizedFlowGuide(ErpSummaryNet(...), ...)``, and
    step a few times asserting a finite loss.
    """
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam

    net = _net()
    n, m = net["n"], net["m"]
    forward = net["forward"]
    packer = ERPDCMPacker(N=n, M=m)
    dataset = [_random_param_dict(n, m, seed=s) for s in range(8)]
    packer.fit_standardization(dataset)

    nc = net["l_full"].shape[0]
    # Simulated observed scalp from one packed draw through the FIXED forward.
    theta0 = forward.pack_params(**dataset[0])
    placeholder = torch.zeros(2, _NS, nc, dtype=_F64)
    observed = torch.nan_to_num(forward.predict(theta0, placeholder, n)).reshape(
        2, _NS, nc
    )

    summary = ErpSummaryNet(n_cond=2, ns=_NS, n_channels=nc, embed_dim=32)
    guide = AmortizedFlowGuide(
        summary,
        packer.n_features,
        embed_dim=32,
        packer=packer,
    )

    pyro.clear_param_store()
    svi = SVI(
        amortized_erp_dcm_model,
        guide,
        Adam({"lr": 1e-3}),
        loss=Trace_ELBO(),
    )
    args = (
        observed,
        net["a_masks"],
        net["b_masks"],
        net["c_mask"],
        net["x_design"],
        net["l_full"],
        forward,
        packer,
    )
    losses = [svi.step(*args) for _ in range(3)]
    assert all(math.isfinite(val) for val in losses), losses


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
