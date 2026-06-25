"""Unit tests for the single-source CMC forward (``cmc_neural_mass.py``).

The permutation guard (CMC-02) is written FIRST: it is the load-bearing
correctness check that the free intrinsic parameter ``P.G`` column 0 drives the
``sp`` self-inhibition / precision knob ``G[:, 6]`` (MATLAB ``G(:,7)``) and NOT
``G[:, 0]`` -- the milestone-headline indexing trap (Fact 2). Plus the sigmoid
``-1/2`` baseline, time-constant units (ms -> s), the ``+exp(P.A)`` extrinsic
convention (NOT the fMRI ``-exp/2``), the zero steady state (M1), float64
(CMC-07), and the column-major flatten round-trip (Fact 1).
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.cmc_neural_mass import (
    cmc_f,
    cmc_flatten,
    cmc_sigmoid,
    cmc_unflatten,
    parameterize_cmc,
)

_F64 = torch.float64


def _baseline_p() -> dict[str, torch.Tensor]:
    """Single-source ``P`` struct with all log-scaling params at zero."""
    return {
        "T": torch.zeros(1, 4, dtype=_F64),
        "G": torch.zeros(1, 4, dtype=_F64),
        "C": torch.zeros(1, 1, dtype=_F64),
        "S": torch.zeros(1, 1, dtype=_F64),
    }


def test_permutation_guard() -> None:
    """Perturbing ``P_G[:, 0]`` changes ``G[:, 6]`` and leaves ``G[:, 0]`` fixed.

    CMC-02 / Fact 2: ``P.G`` column 0 -> ``G(:,7)`` MATLAB -> ``G[:, 6]`` Python
    (sp self-inhibition / precision). ``G[:, 0]`` (the ss->ss strength) is NOT a
    free parameter at the single-source level and must stay at its default.
    """
    p_base = _baseline_p()
    g_base = parameterize_cmc(p_base, n=1)["G"]

    p_pert = _baseline_p()
    p_pert["G"] = p_pert["G"].clone()
    p_pert["G"][:, 0] = p_pert["G"][:, 0] + 0.5
    g_pert = parameterize_cmc(p_pert, n=1)["G"]

    assert not torch.allclose(g_pert[:, 6], g_base[:, 6])
    assert torch.allclose(g_pert[:, 0], g_base[:, 0])


def test_sigmoid_baseline() -> None:
    """``cmc_sigmoid`` subtracts the ``-1/2`` baseline; ``R == 2/3`` at ``P_S=0``."""
    p_s = torch.zeros(1, 1, dtype=_F64)
    zero = torch.zeros(1, 8, dtype=_F64)
    assert torch.allclose(cmc_sigmoid(zero, p_s), torch.zeros(1, 8, dtype=_F64))

    # At P_S = 0, R = 2/3 exactly: check against the closed form at a nonzero x.
    x = torch.full((1, 8), 0.3, dtype=_F64)
    expected = 1.0 / (1.0 + torch.exp(-(2.0 / 3.0) * x)) - 0.5
    assert torch.allclose(cmc_sigmoid(x, p_s), expected)


def test_time_constant_units() -> None:
    """``parameterize_cmc`` returns T in SECONDS (T0_MS/1000 * exp(P.T))."""
    params = parameterize_cmc(_baseline_p(), n=1)
    # T0_MS[0] = 2 ms -> 0.002 s at P_T = 0.
    assert torch.allclose(params["T"][:, 0], torch.tensor([0.002], dtype=_F64))
    assert torch.allclose(params["T"][:, 3], torch.tensor([0.028], dtype=_F64))


def test_extrinsic_convention() -> None:
    """CMC uses ``+exp(P.A)`` (not the fMRI ``-exp/2``); A is zero at n=1."""
    import pyro_dcm.forward_models.cmc_neural_mass as cmc_mod

    # The fMRI parameterize_A convention must NOT leak into the CMC forward.
    src = cmc_mod.__file__
    with open(src, encoding="utf-8") as handle:
        text = handle.read()
    assert "parameterize_A" not in text

    a_blocks = parameterize_cmc(_baseline_p(), n=1)["A"]
    assert torch.allclose(a_blocks, torch.zeros_like(a_blocks))


def test_steady_state_zero() -> None:
    """``cmc_f(0, 0, P) == 0`` -- zero is the fixed point (M1, no Newton solve)."""
    x0 = torch.zeros(8, dtype=_F64)
    f0 = cmc_f(x0, torch.zeros(1, dtype=_F64), _baseline_p(), n=1)
    assert torch.allclose(f0, torch.zeros(8, dtype=_F64))


def test_float64() -> None:
    """``cmc_f`` returns float64 and rejects a float32 state (CMC-07 / N1)."""
    x0 = torch.zeros(8, dtype=_F64)
    f0 = cmc_f(x0, torch.zeros(1, dtype=_F64), _baseline_p(), n=1)
    assert f0.dtype == _F64

    with pytest.raises(TypeError):
        cmc_f(
            torch.zeros(8, dtype=torch.float32),
            torch.zeros(1, dtype=_F64),
            _baseline_p(),
            n=1,
        )


def test_column_major_flatten() -> None:
    """Column-major flatten round-trips and preserves state-blocked order."""
    x = torch.arange(8, dtype=_F64).reshape(1, 8)
    x_flat = cmc_flatten(x)
    assert torch.allclose(cmc_unflatten(x_flat, n=1), x)
    # At n=1 the flat vector is exactly the 8-state block in order.
    assert torch.allclose(x_flat, torch.arange(8, dtype=_F64))
