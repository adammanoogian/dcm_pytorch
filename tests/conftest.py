from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def hemo_params() -> dict[str, float]:
    """Default hemodynamic parameters matching SPM12 spm_fx_fmri.m.

    SPM12 CODE values (H vector):
        H(1) = 0.64  -- signal decay (kappa, s^-1)
        H(2) = 0.32  -- autoregulation (gamma, s^-1)
        H(3) = 2.00  -- transit time (tau, s)
        H(4) = 0.32  -- Grubb's exponent (alpha)
        H(5) = 0.40  -- resting oxygen extraction (E0)

    Note: These differ from Stephan et al. 2007 Table 1 paper values
    (kappa=0.65, gamma=0.41, tau=0.98, E0=0.34). We follow SPM12 code.
    """
    return {
        "kappa": 0.64,
        "gamma": 0.32,
        "tau": 2.0,
        "alpha": 0.32,
        "E0": 0.40,
    }


@pytest.fixture()
def test_A() -> torch.Tensor:
    """Small 3x3 A matrix with known stable eigenvalues.

    Diagonal elements are negative (self-inhibition), ensuring all
    eigenvalues have negative real parts for stability.
    """
    return torch.tensor(
        [
            [-0.5, 0.1, 0.0],
            [0.2, -0.5, 0.1],
            [0.0, 0.3, -0.5],
        ],
        dtype=torch.float64,
    )


@pytest.fixture()
def test_C() -> torch.Tensor:
    """Test C matrix (3 regions, 1 input).

    Only region 0 receives direct driving input.
    """
    return torch.tensor(
        [[1.0], [0.0], [0.0]],
        dtype=torch.float64,
    )


@pytest.fixture()
def device() -> torch.device:
    """Torch device for tests (CPU)."""
    return torch.device("cpu")


@pytest.fixture()
def dtype() -> torch.dtype:
    """Default dtype for numerical precision in tests."""
    return torch.float64
