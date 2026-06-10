"""Task DCM precision-matrix intractability guard tests (VLROBUST-02).

Guards ``TaskDCMForward.build_precision`` against the dense ``(T*N, T*N)``
precision blow-up at fine ``dt`` / long duration (pitfall N1). The tractable
path must return the identity precision unchanged; an oversized ``T*N`` must
fail LOUD with an expected-vs-actual matrix size and the ``dt >= 0.1`` hint.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.inference.forward_models import TaskDCMForward


def _make_forward() -> TaskDCMForward:
    """Build a small task-DCM forward model (N=3, M=1, dt=0.1)."""
    return TaskDCMForward(
        stimulus_fn=lambda t_: torch.zeros(1, dtype=torch.float64),
        c_mask=torch.ones(3, 1, dtype=torch.float64),
        t_eval=torch.arange(0, 10, 0.1, dtype=torch.float64),
    )


@pytest.mark.vl
def test_tractable_precision_returns_identity() -> None:
    """A tractable T*N returns the (ny, ny) identity precision unchanged."""
    forward = _make_forward()
    observed = torch.zeros(60, dtype=torch.float64)

    q_list, nq = forward.build_precision(observed)

    assert nq == 1
    assert len(q_list) == 1
    assert q_list[0].shape == (60, 60)
    assert torch.equal(q_list[0], torch.eye(60, dtype=torch.float64))


@pytest.mark.vl
def test_intractable_precision_raises_with_expected_vs_actual() -> None:
    """An oversized T*N raises ValueError with expected-vs-actual + dt hint."""
    forward = _make_forward()
    observed = torch.zeros(6000, dtype=torch.float64)

    with pytest.raises(ValueError) as exc_info:
        forward.build_precision(observed)

    message = str(exc_info.value)
    assert "6000" in message  # actual size
    assert "5000" in message  # expected cap
    assert "dt >= 0.1" in message  # the floor hint
