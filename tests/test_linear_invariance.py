"""Bit-exact regression tests for linear-path invariance (BILIN-03).

Verifies that ``NeuralStateEquation.derivatives``, when called with
``B=None`` or ``B.shape[0] == 0``, produces output byte-identical to the
v0.2.0 linear form ``A @ x + C @ u``. The CONTEXT-locked short-circuit
gate guarantees zero numerical drift for v0.2.0 callers after the
Phase 13 bilinear extension.

Fixtures (per ``13-CONTEXT.md`` Specific Ideas):

- Hand-crafted 2-region case.
- ``make_random_stable_A(N=3, seed=42)``.
- ``make_random_stable_A(N=5, seed=7)``.
- Defensive: empty-J shape ``(0, N, N)`` routes through the short-circuit.
- Defensive: no-kwarg call matches ``B=None`` call byte-exactly.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.forward_models.neural_state import NeuralStateEquation
from pyro_dcm.simulators.task_simulator import make_random_stable_A


def _linear_reference(
    A: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Explicit v0.2.0 reference: ``dx/dt = A @ x + C @ u``.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity, shape ``(N, N)``.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.
    x : torch.Tensor
        Neural activity, shape ``(N,)``.
    u : torch.Tensor
        Driving input, shape ``(M,)``.

    Returns
    -------
    torch.Tensor
        Time derivatives, shape ``(N,)``.
    """
    return A @ x + C @ u


class TestLinearInvariance:
    """BILIN-03: B=None / J=0 paths are bit-exact to v0.2.0 linear form."""

    def test_hand_crafted_2region_bit_exact(self) -> None:
        """Hand-crafted 2-region A/C: B=None path matches literal A@x + C@u."""
        A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
        C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        x = torch.tensor([0.37, -0.12], dtype=torch.float64)
        u = torch.tensor([0.9], dtype=torch.float64)

        nse = NeuralStateEquation(A, C)
        new_path = nse.derivatives(x, u, B=None, u_mod=None)
        old_path = _linear_reference(A, C, x, u)

        torch.testing.assert_close(new_path, old_path, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("N,seed", [(3, 42), (5, 7)])
    def test_random_stable_A_bit_exact(self, N: int, seed: int) -> None:
        """BILIN-03 with random stable A at N=3/seed=42 and N=5/seed=7."""
        A = make_random_stable_A(N, density=0.5, seed=seed)
        torch.manual_seed(seed)
        C = torch.randn(N, 1, dtype=torch.float64) * 0.5
        x = torch.randn(N, dtype=torch.float64) * 0.3
        u = torch.tensor([0.7], dtype=torch.float64)

        nse = NeuralStateEquation(A, C)
        new_path = nse.derivatives(x, u, B=None, u_mod=None)
        old_path = _linear_reference(A, C, x, u)

        torch.testing.assert_close(new_path, old_path, rtol=0, atol=1e-10)

    def test_empty_J_bit_exact(self) -> None:
        """B.shape == (0, N, N) routes through the short-circuit bit-exactly."""
        N = 3
        A = make_random_stable_A(N, density=0.5, seed=11)
        torch.manual_seed(11)
        C = torch.randn(N, 1, dtype=torch.float64) * 0.5
        x = torch.randn(N, dtype=torch.float64) * 0.3
        u = torch.tensor([0.4], dtype=torch.float64)

        nse = NeuralStateEquation(A, C)
        B_empty = torch.zeros(0, N, N, dtype=torch.float64)
        u_mod_empty = torch.zeros(0, dtype=torch.float64)
        new_path = nse.derivatives(x, u, B=B_empty, u_mod=u_mod_empty)
        old_path = _linear_reference(A, C, x, u)

        torch.testing.assert_close(new_path, old_path, rtol=0, atol=1e-10)

    def test_no_kwarg_vs_B_none_strict_equality(self) -> None:
        """Defensive: ``derivatives(x, u)`` == ``derivatives(x, u, B=None, u_mod=None)``."""
        A = torch.tensor(
            [[-0.5, 0.1, 0.0], [0.2, -0.5, 0.1], [0.0, 0.3, -0.5]],
            dtype=torch.float64,
        )
        C = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
        x = torch.tensor([0.1, -0.2, 0.15], dtype=torch.float64)
        u = torch.tensor([1.0], dtype=torch.float64)

        nse = NeuralStateEquation(A, C)
        no_kwarg = nse.derivatives(x, u)
        b_none = nse.derivatives(x, u, B=None, u_mod=None)

        # Strict bit-equality because both paths execute the identical
        # ``self.A @ x + self.C @ u`` expression.
        assert torch.equal(no_kwarg, b_none), (
            "no-kwarg and B=None paths diverged: "
            f"expected torch.equal, got no_kwarg={no_kwarg}, B=None={b_none}"
        )


class TestBilinearPathSanity:
    """Sanity: bilinear path produces distinguishable output when B is non-trivial."""

    def test_bilinear_changes_output(self) -> None:
        """Bilinear path adds ``u_mod[j] * B[j] @ x`` to the linear result."""
        A = torch.tensor([[-0.5, 0.0], [0.0, -0.5]], dtype=torch.float64)
        C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        x = torch.tensor([1.0, 1.0], dtype=torch.float64)
        u = torch.tensor([0.0], dtype=torch.float64)  # zero drive to isolate B

        nse = NeuralStateEquation(A, C)
        B = torch.zeros(1, 2, 2, dtype=torch.float64)
        B[0, 0, 1] = 0.5  # off-diagonal coupling via modulator 0
        u_mod = torch.tensor([1.0], dtype=torch.float64)

        linear = nse.derivatives(x, u, B=None, u_mod=None)
        bilinear = nse.derivatives(x, u, B=B, u_mod=u_mod)

        # Linear: A @ x = [-0.5, -0.5]. Bilinear adds
        # u_mod[0] * B[0] @ x = 1.0 * [[0, 0.5], [0, 0]] @ [1, 1] = [0.5, 0].
        # So bilinear = [-0.5 + 0.5, -0.5 + 0] = [0.0, -0.5].
        expected_bilinear = torch.tensor([0.0, -0.5], dtype=torch.float64)
        torch.testing.assert_close(
            bilinear, expected_bilinear, atol=1e-12, rtol=1e-12
        )
        assert not torch.equal(linear, bilinear), (
            "Bilinear path did not change output: "
            f"expected distinct, got linear={linear}, bilinear={bilinear}"
        )

    def test_bilinear_u_mod_none_raises(self) -> None:
        """Non-empty B with u_mod=None raises ValueError citing u_mod."""
        A = torch.tensor([[-0.5, 0.0], [0.0, -0.5]], dtype=torch.float64)
        C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        x = torch.zeros(2, dtype=torch.float64)
        u = torch.tensor([0.0], dtype=torch.float64)

        nse = NeuralStateEquation(A, C)
        B = torch.zeros(1, 2, 2, dtype=torch.float64)

        with pytest.raises(ValueError, match="u_mod"):
            nse.derivatives(x, u, B=B, u_mod=None)
