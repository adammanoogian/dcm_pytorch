"""Neural state equation for Dynamic Causal Modeling.

Implements the bilinear neural state equation from [REF-001] Eq. 1
(Friston, Harrison & Penny, 2003).

This module provides the linear form dx/dt = Ax + Cu (without modulatory
B matrices, which are deferred to a later extension). The A matrix
parameterization follows SPM12 spm_fx_fmri.m conventions, guaranteeing
negative self-connections via the transform a_ii = -exp(A_free_ii) / 2.
"""

from __future__ import annotations

import torch


def parameterize_A(A_free: torch.Tensor) -> torch.Tensor:
    """Convert free parameters to effective connectivity matrix.

    Implements SPM12 A matrix parameterization convention [REF-001]:
    - Diagonal: ``a_ii = -exp(A_free_ii) / 2`` (guarantees negative
      self-connections, default -0.5 Hz when free parameter is 0)
    - Off-diagonal: ``a_ij = A_free_ij`` (unconstrained, in Hz)

    Parameters
    ----------
    A_free : torch.Tensor
        Free (unconstrained) parameters, shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix A, shape ``(N, N)``, with
        guaranteed negative diagonal.

    Notes
    -----
    SPM12 source: ``spm_fx_fmri.m`` lines for self-inhibition.
    See also SPM/DCM_units wikibook for A matrix conventions.

    Examples
    --------
    >>> import torch
    >>> A_free = torch.zeros(2, 2, dtype=torch.float64)
    >>> A = parameterize_A(A_free)
    >>> A.diagonal()  # tensor([-0.5000, -0.5000])
    """
    N = A_free.shape[0]
    diag_mask = torch.eye(N, dtype=torch.bool, device=A_free.device)
    A = A_free.clone()
    A[diag_mask] = -torch.exp(A_free[diag_mask]) / 2.0
    return A


class NeuralStateEquation:
    """Bilinear neural state equation dx/dt = Ax + Cu.

    Implements [REF-001] Eq. 1 (Friston, Harrison & Penny, 2003),
    restricted to the linear case (no modulatory B matrices).

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``.
        Diagonal elements should be negative for stability.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.

    Attributes
    ----------
    A : torch.Tensor
        Effective connectivity matrix.
    C : torch.Tensor
        Driving input weights.

    Examples
    --------
    >>> A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]])
    >>> C = torch.tensor([[1.0], [0.0]])
    >>> nse = NeuralStateEquation(A, C)
    >>> x = torch.tensor([0.1, 0.0])
    >>> u = torch.tensor([1.0])
    >>> dx = nse.derivatives(x, u)
    """

    def __init__(self, A: torch.Tensor, C: torch.Tensor) -> None:
        self.A = A
        self.C = C

    def derivatives(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute neural state derivatives dx/dt = Ax + Cu.

        Implements [REF-001] Eq. 1 (linear form).

        Parameters
        ----------
        x : torch.Tensor
            Neural activity per region, shape ``(N,)``.
        u : torch.Tensor
            Experimental input, shape ``(M,)``.

        Returns
        -------
        torch.Tensor
            Time derivatives of neural activity, shape ``(N,)``.
        """
        return self.A @ x + self.C @ u
