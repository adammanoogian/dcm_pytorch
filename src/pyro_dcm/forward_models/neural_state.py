"""Neural state equation for DCM (Friston, Harrison & Penny 2003).

Implements the **linear form** ``dx/dt = Ax + Cu`` of the [REF-001] Eq. 1
neural state equation. The full bilinear form
``dx/dt = (A + Sigma_j u_j * B_j) * x + Cu`` -- which gives this module its
historical name -- is supported as an opt-in path via the ``B`` / ``u_mod``
arguments of ``NeuralStateEquation.derivatives`` (added in v0.3.0) and the
``parameterize_B`` / ``compute_effective_A`` utilities in this module.

Callers who do not pass ``B`` see bit-exact v0.2.0 linear behavior.

The A matrix parameterization follows SPM12 ``spm_fx_fmri.m`` conventions,
guaranteeing negative self-connections via the transform
``a_ii = -exp(A_free_ii) / 2``.
"""

from __future__ import annotations

import warnings

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


def parameterize_B(
    B_free: torch.Tensor,
    b_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert free B parameters to masked modulatory B matrices.

    Implements the v0.3.0 bilinear-extension B factory [REF-001] Eq. 1
    (Friston et al. 2003). Applies the binary mask ``b_mask`` elementwise
    to the free parameters ``B_free``. The recommended default mask zeros
    the diagonal (self-modulation disabled) to avoid the A_eff stability
    failure mode (Pitfall B5 per v0.3.0 research notes).

    Parameters
    ----------
    B_free : torch.Tensor
        Free (unconstrained) modulatory parameters, shape ``(J, N, N)``.
        ``J`` is the number of modulatory inputs, ``N`` is the number of
        regions.
    b_mask : torch.Tensor
        Binary mask, shape ``(J, N, N)``. Entries in ``{0, 1}``. Diagonal
        entries ``b_mask[:, i, i]`` should be zero unless the caller
        explicitly opts in to self-modulation (discouraged; see Warns).

    Returns
    -------
    torch.Tensor
        Masked modulatory matrices ``B = B_free * b_mask``, shape
        ``(J, N, N)``.

    Warns
    -----
    DeprecationWarning
        If any diagonal element of ``b_mask`` is non-zero. Self-modulation
        via the B path can push ``max Re(eig(A_eff))`` positive under the
        ``N(0, 1.0)`` prior (D1, Pitfall B5). A future
        ``parameterize_B_safe_diag`` (v0.4+) may add a guaranteed-stable
        transform; until then, callers should keep the diagonal zero.

    Raises
    ------
    ValueError
        If ``B_free`` and ``b_mask`` shapes differ, or if either tensor is
        not 3-D of shape ``(J, N, N)``.

    Notes
    -----
    Off-diagonal elements pass through via pure mask multiplication. No
    ``-exp`` transform and no ``tanh`` bounding: the ``N(0, 1.0)`` prior
    (D1) performs regularization, not the factory.

    Examples
    --------
    >>> import torch
    >>> B_free = torch.randn(2, 3, 3, dtype=torch.float64)
    >>> b_mask = torch.ones(2, 3, 3, dtype=torch.float64)
    >>> # Zero diagonal per recommended default
    >>> for j in range(2):
    ...     b_mask[j].fill_diagonal_(0.0)
    >>> B = parameterize_B(B_free, b_mask)
    >>> B.shape
    torch.Size([2, 3, 3])
    """
    if B_free.shape != b_mask.shape:
        raise ValueError(
            "parameterize_B shape mismatch: expected B_free.shape == "
            f"b_mask.shape, got B_free.shape={tuple(B_free.shape)} and "
            f"b_mask.shape={tuple(b_mask.shape)}."
        )
    if B_free.ndim != 3:
        raise ValueError(
            "parameterize_B expects 3-D stacked tensors (J, N, N); "
            f"got B_free.ndim={B_free.ndim} with shape {tuple(B_free.shape)}."
        )

    # DeprecationWarning if any diagonal mask entry is non-zero (Pitfall B5).
    if b_mask.shape[0] > 0:
        _, N, _ = b_mask.shape
        diag_idx = torch.arange(N, device=b_mask.device)
        diag_entries = b_mask[:, diag_idx, diag_idx]  # shape (J, N)
        if (diag_entries != 0).any():
            warnings.warn(
                "parameterize_B received a b_mask with non-zero diagonal "
                "entries. Self-modulation via B can push "
                "max Re(eig(A_eff)) > 0 under the N(0, 1.0) prior (D1; "
                "Pitfall B5). Consider zeroing b_mask[:, i, i]. A future "
                "parameterize_B_safe_diag (v0.4+) may provide a "
                "guaranteed-stable diagonal transform.",
                DeprecationWarning,
                stacklevel=2,
            )

    return B_free * b_mask


def compute_effective_A(
    A: torch.Tensor,
    B: torch.Tensor,
    u_mod: torch.Tensor,
) -> torch.Tensor:
    """Compose the effective connectivity ``A_eff = A + sum_j u_j * B_j``.

    Implements [REF-001] Eq. 1 (Friston et al. 2003) modulator composition
    for the task-DCM bilinear extension. Uses ``torch.einsum`` for the
    sum over the modulator index.

    Parameters
    ----------
    A : torch.Tensor
        Baseline effective connectivity, shape ``(N, N)``.
    B : torch.Tensor
        Stacked modulatory matrices, shape ``(J, N, N)``. ``J`` is the
        number of modulatory inputs. ``J=0`` (empty zeroth dim) is a
        supported edge case that returns ``A`` unchanged.
    u_mod : torch.Tensor
        Modulator values at the current time, shape ``(J,)``.

    Returns
    -------
    torch.Tensor
        Effective connectivity ``A_eff``, shape ``(N, N)``.

    Raises
    ------
    ValueError
        If ``B`` is not 3-D, ``u_mod`` is not 1-D, or their modulator
        counts (``B.shape[0]`` vs ``u_mod.shape[0]``) disagree.

    Examples
    --------
    >>> import torch
    >>> A = torch.eye(2, dtype=torch.float64) * -0.5
    >>> B = torch.zeros(1, 2, 2, dtype=torch.float64)
    >>> B[0, 0, 1] = 0.3
    >>> u_mod = torch.tensor([1.0], dtype=torch.float64)
    >>> A_eff = compute_effective_A(A, B, u_mod)
    >>> A_eff[0, 1].item()
    0.3
    """
    if B.ndim != 3:
        raise ValueError(
            "compute_effective_A expects B with shape (J, N, N); "
            f"got B.ndim={B.ndim} with shape {tuple(B.shape)}."
        )
    if u_mod.ndim != 1:
        raise ValueError(
            "compute_effective_A expects u_mod with shape (J,); "
            f"got u_mod.ndim={u_mod.ndim} with shape {tuple(u_mod.shape)}."
        )
    if B.shape[0] != u_mod.shape[0]:
        raise ValueError(
            "compute_effective_A modulator-count mismatch: expected "
            f"B.shape[0] == u_mod.shape[0], got B.shape[0]={B.shape[0]} "
            f"vs u_mod.shape[0]={u_mod.shape[0]}."
        )

    # Empty-J short-circuit: einsum over a zero-length axis is well-defined
    # (returns zeros of shape (N, N)), but the explicit branch returns A
    # bit-exactly without any arithmetic.
    if B.shape[0] == 0:
        return A

    # Einsum: for each j, add u_mod[j] * B[j] to A.
    return A + torch.einsum("j,jnm->nm", u_mod, B)


class NeuralStateEquation:
    """Neural state equation dx/dt = Ax + Cu (linear form; bilinear B-matrix path added in v0.3.0).

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
