"""BOLD signal observation equation.

Implements the BOLD signal equation from [REF-002] Eq. 6
(Stephan et al. 2007), mapping hemodynamic states (blood volume,
deoxyhemoglobin) to observed percent signal change.

This is a purely algebraic observation function (NOT an ODE). It is
applied after ODE integration to convert hemodynamic states into the
measured BOLD signal. Corresponds to SPM12 spm_gx_fmri.m.
"""

from __future__ import annotations

import torch


def bold_signal(
    v: torch.Tensor,
    q: torch.Tensor,
    E0: float = 0.40,
    V0: float = 0.02,
) -> torch.Tensor:
    """Compute BOLD percent signal change from hemodynamic states.

    Implements [REF-002] Eq. 6 (Stephan et al. 2007):

        y = V0 * (k1 * (1 - q) + k2 * (1 - q/v) + k3 * (1 - v))

    where:
        k1 = 7.0 * E0
        k2 = 2.0
        k3 = 2.0 * E0 - 0.2

    This is the simplified Buxton form. E0=0.40 is the SPM12 code
    default (not 0.34 from Stephan 2007 paper).

    Parameters
    ----------
    v : torch.Tensor
        Blood volume in LINEAR space (not log), shape ``(..., N)``.
        At steady state, v = 1.
    q : torch.Tensor
        Deoxyhemoglobin content in LINEAR space (not log),
        shape ``(..., N)``. At steady state, q = 1.
    E0 : float, optional
        Resting oxygen extraction fraction. Default 0.40 (SPM12 code).
    V0 : float, optional
        Resting venous blood volume fraction. Default 0.02.

    Returns
    -------
    torch.Tensor
        BOLD percent signal change, shape ``(..., N)``.
        At steady state (v=1, q=1), returns zero.

    Notes
    -----
    The BOLD signal should produce percent signal change in the
    realistic range of approximately 0.5--5% for physiological
    hemodynamic states near steady state.

    SPM12 source: ``spm_gx_fmri.m``.

    References
    ----------
    [REF-002] Stephan et al. (2007), Eq. 6.
    [REF-030] Buxton, Wong & Frank (1998) — original Balloon model.

    Examples
    --------
    >>> import torch
    >>> v = torch.ones(3, dtype=torch.float64)
    >>> q = torch.ones(3, dtype=torch.float64)
    >>> bold_signal(v, q)  # tensor([0., 0., 0.]) at steady state
    """
    k1 = 7.0 * E0
    k2 = 2.0
    k3 = 2.0 * E0 - 0.2
    return V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
