"""Gaussian evoked drive for DCM-for-evoked-responses (CMC-05).

Ports the exogenous-input generator ``spm_erp_u.m`` (SPM12,
``$Id: spm_erp_u.m 7679``) into pure torch (float64). Each input channel is a
Gaussian bump in peristimulus time, optionally mixed with a sustained
(cumulative) component, scaled by 32 (the ``spm_erp_u.m:63`` convention that the
CMC forward then re-scales by ``exp(P.C)`` inside ``cmc_f``).

References
----------
SPM12 ``spm_erp_u.m:42-64`` -- Gaussian bump; ms timebase (``:46``);
delay ``M.ons + 128*P.R[:,0]``, dispersion ``M.dur*exp(P.R[:,1])``,
sustained-mix ``M.sus*exp(P.R[:,2])``; the 32-scaling (``:63``).
"""

from __future__ import annotations

import torch
from torch import Tensor

_F64 = torch.float64


def erp_gaussian_input(
    t_s: Tensor,
    p_r: Tensor,
    ons_ms: float = 60.0,
    dur_ms: float = 16.0,
    sus: float = 0.0,
) -> Tensor:
    """Gaussian evoked drive ``u(t)`` per input channel (spm_erp_u.m:42-64).

    Parameters
    ----------
    t_s : torch.Tensor
        Peristimulus time grid in SECONDS, shape ``(ns,)``.
    p_r : torch.Tensor
        Onset/dispersion/sustained log-parameters, shape ``(n_inp, 2)`` or
        ``(n_inp, 3)`` (the third column is the sustained-mix log-parameter; a
        1-D ``(2,)`` / ``(3,)`` input is treated as a single channel).
    ons_ms : float, optional
        Onset ``M.ons`` in ms. Default 60 (``spm_gen_erp.m``).
    dur_ms : float, optional
        Dispersion ``M.dur`` in ms. Default 16 (set EXPLICITLY, not the 32-ms
        fallback; pitfall N3).
    sus : float, optional
        Sustained level ``M.sus``. Default 0 (the sustained-mix term is kept
        even at ``sus = 0``; pitfall N4).

    Returns
    -------
    torch.Tensor
        Evoked drive, shape ``(ns, n_inp)``, float64, 32-scaled.
    """
    t_s = torch.as_tensor(t_s, dtype=_F64)
    p_r = torch.as_tensor(p_r, dtype=_F64)
    if p_r.ndim == 1:
        p_r = p_r.reshape(1, -1)
    n_inp = p_r.shape[0]

    t_ms = t_s * 1000.0  # ms timebase internally (spm_erp_u.m:46)
    cols: list[Tensor] = []
    for i in range(n_inp):
        delay = ons_ms + 128.0 * p_r[i, 0]  # spm_erp_u.m:48
        scale = dur_ms * torch.exp(p_r[i, 1])  # spm_erp_u.m:49
        bump = torch.exp(-((t_ms - delay) ** 2) / (2.0 * scale**2))  # :50

        p_r2 = p_r[i, 2] if p_r.shape[1] >= 3 else torch.zeros((), dtype=_F64)
        prop = sus * torch.exp(p_r2)  # M.sus*exp(P.R[:,2]) (spm_erp_u.m:52)
        # Sustained-mix kept even when prop == 0 (pitfall N4).
        bump = prop * torch.cumsum(bump, dim=0) / torch.sum(bump) + bump * (1.0 - prop)

        cols.append(32.0 * bump)  # the 32-scaling lives here (spm_erp_u.m:63)
    return torch.stack(cols, dim=1)  # (ns, n_inp)
