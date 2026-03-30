"""Balloon-Windkessel hemodynamic model in log-space.

Implements the hemodynamic ODEs from [REF-002] Eq. 2-5
(Stephan et al. 2007), using the log-space state representation
from SPM12 spm_fx_fmri.m for numerical stability.

Hemodynamic states are stored in log-space (lnf, lnv, lnq) to enforce
positivity of blood flow, volume, and deoxyhemoglobin. The vasodilatory
signal s remains in linear space.

SPM12 CODE parameter defaults are used (not the Stephan 2007 paper values).
"""

from __future__ import annotations

import torch

# SPM12 spm_fx_fmri.m default hemodynamic parameters (H vector).
#
# These are the AUTHORITATIVE implementation values from SPM12 source code:
#   H(1) = 0.64   signal decay (kappa, s^-1)
#   H(2) = 0.32   autoregulation (gamma, s^-1)
#   H(3) = 2.00   transit time (tau, s)
#   H(4) = 0.32   Grubb's exponent (alpha)
#   H(5) = 0.40   resting oxygen extraction (E0)
#
# DISCREPANCY NOTE: Stephan et al. 2007 Table 1 cites different values:
#   kappa=0.65, gamma=0.41, tau=0.98, E0=0.34
# We follow SPM12 code per the project rule "when in doubt, follow SPM
# convention." The free parameters (P.decay, P.transit) in SPM relate
# these two sets of values via exponential scaling.
DEFAULT_HEMO_PARAMS: dict[str, float] = {
    "kappa": 0.64,
    "gamma": 0.32,
    "tau": 2.0,
    "alpha": 0.32,
    "E0": 0.40,
}


class BalloonWindkessel:
    """Balloon-Windkessel hemodynamic ODE in log-space.

    Implements [REF-002] Eq. 2-5 (Stephan et al. 2007) with log-space
    hemodynamic states following SPM12 spm_fx_fmri.m convention.

    State variables per region (N regions total):
    - ``x``: neural activity (drives hemodynamics), linear space
    - ``s``: vasodilatory signal, linear space
    - ``lnf``: log blood flow (f = exp(lnf))
    - ``lnv``: log blood volume (v = exp(lnv))
    - ``lnq``: log deoxyhemoglobin (q = exp(lnq))

    Parameters
    ----------
    kappa : torch.Tensor or float
        Signal decay rate (s^-1), shape ``(N,)`` or scalar.
    gamma : torch.Tensor or float
        Flow-dependent elimination rate (s^-1).
    tau : torch.Tensor or float
        Hemodynamic transit time (s).
    alpha : torch.Tensor or float
        Grubb's exponent (vessel stiffness).
    E0 : torch.Tensor or float
        Resting oxygen extraction fraction.

    Examples
    --------
    >>> import torch
    >>> bw = BalloonWindkessel()  # SPM12 defaults
    >>> x = torch.tensor([0.5, 0.0])
    >>> s = torch.zeros(2)
    >>> lnf = lnv = lnq = torch.zeros(2)
    >>> ds, dlnf, dlnv, dlnq = bw.derivatives(x, s, lnf, lnv, lnq)
    """

    def __init__(
        self,
        kappa: torch.Tensor | float = DEFAULT_HEMO_PARAMS["kappa"],
        gamma: torch.Tensor | float = DEFAULT_HEMO_PARAMS["gamma"],
        tau: torch.Tensor | float = DEFAULT_HEMO_PARAMS["tau"],
        alpha: torch.Tensor | float = DEFAULT_HEMO_PARAMS["alpha"],
        E0: torch.Tensor | float = DEFAULT_HEMO_PARAMS["E0"],
    ) -> None:
        self.kappa = kappa
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.E0 = E0

    def derivatives(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        lnf: torch.Tensor,
        lnv: torch.Tensor,
        lnq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hemodynamic state derivatives in log-space.

        Implements [REF-002] Eq. 2-5 with log-space chain rule:

        Eq. 2: ds/dt = x - kappa * s - gamma * (f - 1)
        Eq. 3: d(lnf)/dt = s / f
        Eq. 4: d(lnv)/dt = (f - fv) / (tau * v)
        Eq. 5: d(lnq)/dt = (f * E_f / E0 - fv * q / v) / (tau * q)

        where f = exp(lnf), v = exp(lnv), q = exp(lnq),
        fv = v^(1/alpha) is venous outflow, and
        E_f = 1 - (1 - E0)^(1/f) is oxygen extraction.

        Parameters
        ----------
        x : torch.Tensor
            Neural activity driving hemodynamics, shape ``(N,)``.
        s : torch.Tensor
            Vasodilatory signal, shape ``(N,)``.
        lnf : torch.Tensor
            Log blood flow, shape ``(N,)``.
        lnv : torch.Tensor
            Log blood volume, shape ``(N,)``.
        lnq : torch.Tensor
            Log deoxyhemoglobin content, shape ``(N,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (ds, dlnf, dlnv, dlnq) — time derivatives of hemodynamic
            states, each shape ``(N,)``.

        Notes
        -----
        The vasodilatory signal equation (Eq. 2) uses f in linear space
        (not log), so lnf is exponentiated first. SPM12 applies the same
        convention in spm_fx_fmri.m.

        Blood flow is clamped to f >= 1e-6 before computing oxygen
        extraction to prevent NaN from 1/f divergence (see pitfall 7
        in RESEARCH.md).
        """
        # Exponentiate log-space states [REF-002]
        # Clamp lnf to prevent numerical issues in extraction function
        lnf_safe = torch.clamp(lnf, min=-14.0)
        f = torch.exp(lnf_safe)  # blood flow, shape (N,)
        v = torch.exp(lnv)  # blood volume, shape (N,)
        q = torch.exp(lnq)  # deoxyhemoglobin, shape (N,)

        # Venous outflow: fv = v^(1/alpha) [REF-002] Eq. 4
        fv = v.pow(1.0 / self.alpha)

        # Oxygen extraction fraction [REF-002] Eq. 5
        # E_f = 1 - (1 - E0)^(1/f)
        # Clamp f to avoid division by zero
        f_safe = torch.clamp(f, min=1e-6)
        E_f = 1.0 - (1.0 - self.E0) ** (1.0 / f_safe)

        # Vasodilatory signal [REF-002] Eq. 2
        # ds/dt = x - kappa * s - gamma * (f - 1)
        ds = x - self.kappa * s - self.gamma * (f - 1.0)

        # Log blood flow [REF-002] Eq. 3 + chain rule
        # d(lnf)/dt = (1/f) * df/dt = s / f
        dlnf = s / f_safe

        # Log blood volume [REF-002] Eq. 4 + chain rule
        # d(lnv)/dt = (1/v) * dv/dt = (f - fv) / (tau * v)
        dlnv = (f - fv) / (self.tau * v)

        # Log deoxyhemoglobin [REF-002] Eq. 5 + chain rule
        # d(lnq)/dt = (1/q) * dq/dt
        #           = (f * E_f / E0 - fv * q / v) / (tau * q)
        dlnq = (f * E_f / self.E0 - fv * q / v) / (self.tau * q)

        return ds, dlnf, dlnv, dlnq
