"""Single-source canonical-microcircuit (CMC) neural-mass forward.

Ports a SINGLE source (``n = 1``) of the SPM12 canonical-microcircuit forward
``spm_fx_cmc.m`` (SPM12, ``$Id: spm_fx_cmc.m 7279``) into pure torch (float64,
zero new deps). Four populations / eight states (David & Friston 2003): spiny
stellate (ss), superficial pyramidal (sp), inhibitory interneurons (ii), and
deep pyramidal (dp).

At ``n = 1`` the extrinsic coupling blocks ``A{1..4}`` are identically zero
(``spm_fx_cmc.m:73-75``, the ``else A = {0,0,0,0}`` branch), so this module is
the intrinsic-dynamics foundation only; network coupling and condition
modulation ``B`` arrive in Phase 34.

This forward does NOT reuse the fMRI A-matrix parameterisation in
:mod:`pyro_dcm.forward_models.neural_state` (the ``a_ii = -exp(.)/2``
self-inhibition convention). CMC uses the log-normal ``+exp(P.*)`` scaling of
fixed structural strengths with structural signs baked into the equations of
motion (``spm_fx_cmc.m:69-72``).

State column layout (``spm_fx_cmc.m:6-14``), 0-indexed, shape ``(n, 8)``:

==========  =============  =====================================
0-idx col   population     quantity
==========  =============  =====================================
``x[:,0]``  ss             voltage
``x[:,1]``  ss             conductance
``x[:,2]``  sp             voltage
``x[:,3]``  sp             conductance
``x[:,4]``  ii             current
``x[:,5]``  ii             conductance
``x[:,6]``  dp             voltage
``x[:,7]``  dp             conductance
==========  =============  =====================================

Limitations
-----------
(a) **Delay-free (``D = I``).** Conduction delays are deliberately dropped: the
delay operator is the identity, applied via the ``spm_fx_cmc_nodelay.m`` wrapper
rather than SPM's delayed ``spm_fx_cmc.m``. The SPM parity fixtures were
themselves generated delay-free, so parity is apples-to-apples. Consequently this
forward reproduces SPM's evoked *amplitude* and *shape* but NOT delay-sensitive
*timing* (e.g. the forward/backward loop latency expressed as a difference-wave
peak shift). This is fine for the ratio / direction readouts this port is used
for, but is a real limitation if the model is ever fit to real MEG/EEG *timing*.
(b) **``P.M`` term not implemented.** The SPM ``P.M`` dp-voltage-gated modulatory
self-inhibition term (``spm_fx_cmc.m:158-160``) is NOT implemented here and is
SILENTLY IGNORED if a caller passes ``P["M"]``.

References
----------
SPM12 ``spm_fx_cmc.m`` -- equations of motion (``:171-198``), sigmoid
(``:90-94``), exogenous input (``:86,107``), permutation ``j`` (``:151``),
time-constant + intrinsic transforms (``:114,148-154``), ``spm_vec``
column-major flatten (``:199``). David, O. & Friston, K.J. (2003), "A neural
mass model for MEG/EEG: coupling and neuronal dynamics", NeuroImage 20,
1743-1755.
"""

from __future__ import annotations

import torch
from torch import Tensor

_F64 = torch.float64

# Intrinsic-strength permutation: MATLAB j = [7 2 3 4 1 5 6 8 9 10] (1-indexed)
# -> 0-indexed (spm_fx_cmc.m:151). Free P.G column 0 maps to J_PERM[0] == 6, the
# sp -> sp self-inhibition / precision knob (Fact 2).
J_PERM: tuple[int, ...] = (6, 1, 2, 3, 0, 4, 5, 7, 8, 9)

# Fixed defaults (spm_fx_cmc.m:47-49).
G0 = torch.tensor([4, 4, 8, 4, 4, 2, 4, 4, 2, 1], dtype=_F64) * 200.0
T0_MS = torch.tensor([2, 2, 16, 28], dtype=_F64)
E0 = torch.tensor([1, 0.5, 1, 0.5], dtype=_F64) * 200.0


def cmc_sigmoid(x: Tensor, p_s: Tensor) -> Tensor:
    """Voltage-firing sigmoid with the ``-1/2`` baseline (spm_fx_cmc.m:90-94).

    ``R = (2/3) * exp(P.S)``; ``S = 1/(1 + exp(-R*x)) - 1/2``. The ``-1/2``
    baseline (``F - 1/(1+exp(0))``) is load-bearing: it makes ``S(0) = 0`` so
    the zero state is a fixed point.

    Parameters
    ----------
    x : torch.Tensor
        State, shape ``(n, 8)``.
    p_s : torch.Tensor
        Sigmoid-slope log parameter ``P.S``, broadcastable to ``x``
        (shape ``(n, 1)`` or scalar).

    Returns
    -------
    torch.Tensor
        Firing deviation ``S``, shape ``(n, 8)``.
    """
    slope = (2.0 / 3.0) * torch.exp(p_s)
    return 1.0 / (1.0 + torch.exp(-slope * x)) - 0.5


def cmc_flatten(x: Tensor) -> Tensor:
    """Column-major (``spm_vec``) flatten of ``(n, 8)`` -> ``(8n,)``.

    MATLAB ``spm_vec`` is column-major (``spm_fx_cmc.m:199``); the torch
    equivalent is ``x.T.reshape(-1)``.
    """
    return x.transpose(-2, -1).reshape(-1)


def cmc_unflatten(x_flat: Tensor, n: int) -> Tensor:
    """Inverse of :func:`cmc_flatten`: ``(8n,)`` -> ``(n, 8)`` (column-major)."""
    return x_flat.reshape(8, n).transpose(-2, -1)


def parameterize_cmc(p: dict[str, Tensor], n: int) -> dict[str, Tensor]:
    """Map free log-parameters to CMC strengths (spm_fx_cmc.m:114-160).

    Builds the synaptic time constants ``T`` (seconds), the 10 intrinsic
    strengths ``G`` (with the ``J_PERM`` permutation), the input gain ``C``,
    and the extrinsic blocks ``A`` (identically zero at ``n = 1``,
    ``spm_fx_cmc.m:73-75``). This is NOT a generalisation of the fMRI
    A-matrix transform: CMC uses ``+exp(P.*)`` log-normal scaling of fixed
    structural strengths.

    Parameters
    ----------
    p : dict
        Free parameters. Keys (each optional, default zeros): ``"T"``
        ``(n, 4)``, ``"G"`` ``(n, 4)``, ``"C"`` ``(n, n_inp)``, ``"S"``
        ``(n, 1)``.
    n : int
        Number of sources (1 for Phase 33).

    Returns
    -------
    dict
        ``{"T": (n,4), "G": (n,10), "C": (n,n_inp), "A": (4,n,n), "S": (n,1)}``.
    """
    p_t = p.get("T", torch.zeros(n, 4, dtype=_F64))
    p_g = p.get("G", torch.zeros(n, 4, dtype=_F64))
    p_c = p.get("C", torch.zeros(n, 1, dtype=_F64))
    p_s = p.get("S", torch.zeros(n, 1, dtype=_F64))

    # Time constants: ms -> seconds, then exp-scale all 4 free entries
    # (spm_fx_cmc.m:114,148-150).
    t = (torch.ones(n, 1, dtype=_F64) * T0_MS) / 1000.0 * torch.exp(p_t)

    # Intrinsic strengths: G(:, J_PERM[i]) *= exp(P.G[:, i]) for i in 0..3
    # (spm_fx_cmc.m:151-154). Build a multiplier and scatter into permuted cols.
    mult = torch.ones(n, 10, dtype=_F64)
    mult[:, list(J_PERM[:4])] = torch.exp(p_g)
    g = (torch.ones(n, 1, dtype=_F64) * G0) * mult

    # Input gain C = exp(P.C) (spm_fx_cmc.m:86); extrinsic A zero at n=1.
    c = torch.exp(p_c)
    a = torch.zeros(4, n, n, dtype=_F64)

    return {"T": t, "G": g, "C": c, "A": a, "S": p_s}


def cmc_f(x_flat: Tensor, u: Tensor, p: dict[str, Tensor], n: int = 1) -> Tensor:
    """CMC equations of motion ``dx/dt`` (spm_fx_cmc.m:171-198).

    Single-source intrinsic dynamics: the four conductance equations (granular
    ss, supragranular sp, ii, infragranular dp) plus the four voltage integrals
    ``f[:,0]=x[:,1]`` etc. Extrinsic ``A`` terms vanish at ``n = 1``.

    Parameters
    ----------
    x_flat : torch.Tensor
        Flat state, shape ``(8n,)``, float64 (column-major, :func:`cmc_flatten`).
    u : torch.Tensor or float
        Per-sample exogenous input, shape ``(n_inp,)``.
    p : dict
        Free-parameter struct (see :func:`parameterize_cmc`).
    n : int, optional
        Number of sources. Default 1.

    Returns
    -------
    torch.Tensor
        ``dx/dt`` flat, shape ``(8n,)``, float64.

    Raises
    ------
    TypeError
        If ``x_flat`` is not float64 (expected ``torch.float64``).
    """
    if x_flat.dtype != _F64:
        raise TypeError(
            f"x_flat must be float64; expected {_F64}, got {x_flat.dtype} "
            "(the CMC forward runs entirely in float64, CMC-07)"
        )
    x = cmc_unflatten(x_flat, n)  # (n, 8)
    params = parameterize_cmc(p, n)
    t, g, c, p_s = params["T"], params["G"], params["C"], params["S"]

    s = cmc_sigmoid(x, p_s)  # (n, 8)
    u_vec = torch.atleast_1d(torch.as_tensor(u, dtype=_F64))
    big_u = (c @ u_vec) * 32.0  # exogenous input (spm_fx_cmc.m:86,107), (n,)

    # Granular -- spiny stellate (spm_fx_cmc.m:171-174).
    uu_g = big_u - g[:, 0] * s[:, 0] - g[:, 2] * s[:, 4] - g[:, 1] * s[:, 2]
    f1 = (uu_g - 2.0 * x[:, 1] - x[:, 0] / t[:, 0]) / t[:, 0]

    # Supragranular -- superficial pyramidal (spm_fx_cmc.m:177-180).
    uu_sp = g[:, 7] * s[:, 0] - g[:, 6] * s[:, 2]
    f3 = (uu_sp - 2.0 * x[:, 3] - x[:, 2] / t[:, 1]) / t[:, 1]

    # Supragranular -- inhibitory interneurons (spm_fx_cmc.m:183-186).
    uu_ii = g[:, 4] * s[:, 0] + g[:, 5] * s[:, 6] - g[:, 3] * s[:, 4]
    f5 = (uu_ii - 2.0 * x[:, 5] - x[:, 4] / t[:, 2]) / t[:, 2]

    # Infragranular -- deep pyramidal (spm_fx_cmc.m:189-192).
    uu_dp = -g[:, 9] * s[:, 6] - g[:, 8] * s[:, 4]
    f7 = (uu_dp - 2.0 * x[:, 7] - x[:, 6] / t[:, 3]) / t[:, 3]

    # Voltages are integrals of conductances (spm_fx_cmc.m:195-198).
    f0 = x[:, 1]
    f2 = x[:, 3]
    f4 = x[:, 5]
    f6 = x[:, 7]

    f = torch.stack([f0, f1, f2, f3, f4, f5, f6, f7], dim=1)  # (n, 8)
    return cmc_flatten(f)
