"""Single-source CMC priors and steady state (CMC-04).

Transcribes ``spm_cmc_priors.m`` (SPM12, ``$Id: spm_cmc_priors.m 7279``) into
data tables for the single-source (``n = 1``) canonical microcircuit: log-normal
priors whose scaling parameters have prior mean 0 and the variances below.
Also provides the zero steady state used to freeze the integrator's Jacobian.

The extrinsic ``A`` priors and the delay ``P.D`` priors are Phase-34+ detail;
for Phase 33 only the single-source diagonal entries (``T``, ``G``, ``S``,
``C``, ``R``) are needed, and ``P.D`` is omitted/zeroed
(``spm_cmc_priors.m:123``).

References
----------
SPM12 ``spm_cmc_priors.m`` -- ``P.T`` var 1/32 (``:121``), ``P.G`` var 1/32
(``:122``), ``P.S`` var 1/64 (``:124``), ``P.C`` mean ``mask*32-32`` var
``mask/32`` (``:114-116``), ``P.A`` var 1/16 (``:80-81``), ``P.R`` var
``[1/16, 1/16]`` (``:133``). Zero steady state: ``spm_dcm_neural_x.m:70-72``
(the CMC ``otherwise`` branch returns ``M.x`` unchanged -- no Newton solve).
"""

from __future__ import annotations

import torch
from torch import Tensor

_F64 = torch.float64

#: Free log-parameter value mapping an ABSENT CMC connection (mask == 0) to a dead
#: edge. The CMC parameterises strengths as ``exp(P) * E0``, so an absent ``A``/``C``
#: edge must map to a strongly NEGATIVE free value (``exp(-32) * E0 ~ 1e-12``), NOT 0
#: (``exp(0) * E0 = E0`` would be a LIVE edge). This is the ``mask*32-32`` DEAD limit
#: of the ``spm_cmc_priors.m:114-116`` free-log convention -- the single source of
#: truth shared by the SVI ERP model and the VL ERP forward. The fixture-side
#: ``validation.export_to_mat._MS_A_DEAD`` holds the same value independently (the
#: parity ground truth must not import the code under test); the parity tests enforce
#: their equality.
ERP_DEAD_FREE: float = -32.0


def cmc_prior_moments(
    a_mask: Tensor,
    c_mask: Tensor,
    n: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Prior means ``E`` and variances ``V`` for the single-source CMC.

    Transcribes ``spm_cmc_priors.m`` (Fact 6). The extrinsic ``A`` moments are
    included only when ``n > 1`` (zero/absent at the single-source level,
    ``spm_fx_cmc.m:73-75``).

    Parameters
    ----------
    a_mask : torch.Tensor
        Extrinsic connection mask, shape ``(n, n)``. Used only when ``n > 1``.
    c_mask : torch.Tensor
        Input mask, shape ``(n, n_inp)`` (1 where an input drives a source).
    n : int
        Number of sources.

    Returns
    -------
    e_moments : dict
        Prior means, keys ``"T"``, ``"G"``, ``"S"``, ``"C"``, ``"R"``
        (and ``"A"`` when ``n > 1``).
    v_moments : dict
        Prior variances, same keys.
    """
    n_inp = c_mask.shape[1]
    e_moments: dict[str, Tensor] = {
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": torch.zeros(n, 4, dtype=_F64),
        "S": torch.zeros(n, 1, dtype=_F64),
        "C": c_mask.to(_F64) * 32.0 - 32.0,  # spm_cmc_priors.m:114-116
        "R": torch.zeros(n_inp, 2, dtype=_F64),  # spm_cmc_priors.m:133
    }
    v_moments: dict[str, Tensor] = {
        "T": torch.full((n, 4), 1.0 / 32.0, dtype=_F64),  # :121
        "G": torch.full((n, 4), 1.0 / 32.0, dtype=_F64),  # :122
        "S": torch.full((n, 1), 1.0 / 64.0, dtype=_F64),  # :124
        "C": c_mask.to(_F64) / 32.0,  # :114-116
        "R": torch.full((n_inp, 2), 1.0 / 16.0, dtype=_F64),  # :133
    }
    if n > 1:
        # Extrinsic priors (Phase 34): mean mask*32-32, var mask/16 (:80-81).
        e_moments["A"] = a_mask.to(_F64) * 32.0 - 32.0
        v_moments["A"] = a_mask.to(_F64) / 16.0
    return e_moments, v_moments


def cmc_steady_state(n: int) -> Tensor:
    """Zero steady state ``x0 = zeros(n, 8)`` (M1, no Newton solve).

    For ``spm_fx_cmc`` the steady-state solver hits the ``otherwise`` branch of
    ``spm_dcm_neural_x.m:70-72`` which returns ``M.x`` unchanged, and ``M.x`` is
    initialised to ``sparse(n, 8)`` zeros. At ``x = 0, u = 0`` the sigmoid gives
    ``S = F - 1/2 = 0`` and the input is 0, so ``f(0, 0) = 0`` -- zero IS the
    fixed point.

    Parameters
    ----------
    n : int
        Number of sources.

    Returns
    -------
    torch.Tensor
        Zero steady state, shape ``(n, 8)``, float64.
    """
    x0 = torch.zeros(n, 8, dtype=_F64)
    assert torch.all(x0 == 0), (
        f"CMC steady state must be exactly zeros(n, 8); got nonzero entries {x0}"
    )
    return x0
