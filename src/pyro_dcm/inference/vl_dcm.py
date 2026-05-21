"""DCM forward-function bridge for Variational Laplace.

Creates differentiable forward functions ``θ_flat → predicted_BOLD``
that reuse the existing forward model pipeline (neural state ODE,
Balloon-Windkessel, BOLD observation) without going through Pyro.

The ``make_task_dcm_forward`` factory returns everything the VL
optimizer needs: the forward function, prior mean, prior covariance,
and a parameter name map for interpreting the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.models.task_dcm_model import B_PRIOR_VARIANCE
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
    merge_piecewise_inputs,
)


@dataclass
class ParamLayout:
    """Maps flat parameter vector indices to named parameter blocks.

    Attributes
    ----------
    names : list[str]
        Human-readable name for each element of ``theta_flat``.
    slices : dict[str, slice]
        Maps block name (``'A_free'``, ``'C'``, ``'log_noise_prec'``,
        ``'B_free_0'``, ...) to its slice in ``theta_flat``.
    N : int
        Number of brain regions.
    M : int
        Number of driving inputs.
    J : int
        Number of modulators (0 for linear DCM).
    """

    names: list[str]
    slices: dict[str, slice]
    N: int
    M: int
    J: int


def _build_param_layout(
    N: int,
    M: int,
    J: int,
) -> ParamLayout:
    """Build the flat parameter layout for task DCM.

    Layout: [A_free (N*N), C (N*M), log_noise_prec (1), B_free_0 (N*N), ...]

    Parameters
    ----------
    N : int
        Number of regions.
    M : int
        Number of driving inputs.
    J : int
        Number of modulators.

    Returns
    -------
    ParamLayout
    """
    names: list[str] = []
    slices: dict[str, slice] = {}
    idx = 0

    # A_free: N*N elements
    n_a = N * N
    slices["A_free"] = slice(idx, idx + n_a)
    for i in range(N):
        for j in range(N):
            names.append(f"A_free[{i},{j}]")
    idx += n_a

    # C: N*M elements
    n_c = N * M
    slices["C"] = slice(idx, idx + n_c)
    for i in range(N):
        for j_c in range(M):
            names.append(f"C[{i},{j_c}]")
    idx += n_c

    # log_noise_prec: 1 element
    slices["log_noise_prec"] = slice(idx, idx + 1)
    names.append("log_noise_prec")
    idx += 1

    # B_free_j: N*N elements each
    for j_b in range(J):
        n_b = N * N
        slices[f"B_free_{j_b}"] = slice(idx, idx + n_b)
        for i in range(N):
            for j_c in range(N):
                names.append(f"B_free_{j_b}[{i},{j_c}]")
        idx += n_b

    return ParamLayout(names=names, slices=slices, N=N, M=M, J=J)


def _build_prior(
    layout: ParamLayout,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    b_masks: list[torch.Tensor] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prior mean and covariance matching ``task_dcm_model`` priors.

    Parameters
    ----------
    layout : ParamLayout
        Flat parameter layout.
    a_mask : torch.Tensor
        Binary A mask, shape ``(N, N)``.
    c_mask : torch.Tensor
        Binary C mask, shape ``(N, M)``.
    b_masks : list of torch.Tensor or None
        Per-modulator B masks.
    dtype : torch.dtype

    Returns
    -------
    prior_mean : torch.Tensor, shape ``(P,)``
    prior_cov : torch.Tensor, shape ``(P, P)``
    """
    P = len(layout.names)
    prior_mean = torch.zeros(P, dtype=dtype)
    prior_var = torch.ones(P, dtype=dtype)

    # A_free ~ N(0, 1/64) per task_dcm_model
    s = layout.slices["A_free"]
    prior_var[s] = 1.0 / 64.0
    mask_flat = a_mask.flatten()
    # Masked-out A elements: very small variance (effectively fixed)
    for i, m in enumerate(mask_flat):
        if m.item() == 0.0:
            prior_var[s.start + i] = 1e-10

    # C ~ N(0, 1) per task_dcm_model
    s = layout.slices["C"]
    prior_var[s] = 1.0
    c_mask_flat = c_mask.flatten()
    for i, m in enumerate(c_mask_flat):
        if m.item() == 0.0:
            prior_var[s.start + i] = 1e-10

    # log_noise_prec: broad prior (log-space)
    s = layout.slices["log_noise_prec"]
    prior_var[s] = 4.0

    # B_free_j ~ N(0, B_PRIOR_VARIANCE) per task_dcm_model
    if b_masks is not None:
        for j_b, bm in enumerate(b_masks):
            s = layout.slices[f"B_free_{j_b}"]
            prior_var[s] = B_PRIOR_VARIANCE
            bm_flat = bm.flatten()
            for i, m in enumerate(bm_flat):
                if m.item() == 0.0:
                    prior_var[s.start + i] = 1e-10

    prior_cov = torch.diag(prior_var)
    return prior_mean, prior_cov


def make_task_dcm_forward(
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    stimulus: PiecewiseConstantInput,
    t_eval: torch.Tensor,
    TR: float,
    dt: float = 0.5,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
    hemo_params: dict[str, float] | None = None,
) -> tuple[
    callable,
    torch.Tensor,
    torch.Tensor,
    ParamLayout,
]:
    """Create a VL-compatible forward function for task DCM.

    Builds a closure ``forward_fn(theta_flat) -> predicted_bold_flat``
    that unpacks parameters, applies masks and transforms, runs the
    coupled neural-hemodynamic ODE, and returns predicted BOLD at TR
    resolution. Reuses all existing forward model code.

    Parameters
    ----------
    a_mask : torch.Tensor
        Binary A mask, shape ``(N, N)``.
    c_mask : torch.Tensor
        Binary C mask, shape ``(N, M)``.
    stimulus : PiecewiseConstantInput
        Driving input stimulus.
    t_eval : torch.Tensor
        Fine time grid for ODE integration, shape ``(T_fine,)``.
    TR : float
        Repetition time in seconds.
    dt : float
        ODE step size.
    b_masks : list of torch.Tensor or None
        Per-modulator B masks (None for linear DCM).
    stim_mod : PiecewiseConstantInput or None
        Modulatory stimulus (required when ``b_masks`` is non-empty).
    hemo_params : dict or None
        Hemodynamic parameters; None for SPM defaults.

    Returns
    -------
    forward_fn : callable
        ``theta_flat (P,) -> predicted_bold_flat (T*N,)``
    prior_mean : torch.Tensor, shape ``(P,)``
    prior_cov : torch.Tensor, shape ``(P, P)``
    layout : ParamLayout
        Maps indices in ``theta_flat`` to named parameters.
    """
    N = a_mask.shape[0]
    M = c_mask.shape[1]
    J = len(b_masks) if b_masks is not None else 0
    dtype = a_mask.dtype

    if J > 0 and stim_mod is None:
        raise ValueError(
            "stim_mod is required when b_masks is non-empty; got None."
        )

    layout = _build_param_layout(N, M, J)
    prior_mean, prior_cov = _build_prior(layout, a_mask, c_mask, b_masks, dtype)

    # Pre-build merged input for bilinear mode
    if J > 0:
        merged_input = merge_piecewise_inputs(stimulus, stim_mod)
    else:
        merged_input = None

    T_obs = len(t_eval[:: round(TR / dt)])
    step = round(TR / dt)

    def forward_fn(theta_flat: torch.Tensor) -> torch.Tensor:
        """Map flat parameters to predicted BOLD (T*N,)."""
        # Unpack A_free
        a_free_flat = theta_flat[layout.slices["A_free"]]
        A_free = a_free_flat.reshape(N, N) * a_mask
        A = parameterize_A(A_free)

        # Unpack C
        c_flat = theta_flat[layout.slices["C"]]
        C = c_flat.reshape(N, M) * c_mask

        # Unpack B if bilinear
        B_stacked = None
        if J > 0:
            B_free_list = []
            for j_b in range(J):
                bf_flat = theta_flat[layout.slices[f"B_free_{j_b}"]]
                B_free_j = bf_flat.reshape(N, N)
                B_free_list.append(B_free_j)
            B_free_stacked = torch.stack(B_free_list, dim=0)
            b_mask_stacked = torch.stack(list(b_masks), dim=0)
            B_stacked = parameterize_B(B_free_stacked, b_mask_stacked)

        # Build ODE system
        if B_stacked is not None:
            system = CoupledDCMSystem(
                A, C, merged_input,
                hemo_params,
                B=B_stacked,
                n_driving_inputs=M,
            )
        else:
            system = CoupledDCMSystem(A, C, stimulus, hemo_params)

        y0 = make_initial_state(N, dtype=dtype)
        solution = integrate_ode(
            system, y0, t_eval, method="rk4", step_size=dt,
        )

        # Extract BOLD
        lnv = solution[:, 3 * N: 4 * N]
        lnq = solution[:, 4 * N: 5 * N]
        v = torch.exp(lnv)
        q = torch.exp(lnq)
        bold_fine = bold_signal(v, q)

        predicted_bold = bold_fine[::step][:T_obs]

        # NaN guard
        if torch.isnan(predicted_bold).any() or torch.isinf(predicted_bold).any():
            return torch.zeros(T_obs * N, dtype=dtype)

        return predicted_bold.reshape(-1)

    return forward_fn, prior_mean, prior_cov, layout


def unpack_theta(
    theta: torch.Tensor,
    layout: ParamLayout,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    b_masks: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Unpack a flat parameter vector into named DCM parameters.

    Parameters
    ----------
    theta : torch.Tensor
        Flat parameter vector, shape ``(P,)``.
    layout : ParamLayout
        Parameter layout from ``make_task_dcm_forward``.
    a_mask : torch.Tensor
        Binary A mask.
    c_mask : torch.Tensor
        Binary C mask.
    b_masks : list of torch.Tensor or None

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: ``'A'``, ``'A_free'``, ``'C'``, ``'noise_prec'``,
        and optionally ``'B_0'``, ``'B_free_0'``, etc.
    """
    N, M, J = layout.N, layout.M, layout.J

    A_free = theta[layout.slices["A_free"]].reshape(N, N) * a_mask
    A = parameterize_A(A_free)
    C = theta[layout.slices["C"]].reshape(N, M) * c_mask
    log_np = theta[layout.slices["log_noise_prec"]]
    noise_prec = torch.exp(log_np).item()

    result: dict[str, torch.Tensor] = {
        "A": A.detach(),
        "A_free": A_free.detach(),
        "C": C.detach(),
        "noise_prec": torch.tensor(noise_prec),
    }

    if b_masks is not None:
        for j_b in range(J):
            bf = theta[layout.slices[f"B_free_{j_b}"]].reshape(N, N)
            bm = b_masks[j_b]
            B_j = parameterize_B(
                bf.unsqueeze(0), bm.unsqueeze(0)
            ).squeeze(0)
            result[f"B_{j_b}"] = B_j.detach()
            result[f"B_free_{j_b}"] = bf.detach()

    return result
