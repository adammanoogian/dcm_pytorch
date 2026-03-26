"""Regression DCM end-to-end simulator.

Ties together BOLD generation, frequency-domain regressor construction,
and VB inversion into a single pipeline for parameter recovery testing.

This module provides:
- ``make_stable_A_rdcm``: Generate a stable A matrix for rDCM simulation.
- ``make_block_stimulus_rdcm``: Generate block-design stimulus inputs.
- ``simulate_rdcm``: End-to-end rDCM simulation and inversion.

All real operations use ``torch.float64``.

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
Julia source: RegressionDynamicCausalModeling.jl (generate_BOLD.jl,
rigid_inversion.jl, sparse_inversion.jl).
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.rdcm_forward import (
    create_regressors,
    generate_bold,
    get_hrf,
)
from pyro_dcm.forward_models.rdcm_posterior import (
    rigid_inversion,
    sparse_inversion,
)


def make_stable_A_rdcm(
    nr: int,
    density: float = 0.5,
    diag_range: tuple[float, float] = (-0.8, -0.2),
    offdiag_range: tuple[float, float] = (-0.3, 0.3),
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a stable A matrix suitable for rDCM simulation.

    Creates an endogenous connectivity matrix with the given density
    of off-diagonal connections and verifies that all eigenvalues
    have negative real parts (stability). If unstable, scales A
    until stable.

    Parameters
    ----------
    nr : int
        Number of regions.
    density : float, optional
        Fraction of off-diagonal connections present. Default 0.5.
    diag_range : tuple of float, optional
        Range for diagonal elements (self-inhibition).
        Default ``(-0.8, -0.2)``.
    offdiag_range : tuple of float, optional
        Range for off-diagonal elements. Default ``(-0.3, 0.3)``.
    seed : int or None, optional
        Random seed for reproducibility. Default ``None``.

    Returns
    -------
    tuple of torch.Tensor
        ``(A, a_mask)`` where ``A`` has shape ``(nr, nr)`` float64
        and ``a_mask`` has shape ``(nr, nr)`` float64 with 1s on
        diagonal and where off-diagonal connections exist.

    Notes
    -----
    Stability is enforced by checking eigenvalues and scaling A
    if needed. The mask always has 1s on the diagonal.
    """
    dtype = torch.float64
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    A = torch.zeros(nr, nr, dtype=dtype)
    a_mask = torch.eye(nr, dtype=dtype)

    # Diagonal: uniform in diag_range
    diag_vals = (
        torch.rand(nr, generator=gen, dtype=dtype)
        * (diag_range[1] - diag_range[0])
        + diag_range[0]
    )
    A.fill_diagonal_(0)
    for i in range(nr):
        A[i, i] = diag_vals[i]

    # Off-diagonal: sample binary mask with given density
    n_offdiag = nr * (nr - 1)
    n_present = int(round(density * n_offdiag))

    # Generate random off-diagonal mask
    offdiag_probs = torch.rand(n_offdiag, generator=gen, dtype=dtype)
    _, top_idx = torch.topk(offdiag_probs, n_present)
    offdiag_mask_flat = torch.zeros(n_offdiag, dtype=dtype)
    offdiag_mask_flat[top_idx] = 1.0

    # Map flat indices back to off-diagonal positions
    idx = 0
    for i in range(nr):
        for j in range(nr):
            if i != j:
                if offdiag_mask_flat[idx] > 0:
                    val = (
                        torch.rand(1, generator=gen, dtype=dtype).item()
                        * (offdiag_range[1] - offdiag_range[0])
                        + offdiag_range[0]
                    )
                    A[i, j] = val
                    a_mask[i, j] = 1.0
                idx += 1

    # Verify stability: all eigenvalues have negative real parts
    eigvals = torch.linalg.eigvals(A)
    max_real = eigvals.real.max().item()

    if max_real >= 0:
        # Scale A to make stable
        scale = 0.9 / (1.0 + max_real)
        A = A * scale
        # Re-check
        eigvals = torch.linalg.eigvals(A)
        max_real = eigvals.real.max().item()
        if max_real >= 0:
            # Fallback: subtract from diagonal
            A -= (max_real + 0.1) * torch.eye(nr, dtype=dtype)

    return A, a_mask


def make_block_stimulus_rdcm(
    n_time: int,
    n_inputs: int,
    u_dt: float,
    block_duration: float = 20.0,
    rest_duration: float = 20.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate block design stimulus for rDCM.

    Creates alternating on/off blocks for each input channel,
    with each input shifted by a random offset.

    Parameters
    ----------
    n_time : int
        Number of time steps.
    n_inputs : int
        Number of stimulus inputs.
    u_dt : float
        Time step (seconds).
    block_duration : float, optional
        Duration of each on-block (seconds). Default 20.0.
    rest_duration : float, optional
        Duration of each rest block (seconds). Default 20.0.
    seed : int or None, optional
        Random seed for reproducibility. Default ``None``.

    Returns
    -------
    torch.Tensor
        Stimulus matrix, shape ``(n_time, n_inputs)``, float64.
        Values are 0.0 or 1.0.

    Notes
    -----
    Each input channel has its own block pattern, shifted by a
    random offset to avoid perfect correlation between inputs.
    """
    dtype = torch.float64
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    u = torch.zeros(n_time, n_inputs, dtype=dtype)
    cycle = block_duration + rest_duration
    block_samples = int(round(block_duration / u_dt))
    cycle_samples = int(round(cycle / u_dt))

    for inp in range(n_inputs):
        # Random offset for this input
        offset = int(
            torch.randint(
                0, cycle_samples, (1,), generator=gen
            ).item()
        )
        for t in range(n_time):
            t_shifted = (t + offset) % cycle_samples
            if t_shifted < block_samples:
                u[t, inp] = 1.0

    return u


def simulate_rdcm(
    A: torch.Tensor,
    C: torch.Tensor,
    u: torch.Tensor,
    u_dt: float,
    y_dt: float,
    SNR: float = 3.0,
    a_mask: torch.Tensor | None = None,
    c_mask: torch.Tensor | None = None,
    mode: str = "rigid",
    sparse_kwargs: dict | None = None,
    confound_cols: int = 1,
    seed: int | None = None,
) -> dict:
    """End-to-end rDCM simulation and inversion.

    Generates synthetic BOLD data from a known connectivity matrix,
    constructs frequency-domain regressors, and runs VB inversion
    to recover the connectivity parameters.

    Steps:
    1. Generate BOLD: ``bold_result = generate_bold(A, C, u, ...)``.
    2. Create regressors: ``X, Y, N_eff = create_regressors(...)``.
    3. If ``a_mask`` is ``None``, use ``ones(nr, nr)``.
       If ``c_mask`` is ``None``, use ``ones(nr, nu)``.
    4. Run inversion (rigid or sparse).
    5. Combine results with ground truth info.

    Parameters
    ----------
    A : torch.Tensor
        Ground truth connectivity matrix, shape ``(nr, nr)``, float64.
    C : torch.Tensor
        Ground truth input weights, shape ``(nr, nu)``, float64.
    u : torch.Tensor
        Stimulus input, shape ``(N_u, nu)``, float64.
    u_dt : float
        Input sampling interval (seconds).
    y_dt : float
        BOLD sampling interval / TR (seconds).
    SNR : float, optional
        Signal-to-noise ratio for BOLD generation. Default 3.0.
    a_mask : torch.Tensor or None, optional
        Binary architecture mask for A, shape ``(nr, nr)``.
        If ``None``, uses ``ones(nr, nr)`` (full connectivity).
    c_mask : torch.Tensor or None, optional
        Binary mask for C, shape ``(nr, nu)``.
        If ``None``, uses ``ones(nr, nu)``.
    mode : str, optional
        Inversion mode: ``'rigid'`` or ``'sparse'``. Default ``'rigid'``.
    sparse_kwargs : dict or None, optional
        Additional keyword arguments for ``sparse_inversion``.
        E.g. ``{'n_reruns': 10, 'p0': 0.5}``. Default ``None``.
    confound_cols : int, optional
        Number of confound columns. Default 1.
    seed : int or None, optional
        Random seed for reproducibility (affects BOLD noise).
        Default ``None``.

    Returns
    -------
    dict
        Combined results with keys:
        - ``'A_true'``: ground truth A matrix.
        - ``'C_true'``: ground truth C matrix.
        - ``'A_mu'``: posterior mean A.
        - ``'C_mu'``: posterior mean C.
        - ``'y'``: noisy BOLD, shape ``(N_y, nr)``.
        - ``'y_clean'``: clean BOLD, shape ``(N_y, nr)``.
        - ``'F_total'``: total free energy (scalar).
        - ``'F_per_region'``: per-region free energies.
        - ``'iterations_per_region'``: iteration counts.
        - ``'mu_per_region'``: per-region posterior means.
        - ``'Sigma_per_region'``: per-region posterior covariances.
        - ``'z_per_region'``: (sparse only) binary indicators.
        - ``'mode'``: ``'rigid'`` or ``'sparse'``.

    References
    ----------
    [REF-020] Frassle et al. (2017). Julia ``generate_BOLD.jl``,
    ``rigid_inversion.jl``, ``sparse_inversion.jl``.

    See Also
    --------
    generate_bold : BOLD signal generation.
    create_regressors : Frequency-domain design matrix.
    rigid_inversion : Fixed-architecture VB inversion.
    sparse_inversion : ARD-based sparse VB inversion.
    """
    nr = A.shape[0]
    nu = C.shape[1]
    dtype = torch.float64

    # Set seed for reproducible noise
    if seed is not None:
        torch.manual_seed(seed)

    # Step 1: Generate BOLD
    bold_result = generate_bold(A, C, u, u_dt, y_dt, SNR)

    # Step 2: Create regressors
    X, Y, N_eff = create_regressors(
        bold_result["hrf"],
        bold_result["y"],
        u,
        u_dt,
        y_dt,
    )

    # Step 3: Default masks
    if a_mask is None:
        a_mask = torch.ones(nr, nr, dtype=dtype)
    if c_mask is None:
        c_mask = torch.ones(nr, nu, dtype=dtype)

    # Step 4: Run inversion
    if mode == "rigid":
        inv_result = rigid_inversion(
            X, Y, a_mask, c_mask, confound_cols
        )
    elif mode == "sparse":
        kwargs = sparse_kwargs if sparse_kwargs is not None else {}
        inv_result = sparse_inversion(
            X, Y, a_mask, c_mask, confound_cols, **kwargs
        )
    else:
        msg = f"Unknown mode '{mode}'. Use 'rigid' or 'sparse'."
        raise ValueError(msg)

    # Step 5: Combine results
    result: dict = {
        "A_true": A,
        "C_true": C,
        "y": bold_result["y"],
        "y_clean": bold_result["y_clean"],
        "mode": mode,
    }
    result.update(inv_result)

    return result
