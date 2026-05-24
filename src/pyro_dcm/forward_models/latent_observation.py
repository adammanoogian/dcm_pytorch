"""Observation equation for latent circuit DCM.

Maps neural state trajectories to observed measurements via a linear
observation model ``y = C_obs @ x + noise``. For v0.6.0, the
observation matrix C_obs is always identity (direct observation of
neural states). The module supports arbitrary C_obs for future learned
projection (v0.7.0+, MODEL-04).

This separates the observation equation from the generative model
function, enabling reuse across model variants and unit testing of
the observation computation in isolation.

References
----------
OBS-02 from REQUIREMENTS-v0.6.0.md -- Standalone observation function.
"""

from __future__ import annotations

import torch


def direct_observation(
    x: torch.Tensor,
    C_obs: torch.Tensor,
    noise_prec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute observation mean and noise std from neural state trajectories.

    Implements the linear observation equation ``y_mean = C_obs @ x``
    (or equivalently ``x @ C_obs.T`` for batched time points) and
    derives ``noise_std = (1 / noise_prec).sqrt()``.

    This function is a pure deterministic computation -- it does NOT
    call ``pyro.sample``. Pyro sampling happens in the model function
    that calls this.

    For v0.6.0, C_obs is always identity (P=N), so ``y_mean = x``.
    The function supports arbitrary C_obs for future learned projection
    (v0.7.0+, MODEL-04).

    Parameters
    ----------
    x : torch.Tensor
        Neural state trajectories. Shape ``(T, N)`` for a time series
        of T points across N latent dimensions, or ``(N,)`` for a
        single time point.
    C_obs : torch.Tensor
        Observation matrix, shape ``(P, N)`` where P is the
        observation dimension. When C_obs is the N x N identity matrix,
        this reduces to ``y_mean = x``.
    noise_prec : torch.Tensor
        Observation noise precision. Scalar or shape ``(P,)`` for
        per-dimension precision. ``noise_std = (1 / noise_prec).sqrt()``.

    Returns
    -------
    y_mean : torch.Tensor
        Predicted observation mean. Shape ``(T, P)`` when x is
        ``(T, N)``, or ``(P,)`` when x is ``(N,)``.
    noise_std : torch.Tensor
        Observation noise standard deviation. Scalar or shape ``(P,)``.

    Notes
    -----
    Implements OBS-02 from REQUIREMENTS-v0.6.0.md.

    The computation uses ``x @ C_obs.T`` rather than ``(C_obs @ x.T).T``
    for efficiency with batched time points (avoids two transposes).

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003) -- DCM observation model.
    OBS-02 REQUIREMENTS-v0.6.0.md -- Standalone observation function.
    """
    # Compute observation mean: y_mean = x @ C_obs.T
    # For x shape (T, N) and C_obs shape (P, N): result is (T, P)
    # For x shape (N,) and C_obs shape (P, N): result is (P,)
    y_mean = x @ C_obs.T

    # Compute noise standard deviation from precision
    noise_std = (1.0 / noise_prec).sqrt()

    return y_mean, noise_std
