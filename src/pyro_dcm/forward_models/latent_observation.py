"""Direct observation model for latent-circuit DCM (v0.6.0).

Implements the identity observation equation for latent-circuit fitting:
y = C_obs @ x + noise, where C_obs is fixed at the identity matrix for
v0.6.0 (rotation ambiguity pitfall LC5 deferred to v0.7.0+).

The ``direct_observation`` function is a pure deterministic computation that
maps latent neural activity x(t) to predicted observations y_mean(t) and
returns the noise standard deviation derived from the Pyro-sampled noise
precision. The Pyro likelihood call is handled by the calling model
(``latent_circuit_dcm_model``), not here.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
    The latent state x evolves under the bilinear DCM; ``direct_observation``
    provides the measurement equation y = x (C_obs = I, v0.6.0).
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md -- C_obs
    identity constraint (pitfall LC5), v0.6.0 scope definition.
"""

from __future__ import annotations

import torch


def direct_observation(
    x: torch.Tensor,
    C_obs: torch.Tensor,
    noise_prec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute predicted observations from latent neural activity.

    Maps latent state trajectories through a fixed linear observation matrix
    and converts a noise precision sample into noise standard deviation.
    For v0.6.0, ``C_obs`` is always the identity matrix so ``y_mean = x``
    exactly. The function supports a general ``C_obs`` for forward
    compatibility with v0.7.0+ learned observation operators.

    Implements: y_mean(t) = C_obs @ x(t) per [REF-001] Eq. 1 observation
    equation. Noise model: noise_std = 1 / sqrt(noise_prec).

    Parameters
    ----------
    x : torch.Tensor
        Latent neural activity trajectories, shape ``(T, N)`` where T is
        the number of time points and N is the number of latent dimensions.
        dtype must be ``torch.float64``.
    C_obs : torch.Tensor
        Observation matrix, shape ``(N_obs, N)`` where N_obs is the number
        of observed dimensions. For v0.6.0 this is always the identity
        matrix ``torch.eye(N)``, so N_obs == N and y_mean == x.
        dtype must be ``torch.float64``.
    noise_prec : torch.Tensor
        Scalar noise precision (reciprocal of variance). Sampled from a
        Gamma prior in the calling Pyro model. Shape ``()`` (scalar).
        dtype must be ``torch.float64``.

    Returns
    -------
    y_mean : torch.Tensor
        Predicted observation mean, shape ``(T, N_obs)``. When C_obs is
        the identity, this equals x exactly with shape ``(T, N)``.
    noise_std : torch.Tensor
        Scalar noise standard deviation, shape ``()``. Computed as
        ``1 / sqrt(noise_prec)``.

    Notes
    -----
    This is a pure deterministic computation with no Pyro sample sites.
    The likelihood observation ``pyro.sample("obs", ...)`` is issued by
    the calling ``latent_circuit_dcm_model``.

    For v0.6.0, C_obs is always the identity so ``y_mean = x @ C_obs.T``
    reduces to x element-wise. The general matmul is preserved for
    forward compatibility.

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state
        equation and observation model.
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md --
        C_obs identity constraint (pitfall LC5).

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(100, 4, dtype=torch.float64)
    >>> C_obs = torch.eye(4, dtype=torch.float64)
    >>> noise_prec = torch.tensor(1.0, dtype=torch.float64)
    >>> y_mean, noise_std = direct_observation(x, C_obs, noise_prec)
    >>> y_mean.shape
    torch.Size([100, 4])
    >>> torch.allclose(y_mean, x)  # identity C_obs
    True
    >>> noise_std.shape
    torch.Size([])
    """
    # y_mean(t) = C_obs @ x(t) for each time point t.
    # x shape: (T, N), C_obs shape: (N_obs, N).
    # (x @ C_obs.T) has shape (T, N_obs) -- efficient batched matmul.
    y_mean = x @ C_obs.T

    # noise_std = 1 / sqrt(noise_prec): inverse transform of Gamma sample.
    noise_std = (1.0 / noise_prec).sqrt()

    return y_mean, noise_std
