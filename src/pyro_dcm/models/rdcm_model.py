"""Pyro generative model for regression DCM (rDCM).

Wraps the rDCM frequency-domain regression likelihood for use in
Pyro's SVI framework. This model is for ELBO comparison and
amortization (Phase 7) -- NOT for routine rDCM inference.

For routine rDCM inference, use the analytic VB functions directly:
- ``pyro_dcm.forward_models.rdcm_posterior.rigid_inversion``
- ``pyro_dcm.forward_models.rdcm_posterior.sparse_inversion``

The model samples per-region parameter vectors (theta_r) with sizes
determined by the active connections in each region's row of a_mask
and c_mask, plus confound columns. Each region can have a different
number of active connections, so a Python loop (not a plate) is used
for parameter sampling.

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
[REF-021] Frassle et al. (2018), NeuroImage 155, 406-421.
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist


def rdcm_model(
    Y: torch.Tensor,
    X: torch.Tensor,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    confound_cols: int = 1,
) -> None:
    """Pyro generative model for regression DCM.

    Wraps the frequency-domain regression likelihood from [REF-020]
    for ELBO comparison across DCM variants and amortized guide
    training in Phase 7. The analytic VB (``rigid_inversion``,
    ``sparse_inversion``) remains the PRIMARY inference method for
    rDCM.

    For each region r, the model:

    1. Builds the set of active column indices from a_mask and c_mask.
    2. Selects relevant columns from X to form X_r (N_eff, D_r).
    3. Filters NaN rows (from ``reduce_zeros``).
    4. Samples theta_r ~ N(0, I) of size D_r.
    5. Samples noise_prec_r ~ Gamma(2, 1).
    6. Evaluates Gaussian likelihood: obs_r ~ N(X_r @ theta_r, 1/sqrt(prec)).

    Parameters
    ----------
    Y : torch.Tensor
        Frequency-domain data from ``create_regressors``, shape
        ``(N_eff, nr)``, dtype float64. May contain NaN for rows
        excluded by ``reduce_zeros``.
    X : torch.Tensor
        Design matrix from ``create_regressors``, shape
        ``(N_eff, D)`` where ``D = nr + nu + nc``, dtype float64.
        May contain NaN for excluded rows.
    a_mask : torch.Tensor
        Binary mask for A connections, shape ``(nr, nr)``, float64.
    c_mask : torch.Tensor
        Binary mask for C connections, shape ``(nr, nu)``, float64.
    confound_cols : int, optional
        Number of confound columns in X. Default 1.

    Notes
    -----
    - Uses a Python loop over regions, NOT a pyro.plate for parameters,
      because each region can have a different D_r (number of active
      connections).
    - Prior on theta is N(0, 1) -- intentionally broader than the
      analytic VB priors. The Pyro model serves ELBO comparison and
      amortization, not primary inference.
    - The Y and X inputs come pre-computed from ``create_regressors``.
      This model does NOT call ``create_regressors`` internally.
    - ``.to_event(1)`` on the likelihood treats the frequency-domain
      data points within each region as a single observation event.

    References
    ----------
    [REF-020] Frassle et al. (2017), Eq. 4-8.
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    nc = confound_cols

    for r in range(nr):
        # --- Build active column indices for this region ---
        # A columns: indices where a_mask[r, :] is 1
        a_active = torch.where(a_mask[r, :] > 0.5)[0]
        # C columns: offset by nr in X
        c_active = torch.where(c_mask[r, :] > 0.5)[0] + nr
        # Confound columns: always all nc columns
        conf_indices = torch.arange(nr + nu, nr + nu + nc)

        # Combine all active column indices
        active_cols = torch.cat([a_active, c_active, conf_indices])
        D_r = active_cols.shape[0]

        # --- Select relevant columns from X ---
        X_r = X[:, active_cols]  # (N_eff, D_r)

        # --- Filter NaN rows ---
        valid = ~torch.isnan(Y[:, r])
        Y_r_valid = Y[valid, r]  # (n_valid,)
        X_r_valid = X_r[valid]   # (n_valid, D_r)

        # --- Sample theta_r: per-region parameter vector ---
        theta_r = pyro.sample(
            f"theta_{r}",
            dist.Normal(
                torch.zeros(D_r, dtype=torch.float64),
                torch.ones(D_r, dtype=torch.float64),
            ).to_event(1),
        )

        # --- Sample noise precision ---
        noise_prec_r = pyro.sample(
            f"noise_prec_{r}",
            dist.Gamma(
                torch.tensor(2.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            ),
        )

        # --- Predicted Y_r ---
        pred_r = X_r_valid @ theta_r  # (n_valid,)

        # --- Noise standard deviation ---
        noise_std = (1.0 / noise_prec_r).sqrt()

        # --- Likelihood ---
        pyro.sample(
            f"obs_{r}",
            dist.Normal(pred_r, noise_std).to_event(1),
            obs=Y_r_valid,
        )
