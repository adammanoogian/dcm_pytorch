"""Hybrid VAE-DCM: physics-informed variational autoencoder for DCM.

The hybrid VAE-DCM uses a bilinear DCM forward model as the decoder of a
variational autoencoder. The latent space is interpretable as DCM
connectivity parameters (A, C, x0, noise_prec), and the decoder
integrates the neural state ODE [REF-001] Eq. 1 to produce predicted
trajectories.

**Architecture overview:**

- **Model (decoder/generative):** ``hybrid_vae_dcm_model`` samples a
  packed latent vector from a standard normal prior in standardized
  space, unstandardizes and unpacks it via ``LatentCircuitDCMPacker``
  into DCM parameters, runs ``CoupledDCMSystem(hemodynamic=False)``
  ODE integration as the decoder, and evaluates a Gaussian likelihood
  on observed trajectories.

- **Guide (encoder/recognition):** ``HybridVAEDCMGuide`` wraps a
  ``DCMEncoderNet`` that maps observed trajectories to approximate
  posterior parameters (z_loc, z_scale) and samples the latent via
  ``pyro.sample("_latent", N(z_loc, z_scale))``.

Both model and guide share a single ``_latent`` sample site for Pyro
ELBO compatibility. This follows the wrapper model pattern from
``amortized_wrappers.py`` adapted for the latent-circuit domain.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state
    equation dx/dt = Ax + Cu.
25-RESEARCH.md: Hybrid VAE-DCM architecture and training strategy.
"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
)


def hybrid_vae_dcm_model(
    observed_trajectories: torch.Tensor,
    stimulus: object,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float,
    packer: LatentCircuitDCMPacker,
) -> None:
    """Pyro generative model for hybrid VAE-DCM (decoder).

    Samples a packed latent vector from a standard normal prior in
    standardized space, unstandardizes and unpacks via the packer into
    DCM parameters (A_free, C, x0, noise_prec), runs the bilinear
    neural state ODE as the decoder, and evaluates a Gaussian
    likelihood on observed trajectories.

    The single ``_latent`` sample site matches the
    ``HybridVAEDCMGuide`` which also samples ``_latent``. This is
    the wrapper model pattern that keeps Pyro's automatic ELBO
    working.

    Implements [REF-001] Eq. 1: dx/dt = Ax + Cu(t), where A is
    parameterized via ``parameterize_A`` (negative diagonal enforced)
    and the ODE is integrated via ``CoupledDCMSystem(hemodynamic=False)``.

    Parameters
    ----------
    observed_trajectories : torch.Tensor
        Observed latent-state trajectories, shape ``(T, N)`` where T
        is the number of time points and N is the number of latent
        dimensions. dtype must be ``torch.float64``.
    stimulus : PiecewiseConstantInput or dict
        Piecewise-constant driving stimulus. If dict, must have
        ``'times'`` and ``'values'`` keys (auto-converted).
    a_mask : torch.Tensor
        Binary structural mask for A, shape ``(N, N)``.
    c_mask : torch.Tensor
        Binary structural mask for C, shape ``(N, M)``.
    t_eval : torch.Tensor
        Fine time grid for ODE integration, shape ``(T_ode,)``.
    dt : float
        ODE integration step size in seconds.
    packer : LatentCircuitDCMPacker
        Parameter packer with fitted standardization. Must have
        ``fit_standardization`` already called.

    Notes
    -----
    **Prior:** N(0, I) in standardized space. The packer's
    standardization was fit to the data-generating distribution, so
    N(0, I) approximately matches the actual parameter priors after
    unstandardization.

    **NaN guard:** When ODE integration diverges (NaN/Inf in predicted
    trajectories), the predictions are replaced with detached zeros.
    This produces a large finite ELBO penalty with zero gradient,
    preventing NaN gradients from corrupting the encoder parameters.

    **Observation model:** Identity C_obs (v0.6.0, pitfall LC5).

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1.
    25-RESEARCH.md: Hybrid VAE-DCM architecture.
    """
    N = a_mask.shape[0]
    T = observed_trajectories.shape[0]
    n = packer.total_dim

    # --- Sample packed latent from standard normal prior ---
    z_std = pyro.sample(
        "_latent",
        dist.Normal(
            torch.zeros(n, dtype=torch.float64),
            torch.ones(n, dtype=torch.float64),
        ).to_event(1),
    )

    # --- Unstandardize and unpack ---
    z = packer.unstandardize(z_std)
    params = packer.unpack(z)

    # --- Extract and mask parameters ---
    A_free = params["A_free"] * a_mask.to(dtype=params["A_free"].dtype)
    C = params["C"] * c_mask.to(dtype=params["C"].dtype)
    x0 = params["x0"]
    # Log-space contract: noise_prec stored in log-space in packed vec
    noise_prec = params["noise_prec"].exp()

    # --- Parameterize A (enforces negative diagonal) ---
    A = pyro.deterministic("A", parameterize_A(A_free))

    # --- Build ODE system (latent-circuit mode, no hemodynamics) ---
    if isinstance(stimulus, dict):
        input_fn = PiecewiseConstantInput(
            stimulus["times"], stimulus["values"],
        )
    else:
        input_fn = stimulus

    system = CoupledDCMSystem(A, C, input_fn, hemodynamic=False)

    # --- Integrate ODE with learned initial conditions ---
    solution = integrate_ode(
        system, x0, t_eval, method="rk4", step_size=dt,
    )

    # --- Predicted trajectories (truncate to match observed) ---
    predicted_trajectories = solution[:T]

    # --- NaN guard (mirrors amortized_wrappers pattern) ---
    if (
        torch.isnan(predicted_trajectories).any()
        or torch.isinf(predicted_trajectories).any()
    ):
        predicted_trajectories = torch.zeros_like(
            predicted_trajectories,
        ).detach()
    pyro.deterministic("predicted_trajectories", predicted_trajectories)

    # --- Direct observation: identity C_obs for v0.6.0 ---
    C_obs = torch.eye(N, dtype=torch.float64)
    y_mean, noise_std = direct_observation(
        predicted_trajectories, C_obs, noise_prec,
    )

    # --- Gaussian likelihood ---
    pyro.sample(
        "obs",
        dist.Normal(y_mean, noise_std).to_event(2),
        obs=observed_trajectories,
    )


class HybridVAEDCMGuide(nn.Module):
    """Pyro guide (encoder/recognition network) for hybrid VAE-DCM.

    Wraps a ``DCMEncoderNet`` as a Pyro guide that maps observed
    trajectories to an approximate posterior over the packed DCM
    parameter vector. The guide samples a single ``_latent`` site
    from ``N(z_loc, z_scale)`` where ``z_loc`` and ``z_scale`` are
    the outputs of the encoder network.

    Parameters
    ----------
    encoder_net : DCMEncoderNet
        1D-CNN encoder mapping ``(T, N)`` trajectories to
        ``(z_loc, z_scale)`` each of shape ``(latent_dim,)``.
    packer : LatentCircuitDCMPacker
        Parameter packer for unstandardize/unpack operations in
        ``sample_posterior``.

    Notes
    -----
    The ``_latent`` sample site name matches
    ``hybrid_vae_dcm_model`` exactly. Pyro's ELBO computes
    KL[q(_latent) || p(_latent)] automatically.

    The guide accepts ``*args, **kwargs`` beyond
    ``observed_trajectories`` to absorb additional model arguments
    (stimulus, masks, etc.) that SVI passes through.

    Examples
    --------
    >>> enc = DCMEncoderNet(4, packer.total_dim).double()
    >>> guide = HybridVAEDCMGuide(enc, packer)
    >>> obs = torch.randn(200, 4, dtype=torch.float64)
    >>> z = guide(obs)  # samples _latent
    """

    def __init__(
        self,
        encoder_net: DCMEncoderNet,
        packer: LatentCircuitDCMPacker,
    ) -> None:
        super().__init__()
        self.encoder_net = encoder_net
        self.packer = packer

    def forward(
        self,
        observed_trajectories: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Pyro guide: encode observations and sample ``_latent``.

        Parameters
        ----------
        observed_trajectories : torch.Tensor
            Observed trajectories, shape ``(T, N)``.
        *args : object
            Additional arguments (passed by SVI, absorbed here).
        **kwargs : object
            Additional keyword arguments (absorbed).

        Returns
        -------
        torch.Tensor
            Sampled standardized latent vector, shape
            ``(latent_dim,)``.
        """
        pyro.module("hybrid_vae_dcm_encoder", self.encoder_net)
        z_loc, z_scale = self.encoder_net(observed_trajectories)
        z_std = pyro.sample(
            "_latent",
            dist.Normal(z_loc, z_scale).to_event(1),
        )
        return z_std

    def sample_posterior(
        self,
        observed_trajectories: torch.Tensor,
        n_samples: int = 1000,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples via forward pass (no SVI needed).

        Parameters
        ----------
        observed_trajectories : torch.Tensor
            Observed trajectories, shape ``(T, N)``.
        n_samples : int, optional
            Number of posterior samples. Default 1000.

        Returns
        -------
        dict of str to torch.Tensor
            Unpacked parameter samples with keys ``A_free``
            ``(n_samples, N, N)``, ``C`` ``(n_samples, N, M)``,
            ``x0`` ``(n_samples, N)``, ``noise_prec``
            ``(n_samples,)``. Note: ``noise_prec`` is in log-space;
            caller must ``.exp()`` for positive precision.
        """
        self.eval()
        with torch.no_grad():
            z_loc, z_scale = self.encoder_net(observed_trajectories)
            z_std = dist.Normal(z_loc, z_scale).sample((n_samples,))
            # Unstandardize each sample
            z = self.packer.unstandardize(z_std)
            # Unpack each sample individually (packer expects 1D)
            results: dict[str, list[torch.Tensor]] = {
                "A_free": [],
                "C": [],
                "x0": [],
                "noise_prec": [],
            }
            for i in range(n_samples):
                params_i = self.packer.unpack(z[i])
                for key in results:
                    results[key].append(params_i[key])
            return {
                key: torch.stack(vals, dim=0)
                for key, vals in results.items()
            }
