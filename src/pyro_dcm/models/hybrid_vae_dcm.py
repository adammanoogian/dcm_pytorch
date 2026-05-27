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

**Training infrastructure:**

- ``generate_synthetic_vae_dataset``: Creates diverse DCM parameter
  sets and simulates trajectories for training/validation.
- ``train_hybrid_vae_dcm``: SVI training loop with KL annealing
  (beta warmup) for stable ODE decoder training.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state
    equation dx/dt = Ax + Cu.
25-RESEARCH.md: Hybrid VAE-DCM architecture and training strategy.
"""

from __future__ import annotations

import logging

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
)

logger = logging.getLogger(__name__)


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
    and the ODE is integrated via
    ``CoupledDCMSystem(hemodynamic=False)``.

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
            stimulus["times"],
            stimulus["values"],
        )
    else:
        input_fn = stimulus

    system = CoupledDCMSystem(A, C, input_fn, hemodynamic=False)

    # --- Integrate ODE with learned initial conditions ---
    solution = integrate_ode(
        system,
        x0,
        t_eval,
        method="rk4",
        step_size=dt,
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
        predicted_trajectories,
        C_obs,
        noise_prec,
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


def generate_synthetic_vae_dataset(
    n_samples: int,
    n_regions: int = 4,
    n_inputs: int = 1,
    duration: float = 5.0,
    dt: float = 0.01,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic training data for hybrid VAE-DCM.

    Creates diverse DCM parameter sets and simulates latent-state
    trajectories using ``simulate_latent_circuit``. Each sample has
    a unique stable A matrix, random C, random initial conditions,
    and random observation noise precision. Trajectories are simulated
    with a simple block stimulus (on at t=1-2s).

    Parameters
    ----------
    n_samples : int
        Number of training examples to generate.
    n_regions : int, optional
        Number of latent dimensions (N). Default 4.
    n_inputs : int, optional
        Number of driving inputs (M). Default 1.
    duration : float, optional
        Simulation duration in seconds. Default 5.0.
    dt : float, optional
        ODE integration step size in seconds. Default 0.01.
    seed : int, optional
        Base random seed for reproducibility. Default 42.

    Returns
    -------
    list of dict
        Each dict contains:

        - ``"observed"``: Noisy trajectory, shape ``(T, N)``,
          dtype ``float64``.
        - ``"A"``: True connectivity matrix, shape ``(N, N)``.
        - ``"C"``: True driving input weights, shape ``(N, M)``.
        - ``"x0"``: True initial conditions, shape ``(N,)``.
        - ``"noise_prec"``: True noise precision, scalar.
        - ``"stimulus"``: ``PiecewiseConstantInput`` instance.
        - ``"t_eval"``: Time grid, shape ``(T,)``.
        - ``"a_mask"``: Connectivity mask, shape ``(N, N)``,
          all ones.
        - ``"c_mask"``: Input mask, shape ``(N, M)``, all ones.

    Notes
    -----
    The block stimulus is created via ``make_block_stimulus`` with
    parameters chosen so that the on-period falls within [1s, 2s]
    for a duration >= 3s. For shorter durations, a single block
    of 0.5s duration at t=0 is used.

    All tensors use ``torch.float64`` for numerical stability in
    ODE integration.

    References
    ----------
    25-RESEARCH.md: Synthetic validation strategy for hybrid VAE-DCM.

    Examples
    --------
    >>> dataset = generate_synthetic_vae_dataset(10, n_regions=3)
    >>> len(dataset)
    10
    >>> dataset[0]["observed"].shape
    torch.Size([500, 3])
    """
    dtype = torch.float64
    dataset: list[dict[str, torch.Tensor]] = []

    # Create a block stimulus: on at t=1-2s if duration allows,
    # otherwise a short block at t=0.
    if duration >= 3.0:
        stim_dict = make_block_stimulus(
            n_blocks=1,
            block_duration=1.0,
            rest_duration=duration - 1.0,
            n_inputs=n_inputs,
            dtype=dtype,
        )
        # Shift onset to t=1s by prepending a rest period.
        times_shifted = torch.cat([
            torch.tensor([0.0], dtype=dtype),
            stim_dict["times"] + 1.0,
        ])
        values_shifted = torch.cat([
            torch.zeros(1, n_inputs, dtype=dtype),
            stim_dict["values"],
        ])
        stim = PiecewiseConstantInput(times_shifted, values_shifted)
    else:
        stim_dict = make_block_stimulus(
            n_blocks=1,
            block_duration=duration * 0.3,
            rest_duration=duration * 0.7,
            n_inputs=n_inputs,
            dtype=dtype,
        )
        stim = PiecewiseConstantInput(
            stim_dict["times"], stim_dict["values"],
        )

    # Masks: fully connected
    a_mask = torch.ones(n_regions, n_regions, dtype=dtype)
    c_mask = torch.ones(n_regions, n_inputs, dtype=dtype)

    for i in range(n_samples):
        torch.manual_seed(seed + i)

        # Random stable A matrix
        A = make_stable_latent_circuit_A(n_regions, seed=seed + i)

        # Random C: sample from N(0, 0.5)
        C = 0.5 * torch.randn(n_regions, n_inputs, dtype=dtype)

        # Random initial conditions: small perturbations
        x0 = 0.1 * torch.randn(n_regions, dtype=dtype)

        # Random noise precision: Uniform(5, 50)
        noise_prec = 5.0 + 45.0 * torch.rand(1, dtype=dtype).item()
        noise_prec_t = torch.tensor(noise_prec, dtype=dtype)

        # Simulate trajectory
        sim_result = simulate_latent_circuit(
            A, C, stim, duration=duration, dt=dt,
            SNR=-1.0,  # No noise from simulator; we add our own
            seed=seed + i + n_samples,
        )

        clean_traj = sim_result["trajectories"]  # (T, N) noise-free
        t_eval = sim_result["times"]

        # Skip diverged simulations
        if sim_result["simulation_diverged"]:
            logger.warning(
                "Sample %d diverged during simulation, "
                "regenerating with stronger self-inhibition.",
                i,
            )
            # Retry with stronger self-inhibition
            A = make_stable_latent_circuit_A(
                n_regions, seed=seed + i,
                self_inhibition=2.0,
            )
            sim_result = simulate_latent_circuit(
                A, C, stim, duration=duration, dt=dt,
                SNR=-1.0, seed=seed + i + n_samples,
            )
            clean_traj = sim_result["trajectories"]
            t_eval = sim_result["times"]
            if sim_result["simulation_diverged"]:
                continue  # Skip this sample entirely

        # Add Gaussian noise: std = 1 / sqrt(noise_prec)
        noise_std = 1.0 / noise_prec**0.5
        noise = noise_std * torch.randn_like(clean_traj)
        observed = clean_traj + noise

        dataset.append({
            "observed": observed,
            "A": A,
            "C": C,
            "x0": x0,
            "noise_prec": noise_prec_t,
            "stimulus": stim,
            "t_eval": t_eval,
            "a_mask": a_mask,
            "c_mask": c_mask,
        })

    return dataset


def train_hybrid_vae_dcm(
    model_fn: object,
    guide: HybridVAEDCMGuide,
    train_data: list[dict[str, torch.Tensor]],
    n_epochs: int = 100,
    warmup_epochs: int = 20,
    lr: float = 1e-3,
    clip_norm: float = 10.0,
    log_every: int = 10,
) -> dict[str, list[float]]:
    """Train hybrid VAE-DCM with KL annealing via SVI.

    Implements the standard Pyro KL annealing pattern: the entire
    model is scaled by ``beta = min(1.0, epoch / warmup_epochs)``
    during the warmup period. This prevents posterior collapse and
    ODE decoder divergence during early training when the encoder
    is untrained.

    Parameters
    ----------
    model_fn : callable
        Pyro generative model (typically ``hybrid_vae_dcm_model``).
    guide : HybridVAEDCMGuide
        Pyro guide (encoder/recognition network).
    train_data : list of dict
        Training dataset from ``generate_synthetic_vae_dataset``.
        Each dict must have keys: ``"observed"``, ``"stimulus"``,
        ``"a_mask"``, ``"c_mask"``, ``"t_eval"``.
    n_epochs : int, optional
        Number of training epochs. Default 100.
    warmup_epochs : int, optional
        Number of KL warmup epochs (beta: 0 -> 1). Default 20.
    lr : float, optional
        Learning rate for ClippedAdam. Default 1e-3.
    clip_norm : float, optional
        Gradient clip norm for ClippedAdam. Default 10.0.
    log_every : int, optional
        Print progress every ``log_every`` epochs. Default 10.

    Returns
    -------
    dict
        Training results with keys:

        - ``"losses"``: Per-epoch average ELBO losses,
          length ``n_epochs``.
        - ``"betas"``: Per-epoch KL annealing schedule,
          length ``n_epochs``.

    Notes
    -----
    **KL annealing:** The entire model is scaled by beta, following
    the standard Pyro VAE tutorial pattern. During warmup (beta < 1),
    the reduced likelihood weight makes training more conservative,
    which stabilizes ODE decoder training. This is equivalent to the
    beta-VAE objective with time-varying beta.

    **NaN handling:** Individual SVI steps may produce NaN losses due
    to ODE divergence (the model's NaN guard prevents gradient
    corruption but the loss is still NaN). NaN losses are excluded
    from epoch averages.

    References
    ----------
    25-RESEARCH.md: KL annealing strategy for ODE-based decoders.

    Examples
    --------
    >>> dataset = generate_synthetic_vae_dataset(50, n_regions=3)
    >>> packer = LatentCircuitDCMPacker(3, 1, ...)  # with fit
    >>> enc = DCMEncoderNet(3, packer.total_dim).double()
    >>> guide = HybridVAEDCMGuide(enc, packer)
    >>> result = train_hybrid_vae_dcm(
    ...     hybrid_vae_dcm_model, guide, dataset,
    ...     n_epochs=50, warmup_epochs=10,
    ... )
    >>> len(result["losses"])
    50
    """
    optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm})

    # Extract dt from first training example's t_eval
    first_t_eval = train_data[0]["t_eval"]
    dt = float(first_t_eval[1] - first_t_eval[0])

    epoch_losses: list[float] = []
    beta_schedule: list[float] = []

    for epoch in range(n_epochs):
        # KL annealing: beta ramps from ~0 to 1 over warmup_epochs.
        # Clamp beta >= 1e-6 to avoid poutine.scale(scale=0) error.
        beta = min(1.0, max(1e-6, epoch / max(1, warmup_epochs)))
        beta_schedule.append(beta)

        # Create scaled model for this epoch's beta
        scaled_model = (
            poutine.scale(model_fn, scale=beta)
            if beta < 1.0
            else model_fn
        )

        # Create fresh SVI with current scaled model
        svi = SVI(scaled_model, guide, optimizer, loss=Trace_ELBO())

        # Shuffle training data
        perm = torch.randperm(len(train_data))
        step_losses: list[float] = []

        for idx in perm:
            ex = train_data[int(idx.item())]
            loss = svi.step(
                observed_trajectories=ex["observed"],
                stimulus=ex["stimulus"],
                a_mask=ex["a_mask"],
                c_mask=ex["c_mask"],
                t_eval=ex["t_eval"],
                dt=dt,
                packer=guide.packer,
            )

            # Track finite losses only
            if torch.isfinite(torch.tensor(loss)):
                step_losses.append(loss)

        # Epoch average loss (NaN if all steps diverged)
        if step_losses:
            avg_loss = sum(step_losses) / len(step_losses)
        else:
            avg_loss = float("nan")
        epoch_losses.append(avg_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            n_finite = len(step_losses)
            n_total = len(train_data)
            logger.info(
                "Epoch %d/%d | beta=%.3f | loss=%.2f "
                "| finite=%d/%d",
                epoch + 1,
                n_epochs,
                beta,
                avg_loss,
                n_finite,
                n_total,
            )

    return {"losses": epoch_losses, "betas": beta_schedule}
