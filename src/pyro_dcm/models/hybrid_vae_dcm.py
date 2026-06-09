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

- ``generate_synthetic_vae_dataset``: Generates diverse DCM parameter
  sets and simulates latent-state trajectories for training.
- ``train_hybrid_vae_dcm``: Training loop with KL annealing (beta
  warmup 0 -> 1) and gradient clipping for stable ODE decoder training.

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
import pyro.infer
import pyro.optim
import pyro.poutine
import torch
import torch.nn as nn

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.guides.dcm_encoder_net import DCMEncoderNet
from pyro_dcm.guides.parameter_packing import LatentCircuitDCMPacker
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
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


def masked_sign_recovery(
    pred: torch.Tensor,
    true: torch.Tensor,
    magnitude_threshold: float = 0.1,
) -> float:
    """Sign-recovery fraction over entries where ``|true| > threshold``.

    Connectivity matrices are sparse: most off-diagonal entries are exactly
    zero (absent connections). An unmasked sign comparison is meaningless on
    those entries because ``torch.sign(0) == 0`` can never equal the sign of a
    non-zero prediction -- every structural zero becomes a guaranteed mismatch,
    deflating the score (this caused the spurious Phase 25 HVAE-02 0.44). This
    masks to the genuinely non-zero ground-truth entries, matching the
    convention in ``benchmarks.runners.latent_circuit_recovery`` (B sign
    recovery on ``|B_true| > 0.1``).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted parameter tensor.
    true : torch.Tensor
        Ground-truth parameter tensor, same shape as ``pred``.
    magnitude_threshold : float, optional
        Entries with ``|true| <= threshold`` are excluded. Default 0.1.

    Returns
    -------
    float
        Fraction of eligible entries with matching sign, or ``nan`` if no
        entry exceeds the threshold.
    """
    mask = true.abs() > magnitude_threshold
    if not bool(mask.any()):
        return float("nan")
    match = (torch.sign(pred) == torch.sign(true))[mask]
    return float(match.float().mean().item())


def generate_synthetic_vae_dataset(
    n_samples: int,
    n_regions: int = 4,
    n_inputs: int = 1,
    duration: float = 5.0,
    dt: float = 0.01,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic DCM dataset for hybrid VAE-DCM training.

    For each sample, generates a random stable A matrix, random C and
    x0, a random noise precision, simulates the latent-circuit ODE,
    and adds Gaussian observation noise. Returns a list of dicts ready
    for training.

    Parameters
    ----------
    n_samples : int
        Number of synthetic examples to generate.
    n_regions : int, optional
        Number of latent dimensions (N). Default 4.
    n_inputs : int, optional
        Number of driving inputs (M). Default 1.
    duration : float, optional
        Simulation duration in seconds. Default 5.0.
    dt : float, optional
        ODE integration step size. Default 0.01.
    seed : int, optional
        Base random seed. Each sample uses ``seed + i``. Default 42.

    Returns
    -------
    list of dict
        Each dict contains:

        - ``"observed"``: Noisy trajectory, shape ``(T, N)``, float64.
        - ``"A"``: True A matrix, shape ``(N, N)``.
        - ``"A_free"``: Free A parameters (same as A for this
          generator since ``make_stable_latent_circuit_A`` produces
          the parameterized A directly).
        - ``"C"``: Driving input weights, shape ``(N, M)``.
        - ``"x0"``: Initial conditions, shape ``(N,)``.
        - ``"noise_prec"``: Noise precision (scalar).
        - ``"stimulus"``: ``PiecewiseConstantInput`` instance.
        - ``"t_eval"``: Time grid, shape ``(T,)``.
        - ``"a_mask"``: All-ones mask, shape ``(N, N)``.
        - ``"c_mask"``: All-ones mask, shape ``(N, M)``.

    Notes
    -----
    The stimulus is a simple block design: ON from 20-40% of duration,
    OFF otherwise. This provides a clean driving signal for validating
    the encoder's ability to recover connectivity from stimulus-driven
    dynamics.

    Examples
    --------
    >>> data = generate_synthetic_vae_dataset(10, n_regions=3, duration=2.0)
    >>> len(data)
    10
    >>> data[0]["observed"].shape[1]
    3
    """
    dataset: list[dict[str, torch.Tensor]] = []
    N, M = n_regions, n_inputs
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.ones(N, M, dtype=torch.float64)

    # Block stimulus: ON from 20% to 40% of duration
    on_start = 0.2 * duration
    on_end = 0.4 * duration
    stim_times = torch.tensor(
        [0.0, on_start, on_end, duration],
        dtype=torch.float64,
    )
    stim_values = torch.zeros(4, M, dtype=torch.float64)
    stim_values[1, :] = 1.0  # ON during [on_start, on_end)

    for i in range(n_samples):
        rng = torch.Generator().manual_seed(seed + i)

        # Random stable A
        A = make_stable_latent_circuit_A(N, seed=seed + i)

        # Random C ~ N(0, 0.5)
        C = 0.5 * torch.randn(N, M, dtype=torch.float64, generator=rng)

        # Random x0 ~ N(0, 0.1)
        x0 = 0.1 * torch.randn(N, dtype=torch.float64, generator=rng)

        # Random noise_prec ~ Uniform(5, 50)
        noise_prec_val = 5.0 + 45.0 * torch.rand(
            1, dtype=torch.float64, generator=rng,
        ).item()
        noise_prec = torch.tensor(noise_prec_val, dtype=torch.float64)

        # Create stimulus
        stimulus = PiecewiseConstantInput(stim_times, stim_values)

        # Simulate (noiseless)
        result = simulate_latent_circuit(
            A, C, stimulus, duration=duration, dt=dt,
            SNR=-1, solver="rk4", seed=None,
        )

        if result["simulation_diverged"]:
            logger.warning("Sample %d diverged, skipping.", i)
            continue

        traj_clean = result["trajectories"]  # (T, N)
        t_eval = result["times"]

        # Add noise: std = 1 / sqrt(noise_prec)
        noise_std = 1.0 / noise_prec_val**0.5
        rng2 = torch.Generator().manual_seed(seed + n_samples + i)
        noise = noise_std * torch.randn(
            traj_clean.shape, dtype=torch.float64, generator=rng2,
        )
        observed = traj_clean + noise

        # Use initial state from simulation for x0 ground truth
        # The simulation starts from zeros, but we set x0 as the
        # encoder target (perturbation from zero baseline).
        dataset.append({
            "observed": observed,
            "A": A,
            "A_free": A.clone(),
            "C": C,
            "x0": x0,
            "noise_prec": noise_prec,
            "stimulus": stimulus,
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
    dt: float = 0.01,
) -> dict[str, list[float]]:
    """Train hybrid VAE-DCM with KL annealing.

    Uses Pyro's SVI with Trace_ELBO and ClippedAdam optimizer. KL
    annealing linearly increases beta from 0 to 1 over ``warmup_epochs``
    by scaling the entire model via ``pyro.poutine.scale``. This is the
    standard Pyro KL annealing pattern (Pyro VAE tutorial).

    Parameters
    ----------
    model_fn : callable
        Pyro model function (``hybrid_vae_dcm_model``).
    guide : HybridVAEDCMGuide
        Encoder guide wrapping a ``DCMEncoderNet``.
    train_data : list of dict
        Training examples from ``generate_synthetic_vae_dataset``.
    n_epochs : int, optional
        Total training epochs. Default 100.
    warmup_epochs : int, optional
        Number of epochs for beta warmup (0 -> 1). Default 20.
    lr : float, optional
        Learning rate for ClippedAdam. Default 1e-3.
    clip_norm : float, optional
        Gradient clipping norm. Default 10.0.
    log_every : int, optional
        Print progress every N epochs. Default 10.
    dt : float, optional
        ODE integration step size. Default 0.01.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"losses"``: Per-epoch average loss, length ``n_epochs``.
        - ``"betas"``: Per-epoch beta values, length ``n_epochs``.

    Notes
    -----
    The training loop iterates over examples one-by-one (not batched)
    because the ODE decoder does not support batched integration.

    KL annealing scales the ENTIRE model (likelihood + prior). During
    warmup (beta < 1), this makes training more conservative but
    avoids the complexity of selectively scaling only the KL term.
    This is the standard approach from the Pyro VAE tutorial.

    Examples
    --------
    >>> data = generate_synthetic_vae_dataset(20, n_regions=3, duration=2.0)
    >>> packer = LatentCircuitDCMPacker(3, 1, ...)
    >>> packer.fit_standardization(...)
    >>> enc = DCMEncoderNet(3, packer.total_dim).double()
    >>> guide = HybridVAEDCMGuide(enc, packer)
    >>> result = train_hybrid_vae_dcm(
    ...     hybrid_vae_dcm_model, guide, data, n_epochs=10,
    ... )
    >>> len(result["losses"])
    10
    """
    pyro.clear_param_store()

    optimizer = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})

    # Mutable container for beta; the scaled_model closure reads it.
    beta_container: list[float] = [0.0]

    def scaled_model(*args: object, **kwargs: object) -> None:
        with pyro.poutine.scale(scale=beta_container[0]):
            model_fn(*args, **kwargs)

    svi = pyro.infer.SVI(
        scaled_model, guide, optimizer,
        loss=pyro.infer.Trace_ELBO(),
    )

    losses: list[float] = []
    betas: list[float] = []

    for epoch in range(n_epochs):
        beta = min(1.0, max(1e-3, epoch / max(1, warmup_epochs)))
        beta_container[0] = beta
        betas.append(beta)

        epoch_loss = 0.0
        n_valid = 0

        # Shuffle training data
        perm = torch.randperm(len(train_data))

        for idx in perm:
            ex = train_data[int(idx.item())]

            try:
                loss = svi.step(
                    observed_trajectories=ex["observed"],
                    stimulus=ex["stimulus"],
                    a_mask=ex["a_mask"],
                    c_mask=ex["c_mask"],
                    t_eval=ex["t_eval"],
                    dt=dt,
                    packer=guide.packer,
                )
            except Exception:
                logger.debug(
                    "SVI step failed for sample %d, skipping.", idx,
                )
                continue

            if not (torch.isnan(torch.tensor(loss))
                    or torch.isinf(torch.tensor(loss))):
                epoch_loss += loss
                n_valid += 1

        avg_loss = epoch_loss / max(1, n_valid)
        losses.append(avg_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d  beta=%.3f  loss=%.4f  (%d/%d valid)",
                epoch + 1, n_epochs, beta, avg_loss,
                n_valid, len(train_data),
            )

    return {"losses": losses, "betas": betas}
