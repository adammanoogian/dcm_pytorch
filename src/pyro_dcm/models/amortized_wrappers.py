"""Wrapper models for amortized inference with normalizing flow guides.

These wrapper models restructure the standard DCM Pyro models to use a
single packed latent vector ``_latent`` instead of multiple named sample
sites. This solves the Pyro site-matching problem: the normalizing flow
guide outputs a single vector, and both model and guide must share the
same sample site names for Pyro's automatic ELBO to work correctly.

**Pattern:** The wrapper model samples ``_latent`` from a standard
normal prior (in standardized space), unstandardizes it via the packer,
deterministically unpacks into named parameters, runs the same forward
model as the original Pyro model, and conditions on observed data via
the ``obs=`` kwarg in the likelihood site.

**rDCM amortized guide deferral:** rDCM amortized guide is intentionally
skipped because the analytic VB posterior is exact for the conjugate rDCM
model. The analytic posterior from regression DCM already provides
closed-form inference, so an amortized neural guide would add complexity
with no accuracy benefit. See 07-RESEARCH.md Section 2 (Architecture
Patterns, Regression DCM Flow) for the full rationale.

References
----------
[REF-042] Radev et al. (2020). BayesFlow.
[REF-043] Cranmer, Brehmer & Louppe (2020). SBI frontier.
07-RESEARCH.md: Wrapper model pattern (Section 3, Final recommendation).
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist

from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.guides.parameter_packing import (
    SpectralDCMPacker,
    TaskDCMPacker,
)
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
)


def _sample_latent_and_unpack(
    packer: TaskDCMPacker | SpectralDCMPacker,
) -> dict[str, torch.Tensor]:
    """Sample packed latent from N(0,1) prior and unpack.

    The prior is standard normal in standardized space because the
    packer's standardization was fit to match the data-generating
    distribution, so N(0,1) approximately matches the actual prior
    after standardization.

    Parameters
    ----------
    packer : TaskDCMPacker or SpectralDCMPacker
        Parameter packer with fitted standardization.

    Returns
    -------
    dict
        Unpacked parameter dictionary (values still in packed
        space -- caller handles exp() for positive params).
    """
    n = packer.n_features
    z_std = pyro.sample(
        "_latent",
        dist.Normal(
            torch.zeros(n, dtype=torch.float64),
            torch.ones(n, dtype=torch.float64),
        ).to_event(1),
    )
    z = packer.unstandardize(z_std)
    return packer.unpack(z)


def _run_task_forward_model(
    A: torch.Tensor,
    C: torch.Tensor,
    noise_prec: torch.Tensor,
    observed_bold: torch.Tensor,
    stimulus: object,
    t_eval: torch.Tensor,
    TR: float,
    dt: float,
) -> None:
    """Run task DCM forward model and evaluate likelihood.

    Reuses the same ODE integration, BOLD extraction, and Gaussian
    likelihood as ``task_dcm_model``. Factored out to keep wrapper
    functions under 50 lines.

    Parameters
    ----------
    A : torch.Tensor
        Parameterized connectivity matrix, shape ``(N, N)``.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.
    noise_prec : torch.Tensor
        Noise precision (positive scalar).
    observed_bold : torch.Tensor
        Observed BOLD, shape ``(T, N)``.
    stimulus : PiecewiseConstantInput or dict
        Stimulus function or dict with ``times`` and ``values``.
    t_eval : torch.Tensor
        Fine time grid for ODE integration.
    TR : float
        Repetition time in seconds.
    dt : float
        ODE step size in seconds.
    """
    N = A.shape[0]
    T = observed_bold.shape[0]

    # Convert dict stimulus to PiecewiseConstantInput if needed
    if isinstance(stimulus, dict):
        stimulus = PiecewiseConstantInput(
            stimulus["times"], stimulus["values"],
        )

    system = CoupledDCMSystem(A, C, stimulus)
    y0 = make_initial_state(N, dtype=torch.float64)
    solution = integrate_ode(system, y0, t_eval, method="rk4", step_size=dt)

    lnv = solution[:, 3 * N : 4 * N]
    lnq = solution[:, 4 * N : 5 * N]
    bold_fine = bold_signal(torch.exp(lnv), torch.exp(lnq))

    step = round(TR / dt)
    predicted_bold = bold_fine[::step][:T]

    # NaN protection: ODE can diverge for extreme parameter samples
    # from the untrained flow. Detach and replace with zeros to
    # produce a large finite penalty with zero gradient, preventing
    # NaN gradients from corrupting the flow parameters.
    if torch.isnan(predicted_bold).any():
        predicted_bold = torch.zeros_like(predicted_bold).detach()
    pyro.deterministic("predicted_bold", predicted_bold)

    noise_std = (1.0 / noise_prec).sqrt().clamp(min=1e-6)
    pyro.sample(
        "obs",
        dist.Normal(predicted_bold, noise_std).to_event(2),
        obs=observed_bold,
    )


def amortized_task_dcm_model(
    observed_bold: torch.Tensor,
    stimulus: object,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    TR: float,
    dt: float,
    packer: TaskDCMPacker,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: object | None = None,
) -> None:
    """Wrap task DCM for amortized inference via packed latent vector.

    Samples a single packed ``_latent`` vector from a standard normal
    prior (in standardized space), unstandardizes and unpacks into
    A_free, C, noise_prec, then runs the same forward model as
    ``task_dcm_model`` [REF-001] Eq. 1, [REF-002] Eq. 2-6.

    The single ``_latent`` site matches the ``AmortizedFlowGuide``
    which also samples only ``_latent``. This is the wrapper model
    pattern from 07-RESEARCH.md that solves the site-matching problem.

    Parameters
    ----------
    observed_bold : torch.Tensor
        Observed BOLD, shape ``(T, N)``, dtype float64.
    stimulus : PiecewiseConstantInput or dict
        Piecewise-constant stimulus function, or dict with
        ``times`` and ``values`` keys (auto-converted).
    a_mask : torch.Tensor
        Binary mask for A, shape ``(N, N)``, dtype float64.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(N, M)``, dtype float64.
    t_eval : torch.Tensor
        Fine time grid for ODE integration.
    TR : float
        Repetition time in seconds.
    dt : float
        ODE step size in seconds.
    packer : TaskDCMPacker
        Packer with fitted standardization.
    b_masks : list of torch.Tensor or None, optional
        Per-modulator binary structural masks, each shape ``(N, N)``.
        Default ``None``. Present for API symmetry with
        ``task_dcm_model``; non-empty ``b_masks`` raises
        ``NotImplementedError`` (bilinear amortized inference deferred
        to v0.3.1 per D5).
    stim_mod : PiecewiseConstantInput or None, optional
        Modulatory stimulus; ignored in linear mode. Default ``None``.

    Notes
    -----
    The prior on ``_latent`` is N(0, I) in standardized space. Since
    the standardization was fit to the data-generating distribution,
    this approximately matches the actual parameter priors after the
    inverse transform.

    **Bilinear support:** Not implemented in v0.3.0. Calling this
    wrapper with non-empty ``b_masks`` raises ``NotImplementedError``
    with a reference to v0.3.1 (per D5; see ``.planning/STATE.md``).
    The packer's fixed ``n_features = N*N + N*M + 1`` cannot
    accommodate ``J*N*N`` bilinear terms (see v0.3.0 PITFALLS.md B3).
    Use ``create_guide(task_dcm_model) + run_svi`` for bilinear DCM.
    The keyword-only ``b_masks`` and ``stim_mod`` arguments exist for
    API symmetry with ``task_dcm_model`` and to make the v0.3.1
    deferral message user-visible.

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1.
    [REF-002] Stephan et al. (2007), Eq. 2-6.
    [REF-042] Radev et al. (2020). BayesFlow.
    """
    # MODEL-07: amortized bilinear inference is deferred to v0.3.1 per D5.
    # Refusal fires on b_masks with non-empty list; None and [] both pass
    # through to the linear body (API symmetry with task_dcm_model).
    if b_masks is not None and len(b_masks) > 0:
        raise NotImplementedError(
            "amortized_task_dcm_model does not support bilinear (B) sample "
            "sites in v0.3.0 per D5 (.planning/STATE.md). Bilinear amortized "
            "inference is deferred to v0.3.1; the packer's fixed n_features "
            "= N*N + N*M + 1 cannot accommodate J*N*N bilinear terms (see "
            "v0.3.0 PITFALLS.md B3). Use the SVI path via "
            "create_guide(task_dcm_model) + run_svi for bilinear DCM."
        )
    # stim_mod is silently ignored in linear mode (API symmetry; no-op if
    # b_masks is None).
    del stim_mod  # mark intentionally unused in linear mode

    params = _sample_latent_and_unpack(packer)

    A_free = params["A_free"] * a_mask
    A = pyro.deterministic("A", parameterize_A(A_free))
    C = params["C"] * c_mask
    # Log-space contract: noise_prec is in log-space in packed vector
    noise_prec = params["noise_prec"].exp()

    _run_task_forward_model(
        A, C, noise_prec, observed_bold, stimulus, t_eval, TR, dt,
    )


def amortized_spectral_dcm_model(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    packer: SpectralDCMPacker,
) -> None:
    """Wrap spectral DCM for amortized inference via packed latent vector.

    Samples a single packed ``_latent`` vector, unpacks into A_free,
    noise_a, noise_b, noise_c, csd_noise_scale, then runs the same
    spectral DCM forward model as ``spectral_dcm_model`` [REF-010]
    Eq. 3-10.

    Parameters
    ----------
    observed_csd : torch.Tensor
        Observed CSD, shape ``(F, N, N)``, dtype complex128.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, dtype float64.
    a_mask : torch.Tensor
        Binary mask for A, shape ``(N, N)``, dtype float64.
    packer : SpectralDCMPacker
        Packer with fitted standardization.

    References
    ----------
    [REF-010] Friston, Kahan, Biswal & Razi (2014), Eq. 3-10.
    [REF-042] Radev et al. (2020). BayesFlow.
    """
    params = _sample_latent_and_unpack(packer)

    A_free = params["A_free"] * a_mask
    A = pyro.deterministic("A", parameterize_A(A_free))
    noise_a = params["noise_a"]
    noise_b = params["noise_b"]
    noise_c = params["noise_c"]
    # Log-space contract: csd_noise_scale in log-space
    csd_noise_scale = params["csd_noise_scale"].exp()

    predicted_csd = spectral_dcm_forward(A, freqs, noise_a, noise_b, noise_c)
    pyro.deterministic("predicted_csd", predicted_csd)

    pred_real = decompose_csd_for_likelihood(predicted_csd)
    obs_real = decompose_csd_for_likelihood(observed_csd)

    pyro.sample(
        "obs_csd",
        dist.Normal(pred_real, csd_noise_scale).to_event(1),
        obs=obs_real,
    )
