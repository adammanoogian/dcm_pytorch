"""Tests for Predictive-based posterior extraction from all guide types.

Verifies that ``extract_posterior_params`` works identically for all six
supported guide types, returns the backward-compatible ``'median'`` key,
produces ``std=0`` for ``AutoDelta``, and respects ``num_samples``.
"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import pytest
import torch

from pyro_dcm.models.guides import (
    GUIDE_REGISTRY,
    create_guide,
    extract_posterior_params,
    run_svi,
)


# ------------------------------------------------------------------
# Minimal Pyro model for extraction tests
# ------------------------------------------------------------------


def _simple_model(obs: torch.Tensor) -> None:
    """Two-site model: a, b -> obs = Normal(a + b, 1).

    Parameters
    ----------
    obs : torch.Tensor
        Observed value.
    """
    a = pyro.sample("a", dist.Normal(0.0, 1.0))
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    pyro.sample("obs", dist.Normal(a + b, 1.0), obs=obs)


_OBS = torch.tensor(3.0, dtype=torch.float64)
_MODEL_ARGS = (_OBS,)


# ------------------------------------------------------------------
# Helper: train a guide for a given type
# ------------------------------------------------------------------


def _train_guide(guide_type: str, num_steps: int = 30) -> tuple:
    """Train a guide and return (guide_or_post_guide, model_args).

    For ``auto_laplace``, returns the post-Laplace guide.

    Parameters
    ----------
    guide_type : str
        One of the keys in ``GUIDE_REGISTRY``.
    num_steps : int
        SVI steps to run.

    Returns
    -------
    tuple
        (trained guide, model_args tuple)
    """
    pyro.clear_param_store()
    guide = create_guide(
        _simple_model, guide_type=guide_type, init_scale=0.1,
    )
    result = run_svi(
        _simple_model,
        guide,
        _MODEL_ARGS,
        num_steps=num_steps,
        lr=0.01,
        elbo_type="trace_elbo",
        guide_type=guide_type,
    )
    # For auto_laplace, use the post-Laplace guide
    extract_guide = result.get("guide", guide)
    return extract_guide, _MODEL_ARGS


# ------------------------------------------------------------------
# 1. Parametrized extraction for all 6 guide types
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    sorted(GUIDE_REGISTRY.keys()),
    ids=sorted(GUIDE_REGISTRY.keys()),
)
def test_extract_posterior_all_guide_types(
    guide_type: str,
) -> None:
    """extract_posterior_params returns per-site mean/std/samples."""
    guide, model_args = _train_guide(guide_type)
    posterior = extract_posterior_params(
        guide, model_args, num_samples=50,
    )

    # Must have latent sites a and b
    for site in ("a", "b"):
        assert site in posterior, (
            f"Site {site!r} missing for guide_type={guide_type!r}"
        )
        info = posterior[site]
        assert "mean" in info
        assert "std" in info
        assert "samples" in info
        assert info["samples"].shape[0] == 50, (
            f"Expected 50 samples, got {info['samples'].shape[0]}"
        )
        assert torch.isfinite(info["mean"]).all()
        assert torch.isfinite(info["std"]).all()


# ------------------------------------------------------------------
# 2. AutoDelta std=0 test
# ------------------------------------------------------------------


def test_extract_posterior_auto_delta_std_zero() -> None:
    """AutoDelta extraction produces std=0 for all sites."""
    guide, model_args = _train_guide("auto_delta")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=100,
    )

    for site in ("a", "b"):
        std = posterior[site]["std"]
        assert torch.allclose(
            std, torch.zeros_like(std), atol=1e-6,
        ), (
            f"AutoDelta site {site!r} std should be 0, "
            f"got {std.item():.6f}"
        )


# ------------------------------------------------------------------
# 3. Backward-compatible median key test
# ------------------------------------------------------------------


def test_extract_posterior_median_key_backward_compat() -> None:
    """Extraction result has 'median' key mapping site names to means."""
    guide, model_args = _train_guide("auto_normal")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=50,
    )

    assert "median" in posterior
    median = posterior["median"]
    assert isinstance(median, dict)

    for site in ("a", "b"):
        assert site in median, (
            f"Site {site!r} missing from median dict"
        )
        # median[site] should match the per-site mean
        assert torch.allclose(
            median[site], posterior[site]["mean"], atol=1e-6,
        )


# ------------------------------------------------------------------
# 4. Sample-based quantiles test
# ------------------------------------------------------------------


def test_sample_based_quantiles() -> None:
    """Quantiles computed from samples bracket the posterior mean."""
    guide, model_args = _train_guide("auto_normal", num_steps=50)
    posterior = extract_posterior_params(
        guide, model_args, num_samples=500,
    )

    for site in ("a", "b"):
        samples = posterior[site]["samples"].float()
        q025 = torch.quantile(samples, 0.025, dim=0)
        q975 = torch.quantile(samples, 0.975, dim=0)
        mean = posterior[site]["mean"]

        # Mean must be within the 95% CI
        assert q025 <= mean <= q975, (
            f"Site {site!r}: mean={mean.item():.4f} not in "
            f"[{q025.item():.4f}, {q975.item():.4f}]"
        )

        # CI width should be positive (non-degenerate)
        assert (q975 - q025) > 0, (
            f"Site {site!r}: CI width is zero"
        )


# ------------------------------------------------------------------
# 5. num_samples parameter test
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [10, 100, 500])
def test_num_samples_controls_sample_count(
    n_samples: int,
) -> None:
    """num_samples parameter controls number of posterior samples."""
    guide, model_args = _train_guide("auto_normal")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=n_samples,
    )

    for site in ("a", "b"):
        assert posterior[site]["samples"].shape[0] == n_samples, (
            f"Expected {n_samples} samples for {site!r}, "
            f"got {posterior[site]['samples'].shape[0]}"
        )


# ------------------------------------------------------------------
# 6. AutoLaplace extraction test
# ------------------------------------------------------------------


def test_extract_posterior_auto_laplace() -> None:
    """AutoLaplace post-guide produces non-zero std."""
    guide, model_args = _train_guide("auto_laplace")
    posterior = extract_posterior_params(
        guide, model_args, num_samples=200,
    )

    for site in ("a", "b"):
        assert site in posterior
        std = posterior[site]["std"]
        # Post-Laplace approximation should have non-zero uncertainty
        assert std.item() > 0, (
            f"AutoLaplace site {site!r} std should be >0, "
            f"got {std.item():.6f}"
        )


# ------------------------------------------------------------------
# Bilinear posterior extraction (MODEL-05, Phase 15-03)
# ------------------------------------------------------------------


class TestExtractPosteriorBilinear:
    """MODEL-05: extract_posterior_params returns per-modulator B_j medians.

    extract_posterior_params is already site-agnostic -- any new sample or
    deterministic site (B_free_j, B) automatically appears in its output.
    This test verifies the passive claim on a live bilinear SVI run.

    Portability note (Plan 15-03 checker Blocker 2 resolution):
    We call Predictive DIRECTLY with explicit return_sites=['A_free', 'C',
    'noise_prec', 'B_free_0', 'B'] to make the 'B' deterministic-site
    assertion portable across Pyro 1.9+ patch versions regardless of whether
    Predictive(return_sites=None) default behavior includes deterministic
    sites. We then also exercise extract_posterior_params with the default
    return_sites=None to check the B_free_0 assertion (which holds across
    all versions since B_free_0 is a pyro.sample site, always included).
    """

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog) -> None:
        """Silence pyro_dcm.stability WARNING spam during bilinear SVI."""
        import logging
        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    def test_extract_posterior_includes_bilinear_sites(self) -> None:
        """Posterior via explicit-return_sites Predictive contains B_free_0 and B.

        Also exercises extract_posterior_params (default return_sites) for
        the B_free_0 assertion, which holds across all Pyro versions since
        B_free_0 is a stochastic pyro.sample site (always included).
        """
        from functools import partial

        from pyro.infer import SVI, Predictive, Trace_ELBO

        # IMPORTANT: do NOT import run_svi -- we use a bare SVI loop because
        # run_svi takes a positional model_args tuple, which cannot forward
        # task_dcm_model's keyword-only b_masks / stim_mod kwargs. Unused
        # imports would fail ruff's F401 unused-import check.
        from pyro_dcm.models.guides import create_guide
        from pyro_dcm.models.task_dcm_model import task_dcm_model
        from pyro_dcm.simulators.task_simulator import (
            make_block_stimulus,
            make_epoch_stimulus,
            make_random_stable_A,
            simulate_task_dcm,
        )
        from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

        N, M, J = 3, 1, 1
        A = make_random_stable_A(N, density=0.5, seed=42)
        C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
        stim = make_block_stimulus(
            n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
        )
        res = simulate_task_dcm(
            A, C, stim, duration=30.0, dt=0.01, TR=2.0, SNR=5.0, seed=7,
        )
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        t_eval = torch.arange(0, 30.0, 0.5, dtype=torch.float64)
        b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
        b_mask_0[1, 0] = 1.0
        b_masks = [b_mask_0]
        stim_mod_dict = make_epoch_stimulus(
            event_times=[10.0], event_durations=[10.0],
            event_amplitudes=[1.0],
            duration=30.0, dt=0.01, n_inputs=1,
        )
        stim_mod = PiecewiseConstantInput(
            stim_mod_dict["times"], stim_mod_dict["values"],
        )

        pyro.clear_param_store()
        guide = create_guide(task_dcm_model, init_scale=0.005)  # L2

        # 20 SVI steps via a bare loop. task_dcm_model takes keyword-only
        # b_masks / stim_mod kwargs; svi.step(**kwargs) forwards them.
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0},
        )
        svi = SVI(task_dcm_model, guide, optimizer, loss=Trace_ELBO())
        model_kwargs = dict(
            observed_bold=res["bold"],
            stimulus=res["stimulus"],
            a_mask=a_mask,
            c_mask=c_mask,
            t_eval=t_eval,
            TR=2.0,
            dt=0.5,
            b_masks=b_masks,
            stim_mod=stim_mod,
        )
        for _ in range(20):
            svi.step(**model_kwargs)

        # --- Portable assertion path: direct Predictive call with explicit
        # return_sites that include BOTH the stochastic site 'B_free_0' AND
        # the deterministic site 'B'. This works across Pyro 1.9+ patch
        # versions regardless of Predictive(return_sites=None) defaults.
        # Predictive invokes the model as model(*args); partial supplies
        # bilinear kwargs since Predictive does not accept model_kwargs.
        bilinear_model = partial(
            task_dcm_model, b_masks=b_masks, stim_mod=stim_mod,
        )
        model_args = (
            res["bold"], res["stimulus"], a_mask, c_mask, t_eval, 2.0, 0.5,
        )
        return_sites = ["A_free", "C", "noise_prec", "B_free_0", "B"]
        predictive = Predictive(
            bilinear_model,
            guide=guide,
            num_samples=10,
            return_sites=return_sites,
        )
        with torch.no_grad():
            samples = predictive(*model_args)

        # MODEL-05 gate (portable path): B_free_0 (raw) and B (deterministic,
        # masked) are both present because we asked for them explicitly.
        assert "B_free_0" in samples, (
            f"Missing 'B_free_0' in explicit-return_sites Predictive "
            f"output; keys: {sorted(samples.keys())}. MODEL-05 not "
            f"satisfied."
        )
        # B_free_0 samples have shape (num_samples, N, N).
        assert samples["B_free_0"].shape[-2:] == (N, N), (
            f"B_free_0 shape tail: expected (..., {N}, {N}), got "
            f"{tuple(samples['B_free_0'].shape)}"
        )
        assert "B" in samples, (
            f"Missing 'B' deterministic in explicit-return_sites Predictive "
            f"output; keys: {sorted(samples.keys())}. L3 guard may have "
            f"failed in Plan 15-01, or Predictive failed to honor "
            f"return_sites for deterministic sites."
        )
        # B samples have shape (num_samples, J, N, N).
        assert samples["B"].shape[-3:] == (J, N, N), (
            f"B shape tail: expected (..., {J}, {N}, {N}), got "
            f"{tuple(samples['B'].shape)}"
        )

        # --- Supplementary assertion: exercise extract_posterior_params with
        # its default (return_sites=None). B_free_0 is a stochastic
        # pyro.sample site and is always included regardless of Pyro version.
        # We do NOT require 'B' here to avoid Pyro-version coupling; that
        # assertion is already covered via the explicit return_sites path
        # above.
        posterior = extract_posterior_params(
            guide, model_args, model=bilinear_model, num_samples=10,
        )
        assert "B_free_0" in posterior, (
            f"Missing 'B_free_0' in extract_posterior_params output; keys: "
            f"{sorted(posterior.keys())}. MODEL-05 not satisfied for the "
            f"site-agnostic posterior-extraction path."
        )
        assert posterior["B_free_0"]["mean"].shape == (N, N), (
            f"B_free_0 posterior mean shape: expected ({N}, {N}), got "
            f"{posterior['B_free_0']['mean'].shape}"
        )
        # Linear sites still present (regression check).
        assert "A_free" in posterior
        assert "C" in posterior
        assert "noise_prec" in posterior
        # Backward-compat median dict also contains B_free_0.
        assert "B_free_0" in posterior["median"]
