"""SVI step timing profiler for DCM benchmark analysis.

Decomposes per-step wall-clock time into forward model, guide
evaluation, and ELBO+gradient components using ``pyro.poutine.trace``.
Implements Option C from 11-RESEARCH.md Section 6.2: profile a few
steps after training to capture steady-state timing.

Reports median and IQR (not just means) per STATE.md risk P12.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import numpy as np
import pyro
import pyro.poutine as poutine
import torch

from benchmarks.fixtures import load_fixture
from pyro_dcm.models import (
    create_guide,
    run_svi,
    spectral_dcm_model,
)
from pyro_dcm.models.spectral_dcm_model import (
    decompose_csd_for_likelihood,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


def profile_svi_step(
    model: Callable[..., Any],
    guide: Callable[..., Any],
    elbo: pyro.infer.ELBO,
    model_args: tuple[Any, ...],
    n_steps: int = 10,
) -> dict[str, Any]:
    """Profile per-step wall-clock timing of SVI components.

    Runs ``n_steps`` profiling iterations on a trained model/guide
    pair, timing each component separately with
    ``time.perf_counter``.

    Parameters
    ----------
    model : callable
        Pyro model function.
    guide : callable
        Trained Pyro guide function.
    elbo : pyro.infer.ELBO
        ELBO loss object (e.g., ``Trace_ELBO()``).
    model_args : tuple
        Positional arguments for model and guide.
    n_steps : int, optional
        Number of profiling steps. Default 10.

    Returns
    -------
    dict[str, Any]
        Timing results with keys:

        - ``"forward_times"``: list of per-step forward model times
        - ``"guide_times"``: list of per-step guide evaluation times
        - ``"gradient_times"``: list of per-step ELBO+backward times
        - ``"forward_median"``: median forward time (seconds)
        - ``"guide_median"``: median guide time (seconds)
        - ``"gradient_median"``: median gradient time (seconds)
        - ``"total_median"``: sum of three medians (seconds)
        - ``"forward_pct"``: forward percentage of total
        - ``"guide_pct"``: guide percentage of total
        - ``"gradient_pct"``: gradient percentage of total
    """
    forward_times: list[float] = []
    guide_times: list[float] = []
    gradient_times: list[float] = []

    for _ in range(n_steps):
        # Time forward model via poutine.trace
        t0 = time.perf_counter()
        poutine.trace(model).get_trace(*model_args)
        t1 = time.perf_counter()
        forward_times.append(t1 - t0)

        # Time guide evaluation via poutine.trace
        t2 = time.perf_counter()
        poutine.trace(guide).get_trace(*model_args)
        t3 = time.perf_counter()
        guide_times.append(t3 - t2)

        # Time ELBO + backward pass
        t4 = time.perf_counter()
        loss = elbo.differentiable_loss(
            model, guide, *model_args,
        )
        loss.backward()
        t5 = time.perf_counter()
        gradient_times.append(t5 - t4)

        # Zero gradients to avoid accumulation
        for param in pyro.get_param_store().values():
            if hasattr(param, "grad") and param.grad is not None:
                param.grad.zero_()

    fwd_med = statistics.median(forward_times)
    guide_med = statistics.median(guide_times)
    grad_med = statistics.median(gradient_times)
    total_med = fwd_med + guide_med + grad_med

    # Avoid division by zero
    if total_med < 1e-15:
        total_med = 1e-15

    return {
        "forward_times": forward_times,
        "guide_times": guide_times,
        "gradient_times": gradient_times,
        "forward_median": fwd_med,
        "guide_median": guide_med,
        "gradient_median": grad_med,
        "total_median": total_med,
        "forward_pct": fwd_med / total_med * 100.0,
        "guide_pct": guide_med / total_med * 100.0,
        "gradient_pct": grad_med / total_med * 100.0,
    }


# SVI guide types to profile (exclude rDCM -- analytic VB, no SVI)
_SVI_GUIDE_TYPES: list[str] = [
    "auto_delta",
    "auto_normal",
    "auto_lowrank_mvn",
    "auto_mvn",
    "auto_iaf",
    "auto_laplace",
]


def _build_spectral_model_args(
    n_regions: int,
    seed: int,
    fixtures_dir: str | None = None,
) -> tuple[tuple[Any, ...], torch.Tensor]:
    """Build spectral DCM model args from fixture or inline generation.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    seed : int
        Random seed.
    fixtures_dir : str or None
        Path to fixtures directory.

    Returns
    -------
    tuple
        ``(model_args, A_true)`` where model_args is the tuple
        passed to the spectral DCM model.
    """
    N = n_regions
    snr = 10.0

    if fixtures_dir is not None:
        data = load_fixture("spectral", N, 0, fixtures_dir)
        A_true = data["A_true"]
        noisy_csd = torch.complex(
            data["noisy_csd_real"].to(torch.float64),
            data["noisy_csd_imag"].to(torch.float64),
        )
        sim_freqs = data["freqs"]
    else:
        A_true = make_stable_A_spectral(N, seed=seed)
        sim = simulate_spectral_dcm(
            A_true, TR=2.0, n_freqs=32, seed=seed,
        )
        obs_real = decompose_csd_for_likelihood(sim["csd"])
        signal_power = obs_real.pow(2).mean().sqrt()
        noise_std = signal_power / snr
        torch.manual_seed(seed + 1000)
        noisy_obs = (
            obs_real + noise_std * torch.randn_like(obs_real)
        )
        F, n, _ = sim["csd"].shape
        half = F * n * n
        noisy_real = noisy_obs[:half].reshape(F, n, n)
        noisy_imag = noisy_obs[half:].reshape(F, n, n)
        noisy_csd = torch.complex(noisy_real, noisy_imag)
        sim_freqs = sim["freqs"]

    a_mask = torch.ones(N, N, dtype=torch.float64)
    model_args = (noisy_csd, sim_freqs, a_mask, N)
    return model_args, A_true


def profile_all_guides(
    variant: str = "spectral",
    n_regions: int = 3,
    fixtures_dir: str | None = None,
    seed: int = 42,
    n_profile_steps: int = 10,
    n_svi_steps: int = 500,
) -> dict[str, dict[str, Any]]:
    """Profile SVI step timing for all guide types on one dataset.

    For a single representative dataset (index 0), trains each SVI
    guide type and then calls ``profile_svi_step`` on the trained
    guide. Only profiles SVI variants (skips rDCM analytic VB).

    Currently supports ``variant="spectral"`` only.

    Parameters
    ----------
    variant : str, optional
        DCM variant. Default ``"spectral"``.
    n_regions : int, optional
        Number of brain regions. Default 3.
    fixtures_dir : str or None, optional
        Path to fixtures directory. If None, generates inline.
    seed : int, optional
        Random seed. Default 42.
    n_profile_steps : int, optional
        Number of profiling steps per guide. Default 10.
    n_svi_steps : int, optional
        Number of SVI training steps before profiling. Default 500.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from guide_type key to profile results dict
        (output of ``profile_svi_step``).

    Raises
    ------
    ValueError
        If ``variant`` is not ``"spectral"``.
    """
    if variant != "spectral":
        msg = (
            f"profile_all_guides currently only supports "
            f"variant='spectral', got {variant!r}"
        )
        raise ValueError(msg)

    N = n_regions
    results: dict[str, dict[str, Any]] = {}

    # Build model args once for all guides
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)

    model_args, _A_true = _build_spectral_model_args(
        N, seed, fixtures_dir,
    )

    for guide_type in _SVI_GUIDE_TYPES:
        # Skip auto_mvn if n_regions > 7
        if guide_type == "auto_mvn" and N > 7:
            print(
                f"  Skipping {guide_type} (N={N} > 7, "
                f"memory limit)"
            )
            continue

        print(f"  Profiling {guide_type}...")
        try:
            pyro.clear_param_store()
            torch.manual_seed(seed)

            guide = create_guide(
                spectral_dcm_model,
                init_scale=0.01,
                guide_type=guide_type,
                n_regions=N,
            )

            # Train the guide
            svi_result = run_svi(
                spectral_dcm_model, guide, model_args,
                num_steps=n_svi_steps, lr=0.01,
                clip_norm=10.0, lr_decay_factor=0.1,
                guide_type=guide_type,
            )

            # Use post-Laplace guide if available
            profile_guide = svi_result.get("guide", guide)

            # Build ELBO for profiling
            from pyro.infer import Trace_ELBO
            elbo = Trace_ELBO(num_particles=1)

            profile = profile_svi_step(
                spectral_dcm_model,
                profile_guide,
                elbo,
                model_args,
                n_steps=n_profile_steps,
            )
            results[guide_type] = profile
            print(
                f"    total_median={profile['total_median']:.4f}s "
                f"(fwd={profile['forward_pct']:.0f}%, "
                f"guide={profile['guide_pct']:.0f}%, "
                f"grad={profile['gradient_pct']:.0f}%)"
            )

        except (RuntimeError, ValueError) as e:
            print(f"    FAILED: {e}")

    return results
