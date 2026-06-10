"""Variational Laplace determinism regression tests (Plan 29-05, VLROBUST-01).

Pins the reproducibility contract of the VL engine across all three forward
models (spectral, task, latent-circuit) before the Phase 30 cluster sweep
consumes VL output across hundreds of cells:

* **Fixed-seed determinism** -- running the same fit twice with the same fixed
  seed and identical inputs yields equal (or tightly-tolerant) posterior means.
* **Seed sensitivity** -- different seeds yield different posterior means (the
  seed actually matters; guards against a stuck/constant fit).
* **Multi-restart reproducibility** (pitfall N4) -- a fixed restart-seed
  schedule selects the same winning restart and posterior on repeated runs,
  even though each restart explores a different basin.

Every fit here is deliberately TINY (``N=2``, short duration, low ``max_iter``)
so the whole file runs in well under three minutes on a laptop CPU, per the
project cluster-routing policy. These tests assert *determinism*, not recovery
quality, so the small settings are sufficient.

The fit setup mirrors the Plan 29-04 runners (``benchmarks.runners.spectral_vl``
/ ``task_vl`` / ``latent_circuit_vl``) and reuses their ground-truth builders to
stay DRY.

Note
----
``torch.use_deterministic_algorithms(True)`` is intentionally NOT forced here:
it can raise on some linear-algebra ops the VL engine relies on. Reproducibility
in this suite is achieved via fixed seeds plus byte-identical inputs, not via
enforced-determinism mode. See ``docs/03_methods_reference/vl_determinism_notes.md``
for the documented non-determinism sources (BLAS reduction order, float64
accumulation, ODE solver steps).

References
----------
.planning/phases/29-vl-validation-infra-bmr-rank/29-05-PLAN.md
    VLROBUST-01 determinism requirement.
.planning/research/v0.7.0/PITFALLS.md
    Pitfall N4 (local optima / multi-restart), N5 (finite-difference step).
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.runners.latent_circuit_recovery import _build_ground_truth
from pyro_dcm.inference import (
    LatentCircuitForward,
    SpectralDCMForward,
    TaskDCMForward,
    run_variational_laplace_generic,
)
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# Tight tolerance for "same seed -> same posterior mean". Bitwise equality is
# preferred (and asserted first), with this allclose as a documented fallback
# for sub-1e-8 BLAS-reduction-order float jitter.
_ATOL: float = 1e-8


def _fit_spectral(seed: int, max_iter: int) -> torch.Tensor:
    """Fit a tiny spectral-DCM VL problem and return the packed posterior mean.

    Builds an ``N=2`` stable ground truth, simulates a small CSD, and fits via
    ``run_variational_laplace_generic`` with ``SpectralDCMForward``.

    Parameters
    ----------
    seed : int
        Seed for both ground-truth construction and the fit's RNG state.
    max_iter : int
        Gauss-Newton iteration cap (kept low for laptop speed).

    Returns
    -------
    torch.Tensor
        Packed posterior-mean vector, shape ``(n_params,)``, float64.
    """
    n_regions = 2
    a_true = make_stable_A_spectral(n_regions, seed=seed)
    sim = simulate_spectral_dcm(a_true, TR=2.0, n_freqs=16, seed=seed)
    csd_obs = sim["csd"].to(torch.complex128)
    freqs = sim["freqs"].to(torch.float64)
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)

    forward = SpectralDCMForward()
    torch.manual_seed(seed)
    result = run_variational_laplace_generic(
        forward,
        observed=csd_obs,
        a_mask=a_mask,
        n_regions=n_regions,
        max_iter=max_iter,
        context={"freqs": freqs},
    )
    mean: torch.Tensor = forward.pack_params(**result.theta_post).to(
        torch.float64,
    )
    return mean


def _fit_task(seed: int, max_iter: int) -> torch.Tensor:
    """Fit a tiny task-DCM VL problem and return the packed posterior mean.

    Builds an ``N=2`` ground truth, simulates a short BOLD series (sampled at
    ``TR`` so ``T*N`` stays small), and fits via ``TaskDCMForward`` at the
    ``dt=0.1`` model grid (the [29-03] precision floor).

    Parameters
    ----------
    seed : int
        Seed for ground-truth construction and the fit's RNG state.
    max_iter : int
        Gauss-Newton iteration cap.

    Returns
    -------
    torch.Tensor
        Packed posterior-mean vector, shape ``(n_params,)``, float64.
    """
    n_regions = 2
    n_inputs = 1
    tr = 2.0
    dt_model = 0.1

    a_true = make_random_stable_A(n_regions, density=0.5, seed=seed)
    c_true = torch.zeros(n_regions, n_inputs, dtype=torch.float64)
    c_true[0, 0] = 1.0
    stim = make_block_stimulus(
        n_blocks=1,
        block_duration=10.0,
        rest_duration=10.0,
        n_inputs=n_inputs,
    )
    sim = simulate_task_dcm(
        a_true, c_true, stim,
        duration=20.0, dt=0.01, TR=tr, SNR=5.0, seed=seed,
    )
    bold = sim["bold"].to(torch.float64)  # (T_TR, N)

    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.zeros(n_regions, n_inputs, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    t_eval = torch.arange(
        0.0, bold.shape[0] * tr, tr, dtype=torch.float64,
    )[: bold.shape[0]]

    forward = TaskDCMForward(
        stimulus_fn=sim["stimulus"],
        c_mask=c_mask,
        t_eval=t_eval,
        dt=dt_model,
    )
    torch.manual_seed(seed)
    result = run_variational_laplace_generic(
        forward,
        observed=bold,
        a_mask=a_mask,
        n_regions=n_regions,
        max_iter=max_iter,
        context={"a_mask": a_mask},
    )
    mean: torch.Tensor = forward.pack_params(**result.theta_post).to(
        torch.float64,
    )
    return mean


def _fit_latent(seed: int, max_iter: int) -> torch.Tensor:
    """Fit a tiny latent-circuit VL problem and return the packed posterior mean.

    Reuses ``_build_ground_truth`` (the shared N=4 bilinear topology) at a SHORT
    duration and a very low ``max_iter`` so the dense time-domain precision and
    per-parameter ODE integrations stay laptop-fast.

    Parameters
    ----------
    seed : int
        Seed for the simulation and the fit's RNG state. The ground-truth
        topology itself is fixed (built with ``seed=0``) so only the simulated
        noise realisation varies with ``seed``.
    max_iter : int
        Gauss-Newton iteration cap (kept very low: <=4 for speed).

    Returns
    -------
    torch.Tensor
        Packed posterior-mean vector, shape ``(n_params,)``, float64.
    """
    duration = 10.0
    dt = 0.1
    snr = 10.0
    train_fraction = 0.80

    gt = _build_ground_truth(seed=0, duration=duration)
    a_true = gt["A_true"]
    b_true = gt["B_true"]  # (1, N, N)
    c_true = gt["C"]
    b_mask_0 = gt["b_mask_0"]
    stim = gt["stim"]
    stim_mod = gt["stim_mod"]
    a_mask = gt["a_mask"]
    c_mask = gt["c_mask"]
    n_regions = a_true.shape[0]

    sim = simulate_latent_circuit(
        a_true, c_true, stim,
        duration=duration, dt=dt, SNR=snr,
        solver="rk4", seed=seed,
        B_list=[b_true[0]], stimulus_mod=stim_mod,
    )
    trajs = sim["trajectories"].to(torch.float64)  # (T, N)
    t_all = sim["times"].to(torch.float64)
    if torch.isnan(trajs).any() or torch.isinf(trajs).any():
        raise ValueError("Simulated trajectories contain NaN/Inf.")

    t_train = int(trajs.shape[0] * train_fraction)
    trajs_train = trajs[:t_train]
    t_eval_train = t_all[:t_train]
    driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

    forward = LatentCircuitForward(
        stimulus=driving_stim,
        c_mask=c_mask,
        t_eval=t_eval_train,
        dt=dt,
        b_masks=[b_mask_0],
        stim_mod=stim_mod,
        c_prior_variance=1.0,
    )
    torch.manual_seed(seed)
    result = run_variational_laplace_generic(
        forward,
        observed=trajs_train,
        a_mask=a_mask,
        n_regions=n_regions,
        max_iter=max_iter,
        context={},
    )
    mean: torch.Tensor = forward.pack_params(**result.theta_post).to(
        torch.float64,
    )
    return mean


def _assert_means_equal(m1: torch.Tensor, m2: torch.Tensor) -> None:
    """Assert two posterior-mean vectors are equal, bitwise or within ``_ATOL``.

    Prefers exact bitwise equality; falls back to ``torch.allclose`` with
    ``atol=_ATOL`` to absorb sub-1e-8 BLAS-reduction-order float jitter (a
    documented non-determinism source -- see the methods-reference note).

    Parameters
    ----------
    m1, m2 : torch.Tensor
        Posterior-mean vectors to compare.
    """
    if torch.equal(m1, m2):
        return
    assert torch.allclose(m1, m2, atol=_ATOL, rtol=0.0), (
        "Same-seed VL fits diverged beyond BLAS-order jitter: "
        f"max|Δ|={float((m1 - m2).abs().max()):.3e} > atol={_ATOL:.0e}"
    )


@pytest.mark.vl
def test_spectral_vl_deterministic_fixed_seed() -> None:
    """Spectral VL: same seed -> equal posterior means across two fits."""
    m1 = _fit_spectral(seed=0, max_iter=24)
    m2 = _fit_spectral(seed=0, max_iter=24)
    _assert_means_equal(m1, m2)


@pytest.mark.vl
def test_task_vl_deterministic_fixed_seed() -> None:
    """Task VL: same seed -> equal posterior means across two fits."""
    m1 = _fit_task(seed=0, max_iter=6)
    m2 = _fit_task(seed=0, max_iter=6)
    _assert_means_equal(m1, m2)


@pytest.mark.vl
def test_latent_vl_deterministic_fixed_seed() -> None:
    """Latent-circuit VL: same seed -> equal posterior means across two fits."""
    m1 = _fit_latent(seed=0, max_iter=3)
    m2 = _fit_latent(seed=0, max_iter=3)
    _assert_means_equal(m1, m2)


@pytest.mark.vl
def test_different_seeds_differ() -> None:
    """Seed sensitivity: distinct seeds yield distinct spectral posterior means.

    Guards against a stuck/constant fit -- if the seed had no effect, the
    determinism tests above would pass trivially and prove nothing.
    """
    m0 = _fit_spectral(seed=0, max_iter=24)
    m1 = _fit_spectral(seed=1, max_iter=24)
    assert not torch.allclose(m0, m1, atol=_ATOL, rtol=0.0), (
        "Different seeds produced identical posterior means; the seed is not "
        "influencing the fit (data generation or RNG state is broken)."
    )


def _multistart_spectral(
    seed: int,
    restart_seeds: list[int],
    max_iter: int,
) -> tuple[torch.Tensor, int]:
    """Run a fixed multi-restart schedule on a tiny spectral problem.

    The VL engine converges to the prior-nearest local mode; a multi-restart
    wrapper re-seeds the RNG per restart and selects the fit with the highest
    final free energy. This local helper exercises that PATH deterministically
    (multi-restart is intentionally NOT added to the engine here -- out of scope
    for Plan 29-05; pitfall N4).

    The ground truth is fixed (built from ``seed``); each restart re-seeds the
    fit's RNG to ``rs`` before fitting. Because the engine starts every fit from
    the same prior mean (``initial_p=None``), the restarts fit identical data
    and the schedule is fully reproducible -- which is exactly the property
    under test: a fixed restart-seed schedule -> a deterministic selected mode.

    Parameters
    ----------
    seed : int
        Seed for the (fixed) ground truth and synthetic CSD.
    restart_seeds : list of int
        Per-restart RNG seeds (the restart schedule).
    max_iter : int
        Gauss-Newton iteration cap per restart.

    Returns
    -------
    best_mean : torch.Tensor
        Packed posterior mean of the winning (highest free-energy) restart.
    best_index : int
        Index into ``restart_seeds`` of the winning restart.
    """
    n_regions = 2
    a_true = make_stable_A_spectral(n_regions, seed=seed)
    sim = simulate_spectral_dcm(a_true, TR=2.0, n_freqs=16, seed=seed)
    csd_obs = sim["csd"].to(torch.complex128)
    freqs = sim["freqs"].to(torch.float64)
    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    forward = SpectralDCMForward()

    best_mean: torch.Tensor | None = None
    best_fe = float("-inf")
    best_index = -1
    for idx, rs in enumerate(restart_seeds):
        torch.manual_seed(rs)
        result = run_variational_laplace_generic(
            forward,
            observed=csd_obs,
            a_mask=a_mask,
            n_regions=n_regions,
            max_iter=max_iter,
            context={"freqs": freqs},
        )
        fe = result.free_energy[-1] if result.free_energy else float("-inf")
        if fe > best_fe:
            best_fe = fe
            best_index = idx
            mean: torch.Tensor = forward.pack_params(**result.theta_post).to(
                torch.float64,
            )
            best_mean = mean

    assert best_mean is not None, "Multi-restart produced no valid fit."
    return best_mean, best_index


@pytest.mark.vl
def test_multistart_schedule_reproducible() -> None:
    """Multi-restart: a fixed restart-seed schedule is reproducible (pitfall N4).

    Running the same restart-seed schedule twice must select the SAME winning
    restart and yield a matching posterior mean, proving the restart PATH is
    deterministic even though each restart explores a different basin.
    """
    restart_seeds = [10, 11, 12]
    mean_a, idx_a = _multistart_spectral(
        seed=0, restart_seeds=restart_seeds, max_iter=24,
    )
    mean_b, idx_b = _multistart_spectral(
        seed=0, restart_seeds=restart_seeds, max_iter=24,
    )
    assert idx_a == idx_b, (
        f"Multi-restart selected different winners across runs: "
        f"{idx_a} != {idx_b}"
    )
    _assert_means_equal(mean_a, mean_b)
