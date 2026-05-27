"""BMR vs brute-force ELBO validation test.

Validates that Bayesian Model Reduction (BMR) agrees with brute-force
ELBO model comparison on a synthetic latent-circuit DCM. Fits a full
model, then compares BMR-based ranking of reduced architectures against
independent SVI fits.

References
----------
[REF-070] Friston & Penny (2011) -- Post hoc Bayesian model selection.
"""

from __future__ import annotations

import time

import pyro
import pytest
import torch
from scipy.stats import spearmanr

from pyro_dcm.model_selection import (
    bayesian_model_reduction,
    make_reduced_prior_zero_connection,
)
from pyro_dcm.models import (
    LC_A_PRIOR_VARIANCE,
    create_guide,
    latent_circuit_dcm_model,
    run_svi,
)
from pyro_dcm.simulators.latent_circuit_simulator import simulate_latent_circuit
from pyro_dcm.simulators.task_simulator import make_block_stimulus


def _build_posterior_from_autonormal(
    n_a_free: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract A_free posterior mean and diagonal covariance from param store.

    Reads the AutoNormal guide's ``locs.A_free`` and ``scales.A_free``
    parameters from the Pyro param store and constructs a flattened
    mean vector and diagonal covariance matrix for the A_free elements.

    Parameters
    ----------
    n_a_free : int
        Expected total number of A_free elements (N*N for the full
        A_free matrix).

    Returns
    -------
    posterior_mean : torch.Tensor, shape (n_a_free,)
        Flattened posterior mean of A_free.
    posterior_cov : torch.Tensor, shape (n_a_free, n_a_free)
        Diagonal posterior covariance of A_free.
    """
    store = pyro.get_param_store()
    loc = store["AutoNormal.locs.A_free"].detach().clone().flatten()
    # AutoNormal stores the positive scale directly (softplus constraint
    # is applied internally by Pyro's param store).
    scale = store["AutoNormal.scales.A_free"].detach().clone().flatten()

    assert loc.shape[0] == n_a_free, (
        f"Expected {n_a_free} A_free params, got {loc.shape[0]}"
    )

    posterior_mean = loc.to(torch.float64)
    posterior_cov = torch.diag(scale.to(torch.float64) ** 2)

    return posterior_mean, posterior_cov


def _build_prior_for_a_free(
    n_a_free: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the prior mean and covariance for A_free parameters.

    Parameters
    ----------
    n_a_free : int
        Total number of A_free elements (N*N).

    Returns
    -------
    prior_mean : torch.Tensor, shape (n_a_free,)
        Prior mean (zeros).
    prior_cov : torch.Tensor, shape (n_a_free, n_a_free)
        Prior covariance (diagonal, LC_A_PRIOR_VARIANCE on diagonal).
    """
    prior_mean = torch.zeros(n_a_free, dtype=torch.float64)
    prior_cov = LC_A_PRIOR_VARIANCE * torch.eye(
        n_a_free, dtype=torch.float64
    )
    return prior_mean, prior_cov


def _a_index(i: int, j: int, n: int) -> int:
    """Map A_free[i, j] to flattened index (row-major).

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Flattened row-major index.
    """
    return i * n + j


@pytest.mark.slow
def test_bmr_agrees_with_elbo_ranking() -> None:
    """BMR ranking agrees with brute-force ELBO ranking.

    Generates a 3-region latent circuit with known connectivity:
    connections 0->1 and 1->2 present, 2->0 absent. Fits a full model
    (all A connections free), then uses BMR to score single-connection
    reductions. Independently fits each reduced model via SVI. Asserts
    that BMR and ELBO agree on direction.

    Timing constraint: duration=1.0, dt=0.05 (T=20 timepoints),
    num_steps=300 per fit. Total ~2 min on laptop.
    """
    total_start = time.perf_counter()

    # ---------------------------------------------------------------
    # 1. Generate synthetic data -- SMALL for speed
    # ---------------------------------------------------------------
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    N = 3

    # Ground-truth A with known connectivity:
    #   connection 0->1 (A[1,0]): present, strength 0.3
    #   connection 1->2 (A[2,1]): present, strength 0.25
    #   connection 2->0 (A[0,2]): ABSENT (zero)
    # Self-connections are always present (negative diagonal).
    A_true = torch.zeros(N, N, dtype=torch.float64)
    A_true[0, 0] = -1.0
    A_true[1, 1] = -1.0
    A_true[2, 2] = -1.0
    A_true[1, 0] = 0.3   # 0 -> 1
    A_true[2, 1] = 0.25  # 1 -> 2
    # A_true[0, 2] = 0.0  # 2 -> 0 absent (already zero)

    C_true = torch.zeros(N, 1, dtype=torch.float64)
    C_true[0, 0] = 0.5

    # Short stimulus: 2 blocks of 0.2s ON / 0.2s OFF
    stim = make_block_stimulus(
        n_blocks=2, block_duration=0.2, rest_duration=0.2,
    )

    # Simulate with high SNR, SHORT duration for speed
    sim = simulate_latent_circuit(
        A_true, C_true, stim,
        duration=1.0, dt=0.05,
        SNR=10.0, seed=42,
    )
    assert not sim["simulation_diverged"], "Simulation diverged"

    observed_traj = sim["trajectories"]   # shape (T, N), T=20
    t_eval = sim["times"]                 # shape (T,)
    stimulus_fn = sim["stimulus"]         # PiecewiseConstantInput

    print(f"\nData: T={observed_traj.shape[0]}, N={N}")

    # ---------------------------------------------------------------
    # 2. Fit FULL model (all A connections free)
    # ---------------------------------------------------------------
    a_mask_full = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    model_args_full = (
        observed_traj, stimulus_fn, a_mask_full, c_mask, t_eval, 0.05,
    )

    num_steps = 300

    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    guide_full = create_guide(
        latent_circuit_dcm_model,
        guide_type="auto_normal",
        init_scale=0.01,
    )
    svi_result_full = run_svi(
        latent_circuit_dcm_model,
        guide_full,
        model_args=model_args_full,
        num_steps=num_steps,
        lr=0.01,
    )
    full_elbo = svi_result_full["final_loss"]
    print(f"Full model ELBO (neg): {full_elbo:.2f}")

    # ---------------------------------------------------------------
    # 3. Extract posterior mean/cov for BMR (A_free only)
    # ---------------------------------------------------------------
    n_a = N * N  # 9 A_free parameters
    posterior_mean, posterior_cov = _build_posterior_from_autonormal(n_a)
    prior_mean, prior_cov = _build_prior_for_a_free(n_a)

    # ---------------------------------------------------------------
    # 4. Define reduced architectures (single-connection pruning)
    # ---------------------------------------------------------------
    # A_free[i,j] -> flat index = i*N + j
    # Connection j->i maps to A[i,j]
    idx_2to0 = _a_index(0, 2, N)  # A[0,2], connection 2->0 (ABSENT)
    idx_0to1 = _a_index(1, 0, N)  # A[1,0], connection 0->1 (PRESENT)
    idx_0to2 = _a_index(2, 0, N)  # A[2,0], connection 0->2 (ABSENT)

    models = {
        "prune_2to0": {
            "description": "Prune truly-absent 2->0 (A[0,2])",
            "prune_indices": [idx_2to0],
            "a_mask": a_mask_full.clone(),
        },
        "prune_0to1": {
            "description": "Prune truly-present 0->1 (A[1,0])",
            "prune_indices": [idx_0to1],
            "a_mask": a_mask_full.clone(),
        },
        "prune_both_absent": {
            "description": "Prune two absent: 2->0, 0->2",
            "prune_indices": [idx_2to0, idx_0to2],
            "a_mask": a_mask_full.clone(),
        },
    }
    # Set up a_masks for brute-force ELBO fits
    models["prune_2to0"]["a_mask"][0, 2] = 0.0
    models["prune_0to1"]["a_mask"][1, 0] = 0.0
    models["prune_both_absent"]["a_mask"][0, 2] = 0.0
    models["prune_both_absent"]["a_mask"][2, 0] = 0.0

    # ---------------------------------------------------------------
    # 5. BMR scoring (analytical -- milliseconds)
    # ---------------------------------------------------------------
    bmr_start = time.perf_counter()
    bmr_scores: dict[str, float] = {}
    for name, spec in models.items():
        reduced_mean, reduced_cov = make_reduced_prior_zero_connection(
            prior_mean, prior_cov, spec["prune_indices"],
        )
        delta_f, _, _ = bayesian_model_reduction(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            reduced_mean,
            reduced_cov,
        )
        bmr_scores[name] = delta_f
    bmr_time = time.perf_counter() - bmr_start

    # Add full model baseline
    bmr_scores["full_model"] = 0.0

    # ---------------------------------------------------------------
    # 6. Brute-force ELBO: fit each reduced model independently
    # ---------------------------------------------------------------
    elbo_start = time.perf_counter()
    elbo_scores: dict[str, float] = {}
    elbo_scores["full_model"] = full_elbo

    for name, spec in models.items():
        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        pyro.clear_param_store()

        reduced_a_mask = spec["a_mask"]
        model_args_reduced = (
            observed_traj, stimulus_fn, reduced_a_mask, c_mask,
            t_eval, 0.05,
        )
        guide_reduced = create_guide(
            latent_circuit_dcm_model,
            guide_type="auto_normal",
            init_scale=0.01,
        )
        svi_result_reduced = run_svi(
            latent_circuit_dcm_model,
            guide_reduced,
            model_args=model_args_reduced,
            num_steps=num_steps,
            lr=0.01,
        )
        elbo_scores[name] = svi_result_reduced["final_loss"]
        print(f"  {name}: ELBO={svi_result_reduced['final_loss']:.2f}")

    elbo_time = time.perf_counter() - elbo_start

    # ---------------------------------------------------------------
    # 7. Compare rankings
    # ---------------------------------------------------------------
    print("\n=== BMR Scores (delta_F, higher = better) ===")
    for name in sorted(bmr_scores, key=lambda k: bmr_scores[k], reverse=True):
        print(f"  {name}: {bmr_scores[name]:.4f}")

    print("\n=== ELBO Scores (negative ELBO, lower = better) ===")
    for name in sorted(elbo_scores, key=lambda k: elbo_scores[k]):
        print(f"  {name}: {elbo_scores[name]:.4f}")

    # Convert ELBO to delta: lower neg-ELBO is better,
    # so delta = -(ELBO_reduced - ELBO_full) = ELBO_full - ELBO_reduced
    # Positive delta_elbo means reduced model is better.
    elbo_deltas: dict[str, float] = {}
    for name in models:
        elbo_deltas[name] = full_elbo - elbo_scores[name]
    elbo_deltas["full_model"] = 0.0

    elbo_ranking = sorted(
        elbo_deltas.keys(),
        key=lambda k: elbo_deltas[k],
        reverse=True,
    )

    print("\n=== ELBO Delta (positive = better than full) ===")
    for name in elbo_ranking:
        print(f"  {name}: {elbo_deltas[name]:.4f}")

    print(f"\nBMR time:  {bmr_time:.3f}s")
    print(f"ELBO time: {elbo_time:.3f}s")
    print(f"Speedup:   {elbo_time / max(bmr_time, 1e-6):.0f}x")

    total_time = time.perf_counter() - total_start
    print(f"Total time: {total_time:.1f}s")

    # --- Assertion 1: BMR identifies pruning absent connection as best ---
    bmr_ranking = sorted(
        bmr_scores.keys(),
        key=lambda k: bmr_scores[k],
        reverse=True,
    )
    truly_absent_models = {"prune_2to0", "prune_both_absent"}
    assert bmr_ranking[0] in truly_absent_models, (
        f"BMR top-1 model is {bmr_ranking[0]}, "
        f"expected one of {truly_absent_models}"
    )

    # --- Assertion 2: Pruning a present connection is NOT best ---
    assert bmr_ranking[0] != "prune_0to1", (
        "BMR incorrectly ranks pruning a present connection as best"
    )
    assert elbo_ranking[0] != "prune_0to1", (
        "ELBO incorrectly ranks pruning a present connection as best"
    )

    # --- Assertion 3: Positive Spearman correlation ---
    all_names = sorted(bmr_scores.keys())
    bmr_vals = [bmr_scores[n] for n in all_names]
    elbo_delta_vals = [elbo_deltas[n] for n in all_names]

    rho, pval = spearmanr(bmr_vals, elbo_delta_vals)
    print(f"\nSpearman correlation: rho={rho:.3f}, p={pval:.3f}")

    assert rho > 0, (
        f"Expected positive Spearman correlation between BMR and "
        f"ELBO delta, got rho={rho:.3f}"
    )

    # --- Assertion 4: BMR is much faster ---
    print(f"BMR is {elbo_time / max(bmr_time, 1e-6):.0f}x faster than ELBO")
    assert bmr_time < elbo_time, (
        f"Expected BMR ({bmr_time:.3f}s) to be faster than "
        f"brute-force ELBO ({elbo_time:.3f}s)"
    )
