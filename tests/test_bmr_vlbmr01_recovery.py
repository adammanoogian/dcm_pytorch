"""VLBMR-01: BMR relative-ranking structure recovery on a real VL posterior.

This is the PRIMARY, defensible model-comparison result for v0.7.0. A SPARSE
spectral circuit is fit with Variational Laplace, the full ``A_free`` posterior
covariance is sliced out, and
:func:`pyro_dcm.model_selection.bmr.rank_connections` ranks the off-diagonal
connections by single-prune cost. The claim is that the top-K most-essential
edges equal the true present edges, with a positive separation gap and the cut
landing at K.

The result is RELATIVE ranking + separation gap, NEVER absolute delta-F. Under
VL the Laplace posterior is sharply overconfident at high SNR, so every
single-prune delta-F is driven deeply negative -- present or absent edges alike
(pitfall C1 / cluster job 55772525: a truly-absent edge scored
delta_F = -115.9, indistinguishable by sign). This module therefore asserts on
the ORDERING and the separation cut only, and NEVER gates on an absolute-nat
threshold for ``separation_gap``.

The present edges are RECIPROCAL (bidirectional) by construction. A purely
feed-forward chain (e.g. ``0->1->2->3``) is NOT identifiable from a stationary
spectral CSD: a strictly lower-triangular ``A`` with isotropic noise produces a
cross-spectral density bit-identical to the empty graph (verified: relative CSD
difference 0.0), so VL recovers the zero off-diagonal and BMR has no signal to
rank. Reciprocal edges create the cross-spectral coupling spectral DCM can see,
making the true structure recoverable -- this is a genuine spectral-DCM
identifiability property, documented in 31-01-SUMMARY.md, not a test artifact.

References
----------
.planning/phases/31-bmr-validation-tempering/31-01-PLAN.md
    VLBMR-01 objective and gating contract.
pyro_dcm.model_selection.bmr.rank_connections
    The relative single-prune ranking under test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.bmr_recovery import (
    bmr_tensors_from_vl_result,
    make_sparse_ground_truth_A,
    offdiag_indices,
)
from pyro_dcm.inference import (  # type: ignore[import-untyped]
    SpectralDCMForward,
    run_variational_laplace_generic,
)
from pyro_dcm.model_selection.bmr import (  # type: ignore[import-untyped]
    rank_connections,
)
from pyro_dcm.simulators.spectral_simulator import (  # type: ignore[import-untyped]
    simulate_spectral_dcm,
)

# Five seeds suffice for a binary structure-recovery assertion (each spectral
# VL fit is ~1.4-7.5s; 2 cases x 5 seeds stays well under the 3-min budget).
SEEDS = list(range(42, 47))
PRIOR_VARIANCE = 1.0 / 64.0


@pytest.mark.vl
@pytest.mark.parametrize(
    ("n_regions", "present_edges"),
    [
        (2, [(0, 1), (1, 0)]),
        (4, [(1, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3)]),
    ],
    ids=["spectral_N2_K2", "spectral_N4_K6"],
)
def test_vlbmr01_relative_ranking_recovers_structure(
    n_regions: int,
    present_edges: list[tuple[int, int]],
) -> None:
    """Top-K essential off-diagonal edges == true present edges across seeds.

    For each seed: simulate a sparse spectral circuit, VL-fit the full
    (all-ones ``a_mask``) model, slice the full ``A_free`` covariance, rank the
    off-diagonal connections by single-prune cost, and check the K most
    essential equal the true present edges (set equality). The aggregate gate
    requires recovery for every seed; per recovered seed the separation gap is
    positive and (when an absent prunable edge exists) the cut lands at K. No
    absolute-delta-F threshold is asserted (pitfall C1).
    """
    true_idx = {i * n_regions + j for (i, j) in present_edges}
    k = len(true_idx)
    # The "cut lands at K" gate (separation_after_rank == K) only makes sense
    # when some prunable edge is genuinely ABSENT -- i.e. there is an
    # essential/non-essential boundary to detect. For N=2 the only identifiable
    # sparse structure is the fully reciprocal pair (both off-diagonals
    # present; a single directed N=2 edge is feed-forward and unidentifiable),
    # so K == len(prunable): every edge is essential and the cut is degenerate
    # by construction. There we assert recovery + a positive gap only.
    n_prunable = n_regions * (n_regions - 1)
    has_absent_edges = k < n_prunable

    recovered_flags: list[bool] = []
    failures: list[str] = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        A_true = make_sparse_ground_truth_A(n_regions, present_edges)
        sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=32, seed=seed)
        csd_obs = sim["csd"].to(torch.complex128)
        freqs = sim["freqs"].double()

        forward = SpectralDCMForward()
        result = run_variational_laplace_generic(
            forward,
            observed=csd_obs,
            a_mask=torch.ones(n_regions, n_regions, dtype=torch.float64),
            n_regions=n_regions,
            max_iter=64,
            prior_variance=PRIOR_VARIANCE,
            context={"freqs": freqs},
        )

        pm, pc, prm, prc = bmr_tensors_from_vl_result(
            result, n_regions, prior_variance=PRIOR_VARIANCE,
        )
        prunable = offdiag_indices(n_regions)
        ranked = rank_connections(pm, pc, prm, prc, prunable)

        topk = {entry["index"] for entry in ranked["ranked"][:k]}
        recovered = topk == true_idx
        recovered_flags.append(recovered)

        print(
            f"  N={n_regions} seed={seed}: recovered={recovered} "
            f"topk={sorted(topk)} true={sorted(true_idx)} "
            f"separation_gap={ranked['separation_gap']:.4f} "
            f"separation_after_rank={ranked['separation_after_rank']}"
        )

        if recovered:
            assert ranked["separation_gap"] > 0, (
                f"N={n_regions} seed={seed}: separation_gap must be > 0 "
                f"when structure is recovered; got "
                f"{ranked['separation_gap']:.6f}"
            )
            if has_absent_edges:
                assert ranked["separation_after_rank"] == k, (
                    f"N={n_regions} seed={seed}: separation cut must land at "
                    f"K; expected separation_after_rank == {k}, got "
                    f"{ranked['separation_after_rank']}"
                )
        else:
            failures.append(
                f"seed={seed}: topk={sorted(topk)} != true={sorted(true_idx)}"
            )

    n_recovered = sum(recovered_flags)
    print(
        f"N={n_regions}: recovered {n_recovered}/{len(SEEDS)} seeds "
        f"(true present edges {sorted(true_idx)}, K={k})"
    )

    assert n_recovered == len(SEEDS), (
        f"N={n_regions}: VLBMR-01 expected all {len(SEEDS)} seeds to recover "
        f"the true structure; got {n_recovered}/{len(SEEDS)}. "
        f"Failures: {failures}"
    )
