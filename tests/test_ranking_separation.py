"""Ranking comparisons must report the evidence separation, not just agreement.

A disagreement between two engines on a pair of models whose free energies
differ by 0.2 nats is not the same finding as one on a pair separated by 40
nats: the first is a coin-flip on a near-tie, the second a real conflict. Job
55234748 seed 44 produced an agreement rate of 0.667 that could not be
interpreted, because ``compare_model_ranking`` recorded only the boolean.
"""

from __future__ import annotations

from validation.compare_results import compare_model_ranking


def test_pairwise_records_separations() -> None:
    """Each pair carries the SPM and VL evidence differences."""
    scenarios = [
        {"spm_F": 100.0, "pyro_elbo": 100.0},
        {"spm_F": 90.0, "pyro_elbo": 95.0},
    ]
    result = compare_model_ranking(scenarios)
    pair = result["pairwise_results"][0]

    assert pair["spm_delta"] == 10.0, (
        f"expected spm_delta = 100.0 - 90.0 = 10.0, got {pair.get('spm_delta')}"
    )
    assert pair["pyro_delta"] == 5.0, (
        f"expected pyro_delta = 100.0 - 95.0 = 5.0, got {pair.get('pyro_delta')}"
    )
    assert pair["min_abs_separation"] == 5.0, (
        "min_abs_separation is the smaller of the two |deltas| -- the "
        "conservative read of how close the call was; got "
        f"{pair.get('min_abs_separation')}"
    )


def test_near_tie_disagreement_is_flagged_as_close() -> None:
    """A disagreement on a near-tie is marked, so it cannot be over-read."""
    scenarios = [
        {"spm_F": 100.0, "pyro_elbo": 100.0},
        # SPM prefers model 0 by 0.05; VL prefers model 1 by 0.05.
        {"spm_F": 99.95, "pyro_elbo": 100.05},
    ]
    result = compare_model_ranking(scenarios, close_call_nats=1.0)
    pair = result["pairwise_results"][0]

    assert pair["agree"] is False
    assert pair["is_close_call"] is True, (
        "a disagreement with both separations under close_call_nats must be "
        "flagged as a close call"
    )
    assert result["disagreements_close"] == 1
    assert result["disagreements_separated"] == 0


def test_separated_disagreement_is_not_close() -> None:
    """A disagreement on a well-separated pair is NOT excused as a near-tie."""
    scenarios = [
        {"spm_F": 100.0, "pyro_elbo": 100.0},
        # Both engines are confident, and they disagree. This is the serious case.
        {"spm_F": 60.0, "pyro_elbo": 140.0},
    ]
    result = compare_model_ranking(scenarios, close_call_nats=1.0)
    pair = result["pairwise_results"][0]

    assert pair["agree"] is False
    assert pair["is_close_call"] is False
    assert result["disagreements_close"] == 0
    assert result["disagreements_separated"] == 1


def test_agreement_rate_is_unchanged() -> None:
    """The existing agreement_rate contract is preserved."""
    scenarios = [
        {"spm_F": 3.0, "pyro_elbo": 3.0},
        {"spm_F": 2.0, "pyro_elbo": 2.0},
        {"spm_F": 1.0, "pyro_elbo": 0.0},
    ]
    result = compare_model_ranking(scenarios)
    assert result["agreement_rate"] == 1.0
    assert result["total_pairs"] == 3
    assert result["agreements"] == 3
