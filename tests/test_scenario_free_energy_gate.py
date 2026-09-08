"""Matched-F must be checked on every scenario, not just the full model.

The parity suite fits several ``a_mask`` scenarios for the ranking sweep but
gated matched-F on the full model alone, discarding the reduced-model free
energies after ranking. Job 55234938 seed 44 showed a full-model relative error
of 0.0162 (inside the 5% gate) alongside a ~54-nat divergence on a REDUCED
model -- invisible to every gate. Since BMR and structure selection run on
reduced models, that is the regime that matters most.
"""

from __future__ import annotations

from validation.compare_results import compare_scenario_free_energies


def test_flags_a_diverging_reduced_model() -> None:
    """A reduced model that diverges fails, even when the full model agrees."""
    scenarios = [
        {"name": "full", "spm_F": 289.9352, "pyro_elbo": 294.6323},
        # The seed-44 pattern: full model agrees, a reduced model does not.
        {"name": "reduced_a", "spm_F": 291.2368, "pyro_elbo": 241.5480},
    ]
    result = compare_scenario_free_energies(scenarios, rel_tolerance=0.05)

    per = {r["name"]: r for r in result["per_scenario"]}
    assert per["full"]["within_tolerance"] is True
    assert per["reduced_a"]["within_tolerance"] is False
    assert result["all_within_tolerance"] is False
    assert result["worst_scenario"] == "reduced_a"
    assert result["worst_relative_error"] > 0.05


def test_passes_when_every_scenario_agrees() -> None:
    """All scenarios inside tolerance means the gate passes."""
    scenarios = [
        {"name": "full", "spm_F": 100.0, "pyro_elbo": 101.0},
        {"name": "reduced_a", "spm_F": 90.0, "pyro_elbo": 90.5},
    ]
    result = compare_scenario_free_energies(scenarios, rel_tolerance=0.05)
    assert result["all_within_tolerance"] is True
    assert result["worst_relative_error"] < 0.05


def test_worst_scenario_is_the_largest_relative_error() -> None:
    """``worst_scenario`` names the biggest divergence, not the first failure."""
    scenarios = [
        {"name": "a", "spm_F": 100.0, "pyro_elbo": 108.0},   # 0.08
        {"name": "b", "spm_F": 100.0, "pyro_elbo": 130.0},   # 0.30
        {"name": "c", "spm_F": 100.0, "pyro_elbo": 101.0},   # 0.01
    ]
    result = compare_scenario_free_energies(scenarios, rel_tolerance=0.05)
    assert result["worst_scenario"] == "b"
    assert abs(result["worst_relative_error"] - 0.30) < 1e-9
    assert result["n_scenarios_failing"] == 2
