"""Free energy should track fit quality; flag scenarios where it does not.

Job 55235094 showed VL and SPM diverging by 18-72 nats on the `single_direction`
mask, with the forward, the iteration budget and the mask specification all
eliminated as causes. The remaining question is which engine is right. The
decisive signal is whether an engine reports the HIGHER free energy while
achieving the WORSE fit to the same data -- free energy trades fit against
complexity, so a higher F with a larger residual is only defensible if the
model is correspondingly simpler. Here both engines fit the SAME model under
the SAME mask, so complexity is identical and the trade-off cannot explain it.
"""

from __future__ import annotations

from validation.compare_results import flag_free_energy_fit_inconsistency


def test_flags_higher_f_with_worse_residual() -> None:
    """Higher F but larger residual, same model: flagged."""
    rows = [
        {"name": "single_direction", "vl_F": 161.8, "spm_F": 223.9,
         "vl_residual": 12.0, "spm_residual": 31.0},
    ]
    out = flag_free_energy_fit_inconsistency(rows)
    row = out["per_scenario"][0]

    assert row["higher_f_engine"] == "spm"
    assert row["lower_residual_engine"] == "vl"
    assert row["inconsistent"] is True
    assert out["n_inconsistent"] == 1
    assert out["inconsistent_scenarios"] == ["single_direction"]


def test_consistent_when_same_engine_wins_both() -> None:
    """The engine with the higher F also fits better: consistent."""
    rows = [
        {"name": "full_reciprocal", "vl_F": 294.6, "spm_F": 289.9,
         "vl_residual": 9.0, "spm_residual": 11.0},
    ]
    out = flag_free_energy_fit_inconsistency(rows)
    assert out["per_scenario"][0]["inconsistent"] is False
    assert out["n_inconsistent"] == 0


def test_near_ties_are_not_flagged() -> None:
    """Differences below tolerance are not called inconsistent."""
    rows = [
        {"name": "diagonal_only", "vl_F": 100.0, "spm_F": 100.02,
         "vl_residual": 5.0, "spm_residual": 5.001},
    ]
    out = flag_free_energy_fit_inconsistency(
        rows, f_tolerance=0.1, residual_tolerance=0.01,
    )
    assert out["per_scenario"][0]["inconsistent"] is False


def test_missing_residuals_are_reported_not_guessed() -> None:
    """A scenario without residuals is marked unknown, never assumed fine."""
    rows = [{"name": "x", "vl_F": 1.0, "spm_F": 2.0,
             "vl_residual": None, "spm_residual": None}]
    out = flag_free_energy_fit_inconsistency(rows)
    row = out["per_scenario"][0]
    assert row["inconsistent"] is None
    assert out["n_unknown"] == 1
    assert out["n_inconsistent"] == 0
