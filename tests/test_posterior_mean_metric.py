"""The near-zero guard must key on the parameter, not on one engine's estimate.

`compare_posterior_means` switches from relative to absolute error when
``|ref_A| <= near_zero_threshold``. That test is applied to the REFERENCE (SPM)
estimate alone, so it never fires when SPM reports a sizeable value for a
parameter whose true value is ~0. On the N=2 parity problem the A_free diagonal
is a log-deviation from a fixed -0.5 self-connection -- correct value ~0 -- and
SPM reported -0.0674. The guard stayed off, relative error divided by 0.0674,
and the gate reported 1.0085 for what is a 0.068 absolute disagreement
(job 55234501).

Relative error is unstable when EITHER estimate is near zero, so the guard must
consider both -- and must key on ground truth when it is available.
"""

from __future__ import annotations

import numpy as np

from validation.compare_results import compare_posterior_means


def test_guard_fires_when_the_pyro_estimate_is_near_zero() -> None:
    """A near-zero VL estimate takes the absolute branch even if SPM's is not."""
    pyro = np.array([[0.00057]])
    ref = np.array([[-0.06741]])

    out = compare_posterior_means(
        pyro, ref, near_zero_threshold=0.01, near_zero_atol=0.02,
    )
    err = float(out["element_errors"][0, 0])

    assert abs(err - 0.06798) < 1e-4, (
        "expected the ABSOLUTE difference 0.06798 (the guard should fire "
        f"because the pyro estimate 0.00057 is near zero), got {err:.6f}. "
        "A value near 1.0085 means the relative branch was taken."
    )
    # It is still a real disagreement: 0.068 exceeds the 0.02 absolute tolerance.
    assert out["within_tolerance"] is False


def test_ground_truth_keys_the_guard_when_supplied() -> None:
    """With true_A given, the near-zero decision follows the TRUE value."""
    # Both engines report sizeable values, but the parameter is truly ~0.
    pyro = np.array([[0.05]])
    ref = np.array([[-0.06]])
    true = np.array([[0.0]])

    out = compare_posterior_means(
        pyro, ref, near_zero_threshold=0.01, near_zero_atol=0.02, true_A=true,
    )
    err = float(out["element_errors"][0, 0])
    assert abs(err - 0.11) < 1e-9, (
        f"expected absolute error 0.11 keyed on the true value, got {err:.6f}"
    )


def test_well_identified_parameters_still_use_relative_error() -> None:
    """Both estimates comfortably non-zero: relative error, unchanged."""
    pyro = np.array([[0.14913]])
    ref = np.array([[0.07281]])

    out = compare_posterior_means(
        pyro, ref, near_zero_threshold=0.01, near_zero_atol=0.02,
    )
    err = float(out["element_errors"][0, 0])
    expected = abs(0.14913 - 0.07281) / 0.07281
    assert abs(err - expected) < 1e-6, (
        f"expected relative error {expected:.6f}, got {err:.6f}"
    )


def test_agreeing_matrix_passes() -> None:
    """A genuinely agreeing pair still passes the gate."""
    pyro = np.array([[0.150, 0.100], [0.0005, 0.0004]])
    ref = np.array([[0.151, 0.099], [0.0006, 0.0003]])
    out = compare_posterior_means(pyro, ref)
    assert out["within_tolerance"] is True
