"""Cross-validation tests for task DCM vs SPM12 (VAL-01).

Tests that Pyro SVI posterior means for effective connectivity (A_free)
match SPM12 Variational Laplace posteriors (Ep.A) within 10% relative
error on simulated 3-region task-DCM data.

All tests require MATLAB + SPM12 and are marked with ``@pytest.mark.spm``
and ``@pytest.mark.slow``. Tests auto-skip when MATLAB is unavailable.

References
----------
SPM12 source: spm_dcm_estimate.m (Variational Laplace).
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.run_validation import (
    check_matlab_available,
    run_task_dcm_validation,
)

pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not check_matlab_available(),
        reason="MATLAB/SPM12 not available",
    ),
]


def _print_comparison_table(
    result: dict,
    label: str = "Task DCM",
) -> None:
    """Print element-wise comparison table for debugging.

    Parameters
    ----------
    result : dict
        Output from ``run_task_dcm_validation``.
    label : str
        Label for the table header.
    """
    A_true = result["A_true_free"]
    pyro_A = result["pyro_A_free"]
    spm_A = result["spm_Ep_A"]
    errors = result["comparison"]["element_errors"]
    N = A_true.shape[0]

    print(f"\n{'=' * 60}")
    print(f"  {label} Cross-Validation (seed={result['seed']})")
    print(f"{'=' * 60}")
    print(
        f"  {'Element':<12} {'True':>8} {'Pyro':>8} "
        f"{'SPM12':>8} {'Error':>8}"
    )
    print(f"  {'-' * 48}")

    for i in range(N):
        for j in range(N):
            elem = f"A[{i},{j}]"
            print(
                f"  {elem:<12} {A_true[i, j]:>8.4f} "
                f"{pyro_A[i, j]:>8.4f} "
                f"{spm_A[i, j]:>8.4f} "
                f"{errors[i, j]:>8.4f}"
            )

    comp = result["comparison"]
    print(f"\n  Max error:  {comp['max_relative_error']:.4f}")
    print(f"  Mean error: {comp['mean_relative_error']:.4f}")
    print(f"  Within tol: {comp['within_tolerance']}")
    print(f"  SPM F:      {result['spm_F']:.2f}")
    print(
        f"  Pyro loss:  {result['pyro_final_loss']:.2f}"
    )
    print(f"{'=' * 60}\n")


class TestTaskDCMvsSPM:
    """Cross-validation tests for task DCM against SPM12."""

    def test_task_dcm_vs_spm_relative_error(self) -> None:
        """VAL-01: Task DCM posterior means within tolerance of SPM12.

        Runs task DCM validation with seed=42 and checks:
        - Overall within_tolerance (10% relative / 0.02 absolute)
        - Max relative error < 15% (slack for individual elements)
        - Mean relative error < 10%
        """
        result = run_task_dcm_validation(seed=42)
        _print_comparison_table(result)

        comp = result["comparison"]
        assert comp["within_tolerance"], (
            f"Task DCM vs SPM12 out of tolerance. "
            f"Max error: {comp['max_relative_error']:.4f}, "
            f"Mean error: {comp['mean_relative_error']:.4f}"
        )
        assert comp["max_relative_error"] < 0.15, (
            f"Max relative error {comp['max_relative_error']:.4f} "
            f">= 0.15"
        )
        assert comp["mean_relative_error"] < 0.10, (
            f"Mean relative error "
            f"{comp['mean_relative_error']:.4f} >= 0.10"
        )

    def test_task_dcm_vs_spm_multiple_seeds(self) -> None:
        """VAL-01: Consistency across multiple random seeds.

        Runs validation for seeds [42, 123, 456] and checks
        that the median max_relative_error is below 10%.
        """
        seeds = [42, 123, 456]
        max_errors = []
        mean_errors = []

        for seed in seeds:
            result = run_task_dcm_validation(seed=seed)
            _print_comparison_table(
                result, label=f"Task DCM seed={seed}"
            )

            comp = result["comparison"]
            max_errors.append(comp["max_relative_error"])
            mean_errors.append(comp["mean_relative_error"])

            if comp["mean_relative_error"] >= 0.10:
                print(
                    f"  WARNING: seed={seed} exceeds 10% "
                    f"mean threshold "
                    f"({comp['mean_relative_error']:.4f}). "
                    f"Root cause: VL vs SVI local optima "
                    f"or parameterization differences."
                )

        median_max = float(np.median(max_errors))
        median_mean = float(np.median(mean_errors))

        print(f"\nMulti-seed summary:")
        print(f"  Max errors:  {max_errors}")
        print(f"  Mean errors: {mean_errors}")
        print(f"  Median max:  {median_max:.4f}")
        print(f"  Median mean: {median_mean:.4f}")

        assert median_max < 0.10, (
            f"Median max_relative_error {median_max:.4f} "
            f">= 0.10 across seeds {seeds}"
        )

    def test_task_dcm_spm_sign_agreement(self) -> None:
        """VAL-01: Sign agreement between SPM12 and Pyro posteriors.

        Checks that off-diagonal elements agree on sign
        (positive/negative/zero with 0.01 tolerance).
        Agreement must be >= 85%.
        """
        result = run_task_dcm_validation(seed=42)
        _print_comparison_table(
            result, label="Task DCM Sign Agreement"
        )

        pyro_A = result["pyro_A_free"]
        spm_A = result["spm_Ep_A"]
        N = pyro_A.shape[0]

        zero_tol = 0.01
        agreements = 0
        total = 0

        print("\nSign agreement details:")
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # skip diagonal

                pyro_sign = _classify_sign(
                    pyro_A[i, j], zero_tol
                )
                spm_sign = _classify_sign(
                    spm_A[i, j], zero_tol
                )
                agree = pyro_sign == spm_sign
                total += 1
                if agree:
                    agreements += 1

                status = "OK" if agree else "MISMATCH"
                print(
                    f"  A[{i},{j}]: Pyro={pyro_sign:>4s} "
                    f"({pyro_A[i, j]:>7.4f}) "
                    f"SPM={spm_sign:>4s} "
                    f"({spm_A[i, j]:>7.4f}) "
                    f"[{status}]"
                )

        rate = agreements / total if total > 0 else 0.0
        print(
            f"\nSign agreement: {agreements}/{total} "
            f"= {rate:.2%}"
        )

        assert rate >= 0.85, (
            f"Sign agreement {rate:.2%} < 85%. "
            f"{agreements}/{total} elements agree."
        )


def _classify_sign(
    value: float, tol: float = 0.01
) -> str:
    """Classify a value as positive, negative, or zero.

    Parameters
    ----------
    value : float
        Value to classify.
    tol : float
        Tolerance for zero classification.

    Returns
    -------
    str
        One of ``'pos'``, ``'neg'``, or ``'zero'``.
    """
    if abs(value) < tol:
        return "zero"
    return "pos" if value > 0 else "neg"
