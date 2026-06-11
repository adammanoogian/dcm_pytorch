"""SPM-gated VL-vs-SPM12 cross-validation test (VLSPM-01/02/03).

Runs the actual cross-validation: fit the Phase 28 Variational-Laplace engine
on a prior-matched reciprocal-asymmetric N=2 spectral DCM problem, inject the
IDENTICAL CSD into SPM12 (Plan 32-01 bridge), run ``spm_dcm_fmri_csd``
(``spm_nlsi_GN``), and assert -- as HARD gates -- that the two engines agree in
free-parameter space (Ep ~10%, S1/S2), on matched free energy (5% on the
identical injected CSD, the user-chosen strict gate), and on relative
cross-model ranking (>= 0.80 over >=3 masks, S3-safe). The asymmetric reciprocal
ground truth must produce an asymmetric SPM ``Ep.A`` (S4); the suite asserts no
forbidden element-wise ``Cp`` metric is present.

All tests require MATLAB + SPM12 and are marked ``@pytest.mark.spm`` +
``@pytest.mark.slow``; they auto-skip when MATLAB is unavailable
(``skipif(not check_matlab_available())``). The local MATLAB license server is
unreachable (FlexLM -15), so on this laptop the test SKIPS; the real run is the
M3 sbatch job ``cluster/sbatch/spm_cross_validation.sbatch`` (32-03 execution
deviation) where MATLAB R2022a + SPM12 are licensed.

References
----------
SPM12 source: spm_dcm_fmri_csd.m, spm_nlsi_GN.m.
.planning/phases/32-spm12-cross-validation/32-03-PLAN.md
    VLSPM-01/02/03 objective and the S1-S4 gating contract.
"""

from __future__ import annotations

import pytest

from validation.run_validation import check_matlab_available
from validation.run_vl_validation import run_vl_spectral_dcm_validation

pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not check_matlab_available(),
        reason="MATLAB/SPM12 not available",
    ),
]


def _print_cross_validation_table(result: dict) -> None:
    """Print the VL-vs-SPM12 comparison table for debugging.

    Surfaces the Ep free-space comparison, the matched-F relative error (the
    headline number), the cross-model ranking agreement, and the S4 asymmetry
    readout so a gate miss is fully visible in the captured output.

    Parameters
    ----------
    result : dict
        Output of :func:`validation.run_vl_validation.run_vl_spectral_dcm_validation`.
    """
    vl_a = result["vl_A_free"]
    spm_a = result["spm_Ep_A"]
    ep = result["ep_comparison"]
    f_cmp = result["matched_f_comparison"]
    ranking = result["ranking"]
    a01, a10 = result["ep_asymmetry"]
    n = vl_a.shape[0]

    print(f"\n{'=' * 64}")
    print(f"  VL vs SPM12 Cross-Validation (seed={result['seed']})")
    print(f"{'=' * 64}")
    print(f"  {'Element':<12} {'VL A_free':>12} {'SPM Ep.A':>12}")
    print(f"  {'-' * 38}")
    for i in range(n):
        for j in range(n):
            print(
                f"  A[{i},{j}]{'':<6} {vl_a[i, j]:>12.5f} {spm_a[i, j]:>12.5f}"
            )

    print("\n  --- Ep free-parameter space (S1/S2, 10% gate) ---")
    print(f"  max_relative_error:  {ep['max_relative_error']:.4f}")
    print(f"  mean_relative_error: {ep['mean_relative_error']:.4f}")
    print(f"  within_tolerance:    {ep['within_tolerance']}")

    print("\n  --- Matched free energy (HEADLINE, strict 5% gate) ---")
    print(f"  VL free_energy[-1]:  {f_cmp['vl_free_energy']:.4f}")
    print(f"  SPM DCM.F:           {f_cmp['spm_F']:.4f}")
    print(f"  relative_error:      {f_cmp['relative_error']:.4f}")
    print(f"  within_tolerance:    {f_cmp['within_tolerance']}")

    print("\n  --- Cross-model ranking (S3-safe, relative, >=0.80) ---")
    print(f"  agreement_rate:      {ranking['agreement_rate']:.4f}")
    print(
        f"  agreements/pairs:    {ranking['agreements']}/{ranking['total_pairs']}"
    )

    print("\n  --- S4 asymmetry readout ---")
    print(f"  Ep.A[0,1]={a01:.5f}  Ep.A[1,0]={a10:.5f}")
    print(f"{'=' * 64}\n")


def test_vl_matches_spm_on_matched_reciprocal_problem() -> None:
    """VL agrees with SPM12 on the matched reciprocal-asymmetric problem.

    HARD gates (pass/fail):

    1. Ep within ~10% in FREE-parameter space (``A_free`` vs ``Ep.A``, S1/S2).
    2. Relative cross-model ranking agreement >= 0.80 (S3-safe; never absolute
       F across masks).
    3. Matched free energy within 5% on the IDENTICAL injected CSD (the
       user-chosen strict gate; the printed table surfaces ``relative_error``
       so a miss is visible).

    Plus the S4 sanity check (asymmetric ground truth => asymmetric SPM
    ``Ep.A``; equality would signal a transpose) and a negative guard that the
    suite performs no forbidden element-wise ``Cp`` comparison.
    """
    result = run_vl_spectral_dcm_validation(seed=42, n_regions=2, max_iter=64)
    _print_cross_validation_table(result)

    # --- Gate 1: Ep within ~10% in free-parameter space (S1/S2) ------------
    assert result["ep_comparison"]["within_tolerance"], (
        f"VL A_free vs SPM Ep.A out of 10% tolerance in free space. "
        f"max={result['ep_comparison']['max_relative_error']:.4f}, "
        f"mean={result['ep_comparison']['mean_relative_error']:.4f}."
    )

    # --- Gate 2: relative cross-model ranking agreement (S3-safe) -----------
    assert result["ranking"]["agreement_rate"] >= 0.80, (
        f"Cross-model ranking agreement "
        f"{result['ranking']['agreement_rate']:.4f} < 0.80 "
        f"({result['ranking']['agreements']}/"
        f"{result['ranking']['total_pairs']} pairs)."
    )

    # --- Gate 3: matched free energy within 5% on the IDENTICAL CSD ---------
    # HARD gate per the user decision (strict-5%-F). A genuine miss is a real
    # result to ESCALATE (report the relative_error), never to silently relax.
    assert result["matched_f_comparison"]["within_tolerance"], (
        f"Matched free energy out of strict 5% tolerance on the IDENTICAL "
        f"injected CSD. relative_error="
        f"{result['matched_f_comparison']['relative_error']:.4f} "
        f"(VL F={result['matched_f_comparison']['vl_free_energy']:.4f}, "
        f"SPM F={result['matched_f_comparison']['spm_F']:.4f})."
    )

    # --- S4 sanity: asymmetric ground truth => asymmetric SPM Ep.A ----------
    a01, a10 = result["ep_asymmetry"]
    assert a01 != a10, (
        f"S4 violation: SPM Ep.A[0,1]={a01} == Ep.A[1,0]={a10}; the "
        f"asymmetric reciprocal ground truth (0.15 vs 0.10) must produce an "
        f"asymmetric Ep.A. Equality signals a transpose in the CSD injection."
    )

    # --- Negative guard: no forbidden element-wise Cp metric (S3) -----------
    # The cross-model criterion is ranking["agreement_rate"] (relative), never
    # absolute F across masks and never an element-wise Cp comparison.
    assert not any(
        "cp" in key.lower() and "compar" in key.lower() for key in result
    ), (
        "result must not contain an element-wise Cp comparison metric "
        "(S3): cross-model agreement is relative ranking only."
    )
