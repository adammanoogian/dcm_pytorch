"""Phase 14 bilinear simulator integration tests.

Covers:
- SIM-03 primary gate: torch.equal bit-exactness when B_list=None vs no-kwarg call.
- SIM-03 secondary gate: bilinear mode produces BOLD distinguishable from linear null.
- SIM-04: return dict contains 'B_list' and 'stimulus_mod' keys.
- SIM-05: dt-invariance at atol=1e-4 under rk4 fixed-step integration.
  - Bilinear mode (primary).
  - Linear mode (L5 regression symmetry).

Fixture notes:
- B off-diagonal magnitude = 0.1 (L4 locked): keeps stability monitor silent.
- Solver = 'rk4' for SIM-05: fixed-step is required for dt-invariance logic
  (14-RESEARCH.md Section 6.3 HIGH-confidence answer).
- SNR = -1 everywhere: no-noise path for deterministic comparison.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.simulators import (
    make_block_stimulus,
    make_epoch_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils import PiecewiseConstantInput


def test_bilinear_arg_none_matches_no_kwarg() -> None:
    """SIM-03: B_list=None bit-exact to pre-Phase-14 no-kwarg call.

    Tests the structural short-circuit: simulate_task_dcm(...) with no bilinear
    kwargs must equal simulate_task_dcm(..., B_list=None, stimulus_mod=None,
    n_driving_inputs=None) byte-for-byte (same seed, same stimulus).
    """
    A = make_random_stable_A(n_regions=3, density=0.5, seed=42)
    # C follows the rest of the task-DCM test suite (0.25 on region 0 only).
    # Using unit-amplitude inputs (e.g. torch.eye(3,1)) destabilizes dopri5's
    # adaptive step for this A (step underflow to 0.0).
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
    stim = make_block_stimulus(
        n_blocks=3, block_duration=20.0, rest_duration=20.0, n_inputs=1,
    )

    # Path 1: pre-Phase-14 call pattern (no bilinear kwargs).
    result_a = simulate_task_dcm(
        A, C, stim, duration=120.0, TR=2.0, SNR=-1, seed=7,
    )

    # Path 2: post-Phase-14 call pattern with explicit None bilinear kwargs.
    result_b = simulate_task_dcm(
        A, C, stim, duration=120.0, TR=2.0, SNR=-1, seed=7,
        B_list=None, stimulus_mod=None, n_driving_inputs=None,
    )

    # Structural bit-exact guarantee.
    assert torch.equal(result_a["bold_clean"], result_b["bold_clean"])
    assert torch.equal(result_a["bold"], result_b["bold"])
    assert torch.equal(result_a["neural"], result_b["neural"])

    # New keys are None in both.
    assert result_a["B_list"] is None and result_b["B_list"] is None
    assert result_a["stimulus_mod"] is None and result_b["stimulus_mod"] is None


def test_bilinear_output_distinguishable_from_linear() -> None:
    """SIM-03: bilinear BOLD numerically differs from linear null on same seed.

    Same A, C, stimulus, seed. Linear mode (B_list=None) vs bilinear mode with
    non-zero B[0, 1, 0] = 0.2 modulated by an epoch stimulus. BOLD difference
    must exceed 0.01 max-abs margin (ample; actual difference will be larger).
    """
    A = make_random_stable_A(n_regions=3, density=0.5, seed=42)
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
    stim = make_block_stimulus(
        n_blocks=3, block_duration=20.0, rest_duration=20.0, n_inputs=1,
    )
    B = torch.zeros(1, 3, 3, dtype=torch.float64)
    B[0, 1, 0] = 0.2   # modulator 0 strengthens 1<-0
    mod = make_epoch_stimulus(
        event_times=[50.0], event_durations=[20.0], event_amplitudes=[1.0],
        duration=120.0, dt=0.01, n_inputs=1,
    )

    common = dict(
        A=A, C=C, stimulus=stim, duration=120.0, TR=2.0, SNR=-1,
        seed=7, solver="rk4",
    )
    result_linear = simulate_task_dcm(**common)
    result_bilinear = simulate_task_dcm(**common, B_list=B, stimulus_mod=mod)

    diff = (
        result_bilinear["bold_clean"] - result_linear["bold_clean"]
    ).abs().max()
    assert diff > 0.01, (
        f"bilinear BOLD indistinguishable from linear null: "
        f"max|diff|={diff.item():.6f} (expected > 0.01)"
    )


def test_return_dict_has_bilinear_keys() -> None:
    """SIM-04: return dict contains 'B_list' and 'stimulus_mod' keys in both modes.

    Linear mode: both None. Bilinear mode: B_list is the stacked tensor,
    stimulus_mod is the constructed PiecewiseConstantInput.
    """
    A = make_random_stable_A(n_regions=3, density=0.5, seed=42)
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
    stim = make_block_stimulus(
        n_blocks=2, block_duration=20.0, rest_duration=20.0, n_inputs=1,
    )

    # Linear mode
    r_linear = simulate_task_dcm(A, C, stim, duration=80.0, SNR=-1, seed=7)
    assert "B_list" in r_linear
    assert "stimulus_mod" in r_linear
    assert r_linear["B_list"] is None
    assert r_linear["stimulus_mod"] is None

    # Bilinear mode
    B = torch.zeros(1, 3, 3, dtype=torch.float64)
    B[0, 1, 0] = 0.1
    mod = make_epoch_stimulus(
        event_times=[30.0], event_durations=[15.0], event_amplitudes=[1.0],
        duration=80.0, dt=0.01, n_inputs=1,
    )
    r_bilin = simulate_task_dcm(
        A, C, stim, duration=80.0, SNR=-1, seed=7, solver="rk4",
        B_list=B, stimulus_mod=mod,
    )
    assert r_bilin["B_list"] is not None
    assert r_bilin["B_list"].shape == (1, 3, 3)
    assert r_bilin["stimulus_mod"] is not None
    # stimulus_mod should be a PiecewiseConstantInput (not a dict).
    assert isinstance(r_bilin["stimulus_mod"], PiecewiseConstantInput)


def test_dt_invariance_bilinear() -> None:
    """SIM-05: dt=0.01 vs dt=0.005 produce equivalent BOLD (atol=1e-4) bilinear.

    Fixed bilinear ground truth, rk4 fixed-step, no noise. Both runs share the
    SAME stim_drive and stim_mod (built at dt=0.005, the finer grid) so only
    the ODE integration step size differs between runs. Expected headroom ~3
    orders of magnitude (14-RESEARCH.md Section 6.4).
    """
    N = 3
    A = parameterize_A(torch.zeros(N, N, dtype=torch.float64))
    # A is diag(-0.5, -0.5, -0.5); guaranteed stable.
    C = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    B = torch.zeros(1, N, N, dtype=torch.float64)
    B[0, 1, 0] = 0.1   # L4 locked: 0.1 not 0.3 to keep monitor silent
    B[0, 2, 1] = 0.1

    duration = 200.0
    TR = 2.0
    stim_drive = make_block_stimulus(
        n_blocks=3, block_duration=30.0, rest_duration=30.0, n_inputs=1,
    )
    # Construct modulator at finer grid dt=0.005 so both runs agree on
    # quantization.
    stim_mod = make_epoch_stimulus(
        event_times=[50.0], event_durations=[40.0], event_amplitudes=[1.0],
        duration=duration, dt=0.005, n_inputs=1,
    )

    common = dict(
        A=A, C=C, stimulus=stim_drive,
        duration=duration, TR=TR,
        SNR=-1,               # no noise
        solver="rk4",         # fixed-step for reproducibility
        dtype=torch.float64,
        seed=42,
        B_list=B, stimulus_mod=stim_mod,
    )
    r_01 = simulate_task_dcm(**common, dt=0.01)
    r_005 = simulate_task_dcm(**common, dt=0.005)

    # Sanity on TR grid.
    assert r_01["times_TR"].shape == r_005["times_TR"].shape
    torch.testing.assert_close(
        r_01["times_TR"], r_005["times_TR"], atol=1e-12, rtol=0.0,
    )
    # Primary SIM-05 gate.
    torch.testing.assert_close(
        r_01["bold_clean"], r_005["bold_clean"], atol=1e-4, rtol=0.0,
    )


def test_dt_invariance_linear() -> None:
    """SIM-05 (L5 regression): dt-invariance for linear mode (B_list=None).

    Mirror of test_dt_invariance_bilinear with no B_list / stimulus_mod.
    """
    N = 3
    A = parameterize_A(torch.zeros(N, N, dtype=torch.float64))
    C = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)

    duration = 200.0
    TR = 2.0
    stim_drive = make_block_stimulus(
        n_blocks=3, block_duration=30.0, rest_duration=30.0, n_inputs=1,
    )

    common = dict(
        A=A, C=C, stimulus=stim_drive,
        duration=duration, TR=TR,
        SNR=-1, solver="rk4", dtype=torch.float64, seed=42,
    )
    r_01 = simulate_task_dcm(**common, dt=0.01)
    r_005 = simulate_task_dcm(**common, dt=0.005)

    torch.testing.assert_close(
        r_01["bold_clean"], r_005["bold_clean"], atol=1e-4, rtol=0.0,
    )
    # Also confirm linear-mode return-dict sanity.
    assert r_01["B_list"] is None and r_005["B_list"] is None
