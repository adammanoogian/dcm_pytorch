"""Unit tests for Phase 14 stimulus construction utilities.

Covers:
- make_event_stimulus (SIM-01): variable-amplitude stick-function stimuli
- make_epoch_stimulus (SIM-02): boxcar-shaped modulatory inputs
- merge_piecewise_inputs: widened PiecewiseConstantInput helper

Overlap semantics (locked per Phase 14 L1 decision): overlapping epochs SUM
amplitudes and emit UserWarning at construction time.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.simulators import (
    make_epoch_stimulus,
    make_event_stimulus,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus
from pyro_dcm.utils import (
    PiecewiseConstantInput,
    merge_piecewise_inputs,
)

# ---------------------------------------------------------------------------
# TestMakeEventStimulus
# ---------------------------------------------------------------------------


class TestMakeEventStimulus:
    """Unit tests for make_event_stimulus (SIM-01)."""

    def test_basic_shape(self) -> None:
        """3 events * 2 transitions + 1 initial zero = 7 breakpoints."""
        result = make_event_stimulus(
            event_times=[1.0, 3.0, 5.0],
            event_amplitudes=1.0,
            duration=10.0,
            dt=0.01,
            n_inputs=1,
        )
        assert result["times"].shape == (7,), result["times"].shape
        assert result["values"].shape == (7, 1), result["values"].shape
        assert result["times"][0].item() == 0.0
        assert result["values"][0, 0].item() == 0.0

    def test_scalar_amplitude_broadcasts(self) -> None:
        """Scalar amplitude broadcasts to all onset rows."""
        result = make_event_stimulus(
            event_times=[1.0, 3.0, 5.0],
            event_amplitudes=1.0,
            duration=10.0,
            dt=0.01,
            n_inputs=1,
        )
        # Onset rows are indices 1, 3, 5 (after initial zero; each event
        # contributes (onset_row, off_row)).
        onset_rows = result["values"][1::2, 0]
        assert torch.allclose(onset_rows, torch.ones_like(onset_rows))

    def test_1d_amplitude_vector(self) -> None:
        """1-D amplitudes map to onset rows in sorted order."""
        result = make_event_stimulus(
            event_times=[1.0, 3.0, 5.0],
            event_amplitudes=[0.5, 1.0, 0.7],
            duration=10.0,
            dt=0.01,
        )
        onset_rows = result["values"][1::2, 0]
        expected = torch.tensor([0.5, 1.0, 0.7], dtype=torch.float64)
        assert torch.allclose(onset_rows, expected)

    def test_2d_amplitude_matrix_multi_input(self) -> None:
        """2-D amplitude matrix preserves column structure across n_inputs."""
        result = make_event_stimulus(
            event_times=[1.0, 3.0],
            event_amplitudes=[[1.0, 0.0], [0.0, 1.0]],
            duration=10.0,
            dt=0.01,
            n_inputs=2,
        )
        # 2 events * 2 transitions + 1 initial zero = 5 breakpoints.
        assert result["values"].shape == (5, 2)
        # Onset rows 1, 3 should match per-event amplitudes.
        assert torch.allclose(
            result["values"][1], torch.tensor([1.0, 0.0], dtype=torch.float64)
        )
        assert torch.allclose(
            result["values"][3], torch.tensor([0.0, 1.0], dtype=torch.float64)
        )

    def test_sorts_unsorted_times(self) -> None:
        """Unsorted event_times are sorted; amplitudes follow the permutation."""
        result = make_event_stimulus(
            event_times=[5.0, 1.0, 3.0],
            event_amplitudes=[0.1, 0.2, 0.3],
            duration=10.0,
            dt=0.01,
        )
        # Times must be sorted ascending.
        diffs = result["times"][1:] - result["times"][:-1]
        assert (diffs >= 0).all()
        # After sorting: t=1 -> amp 0.2, t=3 -> amp 0.3, t=5 -> amp 0.1.
        onset_rows = result["values"][1::2, 0]
        expected = torch.tensor([0.2, 0.3, 0.1], dtype=torch.float64)
        assert torch.allclose(onset_rows, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # Negative event time.
            {
                "event_times": [-0.1],
                "event_amplitudes": [1.0],
                "duration": 10.0,
                "dt": 0.01,
            },
            # Event at t == duration (must be < duration).
            {
                "event_times": [10.0],
                "event_amplitudes": [1.0],
                "duration": 10.0,
                "dt": 0.01,
            },
            # dt = 0.
            {
                "event_times": [1.0],
                "event_amplitudes": [1.0],
                "duration": 10.0,
                "dt": 0.0,
            },
            # duration = 0.
            {
                "event_times": [],
                "event_amplitudes": [],
                "duration": 0.0,
                "dt": 0.01,
            },
        ],
    )
    def test_validation_errors_parameters(self, kwargs: dict) -> None:
        """ValueErrors on out-of-range times and non-positive dt/duration."""
        with pytest.raises(ValueError):
            make_event_stimulus(**kwargs)

    def test_validation_errors_3d_amps(self) -> None:
        """3-D amplitude tensor is rejected with a clear ValueError."""
        with pytest.raises(ValueError, match="ndim"):
            make_event_stimulus(
                event_times=[1.0, 2.0],
                event_amplitudes=torch.zeros(2, 2, 3),
                duration=10.0,
                dt=0.01,
                n_inputs=2,
            )

    def test_same_grid_index_raises(self) -> None:
        """Two events quantizing to the same dt grid index must raise."""
        # With dt=0.01, t=1.001 -> idx=100, t=1.002 -> idx=100.
        with pytest.raises(ValueError, match="same dt grid index"):
            make_event_stimulus(
                event_times=[1.001, 1.002],
                event_amplitudes=[0.5, 0.5],
                duration=10.0,
                dt=0.01,
            )

    def test_stick_pulse_width_equals_dt(self) -> None:
        """Stick pulse width equals dt under PiecewiseConstantInput lookup."""
        result = make_event_stimulus(
            event_times=[2.0],
            event_amplitudes=[1.0],
            duration=10.0,
            dt=0.01,
        )
        input_fn = PiecewiseConstantInput(result["times"], result["values"])
        # t=2.005 is inside [2.00, 2.01): expect amplitude.
        inside = input_fn(torch.tensor(2.005, dtype=torch.float64))
        assert torch.allclose(
            inside, torch.tensor([1.0], dtype=torch.float64)
        )
        # t=2.011 is past 2.01: expect 0.
        outside = input_fn(torch.tensor(2.011, dtype=torch.float64))
        assert torch.allclose(
            outside, torch.tensor([0.0], dtype=torch.float64)
        )

    def test_piecewise_compatibility(self) -> None:
        """Output dict plugs directly into PiecewiseConstantInput."""
        result = make_event_stimulus(
            event_times=[1.0, 3.0],
            event_amplitudes=[0.5, 0.7],
            duration=10.0,
            dt=0.01,
        )
        input_fn = PiecewiseConstantInput(result["times"], result["values"])
        value = input_fn(torch.tensor(0.0, dtype=torch.float64))
        assert value.shape == (1,), value.shape


# ---------------------------------------------------------------------------
# TestMakeEpochStimulus
# ---------------------------------------------------------------------------


class TestMakeEpochStimulus:
    """Unit tests for make_epoch_stimulus (SIM-02)."""

    def test_single_epoch_boxcar(self) -> None:
        """Single 10s epoch produces breakpoints at 0, t_on, t_off."""
        result = make_epoch_stimulus(
            event_times=[5.0],
            event_durations=[10.0],
            event_amplitudes=[1.0],
            duration=30.0,
            dt=0.01,
            n_inputs=1,
        )
        times = result["times"]
        values = result["values"]
        # Must contain 0.0, 5.0, and 15.0 as exact breakpoints.
        assert torch.allclose(times[0], torch.tensor(0.0, dtype=torch.float64))
        # Find 5.0 and 15.0 indices.
        idx_on = int(
            torch.nonzero(
                torch.isclose(
                    times, torch.tensor(5.0, dtype=torch.float64)
                ),
                as_tuple=False,
            )[0, 0].item()
        )
        idx_off = int(
            torch.nonzero(
                torch.isclose(
                    times, torch.tensor(15.0, dtype=torch.float64)
                ),
                as_tuple=False,
            )[0, 0].item()
        )
        assert values[0, 0].item() == 0.0
        assert values[idx_on, 0].item() == pytest.approx(1.0)
        assert values[idx_off, 0].item() == pytest.approx(0.0)

    def test_scalar_duration_broadcasts(self) -> None:
        """Scalar event_durations broadcasts to all events."""
        result = make_epoch_stimulus(
            event_times=[2.0, 10.0, 18.0],
            event_durations=5.0,
            event_amplitudes=[1.0, 1.0, 1.0],
            duration=30.0,
            dt=0.01,
            n_inputs=1,
        )
        input_fn = PiecewiseConstantInput(result["times"], result["values"])
        # Query midpoints of each 5-second epoch.
        for t_on in (2.0, 10.0, 18.0):
            mid = torch.tensor(t_on + 2.5, dtype=torch.float64)
            val = input_fn(mid)
            assert val.item() == pytest.approx(1.0), (t_on, val)

    def test_overlap_sum_and_warning(self) -> None:
        """L1: overlapping epochs SUM amplitudes + emit UserWarning."""
        # Epoch A: [2, 8], amp 1. Epoch B: [5, 11], amp 1. Overlap [5, 8].
        with pytest.warns(UserWarning, match="Overlapping epochs"):
            result = make_epoch_stimulus(
                event_times=[2.0, 5.0],
                event_durations=[6.0, 6.0],
                event_amplitudes=[1.0, 1.0],
                duration=20.0,
                dt=0.01,
                n_inputs=1,
            )
        input_fn = PiecewiseConstantInput(result["times"], result["values"])
        # In overlap window (t=6): expect sum = 2.
        val = input_fn(torch.tensor(6.0, dtype=torch.float64))
        assert val.item() == pytest.approx(2.0)

    def test_clipping_at_duration_warns(self) -> None:
        """Epoch ending past duration emits UserWarning with 'clipped'."""
        with pytest.warns(UserWarning, match="clipped to duration"):
            make_epoch_stimulus(
                event_times=[15.0],
                event_durations=[10.0],
                event_amplitudes=[1.0],
                duration=20.0,
                dt=0.01,
                n_inputs=1,
            )

    def test_piecewise_compatibility(self) -> None:
        """Output dict plugs directly into PiecewiseConstantInput."""
        result = make_epoch_stimulus(
            event_times=[3.0, 12.0],
            event_durations=[4.0, 4.0],
            event_amplitudes=[0.5, 0.8],
            duration=20.0,
            dt=0.01,
            n_inputs=1,
        )
        input_fn = PiecewiseConstantInput(result["times"], result["values"])
        val = input_fn(torch.tensor(0.0, dtype=torch.float64))
        assert val.shape == (1,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # Negative duration.
            {
                "event_times": [1.0],
                "event_durations": [-1.0],
                "event_amplitudes": [1.0],
                "duration": 10.0,
                "dt": 0.01,
            },
            # Duration shape mismatch.
            {
                "event_times": [1.0, 2.0],
                "event_durations": [1.0, 1.0, 1.0],
                "event_amplitudes": [1.0, 1.0],
                "duration": 10.0,
                "dt": 0.01,
            },
            # Negative event_time.
            {
                "event_times": [-0.5],
                "event_durations": [1.0],
                "event_amplitudes": [1.0],
                "duration": 10.0,
                "dt": 0.01,
            },
        ],
    )
    def test_validation_errors(self, kwargs: dict) -> None:
        """ValueErrors on bad durations and times."""
        with pytest.raises(ValueError):
            make_epoch_stimulus(**kwargs)


# ---------------------------------------------------------------------------
# TestMergePiecewiseInputs
# ---------------------------------------------------------------------------


class TestMergePiecewiseInputs:
    """Unit tests for merge_piecewise_inputs."""

    def _build_drive_mod(
        self,
    ) -> tuple[PiecewiseConstantInput, PiecewiseConstantInput]:
        drive_d = make_block_stimulus(
            n_blocks=2, block_duration=5.0, rest_duration=5.0, n_inputs=1
        )
        drive = PiecewiseConstantInput(drive_d["times"], drive_d["values"])
        mod_d = make_event_stimulus(
            event_times=[3.0],
            event_amplitudes=[1.0],
            duration=30.0,
            dt=0.01,
            n_inputs=1,
        )
        mod = PiecewiseConstantInput(mod_d["times"], mod_d["values"])
        return drive, mod

    def test_concatenation_shape(self) -> None:
        """Merged values have M+J columns and unique-union breakpoint rows."""
        drive, mod = self._build_drive_mod()
        merged = merge_piecewise_inputs(drive, mod)
        assert merged.values.shape[1] == 2
        # merged_times is torch.unique of concatenation, so its length equals
        # the unique union length.
        all_times = torch.cat([drive.times, mod.times])
        expected_k = torch.unique(all_times, sorted=True).shape[0]
        assert merged.times.shape[0] == expected_k

    def test_values_at_breakpoints_concat_correctly(self) -> None:
        """At any t*, merged(t*) = concat(drive(t*), mod(t*))."""
        drive, mod = self._build_drive_mod()
        merged = merge_piecewise_inputs(drive, mod)
        # Cover several query points. Include: before any event, inside drive
        # block, in drive rest, inside mod event window, in mod rest after
        # event, and at an exact breakpoint time.
        query_times = [0.0, 0.5, 2.5, 3.005, 3.02, 5.0]
        for t_scalar in query_times:
            t = torch.tensor(t_scalar, dtype=torch.float64)
            expected = torch.cat([drive(t), mod(t)])
            got = merged(t)
            assert torch.allclose(got, expected), (t_scalar, got, expected)

    def test_same_breakpoint_in_both(self) -> None:
        """Shared breakpoint is deduped to a single row with correct values."""
        t_drive = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
        v_drive = torch.tensor(
            [[0.0], [0.3], [0.0]], dtype=torch.float64
        )
        drive = PiecewiseConstantInput(t_drive, v_drive)
        t_mod = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
        v_mod = torch.tensor(
            [[0.0], [0.7], [0.0]], dtype=torch.float64
        )
        mod = PiecewiseConstantInput(t_mod, v_mod)

        merged = merge_piecewise_inputs(drive, mod)
        # t=2.0 must appear exactly once (torch.unique dedup).
        mask = torch.isclose(
            merged.times, torch.tensor(2.0, dtype=torch.float64)
        )
        assert int(mask.sum().item()) == 1
        idx = int(torch.nonzero(mask, as_tuple=False)[0, 0].item())
        expected = torch.cat(
            [drive(torch.tensor(2.0, dtype=torch.float64)),
             mod(torch.tensor(2.0, dtype=torch.float64))]
        )
        assert torch.allclose(merged.values[idx], expected)

    def test_dtype_device_mismatch_raises(self) -> None:
        """Dtype mismatch between drive and mod raises ValueError (no cast)."""
        t_drive = torch.tensor([0.0, 5.0], dtype=torch.float64)
        v_drive = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        drive = PiecewiseConstantInput(t_drive, v_drive)
        t_mod = torch.tensor([0.0, 5.0], dtype=torch.float32)
        v_mod = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        mod = PiecewiseConstantInput(t_mod, v_mod)
        with pytest.raises(ValueError, match="dtype"):
            merge_piecewise_inputs(drive, mod)
