"""Unit tests for the Phase 30 hardened recovery-metric assembler (30-01).

Covers the two metric-hardening guards (masked sign recovery against the
``sign(0)`` artifact; per-region R-squared against the variance-pooled artifact),
the identifiability shrinkage helper, the near-stability-boundary ``A`` exclusion
band, and the per-model SNR-injection mapping. All tests are laptop-fast and
marked ``@pytest.mark.vl`` (registered in ``pyproject.toml`` since Phase 29).

References
----------
.planning/phases/30-recovery-matrix-sweep/30-01-PLAN.md
    Task 3 test specification (VLREC-02 / VLREC-03 hardening proof).
.planning/research/v0.7.0/PITFALLS.md
    R1 (pooled R-squared), R2 (sign(0)), N2 (eig_clamp non-injectivity).
"""

from __future__ import annotations

import json

import pytest
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.recovery_matrix_metrics import (
    NEAR_BOUNDARY_HI,
    NEAR_BOUNDARY_LO,
    assemble_cell_metrics,
    compute_shrinkage_ratio,
    exclude_near_boundary_A,
    resample_A_until_accepted,
    snr_for_model,
)
from benchmarks.runners.spectral_vl import run_spectral_vl
from pyro_dcm.simulators.task_simulator import make_random_stable_A

pytestmark = pytest.mark.vl


def _flat(mat: torch.Tensor) -> list[float]:
    """Row-major flatten a square matrix to a Python float list."""
    return mat.to(torch.float64).flatten().tolist()


def test_masked_sign_recovery_ignores_structural_zeros() -> None:
    """Masked sign recovery scores 1.0 when zeros are excluded (guards R2).

    ``A_true`` has several exact-zero off-diagonals and matching-sign non-zero
    entries. An UNMASKED ``sign(0)`` comparison would deflate the score because
    ``torch.sign(0) == 0`` never matches a non-zero prediction. The masked metric
    must ignore the zeros and report a perfect 1.0.
    """
    a_true = torch.tensor(
        [
            [-0.5, 0.3, 0.0, 0.0],
            [0.0, -0.5, 0.4, 0.0],
            [0.0, 0.0, -0.5, 0.2],
            [-0.3, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )
    # Inferred preserves the sign of every non-zero entry but has noise on the
    # structural zeros (which must be excluded by the mask).
    a_inferred = a_true.clone()
    zero_mask = a_true.abs() <= 0.1
    a_inferred[zero_mask] = 0.05  # wrong sign vs a true zero, but masked out

    cell_result = {
        "variant": "spectral",
        "method": "vl",
        "n_regions": 4,
        "rmse_list": [0.01],
        "n_success": 1,
        "n_failed": 0,
        "a_true_list": [_flat(a_true)],
        "a_inferred_list": [_flat(a_inferred)],
    }
    out = assemble_cell_metrics(cell_result, sign_threshold=0.1)
    assert out["sign_recovery_masked"] == 1.0


def test_per_region_r2_not_pooled() -> None:
    """Reported R-squared is the per-region list median, NOT variance-pooled.

    The assembler consumes a driver-supplied ``r2_per_region_list`` (already the
    ``pooled=False`` per-region reduction) and median-aggregates it. With one
    near-zero-R2 region and others ~1.0, a variance-pooled implementation would
    report a number dominated by the high-variance region; the per-region median
    reported here is distinctly lower.
    """
    # Per-seed per-region-averaged R2 values (already pooled=False upstream).
    # If this were variance-pooled, the silent region would not drag it down.
    r2_per_region_list = [0.5, 0.52, 0.48]  # ~0.5: a failing region pulled it
    cell_result = {
        "variant": "latent_circuit",
        "method": "vl",
        "n_regions": 4,
        "a_rmse_list": [0.1, 0.1, 0.1],
        "n_success": 3,
        "n_failed": 0,
        "r2_per_region_list": r2_per_region_list,
    }
    out = assemble_cell_metrics(cell_result)
    # Median of the per-region list, well below a pooled ~1.0 that a
    # variance-weighted reduction would have reported.
    assert out["r2_per_region"] == pytest.approx(0.5, abs=1e-9)
    assert out["r2_per_region"] < 0.9


def test_shrinkage_ratio() -> None:
    """Shrinkage ratio divides element-wise; non-positive prior raises."""
    ratio = compute_shrinkage_ratio(torch.tensor([0.5]), 1.0)
    assert ratio.tolist() == [0.5]

    with pytest.raises(ValueError, match="strictly positive"):
        compute_shrinkage_ratio(torch.tensor([0.5]), 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_shrinkage_ratio(torch.tensor([0.5]), -1.0)


def test_exclude_near_boundary_band() -> None:
    """A clearly-stable A is accepted; one in [-0.05, 0] is rejected."""
    stable = torch.eye(3, dtype=torch.float64) * -0.5
    assert exclude_near_boundary_A(stable) is True

    # Max Re eig exactly inside the band -> rejected.
    near = torch.eye(3, dtype=torch.float64) * -0.02
    assert exclude_near_boundary_A(near) is False
    # Band edges are inclusive on the rejected side.
    assert NEAR_BOUNDARY_LO < -0.02 < NEAR_BOUNDARY_HI

    # Seeded closure: advance the seed each call so rejects are not regenerated.
    counter = {"i": 0}

    def make_A() -> torch.Tensor:
        seed = 100 + counter["i"]
        counter["i"] += 1
        return make_random_stable_A(4, density=0.5, seed=seed)

    accepted = resample_A_until_accepted(make_A, max_tries=50)
    assert exclude_near_boundary_A(accepted) is True


def test_resample_raises_when_none_accepted() -> None:
    """resample_A_until_accepted raises RuntimeError if every draw is in-band."""
    in_band = torch.eye(2, dtype=torch.float64) * -0.02

    with pytest.raises(RuntimeError, match="max_tries"):
        resample_A_until_accepted(lambda: in_band, max_tries=3)


def test_snr_for_model_per_variant() -> None:
    """Task/latent return SNR kwarg; spectral returns noise log-amplitude."""
    assert snr_for_model("task", 3) == {"SNR": 3.0}
    assert snr_for_model("latent_circuit", 5) == {"SNR": 5.0}

    low = snr_for_model("spectral", 2)
    high = snr_for_model("spectral", 10)
    assert "noise_log_amplitude" in low
    # Higher SNR -> more-negative log-amplitude (less observation noise).
    assert high["noise_log_amplitude"] < low["noise_log_amplitude"]

    with pytest.raises(ValueError, match="Unknown variant"):
        snr_for_model("nonsense", 3)
    with pytest.raises(ValueError, match="snr_level > 0"):
        snr_for_model("spectral", 0)


def test_assemble_cell_metrics_real_runner() -> None:
    """A real spectral VL runner result assembles to a JSON-serializable block."""
    config = BenchmarkConfig(
        variant="spectral",
        method="vl",
        n_regions=2,
        n_datasets=1,
        seed=0,
        quick=True,
        max_iter=8,
    )
    result = run_spectral_vl(config)
    cell = assemble_cell_metrics(result)

    for key in (
        "rmse_a",
        "coverage_95",
        "sign_recovery_masked",
        "shrinkage",
        "n_success",
    ):
        assert key in cell

    # No tensors leak across the JSON boundary.
    assert not any(torch.is_tensor(v) for v in cell.values())
    json.dumps(cell)  # must not raise
