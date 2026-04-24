"""Phase 17 acceptance tests for CircuitViz serializer.

A-series (A-01..A-10) are structural / schema-level acceptance gates. They
run in milliseconds because the serializer performs no SVI / ODE / Pyro work.

Regression tests (5 additional, flat) cover V2 input validation (region_colors
missing + length mismatch), V6 NaN/Inf guard, and V1 extras round-trip.

B-series (B-01, B-02) are @pytest.mark.slow Pyro integration smokes. They
verify that the ``flatten_posterior_for_viz`` helper composes correctly with
``task_dcm_model`` bilinear SVI + ``extract_posterior_params``, producing a
``fitted`` config end-to-end.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from pyro_dcm.utils.circuit_viz import (
    _FIRST_CLASS_KEYS,
    CircuitViz,
    CircuitVizConfig,
    flatten_posterior_for_viz,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_3region_bilinear_cfg() -> dict:
    """3-region bilinear model config for A-series structural tests."""
    return {
        "regions": ["R0", "R1", "R2"],
        "region_colors": ["#111111", "#222222", "#333333"],
        "A_prior_mean": [
            [-0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, -0.5],
        ],
        "B_matrices": {
            "B1": [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        "C_matrix": [[0.5], [0.0], [0.0]],
        "C_inputs": ["u_drive"],
        "B_modulators": {
            "B1": {
                "label": "B1 — mod",
                "color": "#888888",
                "modulator": "u_mod",
                "modulator_display": "u(t)",
            },
        },
        "meta": {
            "title": "Test DCM",
            "subtitle": "3-region bilinear",
            "tags": ["3-region", "SVI"],
        },
        "peb_covariates": [
            {"name": "beta_x", "display": "β_x", "desc": "test cov"},
        ],
    }


@pytest.fixture()
def dummy_posterior_3region() -> dict[str, list[list[float]]]:
    """Return a dummy posterior-means payload for A-02 / A-03."""
    return {
        "A": [
            [-0.48, 0.01, -0.02],
            [0.02, -0.49, 0.00],
            [-0.01, 0.02, -0.50],
        ],
        "B1": [[0.0, 0.0, 0.0], [0.38, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "C": [[0.51], [0.01], [-0.01]],
    }


# ---------------------------------------------------------------------------
# A-series structural acceptance tests (fast; <10s total)
# ---------------------------------------------------------------------------


def test_roundtrip_heart2adapt() -> None:
    """A-01: HEART2ADAPT reference config round-trips byte-equal via extras."""
    ref_path = Path("configs/heart2adapt_dcm_config.json")
    if not ref_path.exists():
        pytest.skip(f"Reference config {ref_path} not present on this branch")
    cfg = CircuitViz.load(ref_path)
    roundtrip = json.loads(json.dumps(cfg.to_dict()))
    original = json.loads(ref_path.read_text())
    assert roundtrip == original, (
        f"Round-trip mismatch: missing={set(original) - set(roundtrip)}, "
        f"extra={set(roundtrip) - set(original)}"
    )
    # Sanity: extras pass-through preserved at least _study + _description.
    assert "_study" in cfg.extras
    assert "_description" in cfg.extras


def test_from_posterior_flips_status(
    minimal_3region_bilinear_cfg: dict,
    dummy_posterior_3region: dict[str, list[list[float]]],
) -> None:
    """A-02: from_posterior flips status='planned'->'fitted' and attaches."""
    planned = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    assert planned.status == "planned"
    assert planned.fitted_params is None

    fitted = CircuitViz.from_posterior(planned, dummy_posterior_3region)
    assert fitted.status == "fitted"
    assert fitted.fitted_params is not None
    assert sorted(fitted.fitted_params.keys()) == ["A", "B1", "C"]


def test_from_posterior_no_mutation(
    minimal_3region_bilinear_cfg: dict,
    dummy_posterior_3region: dict[str, list[list[float]]],
) -> None:
    """A-03: from_posterior does not mutate the planned argument (deepcopy)."""
    planned = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    planned_before = copy.deepcopy(planned)
    _ = CircuitViz.from_posterior(planned, dummy_posterior_3region)
    assert planned.status == planned_before.status
    assert planned.fitted_params == planned_before.fitted_params
    assert planned.matrices == planned_before.matrices


def test_to_dict_top_level_keys(minimal_3region_bilinear_cfg: dict) -> None:
    """A-04: to_dict from a from_model_config output emits exactly 13 keys.

    Note: extras is empty when the config was built via from_model_config, so
    to_dict()'s key set is exactly _FIRST_CLASS_KEYS. Configs loaded from
    external JSON (e.g., HEART2ADAPT) MAY contribute additional keys via the
    extras pass-through -- see ``test_extras_roundtrip_preserved``.
    """
    cfg = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    assert set(cfg.to_dict().keys()) == set(_FIRST_CLASS_KEYS)


def test_mat_order_deterministic() -> None:
    """A-05: mat_order is sort-deterministic across shuffled B_matrix keys."""
    model_cfg = {
        "regions": ["R0", "R1", "R2"],
        "region_colors": ["#111", "#222", "#333"],
        "A_prior_mean": [[-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5]],
        # Insertion order: B3, B1, B2 -- mat_order must still be sorted.
        "B_matrices": {
            "B3": [[0.0] * 3 for _ in range(3)],
            "B1": [[0.0] * 3 for _ in range(3)],
            "B2": [[0.0] * 3 for _ in range(3)],
        },
        "C_matrix": [[0.5], [0.0], [0.0]],
    }
    cfg = CircuitViz.from_model_config(model_cfg)
    assert cfg.mat_order == ["A", "B1", "B2", "B3", "C"]


def test_schema_version(minimal_3region_bilinear_cfg: dict) -> None:
    """A-06: _schema key in emitted JSON is exactly 'dcm_circuit_explorer/v1'."""
    cfg = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    assert cfg.to_dict()["_schema"] == "dcm_circuit_explorer/v1"


def test_vals_are_list_of_list(minimal_3region_bilinear_cfg: dict) -> None:
    """A-07: every matrices[key].vals is list[list[number]] (never tensor)."""
    cfg = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    for key, entry in cfg.matrices.items():
        vals = entry["vals"]
        assert isinstance(vals, list), f"matrices[{key}].vals is not a list"
        for i, row in enumerate(vals):
            assert isinstance(row, list), (
                f"matrices[{key}].vals[{i}] is not a list (got {type(row)})"
            )
            for j, cell in enumerate(row):
                assert isinstance(cell, (int, float)), (
                    f"matrices[{key}].vals[{i}][{j}] is not int/float "
                    f"(got {type(cell)})"
                )


def test_tensor_input_accepted() -> None:
    """A-08: torch.Tensor A_prior_mean is converted lossless to list[list[float]]."""
    model_cfg = {
        "regions": ["R0", "R1"],
        "region_colors": ["#111111", "#222222"],
        "A_prior_mean": torch.tensor(
            [[-0.5, 0.3], [0.1, -0.5]], dtype=torch.float64,
        ),
    }
    cfg = CircuitViz.from_model_config(model_cfg)
    assert cfg.matrices["A"]["vals"] == [[-0.5, 0.3], [0.1, -0.5]]


def test_export_roundtrip(
    tmp_path: Path,
    minimal_3region_bilinear_cfg: dict,
) -> None:
    """A-09: CircuitViz.load(cfg.export(path)).to_dict() == cfg.to_dict()."""
    cfg = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    out = cfg.export(tmp_path / "test.json")
    roundtripped = CircuitViz.load(out)
    assert roundtripped.to_dict() == cfg.to_dict()


def test_empty_optional_collections() -> None:
    """A-10: phenotypes/hypotheses/drugs/peb are empty (not missing) by default."""
    model_cfg = {
        "regions": ["R0", "R1"],
        "region_colors": ["#111", "#222"],
        "A_prior_mean": [[-0.5, 0.0], [0.0, -0.5]],
    }
    cfg = CircuitViz.from_model_config(model_cfg)
    d = cfg.to_dict()
    assert d["phenotypes"] == []
    assert d["hypotheses"] == []
    assert d["drugs"] == []
    assert d["peb"] == {}


# ---------------------------------------------------------------------------
# Additional regression tests (flat; not in a class)
# ---------------------------------------------------------------------------


def test_region_colors_missing_raises() -> None:
    """V2: missing region_colors -> ValueError with expected-vs-actual."""
    model_cfg = {
        "regions": ["R0", "R1"],
        "A_prior_mean": [[-0.5, 0.0], [0.0, -0.5]],
    }
    with pytest.raises(ValueError, match="region_colors"):
        CircuitViz.from_model_config(model_cfg)


def test_region_colors_length_mismatch_raises() -> None:
    """V2: len(region_colors) != len(regions) -> ValueError mentions 'length'."""
    model_cfg = {
        "regions": ["R0", "R1", "R2"],
        "region_colors": ["#111", "#222"],  # only 2, expect 3
        "A_prior_mean": [[-0.5] * 3] * 3,
    }
    with pytest.raises(ValueError, match="length"):
        CircuitViz.from_model_config(model_cfg)


def test_nan_in_posterior_raises(
    minimal_3region_bilinear_cfg: dict,
) -> None:
    """V6: NaN in posterior_means -> ValueError with 'NaN' and location."""
    planned = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    bad = {"A": [[math.nan, 0.0], [0.0, 0.0]]}
    with pytest.raises(ValueError, match="NaN"):
        CircuitViz.from_posterior(planned, bad)


def test_inf_in_posterior_raises(
    minimal_3region_bilinear_cfg: dict,
) -> None:
    """V6: Inf in posterior_means -> ValueError with 'Inf' and location."""
    planned = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
    bad = {"A": [[math.inf, 0.0], [0.0, 0.0]]}
    with pytest.raises(ValueError, match="Inf"):
        CircuitViz.from_posterior(planned, bad)


def test_extras_roundtrip_preserved() -> None:
    """V1: extras dict survives to_dict even when first-class fields empty."""
    cfg = CircuitVizConfig(
        extras={
            "_study": "TEST",
            "node_info": {"n": 3},
            # Collision with first-class key: first-class wins.
            "regions": ["SHOULD_LOSE"],
        },
    )
    d = cfg.to_dict()
    assert d["_study"] == "TEST"
    assert d["node_info"] == {"n": 3}
    # First-class regions (empty default) must win over extras[regions].
    assert d["regions"] == []


# ---------------------------------------------------------------------------
# Pyro integration smoke tests (slow; -m slow)
# ---------------------------------------------------------------------------


class TestPyroIntegration:
    """B-01 / B-02 end-to-end smokes wiring SVI -> flatten -> from_posterior.

    Marked slow because each test spins up ``task_dcm_model`` + Pyro guide +
    ``extract_posterior_params`` with a 3-region bilinear synthetic dataset.
    Kept deliberately short (5-50 SVI steps) -- B-class is NOT a recovery
    test, only a shape-contract smoke.
    """

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        """Silence pyro_dcm.stability WARNING spam during bilinear SVI."""
        import logging

        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    @pytest.mark.slow
    def test_smoke_planned_to_fitted(self) -> None:
        """B-01: end-to-end SVI smoke. planned -> flatten -> fitted.

        Uses a bare SVI loop + ``functools.partial`` to bind the bilinear
        ``b_masks`` / ``stim_mod`` kwargs before passing to
        ``extract_posterior_params`` (Predictive does not accept
        ``model_kwargs``). Pattern mirrors
        ``tests/test_posterior_extraction.py::test_extract_posterior_includes_B_free_and_B``.
        """
        from functools import partial

        import pyro
        from pyro.infer import SVI, Trace_ELBO

        from pyro_dcm.models.guides import (
            create_guide,
            extract_posterior_params,
        )
        from pyro_dcm.models.task_dcm_model import task_dcm_model
        from pyro_dcm.simulators.task_simulator import (
            make_block_stimulus,
            make_epoch_stimulus,
            make_random_stable_A,
            simulate_task_dcm,
        )
        from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        pyro.clear_param_store()

        N, M = 3, 1
        A = make_random_stable_A(N, density=0.5, seed=42)
        C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
        stim = make_block_stimulus(
            n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
        )
        res = simulate_task_dcm(
            A, C, stim, duration=30.0, dt=0.01, TR=2.0, SNR=5.0, seed=7,
        )
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        t_eval = torch.arange(0, 30.0, 0.5, dtype=torch.float64)

        b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
        b_mask_0[1, 0] = 1.0
        b_masks = [b_mask_0]
        stim_mod_dict = make_epoch_stimulus(
            event_times=[10.0], event_durations=[10.0], event_amplitudes=[1.0],
            duration=30.0, dt=0.01, n_inputs=1,
        )
        stim_mod = PiecewiseConstantInput(
            stim_mod_dict["times"], stim_mod_dict["values"],
        )

        guide = create_guide(
            task_dcm_model, guide_type="auto_normal", init_scale=0.005,
        )
        model_args = (
            res["bold"], res["stimulus"], a_mask, c_mask, t_eval, 2.0, 0.5,
        )

        # Planned config: 3-region, one B modulator ("B0" to match B_free_0),
        # one C input.
        model_cfg = {
            "regions": ["R0", "R1", "R2"],
            "region_colors": ["#111", "#222", "#333"],
            "A_prior_mean": [
                [-0.5, 0.0, 0.0],
                [0.0, -0.5, 0.0],
                [0.0, 0.0, -0.5],
            ],
            "B_matrices": {
                "B0": [[0.0] * 3 for _ in range(3)],  # prior mean zero
            },
            "B_modulators": {
                "B0": {"label": "B0 — smoke", "color": "#0088FF"},
            },
            "C_matrix": [[0.25], [0.0], [0.0]],
            "C_inputs": ["u_block"],
        }
        planned = CircuitViz.from_model_config(model_cfg)

        # Bare SVI loop (20 steps); run_svi cannot forward bilinear kwargs
        # through Predictive downstream, and the test is a smoke -- not a
        # recovery gate. 20 steps matches the existing bilinear posterior-
        # extraction integration test in test_posterior_extraction.py.
        optimizer = pyro.optim.ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
        svi = SVI(task_dcm_model, guide, optimizer, loss=Trace_ELBO())
        svi_model_kwargs = dict(
            observed_bold=res["bold"],
            stimulus=res["stimulus"],
            a_mask=a_mask,
            c_mask=c_mask,
            t_eval=t_eval,
            TR=2.0,
            dt=0.5,
            b_masks=b_masks,
            stim_mod=stim_mod,
        )
        for _ in range(20):
            svi.step(**svi_model_kwargs)

        # Extract posterior via a partial-bound bilinear_model (Predictive
        # invokes model(*args) with no kwarg forwarding).
        bilinear_model = partial(
            task_dcm_model, b_masks=b_masks, stim_mod=stim_mod,
        )
        posterior = extract_posterior_params(
            guide, model_args, model=bilinear_model, num_samples=20,
        )
        flat = flatten_posterior_for_viz(
            posterior, planned.mat_order, b_masks=b_masks,
        )

        fitted = CircuitViz.from_posterior(planned, flat)
        d = fitted.to_dict()
        assert d["_status"] == "fitted"
        assert d["fitted_params"] is not None
        # Keys in fitted_params match planned.mat_order.
        assert sorted(d["fitted_params"].keys()) == sorted(planned.mat_order)
        # Shape contract: A is 3x3, B0 is 3x3, C is 3x1.
        assert len(d["fitted_params"]["A"]) == N
        assert len(d["fitted_params"]["A"][0]) == N
        assert len(d["fitted_params"]["B0"]) == N
        assert len(d["fitted_params"]["B0"][0]) == N
        assert len(d["fitted_params"]["C"]) == N
        assert len(d["fitted_params"]["C"][0]) == M

    @pytest.mark.slow
    def test_extract_posterior_shapes_match_serializer(self) -> None:
        """B-02: shape contract -- posterior['A']['mean'] is (3, 3) -> 3x3 list."""
        import pyro

        from pyro_dcm.models.guides import (
            create_guide,
            extract_posterior_params,
            run_svi,
        )
        from pyro_dcm.models.task_dcm_model import task_dcm_model
        from pyro_dcm.simulators.task_simulator import (
            make_block_stimulus,
            make_random_stable_A,
            simulate_task_dcm,
        )

        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        pyro.clear_param_store()

        N, M = 3, 1
        A = make_random_stable_A(N, density=0.5, seed=42)
        C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)
        stim = make_block_stimulus(
            n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
        )
        res = simulate_task_dcm(
            A, C, stim, duration=30.0, dt=0.01, TR=2.0, SNR=5.0, seed=7,
        )
        a_mask = torch.ones(N, N, dtype=torch.float64)
        c_mask = torch.ones(N, M, dtype=torch.float64)
        t_eval = torch.arange(0, 30.0, 0.5, dtype=torch.float64)

        guide = create_guide(
            task_dcm_model, guide_type="auto_normal", init_scale=0.01,
        )
        model_args = (
            res["bold"], res["stimulus"], a_mask, c_mask, t_eval, 2.0, 0.5,
        )
        # Just 5 SVI steps -- we only need parameters in the store.
        _ = run_svi(task_dcm_model, guide, model_args, num_steps=5, lr=0.01)

        posterior = extract_posterior_params(
            guide, model_args, model=task_dcm_model, num_samples=20,
        )
        # In linear mode (no b_masks), posterior has A_free (not A) + C.
        # Assert the shape contract flatten_posterior_for_viz relies on.
        assert "A_free" in posterior or "A" in posterior
        if "A" in posterior:
            a_mean = posterior["A"]["mean"]
        else:
            from pyro_dcm.forward_models.neural_state import parameterize_A

            a_mean = parameterize_A(posterior["A_free"]["mean"])
        assert a_mean.dim() == 2
        assert tuple(a_mean.shape) == (N, N)

        a_list = a_mean.detach().cpu().tolist()
        assert isinstance(a_list, list)
        assert len(a_list) == N
        assert all(isinstance(row, list) and len(row) == N for row in a_list)
        assert all(
            isinstance(cell, (int, float)) for row in a_list for cell in row
        )
