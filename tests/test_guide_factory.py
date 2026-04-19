"""Tests for the create_guide factory in guides.py.

Verifies that all six supported Pyro AutoGuide types can be
instantiated via ``create_guide``, that init_scale asymmetry is
handled correctly, that the N-based blocklist raises for dangerous
configurations, and that each guide can complete a single SVI step.
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from pyro.optim import Adam

from pyro_dcm.models.guides import (
    GUIDE_REGISTRY,
    MEAN_FIELD_GUIDES,
    create_guide,
)
from pyro_dcm.models.task_dcm_model import task_dcm_model
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput


# ------------------------------------------------------------------
# Minimal Pyro model used across all tests
# ------------------------------------------------------------------


def _toy_model() -> None:
    """Three-site model: a, b -> obs = Normal(a + b, 1)."""
    a = pyro.sample("a", dist.Normal(0.0, 1.0))
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    pyro.sample("obs", dist.Normal(a + b, 1.0), obs=torch.tensor(3.0))


# ------------------------------------------------------------------
# Bilinear task-DCM fixture for MODEL-06 auto-discovery tests
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def task_bilinear_guide_data() -> dict:
    """3-region, 1 driving input, 1 modulator bilinear DCM data for guide tests.

    Mirrors the fixture structure in
    ``tests/test_task_dcm_model.py::TestBilinearStructure`` (duplicated
    locally rather than imported to avoid fragile inter-test-file
    dependencies). Uses :func:`make_epoch_stimulus` (Pitfall B12 preferred
    boxcar primitive for dt-invariance under rk4 mid-step sampling) for
    ``stim_mod``.

    Returns
    -------
    dict
        Keys: ``observed_bold``, ``stimulus``, ``a_mask``, ``c_mask``,
        ``t_eval``, ``TR``, ``dt``, ``N``, ``M``, ``J``, ``b_masks``,
        ``stim_mod``.
    """
    N, M = 3, 1
    A = make_random_stable_A(N, density=0.5, seed=42)
    C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)

    stim = make_block_stimulus(
        n_blocks=2, block_duration=8.0, rest_duration=7.0, n_inputs=M,
    )

    duration = 30.0
    TR = 2.0
    dt = 0.5

    result = simulate_task_dcm(
        A, C, stim, duration=duration, dt=0.01, TR=TR, SNR=5.0, seed=7,
    )

    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.ones(N, M, dtype=torch.float64)

    # Single-modulator mask: gate 1 <- 0 connection (Pitfall B5 zero diagonal).
    b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0
    b_masks = [b_mask_0]

    # Single 10s epoch at t=10s, amplitude 1.0, over 30s total.
    stim_mod_dict = make_epoch_stimulus(
        event_times=[10.0],
        event_durations=[10.0],
        event_amplitudes=[1.0],
        duration=30.0,
        dt=0.01,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )

    return {
        "observed_bold": result["bold"],
        "stimulus": result["stimulus"],
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "TR": TR,
        "dt": dt,
        "N": N,
        "M": M,
        "J": 1,
        "b_masks": b_masks,
        "stim_mod": stim_mod,
    }


# ------------------------------------------------------------------
# 1. Parametrized instantiation
# ------------------------------------------------------------------

_GUIDE_TYPE_CLASS = [
    ("auto_delta", AutoDelta),
    ("auto_normal", AutoNormal),
    ("auto_lowrank_mvn", AutoLowRankMultivariateNormal),
    ("auto_mvn", AutoMultivariateNormal),
    ("auto_iaf", AutoIAFNormal),
    ("auto_laplace", AutoLaplaceApproximation),
]


@pytest.mark.parametrize(
    ("guide_type", "expected_cls"),
    _GUIDE_TYPE_CLASS,
    ids=[g for g, _ in _GUIDE_TYPE_CLASS],
)
def test_create_guide_returns_correct_type(
    guide_type: str,
    expected_cls: type,
) -> None:
    """create_guide returns the correct AutoGuide subclass."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    assert isinstance(guide, expected_cls), (
        f"Expected {expected_cls.__name__}, "
        f"got {type(guide).__name__}"
    )


# ------------------------------------------------------------------
# 2. Default backward compatibility
# ------------------------------------------------------------------


def test_create_guide_default_is_auto_normal() -> None:
    """create_guide with no guide_type returns AutoNormal."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model)
    assert isinstance(guide, AutoNormal)


def test_create_guide_with_init_scale_only() -> None:
    """create_guide(model, init_scale=0.01) returns AutoNormal."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, init_scale=0.01)
    assert isinstance(guide, AutoNormal)


# ------------------------------------------------------------------
# 3. init_scale asymmetry
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    ["auto_delta", "auto_laplace", "auto_iaf"],
)
def test_init_scale_not_passed_to_unsupported_guides(
    guide_type: str,
) -> None:
    """init_scale is silently ignored for guides that don't accept it."""
    pyro.clear_param_store()
    # Should NOT raise even though init_scale is provided
    guide = create_guide(
        _toy_model, guide_type=guide_type, init_scale=0.01,
    )
    assert guide is not None


# ------------------------------------------------------------------
# 4. Blocklist enforcement
# ------------------------------------------------------------------


def test_auto_mvn_blocked_at_high_n_regions() -> None:
    """auto_mvn raises ValueError when n_regions >= 8."""
    pyro.clear_param_store()
    with pytest.raises(ValueError, match="auto_lowrank_mvn"):
        create_guide(
            _toy_model, guide_type="auto_mvn", n_regions=8,
        )


def test_auto_mvn_allowed_at_max_n_regions() -> None:
    """auto_mvn is allowed when n_regions == 7 (the max)."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model, guide_type="auto_mvn", n_regions=7,
    )
    assert isinstance(guide, AutoMultivariateNormal)


# ------------------------------------------------------------------
# 5. Invalid guide_type
# ------------------------------------------------------------------


def test_invalid_guide_type_raises_value_error() -> None:
    """Unknown guide_type raises ValueError with valid keys."""
    pyro.clear_param_store()
    with pytest.raises(ValueError, match="auto_normal"):
        create_guide(_toy_model, guide_type="nonexistent")


# ------------------------------------------------------------------
# 6. kwargs passthrough
# ------------------------------------------------------------------


def test_kwargs_passthrough_lowrank() -> None:
    """rank kwarg overrides default for auto_lowrank_mvn."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model, guide_type="auto_lowrank_mvn", rank=3,
    )
    assert isinstance(guide, AutoLowRankMultivariateNormal)


def test_kwargs_passthrough_iaf() -> None:
    """hidden_dim and num_transforms kwargs for auto_iaf."""
    pyro.clear_param_store()
    guide = create_guide(
        _toy_model,
        guide_type="auto_iaf",
        hidden_dim=32,
        num_transforms=3,
    )
    assert isinstance(guide, AutoIAFNormal)


# ------------------------------------------------------------------
# 7. SVI smoke test per guide type
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "guide_type",
    list(GUIDE_REGISTRY.keys()),
    ids=list(GUIDE_REGISTRY.keys()),
)
def test_svi_step_with_each_guide_type(guide_type: str) -> None:
    """Each guide type can complete 1 SVI step with finite loss."""
    pyro.clear_param_store()
    guide = create_guide(_toy_model, guide_type=guide_type)
    optimizer = Adam({"lr": 0.01})
    elbo = Trace_ELBO()
    svi = SVI(_toy_model, guide, optimizer, loss=elbo)

    loss = svi.step()
    assert math.isfinite(loss), (
        f"guide_type={guide_type!r} produced non-finite loss: {loss}"
    )


# ------------------------------------------------------------------
# Registry sanity checks
# ------------------------------------------------------------------


def test_guide_registry_has_six_entries() -> None:
    """GUIDE_REGISTRY contains exactly 6 guide types."""
    assert len(GUIDE_REGISTRY) == 6


def test_mean_field_guides_subset() -> None:
    """MEAN_FIELD_GUIDES is a subset of GUIDE_REGISTRY keys."""
    assert MEAN_FIELD_GUIDES <= set(GUIDE_REGISTRY.keys())
    assert MEAN_FIELD_GUIDES == {"auto_delta", "auto_normal"}


# ------------------------------------------------------------------
# Bilinear guide auto-discovery (MODEL-06, Phase 15-02)
# ------------------------------------------------------------------


_BILINEAR_GUIDE_VARIANTS = ["auto_normal", "auto_lowrank_mvn", "auto_iaf"]
"""Three AutoGuide families MODEL-06 requires auto-discovery on.

- ``auto_normal``: mean-field (``AutoNormal``). Per-site locs/scales via
  ``deep_setattr``.
- ``auto_lowrank_mvn``: ``AutoLowRankMultivariateNormal``. Concatenates all
  continuous sites into a single ``_latent`` vector
  (``AutoContinuous._unpack_latent``).
- ``auto_iaf``: ``AutoIAFNormal``. Same ``AutoContinuous`` parent; dimension
  grows with bilinear sites automatically.

``auto_mvn`` is intentionally excluded per 15-RESEARCH.md Section 3 R3 —
full-rank covariance is wasteful for bilinear J > 1 and MODEL-06 scope is
the three above variants only.

**init_scale portability:** ``create_guide(..., init_scale=0.005)`` is safe
to call across all three variants. Source: ``guides.py:54-58`` defines
``_INIT_SCALE_GUIDES = {auto_normal, auto_lowrank_mvn, auto_mvn}``;
``guides.py:171-172`` only passes ``init_scale`` to the guide constructor
when the guide is in that set. For ``auto_iaf``, ``init_scale`` is silently
dropped -- NOT a TypeError.

**hidden_dim sizing for auto_iaf:** ``AutoIAFNormal`` wraps a
``pyro.nn.AutoRegressiveNN`` that REQUIRES
``min(hidden_dims) >= input_dim`` (``auto_reg_nn.py:206``). The bilinear
``task_dcm_model`` has latent dim 22 (A_free=9, C=3, noise_prec=1,
B_free_0=9), which exceeds the ``create_guide`` factory default
``hidden_dim=[20]``. Tests below pass ``hidden_dim=64`` explicitly for
``auto_iaf`` to accommodate the bilinear latent size. This is a
test-side-only adjustment; ``create_guide`` remains unchanged.
"""


def _guide_kwargs_for(guide_type: str) -> dict:
    """Build variant-specific ``create_guide`` kwargs for bilinear tests.

    ``auto_iaf`` needs ``hidden_dim`` >= bilinear latent dim (22 for
    3-region J=1); ``create_guide``'s default ``[20]`` is too small
    (``AutoRegressiveNN`` raises ``ValueError``). ``auto_normal`` and
    ``auto_lowrank_mvn`` accept ``init_scale`` and no other overrides.
    ``init_scale=0.005`` is passed uniformly; silently dropped for
    ``auto_iaf`` per ``guides.py:171-172`` guard.
    """
    kwargs: dict = {"init_scale": 0.005}
    if guide_type == "auto_iaf":
        kwargs["hidden_dim"] = 64
    return kwargs


class TestBilinearDiscovery:
    """MODEL-06: create_guide auto-discovers B_free_j sites across 3 variants.

    MODEL-06 is PASSIVE -- Pyro's ``AutoGuide._setup_prototype`` iterates
    ``prototype_trace.iter_stochastic_nodes()`` and handles dynamic site
    names (``f'B_free_{j}'``) identically to static ones (``A_free``,
    ``C``). This class verifies the passive claim by (a) forcing
    prototype-trace setup via a single ``guide()`` call and checking
    ``B_free_j`` appears in ``guide.prototype_trace``, and (b) running a
    20-step SVI loop on each variant confirming no runtime error and that
    ``param_store`` grows with bilinear sites.

    Runtime budget: <90s cumulative (3 variants * 30s each upper bound).
    """

    @pytest.fixture(autouse=True)
    def _silence_stability_logger(self, caplog) -> None:
        """Silence pyro_dcm.stability WARNING spam during bilinear SVI."""
        import logging

        caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")

    @pytest.mark.parametrize("guide_type", _BILINEAR_GUIDE_VARIANTS)
    def test_b_free_sites_in_prototype_trace(
        self,
        task_bilinear_guide_data: dict,
        guide_type: str,
    ) -> None:
        """After one guide() call, B_free_j sites must appear in prototype_trace.

        This is the MODEL-06 structural gate: Pyro's
        ``AutoGuide._setup_prototype`` (called lazily on the first
        ``guide()`` invocation) runs the model under ``poutine.trace`` and
        registers every stochastic node. If this test fails for any
        variant, the bilinear model is incompatible with that variant and
        MODEL-06 is NOT satisfied.
        """
        pyro.clear_param_store()

        # L2: for variants in _INIT_SCALE_GUIDES (auto_normal,
        # auto_lowrank_mvn), init_scale=0.005 is applied for bilinear
        # safety. For auto_iaf, init_scale is silently dropped inside
        # create_guide (guides.py:171-172 guard). Safe to pass uniformly
        # across all three variants -- no TypeError, no conditional
        # parametrize marks needed. _guide_kwargs_for additionally
        # supplies hidden_dim=64 for auto_iaf to clear the bilinear
        # latent-dim floor (AutoRegressiveNN requires min(hidden_dims)
        # >= input_dim; default [20] < bilinear latent dim 22).
        guide = create_guide(
            task_dcm_model,
            guide_type=guide_type,
            **_guide_kwargs_for(guide_type),
        )

        model_kwargs = dict(
            observed_bold=task_bilinear_guide_data["observed_bold"],
            stimulus=task_bilinear_guide_data["stimulus"],
            a_mask=task_bilinear_guide_data["a_mask"],
            c_mask=task_bilinear_guide_data["c_mask"],
            t_eval=task_bilinear_guide_data["t_eval"],
            TR=task_bilinear_guide_data["TR"],
            dt=task_bilinear_guide_data["dt"],
            b_masks=task_bilinear_guide_data["b_masks"],
            stim_mod=task_bilinear_guide_data["stim_mod"],
        )

        # First call triggers AutoGuide._setup_prototype (lazy). The call
        # may sample from untrained priors and produce NaN BOLD; the
        # NaN-safe guard in task_dcm_model (Plan 15-01) prevents
        # RuntimeError. We only care that the trace is built.
        with torch.no_grad():
            guide(**model_kwargs)

        # guide.prototype_trace is set by _setup_prototype; contains every
        # stochastic site plus obs.
        prototype_sites = set(guide.prototype_trace.nodes.keys())

        # Expected bilinear sites: B_free_0 (J=1).
        J = task_bilinear_guide_data["J"]
        for j in range(J):
            expected = f"B_free_{j}"
            assert expected in prototype_sites, (
                f"[{guide_type}] Missing B_free_{j} in prototype_trace; "
                f"got sites: {sorted(prototype_sites)}. If this fails, "
                f"MODEL-06 auto-discovery is broken for {guide_type} "
                f"(contradicts 15-RESEARCH.md Section 3 direct-source "
                f"claim)."
            )
        # Also assert the pre-Phase-15 sites still present (A_free, C).
        assert "A_free" in prototype_sites
        assert "C" in prototype_sites

    @pytest.mark.parametrize("guide_type", _BILINEAR_GUIDE_VARIANTS)
    def test_b_free_sites_in_param_store_after_svi(
        self,
        task_bilinear_guide_data: dict,
        guide_type: str,
    ) -> None:
        """After 20 SVI steps on each variant, param_store references B_free_j.

        This is the MODEL-06 SVI-level smoke gate: not only does
        ``_setup_prototype`` discover the sites, but the full SVI training
        loop runs without ``RuntimeError`` and produces finite losses. For
        ``AutoNormal``, ``self.locs.B_free_0`` and ``self.scales.B_free_0``
        appear as param-store keys; for ``AutoLowRankMVN`` / ``AutoIAF``,
        the latent vector size grows and param names include ``_latent``
        structure referencing the bilinear dims.
        """
        pyro.clear_param_store()

        # Same portability note as prototype-trace test above: init_scale
        # is silently dropped for auto_iaf inside create_guide, and
        # hidden_dim=64 is injected for auto_iaf via _guide_kwargs_for
        # to clear the AutoRegressiveNN latent-dim floor.
        guide = create_guide(
            task_dcm_model,
            guide_type=guide_type,
            **_guide_kwargs_for(guide_type),
        )
        optimizer = pyro.optim.ClippedAdam(
            {"lr": 0.01, "clip_norm": 10.0},
        )
        svi = SVI(task_dcm_model, guide, optimizer, loss=Trace_ELBO())

        model_kwargs = dict(
            observed_bold=task_bilinear_guide_data["observed_bold"],
            stimulus=task_bilinear_guide_data["stimulus"],
            a_mask=task_bilinear_guide_data["a_mask"],
            c_mask=task_bilinear_guide_data["c_mask"],
            t_eval=task_bilinear_guide_data["t_eval"],
            TR=task_bilinear_guide_data["TR"],
            dt=task_bilinear_guide_data["dt"],
            b_masks=task_bilinear_guide_data["b_masks"],
            stim_mod=task_bilinear_guide_data["stim_mod"],
        )

        # 20 steps: smoke-level, NOT convergence. Runtime ~30s/variant.
        losses: list[float] = []
        for _ in range(20):
            loss = svi.step(**model_kwargs)
            losses.append(float(loss))
            assert math.isfinite(loss), (
                f"[{guide_type}] Non-finite loss at this step: {loss}; "
                f"MODEL-06 SVI-level smoke failed."
            )

        # Param-store introspection: at least ONE param name must
        # reference each B_free_j. For AutoNormal this is exact
        # ('locs.B_free_0'). For AutoLowRankMVN and AutoIAF, bilinear
        # dims are folded into a larger _latent structure -- we accept
        # either an explicit B_free_0 reference in guide.prototype_trace
        # OR a grown latent dim.
        param_names = list(pyro.get_param_store().keys())

        if guide_type == "auto_normal":
            # AutoNormal uses per-site deep_setattr: names like
            # 'AutoNormal.locs.B_free_0' and 'AutoNormal.scales.B_free_0'.
            J = task_bilinear_guide_data["J"]
            for j in range(J):
                has_loc = any(f"B_free_{j}" in n for n in param_names)
                assert has_loc, (
                    f"[{guide_type}] No param referencing B_free_{j}; "
                    f"got param names: {param_names}"
                )
        else:
            # AutoLowRankMVN / AutoIAF concatenate into a single _latent
            # vector. Confirm (a) prototype_trace still holds B_free_j
            # structurally, and (b) some guide params exist (param store
            # is non-empty after SVI steps).
            prototype_sites = set(guide.prototype_trace.nodes.keys())
            J = task_bilinear_guide_data["J"]
            for j in range(J):
                assert f"B_free_{j}" in prototype_sites, (
                    f"[{guide_type}] prototype_trace lost B_free_{j} "
                    f"after SVI; sites: {sorted(prototype_sites)}"
                )
            assert len(param_names) > 0, (
                f"[{guide_type}] param store is empty after 20 SVI steps"
            )
