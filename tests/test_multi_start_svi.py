"""Tests for multi-start SVI functionality in guides.py.

Verifies:
- Backward compatibility: n_restarts=1 returns identical dict structure
- Multi-restart best-selection logic
- guide_factory requirement enforcement
- Fresh initialization per restart
- NaN resilience (inf penalty, continue to next)
- Param store restoration to best restart's state
"""

from __future__ import annotations

import math
from functools import partial
from unittest.mock import patch

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import SVI

from pyro_dcm.models.guides import create_guide, run_svi

# ------------------------------------------------------------------
# Shared toy model for tests
# ------------------------------------------------------------------


def _conjugate_model(data: torch.Tensor) -> None:
    """Normal-Normal conjugate model for fast, deterministic testing.

    Parameters
    ----------
    data : torch.Tensor
        Observed scalar value.
    """
    mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
    pyro.sample("obs", dist.Normal(mu, 0.1), obs=data)


# ------------------------------------------------------------------
# Test 1: Backward compatibility
# ------------------------------------------------------------------


class TestSingleRestartBackwardCompat:
    """run_svi with n_restarts=1 returns identical dict to pre-Phase-20."""

    def test_single_restart_backward_compat(self) -> None:
        """n_restarts=1 result has exactly {losses, final_loss, num_steps}.

        No 'all_restarts', no 'n_restarts', no 'best_restart_idx'.
        """
        data = torch.tensor(3.0)
        guide = create_guide(
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )
        result = run_svi(
            _conjugate_model,
            guide,
            (data,),
            num_steps=50,
        )

        expected_keys = {"losses", "final_loss", "num_steps"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}. "
            f"Backward compatibility broken: n_restarts=1 must not add "
            f"multi-restart keys."
        )
        assert len(result["losses"]) == 50
        assert result["num_steps"] == 50
        assert not math.isnan(result["final_loss"])

    def test_default_n_restarts_is_one(self) -> None:
        """Default call (no n_restarts arg) returns single-restart dict."""
        data = torch.tensor(3.0)
        guide = create_guide(
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )
        result = run_svi(_conjugate_model, guide, (data,), num_steps=30)

        assert "all_restarts" not in result
        assert "n_restarts" not in result
        assert "best_restart_idx" not in result


# ------------------------------------------------------------------
# Test 2: Multi-restart returns best
# ------------------------------------------------------------------


class TestMultiRestartReturnsBest:
    """run_svi with n_restarts=5 selects lowest final_loss."""

    def test_multi_restart_returns_best(self) -> None:
        """5 restarts: best_restart_idx corresponds to min final_loss."""
        data = torch.tensor(3.0)
        gf = partial(
            create_guide,
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        result = run_svi(
            _conjugate_model,
            gf(),
            (data,),
            num_steps=100,
            n_restarts=5,
            guide_factory=gf,
        )

        # Structure checks
        assert "all_restarts" in result
        assert len(result["all_restarts"]) == 5
        assert result["n_restarts"] == 5
        assert isinstance(result["best_restart_idx"], int)
        assert 0 <= result["best_restart_idx"] < 5

        # Best selection check
        all_final_losses = [
            r["final_loss"] for r in result["all_restarts"]
        ]
        expected_best_loss = min(all_final_losses)
        assert result["final_loss"] == expected_best_loss, (
            f"final_loss {result['final_loss']} != "
            f"min of all restarts {expected_best_loss}. "
            f"Best-selection logic is broken."
        )

        # Verify best_restart_idx matches
        best_idx = result["best_restart_idx"]
        assert (
            result["all_restarts"][best_idx]["final_loss"]
            == result["final_loss"]
        )


# ------------------------------------------------------------------
# Test 3: guide_factory required for n_restarts > 1
# ------------------------------------------------------------------


class TestMultiRestartRequiresGuideFactory:
    """n_restarts > 1 without guide_factory raises ValueError."""

    def test_multi_restart_requires_guide_factory(self) -> None:
        """ValueError raised when n_restarts=3, guide_factory=None."""
        data = torch.tensor(3.0)
        guide = create_guide(
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        with pytest.raises(ValueError, match="guide_factory"):
            run_svi(
                _conjugate_model,
                guide,
                (data,),
                num_steps=50,
                n_restarts=3,
                guide_factory=None,
            )

    def test_multi_restart_explicit_none_guide_factory(self) -> None:
        """Explicitly passing guide_factory=None with n_restarts>1 fails."""
        data = torch.tensor(3.0)
        guide = create_guide(
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        with pytest.raises(ValueError, match="guide_factory"):
            run_svi(
                _conjugate_model,
                guide,
                (data,),
                num_steps=50,
                n_restarts=2,
            )


# ------------------------------------------------------------------
# Test 4: Fresh initialization per restart
# ------------------------------------------------------------------


class TestMultiRestartFreshInit:
    """Each restart begins from independently initialized guide."""

    def test_multi_restart_fresh_init(self) -> None:
        """3 restarts produce 3 independent loss trajectories."""
        data = torch.tensor(3.0)
        gf = partial(
            create_guide,
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        result = run_svi(
            _conjugate_model,
            gf(),
            (data,),
            num_steps=80,
            n_restarts=3,
            guide_factory=gf,
        )

        # Each restart should have its own loss trajectory
        assert len(result["all_restarts"]) == 3
        for i, restart in enumerate(result["all_restarts"]):
            assert "losses" in restart
            assert len(restart["losses"]) == 80
            assert restart["restart"] == i
            assert restart["num_steps"] == 80

        # Loss trajectories should differ (independent starts)
        # Due to random init, first losses should not be identical
        first_losses = [
            r["losses"][0] for r in result["all_restarts"]
        ]
        # At least 2 of 3 first losses should differ
        unique_first = len(set(first_losses))
        assert unique_first >= 2, (
            f"All 3 restarts produced identical first loss "
            f"{first_losses[0]:.6f}. Fresh initialization "
            f"likely not working."
        )


# ------------------------------------------------------------------
# Test 5: NaN resilience
# ------------------------------------------------------------------


class TestMultiRestartNanResilience:
    """NaN restarts are assigned inf and skipped."""

    def test_multi_restart_nan_resilience(self) -> None:
        """If some restarts produce NaN, others still succeed.

        Uses mock to force NaN on specific restarts while allowing
        others to converge normally.
        """
        data = torch.tensor(3.0)
        gf = partial(
            create_guide,
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        call_counter = [0]
        orig_step = SVI.step

        def sometimes_nan_step(self, *args, **kwargs):
            """Return NaN on first 2 restarts (first 2 * num_steps calls)."""
            call_counter[0] += 1
            # Each restart runs num_steps=50 steps.
            # Force NaN on restarts 0 and 1 (first 100 calls).
            if call_counter[0] <= 100:
                # Return NaN on step 5 of each restart (calls 5 and 55)
                if call_counter[0] in (5, 55):
                    return float("nan")
            return orig_step(self, *args, **kwargs)

        with patch.object(SVI, "step", sometimes_nan_step):
            result = run_svi(
                _conjugate_model,
                gf(),
                (data,),
                num_steps=50,
                n_restarts=5,
                guide_factory=gf,
            )

        # At least some restarts should have inf (NaN detected)
        inf_restarts = [
            r for r in result["all_restarts"]
            if r["final_loss"] == float("inf")
        ]
        finite_restarts = [
            r for r in result["all_restarts"]
            if r["final_loss"] != float("inf")
        ]

        assert len(inf_restarts) >= 2, (
            f"Expected at least 2 NaN restarts, got {len(inf_restarts)}"
        )
        assert len(finite_restarts) >= 1, (
            "Expected at least 1 successful restart"
        )

        # Best should be a finite restart
        assert result["final_loss"] != float("inf"), (
            "Best restart should not have inf loss"
        )
        assert result["best_restart_idx"] >= 2, (
            f"Best restart idx should be >= 2 (first 2 had NaN), "
            f"got {result['best_restart_idx']}"
        )

    def test_all_restarts_nan_raises(self) -> None:
        """If ALL restarts produce NaN, RuntimeError is raised."""
        data = torch.tensor(3.0)
        gf = partial(
            create_guide,
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        def always_nan_step(self, *args, **kwargs):
            """Return NaN unconditionally for all SVI steps."""
            return float("nan")

        with (
            patch.object(SVI, "step", always_nan_step),
            pytest.raises(RuntimeError, match="All.*restarts produced NaN"),
        ):
            run_svi(
                _conjugate_model,
                gf(),
                (data,),
                num_steps=50,
                n_restarts=3,
                guide_factory=gf,
            )


# ------------------------------------------------------------------
# Test 6: Param store restoration
# ------------------------------------------------------------------


class TestMultiRestartParamStoreRestored:
    """After multi-start, param store matches best restart."""

    def test_multi_restart_param_store_restored(self) -> None:
        """Param store after run_svi contains best restart's parameters.

        Verifies by checking that the param store state corresponds to
        the restart with the lowest ELBO, not the last restart run.
        """
        data = torch.tensor(3.0)
        gf = partial(
            create_guide,
            _conjugate_model,
            guide_type="auto_normal",
            init_scale=0.1,
        )

        result = run_svi(
            _conjugate_model,
            gf(),
            (data,),
            num_steps=100,
            n_restarts=3,
            guide_factory=gf,
        )

        # Verify param store is non-empty after multi-start
        param_store = pyro.get_param_store()
        assert len(param_store) > 0, (
            "Param store is empty after multi-start SVI"
        )

        # Create a guide from the factory and evaluate loss.
        # The loss should match the best restart's final_loss
        # (approximately, since one more eval on the same params
        # may differ slightly due to stochasticity in ELBO).
        fresh_guide = gf()
        from pyro.infer import Trace_ELBO

        svi = SVI(
            _conjugate_model,
            fresh_guide,
            pyro.optim.ClippedAdam({"lr": 0.001}),
            loss=Trace_ELBO(),
        )
        # Evaluate loss without stepping (just computing ELBO)
        eval_loss = svi.evaluate_loss(data)

        # The eval_loss should be close to best restart's final loss
        # (not exactly equal due to ELBO stochasticity, but within
        # reasonable tolerance for this conjugate model)
        best_loss = result["final_loss"]
        assert abs(eval_loss - best_loss) < 50.0, (
            f"Param store loss {eval_loss:.2f} differs significantly "
            f"from best restart loss {best_loss:.2f}. Param store "
            f"may not be restored to best restart state."
        )

        # Stronger check: the best_restart_idx should point to the
        # restart whose loss we'd approximately reproduce
        best_idx = result["best_restart_idx"]
        last_idx = len(result["all_restarts"]) - 1
        if best_idx != last_idx:
            # If best != last, the restore was non-trivial.
            # Verify we're closer to best than to last restart.
            last_loss = result["all_restarts"][last_idx]["final_loss"]
            if last_loss != float("inf"):
                dist_to_best = abs(eval_loss - best_loss)
                dist_to_last = abs(eval_loss - last_loss)
                # We should be closer to best than last
                # (unless they're very similar)
                if abs(best_loss - last_loss) > 10.0:
                    assert dist_to_best <= dist_to_last, (
                        f"Param store appears to match last restart "
                        f"({last_loss:.2f}) rather than best restart "
                        f"({best_loss:.2f}). Restoration failed."
                    )
