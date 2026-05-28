"""Integration tests for rnn_trainer: train_rnn and eval_rnn_performance.

All tests requiring neurogym are marked with ``@pytest.mark.latent``.
Slow convergence tests (2000+ steps) are additionally marked
``@pytest.mark.slow`` and excluded from routine CI.

Run smoke tests only::

    pytest tests/test_rnn_trainer.py -v -m "latent and not slow"
"""

from __future__ import annotations

import math

import pytest

from pyro_dcm.rnn import ContinuousTimeRNN, eval_rnn_performance, train_rnn


def _make_cddm_rnn(batch_size: int = 16, seq_len: int = 50) -> ContinuousTimeRNN:
    """Construct a ContinuousTimeRNN sized to the CDDM task dimensions.

    Creates a fresh neurogym environment to query ob_size and act_size,
    then returns a small RNN suitable for smoke tests.

    Parameters
    ----------
    batch_size : int, optional
        Batch size for the dataset (unused here, kept for caller context).
    seq_len : int, optional
        Sequence length for the dataset (unused here, kept for caller context).

    Returns
    -------
    ContinuousTimeRNN
        RNN with n_input=ob_size, n_hidden=32, n_output=act_size.
    """
    ngym = pytest.importorskip("neurogym")

    ds = ngym.Dataset(
        "ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        batch_size=batch_size,
        seq_len=seq_len,
    )
    ob_size = ds.env.observation_space.shape[0]
    act_size = ds.env.action_space.n

    return ContinuousTimeRNN(
        n_input=ob_size,
        n_hidden=32,
        n_output=act_size,
    )


@pytest.mark.latent
def test_train_rnn_smoke() -> None:
    """train_rnn runs 50 steps on CDDM and returns a complete result dict.

    Verifies:

    - Return dict has all required keys
    - ``final_loss`` is a finite float
    - ``n_steps_completed == 50``
    - ``len(loss_history) == 50``
    - ``len(accuracy_history) == 50``
    """
    pytest.importorskip("neurogym")

    rnn = _make_cddm_rnn(batch_size=16, seq_len=50)

    result = train_rnn(
        rnn,
        task="ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        n_steps=50,
        batch_size=16,
        seq_len=50,
        log_every=200,  # no logging during 50-step smoke test
    )

    required_keys = {
        "final_loss",
        "final_accuracy",
        "n_steps_completed",
        "loss_history",
        "accuracy_history",
    }
    assert required_keys == set(result.keys()), (
        f"Missing keys: {required_keys - set(result.keys())}; "
        f"Extra keys: {set(result.keys()) - required_keys}"
    )

    assert math.isfinite(result["final_loss"]), (
        f"Expected finite final_loss, got {result['final_loss']}"
    )

    assert result["n_steps_completed"] == 50, (
        f"Expected n_steps_completed=50, got {result['n_steps_completed']}"
    )

    assert len(result["loss_history"]) == 50, (
        f"Expected len(loss_history)=50, got {len(result['loss_history'])}"
    )

    assert len(result["accuracy_history"]) == 50, (
        f"Expected len(accuracy_history)=50, got {len(result['accuracy_history'])}"
    )


@pytest.mark.latent
def test_train_rnn_dimension_mismatch() -> None:
    """train_rnn raises AssertionError when RNN n_input mismatches CDDM ob_size.

    Creates a RNN with n_input=999 (CDDM has ob_size=5) and verifies
    that train_rnn raises an informative AssertionError.
    """
    pytest.importorskip("neurogym")

    # n_output=3 matches CDDM; n_input=999 does not
    rnn = ContinuousTimeRNN(n_input=999, n_hidden=16, n_output=3)

    with pytest.raises(AssertionError, match="n_input mismatch"):
        train_rnn(
            rnn,
            task="ContextDecisionMaking-v0",
            env_kwargs={"dt": 100},
            n_steps=1,
            batch_size=4,
            seq_len=20,
        )


@pytest.mark.latent
def test_eval_rnn_performance() -> None:
    """eval_rnn_performance returns accuracy in [0, 1] and a positive n_trials count.

    Uses an untrained (randomly initialised) RNN, so accuracy need not be
    high -- we only verify the return structure and value ranges.
    """
    pytest.importorskip("neurogym")

    rnn = _make_cddm_rnn(batch_size=16, seq_len=50)

    # Train for a few steps to ensure model is plausible
    train_rnn(
        rnn,
        task="ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        n_steps=10,
        batch_size=8,
        seq_len=30,
        log_every=1000,
    )

    result = eval_rnn_performance(
        rnn,
        task="ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        n_eval_steps=5,
        batch_size=8,
        seq_len=30,
    )

    assert "accuracy" in result, (
        f"Missing 'accuracy' key; got keys: {list(result.keys())}"
    )
    assert "n_trials" in result, (
        f"Missing 'n_trials' key; got keys: {list(result.keys())}"
    )

    acc = result["accuracy"]
    assert isinstance(acc, float), f"Expected float accuracy, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"Expected accuracy in [0, 1], got {acc}"

    n_trials = result["n_trials"]
    assert n_trials > 0, f"Expected positive n_trials, got {n_trials}"


@pytest.mark.latent
@pytest.mark.slow
def test_train_rnn_convergence() -> None:
    """train_rnn achieves >= 70% accuracy on CDDM after 2000 training steps.

    Uses H=64 (moderate capacity). This validates the full training loop
    end-to-end, including Adam convergence and gradient clipping.

    This test is marked slow (expected 2-5 minutes on CPU).
    """
    pytest.importorskip("neurogym")

    import neurogym as ngym

    ds = ngym.Dataset(
        "ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        batch_size=32,
        seq_len=100,
    )
    ob_size = ds.env.observation_space.shape[0]
    act_size = ds.env.action_space.n

    rnn = ContinuousTimeRNN(
        n_input=ob_size,
        n_hidden=64,
        n_output=act_size,
    )

    result = train_rnn(
        rnn,
        task="ContextDecisionMaking-v0",
        env_kwargs={"dt": 100},
        n_steps=2000,
        batch_size=32,
        seq_len=100,
        lr=1e-3,
        grad_clip=1.0,
        criterion_acc=0.85,
        log_every=200,
    )

    final_acc = result["final_accuracy"]
    assert final_acc >= 0.70, (
        f"Expected final_accuracy >= 0.70 after 2000 steps, got {final_acc:.3f}. "
        "If training diverged, check learning rate and gradient clipping."
    )
