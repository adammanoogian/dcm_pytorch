"""RNN training and evaluation utilities for CT-RNN on neurogym tasks.

Implements training infrastructure for ContinuousTimeRNN using standard
PyTorch BPTT with cross-entropy loss and Adam optimizer.

Reference: Langdon & Engel (2025) trainRNNbrain training protocol
(lr=1e-3, grad_clip=1.0, Adam optimizer, full-trial BPTT).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pyro_dcm.rnn.continuous_time_rnn import ContinuousTimeRNN


def train_rnn(
    rnn: ContinuousTimeRNN,
    task: str = "ContextDecisionMaking-v0",
    env_kwargs: dict | None = None,
    n_steps: int = 3000,
    batch_size: int = 32,
    seq_len: int = 100,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    criterion_acc: float = 0.85,
    log_every: int = 100,
    device: torch.device | None = None,
) -> dict:
    """Train a ContinuousTimeRNN on a neurogym task using BPTT.

    Uses Adam optimizer, cross-entropy loss, gradient clipping, and
    optional early stopping on accuracy criterion.

    Reference: Langdon & Engel (2025) trainRNNbrain training protocol.

    Parameters
    ----------
    rnn : ContinuousTimeRNN
        The CT-RNN module to train. Must have ``n_input`` matching the
        task observation space and ``n_output`` matching the action space.
    task : str, optional
        neurogym task name. Default ``"ContextDecisionMaking-v0"``.
    env_kwargs : dict or None, optional
        Keyword arguments for the neurogym environment. Default
        ``{"dt": 100}`` (100 ms per step).
    n_steps : int, optional
        Maximum number of training gradient steps. Default 3000.
    batch_size : int, optional
        Number of trials per gradient step. Default 32.
    seq_len : int, optional
        Sequence length (steps per trial) for the dataset. Default 100.
    lr : float, optional
        Adam learning rate. Default 1e-3.
    grad_clip : float, optional
        Maximum gradient norm for ``nn.utils.clip_grad_norm_``. Default 1.0.
    criterion_acc : float, optional
        Early-stopping accuracy threshold. Training stops after 3
        consecutive log checkpoints at or above this accuracy. Default 0.85.
    log_every : int, optional
        Log and check early-stopping condition every this many steps. Default 100.
    device : torch.device or None, optional
        Device for tensors. If None, uses ``rnn`` parameters' device.

    Returns
    -------
    dict
        Training results with keys:

        - ``"final_loss"`` : float — loss at last step
        - ``"final_accuracy"`` : float — accuracy at last step
        - ``"n_steps_completed"`` : int — number of steps run
        - ``"loss_history"`` : list[float] — loss at every step
        - ``"accuracy_history"`` : list[float] — accuracy at every step

    Raises
    ------
    AssertionError
        If ``rnn.n_input`` does not match the task observation space
        dimension, or ``rnn.n_output`` does not match the action space size.
    ImportError
        If neurogym is not installed.
    """
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required for train_rnn. "
            "Install with: pip install pyro-dcm[latent]"
        ) from e

    if env_kwargs is None:
        env_kwargs = {"dt": 100}

    if device is None:
        device = next(rnn.parameters()).device

    dataset = ngym.Dataset(
        task,
        env_kwargs=env_kwargs,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    ob_size = dataset.env.observation_space.shape[0]
    act_size = dataset.env.action_space.n

    assert rnn.n_input == ob_size, (
        f"RNN n_input mismatch: expected {ob_size} (from task '{task}'), "
        f"got {rnn.n_input}"
    )
    assert rnn.n_output == act_size, (
        f"RNN n_output mismatch: expected {act_size} (from task '{task}'), "
        f"got {rnn.n_output}"
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    final_loss = float("nan")
    final_accuracy = 0.0
    consecutive_above_criterion = 0

    rnn.train()
    rnn.to(device)

    for step in range(n_steps):
        inputs, labels = dataset()

        # inputs: (T, B, ob_size) numpy; labels: (T, B) numpy
        inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
        labels_t = torch.tensor(labels, dtype=torch.long).to(device)

        z, _ = rnn(inputs_t, h0=None)  # z: (T, B, act_size)

        # Flatten time and batch for cross-entropy loss
        z_flat = z.reshape(-1, act_size)    # (T*B, act_size)
        labels_flat = labels_t.reshape(-1)  # (T*B,)

        loss = criterion(z_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(rnn.parameters(), grad_clip)
        optimizer.step()

        step_loss = loss.item()
        step_acc = (z_flat.argmax(dim=1) == labels_flat).float().mean().item()

        loss_history.append(step_loss)
        accuracy_history.append(step_acc)
        final_loss = step_loss
        final_accuracy = step_acc

        if (step + 1) % log_every == 0:
            print(
                f"Step {step + 1}/{n_steps} | "
                f"loss={step_loss:.4f} | "
                f"acc={step_acc:.3f}"
            )
            if step_acc >= criterion_acc:
                consecutive_above_criterion += 1
            else:
                consecutive_above_criterion = 0

            if consecutive_above_criterion >= 3:
                print(
                    f"Early stopping at step {step + 1}: "
                    f"accuracy {step_acc:.3f} >= {criterion_acc} "
                    f"for 3 consecutive checks."
                )
                return {
                    "final_loss": final_loss,
                    "final_accuracy": final_accuracy,
                    "n_steps_completed": step + 1,
                    "loss_history": loss_history,
                    "accuracy_history": accuracy_history,
                }

    return {
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "n_steps_completed": n_steps,
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
    }


def eval_rnn_performance(
    rnn: ContinuousTimeRNN,
    task: str = "ContextDecisionMaking-v0",
    env_kwargs: dict | None = None,
    n_eval_steps: int = 100,
    batch_size: int = 32,
    seq_len: int = 100,
    device: torch.device | None = None,
) -> dict:
    """Evaluate RNN task performance on held-out trials.

    Runs the RNN in evaluation mode (no noise injection) across
    ``n_eval_steps`` fresh batches from the neurogym environment.

    Parameters
    ----------
    rnn : ContinuousTimeRNN
        The trained CT-RNN module to evaluate.
    task : str, optional
        neurogym task name. Default ``"ContextDecisionMaking-v0"``.
    env_kwargs : dict or None, optional
        Keyword arguments for the neurogym environment. Default
        ``{"dt": 100}``.
    n_eval_steps : int, optional
        Number of evaluation batches. Default 100.
    batch_size : int, optional
        Number of trials per evaluation batch. Default 32.
    seq_len : int, optional
        Sequence length per trial. Default 100.
    device : torch.device or None, optional
        Device for tensors. If None, uses ``rnn`` parameters' device.

    Returns
    -------
    dict
        Evaluation results with keys:

        - ``"accuracy"`` : float — mean accuracy across all evaluated trials
        - ``"n_trials"`` : int — total number of (step, trial) pairs evaluated

    Raises
    ------
    ImportError
        If neurogym is not installed.
    """
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required for eval_rnn_performance. "
            "Install with: pip install pyro-dcm[latent]"
        ) from e

    if env_kwargs is None:
        env_kwargs = {"dt": 100}

    if device is None:
        device = next(rnn.parameters()).device

    dataset = ngym.Dataset(
        task,
        env_kwargs=env_kwargs,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    act_size = dataset.env.action_space.n

    rnn.eval()
    rnn.to(device)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for _ in range(n_eval_steps):
            inputs, labels = dataset()
            inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
            labels_t = torch.tensor(labels, dtype=torch.long).to(device)

            z, _ = rnn(inputs_t, h0=None)

            z_flat = z.reshape(-1, act_size)
            labels_flat = labels_t.reshape(-1)

            correct = (z_flat.argmax(dim=1) == labels_flat).sum().item()
            total_correct += int(correct)
            total_samples += labels_flat.numel()

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "n_trials": total_samples,
    }
