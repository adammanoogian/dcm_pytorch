"""Latent trajectory extraction and PCA dimensionality reduction for CT-RNN.

Provides trajectory extraction from trained RNNs, PCA-based dimensionality
reduction to an N-dimensional latent space, an output-R-squared quality gate,
and a z-score normalization utility. These utilities implement requirements
DIM-01, DIM-02, and DIM-03 of the v0.6.0 Latent Circuit DCM milestone.

References
----------
Langdon & Engel (2025) trainRNNbrain (interim cite; formal REF-ID in Phase 25).
Dubreuil et al. (2024) eLife -- pitfall LC3: PCA variance != task relevance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pyro_dcm.rnn.continuous_time_rnn import ContinuousTimeRNN


def extract_trajectories(
    rnn: ContinuousTimeRNN,
    env: Any,
    n_trials_per_condition: int = 50,
    conditions: list[dict] | None = None,
    device: torch.device | None = None,
    max_steps_per_trial: int = 1000,
) -> dict[str, np.ndarray]:
    """Extract hidden-state trajectories from a trained CT-RNN.

    Runs the RNN in evaluation mode (no training noise) on neurogym-generated
    trials and collects the hidden-state trajectory ``h(t)`` for each trial.

    Parameters
    ----------
    rnn : ContinuousTimeRNN
        Trained CT-RNN model. Put in eval mode before trajectory extraction.
    env : neurogym.Env or compatible
        Neurogym environment instance (e.g., ``ContextDecisionMaking-v0``).
        Must have ``observation_space``, ``action_space``, ``dt`` attributes
        and a ``reset()`` / ``step()`` Gymnasium-compatible API.
        Type annotated as ``Any`` to avoid import at module level.
    n_trials_per_condition : int, optional
        Number of trials to run per condition. Default 50.
    conditions : list of dict or None, optional
        List of condition dictionaries (e.g., ``[{"context_id": 0}]``).
        If ``None``, runs ``n_trials_per_condition`` trials with default
        environment sampling and stores them under the key ``"default"``.
    device : torch.device or None, optional
        Device for RNN computation. If ``None``, uses ``cpu``.
    max_steps_per_trial : int, optional
        Maximum number of env steps per trial. Prevents infinite loops
        when the environment never returns ``terminated=True`` (e.g.,
        neurogym 2.2 ContextDecisionMaking). Default 1000.

    Returns
    -------
    dict of str -> np.ndarray
        Maps condition key strings to trajectory arrays of shape
        ``(n_trials, T, H)`` where ``T`` is the trial length and
        ``H = rnn.n_hidden``.
        The returned dict also contains the metadata key
        ``"__meta__"`` : dict with ``dt_seconds`` (float), ``tau`` (float),
        ``alpha`` (float).

    Notes
    -----
    dt/tau metadata is stored in ``result["__meta__"]`` to support Phase 22
    (PIPE-03) time-grid alignment. Neurogym ``env.dt`` is in milliseconds;
    ``dt_seconds = env.dt / 1000`` converts to seconds (pitfall LC10).
    Trajectories are returned as numpy arrays (float32), not tensors, for
    compatibility with downstream PCA and ``.npz`` storage.
    """
    if device is None:
        device = torch.device("cpu")

    rnn = rnn.to(device)
    rnn.eval()

    dt_ms: float = float(getattr(env, "dt", 100.0))
    dt_seconds: float = dt_ms / 1000.0

    result: dict[str, np.ndarray] = {}

    def _run_trials(n_trials: int) -> np.ndarray:
        """Run n_trials and return trajectories (n_trials, T, H)."""
        trajs: list[np.ndarray] = []
        for _ in range(n_trials):
            obs, _ = env.reset() if hasattr(env, "reset") else (env.reset(), {})
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)

            obs_seq: list[np.ndarray] = [obs]
            done = False
            step_count = 0
            while not done and step_count < max_steps_per_trial:
                action = env.action_space.sample()
                step_result = env.step(action)
                obs, _, terminated, truncated, *_ = (
                    step_result
                    if len(step_result) >= 5
                    else (*step_result[:3], False)
                )
                done = terminated or truncated
                obs_seq.append(
                    obs if isinstance(obs, np.ndarray) else np.array(obs)
                )
                step_count += 1

            # obs_seq: list of T observations, each shape (M_in,)
            u_np = np.stack(obs_seq, axis=0).astype(np.float32)  # (T, M_in)
            u_t = torch.tensor(u_np, device=device)  # (T, M_in)

            with torch.no_grad():
                _, h_traj = rnn(u_t)  # h_traj: (T, 1, H) due to unbatched promotion
            # h_traj is (T, 1, H) -> squeeze batch dim -> (T, H)
            trajs.append(h_traj.squeeze(1).cpu().numpy().astype(np.float32))

        # Pad or truncate to common length (use shortest trial)
        min_t = min(t.shape[0] for t in trajs)
        padded = np.stack([t[:min_t] for t in trajs], axis=0)  # (n_trials, T, H)
        return padded

    if conditions is None:
        result["default"] = _run_trials(n_trials_per_condition)
    else:
        for cond in conditions:
            key = "_".join(f"{k}{v}" for k, v in sorted(cond.items()))
            result[key] = _run_trials(n_trials_per_condition)

    result["__meta__"] = np.array(  # type: ignore[assignment]
        [],
        dtype=np.float32,
    )
    # Store metadata as a plain dict accessible via the special key
    # (np.ndarray storage limitation; callers should check isinstance)
    result["__meta__"] = {  # type: ignore[assignment]
        "dt_seconds": dt_seconds,
        "tau": rnn.tau,
        "alpha": rnn.alpha,
    }
    return result


def pca_reduce(
    h_all: np.ndarray,
    n_components: int,
) -> tuple[Any, np.ndarray]:
    """Fit PCA on hidden-state data and return projected coordinates.

    Implements DIM-01: maps H-dimensional CT-RNN hidden states to an
    N-dimensional latent space via principal component analysis.

    Parameters
    ----------
    h_all : np.ndarray, shape (n_samples, H)
        All trajectory samples stacked across trials and time steps.
        Should be the training split only (not the held-out test set) to
        avoid data leakage into the output-R-squared gate.
    n_components : int
        Number of principal components (latent dimensions N).

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object. Use ``pca.components_`` (shape ``(N, H)``) and
        ``pca.explained_variance_ratio_`` for downstream diagnostics.
    projected : np.ndarray, shape (n_samples, N)
        Hidden states projected onto the top-N principal components.

    Notes
    -----
    sklearn is imported lazily inside this function so the module can be
    imported without scikit-learn installed. Raises ``ImportError`` with an
    install hint if sklearn is absent.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for pca_reduce. "
            "Install with: pip install 'pyro-dcm[latent]'"
        ) from exc

    pca = PCA(n_components=n_components)
    projected: np.ndarray = pca.fit_transform(h_all)
    return pca, projected


def variance_explained_diagnostic(pca: Any) -> dict:
    """Compute PCA variance-explained diagnostics and recommend latent dimension N.

    Implements DIM-03: reports cumulative and marginal variance explained per
    principal component and recommends the smallest N where the next component
    adds less than 5% additional variance.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        A fitted PCA object with ``explained_variance_ratio_`` attribute.

    Returns
    -------
    dict
        Keys:

        ``cumulative`` : np.ndarray, shape (n_components,)
            Cumulative fraction of variance explained by top-k components.
        ``marginal`` : np.ndarray, shape (n_components,)
            Fraction of variance explained by each individual component
            (i.e., ``pca.explained_variance_ratio_``).
        ``recommended_n`` : int
            Smallest N such that the (N+1)-th component contributes less
            than 5% additional variance. If no component drops below 5%,
            returns the total number of fitted components.

    Notes
    -----
    The recommendation uses the *next* component's marginal gain: for each
    candidate N (1-indexed), check whether ``marginal[N] < 0.05``.
    The first N satisfying this gives the recommendation.
    """
    marginal: np.ndarray = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative: np.ndarray = np.cumsum(marginal)

    # recommended_n: first N where the NEXT component's gain < 5%
    # i.e., find first index i (0-based) where marginal[i] < 0.05 -> recommended = i
    below_threshold = np.where(marginal < 0.05)[0]
    if len(below_threshold) > 0:
        recommended_n = int(below_threshold[0])
        # Ensure at least 1
        recommended_n = max(recommended_n, 1)
    else:
        recommended_n = len(marginal)

    return {
        "cumulative": cumulative,
        "marginal": marginal,
        "recommended_n": recommended_n,
    }


def output_r_squared_gate(
    h_projected: np.ndarray,
    z_true: np.ndarray,
    w_out: np.ndarray,
    pca: Any,
    threshold: float = 0.90,
) -> dict:
    """Verify that PCA-projected states reconstruct the behavioral readout.

    Implements DIM-02: computes R-squared between the true RNN output and
    the output reconstructed from PCA-projected hidden states. Returns
    ``passed=True`` if ``r_squared >= threshold``.

    The output weight matrix ``W_out`` (shape ``(act_size, H)``) maps hidden
    states to readout logits. In PCA space the effective weights become
    ``W_out_pca = W_out @ pca.components_.T`` (shape ``(act_size, N)``),
    and the reconstructed readout is ``z_pred = h_projected @ W_out_pca.T``.

    Parameters
    ----------
    h_projected : np.ndarray, shape (n_samples, N)
        PCA-projected hidden states (test split).
    z_true : np.ndarray, shape (n_samples, act_size)
        True behavioral readout (pre-softmax logits from the full-dimensional
        RNN). Must be computed from the same samples as ``h_projected``.
    w_out : np.ndarray, shape (act_size, H)
        RNN output weight matrix (``rnn.W_out.detach().numpy()``).
    pca : sklearn.decomposition.PCA
        Fitted PCA object. ``pca.components_`` shape must be ``(N, H)``.
    threshold : float, optional
        DIM-02 gate threshold. Default 0.90 (as specified by DIM-02).

    Returns
    -------
    dict
        Keys:

        ``r_squared`` : float
            Coefficient of determination R2 (may be negative for poor fit).
        ``passed`` : bool
            ``True`` iff ``r_squared >= threshold``.
        ``threshold`` : float
            The threshold used for the gate.

    Notes
    -----
    Guards against zero-variance targets with a ``max(ss_tot, 1e-12)``
    denominator (avoids division by zero when z_true is constant).
    Do NOT call this on the same data used to fit PCA -- use a held-out
    test split to avoid inflated R-squared (pitfall LC3).
    """
    w_out_pca: np.ndarray = w_out @ pca.components_.T  # (act_size, N)
    z_pred: np.ndarray = h_projected @ w_out_pca.T  # (n_samples, act_size)

    ss_res = float(np.sum((z_true - z_pred) ** 2))
    ss_tot = float(np.sum((z_true - z_true.mean(axis=0)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "r_squared": float(r_squared),
        "passed": bool(r_squared >= threshold),
        "threshold": float(threshold),
    }


def zscore_trajectories(
    h_projected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score PCA-projected trajectories to zero mean and unit variance.

    Addresses pitfall LC4: raw PCA trajectories may have amplitudes O(1)-O(10),
    mismatched with DCM priors calibrated for BOLD data. Z-scoring each PC
    independently normalizes amplitude before DCM fitting. Store the returned
    means and stds for inverse-normalization (needed for perturbation experiments
    and interpretation of the fitted A matrix).

    Parameters
    ----------
    h_projected : np.ndarray, shape (n_samples, N)
        PCA-projected hidden-state trajectories across all time points and
        trials.

    Returns
    -------
    z_scored : np.ndarray, shape (n_samples, N)
        Z-scored trajectories with mean ~0 and std ~1 per column.
    means : np.ndarray, shape (N,)
        Per-PC column means used for normalization.
    stds : np.ndarray, shape (N,)
        Per-PC column standard deviations used for normalization.
        Clipped to a minimum of ``1e-8`` to prevent division by zero.

    Notes
    -----
    Inverse transformation: ``h_original = z_scored * stds + means``.
    Clip std to ``max(std, 1e-8)`` before dividing to handle degenerate
    PCs (near-constant after projection).
    """
    means: np.ndarray = h_projected.mean(axis=0)  # (N,)
    stds: np.ndarray = h_projected.std(axis=0)  # (N,)
    stds = np.clip(stds, 1e-8, None)
    z_scored: np.ndarray = (h_projected - means) / stds
    return z_scored, means, stds
