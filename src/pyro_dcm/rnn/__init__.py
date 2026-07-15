from __future__ import annotations

from pyro_dcm.rnn.continuous_time_rnn import ContinuousTimeRNN
from pyro_dcm.rnn.fixed_point_analysis import (
    classify_stability,
    compute_jacobian_at_fp,
    find_fixed_points,
)
from pyro_dcm.rnn.trajectory_pca import (
    extract_trajectories,
    output_r_squared_gate,
    pca_reduce,
    variance_explained_diagnostic,
    zscore_trajectories,
)
from pyro_dcm.rnn.rnn_trainer import eval_rnn_performance, train_rnn

__all__ = [
    # Phase 21-01: CT-RNN module (v0.6.0)
    "ContinuousTimeRNN",
    # Phase 21-02: Training and evaluation (v0.6.0)
    "train_rnn",
    "eval_rnn_performance",
    # Phase 21-03: Fixed-point analysis (v0.6.0)
    "find_fixed_points",
    "compute_jacobian_at_fp",
    "classify_stability",
    # Phase 21-03: Latent extraction and PCA (v0.6.0)
    "extract_trajectories",
    "pca_reduce",
    "output_r_squared_gate",
    "variance_explained_diagnostic",
    "zscore_trajectories",
]
