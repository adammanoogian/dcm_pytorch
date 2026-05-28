"""End-to-end task DCM pipeline demo using MNE-Python IO (PIPE-02).

Demonstrates the complete workflow:

    synthetic MNE Epochs -> epochs_to_timeseries -> task_dcm_model
    (with bilinear B matrices) -> SVI -> posterior A + B matrices
    with recovery metrics.

Intended as a self-contained, copy-pasteable starting point for task DCM
workflows that use MNE-Python data as input.

Expected runtime: ~2-5 min on CPU (100 SVI steps, DURATION=100s, N=3).
For publication-quality recovery use DURATION=300-600s and 1000-1500 SVI
steps submitted to a cluster (Phase 16 pattern).

Produces:
    - Posterior median A matrix (N, N) and recovery RMSE vs ground truth.
    - Posterior median B matrix (J, N, N) and recovery RMSE + sign match.
    - Final SVI loss (mean of last 10 steps).
"""
from __future__ import annotations

import sys
from functools import partial

try:
    import mne
except ImportError:
    print("MNE-Python required. Install with: pip install pyro-dcm[mne]")
    sys.exit(1)

import numpy as np
import torch

from pyro_dcm import (
    PiecewiseConstantInput,
    create_guide,
    extract_posterior_params,
    make_block_stimulus,
    make_event_stimulus,
    parameterize_A,
    parameterize_B,
    run_svi,
    simulate_task_dcm,
    task_dcm_model,
)
from pyro_dcm.io import epochs_to_timeseries


def main() -> None:
    """Run end-to-end task DCM pipeline demo with MNE IO bridge."""
    torch.manual_seed(42)

    # --- 1. Simulation config --------------------------------------------------
    # Deliberately small for local demonstration. For real recovery metrics
    # use DURATION=300-600 and NUM_SVI_STEPS=1000-1500 on the cluster.
    N_REGIONS = 3
    DURATION = 100.0
    DT_SIM = 0.01   # Fine step for forward simulation only
    DT_MODEL = 0.5  # SVI integration step (coarser for efficiency)
    TR = 2.0
    SNR = 3.0
    NUM_SVI_STEPS = 100

    # --- 2. Ground-truth DCM circuit -------------------------------------------
    # Off-diagonal effective connectivity (parameterize_A clamps diagonal
    # negative per SPM12 convention).
    A_free_true = torch.zeros(N_REGIONS, N_REGIONS, dtype=torch.float64)
    A_free_true[1, 0] = 0.3  # region 0 -> 1
    A_free_true[2, 1] = 0.3  # region 1 -> 2
    A_true = parameterize_A(A_free_true)

    # C: driving input into region 0 only.
    C_true = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
    C_true[0, 0] = 0.5

    # B mask + ground truth: modulator gates the 0->1 edge.
    # b_mask shape: (J=1, N, N); J = number of modulatory inputs.
    b_mask = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
    b_mask[0, 1, 0] = 1.0
    B_free_true = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
    B_free_true[0, 1, 0] = 0.4
    B_true = parameterize_B(B_free_true, b_mask)  # (J=1, N, N)

    # --- 3. Stimulus construction ----------------------------------------------
    # Driving input: block design entering region 0.
    stimulus_driving = make_block_stimulus(
        n_blocks=3,
        block_duration=15.0,
        rest_duration=15.0,
        n_inputs=1,
    )

    # Modulator: stick events at fixed times, wrapped in PiecewiseConstantInput.
    stim_mod_dict = make_event_stimulus(
        event_times=[10.0, 40.0, 70.0],
        event_amplitudes=[1.0, 1.0, 1.0],
        duration=DURATION,
        dt=DT_SIM,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"]
    )

    # --- 4. Forward simulation -------------------------------------------------
    B_list = [B_true[j] for j in range(B_true.shape[0])]
    ts_result = simulate_task_dcm(
        A=A_true,
        C=C_true,
        stimulus=stimulus_driving,
        duration=DURATION,
        dt=DT_SIM,
        TR=TR,
        SNR=SNR,
        seed=42,
        B_list=B_list,
        stimulus_mod=stim_mod,
    )
    bold_simulated = ts_result["bold"]  # (T_TR, N) at TR resolution

    # --- 5. Wrap simulated BOLD in MNE Epochs (IO bridge) ----------------------
    # EpochsArray expects (n_epochs, n_channels, n_times); we have one "epoch".
    # BOLD shape is (T_TR, N) so we transpose to (N, T_TR).
    n_timepoints, n_regions = bold_simulated.shape
    ch_names = [f"ROI_{i}" for i in range(n_regions)]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=1.0 / TR,
        ch_types="eeg",
    )
    bold_np = bold_simulated.numpy()  # (T_TR, N)
    # EpochsArray needs (n_epochs, n_channels, n_times)
    bold_epochs_data = bold_np.T[np.newaxis, :, :]  # (1, N, T_TR)
    epochs = mne.EpochsArray(bold_epochs_data, info)

    # --- 6. MNE Epochs -> timeseries tensor ------------------------------------
    # epochs_to_timeseries returns dict with 'timeseries': (T, N).
    io_result = epochs_to_timeseries(epochs, average=True)
    observed_bold = io_result["timeseries"]  # (T_TR, N)

    print(f"Observed BOLD shape: {observed_bold.shape}")
    print(f"Channels: {io_result['ch_names']}")
    print(f"Sampling frequency: {io_result['sfreq']} Hz")

    # --- 7. Set up model args and fit via SVI ----------------------------------
    # t_eval is constructed from DURATION and DT_MODEL, NOT from ts_result times.
    # The spacing of t_eval must equal DT_MODEL (model contract).
    t_eval = torch.arange(
        0.0, DURATION + DT_MODEL / 2.0, DT_MODEL, dtype=torch.float64
    )

    a_mask = torch.ones(N_REGIONS, N_REGIONS, dtype=torch.float64)
    c_mask = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    # b_masks: list of per-modulator (N, N) masks.
    b_masks_list = [b_mask[j] for j in range(b_mask.shape[0])]
    model_kwargs = {"b_masks": b_masks_list, "stim_mod": stim_mod}

    guide = create_guide(
        task_dcm_model, guide_type="auto_normal", init_scale=0.005
    )

    model_args = (
        observed_bold,
        stimulus_driving,
        a_mask,
        c_mask,
        t_eval,
        TR,
        DT_MODEL,
    )

    svi_result = run_svi(
        model=task_dcm_model,
        guide=guide,
        model_args=model_args,
        num_steps=NUM_SVI_STEPS,
        lr=0.02,
        model_kwargs=model_kwargs,
    )

    # --- 8. Extract posterior and recovery metrics -----------------------------
    model_for_pred = partial(task_dcm_model, **model_kwargs)
    posterior = extract_posterior_params(
        guide=guide,
        model_args=model_args,
        model=model_for_pred,
        num_samples=200,
    )

    A_est = posterior["median"]["A"]  # (N, N)
    B_est = posterior["median"]["B"]  # (J, N, N)

    a_rmse = torch.sqrt(((A_est - A_true) ** 2).mean()).item()

    nonzero_idx = b_mask[0].bool()  # (N, N) for modulator j=0
    b_true_vec = B_true[0][nonzero_idx]
    b_est_vec = B_est[0][nonzero_idx]
    b_rmse = torch.sqrt(((b_est_vec - b_true_vec) ** 2).mean()).item()
    b_sign_match = (
        (b_est_vec.sign() == b_true_vec.sign()).float().mean().item()
    )

    print(f"\nFinal SVI loss (last 10 mean): "
          f"{np.mean(svi_result['losses'][-10:]):.2f}")
    print(f"A-RMSE:              {a_rmse:.3f}")
    print(f"B-RMSE (mask=1):     {b_rmse:.3f}")
    print(f"B sign recovery:     {b_sign_match:.2f}")
    print(f"\nA_true:\n{A_true.numpy().round(3)}")
    print(f"A_est:\n{A_est.numpy().round(3)}")
    print(f"\nB_true (non-zero):   {b_true_vec.tolist()}")
    print(f"B_est  (non-zero):   {[round(float(x), 3) for x in b_est_vec]}")


if __name__ == "__main__":
    main()
