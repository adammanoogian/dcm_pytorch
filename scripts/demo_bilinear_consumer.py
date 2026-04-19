"""End-to-end bilinear task-DCM demo for downstream consumers.

Demonstrates the consumer workflow for projects that import pyro-dcm as a
package (e.g. dcm_hgf_mixed_models). Mirrors the canonical example in
docs/02_pipeline_guide/consumer_bilinear_quickstart.md so the doc stays
executable.

Runs in ~30-60 s on CPU. Produces a small, self-contained recovery report.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
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


def main() -> None:
    torch.manual_seed(42)

    # --- 1. Simulation config (deliberately tiny — local-friendly demo) -----
    # For real recovery metrics use 200-600s duration and 300-1500 SVI steps
    # via the cluster pipeline in `cluster/` (Phase 16 pattern). This config
    # targets <5 min local runtime; recovery quality will be rough.
    N_REGIONS = 3
    DURATION = 80.0
    DT = 0.01
    TR = 2.0
    SNR = 3.0
    N_TRIALS = 8
    NUM_SVI_STEPS = 50

    # --- 2. Ground-truth DCM circuit ----------------------------------------
    # Off-diagonal effective connectivity (A_free drives parameterize_A; the
    # diagonal is auto-clamped negative by SPM12 convention).
    A_free_true = torch.zeros(N_REGIONS, N_REGIONS, dtype=torch.float64)
    A_free_true[1, 0] = 0.3  # region 0 -> 1
    A_free_true[2, 1] = 0.3  # region 1 -> 2
    A_true = parameterize_A(A_free_true)

    # C: driving input only into region 0.
    C_true = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
    C_true[0, 0] = 0.5

    # B mask + ground truth: modulator gates 0->1 and 1->2 edges.
    # B tensors are always 3-D (J, N, N) where J = n_modulators.
    b_mask = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
    b_mask[0, 1, 0] = 1.0
    b_mask[0, 2, 1] = 1.0
    B_free_true = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
    B_free_true[0, 1, 0] = 0.4
    B_free_true[0, 2, 1] = 0.3
    B_true = parameterize_B(B_free_true, b_mask)  # (J=1, N, N)

    # --- 3. Mock bridge-style modulator DataFrame ---------------------------
    # In dcm_hgf_mixed_models this comes from select_dcm_modulators().
    rng = np.random.RandomState(0)
    bridge_df = pd.DataFrame(
        {
            "trial_idx": np.arange(N_TRIALS),
            "outcome_time_s": np.linspace(10.0, DURATION - 20.0, N_TRIALS),
            "epsilon2": rng.randn(N_TRIALS) * 0.5,
        }
    )

    # --- 4. Stimulus construction -------------------------------------------
    # Driving input: block design on region 0.
    stimulus_driving = make_block_stimulus(
        n_blocks=3,
        block_duration=15.0,
        rest_duration=10.0,
        n_inputs=1,
    )

    # Modulator: stick events at bridge-reported outcome times, amplitude =
    # channel value. Wrap in PiecewiseConstantInput — task_dcm_model expects
    # the wrapped object, simulate_task_dcm accepts either.
    stim_mod_dict = make_event_stimulus(
        event_times=bridge_df["outcome_time_s"].tolist(),
        event_amplitudes=bridge_df["epsilon2"].tolist(),
        duration=DURATION,
        dt=DT,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"]
    )

    # --- 5. Forward simulation ----------------------------------------------
    # simulate_task_dcm accepts B_list as list of (N,N) — unstack our (J,N,N).
    B_list = [B_true[j] for j in range(B_true.shape[0])]
    sim = simulate_bilinear_bold(
        A_true=A_true,
        C_true=C_true,
        B_list=B_list,
        stimulus_driving=stimulus_driving,
        stim_mod=stim_mod,
        duration=DURATION,
        dt=DT,
        tr=TR,
        snr=SNR,
    )
    observed_bold = sim["bold"]  # (T, N)
    t_eval = sim["times_fine"]  # (T_fine,)

    # --- 6. Fit the DCM (SVI) -----------------------------------------------
    a_mask = torch.ones(N_REGIONS, N_REGIONS, dtype=torch.float64)
    c_mask = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    # b_masks is a list of per-modulator (N,N) masks.
    b_masks_list = [b_mask[j] for j in range(b_mask.shape[0])]
    model_kwargs = {"b_masks": b_masks_list, "stim_mod": stim_mod}

    guide = create_guide(
        task_dcm_model, guide_type="auto_normal", init_scale=0.005
    )

    model_args = (observed_bold, stimulus_driving, a_mask, c_mask, t_eval, TR)

    svi_result = run_svi(
        model=task_dcm_model,
        guide=guide,
        model_args=model_args,
        num_steps=NUM_SVI_STEPS,
        lr=0.02,
        model_kwargs=model_kwargs,
    )

    # --- 7. Extract posterior (bilinear needs model=partial wrapper) --------
    model_for_pred = partial(task_dcm_model, **model_kwargs)
    posterior = extract_posterior_params(
        guide=guide,
        model_args=model_args,
        model=model_for_pred,
        num_samples=200,
    )

    A_est = posterior["median"]["A"]  # (N, N)
    B_est = posterior["median"]["B"]  # (J, N, N)

    # --- 8. Simple recovery metrics -----------------------------------------
    a_rmse = torch.sqrt(((A_est - A_true) ** 2).mean()).item()

    nonzero_idx = b_mask[0].bool()  # (N, N) for modulator 0
    b_true_vec = B_true[0][nonzero_idx]
    b_est_vec = B_est[0][nonzero_idx]  # j=0 (single modulator)
    b_rmse = torch.sqrt(((b_est_vec - b_true_vec) ** 2).mean()).item()
    b_sign_match = (
        (b_est_vec.sign() == b_true_vec.sign()).float().mean().item()
    )

    print(f"Final SVI loss (last 10 mean): "
          f"{np.mean(svi_result['losses'][-10:]):.2f}")
    print(f"A-RMSE:              {a_rmse:.3f}")
    print(f"B-RMSE (mask=1):     {b_rmse:.3f}")
    print(f"B sign recovery:     {b_sign_match:.2f}")
    print(
        f"B_true (non-zero):   "
        f"{b_true_vec.tolist()}"
    )
    print(
        f"B_est  (non-zero):   "
        f"{[round(float(x), 3) for x in b_est_vec]}"
    )


def simulate_bilinear_bold(
    *,
    A_true: torch.Tensor,
    C_true: torch.Tensor,
    B_list: list[torch.Tensor],
    stimulus_driving: dict[str, torch.Tensor],
    stim_mod: dict[str, torch.Tensor],
    duration: float,
    dt: float,
    tr: float,
    snr: float,
) -> dict[str, torch.Tensor]:
    """Thin wrapper around simulate_task_dcm that enforces bilinear kwargs."""
    return simulate_task_dcm(
        A=A_true,
        C=C_true,
        stimulus=stimulus_driving,
        duration=duration,
        dt=dt,
        TR=tr,
        SNR=snr,
        seed=42,
        B_list=B_list,
        stimulus_mod=stim_mod,
    )


if __name__ == "__main__":
    main()
