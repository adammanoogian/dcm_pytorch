"""End-to-end spectral DCM pipeline demo with MNE-Python IO.

Demonstrates the complete spectral DCM workflow for projects that use
MNE-Python data as input:

    MNE Epochs -> epochs_to_csd -> spectral_dcm_model -> SVI -> posterior A

Runs in ~1-2 min on CPU with num_steps=300. Produces:
  - Printed ground-truth and estimated A matrix (3 x 3)
  - A-RMSE recovery metric comparing posterior A to known ground truth
  - Final ELBO loss (mean of last 10 steps)

MNE-Python is required: ``pip install pyro-dcm[mne]``
"""

from __future__ import annotations

import sys

try:
    import mne
except ImportError:
    print("MNE-Python required. Install with: pip install pyro-dcm[mne]")
    sys.exit(1)

import numpy as np
import torch

from pyro_dcm import (
    create_guide,
    extract_posterior_params,
    make_stable_A_spectral,
    parameterize_A,
    run_svi,
    simulate_spectral_dcm,
    spectral_dcm_model,
)
from pyro_dcm.io import epochs_to_csd


def main() -> None:
    """Run the spectral DCM pipeline demo end-to-end.

    Sections
    --------
    1. Ground-truth A matrix
    2. Simulate CSD from ground truth
    3. Create synthetic MNE Epochs
    4. MNE Epochs -> CSD tensor (the IO bridge)
    5. Set up model and fit via SVI
    6. Extract posterior and compute recovery metrics
    """
    torch.manual_seed(42)

    # --- 1. Ground-truth A matrix -------------------------------------------
    # make_stable_A_spectral returns a float64 (N, N) connectivity matrix
    # with guaranteed negative eigenvalues (stable linear system).
    N = 3
    A_true = make_stable_A_spectral(
        n_regions=N, connection_strength=0.1, seed=42
    )
    print("Ground-truth A matrix (A_true):")
    print(A_true.numpy().round(4))

    # --- 2. Simulate CSD from ground truth ----------------------------------
    # simulate_spectral_dcm returns a dict with 'csd' (F, N, N) complex128
    # and 'freqs' (F,) float64, computed via the full spectral DCM forward
    # model (transfer function + neuronal/observation noise).
    sim = simulate_spectral_dcm(A_true, n_freqs=32, seed=42)
    observed_csd: torch.Tensor = sim["csd"]   # (F, N, N), complex128
    freqs: torch.Tensor = sim["freqs"]        # (F,), float64

    print(f"\nSimulated CSD shape:  {tuple(observed_csd.shape)}  "
          f"(freqs x regions x regions)")
    print(f"Frequency range:       {freqs[0].item():.3f} -- "
          f"{freqs[-1].item():.3f} Hz")

    # --- 3. Create synthetic MNE Epochs -------------------------------------
    # Build a minimal EpochsArray to demonstrate the MNE IO path.
    # Amplitudes are scaled to uV (1e-6 V) -- EEG convention in MNE.
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_epochs_mne = 20
    n_times = int(sfreq * 2.0)

    info = mne.create_info(
        ch_names=[f"ROI{i}" for i in range(N)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    data = rng.standard_normal((n_epochs_mne, N, n_times)) * 1e-6
    epochs = mne.EpochsArray(data, info, verbose=False)

    print(f"\nMNE Epochs created:   {n_epochs_mne} epochs  x  {N} channels  "
          f"x  {n_times} samples  (sfreq={sfreq} Hz)")

    # --- 4. MNE Epochs -> CSD tensor (the IO bridge) -----------------------
    # epochs_to_csd wraps mne.time_frequency.csd_multitaper and returns a
    # dict with 'csd' (F, N, N) complex and 'freqs' (F,) float64 -- the
    # same shapes expected by spectral_dcm_model.
    mne_csd_result = epochs_to_csd(epochs, fmin=1.0, fmax=50.0, n_freqs=32)
    mne_csd: torch.Tensor = mne_csd_result["csd"]    # (F, N, N), complex128
    mne_freqs: torch.Tensor = mne_csd_result["freqs"]  # (F,), float64

    print("\nepochs_to_csd output:")
    print(f"  csd shape:   {tuple(mne_csd.shape)}")
    print(f"  freqs shape: {tuple(mne_freqs.shape)}")
    print(f"  freq range:  {mne_freqs[0].item():.3f} -- "
          f"{mne_freqs[-1].item():.3f} Hz")
    print(f"  n_epochs:    {mne_csd_result['n_epochs']}")

    # NOTE: For fitting we use the CSD from simulate_spectral_dcm (Section 2)
    # as observed_csd, not mne_csd. The MNE Epochs above contain pure noise
    # (no DCM structure), so fitting against mne_csd would not yield
    # meaningful A-RMSE recovery metrics. The epochs_to_csd call above
    # demonstrates the IO bridge: in a real workflow, replace observed_csd
    # below with mne_csd and update freqs accordingly.

    # --- 5. Set up model and fit via SVI ------------------------------------
    # Full A-mask: allow all N x N connections (including self-connections --
    # parameterize_A clamps the diagonal to be negative).
    a_mask = torch.ones(N, N, dtype=torch.float64)
    model_args = (observed_csd, freqs, a_mask)

    guide = create_guide(spectral_dcm_model, init_scale=0.01)

    print("\nRunning SVI (300 steps, lr=0.01) ...")
    svi_result = run_svi(
        spectral_dcm_model,
        guide,
        model_args,
        num_steps=300,
        lr=0.01,
    )

    final_loss_mean = float(np.mean(svi_result["losses"][-10:]))
    print(f"Final ELBO loss (last 10 mean): {final_loss_mean:.2f}")

    # --- 6. Extract posterior and compute recovery metrics -----------------
    # extract_posterior_params samples from the trained guide and returns a
    # dict where each Pyro site has keys 'mean', 'std', 'samples'.
    # The A_free site holds the raw free parameters; parameterize_A applies
    # the same transform used inside spectral_dcm_model to get A.
    posterior = extract_posterior_params(guide, model_args)
    A_free_mean: torch.Tensor = posterior["A_free"]["mean"]  # (N, N)
    A_est: torch.Tensor = parameterize_A(A_free_mean)        # (N, N)

    a_rmse = torch.sqrt(((A_est - A_true) ** 2).mean()).item()

    print("\nGround-truth A (A_true):")
    print(A_true.numpy().round(4))
    print("\nEstimated A (A_est, posterior mean):")
    print(A_est.detach().numpy().round(4))
    print(f"\nA-RMSE: {a_rmse:.4f}")


if __name__ == "__main__":
    main()
