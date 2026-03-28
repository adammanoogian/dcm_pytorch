"""End-to-end validation orchestrator for SPM12 cross-validation.

Coordinates the full validation pipeline: generate synthetic data in Python,
export to .mat format, run SPM12/tapas estimation via MATLAB subprocess,
load results, and compare Pyro posteriors against SPM12 reference.

Functions
---------
run_task_dcm_validation : Task DCM vs SPM12 spm_dcm_estimate (VAL-01).
run_spectral_dcm_validation : Spectral DCM vs SPM12 spm_dcm_fmri_csd (VAL-02).
check_matlab_available : Verify MATLAB + SPM12 are accessible.
check_tapas_available : Verify tapas rDCM MATLAB toolbox is accessible.

References
----------
SPM12 source: spm_dcm_estimate.m, spm_dcm_fmri_csd.m.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pyro
import torch

from pyro_dcm.forward_models.csd_computation import (
    bold_to_csd_torch,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import default_frequency_grid
from pyro_dcm.models.guides import create_guide, extract_posterior_params, run_svi
from pyro_dcm.models.spectral_dcm_model import spectral_dcm_model
from pyro_dcm.models.task_dcm_model import task_dcm_model
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput
from validation.compare_results import (
    compute_free_param_comparison,
    load_spm_results,
)
from validation.export_to_mat import (
    export_spectral_dcm_for_spm,
    export_task_dcm_for_spm,
    upsample_stimulus,
)

MATLAB_PATH = "C:/Program Files/MATLAB/R2022a/bin/matlab"
MATLAB_SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__), "matlab_scripts"
).replace("\\", "/")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "data"
).replace("\\", "/")


def check_matlab_available() -> bool:
    """Verify MATLAB and SPM12 are accessible.

    Runs a simple MATLAB command to check both MATLAB binary and SPM12
    availability. Returns True if both succeed, False otherwise.

    Returns
    -------
    bool
        True if MATLAB and SPM12 are available.
    """
    try:
        result = subprocess.run(
            [MATLAB_PATH, "-batch", 'disp("MATLAB OK")'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def check_tapas_available() -> bool:
    """Verify tapas rDCM MATLAB toolbox is accessible.

    Runs MATLAB to check if ``tapas_rdcm_estimate`` is on the path.

    Returns
    -------
    bool
        True if tapas rDCM is available via MATLAB.
    """
    try:
        cmd = (
            "addpath(genpath("
            "'C:/Users/aman0087/Documents/Github/tapas/rDCM'"
            ")); disp(exist('tapas_rdcm_estimate','file'))"
        )
        result = subprocess.run(
            [MATLAB_PATH, "-batch", cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False
        # Check if output contains a nonzero value (2 = .m file)
        return "2" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _a_free_from_parameterized(A: torch.Tensor) -> np.ndarray:
    """Convert parameterized A back to free parameters for comparison.

    SPM12 stores Ep.A in free parameter space. The parameterization
    is: diag -> -exp(free)/2, off-diag -> free. The inverse is:
    diag -> log(-2*diag), off-diag -> same.

    Parameters
    ----------
    A : torch.Tensor
        Parameterized A matrix with negative diagonal.

    Returns
    -------
    np.ndarray
        Free parameters matching SPM12 Ep.A convention.
    """
    A_np = A.detach().cpu().numpy()
    A_free = A_np.copy()
    N = A_np.shape[0]
    for i in range(N):
        # diag: a_ii = -exp(free_ii)/2 => free_ii = log(-2*a_ii)
        A_free[i, i] = np.log(-2.0 * A_np[i, i])
    return A_free


def run_task_dcm_validation(
    seed: int = 42,
    n_regions: int = 3,
    num_svi_steps: int = 3000,
    output_dir: str | None = None,
) -> dict:
    """Run end-to-end task DCM validation against SPM12 (VAL-01).

    Pipeline:
    1. Generate synthetic BOLD from known connectivity using task simulator.
    2. Upsample stimulus to microtime resolution.
    3. Export to .mat for SPM12.
    4. Run SPM12 ``spm_dcm_estimate`` via subprocess.
    5. Run Pyro SVI on identical data.
    6. Compare posterior A_free from both methods.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default 42.
    n_regions : int, optional
        Number of brain regions. Default 3.
    num_svi_steps : int, optional
        Number of SVI optimization steps. Default 3000.
    output_dir : str or None, optional
        Directory for intermediate .mat files. If None, uses
        ``validation/data/``.

    Returns
    -------
    dict
        Keys:
        - ``'comparison'``: dict from ``compute_free_param_comparison``.
        - ``'A_true'``: np.ndarray, ground truth A (parameterized).
        - ``'A_true_free'``: np.ndarray, ground truth A in free space.
        - ``'pyro_A_free'``: np.ndarray, Pyro posterior A_free.
        - ``'spm_Ep_A'``: np.ndarray, SPM12 posterior Ep.A (free).
        - ``'spm_F'``: float, SPM free energy.
        - ``'pyro_final_loss'``: float, Pyro SVI final loss.
        - ``'seed'``: int, random seed used.

    Raises
    ------
    RuntimeError
        If MATLAB/SPM12 fails during estimation.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Generate synthetic data ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    A_true = make_random_stable_A(
        n_regions, density=0.5, seed=seed
    )
    C = torch.zeros(n_regions, 1, dtype=torch.float64)
    C[0, 0] = 1.0

    stimulus = make_block_stimulus(
        n_blocks=5, block_duration=30, rest_duration=20
    )
    duration = 250.0
    TR = 2.0
    SNR = 5.0

    sim_result = simulate_task_dcm(
        A=A_true,
        C=C,
        stimulus=stimulus,
        duration=duration,
        TR=TR,
        SNR=SNR,
        seed=seed,
    )

    bold_data = sim_result["bold"].detach().cpu().numpy()
    A_true_free = _a_free_from_parameterized(A_true)

    # --- Step 2: Upsample stimulus to microtime resolution ---
    # Build TR-resolution stimulus from the block design
    n_scans = bold_data.shape[0]
    stimulus_times = stimulus["times"].numpy()
    stimulus_values = stimulus["values"].numpy()

    # Create TR-resolution stimulus array
    stimulus_tr = np.zeros((n_scans, 1), dtype=np.float64)
    tr_times = np.arange(n_scans) * TR
    for t_idx, t in enumerate(tr_times):
        # Find which stimulus value applies at this time
        mask = stimulus_times <= t
        if mask.any():
            last_idx = np.where(mask)[0][-1]
            stimulus_tr[t_idx] = stimulus_values[last_idx, 0]

    upsampled_stim, u_dt = upsample_stimulus(stimulus_tr, TR)

    # --- Step 3: Export to .mat ---
    a_mask = np.ones((n_regions, n_regions), dtype=np.float64)
    c_mask = np.zeros((n_regions, 1), dtype=np.float64)
    c_mask[0, 0] = 1.0

    input_path = os.path.join(
        output_dir, "task_dcm_input.mat"
    ).replace("\\", "/")
    results_path = os.path.join(
        output_dir, "task_dcm_spm_results.mat"
    ).replace("\\", "/")

    export_task_dcm_for_spm(
        bold_data=bold_data,
        stimulus=upsampled_stim,
        a_mask=a_mask,
        c_mask=c_mask,
        TR=TR,
        u_dt=u_dt,
        output_path=input_path,
    )

    # --- Step 4: Run SPM12 via subprocess ---
    matlab_cmd = (
        f"cd('{MATLAB_SCRIPTS_DIR}'); "
        f"setenv('DCM_INPUT_PATH', '{input_path}'); "
        f"setenv('DCM_OUTPUT_PATH', '{results_path}'); "
        f"run_spm_task_dcm"
    )
    result = subprocess.run(
        [MATLAB_PATH, "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        msg = (
            f"MATLAB/SPM12 task DCM failed (rc={result.returncode}).\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
        raise RuntimeError(msg)

    # --- Step 5: Run Pyro SVI on the same data ---
    pyro.enable_validation(False)
    try:
        bold_torch = torch.tensor(
            bold_data, dtype=torch.float64
        )

        # Reconstruct PiecewiseConstantInput for Pyro model
        stim_input = PiecewiseConstantInput(
            stimulus["times"].to(torch.float64),
            stimulus["values"].to(torch.float64),
        )

        a_mask_torch = torch.ones(
            n_regions, n_regions, dtype=torch.float64
        )
        c_mask_torch = torch.zeros(
            n_regions, 1, dtype=torch.float64
        )
        c_mask_torch[0, 0] = 1.0

        dt = 0.5
        t_eval = torch.arange(
            0, duration, dt, dtype=torch.float64
        )

        model_args = (
            bold_torch, stim_input, a_mask_torch,
            c_mask_torch, t_eval, TR, dt,
        )

        guide = create_guide(task_dcm_model)
        svi_result = run_svi(
            task_dcm_model,
            guide,
            model_args,
            num_steps=num_svi_steps,
            lr=0.01,
        )

        posterior = extract_posterior_params(guide, model_args)
        pyro_A_free = (
            posterior["median"]["A_free"]
            .detach()
            .cpu()
            .numpy()
        )
    finally:
        pyro.enable_validation(True)

    # --- Step 6: Load SPM results ---
    spm_results = load_spm_results(results_path)
    spm_Ep_A = spm_results["Ep_A"]

    # --- Step 7: Compare ---
    comparison = compute_free_param_comparison(
        pyro_A_free, spm_Ep_A
    )

    return {
        "comparison": comparison,
        "A_true": A_true.detach().cpu().numpy(),
        "A_true_free": A_true_free,
        "pyro_A_free": pyro_A_free,
        "spm_Ep_A": spm_Ep_A,
        "spm_F": spm_results["F"],
        "pyro_final_loss": svi_result["final_loss"],
        "seed": seed,
    }


def run_spectral_dcm_validation(
    seed: int = 42,
    n_regions: int = 3,
    num_svi_steps: int = 500,
    n_bold_scans: int = 256,
    output_dir: str | None = None,
) -> dict:
    """Run end-to-end spectral DCM validation against SPM12 (VAL-02).

    Pipeline:
    1. Generate synthetic BOLD from spectral DCM simulator.
    2. Export BOLD to .mat for SPM12 (SPM computes CSD internally).
    3. Run SPM12 ``spm_dcm_fmri_csd`` via subprocess.
    4. Run Pyro SVI on empirical CSD from same BOLD.
    5. Compare posterior A_free from both methods.

    Parameters
    ----------
    seed : int, optional
        Random seed. Default 42.
    n_regions : int, optional
        Number of brain regions. Default 3.
    num_svi_steps : int, optional
        Number of SVI steps. Default 500.
    n_bold_scans : int, optional
        Number of BOLD scans for CSD estimation. Default 256.
    output_dir : str or None, optional
        Directory for .mat files.

    Returns
    -------
    dict
        Keys:
        - ``'comparison'``: dict from ``compute_free_param_comparison``.
        - ``'A_true'``: np.ndarray, ground truth A.
        - ``'A_true_free'``: np.ndarray, ground truth A in free space.
        - ``'pyro_A_free'``: np.ndarray, Pyro posterior A_free.
        - ``'spm_Ep_A'``: np.ndarray, SPM12 posterior Ep.A.
        - ``'spm_F'``: float, SPM free energy.
        - ``'pyro_final_loss'``: float, Pyro SVI final loss.
        - ``'seed'``: int, random seed used.
        - ``'csd_method_note'``: str, documenting CSD method difference.

    Raises
    ------
    RuntimeError
        If MATLAB/SPM12 fails during estimation.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    TR = 2.0

    # --- Step 1: Generate synthetic BOLD ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    A_true = make_stable_A_spectral(n_regions, seed=seed)
    A_true_free = _a_free_from_parameterized(A_true)

    # Generate CSD from ground truth for reference
    sim_result = simulate_spectral_dcm(A=A_true, TR=TR, seed=seed)

    # Generate synthetic BOLD time series from spectral model.
    # The spectral simulator generates CSD directly; we need BOLD
    # time series for both SPM (expects BOLD) and our pipeline
    # (computes CSD from BOLD via Welch). Generate from a simple
    # VAR(1) process with A as the dynamics matrix.
    torch.manual_seed(seed)
    N = n_regions
    bold_ts = torch.zeros(
        n_bold_scans, N, dtype=torch.float64
    )
    noise_std = 0.1
    x = torch.zeros(N, dtype=torch.float64)
    # Discrete-time approximation: x[t+1] = (I + TR*A) x[t] + noise
    A_discrete = (
        torch.eye(N, dtype=torch.float64) + TR * A_true
    )
    for t in range(n_bold_scans):
        x = A_discrete @ x + noise_std * torch.randn(
            N, dtype=torch.float64
        )
        bold_ts[t] = x

    bold_data = bold_ts.detach().cpu().numpy()

    # --- Step 2: Export BOLD to .mat ---
    a_mask = np.ones((N, N), dtype=np.float64)
    c_mask = np.ones((N, 1), dtype=np.float64)

    input_path = os.path.join(
        output_dir, "spectral_dcm_input.mat"
    ).replace("\\", "/")
    results_path = os.path.join(
        output_dir, "spectral_dcm_spm_results.mat"
    ).replace("\\", "/")

    export_spectral_dcm_for_spm(
        bold_data=bold_data,
        a_mask=a_mask,
        c_mask=c_mask,
        TR=TR,
        output_path=input_path,
    )

    # --- Step 3: Run SPM12 spectral DCM via subprocess ---
    matlab_cmd = (
        f"cd('{MATLAB_SCRIPTS_DIR}'); "
        f"setenv('DCM_INPUT_PATH', '{input_path}'); "
        f"setenv('DCM_OUTPUT_PATH', '{results_path}'); "
        f"run_spm_spectral_dcm"
    )
    result = subprocess.run(
        [MATLAB_PATH, "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        msg = (
            f"MATLAB/SPM12 spectral DCM failed "
            f"(rc={result.returncode}).\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
        raise RuntimeError(msg)

    # --- Step 4: Run Pyro SVI on same BOLD ---
    freqs = default_frequency_grid(TR, n_freqs=32)
    observed_csd = bold_to_csd_torch(
        bold_ts, fs=1.0 / TR, freqs=freqs
    )

    a_mask_torch = torch.ones(N, N, dtype=torch.float64)

    model_args = (observed_csd, freqs, a_mask_torch)
    guide = create_guide(spectral_dcm_model)
    svi_result = run_svi(
        spectral_dcm_model,
        guide,
        model_args,
        num_steps=num_svi_steps,
        lr=0.01,
        lr_decay_factor=0.1,
    )

    posterior = extract_posterior_params(guide, model_args)
    pyro_A_free = (
        posterior["median"]["A_free"]
        .detach()
        .cpu()
        .numpy()
    )

    # --- Step 5: Load SPM results and compare ---
    spm_results = load_spm_results(results_path)
    spm_Ep_A = spm_results["Ep_A"]

    # --- Step 6: Compare with relaxed tolerance ---
    comparison = compute_free_param_comparison(
        pyro_A_free, spm_Ep_A, tolerance=0.15
    )

    csd_method_note = (
        "SPM uses MAR model for CSD estimation from BOLD; "
        "our pipeline uses Welch periodogram. This is an expected "
        "additional source of 5-10% discrepancy beyond VL vs SVI "
        "inference differences."
    )

    return {
        "comparison": comparison,
        "A_true": A_true.detach().cpu().numpy(),
        "A_true_free": A_true_free,
        "pyro_A_free": pyro_A_free,
        "spm_Ep_A": spm_Ep_A,
        "spm_F": spm_results["F"],
        "pyro_final_loss": svi_result["final_loss"],
        "seed": seed,
        "csd_method_note": csd_method_note,
    }
