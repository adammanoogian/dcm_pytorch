"""Variational-Laplace vs SPM12 cross-validation orchestrator (VLSPM-01/02/03).

VL-path twin of :func:`validation.run_validation.run_spectral_dcm_validation`.
Where the SVI orchestrator lets SPM recompute the CSD from BOLD via its MAR
model, this fits the Phase 28 Variational-Laplace engine on a prior-matched
reciprocal-edge N=2 spectral DCM problem, injects the IDENTICAL Python-computed
``(F, N, N)`` complex CSD into SPM via the Plan 32-01 same-CSD bridge
(:func:`validation.export_to_mat.export_spectral_dcm_csd_for_spm` +
``run_spm_spectral_dcm_csd_injected.m``), runs ``spm_dcm_fmri_csd``
(``spm_nlsi_GN`` under the hood), and compares the two engines on three axes:

1. **Ep in free-parameter space (S1/S2):** ``result.theta_post["A_free"]`` vs
   SPM ``Ep.A`` (BOTH free params, never the parameterized ``A``) within ~10%
   via :func:`validation.compare_results.compute_free_param_comparison`. The VL
   fit uses the SPM-matched priors ``hyperprior_mean=8.0``,
   ``hyperprior_precision=128.0``, ``prior_mean_a_offset=a_mask/128`` (S2).
2. **Matched free energy (user decision, strict 5% HARD gate):** VL
   ``free_energy[-1]`` vs SPM ``DCM.F`` on the IDENTICAL injected CSD via
   :func:`validation.compare_results.compare_free_energies` (``rel_tolerance=
   0.05``). Like-for-like ONLY because both fit the same CSD (Plan 32-01).
3. **Cross-model ranking (S3-safe):** RELATIVE delta-F across >=3 reduced-model
   ``a_mask`` scenarios via
   :func:`validation.compare_results.compare_model_ranking` (agreement_rate
   >= 0.80). NEVER absolute F across models, NEVER element-wise ``Cp``.

The matched ground truth is reciprocal with ASYMMETRIC strengths
(``A[0,1]=0.15``, ``A[1,0]=0.10``): a feed-forward / lone-edge ``A`` is
CSD-indistinguishable from the empty graph under spectral DCM (Phase 31
identifiability finding, 31-01-D1), so reciprocal edges are mandatory, and the
asymmetry gives the S4 layout check teeth.

The MATLAB binary is resolved from :data:`config.MATLAB_PATH` (env-overridable)
and the SPM12 location is passed to the MATLAB child via the ``SPM12_PATH``
environment variable, so the SAME orchestrator runs on M3 where MATLAB + SPM12
are licensed (research Open Q5).

References
----------
SPM12 source: spm_dcm_fmri_csd.m, spm_nlsi_GN.m.
.planning/phases/32-spm12-cross-validation/32-03-PLAN.md
    VLSPM-01/02/03 objective and gating contract (S1-S4).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import torch

from config import MATLAB_PATH
from pyro_dcm.inference.variational_laplace import run_variational_laplace
from pyro_dcm.simulators.spectral_simulator import simulate_spectral_dcm
from validation.compare_results import (
    compare_free_energies,
    compare_model_ranking,
    compute_free_param_comparison,
    load_spm_results,
)
from validation.export_to_mat import export_spectral_dcm_csd_for_spm
from validation.run_validation import (
    DEFAULT_OUTPUT_DIR,
    MATLAB_SCRIPTS_DIR,
    check_matlab_available,  # noqa: F401 -- re-exported for the SPM-gated test
)

# SPM-matched priors (S2): SPM12 spm_nlsi_GN uses M.hE=8, 1/M.hC=128, and an
# A-prior mean offset of a_mask/128. The VL engine reproduces these exactly so
# the matched-F comparison is like-for-like.
_HYPERPRIOR_MEAN = 8.0
_HYPERPRIOR_PRECISION = 128.0
_PRIOR_MEAN_A_DIVISOR = 128.0

# Reciprocal-ASYMMETRIC ground truth (S4 / Phase 31 identifiability finding).
_EDGE_0_TO_1 = 0.15
_EDGE_1_TO_0 = 0.10
_SELF_CONNECTION = -0.5

_TR = 2.0
_N_FREQS = 32


def _build_reciprocal_asymmetric_A(n_regions: int) -> torch.Tensor:
    """Build a reciprocal-asymmetric ground-truth connectivity matrix.

    Constructs an ``(N, N)`` float64 ``A`` with self-connections on the
    diagonal and a single reciprocal pair ``(0, 1)`` / ``(1, 0)`` set to
    ASYMMETRIC strengths (0.15 vs 0.10). A feed-forward / lone-edge ``A`` is
    CSD-indistinguishable from the empty graph under spectral DCM (Phase 31
    finding, 31-01-D1), so the reciprocal pair is mandatory; the asymmetry
    gives the S4 layout check teeth (a transpose would equalize ``A[0,1]`` and
    ``A[1,0]``).

    Parameters
    ----------
    n_regions : int
        Number of regions ``N`` (must be >= 2 for the reciprocal pair).

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.

    Raises
    ------
    ValueError
        If ``n_regions < 2`` (no reciprocal pair possible) or the resulting
        ``A`` is not stable (max real eigenvalue not ``< 0``). Messages name
        the expected-vs-actual value.
    """
    if n_regions < 2:
        msg = (
            f"reciprocal-asymmetric ground truth needs n_regions >= 2 "
            f"(expected >= 2, got {n_regions})"
        )
        raise ValueError(msg)

    a_true = torch.zeros(n_regions, n_regions, dtype=torch.float64)
    a_true.fill_diagonal_(_SELF_CONNECTION)
    a_true[0, 1] = _EDGE_0_TO_1
    a_true[1, 0] = _EDGE_1_TO_0

    max_real_eig = float(torch.linalg.eigvals(a_true).real.max())
    if not max_real_eig < 0.0:
        msg = (
            f"reciprocal-asymmetric ground-truth A is not stable: max real "
            f"eigenvalue {max_real_eig:.6f} (expected < 0); reduce edge "
            f"strengths or make the self-connection more negative"
        )
        raise ValueError(msg)
    return a_true


def _run_spm_on_csd(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    output_dir: str,
    tag: str,
) -> dict:
    """Inject a CSD into SPM and fit ``spm_dcm_fmri_csd`` on it.

    Exports the EXACT ``observed_csd`` the VL engine fits via the Plan 32-01
    same-CSD bridge, then runs ``run_spm_spectral_dcm_csd_injected.m`` through a
    MATLAB ``-batch`` subprocess (mirrors
    ``validation.run_validation.run_spectral_dcm_validation`` steps 3). The
    SPM12 location is passed to the MATLAB child via the ``SPM12_PATH``
    environment variable so the SAME code runs on M3.

    Parameters
    ----------
    observed_csd : torch.Tensor
        Cross-spectral density, shape ``(F, N, N)``, complex128. The SAME array
        the VL engine fits.
    freqs : torch.Tensor
        Frequency grid in Hz, shape ``(F,)``, float64.
    a_mask : torch.Tensor
        Binary connectivity mask, shape ``(N, N)``, float64.
    output_dir : str
        Directory for the intermediate input/result ``.mat`` files.
    tag : str
        Filename suffix distinguishing concurrent scenario fits.

    Returns
    -------
    dict
        Output of :func:`validation.compare_results.load_spm_results` for this
        fit (``Ep_A``, ``F``, and optional spectral fields).

    Raises
    ------
    RuntimeError
        If the MATLAB / SPM12 subprocess returns non-zero.
    """
    n = a_mask.shape[0]
    input_path = os.path.join(
        output_dir, f"spectral_dcm_csd_input_{tag}.mat"
    ).replace("\\", "/")
    results_path = os.path.join(
        output_dir, f"spectral_dcm_csd_spm_results_{tag}.mat"
    ).replace("\\", "/")

    export_spectral_dcm_csd_for_spm(
        observed_csd=observed_csd.cpu().numpy(),
        freqs=freqs.cpu().numpy(),
        a_mask=a_mask.cpu().numpy(),
        c_mask=np.ones((n, 1), dtype=np.float64),
        TR=_TR,
        output_path=input_path,
    )

    matlab_cmd = (
        f"cd('{MATLAB_SCRIPTS_DIR}'); "
        f"setenv('DCM_INPUT_PATH', '{input_path}'); "
        f"setenv('DCM_OUTPUT_PATH', '{results_path}'); "
        f"run_spm_spectral_dcm_csd_injected"
    )
    # Pass SPM12_PATH through to the MATLAB child; the .m file falls back to its
    # local default when the variable is absent (so laptop + M3 share one code).
    child_env = dict(os.environ)

    result = subprocess.run(
        [str(MATLAB_PATH), "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    if result.returncode != 0:
        msg = (
            f"MATLAB/SPM12 injected-CSD spectral DCM failed "
            f"(tag={tag}, rc={result.returncode}).\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
        raise RuntimeError(msg)

    return load_spm_results(results_path)


def _fit_vl_free_energy(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    n_regions: int,
    max_iter: int,
) -> float:
    """Fit the VL engine under a given ``a_mask`` and return ``free_energy[-1]``.

    Used by the cross-model ranking sweep: each reduced-model scenario re-fits
    the SAME ``observed_csd`` under a different structural ``a_mask`` and yields
    the final VL free energy (higher = better, same convention as SPM ``DCM.F``)
    for the RELATIVE ranking (S3-safe).

    Parameters
    ----------
    observed_csd : torch.Tensor
        Cross-spectral density, shape ``(F, N, N)``, complex128.
    freqs : torch.Tensor
        Frequency grid in Hz, shape ``(F,)``, float64.
    a_mask : torch.Tensor
        Binary connectivity mask for this scenario, shape ``(N, N)``, float64.
    n_regions : int
        Number of regions ``N``.
    max_iter : int
        VL Gauss-Newton iteration cap.

    Returns
    -------
    float
        Final VL free energy ``free_energy[-1]`` for this masked model.
    """
    result = run_variational_laplace(
        observed_csd=observed_csd,
        freqs=freqs,
        a_mask=a_mask,
        N=n_regions,
        max_iter=max_iter,
        hyperprior_mean=_HYPERPRIOR_MEAN,
        hyperprior_precision=_HYPERPRIOR_PRECISION,
        prior_mean_a_offset=a_mask / _PRIOR_MEAN_A_DIVISOR,
    )
    return float(result.free_energy[-1])


def run_vl_spectral_dcm_validation(
    seed: int = 42,
    n_regions: int = 2,
    max_iter: int = 64,
    output_dir: str | None = None,
) -> dict:
    """Cross-validate the VL engine against SPM12 on a matched spectral problem.

    Builds a reciprocal-asymmetric N=2 spectral DCM ground truth, simulates its
    CSD, fits the Phase 28 Variational-Laplace engine with SPM-matched priors,
    injects the IDENTICAL CSD into SPM12 (Plan 32-01 bridge), runs
    ``spm_dcm_fmri_csd``, and compares the two engines in free-parameter space
    (Ep ~10%, S1/S2), matched free energy (5% HARD gate on the identical CSD,
    user decision), and relative cross-model ranking (>= 0.80 over >=3 masks,
    S3-safe). No element-wise ``Cp`` comparison and no absolute-F-across-models
    anywhere.

    Parameters
    ----------
    seed : int, optional
        Random seed for the spectral simulation. Default 42.
    n_regions : int, optional
        Number of brain regions. Default 2 (the identifiable reciprocal pair).
    max_iter : int, optional
        VL Gauss-Newton iteration cap. Default 64.
    output_dir : str or None, optional
        Directory for intermediate ``.mat`` files. If None, uses
        ``validation/data/``.

    Returns
    -------
    dict
        Flat dictionary with keys:

        - ``'ep_comparison'`` : dict -- free-parameter-space Ep comparison
          (``compute_free_param_comparison`` at 10%, S1/S2).
        - ``'matched_f_comparison'`` : dict -- matched-F comparison
          (``compare_free_energies`` at 5%, the headline gate).
        - ``'ranking'`` : dict -- relative cross-model ranking
          (``compare_model_ranking``, S3-safe).
        - ``'vl_A_free'`` : np.ndarray -- VL posterior ``A_free`` (free space).
        - ``'spm_Ep_A'`` : np.ndarray -- SPM ``Ep.A`` (free space).
        - ``'vl_F'`` : float -- VL ``free_energy[-1]`` on the matched problem.
        - ``'spm_F'`` : float -- SPM ``DCM.F`` on the IDENTICAL injected CSD.
        - ``'ep_asymmetry'`` : tuple(float, float) -- ``(Ep.A[0,1], Ep.A[1,0])``
          S4 readout (must differ for the asymmetric ground truth).
        - ``'A_true'`` : np.ndarray -- the reciprocal-asymmetric ground truth.
        - ``'seed'`` : int -- the seed used.
        - ``'n_regions'`` : int -- the number of regions used.

    Raises
    ------
    ValueError
        If the ground truth cannot be built (n_regions < 2 or unstable A).
    RuntimeError
        If the MATLAB / SPM12 subprocess fails.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    n = n_regions

    # --- Step 1: reciprocal-asymmetric matched problem (S4 / Phase 31) ------
    torch.manual_seed(seed)
    np.random.seed(seed)
    a_true = _build_reciprocal_asymmetric_A(n)

    # --- Step 2: simulate the CSD both engines fit -------------------------
    sim = simulate_spectral_dcm(a_true, TR=_TR, n_freqs=_N_FREQS, seed=seed)
    observed_csd = sim["csd"].to(torch.complex128)
    freqs = sim["freqs"].double()
    a_mask_full = torch.ones(n, n, dtype=torch.float64)

    # --- Step 3: VL fit with SPM-matched priors (S2) -----------------------
    vl_result = run_variational_laplace(
        observed_csd=observed_csd,
        freqs=freqs,
        a_mask=a_mask_full,
        N=n,
        max_iter=max_iter,
        hyperprior_mean=_HYPERPRIOR_MEAN,
        hyperprior_precision=_HYPERPRIOR_PRECISION,
        prior_mean_a_offset=a_mask_full / _PRIOR_MEAN_A_DIVISOR,
    )
    # S1: compare FREE-parameter space ("A_free"), NOT the parameterized "A".
    vl_a_free = vl_result.theta_post["A_free"].detach().cpu().numpy()
    vl_f = float(vl_result.free_energy[-1])

    # --- Step 4: SPM side via the SAME-CSD injection (Plan 32-01) -----------
    spm = _run_spm_on_csd(
        observed_csd, freqs, a_mask_full, output_dir, tag="full",
    )
    spm_ep_a = spm["Ep_A"]
    spm_f = float(spm["F"])

    # --- Step 5: compare Ep (free space, 10%) + matched F (5% HARD gate) ----
    ep_comparison = compute_free_param_comparison(
        vl_a_free, spm_ep_a, tolerance=0.10,
    )
    matched_f_comparison = compare_free_energies(
        vl_f, spm_f, rel_tolerance=0.05,
    )
    # S4 readout: the asymmetric reciprocal truth must give asymmetric Ep.A.
    ep_asymmetry = (float(spm_ep_a[0, 1]), float(spm_ep_a[1, 0]))

    # --- Step 6: cross-model ranking over >=3 masks (relative, S3-safe) -----
    # Full reciprocal (correct), single-direction ([1,0] only), diagonal-only.
    # Each scenario re-fits the SAME observed_csd under its a_mask on BOTH
    # engines; we compare only the RELATIVE ordering of F, never absolute F
    # across masks and never element-wise Cp. The "pyro_elbo" key is the
    # literal compare_model_ranking field name; VL free_energy[-1] substitutes
    # (higher = better, same convention as SPM DCM.F).
    eye = torch.eye(n, dtype=torch.float64)
    mask_single = eye.clone()
    mask_single[1, 0] = 1.0
    scenario_masks = {
        "full_reciprocal": a_mask_full,
        "single_direction": mask_single,
        "diagonal_only": eye,
    }
    scenarios: list[dict] = []
    for name, mask in scenario_masks.items():
        vl_f_k = _fit_vl_free_energy(
            observed_csd, freqs, mask, n, max_iter,
        )
        spm_k = _run_spm_on_csd(
            observed_csd, freqs, mask, output_dir, tag=name,
        )
        scenarios.append(
            {"spm_F": float(spm_k["F"]), "pyro_elbo": vl_f_k}
        )
    ranking = compare_model_ranking(scenarios)

    return {
        "ep_comparison": ep_comparison,
        "matched_f_comparison": matched_f_comparison,
        "ranking": ranking,
        "vl_A_free": vl_a_free,
        "spm_Ep_A": spm_ep_a,
        "vl_F": vl_f,
        "spm_F": spm_f,
        "ep_asymmetry": ep_asymmetry,
        "A_true": a_true.cpu().numpy(),
        "seed": seed,
        "n_regions": n_regions,
    }
