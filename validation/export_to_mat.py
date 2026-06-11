"""Export DCM data to SPM12-compatible .mat format.

Provides export functions for all three DCM variants (task, spectral, rDCM)
that create .mat files loadable by SPM12's ``spm_dcm_estimate``,
``spm_dcm_fmri_csd``, and tapas ``tapas_rdcm_estimate``.

Key conventions:
- All scalars are wrapped as ``np.array([[value]])`` (2D) per MATLAB.
- String fields use ``np.array([['text']], dtype=object)``.
- Empty 3D fields use ``np.zeros((N, N, 0))`` for unused modulatory/nonlinear.
- Stimulus must be at microtime resolution (TR/16) for SPM12.

References
----------
SPM12 source: spm_dcm_estimate.m, spm_dcm_fmri_csd.m, spm_dcm_specify_ui.m.
"""

from __future__ import annotations

import numpy as np
import scipy.io


def upsample_stimulus(
    stimulus_tr: np.ndarray,
    TR: float,
    microtime_factor: int = 16,
) -> tuple[np.ndarray, float]:
    """Upsample stimulus from TR resolution to microtime resolution.

    SPM12 expects inputs at microtime resolution (TR / microtime_factor).
    This function upsamples using nearest-neighbor interpolation and pads
    the beginning with 32 zero rows (SPM discards first 32 microtime
    samples internally per ``spm_dcm_specify_ui.m``).

    Parameters
    ----------
    stimulus_tr : np.ndarray
        Stimulus at TR resolution, shape ``(T, M)`` where T is the
        number of scans and M is the number of inputs.
    TR : float
        Repetition time in seconds.
    microtime_factor : int, optional
        Upsampling factor. Default 16 (SPM12 default).

    Returns
    -------
    tuple of (np.ndarray, float)
        ``(upsampled_stimulus, u_dt)`` where upsampled_stimulus has
        shape ``(T * microtime_factor + 32, M)`` and u_dt = TR /
        microtime_factor.
    """
    T, M = stimulus_tr.shape
    u_dt = TR / microtime_factor

    # Nearest-neighbor upsampling: repeat each TR sample microtime_factor times
    upsampled = np.repeat(stimulus_tr, microtime_factor, axis=0)

    # Pad beginning with 32 zero rows (SPM convention)
    padding = np.zeros((32, M), dtype=np.float64)
    upsampled = np.concatenate([padding, upsampled], axis=0)

    return upsampled.astype(np.float64), u_dt


def export_task_dcm_for_spm(
    bold_data: np.ndarray,
    stimulus: np.ndarray,
    a_mask: np.ndarray,
    c_mask: np.ndarray,
    TR: float,
    u_dt: float,
    output_path: str,
) -> None:
    """Export synthetic data as SPM12-compatible task DCM .mat file.

    Builds the complete DCM struct matching ``spm_dcm_estimate``
    requirements. All fields follow the format verified against
    SPM12 source code.

    Parameters
    ----------
    bold_data : np.ndarray
        BOLD time series, shape ``(v, N)`` where v is the number
        of scans and N is the number of regions. Must be float64.
    stimulus : np.ndarray
        Stimulus at microtime resolution, shape ``(T_micro, M)``
        where T_micro is the number of microtime bins and M is the
        number of inputs. Use ``upsample_stimulus`` to convert from
        TR resolution.
    a_mask : np.ndarray
        Binary connectivity mask, shape ``(N, N)``. 1 where
        connections are allowed.
    c_mask : np.ndarray
        Binary driving input mask, shape ``(N, M)``. 1 where
        inputs drive regions.
    TR : float
        Repetition time in seconds.
    u_dt : float
        Stimulus sampling interval in seconds (typically TR/16).
    output_path : str
        Path for the output .mat file.

    Notes
    -----
    SPM12 source: ``spm_dcm_estimate.m`` header for required fields.
    Scalars are wrapped as ``np.array([[value]])`` (2D) per MATLAB.

    See Also
    --------
    upsample_stimulus : Convert TR-resolution stimulus to microtime.
    """
    v = bold_data.shape[0]  # number of scans
    N = bold_data.shape[1]  # number of regions
    M = c_mask.shape[1]     # number of inputs

    DCM = {
        # Connectivity masks
        "a": a_mask.astype(np.float64),
        "b": np.zeros((N, N, 0), dtype=np.float64),
        "c": c_mask.astype(np.float64),
        "d": np.zeros((N, N, 0), dtype=np.float64),
        # Response data
        "Y": {
            "y": bold_data.astype(np.float64),
            "dt": np.array([[TR]]),
            "X0": np.ones((v, 1), dtype=np.float64),
            "name": np.array(
                [[f"R{i + 1}" for i in range(N)]], dtype=object
            ),
        },
        # Input data
        "U": {
            "u": stimulus.astype(np.float64),
            "dt": np.array([[u_dt]]),
            "name": np.array(
                [[f"stim{i + 1}" for i in range(M)]], dtype=object
            ),
        },
        # Dimensions
        "n": np.array([[N]]),
        "v": np.array([[v]]),
        # Timing
        "TE": np.array([[0.04]]),
        "delays": np.ones((1, N)) * TR / 2,
        # Options
        "options": {
            "nonlinear": np.array([[0]]),
            "two_state": np.array([[0]]),
            "stochastic": np.array([[0]]),
            "centre": np.array([[0]]),
            "induced": np.array([[0]]),
            "nograph": np.array([[1]]),
            "maxit": np.array([[128]]),
        },
    }
    scipy.io.savemat(output_path, {"DCM": DCM})


def export_spectral_dcm_for_spm(
    bold_data: np.ndarray,
    a_mask: np.ndarray,
    c_mask: np.ndarray,
    TR: float,
    output_path: str,
) -> None:
    """Export synthetic BOLD as SPM12-compatible spectral DCM .mat file.

    Like task DCM but with ``options.induced = 1`` and
    ``options.analysis = 'CSD'`` to trigger CSD analysis mode.
    SPM12 computes CSD from BOLD internally via MAR model.

    Parameters
    ----------
    bold_data : np.ndarray
        BOLD time series, shape ``(v, N)``. Must be float64.
    a_mask : np.ndarray
        Binary connectivity mask, shape ``(N, N)``.
    c_mask : np.ndarray
        Binary driving input mask, shape ``(N, M)``.
    TR : float
        Repetition time in seconds.
    output_path : str
        Path for the output .mat file.

    Notes
    -----
    For spectral DCM, stimulus is minimal: constant input
    ``U.u = np.ones((T_micro, 1))`` with ``U.dt = TR/16``.
    SPM12 source: ``spm_dcm_fmri_csd.m``.
    """
    v = bold_data.shape[0]
    N = bold_data.shape[1]
    M = c_mask.shape[1]

    # Microtime resolution for U
    microtime_factor = 16
    u_dt = TR / microtime_factor
    T_micro = v * microtime_factor + 32

    # Minimal constant input for spectral DCM
    stimulus = np.ones((T_micro, M), dtype=np.float64)

    DCM = {
        "a": a_mask.astype(np.float64),
        "b": np.zeros((N, N, 0), dtype=np.float64),
        "c": c_mask.astype(np.float64),
        "d": np.zeros((N, N, 0), dtype=np.float64),
        "Y": {
            "y": bold_data.astype(np.float64),
            "dt": np.array([[TR]]),
            "X0": np.ones((v, 1), dtype=np.float64),
            "name": np.array(
                [[f"R{i + 1}" for i in range(N)]], dtype=object
            ),
        },
        "U": {
            "u": stimulus,
            "dt": np.array([[u_dt]]),
            "name": np.array(
                [[f"stim{i + 1}" for i in range(M)]], dtype=object
            ),
        },
        "n": np.array([[N]]),
        "v": np.array([[v]]),
        "TE": np.array([[0.04]]),
        "delays": np.ones((1, N)) * TR / 2,
        "options": {
            "nonlinear": np.array([[0]]),
            "two_state": np.array([[0]]),
            "stochastic": np.array([[0]]),
            "centre": np.array([[0]]),
            "induced": np.array([[1]]),
            "nograph": np.array([[1]]),
            "maxit": np.array([[128]]),
            "analysis": np.array([["CSD"]], dtype=object),
            "order": np.array([[8]]),
        },
    }
    scipy.io.savemat(output_path, {"DCM": DCM})


def export_spectral_dcm_csd_for_spm(
    observed_csd: np.ndarray,
    freqs: np.ndarray,
    a_mask: np.ndarray,
    c_mask: np.ndarray,
    TR: float,
    output_path: str,
    bold_data: np.ndarray | None = None,
) -> None:
    """Inject a precomputed complex CSD into an SPM12 spectral DCM .mat file.

    Unlike ``export_spectral_dcm_for_spm`` (which ships BOLD and lets SPM12
    recompute the CSD internally via its MAR model), this writes the EXACT
    ``(F, N, N)`` complex cross-spectral density the Pyro-DCM VL engine fits
    into ``DCM.Y.csd`` and the matching frequency grid into ``DCM.Y.Hz``. The
    paired MATLAB script ``run_spm_spectral_dcm_csd_injected.m`` then estimates
    on THAT identical data, so a strict matched-free-energy comparison between
    the two engines is meaningful (both evaluate F on the same CSD).

    Layout / pitfall S4 (column-major vs row-major)
    -----------------------------------------------
    ``observed_csd[w, i, j]`` is the C-order ``(F, N, N)`` CSD with ``j``
    varying fastest, then ``i``, then ``w`` (the contract locked by
    ``tests/test_csd_corder_roundtrip.py``). ``scipy.io.savemat`` writes a
    3-D complex array and MATLAB reads it back as ``(F, N, N)``, so
    ``DCM.Y.csd(w, i, j)`` (1-based) equals Python ``observed_csd[w-1, i-1,
    j-1]``. NO transpose is applied here; the injection MATLAB script must
    index accordingly. A silent transpose would corrupt the asymmetric
    off-diagonal structure (e.g. ``csd[w, 0, 1] != csd[w, 1, 0]``), which is
    exactly the bug ``tests/test_csd_injection_roundtrip.py`` guards against.

    Parameters
    ----------
    observed_csd : np.ndarray
        Cross-spectral density, shape ``(F, N, N)``. Cast to
        ``np.complex128`` (pitfall N3). This is the SAME array the VL engine
        fits.
    freqs : np.ndarray
        Frequency grid in Hz, shape ``(F,)``. Cast to ``np.float64``.
    a_mask : np.ndarray
        Binary connectivity mask, shape ``(N, N)``.
    c_mask : np.ndarray
        Binary driving input mask, shape ``(N, M)``.
    TR : float
        Repetition time in seconds.
    output_path : str
        Path for the output .mat file.
    bold_data : np.ndarray or None, optional
        Optional BOLD time series, shape ``(v, N)``, used only to set the
        ``DCM.Y.y`` shape (its values are unused once ``csd`` is injected). If
        None, a minimal float64 zeros placeholder of shape ``(v, N)`` with
        ``v = len(freqs) * 8`` is synthesized so ``DCM.v`` / ``DCM.n`` stay
        consistent.

    Notes
    -----
    Mirrors the struct conventions of ``export_spectral_dcm_for_spm``:
    ``options.induced = 1``, ``options.analysis = 'CSD'``, ``nograph = 1``,
    ``order = 8``, a minimal constant ``U.u`` input at microtime resolution.
    SPM12 source: ``spm_dcm_fmri_csd.m`` (when ``DCM.Y.csd`` is populated the
    internal ``spm_dcm_fmri_csd_data`` CSD estimation is skipped).
    """
    observed_csd = np.asarray(observed_csd).astype(np.complex128)
    freqs = np.asarray(freqs).astype(np.float64)

    N = a_mask.shape[0]
    M = c_mask.shape[1]
    num_freqs = freqs.shape[0]

    if bold_data is not None:
        bold = bold_data.astype(np.float64)
        v = bold.shape[0]
    else:
        v = num_freqs * 8
        bold = np.zeros((v, N), dtype=np.float64)

    # Microtime resolution for the (unused-but-shape-valid) constant input.
    microtime_factor = 16
    u_dt = TR / microtime_factor
    T_micro = v * microtime_factor + 32
    stimulus = np.ones((T_micro, M), dtype=np.float64)

    DCM = {
        "a": a_mask.astype(np.float64),
        "b": np.zeros((N, N, 0), dtype=np.float64),
        "c": c_mask.astype(np.float64),
        "d": np.zeros((N, N, 0), dtype=np.float64),
        "Y": {
            "y": bold,
            "dt": np.array([[TR]]),
            "X0": np.ones((v, 1), dtype=np.float64),
            "name": np.array(
                [[f"R{i + 1}" for i in range(N)]], dtype=object
            ),
            # Injected CSD (pitfall S4: C-order (F, N, N), no transpose).
            "csd": observed_csd,
            "Hz": freqs.reshape(-1, 1),
        },
        "U": {
            "u": stimulus,
            "dt": np.array([[u_dt]]),
            "name": np.array(
                [[f"stim{i + 1}" for i in range(M)]], dtype=object
            ),
        },
        "n": np.array([[N]]),
        "v": np.array([[v]]),
        "TE": np.array([[0.04]]),
        "delays": np.ones((1, N)) * TR / 2,
        "options": {
            "nonlinear": np.array([[0]]),
            "two_state": np.array([[0]]),
            "stochastic": np.array([[0]]),
            "centre": np.array([[0]]),
            "induced": np.array([[1]]),
            "nograph": np.array([[1]]),
            "maxit": np.array([[128]]),
            "analysis": np.array([["CSD"]], dtype=object),
            "order": np.array([[8]]),
        },
    }
    scipy.io.savemat(output_path, {"DCM": DCM})


def export_rdcm_for_tapas(
    bold_data: np.ndarray,
    stimulus: np.ndarray,
    a_mask: np.ndarray,
    c_mask: np.ndarray,
    TR: float,
    u_dt: float,
    output_path: str,
) -> None:
    """Export synthetic data as tapas rDCM-compatible .mat file.

    Uses the same DCM struct format as task DCM since tapas wraps
    SPM functions. No special options needed beyond ``nograph=1``.

    Parameters
    ----------
    bold_data : np.ndarray
        BOLD time series, shape ``(v, N)``. Must be float64.
    stimulus : np.ndarray
        Stimulus at microtime resolution, shape ``(T_micro, M)``.
    a_mask : np.ndarray
        Binary connectivity mask, shape ``(N, N)``.
    c_mask : np.ndarray
        Binary driving input mask, shape ``(N, M)``.
    TR : float
        Repetition time in seconds.
    u_dt : float
        Stimulus sampling interval in seconds.
    output_path : str
        Path for the output .mat file.

    Notes
    -----
    tapas ``tapas_rdcm_estimate`` expects the same DCM struct as
    SPM12. Source: tapas GitHub repository.
    """
    v = bold_data.shape[0]
    N = bold_data.shape[1]
    M = c_mask.shape[1]

    DCM = {
        "a": a_mask.astype(np.float64),
        "b": np.zeros((N, N, 0), dtype=np.float64),
        "c": c_mask.astype(np.float64),
        "d": np.zeros((N, N, 0), dtype=np.float64),
        "Y": {
            "y": bold_data.astype(np.float64),
            "dt": np.array([[TR]]),
            "X0": np.ones((v, 1), dtype=np.float64),
            "name": np.array(
                [[f"R{i + 1}" for i in range(N)]], dtype=object
            ),
        },
        "U": {
            "u": stimulus.astype(np.float64),
            "dt": np.array([[u_dt]]),
            "name": np.array(
                [[f"stim{i + 1}" for i in range(M)]], dtype=object
            ),
        },
        "n": np.array([[N]]),
        "v": np.array([[v]]),
        "TE": np.array([[0.04]]),
        "delays": np.ones((1, N)) * TR / 2,
        "options": {
            "nograph": np.array([[1]]),
        },
    }
    scipy.io.savemat(output_path, {"DCM": DCM})
