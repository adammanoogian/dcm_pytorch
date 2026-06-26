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
        "n": np.array([[N]], dtype=np.float64),
        "v": np.array([[v]], dtype=np.float64),
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
        "n": np.array([[N]], dtype=np.float64),
        "v": np.array([[v]], dtype=np.float64),
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
        "n": np.array([[N]], dtype=np.float64),
        "v": np.array([[v]], dtype=np.float64),
        "TE": np.array([[0.04]]),
        "delays": np.ones((1, N)) * TR / 2,
        "options": {
            "nograph": np.array([[1]]),
        },
    }
    scipy.io.savemat(output_path, {"DCM": DCM})


# --- ERP / CMC single-source fixture export (Phase 33-02, additive) -----------
# Frozen single-source reference locked by 33-01-SUMMARY.md ("Frozen single-source
# reference"): the SAME (x_test, u_test) and integration grid that the pure-torch
# CMC forward evaluates, so the Wave-3 parity gate is element-wise vs-SPM (V4).
_ERP_NS = 128          # M.ns -- number of peristimulus time bins (512 ms window)
_ERP_DT = 0.004        # U.dt in seconds (4 ms) -- spm_gen_erp.m:28-30 default
_ERP_ONS = 60.0        # M.ons in ms -- spm_erp_u.m:46 default onset
_ERP_DUR = 16.0        # M.dur in ms -- spm_erp_u.m:47 default dispersion (N3 explicit)
_ERP_SUS = 0.0         # M.sus -- sustained-input proportion (0 -> pure Gaussian)
_ERP_NSTATES = 8       # CMC states per source (spm_fx_cmc.m), column-major flat (1,8)


def _erp_gaussian_u_grid(
    R: np.ndarray,
    ns: int,
    dt: float,
    ons: float,
    dur: float,
    sus: float,
) -> np.ndarray:
    """Build the frozen Gaussian evoked-input grid ``U.u`` (numpy port of spm_erp_u).

    Transcribes ``spm_erp_u.m:42-64`` (David & Friston 2003 evoked drive): a
    Gaussian bump in peristimulus time with the 32x scaling baked in. ``spm_int_L``
    integrates ``DCM.U.u`` verbatim (it does NOT regenerate the input), so freezing
    the grid here guarantees SPM and torch see the identical drive -- the precondition
    for the ``y_states`` trajectory parity check. With ``sus = 0`` the sustained-mix
    term vanishes and only the third ``R`` column (unused here) would gate it.

    Parameters
    ----------
    R : np.ndarray
        Input-timing log params, shape ``(m, 2)`` (onset shift, log-dispersion).
        ``R = 0`` recovers the default onset/dispersion (peak value 32 at ``ons``).
    ns : int
        Number of peristimulus time bins.
    dt : float
        Time step in seconds.
    ons : float
        Default onset in milliseconds.
    dur : float
        Default dispersion (Gaussian sigma) in milliseconds.
    sus : float
        Sustained-input proportion (``M.sus``); 0 -> pure Gaussian.

    Returns
    -------
    np.ndarray
        Driving input grid, shape ``(ns, m)``, float64.

    Notes
    -----
    SPM12 source: ``spm_erp_u.m:42-64``. ``t_ms = t * 1000`` (the seconds->ms
    convert at ``:46``), ``delay = ons + 128 * R[i,0]``, ``scale = dur * exp(R[i,1])``.
    """
    m = R.shape[0]
    t_ms = np.arange(ns, dtype=np.float64) * dt * 1000.0
    u = np.zeros((ns, m), dtype=np.float64)
    for i in range(m):
        delay = ons + 128.0 * float(R[i, 0])
        scale = dur * np.exp(float(R[i, 1]))
        gaussian = np.exp(-((t_ms - delay) ** 2) / (2.0 * scale**2))
        # sus = 0 -> prop = 0 -> the cumsum sustained-mix term drops out.
        prop = sus  # exp(R[i,2]) would scale it, but R[:,2] is absent at sus=0.
        gaussian = prop * np.cumsum(gaussian) / gaussian.sum() + gaussian * (
            1.0 - prop
        )
        u[:, i] = 32.0 * gaussian
    return u


def export_erp_dcm(
    P: dict[str, np.ndarray] | None = None,
    M_meta: dict[str, float] | None = None,
    output_path: str = "validation/data/erp_single_source_input.mat",
) -> dict[str, object]:
    """Export the single-source CMC-ERP DCM ``.mat`` input for SPM12 fixtures.

    Writes the DCM struct that ``run_spm_erp_dcm.m`` loads via ``load(..., 'DCM')``
    to generate the frozen Phase-33 parity fixtures (``f_field``, ``J0``, ``dtJ``,
    ``Eexp``, ``Q_update``, ``y_states``). APPENDED to this module (the existing
    task / spectral / rDCM exporters are byte-untouched), mirroring their savemat
    conventions: scalars wrapped as ``np.array([[v]])``, string fields as
    ``np.array([[...]], dtype=object)``, dimensions cast to float64 (the int64->double
    ``spm_Ce`` footgun fixed in Phase 32, commit a27828b / decision 32-03 -- savemat
    otherwise writes int64 which breaks SPM internals).

    The defaults encode the frozen single-source reference locked in
    ``33-01-SUMMARY.md``: ``P.T`` zeros(1,4), ``P.G`` zeros(1,4), ``P.C`` zeros(1,1),
    ``P.S`` 0, ``P.R`` zeros(1,2); NO ``P.D`` field (delays stay off -- pitfall M2)
    and NO ``P.A`` blocks (extrinsic coupling is identically zero at n=1). The
    integration grid is ``M.ns = 128``, ``U.dt = 0.004`` s, ``M.ons = 60`` ms,
    ``M.dur = 16`` ms EXPLICIT (pitfall N3); ``M.x = zeros(1,8)`` is the asserted
    fixed point (x0 == 0, M1). The frozen ``f_field`` evaluation point
    ``x_test = 0.1 * ones(1,8)`` and ``u_test = 32.0`` (peak Gaussian, ``P.R = 0``)
    are stored in ``DCM.meta`` so MATLAB and torch evaluate ``cmc_f`` at the SAME
    point.

    Parameters
    ----------
    P : dict of str -> np.ndarray, optional
        Frozen parameter struct. Keys ``T`` (1,4), ``G`` (1,4), ``C`` (1,1),
        ``S`` (1,1), ``R`` (m,2). Defaults to the all-zeros log-scale prior mean
        (the baseline fixture). NO ``D`` / ``A`` keys (pitfall M2 / n=1).
    M_meta : dict of str -> float, optional
        Integration-grid / timing overrides. Keys ``ns``, ``dt``, ``ons``, ``dur``,
        ``sus``, ``n_inputs``. Defaults to the frozen ERP grid above.
    output_path : str, optional
        Path for the output DCM input ``.mat``.

    Returns
    -------
    dict of str -> object
        The provenance metadata actually written to ``DCM.meta`` (dt, ns, ons,
        dur, sus, x_test, u_test, n_states, n_inputs, spm_id placeholder), for the
        caller to log / assert against.

    Notes
    -----
    SPM12 source: ``spm_cmc_priors.m:114-133`` (the single-source prior struct
    shape), ``spm_erp_u.m:42-64`` (the evoked input), ``spm_int_L.m:112-169`` (the
    integrator the fixtures pin). The ``$Id`` provenance string is filled by
    ``run_spm_erp_dcm.m`` at run time (``spm('Ver')``); ``meta.spm_id`` is an empty
    placeholder here.

    See Also
    --------
    export_spectral_dcm_csd_for_spm : The Phase-32 injection bridge this mirrors.
    """
    if P is None:
        P = {
            "T": np.zeros((1, 4), dtype=np.float64),
            "G": np.zeros((1, 4), dtype=np.float64),
            "C": np.zeros((1, 1), dtype=np.float64),
            "S": np.zeros((1, 1), dtype=np.float64),
            "R": np.zeros((1, 2), dtype=np.float64),
        }
    if M_meta is None:
        M_meta = {}

    ns = int(M_meta.get("ns", _ERP_NS))
    dt = float(M_meta.get("dt", _ERP_DT))
    ons = float(M_meta.get("ons", _ERP_ONS))
    dur = float(M_meta.get("dur", _ERP_DUR))
    sus = float(M_meta.get("sus", _ERP_SUS))

    R = np.asarray(P["R"], dtype=np.float64)
    n_inputs = int(M_meta.get("n_inputs", R.shape[0]))

    # Frozen Gaussian evoked drive (spm_int_L integrates this verbatim).
    u_grid = _erp_gaussian_u_grid(R, ns, dt, ons, dur, sus)

    # Frozen f-field evaluation point (Open Question 1, locked in 33-01-SUMMARY).
    x_test = 0.1 * np.ones((1, _ERP_NSTATES), dtype=np.float64)
    u_test = 32.0  # peak Gaussian value at onset with P.R = 0.

    P_struct = {
        "T": np.asarray(P["T"], dtype=np.float64),
        "G": np.asarray(P["G"], dtype=np.float64),
        "C": np.asarray(P["C"], dtype=np.float64),
        "S": np.asarray(P["S"], dtype=np.float64),
        "R": R,
    }  # NO 'D', NO 'A' -- delays off (M2), extrinsic identically zero at n=1.

    meta = {
        "dt": np.array([[dt]]),
        "ns": np.array([[ns]], dtype=np.float64),
        "ons": np.array([[ons]]),
        "dur": np.array([[dur]]),
        "sus": np.array([[sus]]),
        "x_test": x_test,
        "u_test": np.array([[u_test]]),
        "n_states": np.array([[_ERP_NSTATES]], dtype=np.float64),
        "n_inputs": np.array([[n_inputs]], dtype=np.float64),
        "D": np.array([[1.0]]),  # delay operator forced to identity (Fact 4).
        "spm_id": np.array([[""]], dtype=object),  # filled by the .m at run time.
    }

    DCM = {
        "P": P_struct,
        "M": {
            "f": np.array([["spm_fx_cmc_nodelay"]], dtype=object),
            "x": np.zeros((1, _ERP_NSTATES), dtype=np.float64),
            "n": np.array([[_ERP_NSTATES]], dtype=np.float64),
            "m": np.array([[n_inputs]], dtype=np.float64),
            "l": np.array([[1.0]]),
            "ons": np.array([[ons]]),
            "dur": np.array([[dur]]),
            "sus": np.array([[sus]]),
        },
        "U": {
            "u": u_grid,
            "dt": np.array([[dt]]),
            "name": np.array(
                [[f"input{i + 1}" for i in range(n_inputs)]], dtype=object
            ),
        },
        # Dimensions as float64 (int64 -> spm_Ce footgun, Phase 32 / a27828b).
        "n": np.array([[1.0]], dtype=np.float64),
        "v": np.array([[ns]], dtype=np.float64),
        "meta": meta,
    }
    scipy.io.savemat(output_path, {"DCM": DCM})
    return {
        "dt": dt,
        "ns": ns,
        "ons": ons,
        "dur": dur,
        "sus": sus,
        "x_test": x_test,
        "u_test": u_test,
        "n_states": _ERP_NSTATES,
        "n_inputs": n_inputs,
        "spm_id": "",
    }


# =============================================================================
# Multi-source CMC-ERP export (Phase 34-02, EVOK-05/06) -- APPENDED, additive.
# The single-source export_erp_dcm above is byte-untouched.
# =============================================================================

# The canonical 5-source auditory-MMN reference TOPOLOGY (NO MNI coords -- those
# are Phase 36). Sources, 0-indexed: A1L, A1R, STGL, STGR, rIFG.
_MS_SOURCE_NAMES = ("A1L", "A1R", "STGL", "STGR", "rIFG")
_MS_N = 5
_A1L, _A1R, _STGL, _STGR, _RIFG = 0, 1, 2, 3, 4

# Extrinsic edges as [to, from] (the cmc_network_f / spm_fx_cmc convention:
# A[i] @ S routes firing from column `from` into row `to`). Forward blocks A{1}
# (sp->ss) and A{2} (sp->dp) carry the bottom-up edges; backward blocks A{3}
# (dp->sp) and A{4} (dp->ii) carry the top-down edges (spm_fx_cmc.m:171-198).
_MS_FORWARD_EDGES = (
    (_STGL, _A1L),   # A1L  -> STGL
    (_STGR, _A1R),   # A1R  -> STGR
    (_RIFG, _STGL),  # STGL -> rIFG
    (_RIFG, _STGR),  # STGR -> rIFG
)
# Lateral RECIPROCAL pair STGL<->STGR -- both directions live in the forward
# blocks so the (1+4L) lateral reduction fires (spm_fx_cmc.m:79-82;
# erp_coupled_system.parameterize_cmc_network detects recip = a>exp(-8) &
# a.T>exp(-8)). A clean reciprocal test pair (34-RESEARCH Open Q3).
_MS_LATERAL_EDGES = (
    (_STGR, _STGL),  # STGL <-> STGR
    (_STGL, _STGR),
)
_MS_BACKWARD_EDGES = (
    (_STGL, _RIFG),  # rIFG -> STGL
    (_STGR, _RIFG),  # rIFG -> STGR
    (_A1L, _STGL),   # STGL -> A1L
    (_A1R, _STGR),   # STGR -> A1R
)
# Auditory input drives bilateral A1 only (the cortical input recipients).
_MS_INPUT_SOURCES = (_A1L, _A1R)
# Precision (self-inhibition) nodes whose diag(B) modulates G[:,0] (the MMN knob).
_MS_PRECISION_NODES = (_RIFG, _A1L, _A1R)

# Modest, deterministic free-log-space values (mask*32-32 convention, 34-01-D3:
# exp(-32)*E0 << exp(-8) is "off"; the live log-value is "on"). B differs from A
# so the Wave-3 ladder has folding teeth (Q.A = A + X*B element-wise checkable).
_MS_A_LIVE = 0.0       # exp(0)*E0 = E0 (fully on)
_MS_A_DEAD = -32.0     # exp(-32)*E0 ~ 2.5e-12 (off)
_MS_B_EDGE = 0.3       # B on every extrinsic edge (folded into all 4 A blocks)
_MS_B_DIAG = 0.5       # diag(B) at the precision nodes -> Q.G[:,0] (EVOK-02)


def _ms_log_block(edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Build a ``(5,5)`` free-log-space extrinsic block from a ``[to, from]`` list.

    Live edges get ``_MS_A_LIVE`` (``exp(0)*E0`` on); every other entry gets
    ``_MS_A_DEAD`` (``exp(-32)*E0`` off, the ``mask*32-32`` convention,
    34-01-D3).

    Parameters
    ----------
    edges : tuple of (int, int)
        ``(to, from)`` index pairs that are live in this block.

    Returns
    -------
    np.ndarray
        Free-log-space block, shape ``(5, 5)``, float64.
    """
    block = np.full((_MS_N, _MS_N), _MS_A_DEAD, dtype=np.float64)
    for to_i, from_i in edges:
        block[to_i, from_i] = _MS_A_LIVE
    return block


def export_erp_dcm_multisource(
    P: dict[str, np.ndarray] | None = None,
    M_meta: dict[str, float] | None = None,
    output_path: str = "validation/data/erp_multisource_input.mat",
) -> dict[str, object]:
    """Export the 5-source CMC-ERP DCM ``.mat`` input for SPM12 MMN fixtures.

    Writes the multi-source DCM struct that ``run_spm_erp_dcm_multisource.m``
    loads to generate the frozen Wave-3 parity fixtures (per-condition
    ``spm_gen_Q`` ``QA``/``QG``, per-condition frozen ``J0``/``Qupd``, and the
    ``spm_gen_erp`` multi-source trajectory ``y``). APPENDED to this module (the
    single-source :func:`export_erp_dcm` and all other exporters are
    byte-untouched), mirroring their savemat conventions: scalars wrapped as
    ``np.array([[v]])``, string fields as ``np.array([[...]], dtype=object)``,
    and ALL dimensions cast to float64 (the int64->double ``spm_Ce`` footgun
    fixed in Phase 32, commit a27828b / decision 32-03).

    The defaults LOCK the canonical 5-source auditory-MMN reference TOPOLOGY
    (sources A1L, A1R, STGL, STGR, rIFG; NO MNI coordinates -- those are Phase
    36) as explicit ``(5,5)`` extrinsic masks (34-RESEARCH Open Q3):

    * Forward ``A{1}`` (sp->ss) & ``A{2}`` (sp->dp): A1L->STGL, A1R->STGR,
      STGL->rIFG, STGR->rIFG.
    * Backward ``A{3}`` (dp->sp) & ``A{4}`` (dp->ii): rIFG->STGL, rIFG->STGR,
      STGL->A1L, STGR->A1R.
    * Lateral reciprocal STGL<->STGR (added to the forward blocks; triggers the
      ``(1+4L)`` reduction, ``spm_fx_cmc.m:79-82``).
    * Input ``C`` drives A1L and A1R only (bilateral auditory recipients).
    * Condition ``B{1}`` modulates every extrinsic edge AND carries a non-zero
      ``diag(B)`` at rIFG + bilateral A1 (the precision nodes), with design
      ``X = [[0],[1]]`` (standard vs deviant, ``Cnd=2``, ``n_effects=1``).

    The free-log-space ``P`` is built from those masks (``_MS_A_LIVE`` on,
    ``_MS_A_DEAD`` off; ``B`` distinct from ``A`` so the Wave-3 folding check has
    teeth). ``P.A``/``P.B`` are encoded as MATLAB CELL arrays
    (``np.empty((1,k), dtype=object)``); ``U.X`` is ``(Cnd, n_effects)`` double;
    ``M.x = zeros(5,8)``, ``M.n = 40``, ``M.f = 'spm_fx_cmc_nodelay'`` (D=1).

    Parameters
    ----------
    P : dict of str -> np.ndarray, optional
        Free-parameter struct. Keys ``A`` (length-4 list of ``(5,5)``), ``B``
        (length-``n_effects`` list of ``(5,5)``), ``T`` ``(5,4)``, ``G``
        ``(5,4)``, ``C`` ``(5,n_inp)``, ``S`` ``(1,1)`` (scalar slope,
        spm_cmc_priors.m:124), ``R`` ``(n_inp,2)`` and
        ``X`` ``(Cnd,n_effects)``. Defaults to the locked MMN reference above.
    M_meta : dict of str -> float, optional
        Integration-grid / timing overrides (``ns``, ``dt``, ``ons``, ``dur``,
        ``sus``). Defaults to the frozen ERP grid (ns=128, dt=0.004, ons=60,
        dur=16).
    output_path : str, optional
        Path for the output DCM input ``.mat``.

    Returns
    -------
    dict of str -> object
        Provenance metadata (N, edge lists, X, dt/ns/ons/dur, n_inputs), for the
        caller to log / assert against.

    Notes
    -----
    SPM12 source: ``spm_fx_cmc.m:68-82,171-198`` (the extrinsic blocks + lateral
    reduction + the four routes), ``spm_gen_Q.m:24-67`` (the ``B``->all-``A`` +
    ``diag(B)``->``G(:,1)`` folding), ``spm_gen_erp.m:69-86`` (the per-condition
    evoked loop), ``spm_erp_u.m:42-64`` (the Gaussian drive). The ``$Id``
    provenance strings are captured by ``run_spm_erp_dcm_multisource.m`` at run
    time.

    See Also
    --------
    export_erp_dcm : The single-source bridge this mirrors additively.
    """
    if M_meta is None:
        M_meta = {}

    ns = int(M_meta.get("ns", _ERP_NS))
    dt = float(M_meta.get("dt", _ERP_DT))
    ons = float(M_meta.get("ons", _ERP_ONS))
    dur = float(M_meta.get("dur", _ERP_DUR))
    sus = float(M_meta.get("sus", _ERP_SUS))

    n = _MS_N

    if P is None:
        # --- Lock the 5-source reference free-log-space P from the masks. ---
        a_blocks = [
            _ms_log_block(_MS_FORWARD_EDGES + _MS_LATERAL_EDGES),  # A{1} sp->ss
            _ms_log_block(_MS_FORWARD_EDGES + _MS_LATERAL_EDGES),  # A{2} sp->dp
            _ms_log_block(_MS_BACKWARD_EDGES),                     # A{3} dp->sp
            _ms_log_block(_MS_BACKWARD_EDGES),                     # A{4} dp->ii
        ]
        # Condition B{1}: every extrinsic edge + diag at the precision nodes.
        b1 = np.zeros((n, n), dtype=np.float64)
        for to_i, from_i in (
            _MS_FORWARD_EDGES + _MS_LATERAL_EDGES + _MS_BACKWARD_EDGES
        ):
            b1[to_i, from_i] = _MS_B_EDGE
        for node in _MS_PRECISION_NODES:
            b1[node, node] = _MS_B_DIAG
        b_list = [b1]
        # Input C: drive A1L/A1R only (mask*32-32 -> exp(P.C) ~ 0 elsewhere).
        c = np.full((n, 1), _MS_A_DEAD, dtype=np.float64)
        for src in _MS_INPUT_SOURCES:
            c[src, 0] = _MS_A_LIVE
        t = np.zeros((n, 4), dtype=np.float64)
        g = np.zeros((n, 4), dtype=np.float64)
        # P.S is a SCALAR slope (spm_cmc_priors.m:124 E.S = 0): spm_fx_cmc.m:92-93
        # forms R = (2/3)*exp(P.S) then F = sigmoid(-R*x), and -R*x is a MATRIX
        # multiply -- a per-source (n,1) S makes (n,1)*(n,8) fail (34-02-D2).
        s = np.zeros((1, 1), dtype=np.float64)
        r = np.zeros((1, 2), dtype=np.float64)
        x_design = np.array([[0.0], [1.0]], dtype=np.float64)  # std / deviant
    else:
        a_blocks = [np.asarray(P["A"][i], dtype=np.float64) for i in range(4)]
        b_list = [np.asarray(b, dtype=np.float64) for b in P["B"]]
        c = np.asarray(P["C"], dtype=np.float64)
        t = np.asarray(P["T"], dtype=np.float64)
        g = np.asarray(P["G"], dtype=np.float64)
        s = np.asarray(P["S"], dtype=np.float64)
        r = np.asarray(P["R"], dtype=np.float64)
        x_design = np.asarray(P["X"], dtype=np.float64)

    n_inputs = int(c.shape[1])
    n_effects = int(x_design.shape[1])

    # Frozen Gaussian evoked drive (shared across sources; spm_int_L integrates
    # U.u verbatim, so freezing it pins the trajectory parity, pitfall V1).
    u_grid = _erp_gaussian_u_grid(r, ns, dt, ons, dur, sus)

    # P.A / P.B as MATLAB cell arrays (object ndarray); U.X as double.
    a_cell = np.empty((1, 4), dtype=object)
    for i in range(4):
        a_cell[0, i] = a_blocks[i]
    b_cell = np.empty((1, n_effects), dtype=object)
    for i in range(n_effects):
        b_cell[0, i] = b_list[i]

    P_struct = {
        "A": a_cell,
        "B": b_cell,
        "T": t,
        "G": g,
        "C": c,
        "S": s,
        "R": r,
    }  # NO 'D' -- delays forced off via M.f = spm_fx_cmc_nodelay (Fact 4, M2).

    edges_forward = np.asarray(_MS_FORWARD_EDGES, dtype=np.float64)
    edges_lateral = np.asarray(_MS_LATERAL_EDGES, dtype=np.float64)
    edges_backward = np.asarray(_MS_BACKWARD_EDGES, dtype=np.float64)
    input_sources = np.asarray(_MS_INPUT_SOURCES, dtype=np.float64).reshape(1, -1)
    precision_nodes = np.asarray(
        _MS_PRECISION_NODES, dtype=np.float64
    ).reshape(1, -1)

    meta = {
        "N": np.array([[n]], dtype=np.float64),
        "D": np.array([[1.0]]),  # delay operator forced to identity (Fact 4).
        "source_names": np.array([list(_MS_SOURCE_NAMES)], dtype=object),
        "edges_forward": edges_forward,    # [to, from], 0-indexed
        "edges_lateral": edges_lateral,
        "edges_backward": edges_backward,
        "input_sources": input_sources,
        "precision_nodes": precision_nodes,
        "X": x_design,
        "dt": np.array([[dt]]),
        "ns": np.array([[ns]], dtype=np.float64),
        "ons": np.array([[ons]]),
        "dur": np.array([[dur]]),
        "sus": np.array([[sus]]),
        "n_states": np.array([[_ERP_NSTATES]], dtype=np.float64),
        "n_inputs": np.array([[n_inputs]], dtype=np.float64),
        "n_effects": np.array([[n_effects]], dtype=np.float64),
        "spm_id": np.array([[""]], dtype=object),  # filled by the .m at run time.
    }

    DCM = {
        "P": P_struct,
        "M": {
            "f": np.array([["spm_fx_cmc_nodelay"]], dtype=object),
            "x": np.zeros((n, _ERP_NSTATES), dtype=np.float64),
            "n": np.array([[n * _ERP_NSTATES]], dtype=np.float64),
            "m": np.array([[n_inputs]], dtype=np.float64),
            "l": np.array([[n]], dtype=np.float64),
            "ons": np.array([[ons]]),
            "dur": np.array([[dur]]),
            "sus": np.array([[sus]]),
        },
        "U": {
            "u": u_grid,
            "dt": np.array([[dt]]),
            "X": x_design,
            "name": np.array(
                [[f"input{i + 1}" for i in range(n_inputs)]], dtype=object
            ),
        },
        # Dimensions as float64 (int64 -> spm_Ce footgun, Phase 32 / a27828b).
        "n": np.array([[float(n)]], dtype=np.float64),
        "v": np.array([[ns]], dtype=np.float64),
        "meta": meta,
    }
    scipy.io.savemat(output_path, {"DCM": DCM})
    return {
        "N": n,
        "dt": dt,
        "ns": ns,
        "ons": ons,
        "dur": dur,
        "sus": sus,
        "n_inputs": n_inputs,
        "n_effects": n_effects,
        "source_names": list(_MS_SOURCE_NAMES),
        "edges_forward": _MS_FORWARD_EDGES,
        "edges_lateral": _MS_LATERAL_EDGES,
        "edges_backward": _MS_BACKWARD_EDGES,
        "input_sources": _MS_INPUT_SOURCES,
        "precision_nodes": _MS_PRECISION_NODES,
        "X": x_design.tolist(),
        "spm_id": "",
    }
