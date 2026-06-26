"""Laptop round-trip checks for the LFP lead-field DCM input export (Phase 35-02).

Asserts that :func:`validation.export_to_mat.export_erp_dcm_leadfield` writes a
DCM ``.mat`` whose ``scipy.io.loadmat`` round-trip carries the locked 5-source
auditory-MMN reference PLUS the LFP single-dipole spatial spec: ``dipfit.type ==
'LFP'``, ``P.L = ones(1,5)`` (identity diagonal), ``P.J`` one-hot at index 2
(superficial-pyramidal voltage, ``spm_L_priors.m:108``), and all-double dims. No
MATLAB required (FlexLM unreachable locally; the actual ``spm_lx_erp`` run is the
M3 job in Task 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from validation.export_to_mat import (
    export_erp_dcm_leadfield,
    export_erp_dcm_multisource,
)


def test_export_erp_dcm_leadfield_roundtrip(tmp_path: Path) -> None:
    """The written .mat round-trips with the LFP spatial spec + locked net."""
    out = tmp_path / "erp_leadfield_input.mat"
    meta = export_erp_dcm_leadfield(output_path=str(out))

    assert meta["N"] == 5
    assert meta["Nc"] == 5
    assert meta["dipfit_type"] == "LFP"
    assert out.exists()

    mat = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    dcm = mat["DCM"][0, 0]

    # dipfit.type == 'LFP', Ns == Nc == 5.
    dipfit = dcm.dipfit[0, 0]
    assert "LFP" in str(np.asarray(dipfit.type).tolist())
    assert float(dipfit.Ns[0, 0]) == 5.0
    assert float(dipfit.Nc[0, 0]) == 5.0

    # P carries the LFP spatial spec: P.L = ones(1,5), P.J one-hot at index 2.
    p = dcm.P[0, 0]
    assert p.L.shape == (1, 5)
    assert p.L.dtype == np.float64
    np.testing.assert_array_equal(p.L, np.ones((1, 5)))

    assert p.J.shape == (1, 8)
    assert p.J.dtype == np.float64
    expected_j = np.zeros((1, 8))
    expected_j[0, 2] = 1.0  # sp-voltage (spm_L_priors.m:108)
    np.testing.assert_array_equal(p.J, expected_j)
    assert int(np.argmax(p.J.ravel())) == 2

    # P.A still a 1x4 cell of (5,5); P.B a cell of (5,5) -- net topology intact.
    assert p.A.shape == (1, 4)
    for i in range(4):
        assert p.A[0, i].shape == (5, 5)
        assert p.A[0, i].dtype == np.float64
    assert p.B[0, 0].shape == (5, 5)

    # M.x is zeros(5,8); M.f is the D=1 wrapper; M.n == 40.
    m = dcm.M[0, 0]
    assert m.x.shape == (5, 8)
    np.testing.assert_array_equal(m.x, np.zeros((5, 8)))
    assert "spm_fx_cmc_nodelay" in str(np.asarray(m.f).tolist())
    assert float(m.n[0, 0]) == 40.0

    # meta: N=5, Nc=5, D=1, dipfit_type='LFP', P_J / P_L, the edge lists.
    md = dcm.meta[0, 0]
    assert float(md.N[0, 0]) == 5.0
    assert float(md.Nc[0, 0]) == 5.0
    assert float(md.D[0, 0]) == 1.0
    assert "LFP" in str(np.asarray(md.dipfit_type).tolist())
    assert md.edges_forward.shape == (4, 2)
    assert md.edges_backward.shape == (4, 2)
    assert md.edges_lateral.shape == (2, 2)


def test_export_erp_dcm_leadfield_all_double(tmp_path: Path) -> None:
    """All numeric dims cast to double (int64 -> spm_Ce footgun, a27828b)."""
    out = tmp_path / "erp_leadfield_input.mat"
    export_erp_dcm_leadfield(output_path=str(out))

    mat = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    dcm = mat["DCM"][0, 0]
    p = dcm.P[0, 0]
    for name in ("T", "G", "C", "S", "R", "L", "J"):
        arr = np.asarray(getattr(p, name))
        assert arr.dtype == np.float64, f"P.{name} must be float64, got {arr.dtype}"
    assert dcm.n[0, 0].dtype == np.float64
    assert dcm.v[0, 0].dtype == np.float64


def test_export_erp_dcm_leadfield_additive(tmp_path: Path) -> None:
    """The multi-source exporter is unaffected (no P.L / P.J / dipfit leak)."""
    out = tmp_path / "erp_multisource_input.mat"
    export_erp_dcm_multisource(output_path=str(out))

    mat = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    dcm = mat["DCM"][0, 0]
    p = dcm.P[0, 0]
    # The multi-source P must NOT carry the lead-field fields.
    assert not hasattr(p, "L")
    assert not hasattr(p, "J")
    # And no dipfit on the multi-source DCM struct.
    assert not hasattr(dcm, "dipfit")
