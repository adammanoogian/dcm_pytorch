"""Laptop round-trip checks for the 5-source CMC-ERP DCM input export (Phase 34-02).

Asserts that :func:`validation.export_to_mat.export_erp_dcm_multisource` writes a
DCM ``.mat`` whose ``scipy.io.loadmat`` round-trip carries the locked 5-source
auditory-MMN reference: ``P.A`` is a 1x4 cell of ``(5,5)`` blocks, ``P.B`` a cell
of ``(5,5)``, ``U.X`` is ``(2,1)`` double, ``M.x`` is ``zeros(5,8)``, and
``M.f == 'spm_fx_cmc_nodelay'`` (the D=1 wrapper, Fact 4). No MATLAB required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from validation.export_to_mat import export_erp_dcm_multisource


def test_export_erp_dcm_multisource_roundtrip(tmp_path: Path) -> None:
    """The written .mat round-trips with the locked cell / double encodings."""
    out = tmp_path / "erp_multisource_input.mat"
    meta = export_erp_dcm_multisource(output_path=str(out))

    assert meta["N"] == 5
    assert out.exists()

    mat = scipy.io.loadmat(str(out), squeeze_me=False, struct_as_record=False)
    dcm = mat["DCM"][0, 0]

    # P.A is a 1x4 cell, each block (5,5); P.B a cell of (5,5).
    p = dcm.P[0, 0]
    assert p.A.shape == (1, 4)
    for i in range(4):
        assert p.A[0, i].shape == (5, 5)
        assert p.A[0, i].dtype == np.float64
    assert p.B.shape[0] == 1
    assert p.B[0, 0].shape == (5, 5)

    # U.X is (2,1) double (standard / deviant).
    u = dcm.U[0, 0]
    assert u.X.shape == (2, 1)
    assert u.X.dtype == np.float64
    np.testing.assert_array_equal(u.X.ravel(), np.array([0.0, 1.0]))

    # M.x is zeros(5,8); M.f decodes to the D=1 wrapper; M.n == 40.
    m = dcm.M[0, 0]
    assert m.x.shape == (5, 8)
    np.testing.assert_array_equal(m.x, np.zeros((5, 8)))
    assert "spm_fx_cmc_nodelay" in str(np.asarray(m.f).tolist())
    assert float(m.n[0, 0]) == 40.0

    # meta carries the lock: N=5, D=1, the design X, and the edge lists.
    md = dcm.meta[0, 0]
    assert float(md.N[0, 0]) == 5.0
    assert float(md.D[0, 0]) == 1.0
    assert md.edges_forward.shape == (4, 2)
    assert md.edges_backward.shape == (4, 2)
    assert md.edges_lateral.shape == (2, 2)


def test_export_erp_dcm_multisource_b_folding_teeth(tmp_path: Path) -> None:
    """B differs from A on edges and carries a non-zero diag (the EVOK-02 knob).

    ``output_path`` is redirected to ``tmp_path`` even though this test only
    inspects the returned metadata: the exporter's default writes to the
    tracked, byte-frozen ``validation/data/erp_multisource_input.mat``, so
    calling it bare mutates a committed SPM12 fixture as a side effect.
    """
    meta = export_erp_dcm_multisource(
        output_path=str(tmp_path / "erp_multisource_input.mat")
    )
    # Forward + lateral + backward edges are recorded for the Wave-3 ladder.
    assert (2, 0) in meta["edges_forward"]  # A1L -> STGL
    assert (4, 2) in meta["edges_forward"]  # STGL -> rIFG
    assert (2, 3) in meta["edges_lateral"]  # STGR <-> STGL
    assert (0, 2) in meta["edges_backward"]  # STGL -> A1L
    # Input drives bilateral A1; precision nodes are rIFG + bilateral A1.
    assert meta["input_sources"] == (0, 1)
    assert meta["precision_nodes"] == (4, 0, 1)
