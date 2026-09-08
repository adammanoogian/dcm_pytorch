"""C-order CSD round-trip regression tests (VLREC-05, pitfall S4).

Guards the bug class fixed in commit 64e326f: PyTorch's ``.reshape(-1)`` of an
``(F, N, N)`` tensor is C-order, so the linear index is ``w * N * N + i * N + j``
with ``j`` varying FASTEST, then ``i``, then ``w``. The CSD observation precision
builder (``pyro_dcm.inference.csd_precision.compute_csd_precision``) inverts this
map via ``j = idx % N; i = (idx // N) % N; w = idx // (N * N)``. A future edit that
swaps to a column-major (Fortran) layout, or that transposes the complex CSD, would
silently corrupt the asymmetric off-diagonal structure and only surface as
asymmetric A-matrix errors much later (Phase 32 SPM cross-validation).

These tests lock the C-order contract in place with pure tensor algebra (< 1s, no
VL fit, no cluster).
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.inference.csd_precision import compute_csd_precision


def _asymmetric_csd(num_freqs: int, num_regions: int) -> torch.Tensor:
    """Build a deterministic asymmetric complex CSD of shape (F, N, N).

    ``csd[w, i, j] = (w + 1) * 100 + i * 10 + j + 1j * (i - j)``. The real part
    is asymmetric (``csd[w, i, j] != csd[w, j, i]``) and the imaginary part is
    antisymmetric -- exactly the structure a transposition bug would corrupt.
    """
    w = torch.arange(num_freqs).view(num_freqs, 1, 1)
    i = torch.arange(num_regions).view(1, num_regions, 1)
    j = torch.arange(num_regions).view(1, 1, num_regions)
    real = (w + 1) * 100 + i * 10 + j
    imag = i - j
    return (real + 1j * imag).to(torch.complex128)


@pytest.mark.vl
def test_corder_index_roundtrip_recovers_asymmetric_elements() -> None:
    """The C-order (j, i, w) map recovers every element of an asymmetric CSD.

    This is the literal contract the precision builder depends on (commit
    64e326f, pitfall S4).
    """
    num_freqs, num_regions = 3, 3
    csd = _asymmetric_csd(num_freqs, num_regions)
    flat = csd.reshape(-1)

    n = num_regions
    for idx in range(flat.numel()):
        j = idx % n
        i = (idx // n) % n
        w = idx // (n * n)
        assert flat[idx] == csd[w, i, j], (
            f"C-order roundtrip mismatch at idx={idx}: expected "
            f"csd[{w}, {i}, {j}]={csd[w, i, j]}, got flat[{idx}]={flat[idx]}"
        )


@pytest.mark.vl
def test_compute_csd_precision_block_structure_not_transposed() -> None:
    """compute_csd_precision preserves the asymmetric block layout.

    An asymmetric input must yield a layout where the (i=0, j=1) and (i=1, j=0)
    slots land in distinct, correctly ordered positions -- i.e. the builder did
    not symmetrize away the asymmetry (commit 64e326f, pitfall S4).
    """
    num_freqs, num_regions = 3, 3
    csd = _asymmetric_csd(num_freqs, num_regions)

    q_list, nq = compute_csd_precision(csd)

    assert nq == 1
    assert len(q_list) == 1
    dim = num_freqs * num_regions * num_regions
    assert q_list[0].shape == (dim, dim)

    # Within frequency 0's block (flat positions 0 .. N^2 - 1), the C-order map
    # places (i, j) at flat position i * N + j. So (i=0, j=1) -> position 1 and
    # (i=1, j=0) -> position N. These must be distinct, finite, and NOT forced
    # equal (a symmetrizing/transposing bug would collapse them).
    n = num_regions
    pos_01 = 0 * n + 1
    pos_10 = 1 * n + 0
    assert pos_01 != pos_10

    precision = q_list[0]
    assert torch.isfinite(precision[pos_01, pos_01].abs())
    assert torch.isfinite(precision[pos_10, pos_10].abs())
    # The asymmetric input must not be symmetrized into equal off-diagonal slots.
    assert precision[pos_01, pos_10] != precision[pos_10, pos_01]


@pytest.mark.vl
def test_compute_csd_precision_uses_induced_one_norm() -> None:
    """The Q regulariser is MATLAB's induced 1-norm, not an entrywise sum.

    ``spm_dcm_csd_Q.m`` closes with
    ``Q = inv(Q + norm(Q,1)*speye(size(Q))/32)``. MATLAB ``norm(Q,1)`` is the
    INDUCED matrix 1-norm -- the maximum absolute column sum. It is NOT
    ``torch.norm(Q, p=1)``, which for a 2-D tensor is the entrywise sum of
    magnitudes and is larger by up to a factor of the row count.

    Getting this wrong over-regularises Q, which enters the free energy through
    ``L(1) = spm_logdet(iS)*nq/2 - ...`` and cost ~124 nats against SPM12 on the
    N=2 spectral parity problem (DCCN job 55234464). It is also data-scale
    dependent, so it moves whenever the CSD is rescaled.

    Reconstructs the pre-regularisation Q independently and asserts the builder
    inverted it with the induced norm.
    """
    num_freqs, num_regions = 3, 3
    csd = _asymmetric_csd(num_freqs, num_regions)
    n = num_regions
    dim = num_freqs * n * n

    idx = torch.arange(dim)
    j_idx = idx % n
    i_idx = (idx // n) % n
    q_raw = torch.zeros(dim, dim, dtype=csd.dtype)
    for w_f in range(num_freqs):
        block = torch.arange(w_f * n * n, (w_f + 1) * n * n)
        i_b, j_b = i_idx[block], j_idx[block]
        csd_w = csd[w_f]
        q_raw[block.unsqueeze(1), block.unsqueeze(0)] = (
            csd_w[i_b][:, i_b] * csd_w[j_b][:, j_b]
        )

    induced = q_raw.abs().sum(dim=0).max()
    entrywise = torch.norm(q_raw, p=1).real

    # The fixture must actually discriminate between the two norms.
    assert entrywise > induced * 1.5, (
        f"fixture cannot discriminate the norms: entrywise={entrywise:.6e} "
        f"vs induced={induced:.6e}"
    )

    eye = torch.eye(dim, dtype=q_raw.dtype)
    expected = torch.linalg.inv(q_raw + induced * eye / 32)
    wrong = torch.linalg.inv(q_raw + entrywise * eye / 32)

    actual = compute_csd_precision(csd)[0][0]

    err_expected = (actual - expected).abs().max().item()
    err_wrong = (actual - wrong).abs().max().item()
    assert err_expected < 1e-8, (
        "compute_csd_precision did not use MATLAB's induced 1-norm "
        f"(max abs column sum). max|actual - expected| = {err_expected:.6e}, "
        f"but max|actual - entrywise_variant| = {err_wrong:.6e}. Expected the "
        f"induced norm {induced:.6e}, entrywise is {entrywise:.6e}."
    )
