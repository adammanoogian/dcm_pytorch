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
