"""CSD observation precision matching SPM12's spm_dcm_csd_Q.

Computes data-driven Wishart observation precision from cross-spectral
densities, following Camba-Mendez & Kapetanios (2005). The precision
matrix encodes frequency-specific covariance structure for the
vectorised spectral densities.
"""

from __future__ import annotations

import torch


def compute_csd_precision(
    observed_csd: torch.Tensor,
) -> tuple[list[torch.Tensor], int]:
    """Compute observation precision Q from CSD, matching spm_dcm_csd_Q.

    Implements the asymptotic precision for complex cross-spectra:
    Q(Qi, Qj) = CSD(w, i(Qi), i(Qj)) * CSD(w, j(Qi), j(Qj))
    only when w(Qi) == w(Qj) (block-diagonal in frequency), then
    regularizes via Q = inv(Q + ||Q||_1 * I / 32).

    Parameters
    ----------
    observed_csd : torch.Tensor
        Observed cross-spectral density, shape (F, N, N), complex128.
        For cell-array input (list of CSDs), pass the average CSD.

    Returns
    -------
    Q : list[torch.Tensor]
        List of precision component matrices. For fMRI spectral DCM,
        this is a single matrix of shape (F*N^2, F*N^2), complex128.
    nq : int
        Kronecker multiplier: for single-trial fMRI this is 1.
    """
    F, N, _ = observed_csd.shape
    Qn = F * N * N
    device = observed_csd.device

    # Build index arrays matching MATLAB's ind2sub([F, N, N], 1:Qn)
    # MATLAB orders: w varies fastest, then i, then j (column-major)
    # ind2sub([F, N, N], k) for k=1..Qn gives (w, i, j) in MATLAB
    # column-major order: w cycles 1..F, i cycles 1..N, j cycles 1..N
    idx = torch.arange(Qn, device=device)
    w = idx % F           # frequency index (fastest)
    i = (idx // F) % N    # row index
    j = idx // (F * N)    # column index (slowest)

    # Build Q block-by-block per frequency (Q is block-diagonal in freq)
    # For each frequency w_f, the block is of size (N^2, N^2)
    # Block(Qi_local, Qj_local) = CSD(w_f, i_Qi, i_Qj) * CSD(w_f, j_Qi, j_Qj)
    # where local indices run over all (i, j) pairs at that frequency
    Q = torch.zeros(Qn, Qn, dtype=observed_csd.dtype, device=device)

    for w_f in range(F):
        # Indices in the global Q that belong to this frequency
        # In column-major layout: positions w_f, w_f + F, w_f + 2F, ...
        block_idx = torch.arange(N * N, device=device)
        global_idx = w_f + block_idx * F  # map local -> global

        # Extract the (i, j) indices for each position in this freq block
        i_block = i[global_idx]  # shape (N^2,)
        j_block = j[global_idx]  # shape (N^2,)

        # CSD at this frequency: shape (N, N)
        csd_w = observed_csd[w_f]  # (N, N)

        # Q_block(a, b) = csd_w[i_a, i_b] * csd_w[j_a, j_b]
        # Vectorized: outer products of CSD rows/cols
        csd_i = csd_w[i_block]  # (N^2, N) -- rows indexed by i
        csd_j = csd_w[j_block]  # (N^2, N) -- rows indexed by j

        # csd_i[a, :] selects row i_a of csd_w
        # We need csd_w[i_a, i_b] = csd_i[a][i_b]
        # So the (a, b) element is csd_i[a, i_block[b]] * csd_j[a, j_block[b]]
        block = csd_i[:, i_block] * csd_j[:, j_block]  # (N^2, N^2)

        # Place into global Q
        Q[global_idx.unsqueeze(1), global_idx.unsqueeze(0)] = block

    # Regularize and invert: Q = inv(Q + ||Q||_1 * I / 32)
    norm_1 = torch.norm(Q, p=1).real if Q.is_complex() else torch.norm(Q, p=1)
    Q_reg = Q + norm_1 * torch.eye(Qn, dtype=Q.dtype, device=device) / 32
    Q_inv = torch.linalg.inv(Q_reg)

    # Single precision component for fMRI (nh=1)
    # nq = ny / len(Q[0]) -- for single trial, ny = Qn, nq = 1
    return [Q_inv], 1
