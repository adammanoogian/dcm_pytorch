"""Shared CMC network-builder core for the MMN + collision topology modules.

Factors the byte-identical construction logic out of the three public network
builders (:mod:`pyro_dcm.forward_models.mmn_reference`,
:mod:`~pyro_dcm.forward_models.collision_reference`,
:mod:`~pyro_dcm.forward_models.collision_3node_reference`), which previously
re-typed the same ``_edge_mask`` / free-log ``A``-``C`` expansion / between-trial
``B`` folding / lead-field tail. The three builders now differ ONLY in their
topology (node set + edge directions) and their per-node gain mapping; everything
mechanical lives here, so a change to the SPM-parity convention can no longer
drift between the auditory and visual builders.

The free-log ``A``/``C`` convention (``_MS_A_LIVE`` = 0 on, ``_MS_A_DEAD`` = -32
off) and the representative between-trial ``B`` edge/diag values are read from the
locked :mod:`validation.export_to_mat` ``_MS_*`` constants (single source of
truth, SPM12-parity-gated in Phases 34-35) via :func:`read_ms_scalars`.

References
----------
SPM12 ``spm_fx_cmc.m`` -- extrinsic forward / backward / lateral coupling blocks
and the four extrinsic routes into the equations of motion (``:171-198``);
``spm_gen_Q.m`` -- the between-trial ``B`` -> all-``A`` folding and the
``diag(B) -> Q.G(:,1)`` precision path (``:47,65-67``); ``spm_cmc_priors.m`` --
the ``P.C`` mean ``mask*32-32`` free-log convention (``:114-116``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pyro_dcm.forward_models.erp_leadfield import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
)

_F64 = torch.float64

#: Accepted ``fwd_bwd_flag`` model-space toggles (Garrido/Ranlund forward-vs-backward).
VALID_FLAGS = ("forward", "backward", "both")

_Edges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CmcTopology:
    """A between-trial CMC topology: node set + directed extrinsic edges.

    Edges are ``[to, from]`` index pairs matching the SPM ``A{to-row, from-col}``
    routing (``cmc_network_f`` / ``spm_fx_cmc``). Forward and lateral edges populate
    the two forward ``A`` blocks; backward edges populate the two backward blocks.

    Attributes
    ----------
    n : int
        Number of sources.
    source_names : tuple of str
        Human-readable source labels, length ``n``.
    forward, lateral, backward : tuple of (int, int)
        ``[to, from]`` extrinsic edge lists. ``lateral`` is folded into BOTH forward
        blocks so the ``(1 + 4L)`` reciprocal-lateral reduction fires
        (``spm_fx_cmc.m:79-82``); empty for chain/2-node hierarchies.
    inputs : tuple of int
        Source indices the driving input ``C`` enters (granular layer).
    precision : tuple of int
        Source indices carrying the self-inhibition ``diag(B)`` precision knob.
    """

    n: int
    source_names: tuple[str, ...]
    forward: _Edges
    lateral: _Edges
    backward: _Edges
    inputs: tuple[int, ...]
    precision: tuple[int, ...]


def read_ms_scalars() -> dict[str, float]:
    """Read the locked free-log + between-trial ``B`` scalars (single source).

    Lazily imports the SPM12-parity-gated scalar constants from
    :mod:`validation.export_to_mat` so every builder inherits the free-log ``A``/``C``
    convention (``_MS_A_LIVE`` = 0 on, ``_MS_A_DEAD`` = -32 off) and the
    representative between-trial ``B`` edge/diag values verbatim. The lazy import
    keeps the package import light (no ``scipy`` at load time) and mutates NOTHING in
    ``validation.export_to_mat`` -- the locked constants are read only.

    Returns
    -------
    dict of str -> float
        Keys ``a_live`` / ``a_dead`` (free-log on/off) and ``b_edge`` / ``b_diag``
        (the representative between-trial values).
    """
    from validation.export_to_mat import (
        _MS_A_DEAD,
        _MS_A_LIVE,
        _MS_B_DIAG,
        _MS_B_EDGE,
    )

    return {
        "a_live": float(_MS_A_LIVE),
        "a_dead": float(_MS_A_DEAD),
        "b_edge": float(_MS_B_EDGE),
        "b_diag": float(_MS_B_DIAG),
    }


def edge_mask(n: int, edges: _Edges) -> Tensor:
    """Build a ``(n, n)`` binary presence mask from a ``[to, from]`` edge list."""
    mask = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in edges:
        mask[to_i, from_i] = 1.0
    return mask


def validate_flag(fwd_bwd_flag: str) -> None:
    """Raise ``ValueError`` (expected vs actual) if the model-space flag is unknown.

    Parameters
    ----------
    fwd_bwd_flag : str
        Must be one of :data:`VALID_FLAGS`.

    Raises
    ------
    ValueError
        If ``fwd_bwd_flag`` is not one of ``{"forward", "backward", "both"}``.
    """
    if fwd_bwd_flag not in VALID_FLAGS:
        raise ValueError(
            "fwd_bwd_flag must be one of {'forward', 'backward', 'both'}; "
            f"expected one of {VALID_FLAGS}, got {fwd_bwd_flag!r}"
        )


def _selected_extrinsic_edges(topo: CmcTopology, fwd_bwd_flag: str) -> _Edges:
    """Extrinsic edges carrying the ``b_edge`` modulation for the given flag."""
    edges: _Edges = ()
    if fwd_bwd_flag in ("forward", "both"):
        edges = edges + topo.forward + topo.lateral
    if fwd_bwd_flag in ("backward", "both"):
        edges = edges + topo.backward
    return edges


def build_network(topo: CmcTopology, b_edge: float, b_diag: float) -> dict[str, object]:
    """Build the presence-mask network dict for a between-trial CMC topology.

    The forward ``A`` blocks carry the forward + lateral edges (so the ``(1 + 4L)``
    lateral reduction fires); the backward blocks carry the backward edges. The
    between-trial ``B`` places ``b_edge`` on every extrinsic edge and ``b_diag`` on
    the self-inhibition ``diag(B)`` at the precision nodes (``spm_gen_Q.m:65-67``).

    Parameters
    ----------
    topo : CmcTopology
        Node set + directed extrinsic edges.
    b_edge, b_diag : float
        Representative between-trial edge / precision-diagonal values.

    Returns
    -------
    dict of str -> object
        ``{"a_masks": list of 4 (n, n) presence masks (forward, forward, backward,
        backward), "b_masks": [ (n, n) B value matrix ], "c_mask": (n, 1) input
        presence mask, "x_design": (2, 1), "source_names": tuple[str, ...],
        "precision_nodes": tuple[int, ...]}``. All tensors float64.
    """
    n = topo.n
    fwd_mask = edge_mask(n, topo.forward + topo.lateral)
    bwd_mask = edge_mask(n, topo.backward)
    a_masks = [fwd_mask.clone(), fwd_mask.clone(), bwd_mask.clone(), bwd_mask.clone()]

    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in topo.forward + topo.lateral + topo.backward:
        b[to_i, from_i] = b_edge
    for node in topo.precision:
        b[node, node] = b_diag

    c_mask = torch.zeros(n, 1, dtype=_F64)
    for src in topo.inputs:
        c_mask[src, 0] = 1.0

    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)  # standard / deviant

    return {
        "a_masks": a_masks,
        "b_masks": [b],
        "c_mask": c_mask,
        "x_design": x_design,
        "source_names": topo.source_names,
        "precision_nodes": topo.precision,
    }


def cmc_params_from_knobs(
    topo: CmcTopology,
    net: dict[str, object],
    scalars: dict[str, float],
    g_gains: dict[int, float],
    b_diag_gains: dict[int, float],
    fwd_bwd_flag: str,
) -> dict[str, object]:
    """Assemble a ready-to-simulate CMC bundle from per-node precision knobs.

    Shared body of the three ``*_cmc_params`` adapters. Expands the presence masks
    into the free-log ``A``/``C`` blocks, places the flag-selected ``b_edge``
    modulation plus the per-node violated ``diag(B)`` gains, writes the per-node
    superficial-pyramidal self-inhibition onto the FREE ``P.G[node, 0]`` column
    (which drives the parameterised ``G[:,6]`` via the Phase-33 permutation
    ``J_PERM[0] == 6`` -- NEVER index ``G[:,6]`` directly), and appends the identity
    single-dipole LFP lead field.

    Parameters
    ----------
    topo : CmcTopology
        The topology whose edges select the ``b_edge`` placement.
    net : dict
        Output of :func:`build_network` for ``topo`` (supplies ``a_masks``,
        ``c_mask``, ``x_design``).
    scalars : dict of str -> float
        Free-log + ``b_edge`` scalars from :func:`read_ms_scalars`.
    g_gains : dict of int -> float
        Free ``P.G[node, 0]`` sp self-inhibition value per source index.
    b_diag_gains : dict of int -> float
        Violated-condition ``diag(B)`` value per source index.
    fwd_bwd_flag : str
        One of :data:`VALID_FLAGS`; selects which extrinsic blocks carry ``b_edge``.

    Returns
    -------
    dict of str -> object
        ``{"p": free-param struct (keys T (n,4), G (n,4), C (n,1), S (n,1),
        R (n_inp,2), A (list of 4 (n,n)), B ([ (n,n) ])), "a_masks": p["A"],
        "b_masks": p["B"], "c_mask": p["C"], "x_design": (2,1), "l_full":
        (Nc, 8*n) LFP lead field}``.

    Raises
    ------
    ValueError
        If ``fwd_bwd_flag`` is not one of :data:`VALID_FLAGS`.
    """
    validate_flag(fwd_bwd_flag)

    n = topo.n
    a_live = scalars["a_live"]
    a_dead = scalars["a_dead"]
    b_edge = scalars["b_edge"]

    # A / C free-log: presence mask 1 -> _MS_A_LIVE (on), 0 -> _MS_A_DEAD (off).
    a_masks: list[Tensor] = net["a_masks"]  # type: ignore[assignment]
    a_free = [m * (a_live - a_dead) + a_dead for m in a_masks]
    c_mask: Tensor = net["c_mask"]  # type: ignore[assignment]
    c_free = c_mask * (a_live - a_dead) + a_dead

    # Between-trial B: b_edge on the flag-selected extrinsic edges + per-node diag.
    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in _selected_extrinsic_edges(topo, fwd_bwd_flag):
        b[to_i, from_i] = b_edge
    for node, gain in b_diag_gains.items():
        b[node, node] = gain

    # Free intrinsic G: sp self-inhibition on the FREE P.G[:,0] precision column
    # (-> parameterised G[:,6] via J_PERM[0]=6). Direct-G[:,6] indexing is wrong.
    g = torch.zeros(n, 4, dtype=_F64)
    for node, gain in g_gains.items():
        g[node, 0] = gain

    t = torch.zeros(n, 4, dtype=_F64)
    s = torch.zeros(n, 1, dtype=_F64)
    r = torch.zeros(c_free.shape[1], 2, dtype=_F64)

    p: dict[str, object] = {
        "T": t,
        "G": g,
        "C": c_free,
        "S": s,
        "R": r,
        "A": a_free,
        "B": [b],
    }

    l_spatial = lfp_spatial(torch.ones(n, dtype=_F64), n)
    l_full = build_lead_field(cmc_default_pj(), l_spatial)

    return {
        "p": p,
        "a_masks": a_free,
        "b_masks": [b],
        "c_mask": c_free,
        "x_design": net["x_design"],
        "l_full": l_full,
    }
