"""Public 5-source auditory-MMN network + the actinf_physics CMC adapter (Phase 36).

Promotes the locked, parity-verified 5-source auditory mismatch-negativity (MMN)
topology -- previously a private fixture-only constant block in
:mod:`validation.export_to_mat` (the ``_MS_*`` edge lists) -- to a public,
importable network builder so the demo (Plan 36-03), the consumer adapter, and
the downstream ``actinf_physics`` Phase-133 hand-off all share ONE source of
truth instead of re-typing edge lists (a re-typed topology silently diverges from
the SPM12-parity-gated fixture, Phases 34-35).

Two public symbols, both forward-only (no fitting):

* :func:`build_mmn_5source_network` -- the canonical auditory MMN graph
  (A1 L/R -> STG L/R -> rIFG, forward / backward / lateral, C into bilateral A1,
  deviant-vs-standard B with a self-inhibition ``diag(B)`` at the precision nodes
  rIFG + bilateral A1). Returns presence masks; NO MNI coordinates (LFP scope --
  coords are a MUST-VERIFY deferral per Garrido et al. 2009 / Ranlund et al.
  2016, not load-bearing for the LFP demo).
* :func:`mmn_cmc_params` -- the thin ``(sp_inhibition_gain, a1_b_gain,
  rifg_b_gain, fwd_bwd_flag) -> CMC params`` map that turns the precision-sweep
  knobs into a ready-to-simulate forward-only bundle
  (``{p, a_masks, b_masks, c_mask, x_design, l_full}``) the
  :func:`pyro_dcm.simulators.simulate_erp_dcm` loop consumes directly. The
  ``sp_inhibition_gain`` knob drives the FREE ``P.G[:,0]`` at the precision nodes,
  which flows to the parameterised ``G[:,6]`` (sp self-inhibition) via the
  Phase-33 intrinsic permutation ``J_PERM[0] == 6`` -- NEVER index ``G[:,6]``
  directly (the permutation trap).

The topology is held byte-identical to the locked ``_MS_*`` reference by READING
those constants at build time (lazy import, keeping the package import light) --
the topology-equality test reconstructs the masks independently and asserts
element-wise.

References
----------
SPM12 ``spm_fx_cmc.m`` -- extrinsic forward / backward / lateral coupling blocks
and the ``(1 + 4L)`` reciprocal lateral reduction (``:68-82``), the four
extrinsic routes into the equations of motion (``:171-198``). ``spm_gen_Q.m`` --
the between-trial ``B`` -> all-``A`` folding and the ``diag(B) -> Q.G(:,1)``
precision path (``:47,65-67``). Garrido, M.I., Kilner, J.M., Stephan, K.E. &
Friston, K.J. (2009), "The mismatch negativity: a review of underlying
mechanisms", Clinical Neurophysiology 120, 453-463. Ranlund, S. et al. (2016),
"Impaired prefrontal synaptic gain in people with psychosis ...", Human Brain
Mapping 37, 351-365. Adams, R.A., Stephan, K.E., Brown, H.R., Frith, C.D. &
Friston, K.J. (2013), "The computational anatomy of psychosis", Frontiers in
Psychiatry 4, 47 (the precision / synaptic-gain interpretation of the swept
self-inhibition knob).
"""

from __future__ import annotations

import torch
from torch import Tensor

from pyro_dcm.forward_models.erp_leadfield import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
)

_F64 = torch.float64


def _ms_topology() -> dict[str, object]:
    """Read the locked ``_MS_*`` 5-source MMN topology constants (single source).

    Lazily imports the canonical edge lists / precision nodes / B values from
    :mod:`validation.export_to_mat` so the topology stays byte-identical to the
    SPM12-parity-gated fixture (Phases 34-35) and the package import does not pull
    in ``scipy`` at load time. NOTHING in ``validation.export_to_mat`` is mutated
    -- the locked masks are read only.

    Returns
    -------
    dict of str -> object
        Keys ``n`` (int), ``source_names`` (tuple[str, ...]), ``forward`` /
        ``lateral`` / ``backward`` ([to, from] edge tuples), ``inputs`` /
        ``precision`` (source-index tuples), ``b_edge`` / ``b_diag`` /
        ``a_live`` / ``a_dead`` (float scalars).
    """
    from validation.export_to_mat import (
        _MS_A_DEAD,
        _MS_A_LIVE,
        _MS_B_DIAG,
        _MS_B_EDGE,
        _MS_BACKWARD_EDGES,
        _MS_FORWARD_EDGES,
        _MS_INPUT_SOURCES,
        _MS_LATERAL_EDGES,
        _MS_N,
        _MS_PRECISION_NODES,
        _MS_SOURCE_NAMES,
    )

    return {
        "n": _MS_N,
        "source_names": _MS_SOURCE_NAMES,
        "forward": _MS_FORWARD_EDGES,
        "lateral": _MS_LATERAL_EDGES,
        "backward": _MS_BACKWARD_EDGES,
        "inputs": _MS_INPUT_SOURCES,
        "precision": _MS_PRECISION_NODES,
        "b_edge": _MS_B_EDGE,
        "b_diag": _MS_B_DIAG,
        "a_live": _MS_A_LIVE,
        "a_dead": _MS_A_DEAD,
    }


def _edge_mask(n: int, edges: tuple[tuple[int, int], ...]) -> Tensor:
    """Build a ``(n, n)`` binary presence mask from a ``[to, from]`` edge list."""
    mask = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in edges:
        mask[to_i, from_i] = 1.0
    return mask


def build_mmn_5source_network() -> dict[str, object]:
    """Build the canonical 5-source auditory-MMN CMC network (ERPDCM-03).

    Promotes the locked private ``_MS_*`` reference topology (sources, 0-indexed:
    ``A1L=0, A1R=1, STGL=2, STGR=3, rIFG=4``) from :mod:`validation.export_to_mat`
    to a public, importable builder. The graph (edges as ``[to, from]``, matching
    the ``cmc_network_f`` / ``spm_fx_cmc`` ``A[i] @ S`` routing,
    ``spm_fx_cmc.m:171-198``):

    * Forward (``A{1}`` sp->ss, ``A{2}`` sp->dp): A1L->STGL, A1R->STGR,
      STGL->rIFG, STGR->rIFG.
    * Lateral RECIPROCAL STGL<->STGR -- folded into BOTH forward blocks so the
      ``(1 + 4L)`` lateral reduction fires (``spm_fx_cmc.m:79-82``).
    * Backward (``A{3}`` dp->sp, ``A{4}`` dp->ii): rIFG->STGL, rIFG->STGR,
      STGL->A1L, STGR->A1R.
    * Input ``C`` drives bilateral A1 (A1L, A1R) only.
    * Deviant-vs-standard ``B``: ``b_edge`` (0.3) on every extrinsic edge and
      ``b_diag`` (0.5) on the self-inhibition ``diag(B)`` at the precision nodes
      rIFG + bilateral A1 (the MMN gain knob, ``spm_gen_Q.m:65-67``), with design
      ``x_design = [[0], [1]]`` (Cnd=2, n_effects=1; row 0 standard, row 1
      deviant).

    NO MNI coordinates are emitted: the LFP demo is head-model-free (Phase 35),
    so source coordinates are a MUST-VERIFY deferral (Garrido et al. 2009 /
    Ranlund et al. 2016) and are NOT load-bearing here.

    Returns
    -------
    dict of str -> object
        ``{"a_masks": list of 4 (5, 5) presence masks (forward, forward,
        backward, backward), "b_masks": [ (5, 5) B value matrix ], "c_mask":
        (5, 1) input presence mask, "x_design": (2, 1), "source_names":
        ("A1L", "A1R", "STGL", "STGR", "rIFG"), "precision_nodes": (4, 0, 1)}``.
        All tensors float64.
    """
    topo = _ms_topology()
    n = int(topo["n"])  # type: ignore[call-overload]
    forward: tuple[tuple[int, int], ...] = topo["forward"]  # type: ignore[assignment]
    lateral: tuple[tuple[int, int], ...] = topo["lateral"]  # type: ignore[assignment]
    backward: tuple[tuple[int, int], ...] = topo["backward"]  # type: ignore[assignment]
    inputs: tuple[int, ...] = topo["inputs"]  # type: ignore[assignment]
    precision: tuple[int, ...] = topo["precision"]  # type: ignore[assignment]
    b_edge = float(topo["b_edge"])  # type: ignore[arg-type]
    b_diag = float(topo["b_diag"])  # type: ignore[arg-type]

    # Forward blocks carry the bottom-up edges PLUS the reciprocal lateral pair
    # (so (1 + 4L) fires); backward blocks carry the top-down edges.
    fwd_mask = _edge_mask(n, forward + lateral)
    bwd_mask = _edge_mask(n, backward)
    a_masks = [fwd_mask.clone(), fwd_mask.clone(), bwd_mask.clone(), bwd_mask.clone()]

    # Deviant B: b_edge on every extrinsic edge, b_diag on the precision diag.
    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in forward + lateral + backward:
        b[to_i, from_i] = b_edge
    for node in precision:
        b[node, node] = b_diag

    c_mask = torch.zeros(n, 1, dtype=_F64)
    for src in inputs:
        c_mask[src, 0] = 1.0

    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)  # standard / deviant

    return {
        "a_masks": a_masks,
        "b_masks": [b],
        "c_mask": c_mask,
        "x_design": x_design,
        "source_names": tuple(topo["source_names"]),  # type: ignore[arg-type]
        "precision_nodes": precision,
    }


_VALID_FLAGS = ("forward", "backward", "both")


def mmn_cmc_params(
    sp_inhibition_gain: float,
    a1_b_gain: float,
    rifg_b_gain: float,
    fwd_bwd_flag: str = "both",
) -> dict[str, object]:
    """Map the MMN precision knobs to a ready-to-simulate CMC bundle (ERPDCM-05).

    The forward-only ``actinf_physics`` Phase-133 adapter: turns the four
    consumer-facing knobs into the free-parameter struct + lead field that
    :func:`pyro_dcm.simulators.simulate_erp_dcm` consumes verbatim. No fitting.

    Knob wiring:

    * ``sp_inhibition_gain`` -> sets the FREE ``P.G[node, 0] = sp_inhibition_gain``
      for ``node in {rIFG, A1L, A1R}`` (the precision nodes). This free column
      drives the parameterised ``G[:,6]`` (sp self-inhibition) via the Phase-33
      permutation ``J_PERM[0] == 6`` (``spm_fx_cmc.m:151``) -- it is the swept
      precision / synaptic-gain knob (Adams et al. 2013). NEVER index ``G[:,6]``
      directly (the permutation trap).
    * ``a1_b_gain`` -> the deviant ``diag(B)`` at A1L, A1R; ``rifg_b_gain`` -> the
      deviant ``diag(B)`` at rIFG (the between-trial precision modulation,
      ``spm_gen_Q.m:65-67``).
    * ``fwd_bwd_flag`` -> which extrinsic blocks carry the ``b_edge`` modulation
      (the Garrido/Ranlund model-space toggle): ``"forward"`` = the forward +
      lateral edges only, ``"backward"`` = the backward edges only, ``"both"`` =
      all extrinsic edges.

    The A / C blocks use the locked ``_MS_A_LIVE`` (0, on) / ``_MS_A_DEAD`` (-32,
    off) free-log convention; ``B`` carries additive values. The LFP lead field is
    the identity single-dipole map ``build_lead_field(cmc_default_pj(),
    lfp_spatial(ones(5), 5))`` (Phase 35).

    Parameters
    ----------
    sp_inhibition_gain : float
        Free ``P.G[:,0]`` value at the precision nodes (the swept gain).
    a1_b_gain : float
        Deviant ``diag(B)`` value at bilateral A1 (A1L, A1R).
    rifg_b_gain : float
        Deviant ``diag(B)`` value at rIFG.
    fwd_bwd_flag : str, optional
        One of ``{"forward", "backward", "both"}``. Default ``"both"``.

    Returns
    -------
    dict of str -> object
        ``{"p": free-param struct (keys T (5,4), G (5,4), C (5,1), S (5,1),
        R (1,2), A (list of 4 (5,5)), B ([ (5,5) ])), "a_masks": p["A"],
        "b_masks": p["B"], "c_mask": p["C"], "x_design": (2,1), "l_full":
        (Nc, 8*5) LFP lead field}``. Feed directly to
        ``simulate_erp_dcm(bundle["p"], bundle["x_design"], 5,
        l_full=bundle["l_full"])``.

    Raises
    ------
    ValueError
        If ``fwd_bwd_flag`` is not one of ``{"forward", "backward", "both"}``.
    """
    if fwd_bwd_flag not in _VALID_FLAGS:
        raise ValueError(
            "fwd_bwd_flag must be one of {'forward', 'backward', 'both'}; "
            f"expected one of {_VALID_FLAGS}, got {fwd_bwd_flag!r}"
        )

    net = build_mmn_5source_network()
    topo = _ms_topology()
    n = int(topo["n"])  # type: ignore[call-overload]
    forward: tuple[tuple[int, int], ...] = topo["forward"]  # type: ignore[assignment]
    lateral: tuple[tuple[int, int], ...] = topo["lateral"]  # type: ignore[assignment]
    backward: tuple[tuple[int, int], ...] = topo["backward"]  # type: ignore[assignment]
    precision: tuple[int, ...] = topo["precision"]  # type: ignore[assignment]
    a_live = float(topo["a_live"])  # type: ignore[arg-type]
    a_dead = float(topo["a_dead"])  # type: ignore[arg-type]
    b_edge = float(topo["b_edge"])  # type: ignore[arg-type]

    a_idx = {name: i for i, name in enumerate(net["source_names"])}  # type: ignore[arg-type]

    # A / C free-log: presence mask 1 -> _MS_A_LIVE (on), 0 -> _MS_A_DEAD (off).
    a_masks: list[Tensor] = net["a_masks"]  # type: ignore[assignment]
    a_free = [m * (a_live - a_dead) + a_dead for m in a_masks]
    c_mask: Tensor = net["c_mask"]  # type: ignore[assignment]
    c_free = c_mask * (a_live - a_dead) + a_dead

    # Deviant B: b_edge on the flag-selected extrinsic edges + diag gains.
    edges: tuple[tuple[int, int], ...] = ()
    if fwd_bwd_flag in ("forward", "both"):
        edges = edges + forward + lateral
    if fwd_bwd_flag in ("backward", "both"):
        edges = edges + backward
    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in edges:
        b[to_i, from_i] = b_edge
    b[a_idx["A1L"], a_idx["A1L"]] = a1_b_gain
    b[a_idx["A1R"], a_idx["A1R"]] = a1_b_gain
    b[a_idx["rIFG"], a_idx["rIFG"]] = rifg_b_gain

    # Free intrinsic G: sp_inhibition_gain on the FREE P.G[:,0] precision column
    # (-> parameterised G[:,6] via J_PERM[0]=6). Direct-G[:,6] indexing is wrong.
    g = torch.zeros(n, 4, dtype=_F64)
    for node in precision:
        g[node, 0] = sp_inhibition_gain

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
