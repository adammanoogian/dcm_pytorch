"""Visual 3-node V5/MT->SPL/IPS->PMd collision CMC network + actinf adapter.

Extends the frozen 2-node builder (:mod:`pyro_dcm.forward_models.collision_reference`)
to a THREE-node canonical-microcircuit (CMC) predictive-coding hierarchy, motivated
by two physical-inference neuroimaging papers that post-date the 2-node design:

* **Long, Wang, Yang & Chang (2026)** "Dynamic resource allocation orchestrates
  physical simulation in the human brain" (bioRxiv 10.64898/2026.06.21.732202).
  fMRI: bilateral **MT** encodes object motion direction (the sensorimotor,
  object-specific level); **relational** physical variables (relative angle,
  mass ratio) + collision occurrence engage more **anterior frontal-parietal /
  executive-control** cortex. MEG: a slow real-time simulation (occipital ~0.32 s
  -> frontal-parietal ~1.4 s) PLUS a fast **~700 ms pre-contact predictive**
  signal. The MEG millisecond timing is what licenses a distinct FRONTAL ERP node
  (the 2-node design omitted it for lack of timed evidence).
* **Zbaren, Kapur, Meissner & Wenderoth (2024)** "Inferring occluded projectile
  motion changes connectivity within a visuo-fronto-parietal network" (Brain Struct
  Funct, DOI 10.1007/s00429-024-02815-2). fMRI DCM (5 right-hemisphere nodes):
  the winning model modulates **top-down parietal->visual** + bidirectional
  parietal-premotor edges; the dorsomedial **SPL** hub drives visual + SMG in a
  top-down fashion during occluded inference. Supplies the directed-edge template
  and the parietal (SPL/IPS) + premotor (PMd) MNI anchors.

Nodes (0-indexed):

* node 0 = **V5/hMT+** -- sensory motion / launch-kinematics; carries the fitted
  ascending sensory precision ``kappa`` (the schizophrenia visual-gain hook).
  Long: MT = object motion direction. Region provenance: Fischer et al. 2016 (fMRI
  physics-network localizer -- the physics-inference network exists and is
  physics-specific) + Ahuja et al. 2024 (macaque fMRI: motion-sensitive V5/MT-like
  regions ACTIVATE during physical simulation even WITHOUT visual motion -> the
  empirical warrant for MT as the simulation-carrying bottom node); the ``kappa`` ->
  sp self-inhibition precision hook follows the Adams/Bastos lineage (precision =
  superficial-pyramidal gain). Coord (Kolster et al. 2010) right hMT+ ~ MNI [46, -78, 6].
* node 1 = **SPL/IPS (right superior parietal / intraparietal)** -- the relational
  / simulation hub; carries the fitted Newtonian-prior ``omega`` (belief precision).
  Zbaren DCM hub (SPL ~ MNI [20, -70, 50]; SMG/IPS ~ [56, -36, 52]); Long: relational
  variables + spatial working-memory tracking during occlusion. Region provenance:
  Zbaren 2024 (the fMRI-DCM directed-edge hub) + Pramod et al. 2025 (the physics
  network decodes PREDICTED collisions / future states -- the forward-simulation this
  DCM instantiates) + Schwettmann et al. 2019 (invariant physical-property coding in
  the same parietal network).
* node 2 = **PMd (right dorsal premotor / frontal)** -- the higher-order relational
  / predictive-control node; the Long ~700 ms early-predictive + late (~1.4 s)
  frontal-parietal simulation locus; the Zbaren PMd/FEF premotor arm. Region
  provenance: Long et al. 2026 (MEG ms-timing -- the ONLY anchor that licenses a
  distinct FRONTAL ERP node) + Ahuja & Desrochers 2026 (macaque LPFC monotonic
  RAMPING selective to simulation -> reframes this node from static "predictive
  control" to a sequential-monitoring engine that steps a hidden trajectory forward,
  predicting a blunted occlusion-window ramp in SZ). Coord ~ MNI [30, 0, 52].

METHOD-HONESTY CAVEAT (do NOT overclaim ERP timing from fMRI): Fischer 2016,
Pramod 2025, Schwettmann 2019, and the macaque JOCN/CurBiol work (Ahuja 2024/2026)
are **fMRI** -- they establish spatial identity + functional connectivity ONLY, with
NO millisecond timing. Only **Long et al. 2026 (MEG)** licenses the ERP time-course;
Adams lineage is computational/DCM (the intrinsic-gain mechanism). The fMRI anchors
motivate node identity + edges; they are NOT evidence for ERP latency.

Hierarchy (edges as ``[to, from]``, SPM convention ``A{to-row, from-col}``,
matching ``cmc_network_f`` / ``spm_fx_cmc`` routing):

* Forward (``A{1}`` sp->ss, ``A{2}`` sp->dp) -- ascending precision-weighted
  prediction error: V5/MT -> SPL ``[1, 0]``, SPL -> PMd ``[2, 1]``.
* Backward (``A{3}`` dp->sp, ``A{4}`` dp->ii) -- descending top-down prediction
  (the active-inference ``w_topdown`` edges; Zbaren's empirically-modulated
  top-down parietal->visual direction): SPL -> V5/MT ``[0, 1]``, PMd -> SPL
  ``[1, 2]``.
* NO lateral edge (feedforward/feedback chain only, unilateral right).
* Driving input ``C`` enters V5/MT (node 0, granular) only; SPL and PMd are driven
  indirectly via the forward chain.
* Between-trial design = **expected vs violated launch** (the Michotte angle/delay
  contrast), ``x_design = [[0], [1]]`` (row 0 expected, row 1 violated). The
  violated condition raises ascending PE on the forward edges and the precision-node
  ``diag(B)`` at all three nodes.

The per-node precision knob is the CMC superficial-pyramidal **self-inhibition**
(an INVERSE gain): the free ``P.G[node, 0]`` drives the parameterised ``G[:,6]``
(sp->sp self-inhibition) via the intrinsic permutation ``J_PERM[0] == 6``
(``spm_fx_cmc.m:151``) -- NEVER index ``G[:,6]`` directly (the permutation trap).
The validated sweep is monotone non-increasing: a HIGHER self-inhibition value ->
LOWER net sp gain -> SMALLER evoked prediction error. This builder is SIGN-NEUTRAL:
it exposes the per-node self-inhibition knob faithfully; the ``kappa`` -> knob
INVERSION (high kappa -> low self-inhibition value -> large PE) is applied on the
actinf adapter side, not here.

The LFP readout is the identity single-dipole lead field
``build_lead_field(cmc_default_pj(), lfp_spatial(ones(3), 3))`` reading the
superficial-pyramidal voltage (state column index 2 via ``cmc_default_pj``; NEVER
index 6 / dp_V). All free-log A/C + between-trial B scalar conventions are inherited
verbatim from the locked :mod:`validation.export_to_mat` constants (the same source
the MMN + 2-node collision builders read), so only the topology (node set + edge
directions) differs from the SPM12-parity-gated 2-node clone.

References
----------
SPM12 ``spm_fx_cmc.m`` (extrinsic forward/backward coupling + ``J_PERM`` intrinsic
permutation ``:151``); ``spm_gen_Q.m`` (between-trial ``B`` -> all-``A`` folding +
``diag(B) -> Q.G(:,1)`` precision path ``:65-67``). Node set + edge template from
Long et al. 2026 (MEG timing) + Zbaren et al. 2024 (DCM directed edges); the 2-node
provenance is documented in the Phase-133.1 RESEARCH note.

Zotero checklist (commented; NEVER edit the .bib -- the user does the Zotero pass;
real keys in prose ONLY after the entry is confirmed; do NOT invent citation keys):

  [ ] Long, Wang, Yang & Chang (2026) "Dynamic resource allocation orchestrates
      physical simulation in the human brain" bioRxiv. DOI 10.64898/2026.06.21.732202
      [MT=object motion; relational->frontoparietal; ~700 ms early predictive; MEG timing]
  [ ] Zbaren, Kapur, Meissner & Wenderoth (2024) "Inferring occluded projectile
      motion changes connectivity within a visuo-fronto-parietal network" Brain
      Struct Funct. DOI 10.1007/s00429-024-02815-2   [DCM directed edges; SPL/PMd MNI]
  [ ] Bastos, Usrey, Adams, Mangun, Fries & Friston (2012) "Canonical microcircuits
      for predictive coding" Neuron 76:695-711. DOI 10.1016/j.neuron.2012.10.038
  [ ] Adams, Stephan, Brown, Frith & Friston (2013) "The computational anatomy of
      psychosis" Front Psychiatry 4:47. DOI 10.3389/fpsyt.2013.00047
  [ ] Kolster, Peeters & Orban (2010) "Retinotopic organization of human MT/V5 and
      neighbors" J Neurosci 30(29):9801-9820. DOI 10.1523/JNEUROSCI.2069-10.2010
  [ ] Fischer, Mikhael, Tenenbaum & Kanwisher (2016) "Functional neuroanatomy of
      intuitive physical inference" PNAS 113(34):E5072-E5081. DOI 10.1073/pnas.1610344113
      [fMRI physics-network localizer; node identity ONLY, no ms-timing]
  [ ] Pramod, Mieczkowski, Fang, Tenenbaum & Kanwisher (2025) "The physics network
      decodes predicted collisions and future physical states" Sci Adv.
      DOI 10.1126/sciadv.adr7429   [fMRI; forward-simulation / predicted collisions]
  [ ] Ahuja, Yusif Rodriguez, Sheinberg & Desrochers (2026) "LPFC sequential
      monitoring scaffolds visual simulation (Planko)" JOCN 38(7):1297-1306.
      DOI 10.1162/jocn.a.2584   [MACAQUE fMRI; LPFC ramp = the frontal-node algorithm]
  [ ] Ahuja, Yusif Rodriguez, Karkada Ashok, Serre, Desrochers & Sheinberg (2024)
      "Visual simulation in macaques" Curr Biol 34(24):5635-5645.e3.
      DOI 10.1016/j.cub.2024.10.026   [MACAQUE fMRI; MT-like activation w/o visual motion]
  [ ] Schwettmann, Tenenbaum & Kanwisher (2019) "Invariant representations of mass"
      eLife 8:e46619. DOI 10.7554/eLife.46619   [fMRI; parietal physical-property coding]
  [ ] Kiebel, David & Friston (2006) "DCM of evoked responses in EEG/MEG with lead
      field parameterization" NeuroImage 30:1273-1284. DOI 10.1016/j.neuroimage.2005.12.055
# METHOD-HONESTY: Fischer/Pramod/Schwettmann/Ahuja are fMRI (spatial + connectivity
# only); ONLY Long 2026 (MEG) licenses ERP ms-timing; Adams lineage = DCM mechanism.
# NEVER edit the .bib -- the user does the Zotero pass; real keys in prose ONLY after
# each entry is confirmed present. Do NOT invent citation keys.
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

# Visual 3-node topology (0-indexed): V5/MT = 0, SPL/IPS = 1, PMd = 2.
_C3_SOURCE_NAMES: tuple[str, ...] = ("V5/MT", "SPL/IPS", "PMd")
_C3_N = 3
# Forward (ascending PE): MT -> SPL [1, 0], SPL -> PMd [2, 1].
_C3_FORWARD_EDGES: tuple[tuple[int, int], ...] = ((1, 0), (2, 1))
# Backward (descending prediction): SPL -> MT [0, 1], PMd -> SPL [1, 2].
_C3_BACKWARD_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2))
_C3_LATERAL_EDGES: tuple[tuple[int, int], ...] = ()  # feedforward/feedback chain.
_C3_INPUT_SOURCES: tuple[int, ...] = (0,)  # C drives V5/MT only.
_C3_PRECISION_NODES: tuple[int, ...] = (0, 1, 2)  # all nodes carry diag(B) + a knob.

_VALID_FLAGS = ("forward", "backward", "both")

#: Right-hemisphere MNI anchors (documentation only; the LFP-identity readout is
#: head-model-free, so coords are non-load-bearing until/unless an ECD montage is
#: added). V5/MT: Kolster et al. 2010. SPL / PMd: Zbaren et al. 2024 DCM.
C3_MNI_COORDS: dict[str, tuple[int, int, int]] = {
    "V5/MT": (46, -78, 6),
    "SPL/IPS": (20, -70, 50),
    "PMd": (30, 0, 52),
}


def _collision_scalars() -> dict[str, float]:
    """Read the locked free-log + B-value scalar conventions (single source).

    Lazily imports the SPM12-parity-gated scalar constants from
    :mod:`validation.export_to_mat` -- the SAME constants the MMN and 2-node
    collision builders read -- so the 3-node clone inherits the free-log A/C
    convention (``_MS_A_LIVE`` = 0 on, ``_MS_A_DEAD`` = -32 off) and the
    between-trial ``B`` edge/diag values verbatim. NOTHING in
    ``validation.export_to_mat`` is mutated.

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


def _edge_mask(n: int, edges: tuple[tuple[int, int], ...]) -> Tensor:
    """Build a ``(n, n)`` binary presence mask from a ``[to, from]`` edge list."""
    mask = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in edges:
        mask[to_i, from_i] = 1.0
    return mask


def build_collision_3node_network() -> dict[str, object]:
    """Build the 3-node V5/MT->SPL/IPS->PMd visual-collision CMC network.

    Mirrors :func:`pyro_dcm.forward_models.build_collision_2node_network` but for
    the three-node hierarchy (``V5/MT = 0``, ``SPL/IPS = 1``, ``PMd = 2``). The
    graph (edges as ``[to, from]``, matching the ``cmc_network_f`` / ``spm_fx_cmc``
    ``A[i] @ S`` routing):

    * Forward (``A{1}`` sp->ss, ``A{2}`` sp->dp): V5/MT -> SPL ``[1, 0]``,
      SPL -> PMd ``[2, 1]``.
    * Backward (``A{3}`` dp->sp, ``A{4}`` dp->ii): SPL -> V5/MT ``[0, 1]``,
      PMd -> SPL ``[1, 2]``.
    * NO lateral edge (feedforward/feedback chain).
    * Input ``C`` drives V5/MT (node 0) only.
    * Expected-vs-violated ``B``: ``b_edge`` on every extrinsic edge and ``b_diag``
      on the self-inhibition ``diag(B)`` at all three precision nodes, with design
      ``x_design = [[0], [1]]`` (Cnd=2; row 0 expected, row 1 violated).

    NO MNI coordinates are emitted here (LFP-identity readout is head-model-free);
    the right-hemisphere anchors are documented in :data:`C3_MNI_COORDS`.

    Returns
    -------
    dict of str -> object
        ``{"a_masks": list of 4 (3, 3) presence masks (forward, forward, backward,
        backward), "b_masks": [ (3, 3) B value matrix ], "c_mask": (3, 1) input
        presence mask, "x_design": (2, 1), "source_names": ("V5/MT", "SPL/IPS",
        "PMd"), "precision_nodes": (0, 1, 2)}``. All tensors float64.
    """
    scalars = _collision_scalars()
    n = _C3_N
    b_edge = scalars["b_edge"]
    b_diag = scalars["b_diag"]

    fwd_mask = _edge_mask(n, _C3_FORWARD_EDGES + _C3_LATERAL_EDGES)
    bwd_mask = _edge_mask(n, _C3_BACKWARD_EDGES)
    a_masks = [fwd_mask.clone(), fwd_mask.clone(), bwd_mask.clone(), bwd_mask.clone()]

    # Violated B: b_edge on every extrinsic edge, b_diag on the precision diag.
    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in (
        _C3_FORWARD_EDGES + _C3_LATERAL_EDGES + _C3_BACKWARD_EDGES
    ):
        b[to_i, from_i] = b_edge
    for node in _C3_PRECISION_NODES:
        b[node, node] = b_diag

    c_mask = torch.zeros(n, 1, dtype=_F64)
    for src in _C3_INPUT_SOURCES:
        c_mask[src, 0] = 1.0

    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)  # expected / violated

    return {
        "a_masks": a_masks,
        "b_masks": [b],
        "c_mask": c_mask,
        "x_design": x_design,
        "source_names": _C3_SOURCE_NAMES,
        "precision_nodes": _C3_PRECISION_NODES,
    }


def collision_3node_cmc_params(
    v5_sp_gain: float,
    parietal_sp_gain: float,
    frontal_sp_gain: float,
    violation_b_gain: float,
    fwd_bwd_flag: str = "both",
) -> dict[str, object]:
    """Map the 3-node precision knobs to a ready-to-simulate CMC bundle.

    The forward-only actinf adapter: turns the per-node sp self-inhibition knobs +
    the violated-condition modulation into the free-parameter struct + lead field
    that :func:`pyro_dcm.simulators.simulate_erp_dcm` consumes verbatim. No fitting.

    Knob wiring:

    * ``v5_sp_gain`` -> FREE ``P.G[0, 0]`` at V5/MT (node 0); ``parietal_sp_gain``
      -> ``P.G[1, 0]`` at SPL/IPS (node 1); ``frontal_sp_gain`` -> ``P.G[2, 0]`` at
      PMd (node 2). This free column 0 drives the parameterised ``G[:,6]`` (sp
      self-inhibition) via the intrinsic permutation ``J_PERM[0] == 6``
      (``spm_fx_cmc.m:151``) -- the precision / synaptic-gain knob. NEVER index
      ``G[:,6]`` directly (the permutation trap). The knob is self-inhibition (an
      INVERSE gain): a higher value lowers the net sp gain, shrinking the evoked
      prediction error.
    * ``violation_b_gain`` -> the violated-condition ``diag(B)`` value at ALL three
      precision nodes; one shared knob (as in the 2-node builder).
    * ``fwd_bwd_flag`` -> which extrinsic blocks carry the ``b_edge`` modulation:
      ``"forward"`` = the forward edges only, ``"backward"`` = the backward edges
      only, ``"both"`` = all extrinsic edges.

    Parameters
    ----------
    v5_sp_gain : float
        Free ``P.G[0,0]`` sp self-inhibition value at V5/MT (node 0).
    parietal_sp_gain : float
        Free ``P.G[1,0]`` sp self-inhibition value at SPL/IPS (node 1).
    frontal_sp_gain : float
        Free ``P.G[2,0]`` sp self-inhibition value at PMd (node 2).
    violation_b_gain : float
        Violated-condition ``diag(B)`` value at all three precision nodes.
    fwd_bwd_flag : str, optional
        One of ``{"forward", "backward", "both"}``. Default ``"both"``.

    Returns
    -------
    dict of str -> object
        ``{"p": free-param struct (keys T (3,4), G (3,4), C (3,1), S (3,1),
        R (1,2), A (list of 4 (3,3)), B ([ (3,3) ])), "a_masks": p["A"],
        "b_masks": p["B"], "c_mask": p["C"], "x_design": (2,1), "l_full":
        (Nc, 8*3) LFP lead field}``. Feed directly to
        ``simulate_erp_dcm(bundle["p"], bundle["x_design"], 3,
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

    net = build_collision_3node_network()
    scalars = _collision_scalars()
    n = _C3_N
    a_live = scalars["a_live"]
    a_dead = scalars["a_dead"]
    b_edge = scalars["b_edge"]

    # A / C free-log: presence mask 1 -> _MS_A_LIVE (on), 0 -> _MS_A_DEAD (off).
    a_masks: list[Tensor] = net["a_masks"]  # type: ignore[assignment]
    a_free = [m * (a_live - a_dead) + a_dead for m in a_masks]
    c_mask: Tensor = net["c_mask"]  # type: ignore[assignment]
    c_free = c_mask * (a_live - a_dead) + a_dead

    # Violated B: b_edge on the flag-selected extrinsic edges + the shared diag gain.
    edges: tuple[tuple[int, int], ...] = ()
    if fwd_bwd_flag in ("forward", "both"):
        edges = edges + _C3_FORWARD_EDGES + _C3_LATERAL_EDGES
    if fwd_bwd_flag in ("backward", "both"):
        edges = edges + _C3_BACKWARD_EDGES
    b = torch.zeros(n, n, dtype=_F64)
    for to_i, from_i in edges:
        b[to_i, from_i] = b_edge
    for node in _C3_PRECISION_NODES:
        b[node, node] = violation_b_gain

    # Free intrinsic G: per-node sp self-inhibition on the FREE P.G[:,0] precision
    # column (-> parameterised G[:,6] via J_PERM[0]=6). Direct-G[:,6] is wrong.
    g = torch.zeros(n, 4, dtype=_F64)
    g[0, 0] = v5_sp_gain
    g[1, 0] = parietal_sp_gain
    g[2, 0] = frontal_sp_gain

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
