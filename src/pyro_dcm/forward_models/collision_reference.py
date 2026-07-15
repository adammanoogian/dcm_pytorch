"""Visual 2-node V5/MT<->rIPC collision CMC network + the actinf adapter (Phase 133.1).

Clones the auditory-MMN builder pattern (:mod:`pyro_dcm.forward_models.mmn_reference`)
to a VISUAL predictive-coding hierarchy restricted to two canonical-microcircuit
(CMC) nodes:

* node 0 = **V5/hMT+** -- the high-order visual motion area encoding launch
  kinematics (egress angle, delay, velocity); the sensory-precision (kappa) site
  and the schizophrenia visual-gain hook.
* node 1 = **rIPC (right inferior parietal / intraparietal)** -- the
  launching-causality / relational-launch level that issues the top-down
  prediction; the Straube-group SSD causality-perception tDCS target.

The hierarchy (edges as ``[to, from]``, SPM convention ``A{to-row, from-col}``,
matching ``cmc_network_f`` / ``spm_fx_cmc`` routing):

* Forward (``A{1}`` sp->ss, ``A{2}`` sp->dp): V5/MT -> rIPC -- the ascending,
  precision-weighted prediction error (gated by the V5/MT superficial-pyramidal
  gain = sensory precision kappa).
* Backward (``A{3}`` dp->sp, ``A{4}`` dp->ii): rIPC -> V5/MT -- the descending
  top-down prediction (the active-inference ``w_topdown`` edge).
* NO lateral edge (2-node, unilateral right).
* Driving input ``C`` enters V5/MT (node 0, granular) only; rIPC is driven
  indirectly via the forward connection.
* Between-trial design = **expected vs violated launch** (the Michotte angle/delay
  contrast replacing the oddball deviant), ``x_design = [[0], [1]]`` (row 0
  expected/standard, row 1 violated/deviant). The violated condition raises the
  ascending prediction error on the forward edge and the precision-node
  ``diag(B)`` at both nodes (both are precision nodes, like the MMN A1/rIFG nodes).

The per-node precision knob is the CMC superficial-pyramidal **self-inhibition**
(an INVERSE gain): the free ``P.G[node, 0]`` drives the parameterised ``G[:,6]``
(sp->sp self-inhibition) via the intrinsic permutation ``J_PERM[0] == 6``
(``spm_fx_cmc.m:151``) -- NEVER index ``G[:,6]`` directly (the permutation trap).
The validated MMN sweep is monotone non-increasing: a HIGHER self-inhibition value
-> LOWER net sp gain -> SMALLER evoked prediction error / difference wave. This
builder is SIGN-NEUTRAL: it exposes the per-node self-inhibition knob faithfully;
the kappa->knob INVERSION (high kappa -> low self-inhibition value -> large PE) is
applied on the actinf adapter side, not here.

The LFP readout is the identity single-dipole lead field
``build_lead_field(cmc_default_pj(), lfp_spatial(ones(2), 2))`` reading the
superficial-pyramidal voltage (state column index 2 via ``cmc_default_pj``; NEVER
index 6 / dp_V).

The free-log A/C convention (``_MS_A_LIVE`` = 0 / ``_MS_A_DEAD`` = -32) and the
between-trial ``B`` edge/diag values are read from the locked
:mod:`validation.export_to_mat` constants (the same source the MMN builder reads),
so the visual clone inherits the SPM12-parity-disciplined conventions verbatim;
only the topology (node set + edge directions) differs.

References
----------
SPM12 ``spm_fx_cmc.m`` -- extrinsic forward / backward coupling blocks and the
``J_PERM`` intrinsic-gain permutation (``:151``); ``spm_gen_Q.m`` -- the
between-trial ``B`` -> all-``A`` folding and the ``diag(B) -> Q.G(:,1)`` precision
path (``:65-67``). The visual node set + param-map signs are documented in the
Phase-133.1 RESEARCH note.

Zotero checklist (commented; NEVER edit the .bib -- the user does the Zotero pass;
real keys in prose ONLY after the entry is confirmed; do NOT invent citation keys):

  [ ] Bastos, Usrey, Adams, Mangun, Fries & Friston (2012) "Canonical microcircuits
      for predictive coding" Neuron 76:695-711. DOI 10.1016/j.neuron.2012.10.038
  [ ] Adams, Stephan, Brown, Frith & Friston (2013) "The computational anatomy of
      psychosis" Front Psychiatry 4:47. DOI 10.3389/fpsyt.2013.00047
  [ ] Kolster, Peeters & Orban (2010) "Retinotopic organization of human MT/V5 and
      neighbors" J Neurosci 30(29):9801-9820. DOI 10.1523/JNEUROSCI.2069-10.2010
      [V5/hMT+ MNI coords]
  [ ] Straube & Chatterjee (2010) "Space and time in perceptual causality"
      Front Hum Neurosci 4:28. DOI 10.3389/fnhum.2010.00028   [rIPC causality]
  [ ] Straube, Wolk & Chatterjee (2012) "Neural correlates of causality judgment in
      physical and social context" NeuroImage. DOI 10.1016/j.neuroimage.2012.07.012
  [ ] Streiling, Schuelke, Straube et al. (2025) "Choice- and trial-history effects
      on causality perception in Schizophrenia Spectrum Disorder" Schizophrenia
      (Nature). DOI 10.1038/s41537-025-00614-0
  [ ] Kiebel, David & Friston (2006) "DCM of evoked responses in EEG/MEG with lead
      field parameterization" NeuroImage 30:1273-1284.
      DOI 10.1016/j.neuroimage.2005.12.055   [lead field]
"""

from __future__ import annotations

from pyro_dcm.forward_models._cmc_network import (
    CmcTopology,
    build_network,
    cmc_params_from_knobs,
    read_ms_scalars,
)

# Visual 2-node topology (0-indexed): V5/MT = 0, rIPC = 1.
# Forward V5/MT -> rIPC = [to=1, from=0]; backward rIPC -> V5/MT = [to=0, from=1].
# No lateral edge (2-node, unilateral right); C drives V5/MT only; both nodes are
# precision nodes carrying diag(B).
_TOPO = CmcTopology(
    n=2,
    source_names=("V5/MT", "rIPC"),
    forward=((1, 0),),
    lateral=(),
    backward=((0, 1),),
    inputs=(0,),
    precision=(0, 1),
)


def build_collision_2node_network() -> dict[str, object]:
    """Build the 2-node V5/MT<->rIPC visual-collision CMC network (Phase 133.1).

    Mirrors :func:`pyro_dcm.forward_models.build_mmn_5source_network` but for the
    visual 2-node hierarchy (``V5/MT = 0``, ``rIPC = 1``). The graph (edges as
    ``[to, from]``, matching the ``cmc_network_f`` / ``spm_fx_cmc`` ``A[i] @ S``
    routing):

    * Forward (``A{1}`` sp->ss, ``A{2}`` sp->dp): V5/MT -> rIPC (edge ``[1, 0]``).
    * Backward (``A{3}`` dp->sp, ``A{4}`` dp->ii): rIPC -> V5/MT (edge ``[0, 1]``).
    * NO lateral edge (2-node).
    * Input ``C`` drives V5/MT (node 0) only.
    * Expected-vs-violated ``B``: ``b_edge`` on every extrinsic edge and ``b_diag``
      on the self-inhibition ``diag(B)`` at both precision nodes (V5/MT, rIPC),
      with design ``x_design = [[0], [1]]`` (Cnd=2; row 0 expected, row 1
      violated).

    NO MNI coordinates are emitted (LFP-identity readout is head-model-free; coords
    are an ECD-only deferral).

    Returns
    -------
    dict of str -> object
        ``{"a_masks": list of 4 (2, 2) presence masks (forward, forward, backward,
        backward), "b_masks": [ (2, 2) B value matrix ], "c_mask": (2, 1) input
        presence mask, "x_design": (2, 1), "source_names": ("V5/MT", "rIPC"),
        "precision_nodes": (0, 1)}``. All tensors float64.
    """
    scalars = read_ms_scalars()
    return build_network(_TOPO, scalars["b_edge"], scalars["b_diag"])


def collision_cmc_params(
    v5_sp_gain: float,
    ipc_sp_gain: float,
    violation_b_gain: float,
    fwd_bwd_flag: str = "both",
) -> dict[str, object]:
    """Map the visual-collision precision knobs to a ready-to-simulate CMC bundle.

    The forward-only actinf adapter (Phase 133.1): turns the per-node sp
    self-inhibition knobs + the violated-condition modulation into the
    free-parameter struct + lead field that
    :func:`pyro_dcm.simulators.simulate_erp_dcm` consumes verbatim. No fitting.

    Knob wiring:

    * ``v5_sp_gain`` -> sets the FREE ``P.G[0, 0] = v5_sp_gain`` at V5/MT (node 0);
      ``ipc_sp_gain`` -> sets the FREE ``P.G[1, 0] = ipc_sp_gain`` at rIPC (node 1).
      This free column 0 drives the parameterised ``G[:,6]`` (sp self-inhibition)
      via the intrinsic permutation ``J_PERM[0] == 6`` (``spm_fx_cmc.m:151``) -- the
      precision / synaptic-gain knob. NEVER index ``G[:,6]`` directly (the
      permutation trap). The knob is self-inhibition (an INVERSE gain): a higher
      value lowers the net sp gain, shrinking the evoked prediction error.
    * ``violation_b_gain`` -> the violated-condition ``diag(B)`` value at BOTH
      precision nodes (V5/MT and rIPC); one shared knob for the 2-node v1.
    * ``fwd_bwd_flag`` -> which extrinsic blocks carry the ``b_edge`` modulation:
      ``"forward"`` = the forward edge only, ``"backward"`` = the backward edge
      only, ``"both"`` = all extrinsic edges.

    The A / C blocks use the locked ``_MS_A_LIVE`` (0, on) / ``_MS_A_DEAD`` (-32,
    off) free-log convention; ``B`` carries additive values. The LFP lead field is
    the identity single-dipole map ``build_lead_field(cmc_default_pj(),
    lfp_spatial(ones(2), 2))`` reading the superficial-pyramidal voltage (col 2).

    Parameters
    ----------
    v5_sp_gain : float
        Free ``P.G[0,0]`` sp self-inhibition value at V5/MT (node 0).
    ipc_sp_gain : float
        Free ``P.G[1,0]`` sp self-inhibition value at rIPC (node 1).
    violation_b_gain : float
        Violated-condition ``diag(B)`` value at both precision nodes.
    fwd_bwd_flag : str, optional
        One of ``{"forward", "backward", "both"}``. Default ``"both"``.

    Returns
    -------
    dict of str -> object
        ``{"p": free-param struct (keys T (2,4), G (2,4), C (2,1), S (2,1),
        R (1,2), A (list of 4 (2,2)), B ([ (2,2) ])), "a_masks": p["A"],
        "b_masks": p["B"], "c_mask": p["C"], "x_design": (2,1), "l_full":
        (Nc, 8*2) LFP lead field}``. Feed directly to
        ``simulate_erp_dcm(bundle["p"], bundle["x_design"], 2,
        l_full=bundle["l_full"])``.

    Raises
    ------
    ValueError
        If ``fwd_bwd_flag`` is not one of ``{"forward", "backward", "both"}``.
    """
    net = build_collision_2node_network()
    scalars = read_ms_scalars()
    g_gains = {0: v5_sp_gain, 1: ipc_sp_gain}
    b_diag_gains = {node: violation_b_gain for node in _TOPO.precision}
    return cmc_params_from_knobs(
        _TOPO, net, scalars, g_gains, b_diag_gains, fwd_bwd_flag
    )
