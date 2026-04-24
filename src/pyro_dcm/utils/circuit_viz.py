"""DCM Circuit Explorer JSON serializer (``dcm_circuit_explorer/v1``).

Serializes a Pyro-DCM model config and/or fitted SVI posterior into the JSON
schema consumed by ``docs/dcm_circuit_explorer_template.html``. See
``docs/HANDOFF_viz.md`` for the authoritative schema and Phase 17 PLAN for
design decisions V1-V8.

The module exposes three surface-level objects:

- ``CircuitVizConfig`` -- dataclass mirroring the 13-field JSON schema plus a
  pass-through ``extras`` field (V1) for forward-compat keys such as
  ``_study``, ``node_info``, ``svg_edges``.
- ``CircuitViz`` -- factory class with ``from_model_config`` / ``from_posterior``
  / ``load`` static methods.
- ``flatten_posterior_for_viz`` -- module-level helper bridging
  ``extract_posterior_params`` output to the ``from_posterior`` input shape
  (V4; recommended but not auto-invoked).

References
----------
``docs/HANDOFF_viz.md`` -- authoritative schema source-of-truth.
``.planning/phases/17-circuit-visualization-module/17-01-PLAN.md`` -- locked
decisions V1-V8.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:  # pragma: no cover -- import guard for optional numpy support (V8).
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False


_FIRST_CLASS_KEYS: frozenset[str] = frozenset(
    {
        "_schema",
        "_status",
        "meta",
        "palette",
        "regions",
        "region_colors",
        "matrices",
        "mat_order",
        "phenotypes",
        "hypotheses",
        "drugs",
        "peb",
        "fitted_params",
    }
)
"""Exact 13 top-level JSON keys emitted by ``CircuitVizConfig.to_dict()``.

V1 decision: ``CircuitViz.load`` collects every top-level JSON key NOT in this
set into ``cfg.extras`` (``_study``, ``_description``, ``node_info``,
``node_positions``, ``svg_edges``, ``b_overlays``, etc.). ``to_dict`` then
merges ``extras`` after the first-class keys, with first-class keys winning
on any collision. This enables byte-level round-trip of configs authored
outside this module (notably ``configs/heart2adapt_dcm_config.json``).
"""


def _to_list_of_list(x: Any) -> list[list[float]]:
    """Coerce a matrix-like input to ``list[list[float]]`` (V8).

    Accepts ``torch.Tensor``, ``numpy.ndarray`` (if numpy installed), or a
    nested list/tuple of numbers. Any other type raises ``TypeError`` with an
    expected-vs-actual message.

    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray or list of list of float
        Matrix-like input. Must be 2-D; rows may have equal or unequal length
        (no shape validation here -- downstream JSON consumers enforce shape).

    Returns
    -------
    list of list of float
        Nested Python list with plain ``float``/``int`` values, suitable for
        ``json.dumps`` without custom encoders.

    Raises
    ------
    TypeError
        If ``x`` is not one of the supported types.
    """
    if isinstance(x, torch.Tensor):
        result: list[list[float]] = x.detach().cpu().tolist()
        return result
    if _NUMPY_AVAILABLE and isinstance(x, _np.ndarray):
        np_result: list[list[float]] = x.tolist()
        return np_result
    if isinstance(x, (list, tuple)):
        return [list(row) for row in x]
    raise TypeError(
        "Expected torch.Tensor, numpy.ndarray, or list[list[float]]; "
        f"got {type(x).__name__}"
    )


def _validate_no_nan(posterior_means: dict[str, list[list[float]]]) -> None:
    """Raise ``ValueError`` if any flat posterior matrix contains NaN/Inf (V6).

    Called by ``CircuitViz.from_posterior`` BEFORE ``copy.deepcopy`` and by
    ``CircuitVizConfig.export`` BEFORE JSON serialization. The error message
    pinpoints the offending ``(key, row, col)`` so callers can trace the
    failure back to a specific SVI site. Uses ``math.isnan``/``math.isinf``
    on Python floats (cheaper than a torch round-trip).

    Parameters
    ----------
    posterior_means : dict
        Mapping from matrix key (``'A'``, ``'B1'``, ...) to a
        ``list[list[float]]`` payload.

    Raises
    ------
    ValueError
        On the first non-finite cell encountered.
    """
    for key, mat in posterior_means.items():
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                # Allow ints (exact) and floats; only test floats for NaN/Inf.
                if isinstance(val, float):
                    if math.isnan(val):
                        raise ValueError(
                            f"posterior_means[{key!r}] contains NaN at "
                            f"[row={i}, col={j}]; expected finite float for "
                            "dcm_circuit_explorer/v1"
                        )
                    if math.isinf(val):
                        raise ValueError(
                            f"posterior_means[{key!r}] contains Inf at "
                            f"[row={i}, col={j}]; expected finite float for "
                            "dcm_circuit_explorer/v1"
                        )


@dataclass
class CircuitVizConfig:
    """Schema for the DCM Circuit Explorer JSON (``dcm_circuit_explorer/v1``).

    Mirrors the 13 first-class top-level keys documented in
    ``docs/HANDOFF_viz.md`` and adds a pass-through ``extras`` field (V1) for
    forward-compat keys such as ``_study``, ``_description``, ``node_info``,
    ``node_positions``, ``svg_edges``, ``b_overlays`` that the reference
    ``configs/heart2adapt_dcm_config.json`` contains but the renderer treats
    as optional.

    Notes
    -----
    JSON-key vs attribute-name mapping:

    - ``schema`` attribute -> ``_schema`` JSON key
    - ``status`` attribute -> ``_status`` JSON key
    - all other attribute names match their JSON keys verbatim.

    The ``fitted_params`` attribute is ``None`` in the planned state and set
    to a ``dict[str, list[list[float]]]`` after ``CircuitViz.from_posterior``
    attaches SVI means.

    Examples
    --------
    >>> cfg = CircuitVizConfig()
    >>> set(cfg.to_dict().keys()) == {
    ...     "_schema", "_status", "meta", "palette", "regions", "region_colors",
    ...     "matrices", "mat_order", "phenotypes", "hypotheses", "drugs", "peb",
    ...     "fitted_params",
    ... }
    True
    """

    schema: str = "dcm_circuit_explorer/v1"
    status: str = "planned"
    meta: dict[str, Any] = field(default_factory=dict)
    palette: dict[str, Any] = field(default_factory=dict)
    regions: list[str] = field(default_factory=list)
    region_colors: list[str] = field(default_factory=list)
    matrices: dict[str, Any] = field(default_factory=dict)
    mat_order: list[str] = field(default_factory=list)
    phenotypes: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    drugs: list[dict[str, Any]] = field(default_factory=list)
    peb: dict[str, Any] = field(default_factory=dict)
    fitted_params: dict[str, list[list[float]]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` with the 13 first-class keys + extras.

        The 13 first-class keys are emitted first; any additional keys present
        in ``self.extras`` are merged in AFTER, with first-class keys winning
        on any collision (V1). This guarantees round-trip preservation for
        configs authored outside this module.

        Returns
        -------
        dict
            JSON-serializable mapping suitable for ``json.dumps``.
        """
        first_class: dict[str, Any] = {
            "_schema": self.schema,
            "_status": self.status,
            "meta": self.meta,
            "palette": self.palette,
            "regions": self.regions,
            "region_colors": self.region_colors,
            "matrices": self.matrices,
            "mat_order": self.mat_order,
            "phenotypes": self.phenotypes,
            "hypotheses": self.hypotheses,
            "drugs": self.drugs,
            "peb": self.peb,
            "fitted_params": self.fitted_params,
        }
        # Merge extras last; first-class keys win on collision (V1).
        merged: dict[str, Any] = {
            **{k: v for k, v in self.extras.items() if k not in _FIRST_CLASS_KEYS},
            **first_class,
        }
        # Preserve the first-class-first key order by constructing fresh.
        out: dict[str, Any] = dict(first_class)
        for k, v in self.extras.items():
            if k not in _FIRST_CLASS_KEYS:
                out[k] = v
        # ``merged`` above is kept to make the merge semantics explicit in
        # code review; ``out`` is what we return (first-class keys first,
        # extras appended in insertion order).
        del merged
        return out

    def export(self, path: str | Path) -> Path:
        """Write the config JSON to ``path`` and return the path.

        Calls ``_validate_no_nan(self.fitted_params)`` FIRST when
        ``fitted_params is not None`` (V6 scope extension -- also applies to
        ``export`` in addition to ``from_posterior``). This ensures that a
        ``CircuitViz.load(cfg.export(path))`` round-trip never smuggles a
        non-finite float into the JSON.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        Path
            The destination path as a ``pathlib.Path``.

        Raises
        ------
        ValueError
            If ``fitted_params`` contains NaN or Inf (see
            ``_validate_no_nan``).
        """
        if self.fitted_params is not None:
            _validate_no_nan(self.fitted_params)
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


class CircuitViz:
    """Generate DCM Circuit Explorer JSON configs from Pyro-DCM state.

    Three static factory methods:

    - ``from_model_config(model_cfg, *, phenotypes, hypotheses, drugs, peb,
      palette)`` -- build a ``status='planned'`` config from a task_dcm model
      config dict. Matrix values may be ``torch.Tensor``, ``numpy.ndarray``,
      or pre-flattened lists (V8).
    - ``from_posterior(planned, posterior_means)`` -- deepcopy-based upgrade
      of a planned config to ``status='fitted'`` with attached posterior
      means. Validates no NaN/Inf before the deepcopy (V6).
    - ``load(path)`` -- read a ``dcm_circuit_explorer/v1`` JSON back into a
      ``CircuitVizConfig``. Preserves any top-level keys outside the 13-key
      first-class set via ``cfg.extras`` (V1).

    Companion module-level helper ``flatten_posterior_for_viz`` (V4) bridges
    ``extract_posterior_params`` output to the dict shape ``from_posterior``
    expects. It is the RECOMMENDED flatten path but is not auto-invoked:
    the caller retains control over the ``A`` vs ``A_free`` transform choice
    and the ``B`` (stacked) vs ``B_free_j`` (raw + parameterize_B) source
    choice.

    Usage
    -----
    Planned (before fitting):

    >>> cfg = CircuitViz.from_model_config(model_cfg)  # doctest: +SKIP
    >>> cfg.export("configs/my_study_planned.json")  # doctest: +SKIP

    Fitted (after SVI):

    >>> flat = flatten_posterior_for_viz(  # doctest: +SKIP
    ...     posterior, cfg.mat_order, b_masks=b_masks,
    ... )
    >>> fitted = CircuitViz.from_posterior(cfg, flat)  # doctest: +SKIP
    >>> fitted.export("configs/my_study_fitted.json")  # doctest: +SKIP
    """

    @staticmethod
    def from_model_config(
        model_cfg: dict[str, Any],
        *,
        phenotypes: list[dict[str, Any]] | None = None,
        hypotheses: list[dict[str, Any]] | None = None,
        drugs: list[dict[str, Any]] | None = None,
        peb: dict[str, Any] | None = None,
        palette: dict[str, Any] | None = None,
    ) -> CircuitVizConfig:
        """Build a planned-params ``CircuitVizConfig`` from a model config dict.

        Implements V1/V2/V7/V8 decisions:

        - V2: ``region_colors`` is REQUIRED in ``model_cfg`` and must match
          ``regions`` in length. Raises ``ValueError`` with expected-vs-actual
          message otherwise (no matplotlib / tab10 fallback).
        - V7: ``mat_order`` is always computed as
          ``['A'] + sorted(B_matrices.keys()) + (['C'] if C present else [])``,
          independent of caller dict insertion order.
        - V8: ``A_prior_mean``, ``B_matrices[key]``, and ``C_matrix`` may be
          ``torch.Tensor``, ``numpy.ndarray``, or ``list[list[float]]``. All
          are converted to ``list[list[float]]`` via ``_to_list_of_list``.

        Parameters
        ----------
        model_cfg : dict
            Must contain:

            - ``regions`` : ``list[str]``
            - ``region_colors`` : ``list[str]`` (length equal to ``regions``)
            - ``A_prior_mean`` : 2-D matrix-like (see V8)

            May contain:

            - ``B_matrices`` : ``dict[str, matrix-like]``
            - ``B_modulators`` : ``dict[str, dict]`` with optional
              ``label`` / ``color`` / ``modulator`` / ``modulator_display``
              entries. None-valued entries are OMITTED from the emitted
              sub-dict (the renderer uses falsy checks).
            - ``C_matrix`` : 2-D matrix-like. Only included in ``matrices`` /
              ``mat_order`` when non-empty (``len(...) > 0``).
            - ``C_inputs`` : ``list[str]`` (defaults to ``['u']`` when
              ``C_matrix`` is present but no inputs provided).
            - ``meta`` : ``dict`` (shallow-copied into the output).
            - ``peb_covariates`` : ``list[dict]``.
        phenotypes, hypotheses, drugs : list of dict, optional
            Passed through verbatim. Defaults to ``[]``.
        peb : dict, optional
            Overrides any ``peb_covariates`` mapping derived from ``model_cfg``
            (explicit > derived). Defaults to ``{}``.
        palette : dict, optional
            Semantic color-role mapping (distinct from ``region_colors``).
            Defaults to ``{}``.

        Returns
        -------
        CircuitVizConfig
            Config with ``status='planned'`` and ``fitted_params=None``.

        Raises
        ------
        ValueError
            If ``regions`` is missing, if ``region_colors`` is missing (V2),
            if ``len(region_colors) != len(regions)``, or if ``A_prior_mean``
            is missing.
        """
        if "regions" not in model_cfg:
            raise ValueError(
                "model_cfg missing required key 'regions'; "
                "expected list[str], got absent"
            )
        regions = list(model_cfg["regions"])
        if "region_colors" not in model_cfg:
            raise ValueError(
                "model_cfg missing required key 'region_colors'; expected "
                f"list[str] of length {len(regions)}, got absent (V2: no "
                "matplotlib fallback; caller must supply explicit colors)"
            )
        region_colors = list(model_cfg["region_colors"])
        if len(region_colors) != len(regions):
            raise ValueError(
                "region_colors length mismatch: expected "
                f"{len(regions)} (to match regions), got {len(region_colors)}"
            )
        if "A_prior_mean" not in model_cfg:
            raise ValueError(
                "model_cfg missing required key 'A_prior_mean'; expected "
                f"({len(regions)}, {len(regions)}) matrix-like, got absent"
            )

        matrices: dict[str, Any] = {}

        # A matrix first (always present).
        matrices["A"] = {
            "label": "A — Intrinsic",
            "color": "#374151",
            "vals": _to_list_of_list(model_cfg["A_prior_mean"]),
        }

        # B matrices in sort-deterministic key order (V7).
        b_matrices = model_cfg.get("B_matrices", {}) or {}
        b_modulators = model_cfg.get("B_modulators", {}) or {}
        b_keys_sorted = sorted(b_matrices.keys())
        for key in b_keys_sorted:
            mods = b_modulators.get(key, {}) or {}
            entry: dict[str, Any] = {}
            # Emit label/color/modulator/modulator_display only when the
            # modulator sub-dict supplied them (None/missing -> omit).
            for field_name in ("label", "color", "modulator", "modulator_display"):
                val = mods.get(field_name)
                if val is not None:
                    entry[field_name] = val
            entry["vals"] = _to_list_of_list(b_matrices[key])
            matrices[key] = entry

        # C matrix last, when present and non-empty.
        c_present = (
            "C_matrix" in model_cfg
            and model_cfg["C_matrix"] is not None
            and len(model_cfg["C_matrix"]) > 0
        )
        if c_present:
            matrices["C"] = {
                "label": "C — Driving",
                "color": "#6B7280",
                "vals": _to_list_of_list(model_cfg["C_matrix"]),
                "inputs": list(model_cfg.get("C_inputs", ["u"])),
            }

        # mat_order: V7 deterministic construction.
        mat_order: list[str] = ["A"] + b_keys_sorted + (["C"] if c_present else [])

        # peb: explicit kwarg overrides any value derived from model_cfg.
        peb_out: dict[str, Any] = {}
        if "peb_covariates" in model_cfg:
            peb_out["covariates"] = list(model_cfg["peb_covariates"])
        if peb is not None:
            peb_out.update(peb)

        return CircuitVizConfig(
            schema="dcm_circuit_explorer/v1",
            status="planned",
            meta=dict(model_cfg.get("meta", {}) or {}),
            palette=dict(palette or {}),
            regions=regions,
            region_colors=region_colors,
            matrices=matrices,
            mat_order=mat_order,
            phenotypes=list(phenotypes or []),
            hypotheses=list(hypotheses or []),
            drugs=list(drugs or []),
            peb=peb_out,
            fitted_params=None,
        )

    @staticmethod
    def from_posterior(
        planned: CircuitVizConfig,
        posterior_means: dict[str, list[list[float]]],
    ) -> CircuitVizConfig:
        """Attach posterior means to a planned config; switch status to fitted.

        Implements the handoff ``from_posterior`` verbatim (see
        ``docs/HANDOFF_viz.md`` lines 193-212) augmented with V6 NaN/Inf
        validation on ``posterior_means`` BEFORE ``copy.deepcopy`` so the
        error message pinpoints the offending cell before any state is
        duplicated.

        Parameters
        ----------
        planned : CircuitVizConfig
            The planned-state config (typically from ``from_model_config``).
            Not mutated.
        posterior_means : dict
            Mapping from matrix key (``'A'``, ``'B1'``, ...) to a
            ``list[list[float]]`` of posterior means. Typically produced by
            ``flatten_posterior_for_viz(...)``.

        Returns
        -------
        CircuitVizConfig
            A NEW config (deepcopy) with ``status='fitted'`` and
            ``fitted_params = posterior_means``.

        Raises
        ------
        ValueError
            If any cell in any matrix of ``posterior_means`` is NaN or Inf
            (V6; message includes the offending ``(key, row, col)``).
        """
        _validate_no_nan(posterior_means)
        fitted = copy.deepcopy(planned)
        fitted.status = "fitted"
        fitted.fitted_params = posterior_means
        return fitted

    @staticmethod
    def load(path: str | Path) -> CircuitVizConfig:
        """Load an existing JSON config back into a ``CircuitVizConfig``.

        Implements the handoff ``load`` verbatim (see ``docs/HANDOFF_viz.md``
        lines 214-232) augmented with V1 extras collection: every top-level
        JSON key NOT in ``_FIRST_CLASS_KEYS`` is preserved in ``cfg.extras``.
        This enables round-trip of configs authored outside this module
        (notably ``configs/heart2adapt_dcm_config.json``, which contains
        ``_study``, ``_description``, ``node_info``, ``node_positions``,
        ``svg_edges``, ``b_overlays``).

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.

        Returns
        -------
        CircuitVizConfig
            Reconstructed config.
        """
        data = json.loads(Path(path).read_text())
        cfg = CircuitVizConfig()
        cfg.schema = data.get("_schema", "dcm_circuit_explorer/v1")
        cfg.status = data.get("_status", "planned")
        cfg.meta = data.get("meta", {})
        cfg.palette = data.get("palette", {})
        cfg.regions = data.get("regions", [])
        cfg.region_colors = data.get("region_colors", [])
        cfg.matrices = data.get("matrices", {})
        cfg.mat_order = data.get("mat_order", [])
        cfg.phenotypes = data.get("phenotypes", [])
        cfg.hypotheses = data.get("hypotheses", [])
        cfg.drugs = data.get("drugs", [])
        cfg.peb = data.get("peb", {})
        cfg.fitted_params = data.get("fitted_params", None)
        # V1: collect every top-level key outside the first-class set.
        cfg.extras = {k: v for k, v in data.items() if k not in _FIRST_CLASS_KEYS}
        return cfg


def flatten_posterior_for_viz(
    posterior: dict[str, Any],
    mat_order: list[str],
    b_masks: list[torch.Tensor] | None = None,
) -> dict[str, list[list[float]]]:
    """Flatten ``extract_posterior_params`` output to ``from_posterior`` input.

    V4 decision: this helper is the RECOMMENDED bridge from SVI posterior
    output to the ``CircuitViz.from_posterior`` input shape, but is NOT
    auto-invoked -- the caller retains control over:

    - the ``A`` vs ``A_free`` source choice (prefers the deterministic ``A``
      site when present; falls back to ``parameterize_A(A_free)``)
    - the ``B`` (stacked) vs ``B_free_j`` (raw + ``parameterize_B``) source
      choice per modulator.

    Parameters
    ----------
    posterior : dict
        Output of ``pyro_dcm.models.guides.extract_posterior_params``, with
        per-site keys mapping to dicts containing at least a ``'mean'`` tensor.
    mat_order : list of str
        Matrix keys to flatten, typically ``planned.mat_order``. Supported
        patterns: ``'A'``, ``'C'``, and ``'B{j}'`` (e.g., ``'B0'``, ``'B1'``).
    b_masks : list of torch.Tensor, optional
        Per-modulator ``(N, N)`` masks required ONLY when a ``'B{j}'`` key is
        requested AND the deterministic ``'B'`` stacked site is absent from
        ``posterior``. Index convention: ``b_masks[j]`` corresponds to
        ``posterior['B_free_{j}']``.

    Returns
    -------
    dict
        Mapping ``key -> list[list[float]]`` for every ``key`` in
        ``mat_order``. Suitable for ``CircuitViz.from_posterior``.

    Raises
    ------
    ValueError
        If a requested ``B{j}`` key cannot be flattened because neither the
        ``'B'`` stacked site nor a ``'B_free_{j}'`` + ``b_masks[j]`` pair is
        available.

    Notes
    -----
    The numeric index ``j`` in a ``'B{j}'`` key must be a non-negative integer
    parseable by ``int(key[1:])``. Keys like ``'Bfoo'`` raise ``ValueError``.
    """
    from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B

    out: dict[str, list[list[float]]] = {}
    for key in mat_order:
        if key == "A":
            if "A" in posterior:
                mean = posterior["A"]["mean"]
                out["A"] = mean.detach().cpu().tolist()
            elif "A_free" in posterior:
                a_mean = parameterize_A(posterior["A_free"]["mean"])
                out["A"] = a_mean.detach().cpu().tolist()
            else:
                raise ValueError(
                    "flatten_posterior_for_viz: neither 'A' nor 'A_free' site "
                    f"found in posterior; available keys: {sorted(posterior)}"
                )
        elif key == "C":
            if "C" not in posterior:
                raise ValueError(
                    "flatten_posterior_for_viz: 'C' site not found in "
                    f"posterior; available keys: {sorted(posterior)}"
                )
            out["C"] = posterior["C"]["mean"].detach().cpu().tolist()
        elif key.startswith("B") and len(key) > 1:
            try:
                j = int(key[1:])
            except ValueError as exc:
                raise ValueError(
                    "flatten_posterior_for_viz: B-key must be 'B<int>' "
                    f"(got {key!r}); expected e.g. 'B0', 'B1', ..."
                ) from exc
            if "B" in posterior:
                b_mean = posterior["B"]["mean"][j]
                out[key] = b_mean.detach().cpu().tolist()
            else:
                free_key = f"B_free_{j}"
                if free_key not in posterior:
                    raise ValueError(
                        "flatten_posterior_for_viz: cannot flatten "
                        f"{key!r}; neither 'B' stacked site nor {free_key!r} "
                        f"found in posterior (available: {sorted(posterior)})"
                    )
                if b_masks is None:
                    raise ValueError(
                        "flatten_posterior_for_viz: b_masks is required when "
                        f"falling back to {free_key!r} + parameterize_B; got "
                        "None"
                    )
                free_mean = posterior[free_key]["mean"]
                b_j = parameterize_B(
                    free_mean.unsqueeze(0), b_masks[j].unsqueeze(0)
                ).squeeze(0)
                out[key] = b_j.detach().cpu().tolist()
        else:
            raise ValueError(
                "flatten_posterior_for_viz: unsupported mat_order key "
                f"{key!r}; expected 'A', 'C', or 'B<int>'"
            )
    return out
