---
phase: 17
name: Circuit Visualization Module
researched: 2026-04-24
confidence: MEDIUM-HIGH
---

## Summary

Phase 17 ships `src/pyro_dcm/utils/circuit_viz.py` — a pure-Python serializer that
emits `dcm_circuit_explorer/v1` JSON consumed by
`docs/dcm_circuit_explorer_template.html`. The handoff (`docs/HANDOFF_viz.md`,
301 lines) already specifies the `CircuitVizConfig` dataclass verbatim, fully
implements `from_posterior()` and `load()`, and leaves only
`from_model_config()` to build. The acceptance surface is deliberately
*serialization-only* (schema validity, round-trip, planned↔fitted toggle wiring)
— **no RECOV-style posterior-recovery gates apply** per the user's plan-phase
directive.

The work is small-but-risky in exactly one dimension: the handoff dataclass and
the template HTML disagree on which fields exist. The handoff ships fields the
renderer ignores (`palette`, fine-grained `peb` structure) and the renderer
reads fields the handoff omits (`node_positions`, `svg_edges`, `b_overlays`,
bare `fitted_params[key]` arrays vs. any richer structure). Plans must pick a
single source of truth for the schema. Recommendation below: treat the handoff
dataclass as authoritative for the Python API surface, and treat the template's
optional-field set (`node_positions`, `svg_edges`, `b_overlays`) as
pass-through kwargs the user may supply but `from_model_config` need not
synthesize.

**Primary recommendation:** Implement exactly the handoff skeleton, add a
`_validate_matrices_shape()` preflight, emit `mat_order` deterministically from
`model_cfg["B_matrices"]` keys (sorted), default `phenotypes/hypotheses/drugs/
peb` to empty collections so a bare 3-region bilinear DCM produces a
renderer-valid JSON, and gate acceptance on: (i) `json.loads(json.dumps(x.to_dict()))
== x.to_dict()`, (ii) the reference config `heart2adapt_dcm_config.json`
round-trips through `load() → to_dict() → json.dumps`, (iii) `from_posterior`
flips `_status` to `"fitted"` and populates `fitted_params`.

## Class Interface (from HANDOFF_viz.md)

Verbatim from `docs/HANDOFF_viz.md:97-233`. The dataclass and the `from_posterior`
/ `load` methods are fully specified; only `from_model_config` is a stub:

```python
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class CircuitVizConfig:
    """Schema for the DCM Circuit Explorer JSON.

    Mirrors the JSON schema in docs/HANDOFF_viz.md.
    Populated by CircuitViz.from_model() or CircuitViz.from_posterior().
    """
    schema: str = "dcm_circuit_explorer/v1"
    status: str = "planned"          # "planned" | "fitted"
    meta: dict = field(default_factory=dict)
    palette: dict = field(default_factory=dict)
    regions: list[str] = field(default_factory=list)
    region_colors: list[str] = field(default_factory=list)
    matrices: dict = field(default_factory=dict)
    mat_order: list[str] = field(default_factory=list)
    phenotypes: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    drugs: list[dict] = field(default_factory=list)
    peb: dict = field(default_factory=dict)
    fitted_params: dict | None = None

    def to_dict(self) -> dict:
        return {
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

    def export(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


class CircuitViz:
    @staticmethod
    def from_model_config(
        model_cfg: dict,
        *,
        phenotypes: list[dict] | None = None,
        hypotheses: list[dict] | None = None,
        drugs: list[dict] | None = None,
        peb: dict | None = None,
    ) -> CircuitVizConfig:
        """Build a planned-params config from a task_dcm model config dict.

        model_cfg must contain:
          - "regions": list[str]
          - "region_colors": list[str]
          - "A_prior_mean": list[list[float]]    shape (N, N)
          - "B_matrices": dict[str, list[list[float]]]  e.g. {"B1": ..., "B2": ...}
          - "C_matrix": list[list[float]]         shape (N, M)
          - "B_modulators": dict[str, dict]  {key: {label, color, modulator,
                                                    modulator_display}}
          - "C_inputs": list[str]
          - "meta": dict  (title, subtitle, tags)
          - "peb_covariates": list[dict]  [{name, display, desc}]
        """
        ...  # implement

    @staticmethod
    def from_posterior(
        planned: CircuitVizConfig,
        posterior_means: dict[str, list[list[float]]],
    ) -> CircuitVizConfig:
        import copy
        fitted = copy.deepcopy(planned)
        fitted.status = "fitted"
        fitted.fitted_params = posterior_means
        return fitted

    @staticmethod
    def load(path: str | Path) -> CircuitVizConfig:
        data = json.loads(Path(path).read_text())
        cfg = CircuitVizConfig()
        cfg.schema        = data.get("_schema", "dcm_circuit_explorer/v1")
        cfg.status        = data.get("_status", "planned")
        cfg.meta          = data.get("meta", {})
        cfg.palette       = data.get("palette", {})
        cfg.regions       = data.get("regions", [])
        cfg.region_colors = data.get("region_colors", [])
        cfg.matrices      = data.get("matrices", {})
        cfg.mat_order     = data.get("mat_order", [])
        cfg.phenotypes    = data.get("phenotypes", [])
        cfg.hypotheses    = data.get("hypotheses", [])
        cfg.drugs         = data.get("drugs", [])
        cfg.peb           = data.get("peb", {})
        cfg.fitted_params = data.get("fitted_params", None)
        return cfg
```

**Method signatures the planner should lock:**

- `CircuitViz.from_model_config(model_cfg: dict, *, phenotypes=None, hypotheses=None, drugs=None, peb=None) -> CircuitVizConfig`
- `CircuitViz.from_posterior(planned: CircuitVizConfig, posterior_means: dict[str, list[list[float]]]) -> CircuitVizConfig`
- `CircuitViz.load(path: str | Path) -> CircuitVizConfig`
- `CircuitVizConfig.to_dict() -> dict`
- `CircuitVizConfig.export(path: str | Path) -> Path`

Save and to_html methods are NOT in the handoff; do not add them.

## Field Mapping: Pyro-DCM → dcm_circuit_explorer/v1

Column `Source` describes the authoritative origin. "USER" = must be passed in by
the caller; `from_model_config` cannot synthesize it from Pyro-DCM state alone.

| JSON field | Source | Required? (renderer) | Notes |
|---|---|---|---|
| `_schema` | constant `"dcm_circuit_explorer/v1"` | informational | Renderer never reads; for future-compat. |
| `_status` | `"planned"` from `from_model_config`, `"fitted"` from `from_posterior` | informational | Renderer never reads `_status`; it only checks `fitted_params` truthiness (`template.html:418`) to decide whether to show the toggle. |
| `meta.title` | `model_cfg["meta"]["title"]` | optional (defaults `"DCM Circuit Explorer"`, `template.html:408`) | |
| `meta.subtitle` | `model_cfg["meta"]["subtitle"]` | optional (defaults `""`) | |
| `meta.tags[]` | `model_cfg["meta"]["tags"]` | optional (hidden if absent, `template.html:413-414`) | Only the first TWO tags rendered. |
| `palette` | USER (optional; for custom node colors beyond `region_colors`) | optional (defaults `{}`, `template.html:462`) | NOT in handoff's stated `model_cfg` schema; pass-through. `from_model_config` should accept `palette` via `model_cfg.get("palette", {})` — **keyword absent from handoff spec, flag as open question**. |
| `regions[]` | `model_cfg["regions"]` | **REQUIRED** — empty list breaks SVG + matrix rendering | list of N strings |
| `region_colors[]` | `model_cfg["region_colors"]` | optional (falls back to `#888888`, `template.html:464`) | length must equal `len(regions)` if provided |
| `matrices.A.label` | literal `"A — Intrinsic"` (suggested; or `model_cfg.get("A_label")`) | optional | |
| `matrices.A.color` | literal `"#374151"` (suggested) | optional (defaults `#374151`, `template.html:894`) | |
| `matrices.A.vals` | `model_cfg["A_prior_mean"]` as `list[list[float]]` shape `(N,N)` | **REQUIRED** when `"A"` is in `mat_order` | Planned = prior mean. See note below on `parameterize_A` transform. |
| `matrices.A.note` | USER or omit | optional | Renderer does not read `.note`. |
| `matrices.B{j}.vals` | `model_cfg["B_matrices"][key]` as `list[list[float]]` shape `(N,N)` | required when key in `mat_order` | Planned = prior mean (typically all-zeros for a free B-matrix since `B_free ~ N(0, 1)`). |
| `matrices.B{j}.label` | `model_cfg["B_modulators"][key]["label"]` | optional | Used verbatim in sidebar + matrix block header. |
| `matrices.B{j}.color` | `model_cfg["B_modulators"][key]["color"]` | optional | |
| `matrices.B{j}.modulator` | `model_cfg["B_modulators"][key]["modulator"]` | optional | Renderer doesn't read this field directly. |
| `matrices.B{j}.modulator_display` | `model_cfg["B_modulators"][key]["modulator_display"]` | optional-but-important | Presence of `modulator_display` is the renderer's **discriminator** for "this is a B-matrix modulator box" (`template.html:504`). Without it, the top modulator strip is suppressed. |
| `matrices.C.vals` | `model_cfg["C_matrix"]` as `list[list[float]]` shape `(N,M)` | optional | If absent, the C box in the modulator strip is skipped (`template.html:525`). Matrix block in bottom panel is also skipped if `"C"` is not in `mat_order` or vals are empty. |
| `matrices.C.inputs[]` | `model_cfg["C_inputs"]` | optional (defaults `["u"]`, `template.html:898`) | Column headers for C-matrix grid. |
| `matrices.C.label` | literal `"C — Driving"` | optional | |
| `matrices.C.color` | literal `"#6B7280"` | optional | |
| `mat_order[]` | Derived: `["A"] + sorted(B_matrices.keys()) + (["C"] if C_matrix else [])` | optional (falls back to `Object.keys(mats)`, `template.html:831`) | **Recommendation:** always set explicitly to guarantee deterministic ordering across Python versions; dict order is insertion-ordered since 3.7 so `from_model_config` must insert A, then Bs in sorted-key order, then C. |
| `phenotypes[]` | USER kwarg (HEART2ADAPT-specific) | optional (section hidden if empty, `template.html:769`) | Default to `[]` for generic DCMs. |
| `hypotheses[]` | USER kwarg (HEART2ADAPT-specific) | optional (section hidden if empty, `template.html:825`) | Default to `[]`. |
| `drugs[]` | USER kwarg (HEART2ADAPT-specific) | optional (section hidden if empty, `template.html:799`) | Default to `[]`. |
| `peb.description` | `peb` kwarg | optional (block omitted if absent, `template.html:730`) | |
| `peb.covariates[]` | `peb` kwarg, or derived from `model_cfg["peb_covariates"]` | optional | Each entry: `{name, display, desc}`. |
| `fitted_params` | `None` from `from_model_config`; `dict[str, list[list[float]]]` from `from_posterior` | optional | When non-null truthy, `template.html:418` shows the Planned/Fitted toggle. Keys MUST match `mat_order` keys for matrix overlay to work. |
| `node_positions[]` | USER kwarg (optional) | optional (defaults to 4-node HEART2ADAPT layout, `template.html:452-457`) | **NOT in handoff dataclass.** Each entry: `{cx, cy, r, role}`. For N ≠ 4 or non-HEART2ADAPT geometries the default layout is wrong; user must override. Flag as open question. |
| `svg_edges[]` | USER kwarg (optional) | optional (falls back to hardcoded HEART2ADAPT SVG, `template.html:534-548`) | **NOT in handoff dataclass.** Each: `{id, d, stroke, sw, dash, lx, ly, label, target_region, opacity}`. For non-HEART2ADAPT layouts the fallback draws 4-region circuit regardless of actual regions — visible visual bug without user override. |
| `b_overlays` | USER (optional) | optional (defaults `{}`, `template.html:552`) | **NOT in handoff dataclass.** Dict of SVG group specs. |
| `node_info` | USER (optional) | **NEVER read by renderer** (`template.html` has 0 references to `node_info`) | Present in `configs/heart2adapt_dcm_config.json:320-349` but purely documentation. `CircuitViz` should accept it pass-through on `load()` but not expose it in the dataclass — or document as a `palette`-like pass-through dict. **Handoff does not mention `node_info`; flag as open question.** |

### Important note on `A_prior_mean` semantics for Planned view

`task_dcm_model` parameterizes A via `parameterize_A` (`neural_state.py:24`):

- `A[i,i] = -exp(A_free[i,i]) / 2` → at prior mean `E[A_free]=0`,
  `A[i,i] = -0.5` (Hz).
- `A[i,j] = A_free[i,j]` off-diagonal → at prior mean, `A[i,j] = 0`.

The HEART2ADAPT reference config (`configs/heart2adapt_dcm_config.json:32-37`)
uses domain-knowledge *expected* A values (−0.50 on diagonal, +0.30 etc.
off-diagonal), NOT the raw Pyro prior mean. This means `model_cfg["A_prior_mean"]`
is a display-layer convenience — the **user-supplied expected A** if they want
planned-mode to show non-trivial values — NOT a mechanical extraction from the
Pyro prior. For a bare 3-region bilinear DCM with default priors, `A_prior_mean
= [[-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5]]` is the correct serialization of the
Pyro prior mean under parameterize_A. Plan must document this convention
explicitly.

## fitted_params Shape

**From HANDOFF_viz.md:85-89 + template.html:438-444:**

- Type: `dict[str, list[list[float]]] | None`
- Keys: MUST match `mat_order` entries exactly (case-sensitive, e.g. `"B1"` not
  `"b1"`).
- Values: `list[list[float]]` of the **same shape** as the corresponding
  `matrices[key].vals`. For `A` → `(N,N)`. For `Bj` → `(N,N)`. For `C` → `(N,M)`.
- When `fitted_params[key]` is missing, the renderer falls back to
  `matrices[key].vals` (template `matVals(key)` at line 439).
- Per-element stds, samples, or PPDs are **NOT** part of the v1 schema. Only
  point estimates (posterior means / medians) are rendered. Stds are available
  from `extract_posterior_params` (`guides.py:487`) but go unused in v1.
- Renderer shows only cell text `+0.15` / `−0.20` / `0` via `cellText(val)`
  (`template.html:866`); `toFixed(2)` — i.e. **2 decimal places** in the UI.
  JSON can carry higher precision; the display truncates.

### Required input shape from Pyro-DCM

`extract_posterior_params(guide, model_args)` returns (`guides.py:484-492`):

```python
{
  "A_free":    {"mean": Tensor(N,N), "std": ..., "samples": ...},
  "A":         {"mean": Tensor(N,N), ...},   # parameterize_A(A_free)
  "C":         {"mean": Tensor(N,M), ...},
  "noise_prec":{"mean": Tensor(()), ...},
  "B_free_0":  {"mean": Tensor(N,N), ...},   # raw, unmasked
  "B_free_1":  {"mean": Tensor(N,N), ...},
  ...
  "B":         {"mean": Tensor(J,N,N), ...}, # masked, stacked (when present)
  "median":    {"A_free": Tensor, "A": Tensor, "C": Tensor, "B_free_0": ..., ...},
}
```

The caller is responsible for flattening to `dict[str, list[list[float]]]` with
the right keys. Recommended pattern (documented, not baked in):

```python
# User does this explicitly, passes result to from_posterior.
posterior = extract_posterior_params(guide, model_args)
fitted = {
    "A":  posterior["A"]["mean"].tolist(),       # (N,N) after parameterize_A
    "C":  posterior["C"]["mean"].tolist(),       # (N,M)
    "B1": posterior["B"]["mean"][0].tolist(),    # (N,N), first modulator
    "B2": posterior["B"]["mean"][1].tolist(),    # (N,N), second
}
viz_fitted = CircuitViz.from_posterior(planned, fitted)
```

Note: `posterior["B"]` may be absent depending on Pyro version (it is a
`pyro.deterministic` site — `guides.py:431-438` documents the conditional). The
robust alternative is `parameterize_B(posterior["B_free_0"]["mean"].unsqueeze(0),
b_mask_stacked).squeeze(0)`. Whether `from_posterior` should expose a helper for
this or leave it to the caller is an open question.

## Generic vs HEART2ADAPT-domain fields

All HEART2ADAPT-specific fields are optional in both the schema and the renderer.
The renderer explicitly hides the corresponding UI sections when the arrays are
empty:

| Field | Generic/HEART2ADAPT | Renderer behavior when absent/empty |
|---|---|---|
| `phenotypes` | HEART2ADAPT | `sb-phens` section `display:none` (`template.html:769`) |
| `hypotheses` | HEART2ADAPT | `sb-hyps` section `display:none` (`template.html:825`) |
| `drugs` | HEART2ADAPT | `sb-drugs` section `display:none` (`template.html:799`) |
| `peb.covariates` | HEART2ADAPT-ish | Architecture box omits covariate list if empty (`template.html:694`) |
| `peb.description` | HEART2ADAPT-ish | PEB architecture box row omitted (`template.html:730`) |
| `palette` | Generic (optional) | Defaults to `{}`, falls back to region_colors |
| `node_positions` | Generic but layout-dependent | Falls back to 4-node HEART2ADAPT layout — **visible bug for N ≠ 4** |
| `svg_edges` | Generic but layout-dependent | Falls back to hardcoded HEART2ADAPT 4-node SVG — **wrong circuit drawn for any non-HEART2ADAPT study** |
| `b_overlays` | HEART2ADAPT-specific | Falls back to `{}`; inline HEART2ADAPT overlays embedded in `_heart2adaptEdgesSVG` still render. |
| `node_info` | HEART2ADAPT documentation | Never read by renderer. |

**Critical finding for acceptance:** A bare 3-region bilinear DCM produced by
`from_model_config(minimal_cfg)` with empty phenotypes/hypotheses/drugs WILL
load without JS errors, BUT the SVG diagram will render the wrong
(HEART2ADAPT 4-node) circuit because `node_positions` and `svg_edges` default
to hardcoded 4-region assets. The matrix bottom panel and equation panel WILL
correctly reflect the 3-region bilinear model.

This means: for the v0.3.0 acceptance fixture (Phase 16's 3-region bilinear),
either (a) a test-only helper must synthesize `node_positions` for N regions on
a circle, (b) acceptance tests must skip the SVG render assertion and test
only the JSON data layer, or (c) SVG generation is deferred to v0.3.1. Decision
needed during planning — **user's "visualizer-distinct acceptance" directive
strongly implies option (b) is preferred**: acceptance = JSON correctness, SVG
render fidelity is a manual / later concern.

## Renderer behavior on missing fields

Skim of `docs/dcm_circuit_explorer_template.html` for each optional field. "Skip"
means section hidden; "fallback" means a default is used; "JS error" means
uncaught exception.

| Field | Missing behavior | Line |
|---|---|---|
| `meta` | `cfg.meta?.title` short-circuits to default strings | 408-414 |
| `palette` | `CFG.palette \|\| {}` → empty object, lookups return `undefined` safely | 462 |
| `regions` | `CFG.regions \|\| []` → empty → no nodes drawn, no matrix rows. No error. | 463 |
| `region_colors` | Falls back to `'#888888'` per region | 464 |
| `matrices` | `CFG.matrices \|\| {}` → empty; bottom panel empty, eq panel empty | 466, 678 |
| `mat_order` | Falls back to `Object.keys(mats)` (insertion-ordered) | 503, 831, 879 |
| `phenotypes` | Section hidden via `display:none` | 759, 769 |
| `hypotheses` | Section hidden | 808, 825 |
| `drugs` | Section hidden | 777, 799 |
| `peb` | `CFG.peb \|\| {}`; arch box text partially omitted | 679, 730 |
| `fitted_params` | `null` → toggle hidden, fitted badge absent | 418-422, 880 |
| `node_positions` | Falls back to 4-node HEART2ADAPT layout `DEFAULT_NODE_POS` | 452-457, 465 |
| `svg_edges` | Falls back to `_heart2adaptEdgesSVG()` (hardcoded 4-region paths) | 534-548 |
| `b_overlays` | `CFG.b_overlays \|\| {}` → no custom overlays. HEART2ADAPT defaults still present as inline `<g>` in `_heart2adaptEdgesSVG`. | 552 |
| `matrices[key].inputs` (for C) | Falls back to `['u']` | 898 |
| `matrices[key].modulator_display` | If absent on a B-matrix, that B-matrix is excluded from the modulator strip at the top (`bmods.filter` at 504). It still appears in the bottom matrix grid. | 503-504 |
| `matrices[key].color` | Falls back to `'#374151'` or `'#888888'` depending on code path | 894, 515 |

**No uncaught JS errors** observed in the field-access paths for any of the
optional fields above. This means a minimal JSON `{_schema, regions,
region_colors, matrices}` renders (with wrong default geometry for non-4-region
circuits).

## Acceptance evidence (testable)

Per the user's directive — visualizer acceptance is distinct from recovery
acceptance. Concrete pytest assertions:

### A. Structural / schema tests (pure data layer, fast)

1. **A-01 Round-trip equality for HEART2ADAPT reference.**
   ```python
   cfg = CircuitViz.load("configs/heart2adapt_dcm_config.json")
   roundtrip = json.loads(json.dumps(cfg.to_dict()))
   assert roundtrip == json.loads(Path("configs/heart2adapt_dcm_config.json").read_text())
   ```
   Exposes any silent field dropping (e.g. `node_info` if not pass-through,
   `palette`, `_study`, `_description` — the latter two are NOT in the handoff
   dataclass; this assertion WILL FAIL unless they are added or the reference
   file is trimmed).

   **CONFLICT (flag):** `configs/heart2adapt_dcm_config.json` contains
   `_study`, `_description`, and `node_info` keys that the handoff dataclass
   does NOT preserve. The round-trip assertion either (a) needs to be weakened
   to only check handoff-schema fields, or (b) `CircuitVizConfig` must gain
   pass-through storage for unknown keys.

2. **A-02 `from_posterior` flips status and populates `fitted_params`.**
   ```python
   planned = CircuitViz.from_model_config(minimal_3region_bilinear_cfg)
   assert planned.status == "planned"
   assert planned.fitted_params is None
   fitted = CircuitViz.from_posterior(planned, {"A": [[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]], "B1": ...})
   assert fitted.status == "fitted"
   assert fitted.fitted_params is not None
   assert list(fitted.fitted_params.keys()) == ["A", "B1"]
   ```

3. **A-03 `from_posterior` does not mutate the planned instance.**
   ```python
   fitted = CircuitViz.from_posterior(planned, posterior)
   assert planned.status == "planned"
   assert planned.fitted_params is None   # deepcopy semantics preserved
   ```

4. **A-04 `to_dict` output has exactly the expected top-level keys.**
   ```python
   keys = set(planned.to_dict().keys())
   assert keys == {"_schema", "_status", "meta", "palette", "regions",
                   "region_colors", "matrices", "mat_order", "phenotypes",
                   "hypotheses", "drugs", "peb", "fitted_params"}
   ```

5. **A-05 `mat_order` derivation is deterministic.**
   ```python
   cfg = CircuitViz.from_model_config(cfg_with_Bs={"B2": ..., "B1": ..., "B3": ...})
   assert cfg.mat_order == ["A", "B1", "B2", "B3", "C"]  # sorted, A first, C last
   ```

6. **A-06 Schema field `_schema` is exactly `"dcm_circuit_explorer/v1"`.**

7. **A-07 All `matrices[key].vals` are `list[list[float]]` (not nested ndarray,
   not torch tensor).**
   ```python
   for key, m in cfg.to_dict()["matrices"].items():
       assert isinstance(m["vals"], list)
       for row in m["vals"]:
           assert isinstance(row, list)
           for v in row:
               assert isinstance(v, (int, float))
   ```

8. **A-08 `from_model_config` accepts a torch-tensor-valued `model_cfg` and
   serializes correctly.**
   ```python
   cfg = CircuitViz.from_model_config({
       ..., "A_prior_mean": torch.tensor([[-0.5, 0.3], [0.1, -0.5]]),
   })
   assert cfg.matrices["A"]["vals"] == [[-0.5, 0.3], [0.1, -0.5]]
   ```

9. **A-09 `export(path)` writes valid JSON that `load(path)` reads back
   equal.**
   ```python
   out = cfg.export(tmp_path / "test.json")
   roundtripped = CircuitViz.load(out)
   assert roundtripped.to_dict() == cfg.to_dict()
   ```

10. **A-10 Empty optional collections pass through as `[]` / `{}` (not missing
    keys).**
    ```python
    cfg = CircuitViz.from_model_config(minimal_cfg)  # no phenotypes etc.
    d = cfg.to_dict()
    assert d["phenotypes"] == []
    assert d["hypotheses"] == []
    assert d["drugs"] == []
    assert d["peb"] == {}
    ```

### B. Integration tests (with Pyro-DCM fitting state)

11. **B-01 End-to-end planned→fitted workflow on a 3-region bilinear fixture.**
    Build a minimal `task_dcm_model` with `n_regions=3`, run `run_svi` for 50
    steps (NOT a recovery test, just smoke), call `extract_posterior_params`,
    flatten to the `fitted_params` dict form, pass through `from_posterior`,
    assert `to_dict()` schema is valid.

12. **B-02 `extract_posterior_params` output shapes match serializer
    expectations.** Purely a documentation assertion that
    `posterior["A"]["mean"]` is 2D and `.tolist()` produces `list[list[float]]`
    of the expected `(N,N)` shape.

### C. NOT tested in Phase 17 (deferred / out of scope)

- ❌ No RECOV-style RMSE/coverage/shrinkage gates (per user directive).
- ❌ No browser-side JS-error-free assertion. Headless browser integration
  (playwright / puppeteer) is out of scope for v0.3.0.
- ❌ No SVG render fidelity check (the HEART2ADAPT 4-node default will visually
  misrepresent a 3-region circuit; this is a known limitation, not a bug to fix
  in Phase 17).
- ❌ No amortized-guide path — D5 defers bilinear amortization to v0.3.1.

## Upstream read-only consumers

Phase 17 depends on the following existing Pyro-DCM surfaces. **Phase 17 MUST
NOT modify any of these**; if any change is needed, it's a plan-level risk.

| Upstream module | Function / symbol | File:line | Circuit_viz role |
|---|---|---|---|
| `pyro_dcm.models` | `task_dcm_model` | `src/pyro_dcm/models/task_dcm_model.py:119-399` | Read-only. Circuit_viz serializes its config; never calls it. The model takes `b_masks: list[Tensor] \| None, stim_mod: PiecewiseConstantInput \| None` — these are the **structural** inputs that `model_cfg` needs to synthesize `B_matrices` keys. |
| `pyro_dcm.models` | `extract_posterior_params` | `src/pyro_dcm/models/guides.py:366-495` | Read-only. Its return dict is the canonical input to the user's hand-rolled `posterior_means` payload for `from_posterior`. The **caller** flattens, not `circuit_viz`. |
| `pyro_dcm.models` | `create_guide`, `run_svi` | `src/pyro_dcm/models/guides.py` | Only referenced in docstrings/examples. No runtime dep. |
| `pyro_dcm.forward_models.neural_state` | `parameterize_A`, `parameterize_B` | `src/pyro_dcm/forward_models/neural_state.py:24,62` | Relevant for docstring explanation of the log-diagonal convention; not called at runtime by `circuit_viz`. |
| `pyro_dcm.utils` | (none currently — no JSON/tensor helpers to reuse) | `src/pyro_dcm/utils/__init__.py` | The only existing util is `ode_integrator`. Circuit_viz is the second util; re-export via `utils/__init__.py`. |

**No API changes required upstream.** Phase 17 is purely additive:
- NEW file `src/pyro_dcm/utils/circuit_viz.py`
- EDIT `src/pyro_dcm/utils/__init__.py` to re-export `CircuitViz` and
  `CircuitVizConfig` (3-line additive change, precedent at
  `utils/__init__.py:3-15`)
- NEW file `tests/test_circuit_viz.py`
- (Optional) NEW file `scripts/export_circuit_viz.py` — handoff marks optional.

## Risks / Pitfalls

1. **Handoff ↔ template schema drift (HIGH).** The handoff dataclass omits
   `node_positions`, `svg_edges`, `b_overlays`, `node_info`, and the `_study` /
   `_description` top-level fields present in
   `configs/heart2adapt_dcm_config.json`. A naive round-trip test will fail.
   **Mitigation:** either (a) declare the handoff dataclass authoritative and
   trim the reference config to match, or (b) add a pass-through `extras: dict
   = field(default_factory=dict)` field and preserve unknown keys on `load()`.
   **Recommend (b)** — non-breaking, preserves forward-compat. Planner must
   decide.

2. **Schema source-of-truth ownership (MEDIUM).** Both
   `dcm_circuit_explorer_template.html` and `src/pyro_dcm/utils/circuit_viz.py`
   encode the `dcm_circuit_explorer/v1` schema. There is no JSON-schema file
   (`.json` schema definition) to validate against. Over the v0.3.1+ horizon
   these will drift. **Mitigation:** (a) add a `docs/dcm_circuit_explorer_v1.schema.json`
   file that `circuit_viz.py` validates against with `jsonschema` (new dep), OR
   (b) add a `HANDOFF_viz.md`-level acceptance test that parses the schema
   block from the markdown and asserts the dataclass matches. **Recommend (b)**
   — no new dep, handoff is the canonical doc. Flag for planner.

3. **Float precision and reproducibility (MEDIUM).** `torch.float64` tensors
   serialized via `.tolist()` produce Python floats (IEEE 754 double). JSON
   preserves double precision. **No precision loss** across
   `torch → list → json.dumps → json.loads → list`. However, **display
   precision** in the HTML is `.toFixed(2)` (2 decimals); the JSON itself can
   store any precision. Document this: users comparing fitted JSONs byte-wise
   will see full-precision floats that differ across runs even when the UI
   shows identical values. **Mitigation:** optional `round_to: int = 4` kwarg
   on `export()` — flag to planner, handoff doesn't mention it.

4. **Deterministic `mat_order` emission (LOW).** Python dicts preserve
   insertion order since 3.7. If `model_cfg["B_matrices"]` is a user-built
   dict with arbitrary key order (e.g. `{"B2": ..., "B1": ...}`), `mat_order`
   must be derived deterministically. **Mitigation:** always emit
   `["A"] + sorted(B_matrices.keys()) + (["C"] if "C_matrix" in model_cfg and
   len(C_matrix) else [])`. Do not trust insertion order.

5. **Module placement + package-data shipping (LOW).** The file lives at
   `src/pyro_dcm/utils/circuit_viz.py` — `utils/__init__.py` exists and its
   export convention is clear (`utils/__init__.py:3-15`). The
   `dcm_circuit_explorer_template.html` is under `docs/` — NOT under `src/`
   and NOT currently shipped as package-data. `pyproject.toml` (`packages =
   ["src/pyro_dcm"]`) excludes it. **Decision needed:** should users get the
   HTML template bundled with the wheel, or download from GitHub? **Recommend
   documenting the URL in the CircuitViz docstring** and deferring package-data
   wiring to v0.3.1 (matches D5-style "visualizer is a utility, not core
   inference"). Similarly `configs/heart2adapt_dcm_config.json` is not
   package-data — `CircuitViz.load()` takes an arbitrary path, so this is fine.

6. **Color conventions for non-HEART2ADAPT studies (LOW).** The handoff's
   `model_cfg["region_colors"]` is user-supplied. There is NO fallback
   generation (matplotlib default cycle, `tab10`, etc.). **Mitigation:** on
   `from_model_config`, if `region_colors` is missing, either raise (explicit)
   or synthesize from `itertools.cycle(["#534AB7", "#D85A30", "#1D9E75",
   "#888780", "#185FA5", "#9A7820", "#CC8800", "#374151"])` — the HEART2ADAPT
   palette. **Recommend: raise with a descriptive message** (matches project
   "expected vs actual" convention).

7. **`peb_covariates` signature ambiguity (LOW).** The handoff's `model_cfg`
   spec says `peb_covariates: list[dict]` with `[{name, display, desc}]`,
   but the JSON schema places these under `peb.covariates[]`. The reference
   config has BOTH `meta.peb_covariates: list[str]` (names only, for tags)
   AND `peb.covariates: list[dict]` (full). **Clarification needed:** the
   `peb` kwarg is separate from `model_cfg["peb_covariates"]`. Recommend:
   `model_cfg["peb_covariates"]` is a `list[dict]` consumed as-is into
   `peb.covariates`; `meta.peb_covariates` (list of names) is derived as
   `[c["name"] for c in peb_covariates]` — or accept both and pass through.

8. **`docs/HANDOFF_viz.md:167-190` lists a `from_model()` alternate name in
   the class docstring, but the static method is `from_model_config`.** The
   first docstring example (`usage — planned: cfg = CircuitViz.from_model_config(...)`)
   is consistent; the preceding class-level docstring says `from_model()`.
   **Cosmetic;** just use `from_model_config` consistently.

9. **NaN/Inf in posterior means (MEDIUM for Phase 17, HIGH in Phase 16
   context).** If SVI diverges (which Phase 16 actively defends against via
   the NaN-safe zero-fill guard — `task_dcm_model.py:379`), the posterior
   means can still be NaN or inf. `json.dumps` **raises `ValueError` on
   NaN/Inf by default** (`allow_nan=True` is the default, actually, but
   produces non-compliant JSON the template's `JSON.parse` will reject).
   **Mitigation:** `from_posterior` should validate that `posterior_means`
   contains no NaN/Inf before deepcopy — raise with expected-vs-actual message
   pointing at the offending matrix key and index.

## Proposed Plan Decomposition (advisory)

Based on coupling analysis, a single plan (17-01) is feasible — the surface is
small (~150 LOC source + ~200 LOC tests). A 2-plan split for parallelization:

- **Plan 17-01: `CircuitViz` core + unit tests.**
  - Implement `CircuitVizConfig` dataclass per handoff verbatim.
  - Implement `from_model_config`, `from_posterior`, `load`, `export`,
    `to_dict`.
  - Mat_order derivation, NaN validation, deterministic color-raise policy.
  - `tests/test_circuit_viz.py`: A-01 through A-10 from above.
  - Update `src/pyro_dcm/utils/__init__.py` to re-export.

- **Plan 17-02: Pyro integration smoke test + docs + (optional) CLI.**
  - `tests/test_circuit_viz.py::TestPyroIntegration` (B-01, B-02): end-to-end
    from `run_svi` → `extract_posterior_params` → `from_posterior` → JSON.
  - Docstring examples and brief `docs/02_pipeline_guide/circuit_viz.md`
    usage doc.
  - (Optional) `scripts/export_circuit_viz.py` CLI per handoff section "Files
    to create".

**Recommendation:** Single plan (17-01) combining both — the total surface is
too small to justify the overhead of a 2-plan split, and the Pyro integration
test is a smoke check (not a full runner), so there is no parallelization
benefit. Plan should explicitly defer the browser-side render-fidelity check
to v0.3.1. Not binding — planner to decide.

## Open Questions

1. **Schema conflict: should `node_info`, `_study`, `_description`, and
   HEART2ADAPT `node_positions`/`svg_edges`/`b_overlays` be first-class fields
   on `CircuitVizConfig`, or stored under a pass-through `extras` dict?**
   - What we know: the handoff omits them entirely; the renderer ignores
     `node_info`/`_study`/`_description` but reads the other three; the
     reference JSON includes all of them.
   - What's unclear: is the handoff intentionally narrow (v1 = minimum viable
     subset) or was it authored before those extensions?
   - Recommendation: planner asks the user. Default = add `extras: dict`
     pass-through so the round-trip test passes without schema bloat.

2. **`palette` vs `region_colors`.** The handoff spec for `model_cfg` does NOT
   list `palette`, but the `CircuitVizConfig` dataclass has a `palette: dict`
   field and the reference config sets it. The renderer uses `palette.vs`,
   `palette.amy` etc. for B-matrix color matching in the eq panel
   (`template.html:687-708`) — these are **semantic color-role names**, not
   region names. Should `from_model_config` accept `palette` as a kwarg or
   derive it from `B_modulators[key].color`?
   - Recommendation: accept `palette` as an optional kwarg parallel to
     `phenotypes`/`drugs`/etc. Not auto-derived.

3. **Should the schema be codified as a JSON Schema file?** (Risk 2 above.)
   - Recommendation: deferred to v0.3.1. Phase 17 = runtime dataclass is the
     informal schema source-of-truth; a JSON-schema file is a v0.3.1 polish
     task.

4. **Should `from_posterior` accept the raw `extract_posterior_params` output
   dict directly, with automatic flattening, or require the caller to
   pre-flatten to `dict[str, list[list[float]]]`?**
   - Handoff's signature is unambiguous: pre-flattened.
   - Pro-flatten argument: convenience for the most common Pyro-DCM workflow.
   - Pro-explicit argument: caller controls the A-transform (`A` vs `A_free`),
     the B-source (`B` stacked vs `B_free_j` raw), and the masking.
   - Recommendation: keep the handoff signature (pre-flattened). Add a
     sibling helper `CircuitViz._posterior_to_matrices(posterior, mat_order,
     b_masks=None)` as optional convenience, but mark the primary API as the
     explicit form.

5. **How are `CircuitVizConfig.schema` (handoff name) vs `_schema` (JSON key)
   reconciled?** The dataclass uses `schema: str` — shadows Python's builtin
   `schema` module name, but not dangerously. `to_dict()` renames to `_schema`.
   This is correct per the reference JSON.
   - Recommendation: document the rename in the dataclass docstring. No code
     change.

6. **Should there be a `validate()` method on `CircuitVizConfig` that checks
   matrix shapes, region/color length parity, mat_order keys match
   matrices.keys(), fitted_params key/shape conformance, no NaN?**
   - Handoff doesn't mention one. Would centralize the NaN / shape /
     consistency checks currently scattered across `from_model_config`,
     `from_posterior`, and `export`.
   - Recommendation: add a private `_validate()` called from `export()` and
     `to_dict()` exit, raising `ValueError` with expected-vs-actual messages.
     Not a new public API.

---

## Sources

### Primary (HIGH confidence, in-repo authoritative)

- `docs/HANDOFF_viz.md:1-301` — class skeleton, JSON schema, field mapping,
  workflow.
- `configs/heart2adapt_dcm_config.json:1-363` — full reference fixture.
- `docs/dcm_circuit_explorer_template.html:365-1057` — consumer behavior
  (field access paths, optional-field fallbacks, toggle logic).
- `src/pyro_dcm/models/task_dcm_model.py:119-399` — upstream model,
  `b_masks`/`stim_mod`/A/C shapes.
- `src/pyro_dcm/models/guides.py:366-495` — `extract_posterior_params` return
  shape.
- `src/pyro_dcm/forward_models/neural_state.py:24-105` — `parameterize_A`,
  `parameterize_B` conventions.
- `src/pyro_dcm/utils/__init__.py:1-15` — utils package export convention.
- `src/pyro_dcm/models/__init__.py:1-35` — `extract_posterior_params` is
  publicly exported.
- `benchmarks/runners/task_bilinear.py:410-489` — real-world
  `extract_posterior_params` consumer pattern; confirms the flatten-by-caller
  idiom and `B_true`/`B_free_{j}`/`B` key usage.
- `pyproject.toml:1-75` — packaging layout; no additional deps needed for
  Phase 17 (pure stdlib: `json`, `pathlib`, `dataclasses`, `copy`; plus
  `torch` for tensor-to-list conversion).
- `.planning/STATE.md:20-89` — v0.3.0 decisions D1-D5 relevant to scope
  (bilinear-only, SVI-only, B_free prior variance = 1.0).

### Secondary (MEDIUM confidence, derived)

- Field presence-vs-absence matrix in the template: derived by grep of
  `CFG.` reads in `docs/dcm_circuit_explorer_template.html`.
- `mat_order` determinism recommendation: inferred from dict-ordering
  semantics and `template.html:503,831,879` fallback behavior; no
  authoritative spec statement.

### Tertiary (LOW confidence, assumption)

- JSON-schema codification recommendation: no prior art in repo; ecosystem
  norm (jsonschema Python lib) but speculative for this project.
- Browser-headless testing as out-of-scope: inferred from absence of any
  existing JS/browser testing infrastructure; no contradictory signal in
  repo.

## Metadata

**Confidence breakdown:**
- Class interface: HIGH — verbatim from handoff.
- Field mapping: MEDIUM-HIGH — handoff clear, but `node_info`/`palette`/
  extension fields are underspecified in handoff vs. reference JSON.
- fitted_params shape: HIGH — cross-verified against
  `extract_posterior_params` + template `matVals`.
- Renderer behavior: HIGH — direct source grep of template.html.
- Acceptance evidence: MEDIUM-HIGH — user directive clear, but round-trip
  test is blocked on the extras-field decision.
- Risks: MEDIUM — schema drift and NaN in posterior are concrete; color
  fallback and packaging are style questions.

**Research date:** 2026-04-24
**Valid until:** 30 days (stable — no fast-moving upstream deps). Invalidated
only if the handoff doc or the template HTML is edited.

## RESEARCH COMPLETE
