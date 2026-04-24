# DCM Circuit Explorer — Handoff for Claude Code

## What exists

| File | Location | Purpose |
|------|----------|---------|
| `dcm_circuit_explorer.html` | `Approach Avoid Anxiety/` | Fully baked HEART2ADAPT explorer (hardcoded, standalone, final) |
| `dcm_circuit_explorer_template.html` | `docs/` | Generic config-driven template — loads any DCM config JSON |
| `heart2adapt_dcm_config.json` | `configs/` | HEART2ADAPT config (planned params, `fitted_params: null`) |

---

## Goal

Implement `src/pyro_dcm/utils/circuit_viz.py` — a Python class that serialises a DCM model config (and optionally fitted posterior means) into the JSON schema consumed by `dcm_circuit_explorer_template.html`.

---

## JSON Schema (`_schema: "dcm_circuit_explorer/v1"`)

See `configs/heart2adapt_dcm_config.json` for the full reference. Key fields:

```json
{
  "_schema": "dcm_circuit_explorer/v1",
  "_status": "planned | fitted",

  "meta": {
    "title": "DCM Circuit Explorer",
    "subtitle": "Study name",
    "tags": ["n-region", "SVI"],
    "peb_covariates": ["beta_rew", "omega"]
  },

  "regions": ["dACC", "AMY", "VS", "vACC"],
  "region_colors": ["#534AB7", "#D85A30", "#1D9E75", "#888780"],

  "matrices": {
    "A":  { "label": "A — Intrinsic",  "color": "#374151", "vals": [[...], ...] },
    "B1": { "label": "B₁ — ε₂",       "color": "#1D9E75", "vals": [[...], ...],
            "modulator": "epsilon_2(t)", "modulator_display": "ε₂(t) — reward PE" },
    "B2": { "label": "B₂ — ε₃",       "color": "#D85A30", "vals": [[...], ...],
            "modulator": "epsilon_3(t)", "modulator_display": "ε₃(t) — volatility PE" },
    "C":  { "label": "C — Driving",    "color": "#6B7280", "vals": [[...], ...],
            "inputs": ["u_driving"] }
  },
  "mat_order": ["A", "B1", "B2", "C"],

  "phenotypes": [
    {
      "id": "P1", "label": "Balanced", "sub": "β_rew=2.5 · ω=−3.0",
      "color": "#534AB7",
      "hgf_params": { "beta_rew": 2.5, "omega": -3.0 },
      "hl_nodes": ["dacc","amy"], "hl_edges": ["e-amy-dacc"],
      "b_overlays": ["b1-overlay"], "hl_cells": { "B1": [[0,2]] },
      "desc": "Full description shown on hover.",
      "dcm_signature": { "A_overrides": {}, "B1_scale": 1.0 }
    }
  ],

  "hypotheses": [
    {
      "id": "H1", "label": "ε₂ → VS→dACC", "short_desc": "...",
      "color": "#1D9E75",
      "hl_nodes": ["vs","dacc"], "hl_edges": ["e-vs-dacc"],
      "b_overlays": ["b1-overlay"], "hl_cells": { "B1": [[0,2]] },
      "modulator": "ε₂(t)", "mat_param": "B₁[0,2] = +0.15", "test_param": "B₁[0,2] > 0",
      "desc": "Full description."
    }
  ],

  "drugs": [ ... ],   // optional — same structure as heart2adapt_dcm_config.json

  "peb": {
    "description": "θᵢ = μ + X·β + εᵢ",
    "covariates": [
      { "name": "beta_rew", "display": "β_rew", "desc": "reward sensitivity" }
    ]
  },

  "fitted_params": null   // or { "A": [[...]], "B1": [[...]], ... } when status = "fitted"
}
```

**Rules:**
- `matrices[key].vals`: always `list[list[float]]`, shape `(n_regions, n_regions)` or `(n_regions, n_inputs)` for C
- `fitted_params`: same dict structure as `matrices` but values are posterior means (from `guide.median()` or NUTS samples mean). When non-null, the HTML shows a **Planned / Fitted** toggle in the topbar.
- `hl_cells` keys match `mat_order` keys exactly (e.g. `"B1"`, not `"b1"`)
- `b_overlays` are SVG `<g>` element IDs; for the HEART2ADAPT default layout these are `b1-overlay`, `b2-overlay`, `b3a-overlay`, `b3b-overlay`

---

## Python class to implement

**File:** `src/pyro_dcm/utils/circuit_viz.py`

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
        """Write config JSON and return path."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


class CircuitViz:
    """Generate DCM Circuit Explorer JSON configs from Pyro-DCM model state.

    Usage — planned (before fitting):
        cfg = CircuitViz.from_model_config(task_dcm_config)
        cfg.export("configs/my_study_planned.json")

    Usage — fitted (after SVI):
        posterior_means = {
            "A":  guide.median()["A"].detach().numpy().tolist(),
            "B1": guide.median()["B1"].detach().numpy().tolist(),
        }
        cfg = CircuitViz.from_posterior(planned_cfg, posterior_means)
        cfg.export("configs/my_study_fitted.json")
    """

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

        Parameters
        ----------
        model_cfg : dict
            Must contain:
              - "regions": list[str]
              - "region_colors": list[str]
              - "A_prior_mean": list[list[float]]    shape (N, N)
              - "B_matrices": dict[str, list[list[float]]]  e.g. {"B1": ..., "B2": ...}
              - "C_matrix": list[list[float]]         shape (N, M)
              - "B_modulators": dict[str, dict]  {key: {label, color, modulator, modulator_display}}
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
        """Attach posterior means to a planned config, switching status to 'fitted'.

        Parameters
        ----------
        planned : CircuitVizConfig
            The planned config (output of from_model_config).
        posterior_means : dict
            Keys match mat_order; values are list[list[float]] of posterior means.
            Typically: guide.median()[key].detach().tolist()
        """
        import copy
        fitted = copy.deepcopy(planned)
        fitted.status = "fitted"
        fitted.fitted_params = posterior_means
        return fitted

    @staticmethod
    def load(path: str | Path) -> CircuitVizConfig:
        """Load an existing JSON config back into a CircuitVizConfig."""
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

---

## Implementation notes for `from_model_config`

The HEART2ADAPT `task_dcm_model.py` config is a Python dict / dataclass. The key mapping is:

| JSON field | Pyro-DCM source |
|---|---|
| `matrices.A.vals` | Prior mean of A (log-space → real via `torch.exp` for diagonal, raw for off-diagonal) |
| `matrices.B1.vals` | `B_matrices["B1"]` prior mean |
| `matrices.C.vals` | `C_matrix` |
| `fitted_params.A` | `guide.median()["A_tril"]` reconstructed → full matrix |
| `fitted_params.B1` | `guide.median()["B1"]` |

The `selectNode` click in the HTML highlights A-matrix rows/cols — so `A.vals` shape must be exactly `(N, N)`.

---

## Workflow: before and after fitting

```python
# 1. Before fitting — inspect circuit before running SVI
from pyro_dcm.utils.circuit_viz import CircuitViz
from configs.heart2adapt_dcm_config import HEART2ADAPT_MODEL_CFG  # your config dict

planned = CircuitViz.from_model_config(
    HEART2ADAPT_MODEL_CFG,
    phenotypes=PHENS,      # from configs/heart2adapt_dcm_config.json
    hypotheses=HYPS,
    drugs=DRUGS,
    peb=PEB_CFG,
)
planned.export("configs/heart2adapt_planned.json")
# → open docs/dcm_circuit_explorer_template.html, load this JSON

# 2. After fitting — attach posterior means
posterior = {
    "A":  guide.median()["A"].detach().tolist(),
    "B1": guide.median()["B1"].detach().tolist(),
    "B2": guide.median()["B2"].detach().tolist(),
    "B3": guide.median()["B3"].detach().tolist(),
}
fitted = CircuitViz.from_posterior(planned, posterior)
fitted.export("configs/heart2adapt_fitted.json")
# → HTML shows "Planned / Fitted" toggle — click to compare
```

---

## Files to create

1. `src/pyro_dcm/utils/circuit_viz.py` — implement `CircuitViz.from_model_config()`
2. `tests/test_circuit_viz.py` — smoke test: load `configs/heart2adapt_dcm_config.json` via `CircuitViz.load()`, call `from_posterior` with dummy posterior means, assert JSON roundtrip is valid
3. (Optional) `scripts/export_circuit_viz.py` — CLI: `python scripts/export_circuit_viz.py --config heart2adapt --posterior runs/latest/posterior.pt`

---

## How to open the visualiser

```bash
# serve locally so fetch() works for demo config
python -m http.server 8080 --directory dcm_pytorch/
# then open: http://localhost:8080/docs/dcm_circuit_explorer_template.html
# click "Load demo (HEART2ADAPT)" → loads configs/heart2adapt_dcm_config.json
```

Or open `dcm_circuit_explorer_template.html` directly in a browser and use **Load JSON config** to pick any exported config file.
