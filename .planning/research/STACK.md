# Technology Stack

**Project:** Pyro-DCM -- MNE-Python Integration (Testing, Pipelines, Documentation)
**Researched:** 2026-05-21
**Scope:** Stack additions for testing MNE IO loaders, end-to-end pipeline scripts, and API documentation. Existing core stack (PyTorch, Pyro, torchdiffeq, Zuko, NumPyro, scipy) is validated and NOT re-evaluated here.

## Recommended Stack Additions

### Testing Infrastructure

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytest (existing) | >=9.0 | Test framework | Already in `[dev]` extra. No version bump needed; `pytest.importorskip` is the core mechanism for optional-dep gating. |
| pytest-cov | >=7.0 | Coverage reporting for IO module | Needed to verify IO code is actually exercised. Lightweight, integrates with existing pytest. No alternative warranted. |

### MNE Test Data Strategy (NO new dependencies)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| mne.create_info | (bundled with mne>=1.6) | Build synthetic Info objects for test fixtures | Already available via MNE. Zero additional deps. |
| mne.io.RawArray | (bundled) | Create Raw objects from NumPy arrays | Construct test Raw with known data, no disk IO. |
| mne.EpochsArray | (bundled) | Create Epochs from NumPy arrays | Construct test Epochs with known shape/values, no file dependencies. |
| mne.SourceEstimate | (bundled) | Create synthetic source estimates | Constructor takes `(data, vertices, tmin, tstep)` -- fully synthetic. |
| mne.Label | (bundled) | Create test ROI labels | Can be constructed from vertex arrays programmatically. |
| mne_bids.BIDSPath + write_raw_bids | (bundled with mne-bids>=0.14) | Create minimal BIDS datasets in tmpdir | `write_raw_bids` + `make_dataset_description` create valid BIDS structure from synthetic RawArray. Tests against real BIDS reader, not mocks. |

### Documentation System

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Sphinx | >=8.0 | Documentation generator | Industry standard for scientific Python. Autodoc parses NumPy docstrings natively via napoleon. MNE-Python itself uses Sphinx. |
| sphinx.ext.napoleon | (bundled with Sphinx) | NumPy-style docstring support | Project already uses NumPy docstrings enforced by ruff `convention = "numpy"`. Napoleon is bundled, zero-config. |
| sphinx.ext.autodoc | (bundled with Sphinx) | Auto-generate API docs from docstrings | Core requirement: every public function gets an API page without manual duplication. |
| sphinx.ext.intersphinx | (bundled with Sphinx) | Cross-link to MNE, PyTorch, Pyro docs | Users click `mne.Epochs` in our docs and land on MNE's page. Critical for IO module docs. |
| sphinx-autodoc-typehints | >=3.10 | Render Python type annotations in docs | Project uses `from __future__ import annotations` everywhere; this extension renders `str | None` properly in built docs. Without it, type hints appear as raw strings. |
| furo | >=2024.1 | Sphinx theme | Clean, modern, responsive. Used by pip, urllib3, attrs. Simpler than pydata-sphinx-theme, better than RTD theme. Dark mode included. |
| myst-parser | >=4.0 | Write Sphinx docs in Markdown | Existing docs are all Markdown (docs/*.md). MyST lets Sphinx consume them directly instead of forcing an RST rewrite. Supports cross-references, admonitions, all Sphinx directives. |

### Notebook Documentation (DEFERRED -- see rationale below)

| Technology | Version | Purpose | Decision |
|------------|---------|---------|----------|
| myst-nb | 1.4.0 | Execute Jupyter notebooks in Sphinx | DEFER. No notebooks exist yet. Add when pipeline tutorials are written as .ipynb. |
| nbsphinx | 0.9.8 | Alternative notebook renderer | SKIP. If notebooks are added, prefer myst-nb (same MyST ecosystem, better caching). |

### Pipeline Scripts (NO new dependencies)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| argparse (stdlib) | -- | CLI argument parsing for pipeline scripts | Stdlib, zero deps. Pipeline scripts are simple enough that Click/Typer are overkill. |
| logging (stdlib) | -- | Pipeline progress reporting | Already used in pyro_dcm (NullHandler pattern). Pipeline scripts configure a StreamHandler. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Test data | Synthetic via RawArray/EpochsArray | `mne.datasets.sample` (1.9 GB download) | Too large for CI, too slow for unit tests, introduces network dependency. Synthetic data gives full control over ground truth. |
| Test data | Synthetic via RawArray/EpochsArray | unittest.mock / MagicMock on MNE objects | Mocking MNE internals is fragile -- CSD computation calls internal methods that mocks cannot replicate. Real MNE objects from arrays test the actual code paths. |
| Test BIDS | write_raw_bids to tmpdir | Pre-committed BIDS fixture files | Binary files in git are an anti-pattern. write_raw_bids creates valid BIDS from synthetic data, tests the real reader, and cleans up via pytest tmp_path. |
| Docs theme | furo | pydata-sphinx-theme | pydata-sphinx-theme is heavier, designed for NumPy/pandas scale. Furo is right-sized for a research package. |
| Docs theme | furo | sphinx-rtd-theme | RTD theme looks dated, poor mobile support, limited customization. |
| Docs format | myst-parser (Markdown) | reStructuredText | All existing docs are .md. Forcing RST would require rewriting docs/ directory. MyST gives full Sphinx power with Markdown syntax. |
| Docs notebooks | myst-nb (deferred) | nbsphinx | myst-nb integrates with myst-parser (same parser), supports jupyter-cache for faster builds, and is the executablebooks ecosystem standard. |
| Coverage | pytest-cov | coverage.py directly | pytest-cov wraps coverage.py with better pytest integration (automatic subprocess coverage, --cov flag). |
| CLI framework | argparse (stdlib) | Click / Typer | Pipeline scripts take 3-5 arguments. Click adds a dependency for no real gain at this scale. |

## What NOT to Add

| Temptation | Why Not |
|------------|---------|
| `pytest-mne` (hypothetical) | Does not exist as a package. MNE's own test suite uses RawArray/EpochsArray + pytest directly. |
| `mne.datasets.testing` | 2+ GB download, designed for MNE's internal CI. Our loaders need 3-channel, 100-sample test data, not a full Neuromag dataset. |
| `mne.datasets.sample` | 1.9 GB. Same issue. Useful for documentation examples but not for unit tests. |
| `hypothesis` (property-based testing) | IO loaders have fixed contracts (MNE object in, dict of tensors out). Property-based testing adds complexity without value here. |
| `sphinx-gallery` | Generates docs from Python scripts. We have no gallery-style examples yet. Add later if needed. |
| `jupyter-sphinx` | Superseded by myst-nb for notebook embedding. |
| `numpydoc` (Sphinx extension) | Redundant with sphinx.ext.napoleon, which handles NumPy docstrings natively and is bundled with Sphinx. |
| `MkDocs` / `mkdocstrings` | MkDocs cannot parse NumPy docstrings as well as Sphinx+napoleon. autodoc integration is weaker. Scientific Python ecosystem overwhelmingly uses Sphinx (NumPy, SciPy, MNE, scikit-learn, PyTorch). |
| `tox` / `nox` | Project uses simple `pip install -e .[dev,mne]` workflow. Multi-environment testing adds complexity not needed for a single-maintainer research package. |

## pyproject.toml Changes

```toml
[project.optional-dependencies]
mne = [
    "mne>=1.6",
    "mne-bids>=0.14",
]
dev = [
    "pytest",
    "pytest-cov",        # NEW: coverage for IO module
    "ruff",
    "mypy",
]
docs = [                 # NEW: documentation build
    "sphinx>=8.0",
    "furo>=2024.1",
    "myst-parser>=4.0",
    "sphinx-autodoc-typehints>=3.10",
]
```

```bash
# Testing (with MNE)
pip install -e ".[dev,mne]"
pytest tests/ --cov=pyro_dcm.io

# Documentation build
pip install -e ".[docs]"
sphinx-build docs/ docs/_build/html

# Full development install
pip install -e ".[dev,mne,docs]"
```

## pytest Configuration Additions

```toml
# In pyproject.toml [tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "spm: marks tests requiring MATLAB + SPM12",
    "tapas: marks tests requiring tapas rDCM MATLAB toolbox",
    "mne: marks tests requiring mne and mne-bids optional deps",  # NEW
]
```

## Testing Pattern: importorskip + Synthetic Data

The recommended pattern for all MNE IO tests:

```python
"""Tests for pyro_dcm.io.mne_loader."""
from __future__ import annotations

import numpy as np
import pytest
import torch

# Skip entire module if MNE not installed
mne = pytest.importorskip("mne", minversion="1.6")

from pyro_dcm.io import epochs_to_csd, epochs_to_timeseries


@pytest.fixture()
def synth_epochs() -> mne.Epochs:
    """Create minimal synthetic Epochs for testing."""
    n_channels, n_times, n_epochs = 3, 100, 5
    sfreq = 100.0
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    return mne.EpochsArray(data, info, events=events, tmin=-0.2)
```

For BIDS loader tests:

```python
mne = pytest.importorskip("mne", minversion="1.6")
mne_bids = pytest.importorskip("mne_bids", minversion="0.14")

@pytest.fixture()
def bids_dataset(tmp_path):
    """Create minimal BIDS dataset in temp directory."""
    info = mne.create_info(["EEG001", "EEG002"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.default_rng(0).standard_normal((2, 1000)) * 1e-6, info)
    raw.set_annotations(mne.Annotations(onset=[1.0], duration=[0.0], description=["stimulus"]))
    bids_path = mne_bids.BIDSPath(subject="01", task="test", root=tmp_path, datatype="eeg")
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    return bids_path
```

## SourceEstimate Testing (No FreeSurfer Required)

The `stc_to_roi_timeseries` function requires SourceEstimate, Labels, and SourceSpaces. All can be constructed synthetically:

```python
@pytest.fixture()
def synth_stc_data():
    """Create synthetic source estimate and labels (no FreeSurfer)."""
    n_vertices_lh, n_vertices_rh = 10, 10
    n_times = 50
    vertices = [np.arange(n_vertices_lh), np.arange(n_vertices_rh)]
    data = np.random.default_rng(42).standard_normal((n_vertices_lh + n_vertices_rh, n_times))
    stc = mne.SourceEstimate(data, vertices, tmin=0.0, tstep=0.01)

    # Labels from vertex subsets
    label_lh = mne.Label(vertices=np.arange(5), hemi="lh", name="ROI-lh")
    label_rh = mne.Label(vertices=np.arange(5), hemi="rh", name="ROI-rh")
    return stc, [label_lh, label_rh]
```

Note: `extract_label_time_course` with synthetic SourceSpaces requires more care -- this fixture pattern should be verified against MNE 1.6+ API. The SourceSpaces object may need `vertno` attributes set correctly. This is a MEDIUM confidence recommendation that needs validation during implementation.

## Version Pinning Rationale

| Package | Pin Style | Why |
|---------|-----------|-----|
| mne | `>=1.6` | Floor pin. 1.6 introduced API changes we rely on. No upper bound -- MNE maintains backward compat. Current stable: 1.12.1. |
| mne-bids | `>=0.14` | Floor pin. 0.14 is required for BIDSPath API. Current stable: 0.18.0. |
| sphinx | `>=8.0` | Floor pin. 8.0+ for modern Python support and mathjax4. Current: 9.1.0. |
| furo | `>=2024.1` | Calendar-versioned. Any 2024+ release works with Sphinx >=8.0. Current: 2025.12.19. |
| myst-parser | `>=4.0` | Floor pin. 4.0 is the latest major with Sphinx 8+ compat. Current: 4.0.1. |
| sphinx-autodoc-typehints | `>=3.10` | Floor pin. 3.10+ handles `from __future__ import annotations` correctly. Current: 3.10.2. |
| pytest-cov | (no pin) | Any recent version works. No API surface we depend on. |

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Test data strategy (synthetic) | HIGH | MNE's own test suite uses RawArray/EpochsArray extensively. create_info API verified against MNE 1.12 docs. |
| pytest.importorskip pattern | HIGH | Official pytest API, stable for years, used by MNE, scikit-learn, etc. |
| BIDS test data via write_raw_bids | HIGH | MNE-BIDS 0.18.0 docs confirm write_raw_bids + BIDSPath API. Verified via official docs. |
| Sphinx + napoleon + furo | HIGH | Standard scientific Python stack. All version numbers verified via PyPI as of 2026-05-21. |
| myst-parser for Markdown docs | HIGH | Version 4.0.1 confirmed on PyPI. Sphinx 8+ compatible. |
| SourceEstimate synthetic testing | MEDIUM | SourceEstimate constructor is straightforward, but extract_label_time_course with synthetic SourceSpaces needs API verification during implementation. |

## Sources

- [MNE 1.12.1 documentation -- EpochsArray](https://mne.tools/stable/generated/mne.EpochsArray.html)
- [MNE 1.12.0 documentation -- RawArray](https://mne.tools/stable/generated/mne.io.RawArray.html)
- [MNE 1.12.0 documentation -- SourceEstimate](https://mne.tools/stable/generated/mne.SourceEstimate.html)
- [MNE tutorials -- Creating data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html)
- [MNE-BIDS 0.18.0 -- write_raw_bids](https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html)
- [MNE-BIDS 0.18.0 -- make_dataset_description](https://mne.tools/mne-bids/stable/generated/mne_bids.make_dataset_description.html)
- [pytest -- importorskip](https://docs.pytest.org/en/stable/reference/reference.html)
- [pytest -- skipping](https://docs.pytest.org/en/stable/how-to/skipping.html)
- [Sphinx -- napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [sphinx-autodoc-typehints (PyPI)](https://pypi.org/project/sphinx-autodoc-typehints/)
- [furo theme (PyPI)](https://pypi.org/project/furo/)
- [myst-parser (PyPI)](https://pypi.org/project/myst-parser/)
- [myst-nb (PyPI)](https://pypi.org/project/myst-nb/)
- [pytest-cov 7.1.0](https://pytest-cov.readthedocs.io/)
- [Sphinx (PyPI)](https://pypi.org/project/Sphinx/)
