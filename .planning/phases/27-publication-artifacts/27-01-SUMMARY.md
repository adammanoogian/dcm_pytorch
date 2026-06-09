---
phase: 27-publication-artifacts
plan: 01
subsystem: publication-references-methods
tags: [references, methods, latex, markdown, citations, v0.6.0]
backfilled: 2026-06-09

dependency-graph:
  requires: []
  provides: [v0.6.0-reference-catalog, methods-markdown, methods-latex]
  affects: [27-02]

tech-stack:
  added: []
  patterns: [markdown-reference-table, placeholder-citation-keys, no-bib-edit-rule]

file-tracking:
  key-files:
    created: []
    modified:
      - .planning/REFERENCES.md
      - docs/03_methods_reference/methods.md
      - docs/03_methods_reference/methods.tex

decisions:
  - id: 27-01-D1
    summary: "Placeholder LaTeX citation keys pending Zotero export"
    detail: "methods.tex uses placeholder \\citep{} keys (friston2003, langdon2025, friston2011bmr, goncalves2020, kingma2014) with a top-of-file comment noting they must be updated once references are added to Zotero. references.bib was NOT edited (auto-exported from Zotero)."
  - id: 27-01-D2
    summary: "REFERENCES.md (markdown) is the editable reference catalog, not .bib"
    detail: "All v0.6.0 references were added to the markdown REFERENCES.md table. The Better BibTeX .bib file is left untouched per project rule."

metrics:
  duration: ~2 min (three sequential docs commits)
  completed: 2026-05-28
---

# Phase 27 Plan 01: v0.6.0 References and Methods Extension Summary

Backfilled write-up. Updated the project reference catalog with all v0.6.0
references and extended the methods reference (Markdown + LaTeX) to cover the
full v0.6.0 methodology: bilinear extension, direct observation model,
variational Laplace, DCM interpretability for neural data models, Bayesian
Model Reduction, simulation-based inference for spectral DCM, and hybrid
VAE-DCM. Delivers PUB-02 (reference catalog) and PUB-03 (methods narrative).

## Completed Tasks

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Add v0.6.0 references (REF-071..REF-110) + extend equation table | 7a70494 | .planning/REFERENCES.md |
| 2a | Extend methods.md with v0.6.0 sections 6-10 | 5d9f665 | docs/03_methods_reference/methods.md |
| 2b | Extend methods.tex with matching v0.6.0 LaTeX sections | aa39731 | docs/03_methods_reference/methods.tex |

## What Was Built

### .planning/REFERENCES.md (+67 lines, commit 7a70494)

New reference entries added (REF-070 already pre-existed; this commit added
REF-071 through REF-110), grouped under new section headers:

- **Bayesian Model Reduction**: REF-071 (Rosa et al. 2012). REF-070
  (Friston & Penny 2011) was already present and is cited alongside it.
- **CT-RNN and Latent Circuit Analysis**: REF-080 (Langdon & Engel 2025,
  formal citation pending Zotero), REF-081 (Sussillo & Barak 2013).
- **Simulation-Based Inference**: REF-090 (Goncalves et al. 2020),
  REF-091 (Hashemi et al. 2024).
- **Foundation Models for Neuroimaging**: REF-100 (Thomas et al. 2025,
  pending Zotero), REF-101 (Jiang et al. 2024, LaBraM).
- **Hybrid Generative Models**: REF-110 (Kingma & Welling 2014, VAE).

The "Equation Quick-Reference by Module" table gained new rows mapping
modules to references: `latent_circuit_dcm_model.py` (REF-001, REF-070),
`bmr.py` (REF-070, REF-071), `continuous_time_rnn.py` (REF-080),
`fixed_point_analysis.py` (REF-081), `sbi_spectral.py` (REF-090, REF-091),
`dcm_encoder_net.py` (REF-110), and `variational_laplace.py` (REF-040).

### docs/03_methods_reference/methods.md (+50 lines, commit 5d9f665)

Appended after the existing Sections 1-5. New sections:

- **6. Bilinear Extension and Direct Observation Model**
  - 6.1 Bilinear Neural State Equation
  - 6.2 Direct Observation Model
  - 6.3 Variational Laplace
- **7. DCM Interpretability for Neural Data Models**
  - 7.1 CT-RNN Training
  - 7.2 Neural Data Model Pipeline
- **8. Bayesian Model Reduction**
- **9. Simulation-Based Inference for Spectral DCM**
- **10. Hybrid VAE-DCM**

### docs/03_methods_reference/methods.tex (+59 lines, commit aa39731)

Matching LaTeX fragment sections appended after existing content, with a
top-of-file comment noting that citation keys are placeholders pending
Zotero export:

- `\section{Bilinear Extension and Direct Observation Model}` with
  `\subsection{Bilinear Neural State Equation}` and
  `\subsection{Direct Observation Model}`
- `\section{DCM Interpretability for Neural Data Models}`
- `\section{Bayesian Model Reduction}`
- `\section{Simulation-Based Inference for Spectral DCM}`
- `\section{Hybrid VAE-DCM}`

Equation labels added: `eq:bilinear-state`, `eq:direct-obs`, `eq:ctrnn`,
`eq:bmr-cov`, `eq:bmr-mean`, `eq:sbi-simulator`, `eq:vae-elbo`.
Placeholder `\citep{}` keys used: `friston2003`, `langdon2025`,
`friston2011bmr`, `goncalves2020`, `kingma2014`.

## Deviations from Plan

### Noted Behaviors

**1. [Note] methods.md uses top-level Sections 6-10 rather than 1.6/1.7 + 6-9**

- The plan sketched the bilinear and direct-observation content as
  subsections 1.6/1.7 plus top-level Sections 6-9. As implemented, the
  bilinear extension, direct observation model, and variational Laplace
  were grouped under a single new top-level **Section 6**, shifting the
  interpretability/BMR/SBI/VAE sections to **7-10**. Same content, cleaner
  top-level numbering. No content was dropped.

**2. [Note] REF-070 pre-existed**

- The plan listed REF-070 (Friston & Penny 2011) as a new entry, but it was
  already present in REFERENCES.md. The commit added REF-071 through REF-110,
  consistent with the commit message. REF-070 is referenced from the new
  BMR section and equation table.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 27-01-D1 | Placeholder LaTeX citation keys with Zotero-update note | references.bib is auto-exported from Zotero; cannot hand-write entries |
| 27-01-D2 | Edit markdown REFERENCES.md, never the .bib | Project rule: .bib is Better BibTeX auto-export only |

## Verification Results

| Check | Result |
|-------|--------|
| REFERENCES.md contains REF-071 through REF-110 | PASS |
| New section headers (BMR, CT-RNN, SBI, Foundation Models, Hybrid) present | PASS |
| Equation Quick-Reference table gained 5+ new module rows | PASS |
| methods.md contains "Direct Observation Model" section | PASS |
| methods.md contains "Bayesian Model Reduction" section | PASS |
| methods.md contains "Simulation-Based Inference" section | PASS |
| methods.tex contains matching `\subsection{Direct Observation Model}` | PASS |
| methods.tex uses placeholder `\citep{}` keys with Zotero note | PASS |
| references.bib NOT edited in any of the three commits | PASS |

## Next Phase Readiness

Phase 27-02 (publication figure generation) can proceed: the reference
catalog and methods narrative provide the citation IDs and equation labels
that the figure captions and manuscript LaTeX integration depend on.
