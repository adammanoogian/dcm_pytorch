---
phase: 27-publication-artifacts
plan: 03
backfilled: 2026-06-09
subsystem: publication-artifacts
tags: [equations-reference, cross-consistency, methods, checkpoint, v0.6.0]

dependency-graph:
  requires: [27-01, 27-02]
  provides: [equations-quick-reference-v0.6.0]
  affects: []

tech-stack:
  added: []
  patterns: [markdown-equation-tables, reference-key-index, source-file-index]

file-tracking:
  key-files:
    created: []
    modified:
      - docs/03_methods_reference/equations.md

decisions:
  - id: 27-03-D1
    summary: "Cross-consistency report not persisted as an artifact"
    detail: "The plan's Task 1 step 2 asked for a printed consistency report. No report was committed to .planning/ or docs/ and none is embedded in equations.md. The equations.md update was committed but the consistency check left no recorded output."
  - id: 27-03-D2
    summary: "checkpoint:human-verify gate has no recorded approval"
    detail: "Task 2 was a blocking checkpoint:human-verify gate (review pipeline schematic + methods narrative). No 'approved' resume-signal or equivalent verification record exists in commit history, STATE.md, or ROADMAP.md. Checkpoint not recorded as verified."

metrics:
  duration: ~single-commit doc edit
  completed: 2026-05-28
---

# Phase 27 Plan 03: Equations Quick-Reference Update + Cross-Consistency Summary

Updated the equations quick-reference with all v0.6.0 modules (bilinear extension,
direct observation model, CT-RNN, Bayesian model reduction) and expanded the
reference key and source-file index. The cross-consistency check and the
human-verify checkpoint were specified by the plan but left no committed record.

## Completed Tasks

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Update equations.md and cross-check consistency | 5ac7b05 | docs/03_methods_reference/equations.md |
| 2 | checkpoint:human-verify (pipeline schematic + methods narrative) | (none) | -- |

## What Was Built

### docs/03_methods_reference/equations.md (+55 lines, -2 lines)

Four new equation sections plus expanded indices, all in the existing
Markdown table format (Name / Formula / Reference / Implementation):

1. **Bilinear Extension** (added to Task-Based DCM section):
   - Bilinear state equation `dx/dt = Ax + Sum_j u_j B^(j) x + Cu` -- [REF-001] Eq. 1, `neural_state.py`
   - Effective connectivity `A_eff(t) = A + Sum_j u_j(t) B^(j)` -- [REF-001], `neural_state.py`
   - B parameterization `B_j = mask_j * B_free_j, diag=0` -- SPM12 convention, `task_dcm_model.py`

2. **Direct Observation Model** (new section):
   - Direct observation `y(t) = C_obs x(t) + epsilon` -- Phase 20, `latent_circuit_dcm_model.py`
   - LC A prior `A_free ~ N(0, 1/16)` -- Phase 20-03 decision
   - LC noise model `epsilon ~ N(0, 1/noise_prec)`
   - LC self-inhibition `a_ii = -exp(A_free_ii) * self_inhibition` (default 1.0 Hz)

3. **CT-RNN** (new section):
   - CT-RNN dynamics `tau dh/dt = -h + f(W_rec h + W_in u + b)` -- [REF-080], `continuous_time_rnn.py`
   - Euler discretization `h_{t+1} = h_t + (dt/tau)(-h_t + f(...))` -- [REF-080]
   - Output readout `y = W_out h + b_out` -- [REF-080]
   - Output R-squared gate `R2 >= 0.90` on PCA-projected readout -- Phase 21, `latent_extraction.py`

4. **Bayesian Model Reduction** (new section):
   - Reduced posterior precision `Sigma_r_post^-1 = Sigma_f^-1 + Sigma_r^-1 - Sigma_0^-1` -- [REF-070] Eq. 4-5, `bmr.py`
   - Reduced posterior mean -- [REF-070] Eq. 4-5
   - Change in log evidence `delta_F` -- [REF-070] Eq. 6-8
   - Circuit selection: enumerate `2^k - 1` reduced models, rank by `delta_F` -- [REF-071]

### Reference Key Expansion

Added REF-060 through REF-110 to the reference key table: REF-060 (Pyro),
REF-070 (Friston & Penny 2011 post-hoc BMS), REF-071 (Rosa et al. 2012),
REF-080 (Langdon & Engel 2025), REF-081 (Sussillo & Barak 2013),
REF-090 (Goncalves et al. 2020), REF-091 (Hashemi et al. 2024),
REF-110 (Kingma & Welling 2014 VAE).

### Source File Index Expansion

Added/annotated entries for `models/latent_circuit_dcm_model.py`,
`rnn/continuous_time_rnn.py`, `rnn/latent_extraction.py`, and
`model_selection/bmr.py`; annotated `neural_state.py` and
`task_dcm_model.py` as bilinear.

## Deviations from Plan

### Noted Behaviors

**1. [Gap] Cross-consistency report not persisted**

- Task 1 step 2 specified printing a consistency report (REF-XXX IDs in
  methods.md vs REFERENCES.md, methods.tex citation keys, figure function
  names vs methods sections, flagged placeholder keys).
- No such report was committed to `.planning/`, `docs/`, or embedded in
  equations.md (grep for "consistency", "orphan", "mismatch", "placeholder"
  in the committed file returns nothing).
- The equations.md edit was committed; the consistency check left no recorded
  output.

**2. [Gap] checkpoint:human-verify gate not recorded as verified**

- Task 2 was a blocking `checkpoint:human-verify` gate requiring user review
  of `figures/pipeline_schematic.png` and the methods.md narrative for
  scientific accuracy, with an "approved" resume-signal.
- No "approved" record or equivalent verification note exists in commit
  history, STATE.md, or ROADMAP.md.
- **Checkpoint not recorded as verified.**

**3. [Note] ROADMAP item still unchecked**

- ROADMAP.md line 670 lists `27-03-PLAN.md` with an unchecked `[ ]` box,
  consistent with the checkpoint never being closed out.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 27-03-D1 | Consistency report not persisted as an artifact | Only the equations.md update was committed; the printed check left no recorded output |
| 27-03-D2 | Human-verify checkpoint has no recorded approval | No "approved" signal in git/STATE/ROADMAP; recorded as not verified |

## Verification Results

| Check | Result |
|-------|--------|
| equations.md has Bilinear Extension section | PASS |
| equations.md has Direct Observation Model section | PASS |
| equations.md has CT-RNN section | PASS |
| equations.md has Bayesian Model Reduction section | PASS |
| grep "latent_circuit_dcm_model" in equations.md returns results | PASS |
| grep "continuous_time_rnn" in equations.md returns results | PASS |
| REF-070/REF-080 citations present and resolve in reference key | PASS |
| Cross-consistency report committed as artifact | NOT FOUND |
| checkpoint:human-verify approved | NOT RECORDED |

## Next Phase Readiness

The equations quick-reference is complete for v0.6.0. Before the paper is
assembled, the cross-consistency check should be re-run and its report
captured, and the blocking human-verify checkpoint (pipeline schematic +
methods narrative scientific review, plus Zotero entries for REF-080,
REF-100, and any other pending references) should be completed and recorded.
