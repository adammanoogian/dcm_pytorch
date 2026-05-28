---
created: 2026-05-28T16:30
title: ROI projection step for latent circuit DCM
area: planning
files:
  - src/pyro_dcm/rnn/latent_extraction.py
  - src/pyro_dcm/models/latent_circuit_dcm_model.py
  - src/pyro_dcm/foundation/parcellation.py
---

## Problem

After fitting DCM to PCA-reduced latent states from a trained model (Approach A),
the recovered A matrix describes connectivity between abstract PCA components, not
brain ROIs. There is no step to map circuit nodes back to anatomically meaningful
regions. This limits the neuroscience interpretability of Approach A results.

This is separate from Approach B (foundation models on ROI-level data), which
already operates on parcellated brain regions and does not need this projection.

Blocked until both a behavioral dataset (for model training) AND a neuroimaging
dataset (for spatial grounding of hidden units) are available. The spatial grounding
requires hidden units that map to brain locations (e.g., a model trained on
source-localized MEG parcels), so that PCA loadings have a spatial interpretation.

## Solution

TBD — candidate approaches:

1. Use PCA loading matrix (components_ shape N x H) to characterize spatial
   footprint of each circuit node, if hidden units have spatial structure
2. Train model on parcellated source-localized data so hidden units inherit
   spatial meaning; then PCA loadings give ROI contributions per circuit node
3. Post-hoc correlation between circuit node activations and ROI timeseries
