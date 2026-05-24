---
created: 2026-05-24T16:30
title: Neural ODE DCM (Approach 2)
area: planning
files:
  - .planning/PROJECT.md:74-75
  - .planning/research/PITFALLS.md:127-142
  - .planning/research/ARCHITECTURE.md:30-33
  - .planning/REFERENCES.md:208-235
---

## Problem

v0.6.0 (Latent Circuit DCM) uses bilinear dynamics dx/dt = Ax + sum u_j B_j x + Cu to
distill RNN latent dynamics. This is a first-order Taylor approximation — it works near
fixed points but may miss nonlinear structure. Approach 2 replaces the bilinear constraint
with a learned Neural ODE: dx/dt = f_theta(x, u) where f_theta is a small neural network.
This is more expressive but less directly interpretable (W_rec in a nonlinear ODE != A
matrix in DCM). The key question is whether bilinear suffices (Nozari et al. 2024 says
yes for macroscopic BOLD; may differ for meso-circuit RNN dynamics) or whether learned
dynamics are needed.

This should be a separate milestone (v0.7.0+) after v0.6.0, because v0.6.0's results will
inform whether the bilinear approximation is sufficient. If bilinear DCM achieves R^2 >= 0.80
on RNN latents, the Neural ODE extension may be lower priority. If it fails, Approach 2
becomes urgent.

## Literature Discussed (2026-05-24 session)

- MINDy (Singh et al. 2020, REF-050): neural ODE for connectivity, point estimates only
- Nozari et al. 2024 (REF-051): linear models suffice for macroscopic BOLD (resting-state)
- Durstewitz et al. 2023 review (REF-052): rotational degeneracy and identifiability
- BACE (2025 preprint): behavior-adaptive connectivity as interpretable adjacency graphs
- Switching RNNs (NeurIPS 2024): RNNs with discrete weight switches for neural data
- GP-SLDS (NeurIPS 2024): Gaussian process switching linear dynamical systems
- Multiscale Neural ODEs for effective connectivity (Dec 2024)
- TVB / Virtual Brain Inference (Jirsa group, eLife 2025): whole-brain simulation-based inference
- Friston's rotational degeneracy pitfall (P8 in existing PITFALLS.md)

## Existing Architectural Support

- Protocol-based design (swappable ConnectivityPrior, ObservationModel)
- torchdiffeq already integrated for time-varying RHS
- Future ConnectivityPrior sketches in ARCHITECTURE.md: GPPriorA, SwitchingA, RNNPriorA
- v0.6.0 will provide: direct observation model, RNN training, BMR, latent extraction
- Only the dynamics model (neural state equation) needs swapping

## Key Design Decisions (deferred)

1. Whether to replace or augment the biophysical forward model
2. How to preserve interpretability when dynamics are learned (rotational degeneracy)
3. Coordinate regularization: alignment + cross-prediction penalties vs GNN equivariance
4. Whether to keep BOLD observation model or learn end-to-end
5. How to maintain Bayesian posterior inference with learned dynamics

## Solution

TBD — revisit after v0.6.0 latent circuit work completes. v0.6.0's misspecification
analysis (COMP-04) and linearization quality diagnostic (PIPE-03) will directly inform
whether Neural ODE extension is needed and what regimes it targets.
