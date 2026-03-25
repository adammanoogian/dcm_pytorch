---
type: "research"
scope: "pitfalls"
updated: "2026-03-24"
---

# Pitfalls & Anti-Patterns

## P1: ODE Stiffness in Balloon Model [CRITICAL]

**Problem:** The Balloon-Windkessel model has fast (vasodilatory signal s, ~1s) and
slow (volume v, ~5-10s) time constants. This stiffness ratio can cause explicit solvers
(Euler, RK4) to require very small step sizes or produce NaN.

**Mitigation:**
- Use adaptive step-size solvers (Dopri5 via torchdiffeq) as default
- Test with dt=0.01s for 500s simulations (50,000 steps)
- Monitor `max(abs(dhemo/dt))` — if >100, solver is struggling
- Fallback: semi-implicit integration for hemodynamic states
- Validate: steady-state response to step input should match Buxton (1998) Fig. 3

**Ref:** REF-030, REF-002

---

## P2: A Matrix Eigenvalue Stability [CRITICAL]

**Problem:** If any eigenvalue of A has positive real part, neural dynamics diverge
exponentially. During inference, unconstrained optimization can push A unstable.

**Mitigation:**
- Parameterize A as A = -softplus(diag) + off_diag, ensuring negative diagonal
- Or: project A after each gradient step via A ← A - max(0, λ_max(A) + ε) * I
- Or: use prior that strongly penalizes positive eigenvalues
- Monitor: check `torch.linalg.eigvals(A).real.max()` < 0 after each SVI step

**Ref:** REF-001 (stability requirement is implicit in the DCM formulation)

---

## P3: CSD Normalization Mismatch with SPM [HIGH]

**Problem:** SPM's `spm_csd_mtf` uses specific normalization conventions for the
cross-spectral density that differ from scipy.signal.csd defaults. Mismatch here
will cause Phase 6 validation to fail completely.

**Mitigation:**
- Carefully document which normalization convention is used (one-sided vs two-sided,
  per-Hz vs per-bin, power vs amplitude)
- Implement CSD computation with configurable normalization
- Test: generate white noise, verify flat CSD with known variance
- Cross-validate against SPM on a simple known-spectrum test case BEFORE
  building the full spectral DCM

**Ref:** REF-010, REF-012

---

## P4: Complex Tensor Dtype Issues [MEDIUM]

**Problem:** Spectral DCM requires complex-valued tensors for transfer functions
and CSD. PyTorch complex support is mature but has gotchas:
- `torch.inverse` on complex tensors may be slower than `torch.linalg.solve`
- Autograd through complex operations has edge cases
- Pyro distributions don't natively support complex parameters

**Mitigation:**
- Represent complex CSD as real+imag stacked: shape (F, N, N, 2) not (F, N, N) complex
- Or: use `torch.complex64` / `torch.complex128` and verify autograd works
- Test gradient flow through `(iωI - A)⁻¹` before building full spectral model
- For Pyro: keep latent variables real, only use complex in the forward model

---

## P5: ELBO Gradient Variance [MEDIUM]

**Problem:** The ELBO gradient has high variance when the forward model involves
ODE integration (task DCM) or matrix inversion at many frequencies (spectral DCM).
This causes noisy optimization, especially early in training.

**Mitigation:**
- Use multiple ELBO samples per step (num_particles > 1 in Trace_ELBO)
- Learning rate warmup: start at 1e-5, warm to 1e-3 over 500 steps
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(guide.parameters(), 10.0)`
- Track ELBO variance across particles as a diagnostic
- For spectral DCM: consider subsampling frequencies per gradient step

**Ref:** REF-041

---

## P6: Amortization Gap [MEDIUM]

**Problem:** Amortized guides (normalizing flows) trade per-subject optimality for
speed. The "amortization gap" = per-subject ELBO - amortized ELBO can be large if:
- The training distribution of simulated data doesn't cover the test data well
- The flow architecture doesn't have enough capacity
- The summary network loses critical information

**Mitigation:**
- Train on a WIDE distribution of A matrices (not just the specific test config)
- Monitor amortization gap on held-out subjects
- Fine-tune: run a few SVI steps starting from the amortized posterior
- Architecture search: compare MAF vs NSF vs NAF for each DCM variant

**Ref:** REF-042, REF-043

---

## P7: rDCM Frequency Domain Assumptions [MEDIUM]

**Problem:** Regression DCM assumes stationarity and a specific HRF shape. Violations:
- Non-stationary data produces spectral leakage in DFT
- HRF variability across regions breaks the convolution assumption
- Short time series produce poor frequency resolution

**Mitigation:**
- Use sufficient time series length (>200 TRs for rDCM)
- Validate HRF assumption: compare with FIR-estimated HRFs
- Test with both canonical and flexible HRF basis sets
- Document frequency resolution limits: Δf = 1/(N·TR)

**Ref:** REF-020, REF-021

---

## P8: Rotational Degeneracy (v0.2 concern, monitor in v0.1) [FUTURE]

**Problem:** If neural dynamics are modeled with a neural ODE (v0.2), the state
space can undergo similarity transformations A → PAP⁻¹ that preserve trajectories
but produce scientifically contradictory connectivity matrices.

**Current approach:** Not applicable in v0.1 (bilinear model with explicit ROI-aligned
states). But architectural decisions in v0.1 should not preclude solutions in v0.2.

**Future mitigations:**
- Structural masking of the vector field
- Coordinate regularization (alignment + cross-prediction penalties)
- GNN equivariance constraints
- Total correlation penalties from disentangled representation learning

**Ref:** REF-050 (MINDy), REF-052 (Durstewitz review)

---

## P9: SPM Validation: Laplace vs SVI Posterior Discrepancy [HIGH]

**Problem:** SPM uses Variational Laplace (a second-order Laplace approximation),
not stochastic variational inference. The posteriors may differ in:
- Mode location (VL finds a mode; SVI finds an ELBO-weighted average)
- Posterior width (VL uses Hessian at mode; SVI uses guide flexibility)
- These are EXPECTED differences, not bugs — but must be documented

**Mitigation:**
- Validate on unimodal, well-identified cases where VL ≈ SVI
- Report both MAP-like (mode of guide) and posterior mean from SVI
- If discrepancy > 10%, investigate: is the posterior multimodal? is VL overconfident?
- Use NUTS as the ground-truth arbiter between VL and SVI

**Ref:** REF-040

---

## P10: Memory Pressure from ODE Backpropagation [MEDIUM]

**Problem:** Backpropagating through 50,000 ODE steps (500s at dt=0.01) uses
enormous memory if storing all intermediate states.

**Mitigation:**
- Use `odeint_adjoint` (constant memory, recomputes forward pass)
- Subsample time points: integrate at dt=0.01 but only compare BOLD at TR intervals
- Chunk long simulations: integrate in 50s segments, checkpoint between segments
- Monitor: GPU memory usage should stay < 8GB for 5-region, 500s simulations
