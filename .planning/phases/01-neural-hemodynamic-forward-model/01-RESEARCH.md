# Phase 1: Neural & Hemodynamic Forward Model + Task-DCM Simulator - Research

**Researched:** 2026-03-25
**Domain:** Biophysical hemodynamic modeling, ODE integration, DCM forward model
**Confidence:** HIGH

## Summary

This research covers the complete technical domain for implementing Phase 1: the bilinear neural state equation, Balloon-Windkessel hemodynamic ODE, BOLD signal observation equation, ODE integration via torchdiffeq, and a task-DCM simulator. The core mathematical specifications are well-established in the DCM literature (Friston et al. 2003, Stephan et al. 2007), and the reference implementation in SPM12 provides authoritative guidance on parameterization conventions.

The key technical finding is that SPM12 uses **log-space representation** for hemodynamic states (flow, volume, deoxyhemoglobin) to enforce positivity, and applies the transformation `a = -exp(A_diag)/2` for self-connections to ensure stability. The torchdiffeq library provides the needed ODE solvers (`dopri5`, `rk4`, `euler`) with GPU support and differentiable integration, but requires careful handling of time-varying inputs (experimental stimulus u(t)) via interpolation and `grid_points` for discontinuities. The coupled ODE system [neural; hemodynamic] should be integrated as a single flattened state vector.

**Primary recommendation:** Follow SPM12's `spm_fx_fmri.m` conventions exactly -- log-space hemodynamic states, `a = -exp(A_diag)/2` self-connection parameterization, and the specific parameter defaults from the SPM priors. Use torchdiffeq `dopri5` as the default solver with `grid_points` at stimulus discontinuity times to handle piecewise-constant experimental inputs.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.2+ | Tensor computation, autograd | Required by Pyro and torchdiffeq |
| torchdiffeq | latest (pip) | ODE integration (`odeint`, `odeint_adjoint`) | PyTorch-native, GPU, adjoint method for O(1) memory |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | 1.11+ | `solve_ivp` for validation against torchdiffeq | Cross-validate ODE solutions before trusting torchdiffeq |
| numpy | 1.24+ | Array operations for test data generation | Test fixtures and reference computations |
| pytest | 8.0+ | Test framework | All unit and integration tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchdiffeq | torchode | torchode has true parallel batch integration (independent step sizes per batch element), but less mature and less tested with adjoint method |
| torchdiffeq | torchdyn | Higher-level API but less control over solver options; torchdiffeq is more established |
| dopri5 | implicit_adams | Better for stiff systems but slower; Balloon model stiffness is mild enough for dopri5 with tight tolerances |

**Installation:**
```bash
pip install torch torchdiffeq scipy numpy pytest
```

## Architecture Patterns

### Recommended Project Structure
```
src/pyro_dcm/
  forward_models/
    __init__.py
    neural_state.py       # dx/dt = Ax + Cu  [REF-001 Eq.1]
    balloon_model.py      # ds,df,dv,dq ODEs [REF-002 Eq.2-5]
    bold_signal.py        # y = V0[k1(1-q) + k2(1-q/v) + k3(1-v)] [REF-002 Eq.6]
  utils/
    __init__.py
    ode_integrator.py     # Wrapper around torchdiffeq.odeint
  simulators/
    __init__.py
    task_simulator.py     # End-to-end synthetic BOLD generation
tests/
  test_neural_state.py
  test_balloon.py
  test_bold_signal.py
  test_ode_integrator.py
  test_task_simulator.py
```

### Pattern 1: Composable ODE Modules with Combined Right-Hand Side

**What:** Each forward model module (neural_state, balloon_model) defines its own derivative function. A combiner assembles them into a single ODE right-hand side for torchdiffeq.

**When to use:** Always -- this is the locked decision from CONTEXT.md.

**Example:**
```python
# Source: SPM12 spm_fx_fmri.m + torchdiffeq API
class CoupledDCMSystem(nn.Module):
    """Combined neural + hemodynamic ODE for torchdiffeq.

    State vector layout: [x(N), s(N), lnf(N), lnv(N), lnq(N)]
    where N = number of regions.

    Implements [REF-001] Eq. 1 (neural) + [REF-002] Eq. 2-5 (hemodynamic).
    """
    def __init__(self, A, C, hemo_params, input_fn):
        super().__init__()
        self.A = A              # (N, N) effective connectivity
        self.C = C              # (N, M) driving input weights
        self.hemo_params = hemo_params  # dict of per-region params
        self.input_fn = input_fn  # callable: t -> u(t), shape (M,)

    def forward(self, t, state):
        """Compute dx/dt for the full coupled system.

        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar).
        state : torch.Tensor
            Flattened state vector, shape (5*N,).

        Returns
        -------
        torch.Tensor
            Time derivatives, shape (5*N,).
        """
        N = self.A.shape[0]
        x  = state[:N]            # neural activity
        s  = state[N:2*N]         # vasodilatory signal
        lnf = state[2*N:3*N]     # log blood flow
        lnv = state[3*N:4*N]     # log blood volume
        lnq = state[4*N:5*N]     # log deoxyhemoglobin

        u = self.input_fn(t)      # (M,) experimental input

        # Neural state equation [REF-001 Eq. 1]
        dx = self.A @ x + self.C @ u

        # Hemodynamic states [REF-002 Eq. 2-5]
        # Exponentiate log-space states for positivity
        f = torch.exp(lnf)
        v = torch.exp(lnv)
        q = torch.exp(lnq)

        kappa = self.hemo_params['kappa']  # ~0.65
        gamma = self.hemo_params['gamma']  # ~0.41
        tau   = self.hemo_params['tau']    # ~0.98
        alpha = self.hemo_params['alpha']  # ~0.32
        E0    = self.hemo_params['E0']     # ~0.34

        # Vasodilatory signal
        ds = x - kappa * s - gamma * (f - 1)

        # Blood flow (in log-space: d(lnf)/dt = s/f)
        dlnf = s / f

        # Blood volume (in log-space)
        dlnv = (f - v.pow(1.0 / alpha)) / (tau * v)

        # Deoxyhemoglobin (in log-space)
        E_f = 1.0 - (1.0 - E0).pow(1.0 / f)  # oxygen extraction
        dlnq = (f * E_f / E0 - v.pow(1.0 / alpha) * q / v) / (tau * q)

        return torch.cat([dx, ds, dlnf, dlnv, dlnq])
```

### Pattern 2: Time-Varying Input via Interpolation + grid_points

**What:** Experimental stimuli u(t) are piecewise-constant (block designs) or event-related. The ODE solver needs these as a continuous function of t, with discontinuity handling.

**When to use:** Always for task-based DCM with block/event designs.

**Example:**
```python
# Source: torchdiffeq GitHub issue #128
class PiecewiseConstantInput:
    """Piecewise-constant stimulus function for torchdiffeq.

    Provides u(t) and grid_points for adaptive solver restarts.
    """
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        """
        Parameters
        ----------
        times : torch.Tensor
            Onset times, shape (K,), sorted ascending.
        values : torch.Tensor
            Input values at each onset, shape (K, M).
        """
        self.times = times
        self.values = values

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return u(t) via searchsorted."""
        idx = torch.searchsorted(self.times, t.detach(), right=True) - 1
        idx = idx.clamp(min=0, max=len(self.values) - 1)
        return self.values[idx]

    @property
    def grid_points(self) -> torch.Tensor:
        """Discontinuity times for solver restarts."""
        return self.times

# Usage with torchdiffeq:
solution = odeint(
    ode_func, y0, t_eval,
    method='dopri5',
    rtol=1e-5, atol=1e-7,
    options={'grid_points': input_fn.grid_points, 'eps': 1e-6}
)
```

### Pattern 3: BOLD Signal as Algebraic Observation (Not ODE)

**What:** The BOLD signal equation is purely algebraic -- it maps hemodynamic states (v, q) to observed signal. It is NOT part of the ODE system; it is applied after integration.

**When to use:** Always. bold_signal.py is a function, not an ODE module.

**Example:**
```python
# Source: SPM12 spm_gx_fmri.m, [REF-002] Eq. 6
def bold_signal(v: torch.Tensor, q: torch.Tensor,
                E0: float = 0.34, V0: float = 0.02) -> torch.Tensor:
    """BOLD signal observation equation [REF-002] Eq. 6.

    Parameters
    ----------
    v : torch.Tensor
        Blood volume (linear space, NOT log), shape (..., N).
    q : torch.Tensor
        Deoxyhemoglobin content (linear space), shape (..., N).
    E0 : float
        Resting oxygen extraction fraction.
    V0 : float
        Resting venous blood volume fraction.

    Returns
    -------
    torch.Tensor
        BOLD percent signal change, shape (..., N).
    """
    k1 = 7.0 * E0
    k2 = 2.0
    k3 = 2.0 * E0 - 0.2
    return V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
```

### Pattern 4: nn.Module ODE Function for torchdiffeq

**What:** The ODE function MUST be an `nn.Module` when using `odeint_adjoint`. Parameters (A, C, hemo_params) are stored as module attributes and updated before each solve call.

**When to use:** Always when differentiating through the ODE solution (required for inference later).

**Key insight from torchdiffeq FAQ:** The func must be an `nn.Module` for `odeint_adjoint`. Parameters must be actively used in the `forward` method to receive gradients. Reassigning `nn.Parameter` attributes after `__init__` breaks optimizer registration.

### Anti-Patterns to Avoid

- **Integrating BOLD as part of the ODE:** The BOLD equation is algebraic, not differential. Including it in the ODE state vector wastes computation and adds unnecessary coupling.
- **Linear-space hemodynamic states:** Without positivity constraints, flow/volume/deoxyhemoglobin can go negative during integration, causing NaN. Use log-space states as SPM does.
- **Ignoring grid_points for block designs:** Adaptive solvers (dopri5) will waste enormous numbers of steps trying to integrate through sharp stimulus onsets. Always provide grid_points at discontinuity times.
- **Using numpy interpolation for u(t):** Numpy operations break the PyTorch computation graph. All interpolation must use torch operations for gradient flow.
- **Separate integration of neural and hemodynamic ODEs:** These are coupled (neural activity x drives vasodilatory signal s). They must be integrated as one system to avoid splitting errors.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration | Custom Euler/RK4 loop | `torchdiffeq.odeint` | Adaptive stepping, error control, adjoint backprop, GPU support |
| Adaptive step sizing | Manual step rejection logic | `dopri5` method in torchdiffeq | Embedded Runge-Kutta pairs handle this automatically |
| Batch ODE solving | Loop over subjects | torchdiffeq batch mode (stack y0 as (B, D)) | GPU parallelism, single solver call |
| Discontinuity handling | Manual time-splitting | `grid_points` option in torchdiffeq | Solver handles restart automatically |

**Key insight:** torchdiffeq handles all the numerical ODE machinery. The project's job is to define the correct right-hand side function and the correct observation equation. Do not re-implement any ODE solver internals.

## Common Pitfalls

### Pitfall 1: Hemodynamic State Positivity (CRITICAL)
**What goes wrong:** Blood flow (f), volume (v), and deoxyhemoglobin (q) are physically positive quantities. Without constraints, ODE integration can produce negative values, leading to NaN in downstream computations (e.g., `v^(1/alpha)` with negative v).
**Why it happens:** Large neural inputs or aggressive A matrix values push hemodynamic states through zero.
**How to avoid:** Store hemodynamic states in log-space (lnf, lnv, lnq) and exponentiate when computing derivatives, exactly as SPM12 does. The chain rule transforms the ODE: `d(lnf)/dt = (1/f) * df/dt`.
**Warning signs:** NaN in BOLD output, negative flow/volume values during debugging.

### Pitfall 2: A Matrix Self-Connection Parameterization (CRITICAL)
**What goes wrong:** If diagonal of A is unconstrained, self-connections can become positive, causing exponential divergence of neural states.
**Why it happens:** During optimization/inference, gradient updates can push diagonal elements positive.
**How to avoid:** Follow SPM convention: `A_diag = -exp(A_free_diag) / 2`. This guarantees negative self-connections with a default of -0.5 Hz when the free parameter is 0. Off-diagonal elements remain unconstrained (direct Hz values).
**Warning signs:** Neural state x grows unboundedly, ODE solver takes increasingly many steps.

### Pitfall 3: Stimulus Discontinuities with Adaptive Solvers (HIGH)
**What goes wrong:** Block-design stimuli are piecewise-constant. The derivative of the neural state dx/dt has jumps at stimulus onset/offset times. Adaptive solvers detect this as high local error and shrink step size to near-zero, making integration extremely slow or failing.
**Why it happens:** dopri5 uses polynomial interpolation that assumes smooth RHS.
**How to avoid:** Pass stimulus transition times as `grid_points` to torchdiffeq options. This forces the solver to stop and restart at discontinuity times. Also set `eps=1e-6` to avoid evaluating exactly at the discontinuity.
**Warning signs:** Integration takes >100x expected time, NFE (number of function evaluations) exceeds 100,000.

### Pitfall 4: BOLD Signal Scale Mismatch (MEDIUM)
**What goes wrong:** The BOLD signal should produce ~0.5-5% signal change. If parameters or scaling are wrong, the signal may be orders of magnitude off, making later inference impossible.
**Why it happens:** Mismatch between V0, k1/k2/k3 definitions, or hemodynamic parameter values. Different sources use different conventions (V0=0.02 vs V0=0.04; simplified vs full BOLD constants).
**How to avoid:** Use the simplified Buxton form consistently: k1=7*E0, k2=2, k3=2*E0-0.2, V0=0.02. Validate against a known reference: a 30s block stimulus should produce ~1-3% BOLD change in the directly driven region.
**Warning signs:** BOLD output is <0.01% or >10% signal change for standard block designs.

### Pitfall 5: ODE Tolerances Too Loose or Too Tight (MEDIUM)
**What goes wrong:** Too loose (rtol=1e-3, atol=1e-3) gives inaccurate hemodynamic dynamics; too tight (rtol=1e-10, atol=1e-12) makes integration extremely slow.
**Why it happens:** Default torchdiffeq tolerances may not be tuned for hemodynamic timescales.
**How to avoid:** Use rtol=1e-5, atol=1e-7 as defaults. Validate by comparing against rtol=1e-8, atol=1e-10 reference solution -- if peak BOLD differs by >0.1%, tighten tolerances.
**Warning signs:** Jerky BOLD time courses (too loose), >10 minutes for 500s simulation (too tight).

### Pitfall 6: Memory Pressure from Long Simulations (MEDIUM)
**What goes wrong:** A 500s simulation at dt=0.01 has ~50,000 internal steps. Backpropagating through all of them with standard `odeint` uses enormous memory.
**Why it happens:** Standard backprop stores all intermediate activations.
**How to avoid:** Use `odeint_adjoint` instead of `odeint` for gradient computation. This uses O(1) memory by recomputing the forward pass during backward. For forward-only simulation (data generation), `odeint` is fine.
**Warning signs:** CUDA out-of-memory errors during inference/training; RAM usage >16GB for small networks.

### Pitfall 7: Oxygen Extraction Function Numerical Issues (LOW)
**What goes wrong:** The extraction function `E(f, E0) = 1 - (1 - E0)^(1/f)` can produce NaN when f is very small (near zero) because `1/f -> inf`.
**Why it happens:** Log-space states prevent f from being exactly zero, but very negative lnf values push f close to zero.
**How to avoid:** Clamp f to a minimum value (e.g., f = max(f, 1e-6)) before computing E(f, E0). Or equivalently, clamp lnf to a minimum (e.g., lnf >= -14).
**Warning signs:** NaN specifically in the deoxyhemoglobin derivative.

## Code Examples

### Complete BOLD Forward Model Pipeline
```python
# Source: SPM12 spm_fx_fmri.m + spm_gx_fmri.m, verified against
# [REF-001] Eq. 1, [REF-002] Eq. 2-6

import torch
from torch import nn
from torchdiffeq import odeint

# Step 1: Define hemodynamic parameter defaults [REF-002 Table 1]
DEFAULT_HEMO_PARAMS = {
    'kappa': 0.65,    # s^-1, signal decay
    'gamma': 0.41,    # s^-1, flow-dependent elimination
    'tau':   0.98,    # s, hemodynamic transit time
    'alpha': 0.32,    # Grubb's exponent (vessel stiffness)
    'E0':    0.34,    # resting oxygen extraction fraction
}

# Step 2: BOLD signal constants [REF-002 Eq. 6, simplified Buxton form]
BOLD_CONSTANTS = {
    'V0': 0.02,       # resting venous blood volume fraction
    # k1, k2, k3 derived from E0 at runtime
}

# Step 3: Initial conditions (steady state)
# Neural: x = 0 (no activity)
# Vasodilatory signal: s = 0
# Flow: f = 1 (baseline) -> lnf = 0
# Volume: v = 1 (baseline) -> lnv = 0
# Deoxyhemoglobin: q = 1 (baseline) -> lnq = 0

def make_initial_state(n_regions: int) -> torch.Tensor:
    """Create initial state at hemodynamic steady state."""
    return torch.zeros(5 * n_regions)

# Step 4: Integration
# solution = odeint(coupled_system, y0, t_eval, method='dopri5',
#                   rtol=1e-5, atol=1e-7,
#                   options={'grid_points': stim_times, 'eps': 1e-6})

# Step 5: Extract hemodynamic states and compute BOLD
# v = exp(solution[:, 3*N:4*N])
# q = exp(solution[:, 4*N:5*N])
# bold = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))

# Step 6: Downsample to TR
# bold_observed = bold[::tr_samples]  # or use interpolation
```

### SPM A Matrix Parameterization Convention
```python
# Source: SPM12 spm_fx_fmri.m, SPM/DCM_units wikibook

def parameterize_A(A_free: torch.Tensor) -> torch.Tensor:
    """Convert free parameters to effective connectivity matrix.

    SPM convention:
    - Diagonal: a_ii = -exp(A_free_ii) / 2
      (guarantees negative self-connections, default -0.5 Hz)
    - Off-diagonal: a_ij = A_free_ij (unconstrained, in Hz)

    Parameters
    ----------
    A_free : torch.Tensor
        Free parameters, shape (N, N).

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix A, shape (N, N).
    """
    diag_mask = torch.eye(A_free.shape[0], dtype=torch.bool,
                          device=A_free.device)
    A = A_free.clone()
    A[diag_mask] = -torch.exp(A_free[diag_mask]) / 2.0
    return A
```

### SPM Prior Specifications
```python
# Source: SPM12 spm_dcm_fmri_priors.m

# A matrix priors (one-state DCM, pA=64):
# - Off-diagonal mean: 0 (connection present) or fixed at 0 (absent)
# - Off-diagonal variance: 1/64 = 0.015625
# - Diagonal (free param, before -exp/2 transform):
#     mean: 0 (maps to -0.5 Hz self-connection)
#     variance: 1/64

# C matrix priors:
# - Mean: 0
# - Variance: 1 (for present connections)

# Hemodynamic priors (log-space, centered at default):
# - log(kappa/0.64): mean=0, var=1/256
# - log(tau/2.0): mean=0, var=1/256
# - epsilon (BOLD scaling): mean=0, var=1/256
# Note: SPM uses H(1)=0.64, H(3)=2.0 as base values
# that differ slightly from Stephan 2007's kappa=0.65, tau=0.98
```

### Simulator Pattern
```python
# Source: Project architecture decision

def simulate_task_dcm(
    A: torch.Tensor,           # (N, N) effective connectivity
    C: torch.Tensor,           # (N, M) driving input weights
    stimulus: dict,            # {'times': Tensor, 'values': Tensor}
    hemo_params: dict | None,  # per-region hemodynamic params
    duration: float,           # seconds
    dt: float = 0.01,          # integration step hint (for fixed-step)
    TR: float = 2.0,           # repetition time for BOLD sampling
    SNR: float = 5.0,          # signal-to-noise ratio
    solver: str = 'dopri5',    # ODE solver
    device: str = 'cpu',
) -> dict:
    """Generate synthetic task-DCM BOLD data.

    Returns dict with keys: 'bold', 'neural', 'hemodynamic',
    'times', 'params'.
    """
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Linear-space hemodynamic states | Log-space (lnf, lnv, lnq) in SPM12 | SPM12 (2014+) | Prevents negative physiological values, improves numerical stability |
| Fixed A diagonal | Constrained via `-exp(A)/2` | SPM12 | Guarantees stability without eigenvalue projection |
| Fixed-step Euler integration | Adaptive dopri5 via torchdiffeq | torchdiffeq (2018+) | Automatic error control, GPU support, adjoint backprop |
| Full bilinear model (A+B+C) | Start with A+C only (no modulatory B) | Project decision | Simpler first phase; B matrices added later |
| V0=0.04 (4%) older convention | V0=0.02 (2%) or field-dependent | Stephan 2007+ | More accurate for modern field strengths |

**Deprecated/outdated:**
- Linear-space hemodynamic states: replaced by log-space in SPM12 for numerical stability
- Simple Euler integration for hemodynamic ODE: replaced by adaptive solvers; Euler requires dt<0.001 for stability

## SPM12 Implementation Details (Authoritative Reference)

### State Variable Ordering in spm_fx_fmri.m
Per region, SPM12 uses 5 states:
1. `x(:,1)` -- excitatory neuronal activity (linear space)
2. `x(:,2)` -- vasodilatory signal s (linear space)
3. `x(:,3)` -- regional cerebral blood flow ln(f) (log space)
4. `x(:,4)` -- venous blood volume ln(v) (log space)
5. `x(:,5)` -- deoxyhemoglobin content ln(q) (log space)

### SPM Hemodynamic Defaults (from spm_fx_fmri.m)
```
H(1) = 0.64   -- signal decay (kappa, s^-1)
H(2) = 0.32   -- autoregulation (gamma, s^-1)
H(3) = 2.00   -- transit time (tau * something -- note: NOT 0.98)
H(4) = 0.32   -- Grubb's exponent (alpha)
H(5) = 0.40   -- resting oxygen extraction (E0)
```

**IMPORTANT discrepancy:** SPM12 uses `H(1)=0.64, H(2)=0.32, H(3)=2.0, H(5)=0.40` while Stephan 2007 Table 1 cites `kappa=0.65, gamma=0.41, tau=0.98, E0=0.34`. The SPM12 source code values are the authoritative implementation. The Stephan paper values are from the paper text. **Follow SPM12 code values** per the "when in doubt, follow SPM convention" principle.

### SPM12 Hemodynamic ODEs (Log-Space Form)
From spm_fx_fmri.m, after `exp(x(:,3:5))`:
```
f(:,2) = x(:,1) - sd*x(:,2) - H(2)*(x(:,3) - 1)     [ds/dt]
f(:,3) = x(:,2) / x(:,3)                               [d(lnf)/dt]
f(:,4) = (x(:,3) - x(:,4)^(1/H(4))) / (tt * x(:,4))  [d(lnv)/dt]
f(:,5) = (ff*x(:,3) - fv*x(:,5)/x(:,4)) / (tt*x(:,5)) [d(lnq)/dt]
```
Where:
- `sd = H(1) * exp(P.decay)` (prior mean of P.decay = 0, so default sd = 0.64)
- `tt = H(3) * exp(P.transit)` (prior mean of P.transit = 0, so default tt = 2.0)
- `fv = x(:,4)^(1/H(4))` (outflow)
- `ff = (1 - (1-H(5))^(1/x(:,3))) / H(5)` (extraction/E0)

### SPM12 A Matrix Construction
From spm_fx_fmri.m:
```matlab
SE = diag(P.A);                        % log self-inhibition
EE = P.A - diag(exp(SE)/2 + SE);      % effective connectivity
```
This means: off-diag are direct, diag undergoes `-exp(diag)/2` transformation, and the raw diagonal value is subtracted to avoid double-counting.

### SPM12 BOLD Observation (spm_gx_fmri.m)
States exponentiated: `v = exp(x(:,4)), q = exp(x(:,5))`
BOLD signal: `g = V0*(k1 - k1*q + k2 - k2*q/v + k3 - k3*v)`
Where k1, k2, k3 depend on epsilon parameter (intravascular/extravascular ratio).

## Open Questions

1. **SPM parameter discrepancy: which defaults to use?**
   - What we know: SPM12 code uses H=[0.64, 0.32, 2.0, 0.32, 0.40]; Stephan 2007 paper cites [0.65, 0.41, 0.98, 0.32, 0.34]. The CONTEXT.md references Stephan 2007 values.
   - What's unclear: The `H(3)=2.0` in SPM is a transit time base that gets multiplied by `exp(P.transit)`, while Stephan's `tau=0.98` is the direct transit time. These may be consistent if SPM's `P.transit` prior mean accounts for this difference.
   - Recommendation: Use SPM12 code values as the implementation reference (H(1)=0.64 etc.) since the project rule is "follow SPM convention." Document the discrepancy and how the free parameters (P.decay, P.transit) relate the two.

2. **BOLD constants: simplified vs. field-strength-dependent?**
   - What we know: Simplified form uses k1=7*E0, k2=2, k3=2*E0-0.2, V0=0.02. SPM12 uses the full Stephan 2007 form with epsilon parameter and field-strength constants (1.5T values).
   - What's unclear: Whether to implement the simplified or full form for v0.1.
   - Recommendation: Implement both. Use the simplified form as default for the simulator; make the full field-dependent form available as an option. Cite [REF-002] Eq. 6 for both.

3. **torchdiffeq implicit solvers for stiff cases?**
   - What we know: torchdiffeq has `implicit_adams` solver. The Balloon model has mild stiffness (fast s ~1s, slow v ~5-10s). dopri5 should handle this.
   - What's unclear: Whether extreme parameter values during inference might increase stiffness to the point where dopri5 fails.
   - Recommendation: Default to dopri5. Include `implicit_adams` as a fallback option in the ODE integrator wrapper. Monitor NFE (number of function evaluations) as a stiffness diagnostic.

4. **Batch integration for multiple regions vs. single system?**
   - What we know: torchdiffeq supports batch integration with stacked initial conditions. The hemodynamic model is per-region but coupled through neural activity.
   - What's unclear: Whether the hemodynamic ODEs should use batch dimensions (one batch element per region) or be part of the single coupled state vector.
   - Recommendation: Single coupled state vector [x; s; lnf; lnv; lnq] (shape 5N) because neural states couple regions. Batch dimension reserved for multiple subjects/simulations in the simulator.

## Sources

### Primary (HIGH confidence)
- SPM12 `spm_fx_fmri.m` source code (https://github.com/spm/spm12/blob/main/spm_fx_fmri.m) -- authoritative implementation of neural state + hemodynamic ODE
- SPM12 `spm_gx_fmri.m` source code (https://github.com/spm/spm12/blob/main/spm_gx_fmri.m) -- authoritative BOLD observation equation
- SPM12 `spm_dcm_fmri_priors.m` source code (https://github.com/spm/spm12/blob/main/spm_dcm_fmri_priors.m) -- authoritative prior specifications
- torchdiffeq GitHub repo (https://github.com/rtqichen/torchdiffeq) -- API documentation, FAQ, FURTHER_DOCUMENTATION.md
- torchdiffeq issue #128 (https://github.com/rtqichen/torchdiffeq/issues/128) -- handling time-varying inputs with grid_points
- torchdiffeq issue #178 (https://github.com/rtqichen/torchdiffeq/issues/178) -- passing parameters to ODE function
- SPM/DCM_units wikibook (https://en.wikibooks.org/wiki/SPM/DCM_units) -- A matrix parameterization convention
- DCM tutorial Zeidman et al. 2019 (https://pmc.ncbi.nlm.nih.gov/articles/PMC6711459/) -- complete DCM specification including J matrix and priors

### Secondary (MEDIUM confidence)
- Stephan et al. 2007 via PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC2636182/) -- Balloon model equations and parameter values
- Python Balloon-Windkessel reference impl (https://github.com/ito-takuya/HemodynamicResponseModeling/blob/master/BalloonWindkessel.py) -- independent Python implementation for cross-validation
- DCSEM repo (https://github.com/karahanyilmazer/dcsem) -- Python DCM simulation with balloon model

### Tertiary (LOW confidence)
- Various web search results on BOLD percent signal change (typical 1-3% at 1.5T-3T)
- fMRI SNR literature (typical tSNR 50-100, CNR for activation ~10)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- torchdiffeq API verified from GitHub README, FAQ, and issue tracker; SPM12 source code read directly
- Architecture: HIGH -- patterns derived from SPM12 reference implementation and torchdiffeq documentation; locked decisions from CONTEXT.md
- Pitfalls: HIGH -- log-space states, A matrix parameterization, and grid_points for discontinuities all verified from SPM12 source code and torchdiffeq issues
- SPM parameter defaults: HIGH -- read directly from spm_fx_fmri.m and spm_dcm_fmri_priors.m source code
- BOLD signal scale: MEDIUM -- approximate ranges from literature, exact values depend on parameter choices

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable domain, well-established mathematics)
