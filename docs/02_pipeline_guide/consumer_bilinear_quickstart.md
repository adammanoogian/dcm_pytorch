# Consumer Handoff: Bilinear Task-DCM from `pyro_dcm`

End-to-end example for projects that import `pyro_dcm` as a package. Written
for [`dcm_hgf_mixed_models`](https://github.com/adammanoogian/dcm_hgf_mixed_models)
(v0.3.0 milestone task H2A.2.1–2.2) but applies to any consumer that wants to
simulate + fit a bilinear, task-based DCM.

Scope: 3-region circuit, 1 driving input, 1 modulator. Simulation + SVI fit +
simple recovery metrics. The in-repo demo script (`scripts/demo_bilinear_consumer.py`)
is intentionally tiny (80s BOLD, 8 trials, 50 SVI steps) to run in <5 min
locally. **Any run with production-quality parameters (200-600s BOLD, 300-1500
SVI steps, N>=10 seeds) should be submitted to the cluster** via the
`cluster/` sbatch pattern — see `cluster/README.md` and the Phase 16 runner
at `benchmarks/runners/task_bilinear.py` for the full template.

## 1. Install

Development / local cross-repo:

```bash
# In the consumer project's env (e.g. actinf-py-scripts):
pip install -e ../dcm_pytorch   # sibling path
```

Remote pin (in consumer's `pyproject.toml`, `siblings` extra — update when a
new branch lands):

```toml
siblings = [
    "pyro-dcm @ git+https://github.com/adammanoogian/dcm_pytorch.git@gsd/phase-16-bilinear-recovery-benchmark",
]
```

`pyro_dcm`'s own `pyproject.toml` pulls `torch>=2.0`, `torchdiffeq`,
`pyro-ppl>=1.9`, `scipy`, `numpy`, `zuko>=1.2` transitively. Python 3.10+.

## 2. End-to-end example

A working, self-contained script lives at
`scripts/demo_bilinear_consumer.py` in this repo. The same code is reproduced
inline below for copy-paste.

### 2.1 Imports

```python
from functools import partial

import numpy as np
import pandas as pd
import torch

from pyro_dcm import (
    PiecewiseConstantInput,
    create_guide,
    extract_posterior_params,
    make_block_stimulus,
    make_event_stimulus,
    parameterize_A,
    parameterize_B,
    run_svi,
    simulate_task_dcm,
    task_dcm_model,
)
```

All bilinear-path helpers are re-exported at the top level of `pyro_dcm`. The
submodule paths (`pyro_dcm.forward_models.neural_state`,
`pyro_dcm.simulators.task_simulator`, `pyro_dcm.utils.ode_integrator`) are
still valid and stable — use whichever is more convenient.

### 2.2 Ground-truth circuit

```python
N_REGIONS = 3
DURATION = 80.0    # seconds — bump to 200-600 on cluster
DT = 0.01          # ODE step
TR = 2.0           # fMRI sample interval
SNR = 3.0

# A: intrinsic connectivity. parameterize_A clamps the diagonal negative per
# SPM12 convention; pass free off-diagonal elements only.
A_free_true = torch.zeros(N_REGIONS, N_REGIONS, dtype=torch.float64)
A_free_true[1, 0] = 0.3  # 0 -> 1
A_free_true[2, 1] = 0.3  # 1 -> 2
A_true = parameterize_A(A_free_true)

# C: driving input goes into region 0 only.
C_true = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
C_true[0, 0] = 0.5

# B: modulator gates the 0->1 and 1->2 edges. Tensors are always (J, N, N).
b_mask = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
b_mask[0, 1, 0] = 1.0
b_mask[0, 2, 1] = 1.0
B_free_true = torch.zeros(1, N_REGIONS, N_REGIONS, dtype=torch.float64)
B_free_true[0, 1, 0] = 0.4
B_free_true[0, 2, 1] = 0.3
B_true = parameterize_B(B_free_true, b_mask)  # (J=1, N, N)
```

### 2.3 Bridge DataFrame → `pyro_dcm` stimulus

This is the shape contract: the consumer feeds a DataFrame with
`[trial_idx, outcome_time_s, <channel>]` (exactly what
`dcm_hgf_mixed_models.bridge.hgf_to_dcm.select_dcm_modulators` returns) into
`make_event_stimulus`, then wraps in `PiecewiseConstantInput`.

```python
N_TRIALS = 8
rng = np.random.RandomState(0)
bridge_df = pd.DataFrame({
    "trial_idx": np.arange(N_TRIALS),
    "outcome_time_s": np.linspace(10.0, DURATION - 20.0, N_TRIALS),
    "epsilon2": rng.randn(N_TRIALS) * 0.5,  # stand-in for a real HGF trajectory
})

# Driving: block-design paradigm (independent of modulator).
stimulus_driving = make_block_stimulus(
    n_blocks=3, block_duration=15.0, rest_duration=10.0, n_inputs=1,
)

# Modulator: stick events at the bridge's outcome times, amplitude = channel.
stim_mod_dict = make_event_stimulus(
    event_times=bridge_df["outcome_time_s"].tolist(),
    event_amplitudes=bridge_df["epsilon2"].tolist(),
    duration=DURATION, dt=DT, n_inputs=1,
)
# task_dcm_model expects the wrapped object; simulate_task_dcm accepts either.
stim_mod = PiecewiseConstantInput(
    stim_mod_dict["times"], stim_mod_dict["values"],
)
```

### 2.4 Simulate BOLD

```python
# simulate_task_dcm takes B_list as a Python list of (N, N) tensors.
B_list = [B_true[j] for j in range(B_true.shape[0])]

sim = simulate_task_dcm(
    A=A_true, C=C_true, stimulus=stimulus_driving,
    duration=DURATION, dt=DT, TR=TR, SNR=SNR, seed=42,
    B_list=B_list, stimulus_mod=stim_mod,
)
observed_bold = sim["bold"]        # (T_tr, N)  — noisy observation
t_eval = sim["times_fine"]         # (T_fine,)  — ODE grid
```

### 2.5 Fit the DCM via SVI

```python
a_mask = torch.ones(N_REGIONS, N_REGIONS, dtype=torch.float64)
c_mask = torch.zeros(N_REGIONS, 1, dtype=torch.float64)
c_mask[0, 0] = 1.0

# b_masks is a Python list of per-modulator (N, N) masks.
b_masks_list = [b_mask[j] for j in range(b_mask.shape[0])]
model_kwargs = {"b_masks": b_masks_list, "stim_mod": stim_mod}

guide = create_guide(task_dcm_model, guide_type="auto_normal", init_scale=0.005)

model_args = (observed_bold, stimulus_driving, a_mask, c_mask, t_eval, TR)

svi_result = run_svi(
    model=task_dcm_model,
    guide=guide,
    model_args=model_args,
    num_steps=50,        # bump to 300-1500 for real use (cluster)
    lr=0.02,
    model_kwargs=model_kwargs,   # forwards b_masks + stim_mod into the model
)
```

### 2.6 Extract the posterior

`extract_posterior_params` does not accept `model_kwargs` directly — wrap the
model in `functools.partial` so the kwargs are bound when `Predictive` calls
it internally. This is the same pattern
`benchmarks/runners/task_bilinear.py` uses.

```python
model_for_pred = partial(task_dcm_model, **model_kwargs)

posterior = extract_posterior_params(
    guide=guide,
    model_args=model_args,
    model=model_for_pred,
    num_samples=200,
)

A_est = posterior["median"]["A"]   # (N, N)
B_est = posterior["median"]["B"]   # (J, N, N)
```

### 2.7 Simple recovery metrics

```python
a_rmse = torch.sqrt(((A_est - A_true) ** 2).mean()).item()

nonzero = b_mask[0].bool()
b_true_vec = B_true[0][nonzero]
b_est_vec = B_est[0][nonzero]
b_rmse = torch.sqrt(((b_est_vec - b_true_vec) ** 2).mean()).item()
b_sign_match = (b_est_vec.sign() == b_true_vec.sign()).float().mean().item()

print(f"A-RMSE:          {a_rmse:.3f}")
print(f"B-RMSE (masked): {b_rmse:.3f}")
print(f"B sign recovery: {b_sign_match:.2f}")
```

Expected output on a CPU run of the demo script at its default 50 SVI
steps (seed 42, deliberately tiny — local-friendly):

```
A-RMSE:          ~0.10-0.20      # A picks up signal quickly
B-RMSE (masked): ~0.30-0.40      # B barely moves off its zero prior
B sign recovery: 0.00-0.50       # ~random at 50 steps
```

At **300-600 SVI steps** the B posterior pulls meaningfully off zero:
A-RMSE drops to ~0.05-0.15, B-RMSE to ~0.10-0.25, and B sign recovery
reaches 1.00. At **1500 steps with 10 seeds** (Phase 16 full config), the
acceptance gates in `benchmarks/bilinear_metrics.py` pass. The tiny demo
is designed to verify wiring end-to-end, not to demonstrate recovery
quality — for that, use the cluster runner.

Exact numbers vary with seed, SVI step count, and `init_scale`. For
production-quality recovery metrics, pull `benchmarks/bilinear_metrics.py`'s
helpers (`compute_b_rmse_magnitude`, `compute_sign_recovery_nonzero`,
`compute_coverage_of_zero`, `compute_shrinkage`, `compute_acceptance_gates`)
or reimplement on the consumer side — those aren't yet part of the installed
package.

## 3. API map

| What the consumer wants | `pyro_dcm` entry point | Import path |
|---|---|---|
| Build effective connectivity A from free parameters | `parameterize_A(A_free)` | `pyro_dcm.forward_models.neural_state` (also `pyro_dcm`) |
| Build modulatory B matrices from free parameters + mask | `parameterize_B(B_free, b_mask)` — expects 3-D `(J, N, N)` | `pyro_dcm.forward_models.neural_state` |
| Random stable A | `make_random_stable_A(N, density, seed)` | `pyro_dcm.simulators.task_simulator` (also `pyro_dcm`) |
| Block-design driving input | `make_block_stimulus(...)` | `pyro_dcm.simulators.task_simulator` (also `pyro_dcm`) |
| Stick-event modulatory input | `make_event_stimulus(event_times, event_amplitudes, duration, dt, n_inputs)` | `pyro_dcm.simulators.task_simulator` |
| Boxcar-epoch modulatory input | `make_epoch_stimulus(...)` | `pyro_dcm.simulators.task_simulator` |
| Wrap breakpoints for the ODE integrator | `PiecewiseConstantInput(times, values)` | `pyro_dcm.utils.ode_integrator` |
| Forward simulation (linear or bilinear) | `simulate_task_dcm(A, C, stimulus, ..., B_list=..., stimulus_mod=...)` | `pyro_dcm.simulators.task_simulator` (also `pyro_dcm`) |
| Pyro generative model | `task_dcm_model(observed_bold, stimulus, a_mask, c_mask, t_eval, TR, *, b_masks=None, stim_mod=None)` | `pyro_dcm.models` (also `pyro_dcm`) |
| Guide factory | `create_guide(model, guide_type="auto_normal", init_scale=0.005)` | `pyro_dcm.models` (also `pyro_dcm`) |
| SVI runner | `run_svi(model, guide, model_args, num_steps, lr, model_kwargs=...)` | `pyro_dcm.models` (also `pyro_dcm`) |
| Posterior extraction | `extract_posterior_params(guide, model_args, model=partial(model, **kw), num_samples=200)` | `pyro_dcm.models` (also `pyro_dcm`) |

## 4. Troubleshooting

### `NaN ELBO at step 0`
Init-scale sensitivity. `init_scale=0.005` is the default for the benchmark
runner (Phase 16 L2) but some random seeds produce ground-truth + initial
posterior combinations that overflow the ODE. In the Phase 16 cluster run
(job 54901072), 3 of 10 seeds hit this before any gradient step.

Mitigations, in preference order:

1. Retry the same seed with `init_scale=0.001`. If the first SVI step returns
   NaN, reset the guide and try again once with the tighter init. This is
   tracked as a pending todo in this repo
   (`.planning/todos/pending/2026-04-19-retry-nan-seeds-halved-init-scale.md`)
   — feel free to reimplement on the consumer side while we land it
   upstream.
2. Shrink the ground-truth B magnitudes or A density.
3. Switch guide to `auto_low_rank_mvn` (slower but less init-sensitive).

### `AttributeError: 'builtin_function_or_method' object has no attribute 'shape'`
You passed the `make_event_stimulus` dict directly to `task_dcm_model` as
`stim_mod`. The model's `_validate_bilinear_args` expects a
`PiecewiseConstantInput` instance. Wrap first:
```python
stim_mod = PiecewiseConstantInput(stim_mod_dict["times"], stim_mod_dict["values"])
```

### `ValueError: parameterize_B expects 3-D stacked tensors (J, N, N)`
B tensors are always 3-D, even for a single modulator. Use `(1, N, N)` not
`(N, N)` when building `B_free` and `b_mask`.

### RECOV-06 coverage underestimated under `AutoNormal`
Known limitation of mean-field posteriors (research note N1 in this repo).
Consumers doing coverage-based group analysis should switch the guide to
`auto_low_rank_mvn` or `auto_iaf`. v0.3.1 will ship this as an amortized
fallback.

## 5. Known gaps (do not block local use)

- **Recovery metrics not installable** — `benchmarks/bilinear_metrics.py`
  lives outside `src/` and is not shipped as part of the `pyro-dcm` package.
  Consumers either copy the functions or reimplement until we move the
  module into `src/pyro_dcm/metrics/`.
- **Amortized bilinear path deferred to v0.3.1** — `TaskDCMPacker` and
  `amortized_task_dcm_model` refuse bilinear kwargs with a clear error. SVI
  with `auto_normal`/`auto_low_rank_mvn`/`auto_iaf` is the supported path
  until then.

## 6. File references

- **Canonical example (executable):** `scripts/demo_bilinear_consumer.py`
- **Full Phase 16 runner for reference:** `benchmarks/runners/task_bilinear.py`
- **Recovery metrics (for copy-paste):** `benchmarks/bilinear_metrics.py`
- **Model source:** `src/pyro_dcm/models/task_dcm_model.py`
- **Simulator source:** `src/pyro_dcm/simulators/task_simulator.py`
- **Phase 16 plan and summaries:**
  `.planning/phases/16-bilinear-recovery-benchmark/`
