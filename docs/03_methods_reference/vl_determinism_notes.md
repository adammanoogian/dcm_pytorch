# Variational Laplace Determinism Notes

**Requirement:** VLROBUST-01 (Phase 29). **Pitfalls:** N4 (local optima /
multi-restart), N5 (finite-difference Jacobian step).
**Verified by:** `tests/test_vl_determinism.py`.

This note pins the reproducibility contract of the Variational Laplace (VL)
engine (`pyro_dcm.inference.variational_laplace.run_variational_laplace_generic`)
and documents the known sources of non-determinism, so that any future drift in
posterior estimates is *explainable* rather than mysterious. The Phase 30
recovery sweep consumes VL output across hundreds of cells; a silent change in
engine numerics must be detectable, and this contract is the regression anchor.

## The Determinism Contract

Given:

- a **fixed seed** (used for both ground-truth construction and the fit's RNG
  state, via `torch.manual_seed`),
- **byte-identical inputs** (the same observed CSD / BOLD / latent trajectories,
  `a_mask`, and context), and
- an **identical `max_iter`** and all other fit hyperparameters,

two VL fits yield posterior means that are **equal within `atol = 1e-8`**
(bitwise equality is preferred and asserted first; `torch.allclose` with
`atol=1e-8, rtol=0` is the documented fallback). This holds for all three
forward models:

| Forward model         | Class                  | Data domain                |
| --------------------- | ---------------------- | -------------------------- |
| Spectral DCM          | `SpectralDCMForward`   | cross-spectral density     |
| Task DCM              | `TaskDCMForward`       | BOLD time series (ODE)     |
| Latent-circuit DCM    | `LatentCircuitForward` | latent trajectories (ODE)  |

The engine is **not** run under `torch.use_deterministic_algorithms(True)`: that
mode can raise on some linear-algebra ops the VL engine relies on
(`linalg.solve`, `slogdet`, `cholesky`, `matrix_exp`). Reproducibility here is
achieved through fixed seeds plus identical inputs, not through enforced
determinism. The seed sensitivity test (`test_different_seeds_differ`) confirms
that the seed genuinely drives the fit, so the determinism assertions are not
passing trivially.

## Known Non-Determinism Sources

These can introduce sub-`1e-8` jitter (and, across machines, larger differences).
They are listed so future drift can be attributed correctly.

1. **BLAS / threadpool reduction order.** Multi-threaded BLAS sums partial
   products in a non-fixed order, so the low-order bits of matrix products and
   reductions can vary run to run. For *bitwise* reproducibility on a single
   machine, pin thread counts: `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
   Cross-machine bitwise equality is **not** guaranteed even with pinned threads.

2. **float64 accumulation order.** The ReML M-step and free-energy evaluation
   call `torch.linalg.solve` / `inv`, `slogdet` (`_spm_logdet`), and `cholesky`.
   These accumulate in float64; the accumulation order is implementation-defined
   and can shift the last bits. This is the dominant source of the sub-`1e-8`
   jitter that the `atol=1e-8` fallback absorbs.

3. **ODE solver (rk4) step accumulation.** The task and latent-circuit forward
   models integrate the neural/hemodynamic state with a fixed-step RK4 solver.
   Step-wise floating-point accumulation is deterministic for identical inputs
   and step size, but is sensitive to the integration grid (`dt`) and to the
   above BLAS/accumulation effects within each step.

4. **Finite-difference Jacobian step (pitfall N5).** The Gauss-Newton Jacobian
   is computed by forward finite differences with a fixed step `exp(-8)`
   (matching SPM12's `spm_diff.m`). This step is **deterministic, not random** —
   but it is scale-sensitive: it sets the absolute perturbation regardless of
   parameter magnitude, so it interacts with parameter scaling. It is fixed, so
   it does not contribute run-to-run variability; it is noted here only because
   pitfall N5 flags it as a numerical-conditioning concern, not a determinism
   one.

## Multi-Restart Note (Pitfall N4)

The VL engine converges to the **prior-nearest local mode**. A multi-restart
wrapper varies the per-restart RNG seed (and, where supported, `initial_p` in
the SVD-reduced space) and selects the fit with the highest final free energy.

- The restart **PATH is deterministic** given a fixed restart-seed schedule: the
  same schedule selects the same winning restart and the same posterior on
  repeated runs (verified by `test_multistart_schedule_reproducible`).
- The **SELECTED mode** depends on which basin each restart lands in. This is
  *reproducible* for a fixed schedule but is **not** guaranteed to be the global
  optimum — multiple restarts mitigate, but do not eliminate, the local-optima
  risk that pitfall N4 describes.

## Cross-Machine Caveat

The M3 cluster and the development laptop may use different BLAS builds (MKL vs
OpenBLAS, different versions, different threadpool behaviour). Posterior means
can therefore differ **below `atol ~ 1e-6`** across machines even with identical
seeds and inputs. **Phase 30 must compare VL outputs within a machine, not
bitwise across machines.** The within-machine `atol = 1e-8` contract above is the
correct regression target; a cross-machine comparison should use a looser
tolerance (`~1e-6`) or, preferably, re-run the reference fit on the same machine.
