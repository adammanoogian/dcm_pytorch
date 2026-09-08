---
created: 2026-09-08T00:30
updated: 2026-09-08T16:10
title: VL-vs-SPM free energy differs by 116 nats on a forward that matches to 4e-14
area: spectral / SPM parity
priority: high
files:
  - src/pyro_dcm/forward_models/mar_csd.py
  - src/pyro_dcm/forward_models/spectral_transfer.py
  - src/pyro_dcm/inference/variational_laplace.py
  - validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m
---

## RESOLVED 2026-09-08: the mar2csd fix is correct and the "F regression" was not one

Both open questions from the original writing of this todo are settled. The
`2/ns` fix (`bd2448d`) is right, and the apparent end-to-end regression
(relative_error 0.0990 -> 0.2248) was an artifact of comparing free energies
computed on differently-normalised data.

### 1. The forward now matches SPM exactly (DCCN job 55234376)

Fed SPM's OWN exported components (`S_pE`, `Gu_pE`, `Gn_pE`) back through our
assembly and round-trip, and compared against SPM's own composite `pred_pE`.
This isolates what the fix changed and sidesteps every prior/A-convention
question, because SPM's numbers go in on both sides.

| check | ratio (ours/SPM) | max abs diff |
|---|---|---|
| pre-round-trip assembly | 1.000000 | 0.000e+00 |
| **composite post-round-trip** | **1.000000** | **4.043e-14** |
| round-trip only, SPM pre -> SPM post | 1.000000 | 4.043e-14 |

Before the fix this composite was **0.250000**. It is now exact to machine
precision. `csd_mar_roundtrip` is bit-for-bit SPM's `spm_mar2csd(spm_csd2mar(
y, M.Hz, M.p-1), M.Hz)`.

**Correction to this todo's original expectation:** the round-trip gain is
*not* ~1.0 and should not be. Measured out/in is mean 0.9808, range
0.6226..1.0678 -- and **SPM's own round-trip gives the identical numbers**. A
MAR(7) projection is genuinely lossy; the earlier "an information-preserving
round-trip should return ~1.0" framing was wrong. The 0.246 figure was
diagnostic only because it was a *constant* factor.

Confirmed against SPM source: `spm_mar2csd.m` has
`if nargin < 3, ns = 2*freqs(end); end` then `csd = 2*csd/ns`, and
`spm_csd_fmri_mtf.m:157` calls it with two arguments -- so our
`fs = 2*freqs[-1]` reproduces SPM's default exactly.

### 2. The VL free-energy drop is exactly the data-rescaling term

Reproduced locally by monkeypatching `mar2csd` back to its pre-`bd2448d` form,
so data AND forward revert together (as they did before the fix):

```
data scale (fixed/buggy)   4.00000000 at every element
F_buggy - F_fixed          177.445678
ny*ln(4), ny=128           177.445678
residual                  -2.842171e-14
```

Both recorded F values reproduced exactly (577.4417 -> 399.9960), **same
iteration count (9), both converged**. The VL fit did not change at all; F
shifted by precisely the analytic units term for rescaling the data.

### 3. Why the old 52-nat "agreement" was meaningless

| | VL F | SPM F | gap |
|---|---|---|---|
| before fix (data at 1/4 scale) | 577.4417 | 525.4313 | +52.01 |
| after fix (data at true scale) | 399.9960 | 515.9795 | **-115.98** |

Before the fix VL was fitting quarter-scale data with a quarter-scale forward
-- self-consistent, so it fit well, with F inflated by the +177.45 units term.
SPM's forward never had the bug, so it was fitting quarter-scale data with a
correct forward and had to pay for the mismatch.

Arithmetic: the data rescaling alone should move ANY Gaussian evidence by
-ny*ln(4) = -177.45. SPM's F moved only -9.45, so SPM's fit improved by
**+168.0 nats** when the data stopped being 4x too small.

The old +52.01 gap was therefore the accidental near-cancellation of VL's
+177.45 units inflation against ~168 nats of SPM compensation. It was never a
valid measure of engine agreement. (That the 168 nats is specifically prior-KL
on the noise amplitude parameters is a plausible reading, NOT verified --
SPM's `Ep` was not compared across the two data scales.)

## What is actually open: a 116.0-nat / 22.5% inversion-level gap

This is the first honest measurement of VL-vs-SPM disagreement: identical
injected CSD, identical `Y.Q`/`nq`, and a forward that agrees to 4e-14. The
strict 5% gate is red at 0.2248 and that is now a real, localised number.

It cannot be a forward-model difference and it cannot be a differing free
parameter count -- both engines carry
`A_free + a(2) + b(2) + c(N) + transit(N) + decay(1) + epsilon(1)`.

So it is in the inversion. Candidates, untested:

1. The `L` terms individually. The bridge already exports `Eh`; export SPM's
   three-term `L` breakdown as well and compare term by term rather than on the
   total. This is the highest-value next measurement.
2. `_spm_logdet`. SPM's `spm_logdet` is a *pseudo*-determinant that discards
   near-zero eigenvalues; `L(1)` carries `spm_logdet(iS)*nq/2`. If our port
   takes a true log-determinant the difference is a large constant. Verify
   against SPM on the same matrix.
3. Complex-data handling in `L(1)`: SPM computes `real(e'*iS*e)` with `'` as
   the conjugate transpose, and `ny = length(spm_vec(y))` counts 128 *complex*
   elements. Confirm the port matches on both.
4. Convergence: VL stops at 9 iterations. Confirm SPM's iteration count and
   final `Ep` on the same data.

**Ruled out -- do not re-investigate:** the hyperprior. The harness passes
`hyperprior_mean=8.0` / `hyperprior_precision=128.0` explicitly, so
`_run_vl_generic` sets `hE = full(nh, 8)`, matching SPM's exported `fe.hE = 8`.
The `-log(var(y)) + 4` fallback lives only in `_run_variational_laplace_legacy`,
which is not called here. Separately: that fallback takes `torch.var` of the
*real part* while SPM takes MATLAB `var` of the *complex* vector
(= var(real) + var(imag)). A real deviation, but inert on this path -- worth
fixing before anything depends on the fallback.

## Next step

Export SPM's three-term `L` decomposition from `spm_nlsi_GN` alongside `Eh`
and diff it against the VL engine's `L(1)/L(2)/L(3)`. A 116-nat gap on a
matching forward should localise to one term.

## Fixture freeze

Item (1) has settled, so **forward-parity** fixtures can now be frozen -- the
composite is SPM-exact and stable. Do NOT freeze an F-level fixture: 0.2248 is
a known-red number that this todo expects to move.

## Reproduce

- forward parity (cluster): `~/probes/mar_fwd_parity.{py,sbatch}`, job 55234376
- units term (laptop): scratchpad `mar_scale_probe.py`
