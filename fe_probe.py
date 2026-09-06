"""Dump the VL free-energy decomposition for the Phase 32 matched problem."""
from __future__ import annotations
import math, torch
from pyro_dcm.inference.variational_laplace import _spm_logdet
import validation.run_vl_validation as rv
from pyro_dcm.inference import run_variational_laplace
from pyro_dcm.inference.csd_precision import compute_csd_precision
from pyro_dcm.simulators.spectral_simulator import simulate_spectral_dcm

N = 2
A = rv._build_reciprocal_asymmetric_A(N)
sim = simulate_spectral_dcm(A=A, TR=rv._TR, seed=42)
csd, freqs = sim["csd"], sim["freqs"]
print(f"  observed_csd shape = {tuple(csd.shape)}  dtype={csd.dtype}")
print(f"  n_freqs = {csd.shape[0]}   N = {csd.shape[1]}")
print(f"  complex elements = {csd.numel()}   -> as real pairs = {2*csd.numel()}")
Q, nq = compute_csd_precision(csd)
print(f"  nq (precision components) = {nq}")
res = run_variational_laplace(
    observed_csd=csd, freqs=freqs,
    a_mask=torch.ones(N, N, dtype=torch.float64), max_iter=64,
    hyperprior_mean=8.0, hyperprior_precision=128.0,
    prior_mean_a_offset=torch.ones(N, N, dtype=torch.float64) / 128.0,
)
print(f"  VL free_energy[-1] = {res.free_energy[-1]:.6f}")
print(f"  iterations = {res.n_iterations}  converged = {res.converged}")
print()
print("  --- 2*pi term arithmetic ---")
lg = math.log(2.0 * math.pi)
for label, ny in (("F*N*N complex", csd.numel()),
                  ("2*F*N*N real", 2*csd.numel()),
                  ("F*N (auto only)", csd.shape[0]*csd.shape[1])):
    print(f"    ny={ny:5d} ({label:16s}) -> -ny*log(2pi)/2 = {-ny*lg/2:12.4f}")
print(f"    observed offset (VL - SPM) = 269.8947")
print(f"    269.8947 / (log(2pi)/2) = {269.8947/(lg/2):.3f}   (integer => pure 2pi/ny mismatch)")
