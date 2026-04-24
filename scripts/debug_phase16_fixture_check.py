"""Check ground-truth fixture finite-ness for all Phase-16 seeds.

Tests each seed's BOLD fixture for NaN/Inf content AND probes whether
the sustained-u_mod `A_eff = A + B` is stable-real. A seed whose
ground-truth simulation diverges (NaN BOLD) explains the cluster
step-0-NaN even though the SVI/init-scale path is unaffected.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

from benchmarks.runners.task_bilinear import _make_bilinear_ground_truth


def main(seeds: list[int]) -> int:
    logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)
    print(f"{'seed':>5}  {'eig(A).max_re':>13}  {'eig(A+B).max_re':>15}  "
          f"{'bold_nan':>8}  {'bold_inf':>8}  {'status'}")
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        data = _make_bilinear_ground_truth(3, s)
        A = data["A_true"]
        B = data["B_true"][0]  # (N, N), single modulator
        eig_A = torch.linalg.eigvals(A).real.max().item()
        # During u_mod=1 (epoch ON): A_eff = A + 1 * B
        A_eff_on = A + B
        eig_on = torch.linalg.eigvals(A_eff_on).real.max().item()
        bold = data["bold"]
        n_nan = int(torch.isnan(bold).sum().item())
        n_inf = int(torch.isinf(bold).sum().item())
        status = "CLEAN" if (n_nan == 0 and n_inf == 0) else "CORRUPT"
        print(
            f"{s:>5}  {eig_A:>+13.4f}  {eig_on:>+15.4f}  "
            f"{n_nan:>8}  {n_inf:>8}  {status}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    seeds = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else list(range(42, 52))
    raise SystemExit(main(seeds))
