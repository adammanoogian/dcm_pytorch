#!/usr/bin/env python
r"""Compare SBI (NPE) posteriors with SVI posteriors on synthetic data.

Quantifies the amortization gap between SBI (amortized, single forward
pass) and SVI (per-subject optimization) posterior approximations for
spectral DCM. For each synthetic test subject, generates ground-truth
connectivity, simulates CSD data, obtains posteriors from both methods,
and computes comparison metrics (RMSE, CI overlap, speed ratio).

Usage:
    python scripts/compare_sbi_svi.py \
        --estimator-path results/sbi_spectral/estimator.pt \
        --n-test-subjects 5 --n-svi-steps 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyro
import torch

try:
    import sbi  # noqa: F401
except ImportError:
    print(
        "sbi package required. Install with: pip install 'sbi>=0.22'",
        file=sys.stderr,
    )
    sys.exit(1)

from pyro_dcm.models import (
    create_guide,
    decompose_csd_for_likelihood,
    run_svi,
    spectral_dcm_model,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare SBI vs SVI posteriors on synthetic data",
    )
    parser.add_argument(
        "--estimator-path", type=str, required=True,
        help="Path to saved SBI estimator (estimator.pt)",
    )
    parser.add_argument(
        "--n-test-subjects", type=int, default=5,
        help="Number of synthetic test subjects (default: 5)",
    )
    parser.add_argument(
        "--n-svi-steps", type=int, default=500,
        help="Number of SVI optimization steps (default: 500)",
    )
    parser.add_argument(
        "--n-posterior-samples", type=int, default=5000,
        help="Posterior samples per method (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/sbi_vs_svi",
        help="Output directory (default: results/sbi_vs_svi)",
    )
    return parser.parse_args()


def _extract_svi_samples_as_theta(
    guide: object,
    model_args: tuple,
    a_mask: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Extract SVI posterior samples in the same packed format as SBI.

    Samples from the SVI guide, extracts A_free site values, and packs
    them into flat theta vectors matching the SBI parameter layout
    (free A entries where ``a_mask`` is nonzero).

    Parameters
    ----------
    guide : AutoGuide
        Trained Pyro guide.
    model_args : tuple
        Arguments for the model/guide.
    a_mask : torch.Tensor
        Binary structural mask, shape ``(N, N)``.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    torch.Tensor
        SVI samples in theta-space, shape ``(n_samples, n_free)``.
    """
    theta_list = []
    for _ in range(n_samples):
        trace = pyro.poutine.trace(guide).get_trace(*model_args)
        a_free = trace.nodes["A_free"]["value"]
        # Extract free parameters matching mask layout
        theta_i = a_free[a_mask.bool()]
        theta_list.append(theta_i.detach())
    return torch.stack(theta_list)


def main() -> None:
    """Run SBI vs SVI posterior comparison."""
    args = parse_args()
    N = 3
    F = 32
    TR = 2.0

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SBI vs SVI Posterior Comparison")
    print("=" * 70)
    print(f"  Estimator:       {args.estimator_path}")
    print(f"  Test subjects:   {args.n_test_subjects}")
    print(f"  SVI steps:       {args.n_svi_steps}")
    print(f"  Samples/method:  {args.n_posterior_samples}")
    print(f"  Seed:            {args.seed}")
    print()

    # --- Load trained SBI posterior ---
    posterior = torch.load(
        args.estimator_path,
        map_location="cpu",
        weights_only=False,
    )

    # --- Setup ---
    a_mask = torch.ones(N, N, dtype=torch.float64)

    # Storage for per-subject results
    results: list[dict] = []

    for subj_idx in range(args.n_test_subjects):
        seed_i = args.seed + subj_idx
        print(f"\n--- Subject {subj_idx + 1}/{args.n_test_subjects} "
              f"(seed={seed_i}) ---")

        # Generate ground truth
        A_true = make_stable_A_spectral(N, seed=seed_i)
        sim_result = simulate_spectral_dcm(A_true, TR=TR, n_freqs=F)
        obs_csd = sim_result["csd"]

        # Ground truth theta (free A parameters)
        theta_true = A_true[a_mask.bool()]

        # ---- SBI inference (amortized) ----
        x_obs = decompose_csd_for_likelihood(obs_csd)
        conditioned = posterior.set_default_x(
            x_obs.to(torch.float32),
        )

        t0_sbi = time.time()
        sbi_samples = conditioned.sample(
            (args.n_posterior_samples,),
        ).to(torch.float64)
        sbi_time = time.time() - t0_sbi

        sbi_mean = sbi_samples.mean(dim=0)
        sbi_std = sbi_samples.std(dim=0)

        # ---- SVI inference (per-subject) ----
        pyro.clear_param_store()
        guide = create_guide(spectral_dcm_model, init_scale=0.01)
        freqs = sim_result["freqs"]
        model_args = (obs_csd, freqs, a_mask)

        t0_svi = time.time()
        svi_result = run_svi(
            spectral_dcm_model,
            guide,
            model_args=model_args,
            num_steps=args.n_svi_steps,
        )
        svi_time = time.time() - t0_svi

        # Extract SVI samples in theta-space
        svi_samples = _extract_svi_samples_as_theta(
            guide, model_args, a_mask, args.n_posterior_samples,
        )

        svi_mean = svi_samples.mean(dim=0)
        svi_std = svi_samples.std(dim=0)

        # ---- Metrics ----
        # RMSE of posterior means vs ground truth
        sbi_rmse = (
            (sbi_mean - theta_true).pow(2).mean().sqrt().item()
        )
        svi_rmse = (
            (svi_mean - theta_true).pow(2).mean().sqrt().item()
        )

        # 95% CI overlap (per parameter, averaged)
        ci_overlaps = []
        for p in range(theta_true.shape[0]):
            sbi_lo = sbi_mean[p] - 1.96 * sbi_std[p]
            sbi_hi = sbi_mean[p] + 1.96 * sbi_std[p]
            svi_lo = svi_mean[p] - 1.96 * svi_std[p]
            svi_hi = svi_mean[p] + 1.96 * svi_std[p]

            overlap_lo = max(sbi_lo.item(), svi_lo.item())
            overlap_hi = min(sbi_hi.item(), svi_hi.item())
            overlap = max(0.0, overlap_hi - overlap_lo)

            union_lo = min(sbi_lo.item(), svi_lo.item())
            union_hi = max(sbi_hi.item(), svi_hi.item())
            union = max(1e-16, union_hi - union_lo)

            ci_overlaps.append(overlap / union)
        mean_ci_overlap = sum(ci_overlaps) / len(ci_overlaps)

        subj_result = {
            "subject": subj_idx + 1,
            "seed": seed_i,
            "sbi_rmse": sbi_rmse,
            "svi_rmse": svi_rmse,
            "sbi_time": sbi_time,
            "svi_time": svi_time,
            "ci_overlap": mean_ci_overlap,
            "svi_final_loss": svi_result["final_loss"],
            "theta_true": theta_true,
            "sbi_mean": sbi_mean,
            "svi_mean": svi_mean,
        }
        results.append(subj_result)

        print(f"  SBI: RMSE={sbi_rmse:.4f}, time={sbi_time:.3f}s")
        print(f"  SVI: RMSE={svi_rmse:.4f}, time={svi_time:.3f}s, "
              f"ELBO={svi_result['final_loss']:.1f}")
        print(f"  CI overlap: {mean_ci_overlap:.3f}")

    # ---- Aggregate and print table ----
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    header = (
        f"{'Subj':>6} | {'SBI RMSE':>10} | {'SVI RMSE':>10} | "
        f"{'SBI Time':>10} | {'SVI Time':>10} | {'CI Overlap':>10}"
    )
    print(header)
    print("-" * len(header))

    sbi_rmses = []
    svi_rmses = []
    sbi_times = []
    svi_times = []
    ci_overlaps_all = []

    for r in results:
        print(
            f"{r['subject']:>6d} | {r['sbi_rmse']:>10.4f} | "
            f"{r['svi_rmse']:>10.4f} | {r['sbi_time']:>9.3f}s | "
            f"{r['svi_time']:>9.3f}s | {r['ci_overlap']:>10.3f}"
        )
        sbi_rmses.append(r["sbi_rmse"])
        svi_rmses.append(r["svi_rmse"])
        sbi_times.append(r["sbi_time"])
        svi_times.append(r["svi_time"])
        ci_overlaps_all.append(r["ci_overlap"])

    print("-" * len(header))

    mean_sbi_rmse = sum(sbi_rmses) / len(sbi_rmses)
    mean_svi_rmse = sum(svi_rmses) / len(svi_rmses)
    mean_sbi_time = sum(sbi_times) / len(sbi_times)
    mean_svi_time = sum(svi_times) / len(svi_times)
    mean_ci = sum(ci_overlaps_all) / len(ci_overlaps_all)

    print(
        f"{'Mean':>6} | {mean_sbi_rmse:>10.4f} | "
        f"{mean_svi_rmse:>10.4f} | {mean_sbi_time:>9.3f}s | "
        f"{mean_svi_time:>9.3f}s | {mean_ci:>10.3f}"
    )

    speed_ratio = mean_svi_time / max(mean_sbi_time, 1e-6)
    print(f"\n  Speed ratio (SVI/SBI): {speed_ratio:.1f}x")
    print(f"  Amortization gap (RMSE ratio): "
          f"{mean_sbi_rmse / max(mean_svi_rmse, 1e-6):.2f}x")

    # ---- Save results ----
    save_data = {
        "results": results,
        "mean_sbi_rmse": mean_sbi_rmse,
        "mean_svi_rmse": mean_svi_rmse,
        "mean_sbi_time": mean_sbi_time,
        "mean_svi_time": mean_svi_time,
        "mean_ci_overlap": mean_ci,
        "speed_ratio": speed_ratio,
        "args": vars(args),
    }
    save_path = out_dir / "comparison_results.pt"
    torch.save(save_data, save_path)
    print(f"\n  Saved results to {save_path}")


if __name__ == "__main__":
    main()
