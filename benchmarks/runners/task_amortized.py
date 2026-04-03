"""Task DCM amortized benchmark runner.

Implements amortized inference benchmark for task-based DCM. Uses a
pre-trained normalizing flow guide for fast posterior inference,
comparing against per-subject SVI to compute the amortization gap.

Falls back to CI-scale inline training if pre-trained weights are
not available.

Reuses patterns from tests/test_amortized_benchmark.py.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import (
    compute_amortization_gap,
    compute_coverage_from_ci,
    compute_rmse,
    pearson_corr,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.guides import (
    AmortizedFlowGuide,
    BoldSummaryNet,
    TaskDCMPacker,
)
from pyro_dcm.models import create_guide, run_svi, task_dcm_model
from pyro_dcm.models.amortized_wrappers import amortized_task_dcm_model
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from scripts.generate_training_data import invert_A_to_A_free


def _train_task_guide_inline(
    n_regions: int,
    n_inputs: int,
    n_train: int,
    n_steps: int,
    seed: int,
) -> tuple[AmortizedFlowGuide, TaskDCMPacker, dict]:
    """Train a CI-scale task DCM amortized guide inline.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    n_inputs : int
        Number of stimulus inputs.
    n_train : int
        Number of training datasets.
    n_steps : int
        Number of SVI training steps.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (guide, packer, train_meta) where train_meta has stimulus
        and model config for evaluation.
    """
    pyro.clear_param_store()
    pyro.enable_validation(False)
    torch.manual_seed(seed)

    n_blocks = 2
    block_duration = 15.0
    rest_duration = 15.0
    duration = n_blocks * (block_duration + rest_duration)
    dt = 0.5
    TR = 2.0

    stimulus = make_block_stimulus(
        n_blocks=n_blocks,
        block_duration=block_duration,
        rest_duration=rest_duration,
        n_inputs=n_inputs,
    )

    a_mask = torch.ones(n_regions, n_regions, dtype=torch.float64)
    c_mask = torch.ones(n_regions, n_inputs, dtype=torch.float64)
    t_eval = torch.arange(0, duration, dt, dtype=torch.float64)

    # Generate training data
    data_list: list[torch.Tensor] = []
    params_list: list[dict[str, torch.Tensor]] = []

    for i in range(n_train + 20):  # extra for NaN filtering
        A = make_random_stable_A(n_regions, seed=seed + 500 + i)
        torch.manual_seed(seed + 600 + i)
        C = torch.randn(
            n_regions, n_inputs, dtype=torch.float64,
        ) * c_mask

        try:
            result = simulate_task_dcm(
                A, C, stimulus,
                duration=duration, dt=0.01, TR=TR, SNR=5.0,
                seed=seed + 700 + i,
            )
        except Exception:
            continue

        bold = result["bold"]
        if torch.isnan(bold).any() or torch.isinf(bold).any():
            continue

        A_free = invert_A_to_A_free(A)
        signal_std = result["bold_clean"].std(dim=0)
        signal_var = signal_std.pow(2).mean()
        noise_prec = torch.tensor(
            25.0 / signal_var.item()
            if signal_var.item() > 0 else 1.0,
            dtype=torch.float64,
        )

        data_list.append(bold)
        params_list.append({
            "A_free": A_free,
            "C": C,
            "noise_prec": noise_prec,
        })
        if len(data_list) >= n_train:
            break

    # Create packer and guide
    packer = TaskDCMPacker(n_regions, n_inputs, a_mask, c_mask)
    packer.fit_standardization(params_list)

    net = BoldSummaryNet(n_regions, embed_dim=64).double()
    guide = AmortizedFlowGuide(
        net, packer.n_features,
        embed_dim=64,
        n_transforms=2,
        n_bins=4,
        hidden_features=[64, 64],
        packer=packer,
    ).double()

    svi = SVI(
        amortized_task_dcm_model,
        guide,
        ClippedAdam({"lr": 1e-3, "clip_norm": 10.0}),
        loss=Trace_ELBO(num_particles=1),
    )

    # Epoch-based training
    n_data = len(data_list)
    step_count = 0
    while step_count < n_steps:
        perm = torch.randperm(n_data).tolist()
        for idx in perm:
            if step_count >= n_steps:
                break
            svi.step(
                data_list[idx], stimulus,
                a_mask, c_mask, t_eval, TR, dt, packer,
            )
            step_count += 1

    train_meta = {
        "stimulus": stimulus,
        "a_mask": a_mask,
        "c_mask": c_mask,
        "t_eval": t_eval,
        "TR": TR,
        "dt": dt,
        "duration": duration,
        "n_blocks": n_blocks,
    }

    return guide, packer, train_meta


def run_task_amortized(config: BenchmarkConfig) -> dict[str, Any]:
    """Run task DCM amortized benchmark.

    Loads a pre-trained guide or trains one inline in quick mode.
    Evaluates amortized inference speed and accuracy against
    per-subject SVI on held-out test datasets.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    dict
        Results dict with amortized metrics and amortization gap.
    """
    N = config.n_regions
    M = 1

    # Check for pre-trained guide
    guide_paths = ["models/task_final.pt", "models/task_ci.pt"]
    pretrained_path = None
    for p in guide_paths:
        if os.path.exists(p):
            pretrained_path = p
            break

    if pretrained_path is None and not config.quick:
        print(
            "Pre-trained guide not found. "
            "Run scripts/train_amortized_guide.py first."
        )
        return {
            "status": "skipped",
            "reason": "no_pretrained_guide",
        }

    # Quick mode: train inline
    if pretrained_path is None:
        print("Training CI-scale guide inline...")
        guide, packer, meta = _train_task_guide_inline(
            N, M, n_train=50, n_steps=100, seed=config.seed,
        )
    else:
        print(f"Loading pre-trained guide from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, weights_only=False)
        # Reconstruct guide from checkpoint
        a_mask = checkpoint.get(
            "a_mask",
            torch.ones(N, N, dtype=torch.float64),
        )
        c_mask = checkpoint.get(
            "c_mask",
            torch.ones(N, M, dtype=torch.float64),
        )
        packer = TaskDCMPacker(N, M, a_mask, c_mask)
        if "standardization" in checkpoint:
            packer.mean_ = checkpoint["standardization"]["mean"]
            packer.std_ = checkpoint["standardization"]["std"]

        net = BoldSummaryNet(N, embed_dim=64).double()
        guide = AmortizedFlowGuide(
            net, packer.n_features,
            embed_dim=64,
            n_transforms=2,
            n_bins=4,
            hidden_features=[64, 64],
            packer=packer,
        ).double()
        guide.load_state_dict(checkpoint["guide_state_dict"])

        meta = checkpoint.get("meta", {
            "duration": 60.0,
            "n_blocks": 2,
            "dt": 0.5,
            "TR": 2.0,
        })

    # Generate test data
    n_test = config.n_datasets
    duration = meta.get("duration", 60.0)
    n_blocks = meta.get("n_blocks", 2)
    dt_model = meta.get("dt", 0.5)
    TR = meta.get("TR", 2.0)

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    amort_time_list: list[float] = []
    svi_time_list: list[float] = []
    rmse_ratio_list: list[float] = []
    speed_ratio_list: list[float] = []
    gap_list: list[dict[str, float]] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    n_failed = 0

    stimulus = make_block_stimulus(
        n_blocks=n_blocks,
        block_duration=15.0,
        rest_duration=15.0,
        n_inputs=M,
    )

    for i in range(n_test):
        seed_i = config.seed + 1000 + i
        print(f"Running test dataset {i + 1}/{n_test}...")

        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            # Generate test data
            A_true = make_random_stable_A(N, seed=seed_i)
            C = torch.zeros(N, M, dtype=torch.float64)
            C[0, 0] = 1.0

            sim = simulate_task_dcm(
                A_true, C, stimulus,
                duration=duration, dt=0.01, TR=TR, SNR=5.0,
                seed=seed_i,
            )

            bold = sim["bold"]
            if torch.isnan(bold).any():
                n_failed += 1
                continue

            A_true_free = invert_A_to_A_free(A_true)

            # Amortized forward pass
            guide.eval()
            t0 = time.time()
            with torch.no_grad():
                samples = guide.sample_posterior(bold, n_samples=500)
            amort_elapsed = time.time() - t0

            amort_A_mean = samples["A_free"].mean(dim=0)
            amort_A_std = samples["A_free"].std(dim=0)

            # Metrics from amortized
            rmse_amort = compute_rmse(
                A_true_free, amort_A_mean,
            )

            # 90% CI coverage
            lower = amort_A_mean - 1.645 * amort_A_std
            upper = amort_A_mean + 1.645 * amort_A_std
            coverage = compute_coverage_from_ci(
                A_true_free, lower, upper,
            )

            # Correlation
            corr = pearson_corr(
                A_true_free.flatten(), amort_A_mean.flatten(),
            )

            rmse_list.append(rmse_amort)
            coverage_list.append(coverage)
            amort_time_list.append(amort_elapsed)
            correlation_list.append(corr)
            a_true_list.append(
                A_true_free.flatten().tolist(),
            )
            a_inferred_list.append(
                amort_A_mean.flatten().tolist(),
            )

            # Per-subject SVI for comparison
            pyro.clear_param_store()
            pyro.enable_validation(False)
            try:
                a_mask = torch.ones(
                    N, N, dtype=torch.float64,
                )
                c_mask = torch.zeros(
                    N, M, dtype=torch.float64,
                )
                c_mask[0, 0] = 1.0
                t_eval = torch.arange(
                    0, duration, dt_model, dtype=torch.float64,
                )
                model_args = (
                    bold, stimulus,
                    a_mask, c_mask, t_eval, TR, dt_model,
                )

                svi_guide = create_guide(
                    task_dcm_model, init_scale=0.01,
                )
                t0 = time.time()
                svi_result = run_svi(
                    task_dcm_model, svi_guide, model_args,
                    num_steps=200, lr=0.005,
                    clip_norm=10.0, lr_decay_factor=0.01,
                )
                svi_elapsed = time.time() - t0

                svi_median = svi_guide.median(*model_args)
                svi_A_free = svi_median.get(
                    "A_free",
                    torch.zeros(N, N, dtype=torch.float64),
                )
                rmse_svi = compute_rmse(A_true_free, svi_A_free)

                if rmse_svi > 1e-10:
                    rmse_ratio_list.append(rmse_amort / rmse_svi)
                if svi_elapsed > 1e-10:
                    speed_ratio_list.append(
                        amort_elapsed / svi_elapsed,
                    )

                # Amortization gap from RMSE ratio
                # True ELBO gap requires wrapper model + packer;
                # RMSE ratio is the observable proxy
                gap = compute_amortization_gap(
                    svi_result["final_loss"],
                    svi_result["final_loss"] * (
                        1.0 + max(0.0, rmse_amort / rmse_svi - 1.0)
                    ),
                )
                gap_list.append(gap)
                svi_time_list.append(svi_elapsed)

            except Exception:
                pass
            finally:
                pyro.enable_validation(True)

            print(
                f"  RMSE={rmse_amort:.4f}, "
                f"coverage={coverage:.3f}, "
                f"amort_time={amort_elapsed:.3f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1

    n_success = len(rmse_list)
    if n_success == 0:
        return {
            "status": "all_failed",
            "n_failed": n_failed,
        }

    summary: dict[str, Any] = {
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_amort_time": float(np.mean(amort_time_list)),
        "mean_correlation": float(np.mean(correlation_list)),
        "std_correlation": float(np.std(correlation_list)),
    }

    if rmse_ratio_list:
        summary["mean_rmse_ratio"] = float(
            np.mean(rmse_ratio_list),
        )
    if speed_ratio_list:
        summary["mean_speed_ratio"] = float(
            np.mean(speed_ratio_list),
        )

    return {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "amort_time_list": amort_time_list,
        "svi_time_list": svi_time_list,
        "rmse_ratio_list": rmse_ratio_list,
        "speed_ratio_list": speed_ratio_list,
        "amortization_gap_list": [g for g in gap_list],
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "n_success": n_success,
        "n_failed": n_failed,
        **summary,
        "metadata": {
            "variant": "task",
            "method": "amortized",
            "n_regions": N,
            "pretrained": pretrained_path is not None,
            "quick": config.quick,
        },
    }
