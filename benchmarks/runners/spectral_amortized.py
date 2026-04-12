"""Spectral DCM amortized benchmark runner.

Implements amortized inference benchmark for spectral DCM. Uses a
pre-trained normalizing flow guide for fast posterior inference,
comparing against per-subject SVI to compute the amortization gap.

Falls back to CI-scale inline training if pre-trained weights are
not available.

Reuses patterns from tests/test_amortized_benchmark.py.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from benchmarks.config import BenchmarkConfig
from benchmarks.fixtures import load_fixture
from benchmarks.metrics import (
    compute_amortization_gap,
    compute_coverage_from_ci,
    compute_rmse,
    pearson_corr,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_noise import default_noise_priors
from pyro_dcm.guides import (
    AmortizedFlowGuide,
    CsdSummaryNet,
    SpectralDCMPacker,
)
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    run_svi,
    spectral_dcm_model,
)
from pyro_dcm.models.amortized_wrappers import (
    amortized_spectral_dcm_model,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from scripts.generate_training_data import invert_A_to_A_free


def _generate_spectral_datasets(
    n_datasets: int,
    n_regions: int,
    seed: int,
) -> tuple[
    list[torch.Tensor],
    list[dict[str, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
]:
    """Generate spectral DCM datasets for benchmark.

    Parameters
    ----------
    n_datasets : int
        Number of datasets to generate.
    n_regions : int
        Number of brain regions.
    seed : int
        Base random seed.

    Returns
    -------
    tuple
        (data_list, params_list, freqs, a_mask).
    """
    data_list: list[torch.Tensor] = []
    params_list: list[dict[str, torch.Tensor]] = []
    freqs_ref = None
    a_mask = torch.ones(
        n_regions, n_regions, dtype=torch.float64,
    )

    for i in range(n_datasets + 20):  # extra for filtering
        A = make_stable_A_spectral(n_regions, seed=seed + i)
        torch.manual_seed(seed + n_datasets + i)
        priors = default_noise_priors(n_regions)
        noise_a = priors["a_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )
        noise_b = priors["b_prior_mean"] + 0.1 * torch.randn(
            2, 1, dtype=torch.float64,
        )
        noise_c = priors["c_prior_mean"] + 0.1 * torch.randn(
            2, n_regions, dtype=torch.float64,
        )

        try:
            result = simulate_spectral_dcm(
                A,
                noise_params={
                    "a": noise_a, "b": noise_b, "c": noise_c,
                },
                seed=seed + i,
            )
        except Exception:
            continue

        csd = result["csd"]
        if torch.isnan(csd.real).any() or torch.isinf(csd.real).any():
            continue

        if freqs_ref is None:
            freqs_ref = result["freqs"]

        A_free = invert_A_to_A_free(A)
        data_list.append(csd)
        params_list.append({
            "A_free": A_free,
            "noise_a": noise_a,
            "noise_b": noise_b,
            "noise_c": noise_c,
            "csd_noise_scale": torch.tensor(
                1.0, dtype=torch.float64,
            ),
        })

        if len(data_list) >= n_datasets:
            break

    return data_list, params_list, freqs_ref, a_mask


def _train_spectral_guide_inline(
    n_regions: int,
    n_train: int,
    n_steps: int,
    seed: int,
) -> tuple[
    AmortizedFlowGuide,
    SpectralDCMPacker,
    torch.Tensor,
    torch.Tensor,
]:
    """Train a CI-scale spectral DCM amortized guide inline.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    n_train : int
        Number of training datasets.
    n_steps : int
        Number of SVI steps.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (guide, packer, freqs, a_mask).
    """
    pyro.clear_param_store()
    pyro.enable_validation(False)
    torch.manual_seed(seed)

    data_list, params_list, freqs, a_mask = (
        _generate_spectral_datasets(n_train, n_regions, seed)
    )

    packer = SpectralDCMPacker(n_regions)
    packer.fit_standardization(params_list)

    net = CsdSummaryNet(
        n_regions, n_freqs=32, embed_dim=128,
    ).double()
    guide = AmortizedFlowGuide(
        net, packer.n_features,
        embed_dim=128,
        n_transforms=3,
        hidden_features=[128, 128],
        packer=packer,
    ).double()

    svi = SVI(
        amortized_spectral_dcm_model,
        guide,
        ClippedAdam({
            "lr": 5e-4, "clip_norm": 10.0, "lrd": 0.999,
        }),
        loss=Trace_ELBO(
            num_particles=2, vectorize_particles=False,
        ),
    )

    n_data = len(data_list)
    step_count = 0
    while step_count < n_steps:
        torch.manual_seed(step_count)
        perm = torch.randperm(n_data).tolist()
        for idx in perm:
            if step_count >= n_steps:
                break
            svi.step(data_list[idx], freqs, a_mask, packer)
            step_count += 1

    return guide, packer, freqs, a_mask


def run_spectral_amortized(
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run spectral DCM amortized benchmark.

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

    # Check for pre-trained guide
    guide_paths = [
        "models/spectral_final.pt",
        "models/spectral_ci.pt",
    ]
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

    if pretrained_path is None:
        print("Training CI-scale spectral guide inline...")
        n_train = 200 if not config.quick else 50
        n_train_steps = 500 if not config.quick else 200
        guide, packer, freqs, a_mask = (
            _train_spectral_guide_inline(
                N, n_train, n_train_steps, config.seed,
            )
        )
    else:
        print(f"Loading from {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path, weights_only=False,
        )
        packer = SpectralDCMPacker(N)
        if "standardization" in checkpoint:
            packer.mean_ = checkpoint["standardization"]["mean"]
            packer.std_ = checkpoint["standardization"]["std"]

        net = CsdSummaryNet(
            N, n_freqs=32, embed_dim=128,
        ).double()
        guide = AmortizedFlowGuide(
            net, packer.n_features,
            embed_dim=128,
            n_transforms=3,
            hidden_features=[128, 128],
            packer=packer,
        ).double()
        guide.load_state_dict(checkpoint["guide_state_dict"])

        freqs = checkpoint.get("freqs", None)
        a_mask = torch.ones(N, N, dtype=torch.float64)
        if freqs is None:
            from pyro_dcm.forward_models.spectral_transfer import (
                default_frequency_grid,
            )
            freqs = default_frequency_grid(TR=2.0, n_freqs=32)

    # Generate test data
    n_test = config.n_datasets

    if config.fixtures_dir is not None:
        test_data: list[torch.Tensor] = []
        test_params: list[dict[str, torch.Tensor]] = []
        for fi in range(n_test):
            fdata = load_fixture(
                "spectral", N, fi,
                config.fixtures_dir,
            )
            noisy_csd = torch.complex(
                fdata["noisy_csd_real"].to(torch.float64),
                fdata["noisy_csd_imag"].to(torch.float64),
            )
            test_data.append(noisy_csd)
            test_params.append({
                "A_free": invert_A_to_A_free(
                    fdata["A_true"],
                ),
            })
            freqs = fdata["freqs"]
    else:
        test_seed = config.seed + 5000
        test_data, test_params, test_freqs, _ = (
            _generate_spectral_datasets(
                n_test, N, test_seed,
            )
        )
        if test_freqs is not None:
            freqs = test_freqs

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

    for i in range(len(test_data)):
        print(f"Running test dataset {i + 1}/{len(test_data)}...")

        try:
            csd = test_data[i]
            true_A_free = test_params[i]["A_free"]

            # Amortized forward pass
            guide.eval()
            t0 = time.time()
            with torch.no_grad():
                samples = guide.sample_posterior(
                    csd, n_samples=500,
                )
            amort_elapsed = time.time() - t0

            amort_A_mean = samples["A_free"].mean(dim=0)
            amort_A_std = samples["A_free"].std(dim=0)

            # Metrics
            rmse_amort = compute_rmse(true_A_free, amort_A_mean)
            lower = amort_A_mean - 1.645 * amort_A_std
            upper = amort_A_mean + 1.645 * amort_A_std
            coverage = compute_coverage_from_ci(
                true_A_free, lower, upper,
            )

            # Correlation
            corr = pearson_corr(
                true_A_free.flatten(), amort_A_mean.flatten(),
            )

            rmse_list.append(rmse_amort)
            coverage_list.append(coverage)
            amort_time_list.append(amort_elapsed)
            correlation_list.append(corr)
            a_true_list.append(
                true_A_free.flatten().tolist(),
            )
            a_inferred_list.append(
                amort_A_mean.flatten().tolist(),
            )

            # Compute amortized ELBO BEFORE clear_param_store wipes
            # guide params
            with torch.no_grad():
                elbo_fn = Trace_ELBO(num_particles=5)
                guide.eval()
                amortized_elbo = elbo_fn.loss(
                    amortized_spectral_dcm_model,
                    guide,
                    csd, freqs, a_mask, packer,
                )

            # Per-subject SVI comparison
            pyro.clear_param_store()
            pyro.enable_validation(False)
            try:
                model_args = (csd, freqs, a_mask, N)
                svi_guide = create_guide(
                    spectral_dcm_model,
                    init_scale=0.01,
                    guide_type=config.guide_type,
                    n_regions=N,
                )
                t0 = time.time()
                svi_result = run_svi(
                    spectral_dcm_model, svi_guide, model_args,
                    num_steps=500, lr=0.01,
                    clip_norm=10.0, lr_decay_factor=0.1,
                    elbo_type=config.elbo_type,
                    guide_type=config.guide_type,
                )
                svi_elapsed = time.time() - t0

                svi_extract = svi_result.get(
                    "guide", svi_guide,
                )
                svi_post = extract_posterior_params(
                    svi_extract, model_args,
                )
                svi_A_free = svi_post["median"].get(
                    "A_free",
                    torch.zeros(N, N, dtype=torch.float64),
                )
                rmse_svi = compute_rmse(true_A_free, svi_A_free)

                if rmse_svi > 1e-10:
                    rmse_ratio_list.append(rmse_amort / rmse_svi)
                if svi_elapsed > 1e-10:
                    speed_ratio_list.append(
                        amort_elapsed / svi_elapsed,
                    )

                # SVI ELBO evaluation (guide params fresh from
                # SVI training)
                with torch.no_grad():
                    svi_elbo = elbo_fn.loss(
                        spectral_dcm_model,
                        svi_guide,
                        *model_args,
                    )

                gap = compute_amortization_gap(
                    svi_elbo, amortized_elbo,
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
            "variant": "spectral",
            "method": "amortized",
            "n_regions": N,
            "pretrained": pretrained_path is not None,
            "quick": config.quick,
        },
    }
