#!/usr/bin/env python
r"""Generate .npz fixture files for benchmark runners.

Produces bit-identical synthetic datasets for all three DCM variants
(task, spectral, rDCM) using the project simulators with fixed seeds.
Each variant x region-count combination gets its own subdirectory with
numbered .npz files and a ``manifest.json`` for metadata.

Usage::

    python benchmarks/generate_fixtures.py --variant all --n-regions 3,5,10 \
        --n-datasets 50 --seed 42 --output-dir benchmarks/fixtures

Fixture layout::

    benchmarks/fixtures/
        task_3region/
            dataset_000.npz
            dataset_001.npz
            ...
            manifest.json
        spectral_3region/
            ...
        rdcm_3region/
            ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from pyro_dcm.forward_models.rdcm_forward import generate_bold
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.simulators.rdcm_simulator import (
    make_block_stimulus_rdcm,
    make_stable_A_rdcm,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# ---------------------------------------------------------------------------
# Task DCM fixtures
# ---------------------------------------------------------------------------


def generate_task_fixtures(
    n_regions: int,
    n_datasets: int,
    seed: int,
    output_dir: str,
) -> None:
    """Generate task DCM fixtures.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    n_datasets : int
        Number of datasets to produce.
    seed : int
        Base random seed (incremented per dataset).
    output_dir : str
        Root output directory.
    """
    subdir = Path(output_dir) / f"task_{n_regions}region"
    os.makedirs(subdir, exist_ok=True)

    fields_saved: list[str] = []

    for i in range(n_datasets):
        seed_i = seed + i
        print(
            f"  Generating task_{n_regions}region: "
            f"dataset {i + 1}/{n_datasets}..."
        )

        torch.manual_seed(seed_i)

        A_true = make_random_stable_A(
            n_regions, density=0.5, seed=seed_i,
        )
        C = torch.zeros(n_regions, 1, dtype=torch.float64)
        C[0, 0] = 1.0

        stim = make_block_stimulus(
            n_blocks=5,
            block_duration=15.0,
            rest_duration=15.0,
            n_inputs=1,
        )

        sim = simulate_task_dcm(
            A_true, C, stim,
            duration=90.0, dt=0.01, TR=2.0, SNR=5.0,
            seed=seed_i, solver="rk4",
        )

        save_dict = {
            "A_true": A_true.numpy(),
            "C": C.numpy(),
            "bold": sim["bold"].detach().numpy(),
            "bold_clean": sim["bold_clean"].detach().numpy(),
            "stimulus_times": stim["times"].numpy(),
            "stimulus_values": stim["values"].numpy(),
            "TR": np.array(2.0),
            "SNR": np.array(5.0),
            "duration": np.array(90.0),
            "seed": np.array(seed_i),
        }

        np.savez(
            str(subdir / f"dataset_{i:03d}.npz"),
            **save_dict,
        )

        if i == 0:
            fields_saved = list(save_dict.keys())

    _write_manifest(subdir, n_datasets, seed, n_regions, "task", fields_saved)


# ---------------------------------------------------------------------------
# Task Bilinear DCM fixtures (Phase 16)
# ---------------------------------------------------------------------------


def generate_task_bilinear_fixtures(
    n_regions: int,
    n_datasets: int,
    seed: int,
    output_dir: str,
) -> None:
    """Generate task-bilinear DCM fixtures for the v0.3.0 recovery benchmark.

    3-region network (when ``n_regions=3``), 1 driving input, 1 modulator,
    2 non-zero B elements at positions (1, 0) and (2, 1) forming an
    asymmetric V1->V5->SPL hierarchy (16-CONTEXT.md topology lock). Driving
    stimulus is a block design over 200s; modulator is 4 boxcar epochs of
    12s each at [20, 65, 110, 155] s (SIM-02; blur-robust under rk4 per
    Pitfall B12).

    Ground-truth magnitudes (16-RESEARCH.md Section 3.2):

    - ``B_true[0, 1, 0] = 0.4``, ``B_true[0, 2, 1] = 0.3``
    - ``C[0, 0] = 0.5``
    - Safe-eigenvalue anchor: Gershgorin bound row-sum < 1.0 sustained only
      for 12s epochs -> ``exp(+2.4) ~ 11x`` amplification (safe; far below
      Phase 13 BILIN-06 3-sigma anchor ~ 3e6).

    Parameters
    ----------
    n_regions : int
        Number of brain regions. Phase 16 acceptance locks ``N=3``; higher
        N can be generated but the non-zero B positions remain at (1, 0)
        and (2, 1).
    n_datasets : int
        Number of datasets to produce. RECOV floor is >= 10.
    seed : int
        Base random seed (incremented per dataset).
    output_dir : str
        Root output directory. Subdirectory named
        ``task_bilinear_{n_regions}region``.

    Notes
    -----
    SNR = 3.0 (REQUIREMENTS.md RECOV-03..06 floor); TR = 2.0; duration = 200s;
    simulator ``dt = 0.01``. Each .npz carries the generic fields from
    ``generate_task_fixtures`` plus the bilinear-specific fields:
    ``B_true`` (shape ``(J, N, N)``), ``b_mask_0`` (shape ``(N, N)``;
    one per modulator, indexed by j), ``stim_mod_times``,
    ``stim_mod_values``, and ``J`` (scalar; number of modulators).

    References
    ----------
    .planning/phases/16-bilinear-recovery-benchmark/16-CONTEXT.md (topology)
    .planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md Sec. 3
    .planning/REQUIREMENTS.md RECOV-01, RECOV-02
    """
    subdir = Path(output_dir) / f"task_bilinear_{n_regions}region"
    os.makedirs(subdir, exist_ok=True)
    fields_saved: list[str] = []

    J = 1
    duration = 200.0
    TR = 2.0
    SNR = 3.0
    dt_sim = 0.01

    for i in range(n_datasets):
        seed_i = seed + i
        print(
            f"  Generating task_bilinear_{n_regions}region: "
            f"dataset {i + 1}/{n_datasets}..."
        )

        torch.manual_seed(seed_i)

        # A_true: random stable connectivity (matches v0.2.0 task fixture
        # family for RECOV-03 linear-baseline comparator).
        A_true = make_random_stable_A(
            n_regions, density=0.5, seed=seed_i,
        )

        # C: driving input on region 0 only, amplitude 0.5.
        C = torch.zeros(n_regions, 1, dtype=torch.float64)
        C[0, 0] = 0.5

        # B_true: asymmetric hierarchy V1->V5->SPL.
        B_true = torch.zeros(J, n_regions, n_regions, dtype=torch.float64)
        B_true[0, 1, 0] = 0.4
        B_true[0, 2, 1] = 0.3

        # b_mask: 1 where B is free, 0 elsewhere. Zero-diagonal by design.
        b_mask_0 = torch.zeros(n_regions, n_regions, dtype=torch.float64)
        b_mask_0[1, 0] = 1.0
        b_mask_0[2, 1] = 1.0

        # Driving stimulus: 5 blocks of 20s ON + 20s OFF over 200s.
        stim = make_block_stimulus(
            n_blocks=5,
            block_duration=20.0,
            rest_duration=20.0,
            n_inputs=1,
        )

        # Modulator: 4 epochs of 12s at [20, 65, 110, 155] s.
        stim_mod_dict = make_epoch_stimulus(
            event_times=[20.0, 65.0, 110.0, 155.0],
            event_durations=[12.0] * 4,
            event_amplitudes=[1.0] * 4,
            duration=duration,
            dt=dt_sim,
            n_inputs=1,
        )
        stim_mod = PiecewiseConstantInput(
            stim_mod_dict["times"], stim_mod_dict["values"],
        )

        sim = simulate_task_dcm(
            A_true, C, stim,
            duration=duration, dt=dt_sim, TR=TR, SNR=SNR,
            seed=seed_i, solver="rk4",
            B_list=[B_true[0]],
            stimulus_mod=stim_mod,
        )

        save_dict = {
            "A_true": A_true.numpy(),
            "C": C.numpy(),
            "bold": sim["bold"].detach().numpy(),
            "bold_clean": sim["bold_clean"].detach().numpy(),
            "stimulus_times": stim["times"].numpy(),
            "stimulus_values": stim["values"].numpy(),
            "B_true": B_true.numpy(),
            "b_mask_0": b_mask_0.numpy(),
            "stim_mod_times": stim_mod_dict["times"].numpy(),
            "stim_mod_values": stim_mod_dict["values"].numpy(),
            "J": np.array(J),
            "TR": np.array(TR),
            "SNR": np.array(SNR),
            "duration": np.array(duration),
            "seed": np.array(seed_i),
        }

        np.savez(
            str(subdir / f"dataset_{i:03d}.npz"),
            **save_dict,
        )

        if i == 0:
            fields_saved = list(save_dict.keys())

    _write_manifest(
        subdir, n_datasets, seed, n_regions,
        "task_bilinear", fields_saved,
    )


# ---------------------------------------------------------------------------
# Spectral DCM fixtures
# ---------------------------------------------------------------------------


def generate_spectral_fixtures(
    n_regions: int,
    n_datasets: int,
    seed: int,
    output_dir: str,
) -> None:
    """Generate spectral DCM fixtures with matched noise.

    Noise pattern matches ``spectral_svi.py`` lines 127-140 exactly:
    decompose clean CSD, add SNR-scaled Gaussian noise, reconstruct
    noisy complex CSD.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    n_datasets : int
        Number of datasets to produce.
    seed : int
        Base random seed (incremented per dataset).
    output_dir : str
        Root output directory.
    """
    subdir = Path(output_dir) / f"spectral_{n_regions}region"
    os.makedirs(subdir, exist_ok=True)

    snr = 10.0
    fields_saved: list[str] = []

    for i in range(n_datasets):
        seed_i = seed + i
        print(
            f"  Generating spectral_{n_regions}region: "
            f"dataset {i + 1}/{n_datasets}..."
        )

        A_true = make_stable_A_spectral(n_regions, seed=seed_i)

        sim = simulate_spectral_dcm(
            A_true, TR=2.0, n_freqs=32, seed=seed_i,
        )

        # Noise pattern matching spectral_svi.py lines 127-140
        obs_real = decompose_csd_for_likelihood(sim["csd"])
        signal_power = obs_real.pow(2).mean().sqrt()
        noise_std = signal_power / snr

        torch.manual_seed(seed_i + 1000)
        noisy_obs = obs_real + noise_std * torch.randn_like(obs_real)

        # Reconstruct noisy complex CSD
        F, n, _ = sim["csd"].shape
        half = F * n * n
        noisy_real = noisy_obs[:half].reshape(F, n, n)
        noisy_imag = noisy_obs[half:].reshape(F, n, n)

        # Split clean CSD into real/imag parts
        clean_csd = sim["csd"]

        save_dict = {
            "A_true": A_true.numpy(),
            "csd_real": clean_csd.real.numpy(),
            "csd_imag": clean_csd.imag.numpy(),
            "noisy_csd_real": noisy_real.detach().numpy(),
            "noisy_csd_imag": noisy_imag.detach().numpy(),
            "freqs": sim["freqs"].numpy(),
            "TR": np.array(2.0),
            "n_freqs": np.array(32),
            "seed": np.array(seed_i),
        }

        np.savez(
            str(subdir / f"dataset_{i:03d}.npz"),
            **save_dict,
        )

        if i == 0:
            fields_saved = list(save_dict.keys())

    _write_manifest(
        subdir, n_datasets, seed, n_regions, "spectral", fields_saved,
    )


# ---------------------------------------------------------------------------
# rDCM fixtures
# ---------------------------------------------------------------------------

# Constants matching rdcm_vb.py
_N_TIME = 4000
_U_DT = 0.5
_Y_DT = 2.0
_SNR = 3.0
_N_INPUTS = 2


def generate_rdcm_fixtures(
    n_regions: int,
    n_datasets: int,
    seed: int,
    output_dir: str,
) -> None:
    """Generate rDCM fixtures (BOLD + stimulus, no regressors).

    Matches ``_generate_rdcm_data`` from ``rdcm_vb.py`` exactly for
    A, C, stimulus, and BOLD generation. Regressors are NOT stored
    (runners call ``create_regressors`` themselves).

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    n_datasets : int
        Number of datasets to produce.
    seed : int
        Base random seed (incremented per dataset).
    output_dir : str
        Root output directory.
    """
    subdir = Path(output_dir) / f"rdcm_{n_regions}region"
    os.makedirs(subdir, exist_ok=True)

    fields_saved: list[str] = []

    for i in range(n_datasets):
        seed_i = seed + i
        print(
            f"  Generating rdcm_{n_regions}region: "
            f"dataset {i + 1}/{n_datasets}..."
        )

        A, a_mask = make_stable_A_rdcm(
            n_regions, density=0.5, seed=seed_i,
        )

        C = torch.zeros(n_regions, _N_INPUTS, dtype=torch.float64)
        if n_regions >= 2:
            C[0, 0] = 1.0
            C[1, 1] = 1.0
        else:
            C[0, 0] = 1.0
            C[0, 1] = 1.0

        c_mask = (C != 0).to(torch.float64)

        u = make_block_stimulus_rdcm(
            n_time=_N_TIME, n_inputs=_N_INPUTS,
            u_dt=_U_DT, seed=seed_i,
        )

        torch.manual_seed(seed_i + 10000)
        bold_result = generate_bold(
            A, C, u, u_dt=_U_DT, y_dt=_Y_DT, SNR=_SNR,
        )

        save_dict = {
            "A_true": A.numpy(),
            "C_true": C.numpy(),
            "a_mask": a_mask.numpy(),
            "c_mask": c_mask.numpy(),
            "y": bold_result["y"].detach().numpy(),
            "y_clean": bold_result["y_clean"].detach().numpy(),
            "u": u.numpy(),
            "u_dt": np.array(_U_DT),
            "y_dt": np.array(_Y_DT),
            "SNR": np.array(_SNR),
            "seed": np.array(seed_i),
        }

        np.savez(
            str(subdir / f"dataset_{i:03d}.npz"),
            **save_dict,
        )

        if i == 0:
            fields_saved = list(save_dict.keys())

    _write_manifest(
        subdir, n_datasets, seed, n_regions, "rdcm", fields_saved,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    subdir: Path,
    n_datasets: int,
    seed: int,
    n_regions: int,
    variant: str,
    fields: list[str],
) -> None:
    """Write a manifest.json for a fixture subdirectory.

    Parameters
    ----------
    subdir : Path
        Output subdirectory.
    n_datasets : int
        Number of datasets generated.
    seed : int
        Base random seed.
    n_regions : int
        Number of regions.
    variant : str
        DCM variant name.
    fields : list[str]
        Field names stored in each .npz.
    """
    manifest = {
        "variant": variant,
        "n_regions": n_regions,
        "n_datasets": n_datasets,
        "seed": seed,
        "fields": fields,
    }
    manifest_path = subdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_GENERATORS = {
    "task": generate_task_fixtures,
    "task_bilinear": generate_task_bilinear_fixtures,
    "spectral": generate_spectral_fixtures,
    "rdcm": generate_rdcm_fixtures,
}


def main() -> None:
    """CLI entry point for fixture generation."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark fixture .npz files.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        help=(
            "DCM variant: task, spectral, rdcm, or all. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--n-regions",
        type=str,
        default="3,5,10",
        help=(
            "Comma-separated region counts. Default: 3,5,10."
        ),
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=50,
        help="Number of datasets per variant x size. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/fixtures",
        help="Output directory. Default: benchmarks/fixtures.",
    )
    args = parser.parse_args()

    region_counts = [int(x.strip()) for x in args.n_regions.split(",")]

    if args.variant == "all":
        variants = list(_GENERATORS.keys())
    else:
        variants = [args.variant]

    for variant in variants:
        if variant not in _GENERATORS:
            print(f"Unknown variant: {variant}", file=sys.stderr)
            sys.exit(1)

        gen_fn = _GENERATORS[variant]
        for n_regions in region_counts:
            print(f"\n=== {variant} N={n_regions} ===")
            gen_fn(n_regions, args.n_datasets, args.seed, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
