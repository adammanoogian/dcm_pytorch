"""Isolate step-0 NaN source on Phase-16 bilinear seeds 44/49/50.

Reproduces the cluster-observed NaN at step 0 locally for a chosen seed,
instrumenting every layer between guide sample and ELBO so we can pinpoint
where the NaN first appears (guide log_prob, ODE forward, BOLD, likelihood).

Usage
-----
    python scripts/debug_phase16_nan_seeds.py --seed 44
    python scripts/debug_phase16_nan_seeds.py --seed 44 --init-scale 0.001
    python scripts/debug_phase16_nan_seeds.py --seed 44 --mode linear
    python scripts/debug_phase16_nan_seeds.py --seed 44 --n-probes 50
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoNormal

from benchmarks.runners.task_bilinear import (
    _BILINEAR_INIT_SCALE,
    _DT_MODEL,
    _DURATION,
    _TR,
    _make_bilinear_ground_truth,
)
from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.models import task_dcm_model
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
    merge_piecewise_inputs,
)


def finite_summary(name: str, t: torch.Tensor) -> str:
    """Return finite-ness + range summary of a tensor."""
    if not isinstance(t, torch.Tensor):
        return f"  {name}: not a tensor ({type(t).__name__})"
    with torch.no_grad():
        has_nan = bool(torch.isnan(t).any().item())
        has_inf = bool(torch.isinf(t).any().item())
        if has_nan or has_inf:
            n_nan = int(torch.isnan(t).sum().item())
            n_inf = int(torch.isinf(t).sum().item())
            finite = t[torch.isfinite(t)]
            rng = (
                f"finite_min={finite.min().item():+.3g}, "
                f"finite_max={finite.max().item():+.3g}"
                if finite.numel() > 0 else "no finite values"
            )
            return (
                f"  {name}: shape={tuple(t.shape)} "
                f"NaN={n_nan} Inf={n_inf}; {rng}"
            )
        return (
            f"  {name}: shape={tuple(t.shape)} "
            f"min={t.min().item():+.3g} max={t.max().item():+.3g} "
            f"mean={t.mean().item():+.3g}"
        )


def probe_initial_draws(
    data: dict, init_scale: float, bilinear: bool, n_probes: int = 20,
) -> None:
    """Sample from an AutoNormal guide at init; check forward pass per draw.

    For each draw, run the forward model (up to BOLD) and report where NaN
    first appears. This tests whether the init posterior produces draws
    that push A_eff into unstable territory with any non-trivial frequency.
    """
    print(f"\n== Probe {n_probes} initial guide draws "
          f"({'BILINEAR' if bilinear else 'LINEAR'}, "
          f"init_scale={init_scale}) ==")
    N = data["A_true"].shape[0]
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    t_eval = torch.arange(0, _DURATION, _DT_MODEL, dtype=torch.float64)
    model_args = (
        data["bold"], data["stimulus"],
        a_mask, c_mask, t_eval, _TR, _DT_MODEL,
    )
    model_kwargs: dict = {}
    if bilinear:
        model_kwargs = {
            "b_masks": [data["b_mask_0"]], "stim_mod": data["stim_mod"],
        }

    from functools import partial
    model_for_guide = partial(task_dcm_model, **model_kwargs)

    pyro.clear_param_store()
    guide = AutoNormal(model_for_guide, init_scale=init_scale)

    # Force guide parameter initialization by running one trace.
    with torch.no_grad():
        _ = guide(*model_args)

    # Classify: finite_bold, nan_bold_caught, inf_bold_caught, nan_loss, etc.
    outcomes = {
        "finite_bold_finite_loss": 0,
        "nan_bold_caught": 0,
        "inf_bold_caught": 0,
        "finite_bold_nan_loss": 0,
        "nan_loss_other": 0,
    }
    first_bad_seed: int | None = None
    first_bad_info: dict = {}
    for p in range(n_probes):
        torch.manual_seed(1000 + p)
        pyro.set_rng_seed(1000 + p)
        try:
            with torch.no_grad():
                guide_trace = pyro.poutine.trace(guide).get_trace(*model_args)
                replayed = pyro.poutine.trace(
                    pyro.poutine.replay(model_for_guide, trace=guide_trace),
                ).get_trace(*model_args)
                log_joint = replayed.log_prob_sum()
                log_q = guide_trace.log_prob_sum()
                loss = -(log_joint - log_q).item()

                # Gather the predicted_bold deterministic
                pb = replayed.nodes.get("predicted_bold", {}).get("value")
                raw_bold_nan = (
                    bool(torch.isnan(pb).any().item())
                    if pb is not None else False
                )
                raw_bold_inf = (
                    bool(torch.isinf(pb).any().item())
                    if pb is not None else False
                )

                if math.isnan(loss):
                    if pb is not None and (raw_bold_nan or raw_bold_inf):
                        # Guard should have kicked in; if loss is still NaN,
                        # the guard failed OR there's another NaN source.
                        key = "nan_bold_caught" if raw_bold_nan else "inf_bold_caught"
                        outcomes[key] += 1
                        # Paradox: guard supposedly zeros but we got NaN loss
                        outcomes["nan_loss_other"] += 1
                    else:
                        outcomes["finite_bold_nan_loss"] += 1
                    if first_bad_seed is None:
                        first_bad_seed = 1000 + p
                        first_bad_info = {
                            "loss": loss, "log_joint": log_joint.item(),
                            "log_q": log_q.item(),
                            "pb_has_nan": raw_bold_nan,
                            "pb_has_inf": raw_bold_inf,
                        }
                else:
                    outcomes["finite_bold_finite_loss"] += 1
        except Exception as exc:  # noqa: BLE001
            outcomes.setdefault(f"exception:{type(exc).__name__}", 0)
            outcomes[f"exception:{type(exc).__name__}"] += 1

    print(f"  outcomes: {outcomes}")
    if first_bad_seed is not None:
        print(f"  first bad probe seed={first_bad_seed}: {first_bad_info}")


def manual_forward(
    data: dict, init_scale: float, bilinear: bool,
) -> None:
    """Run model once manually, instrumenting layer-by-layer.

    Uses a HAND-constructed A_free / B_free / C / noise_prec at the
    init_scale's implied posterior *mean* + one sigma, so we can trace
    which layer first produces NaN/Inf.
    """
    print(f"\n== Manual forward at 1-sigma draws "
          f"({'BILINEAR' if bilinear else 'LINEAR'}, "
          f"init_scale={init_scale}) ==")
    N = data["A_true"].shape[0]

    # Prior means are 0, priors sigmas are A: 1/8, C: 1, B: 1. Guide init
    # starts at posterior mean = prior mean for AutoNormal (median at loc=0)
    # with guide-sigma = init_scale. So initial draws are approx
    # N(prior_mean, init_scale) at the START, not at prior sigma.
    torch.manual_seed(0)
    sigma = init_scale
    A_free_draw = sigma * torch.randn(N, N, dtype=torch.float64)
    C_draw = sigma * torch.randn(N, 1, dtype=torch.float64)
    noise_prec_draw = torch.tensor(1.0, dtype=torch.float64)  # prior mean
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    A = parameterize_A(A_free_draw * a_mask)
    eigs_A = torch.linalg.eigvals(A)
    print(finite_summary("A_free_draw", A_free_draw))
    print(finite_summary("A", A))
    print(
        f"  A eigvals real: "
        f"max={eigs_A.real.max().item():+.4f}, "
        f"min={eigs_A.real.min().item():+.4f}"
    )

    B_stacked = None
    if bilinear:
        J = 1
        b_mask_0 = data["b_mask_0"]
        B_free_draw = sigma * torch.randn(J, N, N, dtype=torch.float64)
        b_mask_stacked = b_mask_0.unsqueeze(0)
        B_stacked = parameterize_B(B_free_draw, b_mask_stacked)
        print(finite_summary("B_free_draw", B_free_draw))
        print(finite_summary("B_stacked", B_stacked))

    if bilinear:
        # Merge driving + modulator for the bilinear input_fn.
        stim = data["stimulus"]
        mod = data["stim_mod"]
        merged = merge_piecewise_inputs(
            stim if isinstance(stim, PiecewiseConstantInput)
            else PiecewiseConstantInput(stim["times"], stim["values"]),
            mod if isinstance(mod, PiecewiseConstantInput)
            else PiecewiseConstantInput(mod["times"], mod["values"]),
        )
        system = CoupledDCMSystem(
            A, C_draw, merged, B=B_stacked,
            n_driving_inputs=c_mask.shape[1],
            stability_check_every=0,
        )
    else:
        system = CoupledDCMSystem(A, C_draw, data["stimulus"])

    t_eval = torch.arange(0, _DURATION, _DT_MODEL, dtype=torch.float64)
    y0 = make_initial_state(N, dtype=torch.float64)
    solution = integrate_ode(system, y0, t_eval, method="rk4", step_size=_DT_MODEL)
    print(finite_summary("solution", solution))

    lnv = solution[:, 3 * N:4 * N]
    lnq = solution[:, 4 * N:5 * N]
    print(finite_summary("lnv", lnv))
    print(finite_summary("lnq", lnq))
    v = torch.exp(lnv)
    q = torch.exp(lnq)
    print(finite_summary("v=exp(lnv)", v))
    print(finite_summary("q=exp(lnq)", q))
    bf = bold_signal(v, q)
    print(finite_summary("bold_fine", bf))

    step = round(_TR / _DT_MODEL)
    T = data["bold"].shape[0]
    predicted_bold = bf[::step][:T]
    print(finite_summary("predicted_bold (pre-guard)", predicted_bold))

    # Apply guard.
    if (torch.isnan(predicted_bold).any()
            or torch.isinf(predicted_bold).any()):
        print("  [GUARD] predicted_bold is NaN/Inf; zero-filling + detach")
        predicted_bold_guarded = torch.zeros_like(predicted_bold).detach()
    else:
        predicted_bold_guarded = predicted_bold
    print(finite_summary("predicted_bold (post-guard)", predicted_bold_guarded))

    noise_std = (1.0 / noise_prec_draw).sqrt()
    log_prob = dist.Normal(
        predicted_bold_guarded, noise_std,
    ).to_event(2).log_prob(data["bold"])
    print(f"  likelihood log_prob: {log_prob.item():+.4g} "
          f"(NaN={math.isnan(log_prob.item())})")


def try_svi_step_0(
    data: dict, init_scale: float, bilinear: bool,
) -> dict:
    """Actually attempt SVI step 0 with the configured init_scale.

    Returns outcome dict.
    """
    print(f"\n== Try SVI step 0 "
          f"({'BILINEAR' if bilinear else 'LINEAR'}, "
          f"init_scale={init_scale}) ==")
    N = data["A_true"].shape[0]
    a_mask = torch.ones(N, N, dtype=torch.float64)
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0
    t_eval = torch.arange(0, _DURATION, _DT_MODEL, dtype=torch.float64)
    model_args = (
        data["bold"], data["stimulus"],
        a_mask, c_mask, t_eval, _TR, _DT_MODEL,
    )
    model_kwargs: dict = {}
    if bilinear:
        model_kwargs = {
            "b_masks": [data["b_mask_0"]], "stim_mod": data["stim_mod"],
        }

    pyro.clear_param_store()
    from functools import partial
    from pyro.infer import SVI, Trace_ELBO
    model_bound = partial(task_dcm_model, **model_kwargs)
    guide = AutoNormal(model_bound, init_scale=init_scale)
    opt = pyro.optim.Adam({"lr": 0.005})
    svi = SVI(model_bound, guide, opt, loss=Trace_ELBO())
    try:
        loss = svi.step(*model_args)
        print(f"  step 0 loss = {loss:+.4g} "
              f"(NaN={math.isnan(loss)}, Inf={math.isinf(loss)})")
        return {"loss": loss, "status": "ran"}
    except Exception as exc:  # noqa: BLE001
        print(f"  step 0 raised {type(exc).__name__}: {exc}")
        return {"loss": float("nan"), "status": f"exc:{type(exc).__name__}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                    help="Seed index (e.g. 44, 49, 50 for cluster failures).")
    ap.add_argument("--init-scale", type=float, default=_BILINEAR_INIT_SCALE)
    ap.add_argument("--mode", choices=["bilinear", "linear", "both"],
                    default="both")
    ap.add_argument("--n-probes", type=int, default=20,
                    help="Number of guide draws to sample during probe.")
    ap.add_argument("--n-regions", type=int, default=3)
    args = ap.parse_args()

    # Silence stability warnings per runner convention.
    logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)

    print(f"Building ground-truth fixture for seed {args.seed}...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)
    data = _make_bilinear_ground_truth(args.n_regions, args.seed)

    # Quick sanity on the ground truth itself.
    print("\n== Ground-truth sanity ==")
    print(finite_summary("A_true", data["A_true"]))
    eigs = torch.linalg.eigvals(data["A_true"])
    print(f"  A_true eigvals real max: {eigs.real.max().item():+.4f}")
    print(finite_summary("bold", data["bold"]))
    print(finite_summary("B_true", data["B_true"]))

    modes = ["bilinear", "linear"] if args.mode == "both" else [args.mode]
    for mode in modes:
        bilinear = (mode == "bilinear")
        manual_forward(data, args.init_scale, bilinear)
        probe_initial_draws(data, args.init_scale, bilinear, args.n_probes)
        try_svi_step_0(data, args.init_scale, bilinear)

    return 0


if __name__ == "__main__":
    sys.exit(main())
