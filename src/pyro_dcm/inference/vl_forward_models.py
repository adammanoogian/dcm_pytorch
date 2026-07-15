"""Forward model protocols for Variational Laplace inference.

Defines the ``ForwardModel`` protocol that decouples the VL engine from
any specific DCM variant. Concrete implementations wrap spectral DCM
(frequency-domain CSD) and task DCM (time-domain BOLD ODE integration).

References
----------
[REF-010] Friston et al. (2014). A DCM for resting state fMRI.
[REF-001] Friston et al. (2003). Dynamic causal modelling.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import torch

from pyro_dcm.forward_models.cmc_priors import ERP_DEAD_FREE
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.inference.csd_precision import compute_csd_precision

# Maximum tractable flattened dimension for the dense task-DCM precision
# matrix Q of shape (T*N, T*N). At dt < 0.1 with long duration this blows up
# (e.g. dt=0.01, 100s, N=4 -> T*N = 4e4 -> (4e4, 4e4) dense ~ 13 GB float64).
# Enforces the dt >= 0.1 floor for task DCM VL (VLROBUST-02, pitfall N1).
_TASK_PRECISION_MAX_DIM = 5000


@runtime_checkable
class ForwardModel(Protocol):
    """Protocol for forward models compatible with Variational Laplace.

    Encapsulates all model-specific logic: parameter layout, prior
    covariance structure, observation precision, forward prediction,
    and result construction. The VL engine calls these methods instead
    of hardcoded spectral DCM functions.
    """

    @property
    def residual_is_complex(self) -> bool:
        """True if residuals are complex (spectral CSD), False if real (BOLD)."""
        ...

    def param_count(self, n_regions: int) -> int:
        """Total number of free parameters for ``n_regions`` regions."""
        ...

    def pack_params(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """Flatten named parameter tensors into a single vector."""
        ...

    def unpack_params(
        self, theta: torch.Tensor, n_regions: int,
    ) -> dict[str, torch.Tensor]:
        """Reshape flat parameter vector into named tensors dict."""
        ...

    def build_prior_cov(
        self,
        n_regions: int,
        prior_variance: float,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build diagonal prior variance vector, shape ``(n_params,)``.

        Entries corresponding to absent connections (``a_mask == 0``)
        must be zero so that SVD reduction removes them.
        """
        ...

    def build_precision(
        self, observed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int]:
        """Build observation precision components.

        Returns
        -------
        Q_list : list of torch.Tensor
            Precision basis matrices. For spectral DCM, these are
            frequency-block Wishart matrices. For task DCM, a single
            identity matrix.
        nq : int
            Kronecker multiplier (1 for single-trial data).
        """
        ...

    def predict(
        self,
        theta: torch.Tensor,
        observed: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> torch.Tensor:
        """Run the forward model: parameters -> predicted data.

        Returns the predicted data as a flat vector matching
        ``observed.reshape(-1)`` in shape and dtype.
        """
        ...

    def build_result(
        self,
        theta_final: torch.Tensor,
        a_mask: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> dict:
        """Construct the result dict from converged parameters.

        Returns
        -------
        dict with keys:
            ``theta_post``: dict of named posterior parameter tensors
            ``predicted_output``: predicted data at the posterior mode
        """
        ...


class SpectralDCMForward:
    """Spectral DCM forward model for Variational Laplace.

    Wraps the existing spectral DCM pipeline: transfer function with
    hemodynamic model, neuronal/observation noise spectra, and MAR
    round-trip regularization.

    Parameters
    ----------
    eig_clamp : float or None
        Maximum real part of eigenvalues for A stability.
    mar_order : int
        MAR model order for CSD round-trip (SPM12 default: 7).
    """

    def __init__(
        self,
        eig_clamp: float | None = -1.0 / 32.0,
        mar_order: int = 7,
    ) -> None:
        self._eig_clamp = eig_clamp
        self._mar_order = mar_order

    @property
    def residual_is_complex(self) -> bool:
        return True

    def param_count(self, n_regions: int) -> int:
        N = n_regions
        return N * N + 2 + 2 + N + N + 1 + 1

    def pack_params(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            kwargs["A_free"].reshape(-1),
            kwargs["noise_a"].reshape(-1),
            kwargs["noise_b"].reshape(-1),
            kwargs["noise_c"].reshape(-1),
            kwargs["P_transit"].reshape(-1),
            kwargs["P_decay"].reshape(-1),
            kwargs["P_epsilon"].reshape(-1),
        ])

    def unpack_params(
        self, theta: torch.Tensor, n_regions: int,
    ) -> dict[str, torch.Tensor]:
        N = n_regions
        idx = 0
        result: dict[str, torch.Tensor] = {}
        result["A_free"] = theta[idx : idx + N * N].reshape(N, N)
        idx += N * N
        result["noise_a"] = theta[idx : idx + 2].reshape(2, 1)
        idx += 2
        result["noise_b"] = theta[idx : idx + 2].reshape(2, 1)
        idx += 2
        result["noise_c"] = theta[idx : idx + N].reshape(1, N)
        idx += N
        result["P_transit"] = theta[idx : idx + N]
        idx += N
        result["P_decay"] = theta[idx : idx + 1]
        idx += 1
        result["P_epsilon"] = theta[idx : idx + 1]
        return result

    def build_prior_cov(
        self,
        n_regions: int,
        prior_variance: float,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        N = n_regions
        hemo_var = 1.0 / 256.0
        n_conn_noise = N * N + 2 + 2 + N
        n_hemo = N + 1 + 1
        var_vec = torch.cat([
            torch.full(
                (n_conn_noise,), prior_variance, dtype=torch.float64,
            ),
            torch.full((n_hemo,), hemo_var, dtype=torch.float64),
        ])
        a_mask_flat = a_mask.reshape(-1)
        var_vec[: N * N] *= (a_mask_flat > 0).double()
        return var_vec

    def build_precision(
        self, observed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int]:
        return compute_csd_precision(observed)

    def predict(
        self,
        theta: torch.Tensor,
        observed: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> torch.Tensor:
        N = n_regions
        a_mask = context["a_mask"]
        freqs = context["freqs"]
        params = self.unpack_params(theta, N)
        A = parameterize_A(
            params["A_free"] * a_mask.to(params["A_free"].device),
        )
        pred_csd = spectral_dcm_forward(
            A,
            freqs,
            params["noise_a"],
            params["noise_b"],
            params["noise_c"],
            eig_clamp=self._eig_clamp,
            mar_order=self._mar_order,
            hemodynamic=True,
            P_transit=params["P_transit"],
            P_decay=params["P_decay"],
            P_epsilon=params["P_epsilon"],
        )
        return pred_csd.reshape(-1)

    def build_result(
        self,
        theta_final: torch.Tensor,
        a_mask: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> dict:
        N = n_regions
        freqs = context["freqs"]
        params = self.unpack_params(theta_final, N)
        A = parameterize_A(
            params["A_free"] * a_mask.to(params["A_free"].device),
        )
        pred_csd = spectral_dcm_forward(
            A,
            freqs,
            params["noise_a"],
            params["noise_b"],
            params["noise_c"],
            eig_clamp=self._eig_clamp,
            mar_order=self._mar_order,
            hemodynamic=True,
            P_transit=params["P_transit"],
            P_decay=params["P_decay"],
            P_epsilon=params["P_epsilon"],
        )
        theta_post = {**params, "A": A}
        return {"theta_post": theta_post, "predicted_output": pred_csd}


class TaskDCMForward:
    """Task-based DCM forward model for Variational Laplace.

    Integrates the neural-hemodynamic ODE (bilinear state equation +
    Balloon-Windkessel) and produces predicted BOLD timeseries.

    Parameters
    ----------
    stimulus_fn : callable
        Stimulus function ``u(t) -> (M,)`` for driving inputs.
    c_mask : torch.Tensor
        Binary mask for C matrix, shape ``(N, M)``.
    t_eval : torch.Tensor
        Fine time grid for ODE integration.
    dt : float
        ODE integration step size.

    References
    ----------
    [REF-001] Friston et al. (2003), Eq. 1.
    [REF-002] Stephan et al. (2007), Eq. 2-5.
    """

    def __init__(
        self,
        stimulus_fn: object,
        c_mask: torch.Tensor,
        t_eval: torch.Tensor,
        dt: float = 0.5,
    ) -> None:
        self._stimulus_fn = stimulus_fn
        self._c_mask = c_mask
        self._t_eval = t_eval
        self._dt = dt

    @property
    def residual_is_complex(self) -> bool:
        return False

    def param_count(self, n_regions: int) -> int:
        N = n_regions
        M = self._c_mask.shape[1]
        return N * N + N * M

    def pack_params(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            kwargs["A_free"].reshape(-1),
            kwargs["C_free"].reshape(-1),
        ])

    def unpack_params(
        self, theta: torch.Tensor, n_regions: int,
    ) -> dict[str, torch.Tensor]:
        N = n_regions
        M = self._c_mask.shape[1]
        idx = 0
        result: dict[str, torch.Tensor] = {}
        result["A_free"] = theta[idx : idx + N * N].reshape(N, N)
        idx += N * N
        result["C_free"] = theta[idx : idx + N * M].reshape(N, M)
        return result

    def build_prior_cov(
        self,
        n_regions: int,
        prior_variance: float,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        N = n_regions
        M = self._c_mask.shape[1]
        a_var = torch.full(
            (N * N,), prior_variance, dtype=torch.float64,
        )
        a_mask_flat = a_mask.reshape(-1)
        a_var *= (a_mask_flat > 0).double()
        c_var = torch.ones(N * M, dtype=torch.float64)
        c_mask_flat = self._c_mask.reshape(-1)
        c_var *= (c_mask_flat > 0).double()
        return torch.cat([a_var, c_var])

    def build_precision(
        self, observed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int]:
        """Build the dense identity observation precision for task DCM.

        Returns a single ``(ny, ny)`` identity precision (``ny = T * N``) and a
        Kronecker multiplier of 1. The dense matrix scales as ``(T*N)^2`` in
        memory, so this method fails LOUD when ``ny`` exceeds
        ``_TASK_PRECISION_MAX_DIM``: at ``dt < 0.1`` with long duration the
        precision becomes intractable (e.g. ``dt=0.01, 100s, N=4`` -> ``ny =
        4e4`` -> a ~13 GB float64 matrix). Callers must therefore use
        ``dt >= 0.1`` (and/or shorter duration / fewer regions) so that ``T*N``
        stays tractable (VLROBUST-02, pitfall N1).

        Parameters
        ----------
        observed : torch.Tensor
            Observed BOLD data; ``ny = observed.numel() = T * N``.

        Returns
        -------
        Q_list : list of torch.Tensor
            Single ``(ny, ny)`` identity precision matrix.
        nq : int
            Kronecker multiplier (1 for single-trial task DCM).

        Raises
        ------
        ValueError
            If ``ny > _TASK_PRECISION_MAX_DIM``, with a message reporting the
            actual size, the resulting dense matrix shape, and the expected cap.
        """
        ny = observed.numel()
        if ny > _TASK_PRECISION_MAX_DIM:
            raise ValueError(
                f"Task DCM precision matrix is intractable: observed has "
                f"{ny} elements, producing a dense ({ny}, {ny}) precision "
                f"matrix; expected <= {_TASK_PRECISION_MAX_DIM}. Use dt >= 0.1 "
                f"(and/or shorter duration / fewer regions) so T*N stays "
                f"tractable. See VLROBUST-02."
            )
        Q = torch.eye(ny, dtype=torch.float64, device=observed.device)
        return [Q], 1

    def predict(
        self,
        theta: torch.Tensor,
        observed: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> torch.Tensor:
        from pyro_dcm.forward_models.bold_signal import bold_signal
        from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
        from pyro_dcm.utils.ode_integrator import integrate_ode

        N = n_regions
        a_mask = context["a_mask"]
        params = self.unpack_params(theta, N)
        A = parameterize_A(
            params["A_free"] * a_mask.to(params["A_free"].device),
        )
        C = params["C_free"] * self._c_mask.to(params["C_free"].device)

        system = CoupledDCMSystem(A, C, self._stimulus_fn)
        y0 = torch.zeros(5 * N, dtype=torch.float64)
        solution = integrate_ode(
            system, y0, self._t_eval, method="rk4",
            step_size=self._dt,
        )

        lnv = solution[:, 3 * N : 4 * N]
        lnq = solution[:, 4 * N : 5 * N]
        predicted_bold = bold_signal(torch.exp(lnv), torch.exp(lnq))

        if not torch.isfinite(predicted_bold).all():
            predicted_bold = torch.zeros_like(predicted_bold)

        T_obs = observed.shape[0] if observed.ndim >= 1 else observed.numel()
        if predicted_bold.shape[0] > T_obs:
            predicted_bold = predicted_bold[:T_obs]

        return predicted_bold.reshape(-1)

    def build_result(
        self,
        theta_final: torch.Tensor,
        a_mask: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> dict:
        N = n_regions
        params = self.unpack_params(theta_final, N)
        A = parameterize_A(
            params["A_free"] * a_mask.to(params["A_free"].device),
        )
        C = params["C_free"] * self._c_mask.to(params["C_free"].device)
        theta_post = {**params, "A": A, "C": C}

        predicted = self.predict(
            theta_final, torch.empty(0), N, a_mask=a_mask,
        )
        return {"theta_post": theta_post, "predicted_output": predicted}


class LatentCircuitForward:
    """Latent-circuit (direct-observation, bilinear) forward model for VL.

    Fits a bilinear DCM directly to N-dimensional latent-state trajectories
    with NO hemodynamic model and identity observation (``C_obs = I``),
    matching ``pyro_dcm.models.latent_circuit_dcm_model``. This is the adapter
    that lets the model-agnostic Variational Laplace engine
    (``_run_vl_generic``) fit the Phase 20 latent-circuit DCM -- the existing
    ``SpectralDCMForward`` (CSD/hemodynamic) and ``TaskDCMForward``
    (BOLD ODE, ``A_free + C_free`` only) do not carry bilinear ``B`` and do not
    observe the neural state directly.

    Parameter layout (flat ``theta``)::

        A_free (N*N) + C_free (N*M) + B_free (J*N*N)

    Observation noise precision is NOT a free parameter; VL estimates it as the
    ReML hyperparameter scaling the identity precision ``Q`` (SPM convention).
    The forward model integrates ``dx/dt = (A + sum_j u_j(t) B_j) x + C u(t)``
    via ``CoupledDCMSystem(hemodynamic=False)`` from ``y0 = zeros(N)`` and
    returns the latent trajectory directly (identity observation).

    Parameters
    ----------
    stimulus : PiecewiseConstantInput
        Driving input ``u(t)`` with ``.values`` of shape ``(K, M)``.
    c_mask : torch.Tensor, shape (N, M)
        Binary mask for driving-input weights ``C``.
    t_eval : torch.Tensor, shape (T,)
        ODE time grid; spacing must equal ``dt`` and length must match the
        observed trajectory's time dimension.
    dt : float, optional
        ODE step size in seconds. Default 0.01 (latent dynamics timescale).
    b_masks : list of torch.Tensor or None, optional
        Per-modulator binary ``(N, N)`` masks for the bilinear ``B`` path.
        ``None`` or ``[]`` yields a linear model (no ``B`` parameters).
    stim_mod : PiecewiseConstantInput or None, optional
        Modulator input ``u_mod(t)`` with ``.values`` of shape ``(K, J)``.
        Required when ``b_masks`` is non-empty.
    c_prior_variance : float, optional
        Prior variance for ``C`` entries (model uses ``N(0, 1)``). Default 1.0.
    b_prior_variance : float, optional
        Prior variance for ``B_free`` entries (``LC_B_PRIOR_VARIANCE``).
        Default 1.0. The A-matrix prior variance is supplied separately via the
        VL engine's ``prior_variance`` argument (use ``LC_A_PRIOR_VARIANCE``).

    References
    ----------
    [REF-001] Friston et al. (2003), Eq. 1 (bilinear neural state equation).
    pyro_dcm.models.latent_circuit_dcm_model -- the Pyro/SVI counterpart.
    """

    def __init__(
        self,
        stimulus: object,
        c_mask: torch.Tensor,
        t_eval: torch.Tensor,
        dt: float = 0.01,
        *,
        b_masks: list[torch.Tensor] | None = None,
        stim_mod: object | None = None,
        c_prior_variance: float = 1.0,
        b_prior_variance: float = 1.0,
    ) -> None:
        if b_masks is not None and len(b_masks) == 0:
            b_masks = None
        if b_masks is not None and stim_mod is None:
            raise ValueError(
                "LatentCircuitForward: stim_mod is required when b_masks is "
                "non-empty; got None."
            )
        self._stimulus = stimulus
        self._c_mask = c_mask.to(torch.float64)
        self._t_eval = t_eval.to(torch.float64)
        self._dt = dt
        self._b_masks = (
            [m.to(torch.float64) for m in b_masks]
            if b_masks is not None
            else None
        )
        self._stim_mod = stim_mod
        self._c_prior_variance = c_prior_variance
        self._b_prior_variance = b_prior_variance

    @property
    def residual_is_complex(self) -> bool:
        return False

    @property
    def _n_modulators(self) -> int:
        """Number of bilinear modulators J (0 in the linear short-circuit)."""
        return 0 if self._b_masks is None else len(self._b_masks)

    def param_count(self, n_regions: int) -> int:
        N = n_regions
        M = self._c_mask.shape[1]
        return N * N + N * M + self._n_modulators * N * N

    def pack_params(self, **kwargs: torch.Tensor) -> torch.Tensor:
        parts = [
            kwargs["A_free"].reshape(-1),
            kwargs["C_free"].reshape(-1),
        ]
        if self._n_modulators > 0:
            parts.append(kwargs["B_free"].reshape(-1))
        return torch.cat(parts)

    def unpack_params(
        self, theta: torch.Tensor, n_regions: int,
    ) -> dict[str, torch.Tensor]:
        N = n_regions
        M = self._c_mask.shape[1]
        J = self._n_modulators
        idx = 0
        result: dict[str, torch.Tensor] = {}
        result["A_free"] = theta[idx : idx + N * N].reshape(N, N)
        idx += N * N
        result["C_free"] = theta[idx : idx + N * M].reshape(N, M)
        idx += N * M
        if J > 0:
            result["B_free"] = theta[idx : idx + J * N * N].reshape(J, N, N)
        return result

    def build_prior_cov(
        self,
        n_regions: int,
        prior_variance: float,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        N = n_regions
        M = self._c_mask.shape[1]
        a_var = torch.full((N * N,), prior_variance, dtype=torch.float64)
        a_var *= (a_mask.reshape(-1) > 0).double()
        c_var = torch.full(
            (N * M,), self._c_prior_variance, dtype=torch.float64,
        )
        c_var *= (self._c_mask.reshape(-1) > 0).double()
        parts = [a_var, c_var]
        if self._n_modulators > 0:
            b_mask_flat = torch.stack(self._b_masks, dim=0).reshape(-1)
            b_var = torch.full(
                (b_mask_flat.shape[0],),
                self._b_prior_variance,
                dtype=torch.float64,
            )
            b_var *= (b_mask_flat > 0).double()
            parts.append(b_var)
        return torch.cat(parts)

    def build_precision(
        self, observed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int]:
        ny = observed.numel()
        Q = torch.eye(ny, dtype=torch.float64, device=observed.device)
        return [Q], 1

    def _integrate(
        self, theta: torch.Tensor, n_regions: int, a_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the latent-circuit ODE and return the ``(T, N)`` trajectory."""
        from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
        from pyro_dcm.utils.ode_integrator import (
            integrate_ode,
            merge_piecewise_inputs,
        )

        N = n_regions
        params = self.unpack_params(theta, N)
        A = parameterize_A(params["A_free"] * a_mask.to(torch.float64))
        C = params["C_free"] * self._c_mask

        if self._n_modulators > 0:
            b_mask_stacked = torch.stack(self._b_masks, dim=0)
            B_stacked = parameterize_B(params["B_free"], b_mask_stacked)
            merged = merge_piecewise_inputs(self._stimulus, self._stim_mod)
            system = CoupledDCMSystem(
                A, C, merged,
                hemodynamic=False,
                B=B_stacked,
                n_driving_inputs=self._c_mask.shape[1],
            )
        else:
            system = CoupledDCMSystem(
                A, C, self._stimulus, hemodynamic=False,
            )

        y0 = torch.zeros(N, dtype=torch.float64)
        solution = integrate_ode(
            system, y0, self._t_eval, method="rk4", step_size=self._dt,
        )
        return solution  # (T, N)

    def predict(
        self,
        theta: torch.Tensor,
        observed: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> torch.Tensor:
        a_mask = context["a_mask"]
        solution = self._integrate(theta, n_regions, a_mask)

        if not torch.isfinite(solution).all():
            solution = torch.zeros_like(solution)

        # Align to the observed time dimension when it is known (the main
        # loop passes the (T, N) tensor; the FD Jacobian passes a flat vector,
        # in which case the trajectory length already matches t_eval).
        if observed.ndim >= 2:
            t_obs = observed.shape[0]
            if solution.shape[0] > t_obs:
                solution = solution[:t_obs]
        return solution.reshape(-1)

    def build_result(
        self,
        theta_final: torch.Tensor,
        a_mask: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> dict:
        N = n_regions
        params = self.unpack_params(theta_final, N)
        A = parameterize_A(params["A_free"] * a_mask.to(torch.float64))
        C = params["C_free"] * self._c_mask
        theta_post: dict[str, torch.Tensor] = {**params, "A": A, "C": C}
        if self._n_modulators > 0:
            b_mask_stacked = torch.stack(self._b_masks, dim=0)
            theta_post["B"] = parameterize_B(params["B_free"], b_mask_stacked)

        solution = self._integrate(theta_final, N, a_mask)
        return {"theta_post": theta_post, "predicted_output": solution}


# Free log value mapping an ABSENT CMC connection (mask == 0) to a dead edge --
# single source of truth in ``cmc_priors`` (shared with the SVI ``erp_dcm_model``).
_ERP_DEAD_FREE = ERP_DEAD_FREE


class ERPDCMForward:
    """DCM-for-evoked-responses (CMC) forward model for Variational Laplace.

    The fourth additive ``ForwardModel`` implementor (after ``SpectralDCMForward``,
    ``TaskDCMForward``, ``LatentCircuitForward``), exposing the parity-verified
    Phase-33/34 canonical-microcircuit network forward + condition-B modulation +
    the Phase-35 single-dipole lead field to the model-agnostic VL engine
    (``_run_vl_generic``). NOTHING in the ``ForwardModel`` protocol,
    ``variational_laplace.py``, or the sibling forward classes is edited.

    The forward composes (lazy-imported inside :meth:`predict` so import-time
    coupling stays zero): per-condition ``apply_condition_modulation``
    (``spm_gen_Q``) -> ``integrate_local_linearization`` (``spm_int_L`` exp-Euler)
    of ``cmc_network_f`` (``spm_fx_cmc``) -> ``project_to_scalp``
    (``spm_lx_erp`` ``y = (x - x0) @ L_full.T``), stacked to the canonical internal
    layout ``(Cnd, ns, Nc)`` and flattened C-order at the ``predict`` /
    ``build_precision`` boundary.

    Frozen parameter vector ordering (``param_count`` /
    :meth:`pack_params` / :meth:`unpack_params`)::

        A_free (4*N*N) + C_free (N*M) + T (4*N) + G (4*N) + S (N) + R (2*M)

    where ``M = n_inp``. The lead field ``L`` and contributing-state ``J`` are held
    FIXED (carried in the precomputed ``l_full``) -- they are observation context,
    not recovered parameters, for the forward-parity + protocol round-trip (Open Q3
    RESOLVED). The between-trial ``B`` matrices and the four extrinsic routing
    masks ``a_masks`` ride as constructor args (B2: ERP-specific needs are NOT
    protocol methods); the engine-supplied scalar ``a_mask`` (``context``) is a
    compatibility no-op -- CMC uses its own 4-block ``a_masks`` (Open Q4 RESOLVED).

    Parameters
    ----------
    l_full : torch.Tensor
        Full per-state lead field ``(Nc, 8N)`` (e.g.
        :func:`pyro_dcm.forward_models.build_lead_field`).
    x_design : torch.Tensor
        Between-trial design ``(Cnd, n_effects)``. Row 0 = standard, row 1 =
        deviant.
    a_masks : sequence of torch.Tensor
        The four extrinsic routing-graph masks (fwd sp->ss, fwd sp->dp, bwd
        dp->sp, bwd dp->ii), each ``(N, N)`` binary.
    b_masks : list of torch.Tensor
        The FIXED between-trial ``B`` value matrices (one per between-trial
        effect), each ``(N, N)``. ``B`` is NOT a free parameter in v1.
    c_mask : torch.Tensor
        Driving-input mask ``(N, n_inp)`` binary.
    dt : float, optional
        Integration step (``U.dt``) in seconds. Default 0.004.
    ns : int, optional
        Number of peristimulus samples. Default 128.
    ons_ms : float, optional
        Stimulus onset in ms (``M.ons``). Default 60.
    dur_ms : float, optional
        Gaussian dispersion in ms (``M.dur``). Default 16.
    sus : float, optional
        Sustained-input level (``M.sus``). Default 0.

    References
    ----------
    SPM12 ``spm_fx_cmc.m`` / ``spm_gen_Q.m`` / ``spm_int_L.m`` / ``spm_lx_erp.m``
    (Phase 33/34/35 ports). ``LatentCircuitForward`` -- the additive 4th-implementor
    precedent (``observed.ndim`` FD guard, identity ``build_precision``, lazy import).
    """

    def __init__(
        self,
        l_full: torch.Tensor,
        x_design: torch.Tensor,
        a_masks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        b_masks: list[torch.Tensor],
        c_mask: torch.Tensor,
        *,
        dt: float = 0.004,
        ns: int = 128,
        ons_ms: float = 60.0,
        dur_ms: float = 16.0,
        sus: float = 0.0,
    ) -> None:
        if len(a_masks) != 4:
            raise ValueError(
                "ERPDCMForward: a_masks must hold exactly 4 extrinsic routing "
                f"blocks (fwd/fwd/bwd/bwd); got {len(a_masks)}."
            )
        self._l_full = l_full.to(torch.float64)
        self._x_design = x_design.to(torch.float64)
        self._a_masks = [m.to(torch.float64) for m in a_masks]
        self._b_masks = [b.to(torch.float64) for b in b_masks]
        self._c_mask = c_mask.to(torch.float64)
        self._dt = dt
        self._ns = ns
        self._ons_ms = ons_ms
        self._dur_ms = dur_ms
        self._sus = sus
        # Peristimulus time grid in seconds (spm_gen_erp.m:35), precomputed.
        self._pst = (
            torch.arange(1, ns + 1, dtype=torch.float64) * dt - ons_ms / 1000.0
        )

    @property
    def residual_is_complex(self) -> bool:
        """ERP residuals are real-valued time-domain scalp ERPs."""
        return False

    @property
    def _n_inp(self) -> int:
        """Number of driving inputs ``M`` (columns of ``c_mask``)."""
        return self._c_mask.shape[1]

    def param_count(self, n_regions: int) -> int:
        """Total free params: ``4NN + N*M + 4N + 4N + N + 2M`` (L/J fixed)."""
        N = n_regions
        M = self._n_inp
        # A{1..4} + C + T + G + S + R (L/J fixed in the precomputed l_full).
        return 4 * N * N + N * M + 4 * N + 4 * N + N + 2 * M

    def pack_params(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """Flatten ``A_free, C_free, T, G, S, R`` in the frozen pack order."""
        return torch.cat([
            kwargs["A_free"].reshape(-1),
            kwargs["C_free"].reshape(-1),
            kwargs["T"].reshape(-1),
            kwargs["G"].reshape(-1),
            kwargs["S"].reshape(-1),
            kwargs["R"].reshape(-1),
        ])

    def unpack_params(
        self, theta: torch.Tensor, n_regions: int,
    ) -> dict[str, torch.Tensor]:
        """Reverse the frozen pack order into named CMC free-parameter tensors."""
        N = n_regions
        M = self._n_inp
        idx = 0
        result: dict[str, torch.Tensor] = {}
        result["A_free"] = theta[idx : idx + 4 * N * N].reshape(4, N, N)
        idx += 4 * N * N
        result["C_free"] = theta[idx : idx + N * M].reshape(N, M)
        idx += N * M
        result["T"] = theta[idx : idx + 4 * N].reshape(N, 4)
        idx += 4 * N
        result["G"] = theta[idx : idx + 4 * N].reshape(N, 4)
        idx += 4 * N
        result["S"] = theta[idx : idx + N].reshape(N, 1)
        idx += N
        result["R"] = theta[idx : idx + 2 * M].reshape(M, 2)
        return result

    def build_prior_cov(
        self,
        n_regions: int,
        prior_variance: float,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        """CMC prior variances flattened IN PACK ORDER (cmc_prior_moments).

        ``A`` ``mask/16``, ``C`` ``mask/32``, ``T``/``G`` ``1/32``, ``S`` ``1/64``,
        ``R`` ``1/16`` (``spm_cmc_priors.m``). Absent ``A``/``C`` entries get
        variance 0 so ``_spm_svd`` drops them (same idiom as
        ``LatentCircuitForward.build_prior_cov``). The engine-supplied scalar
        ``a_mask`` / ``prior_variance`` are IGNORED -- CMC uses its own stored
        4-block ``a_masks`` and the fixed ``spm_cmc_priors`` variances.
        """
        N = n_regions
        M = self._n_inp
        # A: 4 routing blocks, var 1/16 on live edges, 0 on absent (spm_cmc:80-81).
        a_var = torch.cat([
            (self._a_masks[i].reshape(-1) > 0).double() / 16.0 for i in range(4)
        ])
        # C: var mask/32 (spm_cmc_priors.m:114-116).
        c_var = (self._c_mask.reshape(-1) > 0).double() / 32.0
        t_var = torch.full((4 * N,), 1.0 / 32.0, dtype=torch.float64)  # :121
        g_var = torch.full((4 * N,), 1.0 / 32.0, dtype=torch.float64)  # :122
        s_var = torch.full((N,), 1.0 / 64.0, dtype=torch.float64)  # :124
        r_var = torch.full((2 * M,), 1.0 / 16.0, dtype=torch.float64)  # :133
        return torch.cat([a_var, c_var, t_var, g_var, s_var, r_var])

    def build_precision(
        self, observed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int]:
        """Identity observation precision over ``Cnd*ns*Nc`` (v1).

        AR(1) ``spm_Q`` temporal precision is deferred to a later milestone (the
        forward-only Phase-35 scope uses the identity).
        """
        ny = observed.numel()
        Q = torch.eye(ny, dtype=torch.float64, device=observed.device)
        return [Q], 1

    def _masked_free(
        self, params: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Apply the routing/input masks, sending absent edges to the dead value.

        Returns the four masked free ``A`` log-blocks (list of ``(N, N)``) and the
        masked free ``C`` log-gain ``(N, M)``. Live entries keep their (recovered)
        free value; absent entries map to ``_ERP_DEAD_FREE`` so ``exp(P) * E0`` is
        negligible (a 0 free value would be a LIVE edge under the CMC parameterisation).
        """
        a_free_list: list[torch.Tensor] = []
        for i in range(4):
            mb = (self._a_masks[i] > 0).double()
            a_free_list.append(
                params["A_free"][i] * mb + _ERP_DEAD_FREE * (1.0 - mb)
            )
        cb = (self._c_mask > 0).double()
        c_free = params["C_free"] * cb + _ERP_DEAD_FREE * (1.0 - cb)
        return a_free_list, c_free

    def predict(
        self,
        theta: torch.Tensor,
        observed: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> torch.Tensor:
        """Per-condition integrate -> project -> stack ``(Cnd, ns, Nc)`` -> flat.

        The canonical internal layout is ``(Cnd, ns, Nc)``; the flat boundary is a
        C-order ``reshape(-1)`` (condition-blocked), matching the engine's
        ``observed.reshape(-1)`` and the identity precision element-for-element
        (locked stacking layout, gap 4). The ``observed.ndim`` guard handles BOTH
        the main-loop ``(Cnd, ns, Nc)`` call (truncate ``ns`` to the observed
        length) AND the flat FD-Jacobian call (no truncation) -- pitfall B3.
        """
        from pyro_dcm.forward_models.erp_coupled_system import (
            apply_condition_modulation,
            cmc_network_f,
        )
        from pyro_dcm.forward_models.erp_input import erp_gaussian_input
        from pyro_dcm.forward_models.erp_leadfield import project_to_scalp
        from pyro_dcm.utils.local_linearization import (
            integrate_local_linearization,
        )

        N = n_regions
        params = self.unpack_params(theta, N)
        a_free_list, c_free = self._masked_free(params)
        p_struct: dict[str, object] = {
            "T": params["T"],
            "G": params["G"],
            "C": c_free,
            "S": params["S"],
            "R": params["R"],
            "A": a_free_list,
            "B": self._b_masks,
        }
        inputs = erp_gaussian_input(
            self._pst, params["R"], self._ons_ms, self._dur_ms, self._sus,
        )
        x0 = torch.zeros(8 * N, dtype=torch.float64, device=theta.device)

        y_list: list[torch.Tensor] = []
        for c in range(self._x_design.shape[0]):
            q = apply_condition_modulation(p_struct, self._x_design[c])

            def f_c(
                v: torch.Tensor, u: torch.Tensor, q: dict = q,
            ) -> torch.Tensor:
                return cmc_network_f(v, u, q, N)

            traj = integrate_local_linearization(f_c, x0, inputs, self._dt)
            if not torch.isfinite(traj).all():
                # The exp-Euler integrator diverged; clamp to a finite (zero)
                # trajectory so the ELBO/free-energy stays finite. Warn, because a
                # zeroed, parameter-independent forward yields a FLAT
                # finite-difference Jacobian region that can mislead Variational
                # Laplace into treating a diverging parameter as inert. A silent
                # clamp here is invisible; make it observable (dedup'd by warnings
                # so a persistent divergence does not spam).
                warnings.warn(
                    "ERPDCMForward.predict: exp-Euler integration produced "
                    f"non-finite states at condition index {c} (dt={self._dt}); "
                    "clamping the trajectory to zeros. Expected all-finite; got "
                    "NaN/Inf. Inspect priors/dt if this recurs -- the zeroed "
                    "forward flattens the VL finite-difference Jacobian.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                traj = torch.zeros_like(traj)
            y_list.append(project_to_scalp(traj, self._l_full))  # (ns, Nc)

        y = torch.stack(y_list, dim=0)  # (Cnd, ns, Nc)
        # Main loop passes (Cnd, ns, Nc); FD Jacobian passes a flat vector whose
        # trajectory length already matches ns (so only truncate when 3-D).
        if observed.ndim >= 3:
            y = y[:, : observed.shape[1]]
        return y.reshape(-1)

    def build_result(
        self,
        theta_final: torch.Tensor,
        a_mask: torch.Tensor,
        n_regions: int,
        **context: object,
    ) -> dict:
        """Build the posterior result: parameterised ``A``/``C`` + ``(Cnd,ns,Nc)``.

        ``theta_post`` keeps the raw free params (``A_free, C_free, T, G, S, R``)
        and adds the parameterised extrinsic ``A`` ``(4, N, N)``, the parameterised
        input gain ``C`` ``(N, M)``, and the fixed ``B`` stack. ``predicted_output``
        is the scalp ERP at the mode, reshaped to the canonical ``(Cnd, ns, Nc)``.
        """
        from pyro_dcm.forward_models.erp_coupled_system import (
            parameterize_cmc_network,
        )

        N = n_regions
        params = self.unpack_params(theta_final, N)
        a_free_list, c_free = self._masked_free(params)
        net = parameterize_cmc_network(
            {
                "T": params["T"],
                "G": params["G"],
                "C": c_free,
                "S": params["S"],
                "A": a_free_list,
            },
            N,
        )
        theta_post: dict[str, torch.Tensor] = {**params}
        theta_post["A"] = net["A"]  # parameterised extrinsic blocks (4, N, N)
        theta_post["C"] = net["C"]  # parameterised input gain exp(P.C)
        if self._b_masks:
            theta_post["B"] = torch.stack(self._b_masks, dim=0)

        cnd = self._x_design.shape[0]
        nc = self._l_full.shape[0]
        predicted_flat = self.predict(theta_final, torch.empty(0), N, **context)
        predicted = predicted_flat.reshape(cnd, self._ns, nc)
        return {"theta_post": theta_post, "predicted_output": predicted}
