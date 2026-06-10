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

from typing import Protocol, runtime_checkable

import torch

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
            options={"step_size": self._dt},
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
