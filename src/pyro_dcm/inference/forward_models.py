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

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.inference.csd_precision import compute_csd_precision


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
        ny = observed.numel()
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
