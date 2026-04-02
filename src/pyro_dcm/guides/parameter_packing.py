"""Parameter packing utilities for amortized inference.

Converts between named Pyro sample site dictionaries and flat
standardized vectors required by Zuko NSF spline transforms. The
spline transforms operate on [-5, 5], so all features must be
standardized to approximately zero mean and unit variance before
passing to the flow.

Two packer classes are provided:

- **TaskDCMPacker**: Packs/unpacks A_free, C, noise_prec for
  task-based DCM (``task_dcm_model``).
- **SpectralDCMPacker**: Packs/unpacks A_free, noise_a, noise_b,
  noise_c, csd_noise_scale for spectral DCM (``spectral_dcm_model``).

LOG-SPACE CONTRACT
------------------
Positive-constrained parameters (``noise_prec``, ``csd_noise_scale``)
are stored in **log-space** in the packed vector. This ensures the NSF
spline flow operates on unconstrained real values. The wrapper model
(07-02) calls ``.exp()`` on the unpacked value to recover the positive
parameter. See 07-RESEARCH.md Pitfall 3 for standardization rationale.

References
----------
07-RESEARCH.md Section 3: Parameter Packing/Unpacking.
07-RESEARCH.md Pitfall 3: Spline domain truncation.
"""

from __future__ import annotations

import torch


class TaskDCMPacker:
    """Pack/unpack task DCM parameters to/from flat vectors.

    Handles the three sample sites from ``task_dcm_model``:
    ``A_free`` (N, N), ``C`` (N, M), and ``noise_prec`` (scalar).

    The packed vector has shape ``(n_features,)`` where
    ``n_features = N*N + N*M + 1``. The ordering is:
    ``[A_free.flatten(), C.flatten(), log(noise_prec)]``.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (N).
    n_inputs : int
        Number of experimental inputs (M).
    a_mask : torch.Tensor
        Binary structural mask for A, shape ``(N, N)``, float64.
    c_mask : torch.Tensor
        Binary structural mask for C, shape ``(N, M)``, float64.

    Attributes
    ----------
    n_features : int
        Total number of features in the packed vector.
    mean_ : torch.Tensor or None
        Per-element mean from ``fit_standardization``. Trailing
        underscore per coding conventions (fitted attribute).
    std_ : torch.Tensor or None
        Per-element standard deviation from ``fit_standardization``.

    Notes
    -----
    ``noise_prec`` is stored in log-space in the packed vector.
    This is the explicit contract with the wrapper model in 07-02,
    which calls ``params["noise_prec"].exp()`` on the unpacked value.
    """

    def __init__(
        self,
        n_regions: int,
        n_inputs: int,
        a_mask: torch.Tensor,
        c_mask: torch.Tensor,
    ) -> None:
        self.n_regions = n_regions
        self.n_inputs = n_inputs
        self.a_mask = a_mask
        self.c_mask = c_mask
        self.n_features = n_regions * n_regions + n_regions * n_inputs + 1

        # Standardization stats (fitted attributes)
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def pack(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack named parameters into a flat vector.

        Parameters
        ----------
        params : dict
            Dictionary with keys ``"A_free"`` (N, N), ``"C"`` (N, M),
            ``"noise_prec"`` (scalar, positive).

        Returns
        -------
        torch.Tensor
            Flat vector of shape ``(n_features,)``. The last element
            is ``log(noise_prec)`` (log-space contract).

        Examples
        --------
        >>> packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
        >>> params = {
        ...     "A_free": torch.randn(3, 3),
        ...     "C": torch.randn(3, 1),
        ...     "noise_prec": torch.tensor(10.0),
        ... }
        >>> z = packer.pack(params)
        >>> z.shape
        torch.Size([13])
        """
        a_flat = params["A_free"].flatten()
        c_flat = params["C"].flatten()
        # Log-space contract: store noise_prec in log-space
        log_prec = torch.log(params["noise_prec"]).reshape(1)
        return torch.cat([a_flat, c_flat, log_prec])

    def unpack(
        self, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Unpack flat vector into named parameter dict.

        Parameters
        ----------
        z : torch.Tensor
            Flat vector of shape ``(..., n_features)``. Supports
            arbitrary batch dimensions.

        Returns
        -------
        dict
            Dictionary with keys ``"A_free"`` (N, N), ``"C"`` (N, M),
            ``"noise_prec"`` (scalar). Note: ``noise_prec`` is still
            in log-space -- caller must call ``.exp()`` for positive
            precision.

        Examples
        --------
        >>> packer = TaskDCMPacker(3, 1, torch.ones(3, 3), torch.ones(3, 1))
        >>> z = torch.randn(13)
        >>> params = packer.unpack(z)
        >>> params["A_free"].shape
        torch.Size([3, 3])
        """
        N, M = self.n_regions, self.n_inputs
        a_end = N * N
        c_end = a_end + N * M

        batch_shape = z.shape[:-1]

        a_free = z[..., :a_end].reshape(*batch_shape, N, N)
        c_val = z[..., a_end:c_end].reshape(*batch_shape, N, M)
        # noise_prec remains in log-space (caller must .exp())
        noise_prec = z[..., c_end]

        return {
            "A_free": a_free,
            "C": c_val,
            "noise_prec": noise_prec,
        }

    def fit_standardization(
        self, dataset: list[dict[str, torch.Tensor]],
    ) -> None:
        """Compute per-element mean and std from training data.

        Packs all parameter dicts (applying log-transform to
        noise_prec), computes elementwise statistics, and stores
        them as ``self.mean_`` and ``self.std_``.

        Parameters
        ----------
        dataset : list of dict
            List of parameter dicts, each with keys ``"A_free"``,
            ``"C"``, ``"noise_prec"`` (positive, raw values).

        Notes
        -----
        Standardization is critical for NSF spline domain [-5, 5].
        See 07-RESEARCH.md Pitfall 3.
        """
        packed = torch.stack([self.pack(d) for d in dataset])
        self.mean_ = packed.mean(dim=0)
        self.std_ = packed.std(dim=0).clamp(min=1e-6)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize packed vector to zero mean, unit variance.

        Parameters
        ----------
        z : torch.Tensor
            Packed parameter vector(s).

        Returns
        -------
        torch.Tensor
            Standardized vector: ``(z - mean) / std``.
        """
        assert self.mean_ is not None, "Call fit_standardization first"
        return (z - self.mean_) / self.std_

    def unstandardize(self, z_std: torch.Tensor) -> torch.Tensor:
        """Reverse standardization.

        Parameters
        ----------
        z_std : torch.Tensor
            Standardized vector(s).

        Returns
        -------
        torch.Tensor
            Original-scale packed vector: ``z_std * std + mean``.
        """
        assert self.mean_ is not None, "Call fit_standardization first"
        return z_std * self.std_ + self.mean_


class SpectralDCMPacker:
    """Pack/unpack spectral DCM parameters to/from flat vectors.

    Handles the five sample sites from ``spectral_dcm_model``:
    ``A_free`` (N, N), ``noise_a`` (2, N), ``noise_b`` (2, 1),
    ``noise_c`` (2, N), and ``csd_noise_scale`` (scalar).

    The packed vector has shape ``(n_features,)`` where
    ``n_features = N*N + 2*N + 2 + 2*N + 1``. The ordering is:
    ``[A_free.flatten(), noise_a.flatten(), noise_b.flatten(),
    noise_c.flatten(), log(csd_noise_scale)]``.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (N).

    Attributes
    ----------
    n_features : int
        Total number of features in the packed vector.
    mean_ : torch.Tensor or None
        Per-element mean from ``fit_standardization``.
    std_ : torch.Tensor or None
        Per-element standard deviation from ``fit_standardization``.

    Notes
    -----
    ``csd_noise_scale`` is stored in log-space in the packed vector.
    Same contract as ``TaskDCMPacker.noise_prec``.
    """

    def __init__(self, n_regions: int) -> None:
        self.n_regions = n_regions
        self.n_features = (
            n_regions * n_regions  # A_free
            + 2 * n_regions        # noise_a
            + 2                    # noise_b
            + 2 * n_regions        # noise_c
            + 1                    # csd_noise_scale
        )

        # Standardization stats (fitted attributes)
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def pack(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack named parameters into a flat vector.

        Parameters
        ----------
        params : dict
            Dictionary with keys ``"A_free"`` (N, N),
            ``"noise_a"`` (2, N), ``"noise_b"`` (2, 1),
            ``"noise_c"`` (2, N), ``"csd_noise_scale"`` (scalar,
            positive).

        Returns
        -------
        torch.Tensor
            Flat vector of shape ``(n_features,)``. The last element
            is ``log(csd_noise_scale)``.

        Examples
        --------
        >>> packer = SpectralDCMPacker(3)
        >>> params = {
        ...     "A_free": torch.randn(3, 3),
        ...     "noise_a": torch.randn(2, 3),
        ...     "noise_b": torch.randn(2, 1),
        ...     "noise_c": torch.randn(2, 3),
        ...     "csd_noise_scale": torch.tensor(1.0),
        ... }
        >>> z = packer.pack(params)
        >>> z.shape
        torch.Size([22])
        """
        a_flat = params["A_free"].flatten()
        na_flat = params["noise_a"].flatten()
        nb_flat = params["noise_b"].flatten()
        nc_flat = params["noise_c"].flatten()
        log_scale = torch.log(params["csd_noise_scale"]).reshape(1)
        return torch.cat([a_flat, na_flat, nb_flat, nc_flat, log_scale])

    def unpack(
        self, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Unpack flat vector into named parameter dict.

        Parameters
        ----------
        z : torch.Tensor
            Flat vector of shape ``(..., n_features)``. Supports
            arbitrary batch dimensions.

        Returns
        -------
        dict
            Dictionary with keys ``"A_free"`` (N, N),
            ``"noise_a"`` (2, N), ``"noise_b"`` (2, 1),
            ``"noise_c"`` (2, N), ``"csd_noise_scale"`` (scalar,
            log-space).

        Examples
        --------
        >>> packer = SpectralDCMPacker(3)
        >>> z = torch.randn(22)
        >>> params = packer.unpack(z)
        >>> params["A_free"].shape
        torch.Size([3, 3])
        """
        N = self.n_regions
        batch_shape = z.shape[:-1]

        idx = 0

        a_free = z[..., idx:idx + N * N].reshape(*batch_shape, N, N)
        idx += N * N

        noise_a = z[..., idx:idx + 2 * N].reshape(*batch_shape, 2, N)
        idx += 2 * N

        noise_b = z[..., idx:idx + 2].reshape(*batch_shape, 2, 1)
        idx += 2

        noise_c = z[..., idx:idx + 2 * N].reshape(*batch_shape, 2, N)
        idx += 2 * N

        csd_noise_scale = z[..., idx]

        return {
            "A_free": a_free,
            "noise_a": noise_a,
            "noise_b": noise_b,
            "noise_c": noise_c,
            "csd_noise_scale": csd_noise_scale,
        }

    def fit_standardization(
        self, dataset: list[dict[str, torch.Tensor]],
    ) -> None:
        """Compute per-element mean and std from training data.

        Parameters
        ----------
        dataset : list of dict
            List of parameter dicts with spectral DCM site names.
            ``csd_noise_scale`` must be positive (raw values).
        """
        packed = torch.stack([self.pack(d) for d in dataset])
        self.mean_ = packed.mean(dim=0)
        self.std_ = packed.std(dim=0).clamp(min=1e-6)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize packed vector to zero mean, unit variance.

        Parameters
        ----------
        z : torch.Tensor
            Packed parameter vector(s).

        Returns
        -------
        torch.Tensor
            Standardized vector: ``(z - mean) / std``.
        """
        assert self.mean_ is not None, "Call fit_standardization first"
        return (z - self.mean_) / self.std_

    def unstandardize(self, z_std: torch.Tensor) -> torch.Tensor:
        """Reverse standardization.

        Parameters
        ----------
        z_std : torch.Tensor
            Standardized vector(s).

        Returns
        -------
        torch.Tensor
            Original-scale packed vector: ``z_std * std + mean``.
        """
        assert self.mean_ is not None, "Call fit_standardization first"
        return z_std * self.std_ + self.mean_
