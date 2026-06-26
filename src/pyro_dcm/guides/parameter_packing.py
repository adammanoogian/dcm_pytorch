"""Parameter packing utilities for amortized inference.

Converts between named Pyro sample site dictionaries and flat
standardized vectors required by Zuko NSF spline transforms. The
spline transforms operate on [-5, 5], so all features must be
standardized to approximately zero mean and unit variance before
passing to the flow.

Three packer classes are provided:

- **TaskDCMPacker**: Packs/unpacks A_free, C, noise_prec for
  task-based DCM (``task_dcm_model``).
- **SpectralDCMPacker**: Packs/unpacks A_free, noise_a, noise_b,
  noise_c, csd_noise_scale for spectral DCM (``spectral_dcm_model``).
- **LatentCircuitDCMPacker**: Packs/unpacks A_free, C, x0,
  noise_prec for hybrid VAE-DCM (``latent_circuit_dcm_model``).
  Uses sparse packing (only non-zero mask entries) for A and C.

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
        # MODEL-07 defense-in-depth: refuse bilinear sample-site keys. The
        # packer's n_features = N*N + N*M + 1 does not accommodate J*N*N
        # bilinear terms; amortized bilinear inference is deferred to v0.3.1
        # per D5 (.planning/STATE.md). The amortized_task_dcm_model wrapper
        # is the primary user surface; this guard catches direct callers
        # building bilinear params dicts for offline analysis (research
        # Section 6 defense-in-depth rationale).
        bilinear_keys = [k for k in params if k.startswith("B_free_")]
        if bilinear_keys:
            raise NotImplementedError(
                f"TaskDCMPacker.pack refuses bilinear sample sites "
                f"{sorted(bilinear_keys)}; amortized bilinear inference is "
                f"deferred to v0.3.1 per D5 (.planning/STATE.md). Use the "
                f"SVI path via create_guide(task_dcm_model) for bilinear DCM."
            )

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


class ERPDCMPacker:
    """Pack/unpack DCM-for-evoked-responses (CMC) parameters to/from vectors.

    Mirrors the FROZEN :class:`ERPDCMForward` parameter ordering for the
    amortized scalp-ERP path, so a packed latent and a VL ``theta`` are
    interchangeable element-for-element::

        A_free (4*N*N) | C_free (N*M) | T (4*N) | G (4*N) | S (N) | R (2*M)

    with unpack shapes ``A_free (4, N, N)``, ``C_free (N, M)``, ``T (N, 4)``,
    ``G (N, 4)``, ``S (N, 1)``, ``R (M, 2)``. The between-trial ``B`` is
    EXCLUDED (held FIXED in the amortized path, mirroring ``ERPDCMForward`` +
    the v0.3.0-D5 "amortized defers B" precedent).

    LOG-SPACE CONTRACT
    ------------------
    Unlike :class:`TaskDCMPacker` / :class:`SpectralDCMPacker` (which ``log()``
    a positive-constrained noise scalar), EVERY CMC free parameter is ALREADY
    unconstrained (``P.*`` free log-params). ``pack``/``unpack`` are therefore
    pure IDENTITY reshapes: there is NO ``.exp()`` at unpack and NO
    positive-constrained scalar. The packed vector is a verbatim
    ``ERPDCMForward.pack_params`` flattening.

    Parameters
    ----------
    N : int
        Number of sources (regions).
    M : int
        Number of driving inputs (columns of ``c_mask``).

    Attributes
    ----------
    n_features : int
        Total number of features in the packed vector
        (``4*N*N + N*M + 4*N + 4*N + N + 2*M``).
    mean_ : torch.Tensor or None
        Per-element mean from ``fit_standardization`` (fitted attribute).
    std_ : torch.Tensor or None
        Per-element standard deviation from ``fit_standardization``.
    """

    def __init__(self, N: int, M: int) -> None:
        self.n_regions = N
        self.n_inputs = M
        self.n_features = (
            4 * N * N  # A_free (4 routing blocks)
            + N * M  # C_free
            + 4 * N  # T
            + 4 * N  # G
            + N  # S
            + 2 * M  # R
        )

        # Standardization stats (fitted attributes)
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def pack(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack named CMC parameters into a flat vector (identity reshape).

        Parameters
        ----------
        params : dict
            Dictionary with keys ``"A_free"`` (4, N, N), ``"C_free"`` (N, M),
            ``"T"`` (N, 4), ``"G"`` (N, 4), ``"S"`` (N, 1), ``"R"`` (M, 2).

        Returns
        -------
        torch.Tensor
            Flat vector of shape ``(n_features,)``. No ``.log()`` is applied --
            every CMC free param is already unconstrained.
        """
        return torch.cat(
            [
                params["A_free"].flatten(),
                params["C_free"].flatten(),
                params["T"].flatten(),
                params["G"].flatten(),
                params["S"].flatten(),
                params["R"].flatten(),
            ]
        )

    def unpack(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack flat vector into named CMC parameters (identity reshape).

        Parameters
        ----------
        z : torch.Tensor
            Flat vector of shape ``(..., n_features)``. Supports arbitrary
            batch dimensions.

        Returns
        -------
        dict
            Dictionary with keys ``"A_free"`` (4, N, N), ``"C_free"`` (N, M),
            ``"T"`` (N, 4), ``"G"`` (N, 4), ``"S"`` (N, 1), ``"R"`` (M, 2).
            No ``.exp()`` is applied -- the values stay in free log-space.
        """
        N, M = self.n_regions, self.n_inputs
        batch_shape = z.shape[:-1]
        idx = 0

        a_free = z[..., idx : idx + 4 * N * N].reshape(*batch_shape, 4, N, N)
        idx += 4 * N * N

        c_free = z[..., idx : idx + N * M].reshape(*batch_shape, N, M)
        idx += N * M

        t = z[..., idx : idx + 4 * N].reshape(*batch_shape, N, 4)
        idx += 4 * N

        g = z[..., idx : idx + 4 * N].reshape(*batch_shape, N, 4)
        idx += 4 * N

        s = z[..., idx : idx + N].reshape(*batch_shape, N, 1)
        idx += N

        r = z[..., idx : idx + 2 * M].reshape(*batch_shape, M, 2)

        return {
            "A_free": a_free,
            "C_free": c_free,
            "T": t,
            "G": g,
            "S": s,
            "R": r,
        }

    def fit_standardization(
        self,
        dataset: list[dict[str, torch.Tensor]],
    ) -> None:
        """Compute per-element mean and std from training data.

        Parameters
        ----------
        dataset : list of dict
            List of parameter dicts with the CMC site names. All values are
            already unconstrained free log-params (no transform applied).

        Notes
        -----
        Standardization is critical for the NSF spline domain ``[-5, 5]``.
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


class LatentCircuitDCMPacker:
    """Pack/unpack DCM parameters for hybrid VAE-DCM.

    Packs ``A_free``, ``C``, ``x0`` (initial conditions), and
    ``noise_prec`` into a single flat vector for the encoder network.
    Includes standardization (``fit_standardization``, ``standardize``,
    ``unstandardize``) for training stability.

    Unlike ``TaskDCMPacker`` which packs the full ``(N, N)`` A matrix,
    this packer uses **sparse packing**: only the non-zero entries
    specified by ``a_mask`` and ``c_mask`` are stored. This is
    appropriate for latent circuit DCM where the connectivity mask
    defines which connections exist.

    Parameters
    ----------
    n_regions : int
        Number of brain regions / latent dimensions (N).
    n_inputs : int
        Number of experimental inputs (M).
    a_mask : torch.Tensor
        Binary structural mask for A, shape ``(N, N)``.
    c_mask : torch.Tensor
        Binary structural mask for C, shape ``(N, M)``.

    Attributes
    ----------
    total_dim : int
        Total number of packed parameters.
    mean_ : torch.Tensor or None
        Per-element mean from ``fit_standardization``.
    std_ : torch.Tensor or None
        Per-element standard deviation from ``fit_standardization``.

    Notes
    -----
    ``noise_prec`` is stored in log-space in the packed vector
    (same contract as ``TaskDCMPacker``).

    Dimension calculation:

    - ``n_a``: number of non-zero entries in ``a_mask``
    - ``n_c``: number of non-zero entries in ``c_mask``
    - ``x0``: ``n_regions``
    - ``noise_prec``: 1

    Total: ``n_a + n_c + n_regions + 1``

    Examples
    --------
    >>> a_mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.float32)
    >>> c_mask = torch.tensor([[1], [0]], dtype=torch.float32)
    >>> packer = LatentCircuitDCMPacker(2, 1, a_mask, c_mask)
    >>> packer.total_dim
    7
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
        self.a_mask = a_mask.bool()
        self.c_mask = c_mask.bool()

        # Sparse dimension counts
        self._n_a = int(self.a_mask.sum().item())
        self._n_c = int(self.c_mask.sum().item())

        # Standardization stats (fitted attributes)
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    @property
    def total_dim(self) -> int:
        """Total number of packed parameters."""
        return self._n_a + self._n_c + self.n_regions + 1

    def pack(
        self,
        a_free: torch.Tensor,
        c: torch.Tensor,
        x0: torch.Tensor,
        noise_prec: torch.Tensor,
    ) -> torch.Tensor:
        """Pack named parameters into a flat vector.

        Parameters
        ----------
        a_free : torch.Tensor
            Effective connectivity matrix, shape ``(N, N)``.
            Only entries where ``a_mask`` is True are packed.
        c : torch.Tensor
            Driving input weights, shape ``(N, M)``.
            Only entries where ``c_mask`` is True are packed.
        x0 : torch.Tensor
            Initial conditions, shape ``(N,)``.
        noise_prec : torch.Tensor
            Observation noise precision (scalar, positive).

        Returns
        -------
        torch.Tensor
            Flat vector of shape ``(total_dim,)``. The last element
            is ``log(noise_prec)`` (log-space contract).

        Examples
        --------
        >>> a_mask = torch.ones(3, 3)
        >>> c_mask = torch.ones(3, 1)
        >>> packer = LatentCircuitDCMPacker(3, 1, a_mask, c_mask)
        >>> flat = packer.pack(
        ...     torch.randn(3, 3), torch.randn(3, 1),
        ...     torch.zeros(3), torch.tensor(10.0),
        ... )
        >>> flat.shape
        torch.Size([13])
        """
        a_vals = a_free[self.a_mask]
        c_vals = c[self.c_mask]
        log_prec = torch.log(noise_prec).reshape(1)
        return torch.cat([a_vals, c_vals, x0.flatten(), log_prec])

    def unpack(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack flat vector into named parameter dict.

        Parameters
        ----------
        flat : torch.Tensor
            Flat vector of shape ``(total_dim,)``.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"A_free"``: shape ``(N, N)``, zeros where mask is False
            - ``"C"``: shape ``(N, M)``, zeros where mask is False
            - ``"x0"``: shape ``(N,)``
            - ``"noise_prec"``: scalar (log-space; caller must
              call ``.exp()`` for positive precision)

        Examples
        --------
        >>> a_mask = torch.ones(3, 3)
        >>> c_mask = torch.ones(3, 1)
        >>> packer = LatentCircuitDCMPacker(3, 1, a_mask, c_mask)
        >>> flat = torch.randn(13)
        >>> params = packer.unpack(flat)
        >>> params["A_free"].shape
        torch.Size([3, 3])
        """
        N, M = self.n_regions, self.n_inputs
        idx = 0

        # A_free: sparse unpack
        a_vals = flat[idx:idx + self._n_a]
        idx += self._n_a
        a_free = torch.zeros(N, N, dtype=flat.dtype, device=flat.device)
        a_free[self.a_mask] = a_vals

        # C: sparse unpack
        c_vals = flat[idx:idx + self._n_c]
        idx += self._n_c
        c_mat = torch.zeros(N, M, dtype=flat.dtype, device=flat.device)
        c_mat[self.c_mask] = c_vals

        # x0
        x0 = flat[idx:idx + N]
        idx += N

        # noise_prec (log-space)
        noise_prec = flat[idx]

        return {
            "A_free": a_free,
            "C": c_mat,
            "x0": x0,
            "noise_prec": noise_prec,
        }

    def fit_standardization(self, samples: torch.Tensor) -> None:
        """Compute per-element mean and std from packed samples.

        Parameters
        ----------
        samples : torch.Tensor
            Stacked packed vectors, shape ``(n_samples, total_dim)``.
            Each row is the output of ``pack()``.

        Notes
        -----
        Standardization is critical for encoder networks that expect
        approximately zero-mean, unit-variance targets.
        """
        self.mean_ = samples.mean(dim=0)
        self.std_ = samples.std(dim=0).clamp(min=1e-6)

    def standardize(self, flat: torch.Tensor) -> torch.Tensor:
        """Standardize packed vector to zero mean, unit variance.

        Parameters
        ----------
        flat : torch.Tensor
            Packed parameter vector(s).

        Returns
        -------
        torch.Tensor
            Standardized vector: ``(flat - mean) / std``.
        """
        assert self.mean_ is not None, "Call fit_standardization first"
        return (flat - self.mean_) / self.std_

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
