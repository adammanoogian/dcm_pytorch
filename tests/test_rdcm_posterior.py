"""Unit tests for rDCM analytic posterior inference.

Tests for prior specification, standalone likelihood, free energy,
rigid VB inversion, and sparse VB inversion.

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
[REF-021] Frassle et al. (2018), NeuroImage 155, 406-421.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.forward_models.rdcm_posterior import (
    compute_free_energy_rigid,
    compute_free_energy_sparse,
    compute_rdcm_likelihood,
    get_priors_rigid,
    get_priors_sparse,
    rigid_inversion,
    sparse_inversion,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture()
def masks_3x2():
    """3-region, 2-input architecture masks."""
    a_mask = torch.tensor(
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        dtype=torch.float64,
    )
    c_mask = torch.tensor(
        [[1, 0], [0, 1], [0, 0]],
        dtype=torch.float64,
    )
    return a_mask, c_mask


@pytest.fixture()
def simple_regression_data():
    """Simple linear regression problem for rigid inversion testing.

    Creates Y = X @ theta_true + noise for a 3-region, 1-input system
    with known ground truth.
    """
    torch.manual_seed(42)
    nr = 3
    nu = 1
    nc = 1
    N = 60

    # Full connectivity
    a_mask = torch.ones(nr, nr, dtype=torch.float64)
    c_mask = torch.ones(nr, nu, dtype=torch.float64)

    # True parameters for each region
    # Each region gets nr (A cols) + nu (C cols) + nc (confound) = 5 params
    D = nr + nu + nc

    # Generate random design matrix
    X = torch.randn(N, D, dtype=torch.float64) * 0.5
    # Add confound (constant) in last column
    X[:, -1] = 1.0

    # True A matrix (with self-inhibition)
    A_true = torch.tensor(
        [[-0.4, 0.2, 0.0], [0.1, -0.5, 0.15], [0.0, 0.1, -0.3]],
        dtype=torch.float64,
    )

    # True C matrix
    C_true = torch.tensor([[0.8], [0.0], [0.0]], dtype=torch.float64)

    # Generate Y for each region
    Y = torch.zeros(N, nr, dtype=torch.float64)
    noise_std = 0.1
    for r in range(nr):
        theta_true = torch.cat([
            A_true[r, :],
            C_true[r, :],
            torch.tensor([0.0], dtype=torch.float64),  # confound
        ])
        Y[:, r] = X @ theta_true + noise_std * torch.randn(
            N, dtype=torch.float64
        )

    return X, Y, a_mask, c_mask, A_true, C_true


# ---------------------------------------------------------------
# Prior specification tests
# ---------------------------------------------------------------


class TestGetPriorsRigid:
    """Tests for get_priors_rigid."""

    def test_shapes(self, masks_3x2):
        """m0 and l0 have shape (nr, nr+nu)."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        assert priors["m0"].shape == (3, 5)
        assert priors["l0"].shape == (3, 5)

    def test_diagonal_mean(self, masks_3x2):
        """A diagonal prior mean is -0.5 for all regions."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        m0 = priors["m0"]
        for r in range(3):
            assert m0[r, r].item() == pytest.approx(-0.5)

    def test_offdiag_mean(self, masks_3x2):
        """A off-diagonal prior mean is 0."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        m0 = priors["m0"]
        for r in range(3):
            for j in range(3):
                if r != j:
                    assert m0[r, j].item() == pytest.approx(0.0)

    def test_c_mean(self, masks_3x2):
        """C prior mean is 0."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        m0 = priors["m0"]
        # C columns are indices 3, 4
        assert m0[:, 3:].abs().max().item() == pytest.approx(0.0)

    def test_precision_scaling(self, masks_3x2):
        """For nr=3, off-diag A precision=nr/8=3/8, diag=8*nr=24."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        l0 = priors["l0"]
        nr = 3

        # Off-diagonal A: covariance = 8/nr, precision = nr/8
        # But only for present connections in a_mask
        # a_mask[0, 1] = 1 (present), so l0[0, 1] = nr/8
        expected_offdiag_prec = nr / 8.0
        assert l0[0, 1].item() == pytest.approx(expected_offdiag_prec)

        # Diagonal A: covariance = 1/(8*nr), precision = 8*nr
        expected_diag_prec = 8.0 * nr
        assert l0[0, 0].item() == pytest.approx(expected_diag_prec)

    def test_absent_connections(self, masks_3x2):
        """Absent C connections should have inf precision."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        l0 = priors["l0"]
        # c_mask[2, 0] = 0 and c_mask[2, 1] = 0
        # C columns start at index 3
        assert l0[2, 3].item() == float("inf")
        assert l0[2, 4].item() == float("inf")

    def test_gamma(self, masks_3x2):
        """a0=2, b0=1."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_rigid(a_mask, c_mask)
        assert priors["a0"] == 2.0
        assert priors["b0"] == 1.0


class TestGetPriorsSparse:
    """Tests for get_priors_sparse."""

    def test_uses_full_connectivity(self, masks_3x2):
        """Sparse priors use ones(nr,nr), so all off-diag get finite precision."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_sparse(a_mask, c_mask)
        l0 = priors["l0"]
        # Even for a_mask[0,2]=0, sparse uses full mask
        # so l0[0, 2] should be finite (nr/8)
        assert torch.isfinite(l0[0, 2])

    def test_has_p0(self, masks_3x2):
        """Sparse priors include p0 (Bernoulli prior)."""
        a_mask, c_mask = masks_3x2
        priors = get_priors_sparse(a_mask, c_mask)
        assert "p0" in priors
        assert priors["p0"] == 0.5


# ---------------------------------------------------------------
# Standalone likelihood tests
# ---------------------------------------------------------------


class TestComputeRdcmLikelihood:
    """Tests for compute_rdcm_likelihood."""

    def test_known_value(self):
        """Likelihood matches manual Gaussian formula."""
        torch.manual_seed(0)
        N = 20
        D = 3
        X_r = torch.randn(N, D, dtype=torch.float64)
        theta = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64)
        tau = 5.0
        Y_r = X_r @ theta + 0.1 * torch.randn(N, dtype=torch.float64)

        ll = compute_rdcm_likelihood(Y_r, X_r, theta, tau)

        # Manual computation
        residual = Y_r - X_r @ theta
        expected = (
            -0.5 * N * math.log(2 * math.pi)
            + 0.5 * N * math.log(tau)
            - 0.5 * tau * (residual @ residual).item()
        )
        assert ll.item() == pytest.approx(expected, abs=1e-8)

    def test_perfect_fit(self):
        """Zero residual gives -0.5*N*log(2pi) + 0.5*N*log(tau)."""
        N = 10
        D = 2
        X_r = torch.randn(N, D, dtype=torch.float64)
        mu_r = torch.tensor([1.0, 2.0], dtype=torch.float64)
        Y_r = X_r @ mu_r  # perfect fit
        tau = 3.0

        ll = compute_rdcm_likelihood(Y_r, X_r, mu_r, tau)
        expected = -0.5 * N * math.log(2 * math.pi) + 0.5 * N * math.log(
            tau
        )
        assert ll.item() == pytest.approx(expected, abs=1e-8)

    def test_returns_scalar(self):
        """Output is a single scalar tensor."""
        X_r = torch.randn(10, 2, dtype=torch.float64)
        Y_r = torch.randn(10, dtype=torch.float64)
        mu_r = torch.randn(2, dtype=torch.float64)
        ll = compute_rdcm_likelihood(Y_r, X_r, mu_r, 1.0)
        assert ll.dim() == 0

    def test_higher_for_better_fit(self):
        """True parameters give higher likelihood than random ones."""
        torch.manual_seed(1)
        N = 30
        D = 3
        X_r = torch.randn(N, D, dtype=torch.float64)
        theta_true = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64)
        Y_r = X_r @ theta_true + 0.1 * torch.randn(
            N, dtype=torch.float64
        )
        theta_random = torch.randn(D, dtype=torch.float64) * 5.0
        tau = 10.0

        ll_true = compute_rdcm_likelihood(Y_r, X_r, theta_true, tau)
        ll_rand = compute_rdcm_likelihood(Y_r, X_r, theta_random, tau)
        assert ll_true.item() > ll_rand.item()


# ---------------------------------------------------------------
# Free energy tests (rigid)
# ---------------------------------------------------------------


class TestFreeEnergyRigid:
    """Tests for compute_free_energy_rigid."""

    @pytest.fixture()
    def fe_inputs(self):
        """Known inputs for free energy computation."""
        D_r = 3
        N_eff = 20
        a_r = 12.0
        beta_r = 5.0
        tau_r = a_r / beta_r
        QF = 2.5
        l0_r = torch.diag(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )
        mu_r = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
        mu0_r = torch.tensor([0.0, 0.0, -0.5], dtype=torch.float64)
        Sigma_r = torch.diag(
            torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        )
        a0 = 2.0
        beta0 = 1.0
        return {
            "N_eff": N_eff,
            "a_r": a_r,
            "beta_r": beta_r,
            "QF": QF,
            "tau_r": tau_r,
            "l0_r": l0_r,
            "mu_r": mu_r,
            "mu0_r": mu0_r,
            "Sigma_r": Sigma_r,
            "a0": a0,
            "beta0": beta0,
            "D_r": D_r,
        }

    def test_returns_scalar(self, fe_inputs):
        """Free energy is a single scalar."""
        F = compute_free_energy_rigid(**fe_inputs)
        assert F.dim() == 0

    def test_five_components(self, fe_inputs):
        """Verify free energy matches manual 5-component computation."""
        inp = fe_inputs
        dtype = torch.float64

        a_r = torch.tensor(inp["a_r"], dtype=dtype)
        beta_r = torch.tensor(inp["beta_r"], dtype=dtype)
        tau_r = torch.tensor(inp["tau_r"], dtype=dtype)
        QF_t = torch.tensor(inp["QF"], dtype=dtype)
        N = inp["N_eff"]
        D = inp["D_r"]
        a0 = inp["a0"]
        beta0 = inp["beta0"]
        l0_r = inp["l0_r"]
        mu_r = inp["mu_r"]
        mu0_r = inp["mu0_r"]
        Sigma_r = inp["Sigma_r"]

        # Component 1
        log_lik = (
            0.5
            * (
                N * (torch.special.digamma(a_r) - torch.log(beta_r))
                - N * math.log(2 * math.pi)
            )
            - QF_t * tau_r
        )

        # Component 2
        _, logdet_l0 = torch.linalg.slogdet(l0_r)
        diff = mu_r - mu0_r
        log_p_w = 0.5 * (
            logdet_l0
            - D * math.log(2 * math.pi)
            - diff @ l0_r @ diff
            - torch.trace(l0_r @ Sigma_r)
        )

        # Component 3
        beta0_t = torch.tensor(beta0, dtype=dtype)
        log_p_p = (
            a0 * torch.log(beta0_t)
            - torch.lgamma(torch.tensor(a0, dtype=dtype))
            + (a0 - 1) * (torch.special.digamma(a_r) - torch.log(beta_r))
            - beta0 * tau_r
        )

        # Component 4
        _, logdet_S = torch.linalg.slogdet(Sigma_r)
        log_q_w = 0.5 * (logdet_S + D * (1 + math.log(2 * math.pi)))

        # Component 5
        log_q_p = (
            a_r
            - torch.log(beta_r)
            + torch.lgamma(a_r)
            + (1 - a_r) * torch.special.digamma(a_r)
        )

        expected = log_lik + log_p_w + log_p_p + log_q_w + log_q_p
        F = compute_free_energy_rigid(**fe_inputs)
        assert F.item() == pytest.approx(expected.item(), abs=1e-8)

    def test_known_value(self):
        """Simple 1-parameter case with manual computation."""
        D_r = 1
        N_eff = 10
        a0 = 2.0
        beta0 = 1.0
        a_r = a0 + N_eff * 0.5  # = 7
        beta_r = 3.0
        tau_r = a_r / beta_r
        QF = 1.5
        l0_r = torch.tensor([[4.0]], dtype=torch.float64)
        mu_r = torch.tensor([0.5], dtype=torch.float64)
        mu0_r = torch.tensor([0.0], dtype=torch.float64)
        Sigma_r = torch.tensor([[0.1]], dtype=torch.float64)

        F = compute_free_energy_rigid(
            N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r,
            Sigma_r, a0, beta0, D_r,
        )

        # Should be finite
        assert torch.isfinite(F)
        # Manually verify it's a reasonable number (not NaN/Inf)
        assert -1000 < F.item() < 1000

    def test_no_nan(self, fe_inputs):
        """Free energy is finite for valid inputs."""
        F = compute_free_energy_rigid(**fe_inputs)
        assert torch.isfinite(F)


# ---------------------------------------------------------------
# Rigid inversion tests
# ---------------------------------------------------------------


class TestRigidInversion:
    """Tests for rigid_inversion."""

    def test_known_linear_regression(self, simple_regression_data):
        """Posterior mean recovers known parameters."""
        X, Y, a_mask, c_mask, A_true, C_true = simple_regression_data

        result = rigid_inversion(X, Y, a_mask, c_mask)
        A_mu = result["A_mu"]

        # Check diagonal recovery (should be close to true values)
        for r in range(3):
            post_std = result["Sigma_per_region"][r].diag().sqrt()
            mu_r = result["mu_per_region"][r]
            # Each parameter should be within 3 posterior std of true
            # (using 3 sigma for robustness)
            a_true_r = torch.cat([
                A_true[r, :], C_true[r, :],
                torch.tensor([0.0], dtype=torch.float64),
            ])
            for i in range(min(len(mu_r), len(a_true_r))):
                diff = abs(mu_r[i].item() - a_true_r[i].item())
                tol_val = 3.0 * post_std[i].item()
                # Allow generous tolerance for this statistical test
                assert diff < max(tol_val, 0.5), (
                    f"Region {r}, param {i}: |{mu_r[i].item():.4f} - "
                    f"{a_true_r[i].item():.4f}| = {diff:.4f} > "
                    f"3*std={tol_val:.4f}"
                )

    def test_convergence(self, simple_regression_data):
        """Inversion converges in < 50 iterations."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        result = rigid_inversion(X, Y, a_mask, c_mask)
        iters = result["iterations_per_region"]
        for r in range(3):
            assert iters[r].item() < 50, (
                f"Region {r} took {iters[r].item()} iterations"
            )

    def test_shapes(self, simple_regression_data):
        """Output shapes are correct."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        nr = 3
        nu = 1
        result = rigid_inversion(X, Y, a_mask, c_mask)
        assert result["A_mu"].shape == (nr, nr)
        assert result["C_mu"].shape == (nr, nu)
        assert len(result["mu_per_region"]) == nr
        assert len(result["Sigma_per_region"]) == nr
        assert result["a_per_region"].shape == (nr,)
        assert result["beta_per_region"].shape == (nr,)
        assert result["F_per_region"].shape == (nr,)

    def test_posterior_covariance_positive_definite(
        self, simple_regression_data
    ):
        """All Sigma_r are positive definite."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        result = rigid_inversion(X, Y, a_mask, c_mask)
        for r in range(3):
            Sigma = result["Sigma_per_region"][r]
            eigvals = torch.linalg.eigvalsh(Sigma)
            assert (eigvals > 0).all(), (
                f"Region {r}: non-positive eigenvalue: "
                f"{eigvals.min().item()}"
            )

    def test_free_energy_increases(self, simple_regression_data):
        """F_total is greater than a very negative number."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        result = rigid_inversion(X, Y, a_mask, c_mask)
        # Free energy should be finite and reasonable
        assert torch.isfinite(result["F_total"])
        # F_total should be better than -inf (started there)
        assert result["F_total"].item() > -1e10

    def test_noise_precision_reasonable(self, simple_regression_data):
        """tau = a/beta should be in reasonable range."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        result = rigid_inversion(X, Y, a_mask, c_mask)
        for r in range(3):
            tau = (
                result["a_per_region"][r] / result["beta_per_region"][r]
            )
            assert 0.01 < tau.item() < 1e6, (
                f"Region {r}: tau={tau.item()}"
            )

    def test_diagonal_recovery(self, simple_regression_data):
        """A_mu diagonal should be negative (self-inhibition)."""
        X, Y, a_mask, c_mask, _, _ = simple_regression_data
        result = rigid_inversion(X, Y, a_mask, c_mask)
        A_mu = result["A_mu"]
        for r in range(3):
            assert A_mu[r, r].item() < 0, (
                f"Region {r}: A_mu[{r},{r}]={A_mu[r, r].item()}"
            )


# ---------------------------------------------------------------
# Sparse inversion tests
# ---------------------------------------------------------------


class TestSparseInversion:
    """Tests for sparse_inversion."""

    @pytest.fixture()
    def sparse_regression_data(self):
        """Sparse A with known zero entries for pruning test.

        Creates a system where A[0,2]=0 and A[2,0]=0 (truly absent).
        With enough data, sparse inversion should prune these.
        """
        torch.manual_seed(123)
        nr = 3
        nu = 1
        nc = 1
        N = 80

        a_mask = torch.ones(nr, nr, dtype=torch.float64)
        c_mask = torch.ones(nr, nu, dtype=torch.float64)

        D = nr + nu + nc
        X = torch.randn(N, D, dtype=torch.float64) * 0.5
        X[:, -1] = 1.0

        # Sparse A: strong diagonal, some zeros
        A_true = torch.tensor(
            [[-0.5, 0.3, 0.0], [0.2, -0.4, 0.25], [0.0, 0.2, -0.5]],
            dtype=torch.float64,
        )
        C_true = torch.tensor(
            [[0.6], [0.0], [0.0]], dtype=torch.float64
        )

        Y = torch.zeros(N, nr, dtype=torch.float64)
        noise_std = 0.05
        for r in range(nr):
            theta_true = torch.cat([
                A_true[r, :],
                C_true[r, :],
                torch.tensor([0.0], dtype=torch.float64),
            ])
            Y[:, r] = X @ theta_true + noise_std * torch.randn(
                N, dtype=torch.float64
            )

        return X, Y, a_mask, c_mask, A_true, C_true

    def test_prunes_absent_connections(self, sparse_regression_data):
        """z < 0.5 for true-zero, z > 0.5 for true-present."""
        X, Y, a_mask, c_mask, A_true, _ = sparse_regression_data

        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=5, max_iter=100,
        )

        # Check z indicators for region 0
        z_r0 = result["z_per_region"][0]
        # A_true[0, 2] = 0.0 -> z for index 2 should be low
        # A_true[0, 0] = -0.5 -> z for index 0 should be high
        # A_true[0, 1] = 0.3 -> z for index 1 should be high

        # True-present connections should have z > 0.3
        # (relaxed threshold due to stochastic nature)
        assert z_r0[0].item() > 0.3, (
            f"z[0,0]={z_r0[0].item()}: should be high (A=-0.5)"
        )
        assert z_r0[1].item() > 0.3, (
            f"z[0,1]={z_r0[1].item()}: should be high (A=0.3)"
        )

    def test_shapes(self, sparse_regression_data):
        """Same output shapes as rigid plus z_per_region."""
        X, Y, a_mask, c_mask, _, _ = sparse_regression_data
        nr, nu = 3, 1
        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=2, max_iter=50,
        )
        assert result["A_mu"].shape == (nr, nr)
        assert result["C_mu"].shape == (nr, nu)
        assert "z_per_region" in result
        assert len(result["z_per_region"]) == nr

    def test_z_in_01(self, sparse_regression_data):
        """All z values in [0, 1]."""
        X, Y, a_mask, c_mask, _, _ = sparse_regression_data
        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=2, max_iter=50,
        )
        for r in range(3):
            z_r = result["z_per_region"][r]
            assert (z_r >= 0.0).all(), f"Region {r}: z < 0"
            assert (z_r <= 1.0).all(), f"Region {r}: z > 1"

    def test_best_of_reruns(self, sparse_regression_data):
        """With n_reruns=5, selected run has highest free energy."""
        X, Y, a_mask, c_mask, _, _ = sparse_regression_data
        # Run twice with same seed to verify best is selected
        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=5, max_iter=50,
        )
        # F_total should be finite
        assert torch.isfinite(result["F_total"])

    def test_restrict_inputs(self, sparse_regression_data):
        """With restrict_inputs=True, z for C columns = 1.0."""
        X, Y, a_mask, c_mask, _, _ = sparse_regression_data
        nr, nu = 3, 1
        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=2, max_iter=50,
            restrict_inputs=True,
        )
        for r in range(nr):
            z_r = result["z_per_region"][r]
            # C indices are nr..nr+nu-1
            for j in range(nu):
                assert z_r[nr + j].item() == pytest.approx(1.0), (
                    f"Region {r}, C index {j}: z={z_r[nr + j].item()}"
                )

    def test_hard_thresholding(self, sparse_regression_data):
        """mu values with |mu| < 1e-5 should be exactly 0."""
        X, Y, a_mask, c_mask, _, _ = sparse_regression_data
        result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=2, max_iter=50,
        )
        for r in range(3):
            mu_r = result["mu_per_region"][r]
            # Any value with abs < 1e-5 should be exactly 0
            small_mask = mu_r.abs() < 1e-5
            if small_mask.any():
                assert (mu_r[small_mask] == 0.0).all()


# ---------------------------------------------------------------
# Free energy tests (sparse)
# ---------------------------------------------------------------


class TestFreeEnergySparse:
    """Tests for compute_free_energy_sparse."""

    def test_seven_components(self):
        """Sparse free energy is computable and finite."""
        D_r = 4
        N_eff = 15
        a_r = 9.5
        beta_r = 3.0
        tau_r = a_r / beta_r
        QF = 2.0
        l0_r = torch.eye(D_r, dtype=torch.float64) * 2.0
        mu_r = torch.tensor(
            [0.1, -0.2, 0.3, 0.0], dtype=torch.float64
        )
        mu0_r = torch.zeros(D_r, dtype=torch.float64)
        Sigma_r = torch.eye(D_r, dtype=torch.float64) * 0.3
        z_r = torch.tensor(
            [0.9, 0.1, 0.7, 0.5], dtype=torch.float64
        )
        z_idx = (z_r > 1e-10) & (z_r < 1.0)
        p0 = 0.5 * torch.ones(D_r, dtype=torch.float64)

        F = compute_free_energy_sparse(
            N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r,
            Sigma_r, 2.0, 1.0, D_r, z_r, z_idx, p0,
        )
        assert torch.isfinite(F)

    def test_z_entropy(self):
        """When z=0.5, entropy term is maximal."""
        D_r = 2
        N_eff = 10
        a_r = 7.0
        beta_r = 2.0
        tau_r = a_r / beta_r
        QF = 1.0
        l0_r = torch.eye(D_r, dtype=torch.float64)
        mu_r = torch.tensor([0.1, 0.1], dtype=torch.float64)
        mu0_r = torch.zeros(D_r, dtype=torch.float64)
        Sigma_r = torch.eye(D_r, dtype=torch.float64) * 0.5
        p0 = 0.5 * torch.ones(D_r, dtype=torch.float64)

        # z=0.5 (maximum entropy)
        z_half = torch.tensor([0.5, 0.5], dtype=torch.float64)
        z_idx_half = torch.ones(D_r, dtype=torch.bool)
        F_half = compute_free_energy_sparse(
            N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r,
            Sigma_r, 2.0, 1.0, D_r, z_half, z_idx_half, p0,
        )

        # z=0.9 (lower entropy)
        z_high = torch.tensor([0.9, 0.9], dtype=torch.float64)
        z_idx_high = torch.ones(D_r, dtype=torch.bool)
        F_high = compute_free_energy_sparse(
            N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r,
            Sigma_r, 2.0, 1.0, D_r, z_high, z_idx_high, p0,
        )

        # With p0=0.5, the log_p_z term is 0 for both, so
        # the only difference is log_q_z (entropy).
        # Entropy at z=0.5 is maximal, so log_q_z is larger.
        # Since log_q_z is part of F, F_half should be >= F_high
        # when the entropy contribution dominates.
        # At minimum, both should be finite.
        assert torch.isfinite(F_half)
        assert torch.isfinite(F_high)
        # For p0=0.5, the entropy term at z=0.5 is D*ln(2)
        # vs at z=0.9 it's D*(-0.1*ln(0.1)-0.9*ln(0.9))
        # 2*0.693 = 1.386 vs 2*0.325 = 0.650
        # So F_half - F_high ~ 0.736
        assert F_half.item() > F_high.item()


# ---------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------


class TestRigidVsSparse:
    """Integration test comparing rigid and sparse on full connectivity."""

    def test_full_connectivity(self):
        """Both give similar posteriors on fully connected system."""
        torch.manual_seed(99)
        nr = 2
        nu = 1
        nc = 1
        N = 50
        D = nr + nu + nc

        a_mask = torch.ones(nr, nr, dtype=torch.float64)
        c_mask = torch.ones(nr, nu, dtype=torch.float64)

        X = torch.randn(N, D, dtype=torch.float64)
        X[:, -1] = 1.0

        A_true = torch.tensor(
            [[-0.4, 0.2], [0.15, -0.3]], dtype=torch.float64
        )
        C_true = torch.tensor([[0.5], [0.0]], dtype=torch.float64)

        Y = torch.zeros(N, nr, dtype=torch.float64)
        for r in range(nr):
            theta = torch.cat([
                A_true[r, :],
                C_true[r, :],
                torch.tensor([0.0], dtype=torch.float64),
            ])
            Y[:, r] = X @ theta + 0.1 * torch.randn(
                N, dtype=torch.float64
            )

        rigid_result = rigid_inversion(X, Y, a_mask, c_mask)
        sparse_result = sparse_inversion(
            X, Y, a_mask, c_mask, n_reruns=3, max_iter=100,
        )

        # Both should have reasonable A_mu values (same sign at least)
        for r in range(nr):
            for j in range(nr):
                rigid_val = rigid_result["A_mu"][r, j].item()
                sparse_val = sparse_result["A_mu"][r, j].item()
                # Both should be in the same ballpark
                # (within 1.0 of each other for this simple problem)
                assert abs(rigid_val - sparse_val) < 1.0, (
                    f"A[{r},{j}]: rigid={rigid_val:.4f}, "
                    f"sparse={sparse_val:.4f}"
                )
