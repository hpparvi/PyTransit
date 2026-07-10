"""Verification tests for limb darkening functions.

Tests the analytical disk-integrated fluxes (ldi_*) against numerical integration
of the intensity profiles (ld_*), and the analytical gradients (ldd_*) against
finite differences.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

from pytransit.models.limb_darkening import (
    ld_uniform, ldi_uniform,
    ld_linear, ldi_linear, ldd_linear,
    ld_quadratic, ldi_quadratic, ldd_quadratic,
    ld_quadratic_tri, ldi_quadratic_tri,
    ld_power_2, ldi_power_2, ldd_power_2,
    ld_power_2_pm, ldi_power_2_pm,
    ld_nonlinear, ldi_nonlinear,
    ld_general, ldi_general,
    ld_square_root, ldi_square_root,
    ld_logarithmic, ldi_logarithmic,
    ld_exponential, ldi_exponential,
    evaluate_ld, evaluate_ldi,
)


def numerical_ldi(ld_func, pv, n=10000):
    """Numerically integrate 2π ∫₀¹ I(μ) μ dμ using scipy.integrate.quad."""
    def integrand(mu):
        return ld_func(np.array([mu]), pv)[0] * mu if hasattr(ld_func(np.array([mu]), pv), '__len__') else ld_func(np.array([mu]), pv) * mu
    result, _ = quad(integrand, 0, 1)
    return 2 * np.pi * result


def numerical_gradient(ld_func, mu_arr, pv, eps=1e-7):
    """Compute finite-difference gradients [dI/dμ, dI/dp₀, dI/dp₁, ...]."""
    n_params = pv.size
    n_mu = mu_arr.size
    grad = np.zeros((1 + n_params, n_mu))

    # dI/dμ by central difference
    for j, mu_val in enumerate(mu_arr):
        mu_p = np.array([mu_val + eps])
        mu_m = np.array([mu_val - eps])
        ip = ld_func(mu_p, pv)
        im = ld_func(mu_m, pv)
        val_p = ip[0] if hasattr(ip, '__len__') else ip
        val_m = im[0] if hasattr(im, '__len__') else im
        grad[0, j] = (val_p - val_m) / (2 * eps)

    # dI/dpᵢ by central difference
    for i in range(n_params):
        pv_p = pv.copy()
        pv_m = pv.copy()
        pv_p[i] += eps
        pv_m[i] -= eps
        ip = ld_func(mu_arr, pv_p)
        im = ld_func(mu_arr, pv_m)
        if hasattr(ip, '__len__'):
            grad[1 + i] = (ip - im) / (2 * eps)
        else:
            grad[1 + i] = (ip - im) / (2 * eps)

    return grad


# ============================================================
# Disk-integrated flux tests
# ============================================================

class TestIntegrals:
    """Test ldi_* against numerical integration of ld_*."""

    def test_uniform(self):
        pv = np.array([])
        assert_allclose(ldi_uniform(pv), numerical_ldi(ld_uniform, pv), rtol=1e-10)

    @pytest.mark.parametrize("u", [0.0, 0.3, 0.6, 1.0])
    def test_linear(self, u):
        pv = np.array([u])
        assert_allclose(ldi_linear(pv), numerical_ldi(ld_linear, pv), rtol=1e-10)

    def test_linear_u0_gives_pi(self):
        """u=0 reduces to uniform disk, integral = π."""
        assert_allclose(ldi_linear(np.array([0.0])), np.pi, rtol=1e-12)

    def test_linear_u1(self):
        """u=1: I(μ)=μ, integral = 2π/3."""
        assert_allclose(ldi_linear(np.array([1.0])), 2 * np.pi / 3, rtol=1e-12)

    @pytest.mark.parametrize("pv", [
        np.array([0.0, 0.0]),
        np.array([0.3, 0.2]),
        np.array([0.5, 0.3]),
        np.array([0.8, 0.5]),
    ])
    def test_quadratic(self, pv):
        assert_allclose(ldi_quadratic(pv), numerical_ldi(ld_quadratic, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.1, 0.1]),
        np.array([0.5, 0.3]),
        np.array([0.8, 0.5]),
        np.array([1.0, 0.5]),
    ])
    def test_quadratic_tri(self, pv):
        assert_allclose(ldi_quadratic_tri(pv), numerical_ldi(ld_quadratic_tri, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.3, 0.7]),
        np.array([0.5, 1.0]),
        np.array([0.8, 0.5]),
        np.array([0.0, 1.0]),
    ])
    def test_power_2(self, pv):
        assert_allclose(ldi_power_2(pv), numerical_ldi(ld_power_2, pv), rtol=1e-10)

    def test_power_2_c0_gives_pi(self):
        """c=0 reduces to uniform disk, integral = π."""
        assert_allclose(ldi_power_2(np.array([0.0, 1.0])), np.pi, rtol=1e-12)

    @pytest.mark.parametrize("pv", [
        np.array([0.3, 0.2]),
        np.array([0.5, 0.3]),
        np.array([0.0, 0.0]),
    ])
    def test_square_root(self, pv):
        assert_allclose(ldi_square_root(pv), numerical_ldi(ld_square_root, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.3, 0.2]),
        np.array([0.5, 0.3]),
        np.array([0.0, 0.0]),
    ])
    def test_logarithmic(self, pv):
        assert_allclose(ldi_logarithmic(pv), numerical_ldi(ld_logarithmic, pv), rtol=1e-8)

    @pytest.mark.parametrize("pv", [
        np.array([0.7, 0.2]),
        np.array([0.8, 0.05]),
    ])
    def test_power_2_pm(self, pv):
        assert_allclose(ldi_power_2_pm(pv), numerical_ldi(ld_power_2_pm, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.4, 0.2, 0.1, 0.05]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.2, -0.1, 0.3, -0.05]),
    ])
    def test_nonlinear(self, pv):
        assert_allclose(ldi_nonlinear(pv), numerical_ldi(ld_nonlinear, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.6]),
        np.array([0.3, 0.15]),
        np.array([0.3, 0.15, 0.05, 0.02]),
    ])
    def test_general(self, pv):
        assert_allclose(ldi_general(pv), numerical_ldi(ld_general, pv), rtol=1e-10)

    @pytest.mark.parametrize("pv", [
        np.array([0.5, 0.1]),
        np.array([0.3, 0.05]),
        np.array([0.0, 0.0]),
    ])
    def test_exponential(self, pv):
        assert_allclose(ldi_exponential(pv), numerical_ldi(ld_exponential, pv), rtol=1e-8)


# ============================================================
# Gradient tests
# ============================================================

class TestGradients:
    """Test ldd_* against finite-difference gradients of ld_*."""

    mu_test = np.array([0.01, 0.1, 0.3, 0.5, 0.8, 0.99])

    @pytest.mark.parametrize("pv", [
        np.array([0.3]),
        np.array([0.7]),
    ])
    def test_linear(self, pv):
        ana = ldd_linear(self.mu_test, pv)
        num = numerical_gradient(ld_linear, self.mu_test, pv)
        assert_allclose(ana, num, atol=1e-6)

    @pytest.mark.parametrize("pv", [
        np.array([0.3, 0.2]),
        np.array([0.5, 0.3]),
    ])
    def test_quadratic(self, pv):
        ana = ldd_quadratic(self.mu_test, pv)
        num = numerical_gradient(ld_quadratic, self.mu_test, pv)
        assert_allclose(ana, num, atol=1e-6)

    @pytest.mark.parametrize("pv", [
        np.array([0.3, 0.7]),
        np.array([0.8, 0.5]),
    ])
    def test_power_2(self, pv):
        # Avoid mu=0 where dI/dmu diverges for alpha < 1
        mu = self.mu_test[self.mu_test > 0.05]
        ana = ldd_power_2(mu, pv)
        num = numerical_gradient(ld_power_2, mu, pv)
        assert_allclose(ana, num, atol=1e-5)


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    """Test edge cases: μ=0, μ=1, zero coefficients."""

    def test_all_ld_at_mu1(self):
        """All limb darkening laws should give I(1) = 1 (or sum of coeffs for general)."""
        mu = np.array([1.0])
        assert_allclose(ld_linear(mu, np.array([0.5])), 1.0)
        assert_allclose(ld_quadratic(mu, np.array([0.3, 0.2])), 1.0)
        assert_allclose(ld_quadratic_tri(mu, np.array([0.5, 0.3])), 1.0)
        assert_allclose(ld_power_2(mu, np.array([0.3, 0.7])), 1.0)
        assert_allclose(ld_nonlinear(mu, np.array([0.1, 0.2, 0.1, 0.05])), 1.0)

    def test_general_at_mu1(self):
        """Giménez (2006) general law at μ=1: I(1) = 1 - Σ cᵢ(1-1) = 1."""
        mu = np.array([1.0])
        assert_allclose(ld_general(mu, np.array([0.5, 0.3])), 1.0)

    def test_general_reduces_to_linear(self):
        """General law with a single coefficient should match the linear law."""
        mu = np.linspace(0.01, 1.0, 50)
        u = 0.6
        assert_allclose(ld_general(mu, np.array([u])), ld_linear(mu, np.array([u])), rtol=1e-12)
        assert_allclose(ldi_general(np.array([u])), ldi_linear(np.array([u])), rtol=1e-12)

    def test_general_reference_formula(self):
        """Giménez (2006): I(μ) = 1 - Σ cᵢ(1 - μ^(i+1))."""
        mu = np.linspace(0.01, 1.0, 50)
        c = np.array([0.3, 0.15, 0.05])
        ref = 1.0 - sum(c[i] * (1.0 - mu ** (i + 1)) for i in range(c.size))
        assert_allclose(ld_general(mu, c), ref)

    def test_all_ld_zero_coeffs(self):
        """Zero limb darkening coefficients reduce all laws to I(μ) = 1."""
        mu = np.array([0.0, 0.5, 1.0])
        assert_allclose(ld_linear(mu, np.array([0.0])), np.ones(3))
        assert_allclose(ld_quadratic(mu, np.array([0.0, 0.0])), np.ones(3))
        assert_allclose(ld_power_2(mu, np.array([0.0, 1.0])), np.ones(3))
        assert_allclose(ld_nonlinear(mu, np.array([0.0, 0.0, 0.0, 0.0])), np.ones(3))

    def test_quadratic_reduces_to_linear(self):
        """Quadratic with v=0 should match linear."""
        mu = np.linspace(0.01, 1.0, 50)
        u = 0.6
        assert_allclose(
            ld_quadratic(mu, np.array([u, 0.0])),
            ld_linear(mu, np.array([u])),
            rtol=1e-12,
        )
        assert_allclose(
            ldi_quadratic(np.array([u, 0.0])),
            ldi_linear(np.array([u])),
            rtol=1e-12,
        )


# ============================================================
# Profile tests against independent reference formulas
# ============================================================

class TestProfiles:
    """Test ld_* profiles against independently written reference formulas."""

    mu = np.linspace(0.01, 1.0, 200)

    def test_square_root(self):
        """Square-root law: I(μ) = 1 - a(1-μ) - b(1-√μ)."""
        a, b = 0.4, 0.3
        assert_allclose(ld_square_root(self.mu, np.array([a, b])),
                        1.0 - a * (1.0 - self.mu) - b * (1.0 - np.sqrt(self.mu)))

    def test_square_root_reduces_to_linear(self):
        """Square-root with b=0 should match linear."""
        u = 0.6
        assert_allclose(ld_square_root(self.mu, np.array([u, 0.0])),
                        ld_linear(self.mu, np.array([u])), rtol=1e-12)
        assert_allclose(ldi_square_root(np.array([u, 0.0])),
                        ldi_linear(np.array([u])), rtol=1e-12)

    def test_logarithmic(self):
        """Logarithmic law: I(μ) = 1 - a(1-μ) - b μ ln(μ)."""
        a, b = 0.5, 0.2
        assert_allclose(ld_logarithmic(self.mu, np.array([a, b])),
                        1.0 - a * (1.0 - self.mu) - b * self.mu * np.log(self.mu))

    def test_logarithmic_reduces_to_linear(self):
        """Logarithmic with b=0 should match linear."""
        u = 0.6
        assert_allclose(ld_logarithmic(self.mu, np.array([u, 0.0])),
                        ld_linear(self.mu, np.array([u])), rtol=1e-12)
        assert_allclose(ldi_logarithmic(np.array([u, 0.0])),
                        ldi_linear(np.array([u])), rtol=1e-12)

    def test_exponential(self):
        """Exponential law: I(μ) = 1 - a(1-μ) - b/(1-exp(μ))."""
        a, b = 0.5, 0.1
        assert_allclose(ld_exponential(self.mu, np.array([a, b])),
                        1.0 - a * (1.0 - self.mu) - b / (1.0 - np.exp(self.mu)))

    def test_power_2_pm_matches_power_2(self):
        """Maxted (2018) h1-h2 parametrization maps to the standard power-2 law."""
        c, alpha = 0.5, 0.7
        h1 = 1.0 - c * (1.0 - 2.0 ** -alpha)
        h2 = c * 2.0 ** -alpha
        assert_allclose(ld_power_2_pm(self.mu, np.array([h1, h2])),
                        ld_power_2(self.mu, np.array([c, alpha])), rtol=1e-12)
        assert_allclose(ldi_power_2_pm(np.array([h1, h2])),
                        ldi_power_2(np.array([c, alpha])), rtol=1e-12)

    def test_new_models_at_mu1(self):
        """New laws must be normalized to I(1) = 1."""
        mu = np.array([1.0])
        assert_allclose(ld_square_root(mu, np.array([0.4, 0.3])), 1.0)
        assert_allclose(ld_logarithmic(mu, np.array([0.5, 0.2])), 1.0)


# ============================================================
# Evaluation helper tests
# ============================================================

class TestEvaluationHelpers:
    """evaluate_ld and evaluate_ldi must handle 1D, 2D, and 3D parameter arrays."""

    mu = np.linspace(0.01, 1.0, 50)
    pv = np.array([0.4, 0.2])

    def test_evaluate_ld_1d(self):
        ldp = evaluate_ld(ld_quadratic, self.mu, self.pv)
        assert ldp.shape == (1, 1, self.mu.size)
        assert_allclose(ldp[0, 0], ld_quadratic(self.mu, self.pv))

    def test_evaluate_ld_2d(self):
        pv = np.tile(self.pv, (3, 1))
        ldp = evaluate_ld(ld_quadratic, self.mu, pv)
        assert ldp.shape == (1, 3, self.mu.size)
        for ipb in range(3):
            assert_allclose(ldp[0, ipb], ld_quadratic(self.mu, self.pv))

    def test_evaluate_ld_3d(self):
        pv = np.tile(self.pv, (2, 3, 1))
        ldp = evaluate_ld(ld_quadratic, self.mu, pv)
        assert ldp.shape == (2, 3, self.mu.size)

    def test_evaluate_ldi(self):
        istar = evaluate_ldi(ldi_quadratic, self.pv)
        assert istar.shape == (1, 1)
        assert_allclose(istar[0, 0], ldi_quadratic(self.pv))


# ============================================================
# RoadRunner integration tests
# ============================================================

class TestRoadRunnerIntegration:
    """The RoadRunner model must evaluate with the named limb darkening models."""

    @pytest.mark.parametrize("ldmodel,ldc", [
        ('square_root', [0.4, 0.3]),
        ('logarithmic', [0.5, 0.2]),
        ('power-2-pm', [0.8, 0.3]),
    ])
    def test_rrmodel_evaluates(self, ldmodel, ldc):
        from pytransit import RoadRunnerModel
        tm = RoadRunnerModel(ldmodel)
        # The transit lasts from about -0.07 to 0.07 days, so the first and
        # last samples are out of transit and the center is in transit.
        time = np.linspace(-0.15, 0.15, 100)
        tm.set_data(time)
        flux = np.atleast_1d(tm.evaluate(0.1, np.array(ldc), 0.0, 2.0, 5.0, 0.5 * np.pi))
        assert np.all(np.isfinite(flux))
        assert flux.min() < 0.99
        assert_allclose(flux[[0, -1]], 1.0)

    def test_rrmodel_batch_evaluation_matches_scalar(self):
        """Batch evaluation with 2D ldc (the LPF convention, one row per pv set)
        must match per-row scalar evaluation."""
        from pytransit import RoadRunnerModel
        time = np.linspace(-0.05, 0.05, 50)
        ldc = np.array([[0.40, 0.20], [0.55, 0.10], [0.30, 0.25]])
        k = np.array([[0.10], [0.11], [0.12]])
        t0 = np.zeros(3)
        p = np.full(3, 2.0)
        a = np.full(3, 5.0)
        i = np.full(3, 0.5 * np.pi)

        tm = RoadRunnerModel('quadratic')
        tm.set_data(time)
        fbatch = tm.evaluate(k, ldc, t0, p, a, i)
        assert fbatch.shape == (3, time.size)

        for j in range(3):
            fscalar = tm.evaluate(k[j, 0], ldc[j], t0[j], p[j], a[j], i[j])
            assert_allclose(fbatch[j], fscalar, rtol=1e-12,
                            err_msg=f'batch pv set {j} does not match scalar evaluation')

    def test_rrmodel_analytic_istar_matches_numerical(self):
        """Transit depths with the analytic disk integral must match the trapezoid fallback."""
        from pytransit import RoadRunnerModel
        pv = np.array([0.4, 0.3])
        time = np.linspace(-0.05, 0.05, 100)

        tm_analytic = RoadRunnerModel('square_root')
        tm_analytic.set_data(time)
        f_analytic = tm_analytic.evaluate(0.1, pv, 0.0, 2.0, 5.0, 0.5 * np.pi)

        tm_numeric = RoadRunnerModel(ld_square_root)
        tm_numeric.set_data(time)
        f_numeric = tm_numeric.evaluate(0.1, pv, 0.0, 2.0, 5.0, 0.5 * np.pi)

        assert_allclose(f_analytic, f_numeric, rtol=1e-6)
