"""Verification tests for limb darkening functions.

Tests the analytical disk-integrated fluxes (ldi_*) against numerical integration
of the intensity profiles (ld_*), and the analytical gradients (ldd_*) against
finite differences.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

from pytransit.backends.numba.limb_darkening.uniform import ld_uniform, ldi_uniform
from pytransit.backends.numba.limb_darkening.linear import ld_linear, ldi_linear, ldd_linear
from pytransit.backends.numba.limb_darkening.quadratic import ld_quadratic, ldi_quadratic, ldd_quadratic
from pytransit.backends.numba.limb_darkening.quadratic_tri import ld_quadratic_tri, ldi_quadratic_tri, ldd_quadratic_tri
from pytransit.backends.numba.limb_darkening.power_2 import ld_power_2, ldi_power_2, ldd_power_2
from pytransit.backends.numba.limb_darkening.nonlinear import ld_nonlinear, ldi_nonlinear, ldd_nonlinear
from pytransit.backends.numba.limb_darkening.general import ld_general, ldi_general, ldd_general


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
        np.array([0.1, 0.2, 0.1, 0.05]),
        np.array([0.3, 0.2, 0.15, 0.1]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.3, 0.1, 0.05]),
    ])
    def test_nonlinear(self, pv):
        assert_allclose(ldi_nonlinear(pv), numerical_ldi(ld_nonlinear, pv), rtol=1e-10)

    def test_nonlinear_zeros_gives_pi(self):
        """All zeros reduces to uniform disk, integral = π."""
        assert_allclose(ldi_nonlinear(np.array([0.0, 0.0, 0.0, 0.0])), np.pi, rtol=1e-12)

    @pytest.mark.parametrize("pv", [
        np.array([0.5, 0.3]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.2, 0.3, 0.1]),
    ])
    def test_general(self, pv):
        assert_allclose(ldi_general(pv), numerical_ldi(ld_general, pv), rtol=1e-10)

    def test_general_single_coeff(self):
        """Single coeff c₀: I(μ) = c₀(1-μ), integral = 2π·c₀·1/(2·3) = πc₀/3."""
        pv = np.array([1.0])
        assert_allclose(ldi_general(pv), np.pi / 3.0, rtol=1e-12)


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
        np.array([0.5, 0.3]),
        np.array([0.8, 0.5]),
    ])
    def test_quadratic_tri(self, pv):
        ana = ldd_quadratic_tri(self.mu_test, pv)
        num = numerical_gradient(ld_quadratic_tri, self.mu_test, pv)
        assert_allclose(ana, num, atol=1e-5)

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

    @pytest.mark.parametrize("pv", [
        np.array([0.1, 0.2, 0.1, 0.05]),
        np.array([0.3, 0.2, 0.15, 0.1]),
    ])
    def test_nonlinear(self, pv):
        # Avoid mu near 0 where c₁/(2√μ) diverges
        mu = self.mu_test[self.mu_test > 0.05]
        ana = ldd_nonlinear(mu, pv)
        num = numerical_gradient(ld_nonlinear, mu, pv)
        assert_allclose(ana, num, atol=1e-5)

    @pytest.mark.parametrize("pv", [
        np.array([0.5, 0.3]),
        np.array([0.2, 0.3, 0.1]),
    ])
    def test_general(self, pv):
        ana = ldd_general(self.mu_test, pv)
        num = numerical_gradient(ld_general, self.mu_test, pv)
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
        """General law at μ=1: I(1) = Σ cᵢ(1-1) = 0."""
        mu = np.array([1.0])
        assert_allclose(ld_general(mu, np.array([0.5, 0.3])), 0.0)

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
