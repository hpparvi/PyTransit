"""Tests for the JAX uniform-disk transit model.

Verifies:
1. Analytical correctness (mid-transit depth = k^2 for full containment)
2. jax.jit compatibility
3. jax.value_and_grad produces finite gradients
4. Edge cases: grazing transit, full containment, no transit
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

# Skip the whole module if the JAX backend isn't available yet (work in progress).
jax_uniform = pytest.importorskip("pytransit.backends.jax.uniformdisk").uniform_model

# Shared orbital parameters for a typical hot Jupiter transit
T0 = 0.0
P = 2.5
A = 8.0
I = np.radians(87.0)
E = 0.0
W = 0.5 * np.pi
TIMES = np.linspace(-0.15, 0.15, 300)


class TestAnalytical:
    """Check against known analytical results."""

    def test_mid_transit_depth(self):
        """At mid-transit with b~0, flux deviation should be -k^2."""
        k = 0.1
        # Use near-zero impact parameter (i ~ 90 deg)
        inc = np.radians(89.99)
        times = jnp.array([0.0])
        flux = np.asarray(jax_uniform(times, k, T0, P, A, inc, E, W))
        np.testing.assert_allclose(flux[0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.05, 0.1, 0.15])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux deviation should be zero."""
        times = jnp.array([0.5, 1.0, -0.5])
        flux = np.asarray(jax_uniform(times, k, T0, P, A, I, E, W))
        np.testing.assert_allclose(flux, 0.0, atol=1e-12)

    def test_transit_is_symmetric(self):
        """Transit light curve should be symmetric about t0 for circular orbit."""
        k = 0.1
        times = jnp.linspace(-0.1, 0.1, 201)
        flux = np.asarray(jax_uniform(times, k, T0, P, A, I, E, W))
        np.testing.assert_allclose(flux, flux[::-1], atol=5e-7)


class TestJIT:
    """Model should work under jax.jit."""

    def test_jit_produces_correct_results(self):
        k = 0.1
        times = jnp.array(TIMES)
        flux_eager = np.asarray(jax_uniform(times, k, T0, P, A, I, E, W))
        flux_jit = np.asarray(jax.jit(jax_uniform)(times, k, T0, P, A, I, E, W))
        np.testing.assert_allclose(flux_jit, flux_eager, atol=1e-6)


class TestGradients:
    """jax.value_and_grad should return finite gradients."""

    def test_grad_wrt_k(self):
        def loss(k):
            return jnp.sum(jax_uniform(jnp.array(TIMES), k, T0, P, A, I, E, W))

        val, grad = jax.value_and_grad(loss)(0.1)
        assert jnp.isfinite(val)
        assert jnp.isfinite(grad)
        assert grad != 0.0  # k definitely affects the flux

    def test_grad_wrt_all_params(self):
        def loss(k, t0, p, a, inc, e, w):
            return jnp.sum(jax_uniform(jnp.array(TIMES), k, t0, p, a, inc, e, w))

        argnums = tuple(range(7))
        val, grads = jax.value_and_grad(loss, argnums=argnums)(0.1, T0, P, A, I, E, W)
        assert jnp.isfinite(val)
        for j, g in enumerate(grads):
            assert jnp.isfinite(g), f"Non-finite gradient for argnum {j}: {g}"


class TestEdgeCases:
    """Edge cases: grazing transit, full containment, no overlap."""

    def test_no_transit(self):
        """Large impact parameter — planet never overlaps the star."""
        inc_no_transit = np.radians(60.0)
        flux = np.asarray(jax_uniform(jnp.array(TIMES), 0.1, T0, P, A, inc_no_transit, E, W))
        np.testing.assert_allclose(flux, 0.0, atol=1e-12)

    def test_grazing_transit(self):
        """Planet barely grazes the stellar disk."""
        k = 0.1
        b_grazing = 0.95
        inc_graze = np.arccos(b_grazing / A)
        flux = np.asarray(jax_uniform(jnp.array(TIMES), k, T0, P, A, inc_graze, E, W))
        # Should have some negative flux values (transit dip)
        assert np.min(flux) < 0.0
        # But much shallower than full transit
        full_flux = np.asarray(jax_uniform(jnp.array(TIMES), k, T0, P, A, I, E, W))
        assert np.min(flux) > np.min(full_flux)

    def test_full_containment(self):
        """Planet fully inside the stellar disk at mid-transit."""
        k = 0.1
        flux = np.asarray(jax_uniform(jnp.array(TIMES), k, T0, P, A, I, E, W))
        # At mid-transit, flux deviation should be -k^2
        mid_idx = len(TIMES) // 2
        np.testing.assert_allclose(flux[mid_idx], -k**2, atol=1e-6)

    def test_eccentric_orbit(self):
        """Eccentric orbit should still produce valid transit."""
        k, e = 0.1, 0.3
        flux = np.asarray(jax_uniform(jnp.array(TIMES), k, T0, P, A, I, e, W))
        # Should have a transit dip
        assert np.min(flux) < 0.0
        # Out-of-transit should be zero
        assert flux[0] == 0.0 or abs(flux[0]) < 1e-10


class TestCustomJVP:
    """Validate custom JVP derivatives against finite differences."""

    PARAM_NAMES = ["k", "t0", "p", "a", "i", "e", "w"]

    @staticmethod
    def _finite_diff_grad(times, params, idx, eps=1e-6):
        """Central finite-difference gradient w.r.t. params[idx]."""
        p_plus = list(params)
        p_minus = list(params)
        p_plus[idx] = params[idx] + eps
        p_minus[idx] = params[idx] - eps
        f_plus = jax_uniform(times, *p_plus)
        f_minus = jax_uniform(times, *p_minus)
        return (f_plus - f_minus) / (2 * eps)

    @pytest.mark.parametrize("idx", range(7))
    def test_circular_orbit_jacobian(self, idx):
        """Custom JVP matches finite differences for circular orbit."""
        times = jnp.linspace(-0.08, 0.08, 50)
        params = [0.1, T0, P, A, I, 0.0, W]

        # JAX Jacobian via the custom JVP
        def model_wrapper(*args):
            return jax_uniform(times, *args)

        jac_jax = jax.jacobian(model_wrapper, argnums=idx)(*[jnp.float64(v) for v in params])
        jac_fd = self._finite_diff_grad(times, params, idx)

        np.testing.assert_allclose(
            np.asarray(jac_jax), np.asarray(jac_fd), atol=1e-4, rtol=1e-3,
            err_msg=f"Jacobian mismatch for {self.PARAM_NAMES[idx]} (circular)"
        )

    @pytest.mark.parametrize("idx", range(7))
    def test_eccentric_orbit_jacobian(self, idx):
        """Custom JVP matches finite differences for eccentric orbit."""
        times = jnp.linspace(-0.08, 0.08, 50)
        params = [0.1, T0, P, A, I, 0.3, W]

        def model_wrapper(*args):
            return jax_uniform(times, *args)

        jac_jax = jax.jacobian(model_wrapper, argnums=idx)(*[jnp.float64(v) for v in params])
        jac_fd = self._finite_diff_grad(times, params, idx)

        np.testing.assert_allclose(
            np.asarray(jac_jax), np.asarray(jac_fd), atol=1e-4, rtol=1e-3,
            err_msg=f"Jacobian mismatch for {self.PARAM_NAMES[idx]} (eccentric)"
        )

    def test_all_gradients_finite_and_nonzero(self):
        """All 7 partial derivatives should be finite; most should be nonzero."""
        times = jnp.linspace(-0.08, 0.08, 50)

        def loss(*args):
            return jnp.sum(jax_uniform(times, *args))

        params = [jnp.float64(v) for v in [0.1, T0, P, A, I, 0.0, W]]
        val, grads = jax.value_and_grad(loss, argnums=tuple(range(7)))(*params)

        assert jnp.isfinite(val)
        for j, g in enumerate(grads):
            assert jnp.isfinite(g), f"Non-finite grad for {self.PARAM_NAMES[j]}: {g}"

        # k, a, i definitely affect transit depth
        for j in [0, 3, 4]:
            assert grads[j] != 0.0, f"Zero grad for {self.PARAM_NAMES[j]}"
