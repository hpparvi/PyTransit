"""Tests for the JAX uniform-disk transit model.

Verifies:
1. Analytical correctness (mid-transit depth = k^2 for full containment)
2. Output shapes for (npt,) layout
3. jax.jit compatibility
4. Custom JVP derivatives match finite differences
5. Edge cases: grazing transit, full containment, no transit
6. Flux matches Numba backend
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from pytransit.backends.jax.udmodel import udmodel as jax_udmodel, udmodel_grad as jax_udmodel_grad

# Shared orbital parameters for a typical hot Jupiter transit
T0 = 0.0
P = 2.5
A = 8.0
I = np.radians(87.0)
E = 0.0
W = 0.5 * np.pi
TIMES = np.linspace(-0.15, 0.15, 300)


def _make_1d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npb=1, nep=1):
    """Build 1D parameter arrays expected by the JAX udmodel."""
    k1 = jnp.full((npb,), k)
    t01 = jnp.full((nep,), t0)
    p1 = jnp.full((nep,), p)
    a1 = jnp.full((nep,), a)
    i1 = jnp.full((nep,), inc)
    e1 = jnp.full((nep,), e)
    w1 = jnp.full((nep,), w)
    return dict(k=k1, t0=t01, p=p1, a=a1, i=i1, e=e1, w=w1)


def _call_udmodel(times, k, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Convenience wrapper calling the JAX udmodel with default single-LC setup."""
    npt = times.size
    params = _make_1d_params(k, t0, p, a, inc, e, w)
    lcids = jnp.zeros(npt, dtype=jnp.int32)
    pbids = jnp.zeros(1, dtype=jnp.int32)
    epids = jnp.zeros(1, dtype=jnp.int32)
    nsamples = jnp.ones(1, dtype=jnp.int32)
    exptimes = jnp.zeros(1, dtype=jnp.float64)
    return jax_udmodel(times, params['k'], params['t0'], params['p'], params['a'],
                       params['i'], params['e'], params['w'],
                       lcids, pbids, epids, nsamples, exptimes,
                       1, 1)


def _call_udmodel_grad(times, k, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Convenience wrapper for the gradient variant."""
    npt = times.size
    params = _make_1d_params(k, t0, p, a, inc, e, w)
    lcids = jnp.zeros(npt, dtype=jnp.int32)
    pbids = jnp.zeros(1, dtype=jnp.int32)
    epids = jnp.zeros(1, dtype=jnp.int32)
    nsamples = jnp.ones(1, dtype=jnp.int32)
    exptimes = jnp.zeros(1, dtype=jnp.float64)
    return jax_udmodel_grad(times, params['k'], params['t0'], params['p'], params['a'],
                            params['i'], params['e'], params['w'],
                            lcids, pbids, epids, nsamples, exptimes,
                            1, 1)


class TestOutputShapes:
    """Verify output array shapes."""

    def test_forward_shape(self):
        """Forward output should have shape (npt,)."""
        flux = _call_udmodel(TIMES, k=0.1)
        assert flux.shape == (TIMES.size,)

    def test_grad_shape(self):
        """Gradient output should have shapes (npt,) and (npt, 7)."""
        flux, dflux = _call_udmodel_grad(TIMES, k=0.1)
        assert flux.shape == (TIMES.size,)
        assert dflux.shape == (TIMES.size, 7)


class TestAnalytical:
    """Check against known analytical results."""

    def test_mid_transit_depth(self):
        """At mid-transit with b~0, flux deviation should be -k^2."""
        k = 0.1
        inc = np.radians(89.99)
        times = jnp.array([0.0])
        flux = np.asarray(_call_udmodel(times, k=k, inc=inc))
        np.testing.assert_allclose(flux[0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.05, 0.1, 0.15])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux deviation should be zero."""
        times = jnp.array([0.5, 1.0, -0.5])
        flux = np.asarray(_call_udmodel(times, k=k))
        np.testing.assert_allclose(flux, 0.0, atol=1e-12)

    def test_transit_is_symmetric(self):
        """Transit light curve should be symmetric about t0 for circular orbit."""
        k = 0.1
        times = jnp.linspace(-0.1, 0.1, 201)
        flux = np.asarray(_call_udmodel(times, k=k))
        np.testing.assert_allclose(flux, flux[::-1], atol=5e-7)

    def test_depth_scales_with_k_squared(self):
        """Mid-transit depth ratio for k=0.1 vs k=0.01 should be ~100."""
        inc = np.radians(89.99)
        times = jnp.array([0.0])
        flux_small = np.asarray(_call_udmodel(times, k=0.01, inc=inc))
        flux_large = np.asarray(_call_udmodel(times, k=0.1, inc=inc))
        ratio = flux_large[0] / flux_small[0]
        np.testing.assert_allclose(ratio, 100.0, rtol=1e-3)


class TestJIT:
    """Model should work under jax.jit."""

    def test_jit_produces_correct_results(self):
        k = 0.1
        times = jnp.array(TIMES)
        flux_eager = np.asarray(_call_udmodel(times, k=k))
        # JIT via the wrapper — udmodel is already jittable
        jitted = jax.jit(jax_udmodel, static_argnums=(13, 14))
        params = _make_1d_params(k)
        npt = times.size
        lcids = jnp.zeros(npt, dtype=jnp.int32)
        pbids = jnp.zeros(1, dtype=jnp.int32)
        epids = jnp.zeros(1, dtype=jnp.int32)
        nsamples = jnp.ones(1, dtype=jnp.int32)
        exptimes = jnp.zeros(1, dtype=jnp.float64)
        flux_jit = np.asarray(jitted(times, params['k'], params['t0'], params['p'], params['a'],
                                     params['i'], params['e'], params['w'],
                                     lcids, pbids, epids, nsamples, exptimes,
                                     1, 1))
        np.testing.assert_allclose(flux_jit, flux_eager, atol=1e-6)


class TestEdgeCases:
    """Edge cases: grazing transit, full containment, no overlap."""

    def test_no_transit(self):
        """Large impact parameter — planet never overlaps the star."""
        inc_no_transit = np.radians(60.0)
        flux = np.asarray(_call_udmodel(jnp.array(TIMES), k=0.1, inc=inc_no_transit))
        np.testing.assert_allclose(flux, 0.0, atol=1e-12)

    def test_grazing_transit(self):
        """Planet barely grazes the stellar disk."""
        k = 0.1
        b_grazing = 0.95
        inc_graze = np.arccos(b_grazing / A)
        flux = np.asarray(_call_udmodel(jnp.array(TIMES), k=k, inc=inc_graze))
        assert np.min(flux) < 0.0
        full_flux = np.asarray(_call_udmodel(jnp.array(TIMES), k=k))
        assert np.min(flux) > np.min(full_flux)

    def test_full_containment(self):
        """Planet fully inside the stellar disk at mid-transit."""
        k = 0.1
        flux = np.asarray(_call_udmodel(jnp.array(TIMES), k=k))
        mid_idx = len(TIMES) // 2
        np.testing.assert_allclose(flux[mid_idx], -k**2, atol=1e-6)

    def test_eccentric_orbit(self):
        """Eccentric orbit should still produce valid transit."""
        k, e = 0.1, 0.3
        flux = np.asarray(_call_udmodel(jnp.array(TIMES), k=k, e=e))
        assert np.min(flux) < 0.0
        assert flux[0] == 0.0 or abs(flux[0]) < 1e-10


class TestGradients:
    """Direct gradient output should be finite and correct."""

    def test_grad_finite(self):
        """All gradient elements should be finite."""
        _, dflux = _call_udmodel_grad(TIMES, k=0.1)
        assert jnp.all(jnp.isfinite(dflux))

    def test_grad_k_nonzero_in_transit(self):
        """Gradient w.r.t. k should be nonzero during transit."""
        times = jnp.array([0.0])
        _, dflux = _call_udmodel_grad(times, k=0.1, inc=np.radians(89.99))
        assert dflux[0, 0] != 0.0  # dk


class TestCustomJVP:
    """Validate custom JVP derivatives against finite differences."""

    PARAM_NAMES = ["k", "t0", "p", "a", "i", "e", "w"]

    @staticmethod
    def _finite_diff_grad(times, k, t0, p, a, inc, e, w, idx, eps=1e-6):
        """Central finite-difference gradient w.r.t. parameter idx."""
        params = [k, t0, p, a, inc, e, w]
        p_plus = list(params)
        p_minus = list(params)
        p_plus[idx] = params[idx] + eps
        p_minus[idx] = params[idx] - eps
        f_plus = _call_udmodel(times, *p_plus)
        f_minus = _call_udmodel(times, *p_minus)
        return (f_plus - f_minus) / (2 * eps)

    @pytest.mark.parametrize("idx", range(7))
    def test_circular_orbit_jacobian(self, idx):
        """Custom JVP matches finite differences for circular orbit."""
        times = jnp.linspace(-0.08, 0.08, 50)
        params = [0.1, T0, P, A, I, 0.0, W]

        jac_fd = np.asarray(self._finite_diff_grad(times, *params, idx=idx))

        # Get analytic Jacobian from udmodel_grad
        _, dflux = _call_udmodel_grad(times, *params)
        jac_analytic = np.asarray(dflux[:, idx])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-4, rtol=1e-3,
            err_msg=f"Jacobian mismatch for {self.PARAM_NAMES[idx]} (circular)"
        )

    @pytest.mark.parametrize("idx", range(7))
    def test_eccentric_orbit_jacobian(self, idx):
        """Custom JVP matches finite differences for eccentric orbit."""
        times = jnp.linspace(-0.08, 0.08, 50)
        params = [0.1, T0, P, A, I, 0.3, W]

        jac_fd = np.asarray(self._finite_diff_grad(times, *params, idx=idx))
        _, dflux = _call_udmodel_grad(times, *params)
        jac_analytic = np.asarray(dflux[:, idx])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-4, rtol=1e-3,
            err_msg=f"Jacobian mismatch for {self.PARAM_NAMES[idx]} (eccentric)"
        )

    def test_all_gradients_finite_and_nonzero(self):
        """All 7 partial derivatives should be finite; most should be nonzero."""
        times = jnp.linspace(-0.08, 0.08, 50)
        _, dflux = _call_udmodel_grad(times, k=0.1, inc=np.radians(89.99))

        assert jnp.all(jnp.isfinite(dflux))
        # k, a, i definitely affect transit depth — sum of absolute gradients > 0
        for j in [0, 3, 4]:
            assert jnp.sum(jnp.abs(dflux[:, j])) > 0.0, \
                f"Zero grad for {self.PARAM_NAMES[j]}"


class TestFluxMatchesNumba:
    """JAX and Numba backends should produce identical results."""

    def test_flux_matches_numba(self):
        """Compare JAX and Numba forward models."""
        from numba import njit
        from pytransit.backends.numba.udmodel import udmodel as nb_udmodel
        nb_jit = njit(nb_udmodel)

        k, inc = 0.1, np.radians(89.99)
        times = np.linspace(-0.1, 0.1, 200)

        # Numba call (still uses 2D params with npv dimension)
        k_nb = np.full((1, 1), k)
        t0_nb = np.full((1, 1), T0)
        p_nb = np.full((1, 1), P)
        a_nb = np.full((1, 1), A)
        i_nb = np.full((1, 1), inc)
        e_nb = np.full((1, 1), E)
        w_nb = np.full((1, 1), W)
        npt = times.size
        lcids = np.zeros(npt, dtype=np.int32)
        pbids = np.zeros(1, dtype=np.int32)
        epids = np.zeros(1, dtype=np.int32)
        nsamples = np.ones(1, dtype=np.int32)
        exptimes = np.zeros(1, dtype=np.float64)
        flux_nb = nb_jit(times, k_nb, t0_nb, p_nb, a_nb, i_nb, e_nb, w_nb,
                         lcids, pbids, epids, nsamples, exptimes, 1, 1, 1)

        # JAX call (1D params, no npv dimension)
        flux_jax = np.asarray(_call_udmodel(jnp.array(times), k=k, inc=inc))

        np.testing.assert_allclose(flux_jax, flux_nb[0], atol=1e-10)
