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
                         lcids, pbids, epids, nsamples, exptimes, 1, 1)

        # JAX call (1D params, no npv dimension)
        flux_jax = np.asarray(_call_udmodel(jnp.array(times), k=k, inc=inc))

        np.testing.assert_allclose(flux_jax, flux_nb[0], atol=1e-10)


class TestValueAndGrad:
    """Verify jax.value_and_grad works through the custom JVP.

    Key pattern: call jax_udmodel directly (not via an inner jax.jit) inside the
    chi2 closure, then wrap the whole thing with jax.jit(jax.value_and_grad(...)).
    Nesting jax.jit inside value_and_grad breaks custom_jvp's nondiff_argnums
    because the outer jit turns nondiff array args into tracers.
    """

    def test_chi2_value_and_grad_single_param(self):
        """jax.value_and_grad of a chi2 function w.r.t. k should return finite results."""
        times = jnp.array(TIMES)
        npt = times.size
        lcids = jnp.zeros(npt, dtype=jnp.int32)
        pbids = jnp.zeros(1, dtype=jnp.int32)
        epids = jnp.zeros(1, dtype=jnp.int32)
        nsamples = jnp.ones(1, dtype=jnp.int32)
        exptimes = jnp.zeros(1, dtype=jnp.float64)

        # Synthetic observations at k=0.1
        k_true = jnp.array([0.1])
        flux_true = jax_udmodel(times, k_true,
                                jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                                jnp.array([I]), jnp.array([E]), jnp.array([W]),
                                lcids, pbids, epids, nsamples, exptimes, 1, 1)
        fobs = flux_true + 0.001 * jax.random.normal(jax.random.PRNGKey(42), flux_true.shape)

        def chi2(k):
            flux = jax_udmodel(times, k,
                               jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                               jnp.array([I]), jnp.array([E]), jnp.array([W]),
                               lcids, pbids, epids, nsamples, exptimes, 1, 1)
            return jnp.sum((fobs - flux) ** 2)

        jchi2 = jax.jit(jax.value_and_grad(chi2))
        value, grad = jchi2(jnp.array([0.1]))

        assert jnp.isfinite(value)
        assert value.shape == ()
        assert grad.shape == (1,)
        assert jnp.all(jnp.isfinite(grad))

    def test_chi2_grad_is_nonzero_away_from_truth(self):
        """Gradient should be nonzero when k differs from true value."""
        times = jnp.array(TIMES)
        npt = times.size
        lcids = jnp.zeros(npt, dtype=jnp.int32)
        pbids = jnp.zeros(1, dtype=jnp.int32)
        epids = jnp.zeros(1, dtype=jnp.int32)
        nsamples = jnp.ones(1, dtype=jnp.int32)
        exptimes = jnp.zeros(1, dtype=jnp.float64)

        # Noiseless observations at k=0.1
        k_true = jnp.array([0.1])
        fobs = jax_udmodel(times, k_true,
                           jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                           jnp.array([I]), jnp.array([E]), jnp.array([W]),
                           lcids, pbids, epids, nsamples, exptimes, 1, 1)

        def chi2(k):
            flux = jax_udmodel(times, k,
                               jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                               jnp.array([I]), jnp.array([E]), jnp.array([W]),
                               lcids, pbids, epids, nsamples, exptimes, 1, 1)
            return jnp.sum((fobs - flux) ** 2)

        jchi2 = jax.jit(jax.value_and_grad(chi2))

        # At the true k, chi2 should be ~0 and gradient ~0
        val_true, grad_true = jchi2(k_true)
        np.testing.assert_allclose(float(val_true), 0.0, atol=1e-20)
        np.testing.assert_allclose(float(grad_true[0]), 0.0, atol=1e-10)

        # At a wrong k, gradient should be nonzero
        val_wrong, grad_wrong = jchi2(jnp.array([0.15]))
        assert float(val_wrong) > 0.0
        assert float(jnp.abs(grad_wrong[0])) > 1e-6

    def test_chi2_grad_matches_finite_diff(self):
        """Gradient from value_and_grad should match finite differences."""
        times = jnp.array(TIMES)
        npt = times.size
        lcids = jnp.zeros(npt, dtype=jnp.int32)
        pbids = jnp.zeros(1, dtype=jnp.int32)
        epids = jnp.zeros(1, dtype=jnp.int32)
        nsamples = jnp.ones(1, dtype=jnp.int32)
        exptimes = jnp.zeros(1, dtype=jnp.float64)

        fobs = jax_udmodel(times, jnp.array([0.1]),
                           jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                           jnp.array([I]), jnp.array([E]), jnp.array([W]),
                           lcids, pbids, epids, nsamples, exptimes, 1, 1)

        def chi2(k):
            flux = jax_udmodel(times, k,
                               jnp.array([T0]), jnp.array([P]), jnp.array([A]),
                               jnp.array([I]), jnp.array([E]), jnp.array([W]),
                               lcids, pbids, epids, nsamples, exptimes, 1, 1)
            return jnp.sum((fobs - flux) ** 2)

        k_test = jnp.array([0.12])
        _, grad = jax.jit(jax.value_and_grad(chi2))(k_test)

        # Finite difference
        eps = 1e-6
        chi2_plus = chi2(k_test + eps)
        chi2_minus = chi2(k_test - eps)
        grad_fd = (chi2_plus - chi2_minus) / (2 * eps)

        np.testing.assert_allclose(float(grad[0]), float(grad_fd), rtol=1e-4)

    def test_multi_epoch_chi2(self):
        """value_and_grad should work with multiple epochs and supersampling."""
        times = jnp.linspace(-0.5, 0.5, 500)
        times_all = jnp.concatenate([times, times + P])
        npt = times_all.size
        lcids = jnp.concatenate([jnp.zeros(500, dtype=jnp.int32),
                                 jnp.ones(500, dtype=jnp.int32)])
        pbids = jnp.zeros(2, dtype=jnp.int32)
        epids = jnp.array([0, 1], dtype=jnp.int32)
        nsamples = jnp.array([1, 10], dtype=jnp.int32)
        exptimes = jnp.array([0.0, 0.05])
        nep = 2

        fobs = jax_udmodel(times_all, jnp.array([0.1]),
                           jnp.array([T0, T0]), jnp.array([P, P]),
                           jnp.array([A, A]), jnp.array([I, I]),
                           jnp.array([E, E]), jnp.array([W, W]),
                           lcids, pbids, epids, nsamples, exptimes, 1, nep)

        def chi2(k):
            flux = jax_udmodel(times_all, k,
                               jnp.array([T0, T0]), jnp.array([P, P]),
                               jnp.array([A, A]), jnp.array([I, I]),
                               jnp.array([E, E]), jnp.array([W, W]),
                               lcids, pbids, epids, nsamples, exptimes, 1, nep)
            return jnp.sum((fobs - flux) ** 2)

        jchi2 = jax.jit(jax.value_and_grad(chi2))
        value, grad = jchi2(jnp.array([0.12]))

        assert jnp.isfinite(value)
        assert float(value) > 0.0
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.abs(grad[0])) > 1e-6


class TestJacobianMatchesNumba:
    """The JAX Jacobian against the analytic Numba one, through the model class."""

    @staticmethod
    def _models(times, lcids=None, epids=None, nsamples=1, exptimes=0.0, **kwargs):
        from pytransit import UniformDiskModel
        models = []
        for backend in ('numba', 'jax'):
            m = UniformDiskModel(backend=backend, return_grad=True)
            m.set_data(times, lcids, epids=epids, nsamples=nsamples, exptimes=exptimes, **kwargs)
            models.append(m)
        return models

    def _compare(self, models, *pars):
        (flux_nb, dflux_nb), (flux_jax, dflux_jax) = [m.evaluate(*pars) for m in models]
        assert np.any(flux_nb < 0.0)
        np.testing.assert_allclose(np.asarray(flux_jax), flux_nb, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(np.asarray(dflux_jax), dflux_nb, rtol=1e-7, atol=1e-9)

    def test_eccentric_orbit(self):
        times = np.linspace(-0.12, 0.12, 151)
        self._compare(self._models(times), 0.1, 0.01, P, A, I, 0.2, 0.4 * np.pi)

    def test_supersampling(self):
        times = np.linspace(-0.12, 0.12, 60)
        self._compare(self._models(times, nsamples=5, exptimes=0.02), 0.1, 0.01, P, A, I, 0.2, 0.4 * np.pi)

    def test_later_epochs_fold_the_period(self):
        """Away from epoch 0 the period derivative includes the folding term."""
        times = 7 * P + np.linspace(-0.12, 0.12, 151)
        models = self._models(times)
        self._compare(models, 0.1, 0.01, P, A, I, 0.2, 0.4 * np.pi)
        _, dflux = models[1].evaluate(0.1, 0.01, P, A, I, 0.2, 0.4 * np.pi)
        assert np.abs(np.asarray(dflux)[:, 2]).max() > 1.0

    def test_transit_centres_with_a_shared_orbit(self):
        """One orbit and a transit centre per epoch: each epoch must use its own centre."""
        t = np.linspace(-0.12, 0.12, 101)
        times = np.concatenate([t, P + t])
        lcids = np.concatenate([np.zeros(t.size, 'int'), np.full(t.size, 1, 'int')])
        models = self._models(times, lcids, epids=np.array([0, 1]), include_orbit_variations=False)
        self._compare(models, 0.1, np.array([0.0, P + 0.03]), P, A, I, 0.2, 0.4 * np.pi)


class TestReverseModeWithSupersampling:
    def test_grad_matches_finite_diff(self):
        """jax.grad must work although the supersampling loop has a traced trip count."""
        times = jnp.linspace(-0.1, 0.1, 40)
        lcids = jnp.zeros(times.size, dtype=jnp.int32)
        ids = jnp.zeros(1, dtype=jnp.int32)
        nsamples = jnp.full(1, 4, dtype=jnp.int32)
        exptimes = jnp.full(1, 0.02)

        def total(k, a):
            pv = _make_1d_params(k, a=a)
            return jnp.sum(jax_udmodel(times, pv['k'], pv['t0'], pv['p'], pv['a'], pv['i'], pv['e'], pv['w'],
                                       lcids, ids, ids, nsamples, exptimes, 1, 1) ** 2)

        g = jax.grad(total, argnums=(0, 1))(0.1, A)
        eps = 1e-6
        fd_k = (total(0.1 + eps, A) - total(0.1 - eps, A)) / (2 * eps)
        fd_a = (total(0.1, A + eps) - total(0.1, A - eps)) / (2 * eps)
        np.testing.assert_allclose([g[0], g[1]], [fd_k, fd_a], rtol=1e-5)
