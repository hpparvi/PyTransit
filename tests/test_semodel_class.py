"""Tests for the public SecondaryEclipseModel class wrapper.

Verifies that the wrapper sets up the data, normalises parameter shapes, threads
the ``rstar`` argument through to the Numba backend, and returns flux (and
gradients) consistent with the backend functions.
"""

import numpy as np
import pytest
from math import radians, pi

from numba import njit
from meepmeep.backends.numba.utils import eclipse_time_offset
from pytransit.models.semodel import SecondaryEclipseModel
from pytransit.backends.numba.semodel import semodel as nb_semodel

# Shared orbital parameters
T0 = 0.0
P = 2.5
A = 8.0
I = radians(89.0)
E = 0.0
W = pi / 2
RSTAR = 1.0
K = 0.1
CENTER = T0 + eclipse_time_offset(P, I, E, W)
TIMES = np.linspace(CENTER - 0.15, CENTER + 0.15, 400)


def _raw_backend(times, k=K, t0=T0, p=P, a=A, i=I, e=E, w=W, rstar=RSTAR):
    """Call the Numba backend directly with a single-LC setup."""
    nbjit = njit(nb_semodel)
    mk = lambda v: np.full((1, 1), v)
    return nbjit(times, mk(k), mk(t0), mk(p), mk(a), mk(i), mk(e), mk(w), rstar,
                 np.zeros(times.size, np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32),
                 np.ones(1, np.int32), np.zeros(1), 1, 1)[0]


class TestConstruction:
    def test_numba_default(self):
        m = SecondaryEclipseModel()
        assert m.backend == 'numba'
        assert m.return_grad is False

    def test_jax_backend_rejected(self):
        with pytest.raises(ValueError):
            SecondaryEclipseModel(backend='jax')


class TestForward:
    def test_shape_and_baseline(self):
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = m.evaluate(K, T0, P, A, I, E, W, rstar=RSTAR)
        assert flux.shape == (TIMES.size,)
        np.testing.assert_allclose(flux.max(), pi * K**2, rtol=1e-12)
        np.testing.assert_allclose(flux.min(), 0.0, atol=1e-9)

    def test_matches_raw_backend(self):
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = m.evaluate(K, T0, P, A, I, E, W, rstar=RSTAR)
        np.testing.assert_allclose(flux, _raw_backend(TIMES), atol=1e-14)

    def test_default_circular_orbit(self):
        """e and w default to a circular orbit when omitted."""
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = m.evaluate(K, T0, P, A, I)
        np.testing.assert_allclose(flux, _raw_backend(TIMES, e=0.0, w=0.0), atol=1e-14)

    def test_rstar_shifts_eclipse(self):
        """A larger stellar radius increases the light-travel delay."""
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        f_small = m.evaluate(K, T0, P, A, I, E, W, rstar=0.1)
        f_large = m.evaluate(K, T0, P, A, I, E, W, rstar=5.0)
        assert np.max(np.abs(f_small - f_large)) > 0.0

    def test_two_dimensional_k(self):
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = m.evaluate(np.array([[K]]), T0, P, A, I, E, W, rstar=RSTAR)
        np.testing.assert_allclose(flux, _raw_backend(TIMES), atol=1e-14)


class TestGradient:
    def test_shapes_and_flux_consistency(self):
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = m.evaluate(K, T0, P, A, I, E, W, rstar=RSTAR)

        mg = SecondaryEclipseModel(return_grad=True)
        mg.set_data(TIMES)
        f, df = mg.evaluate(K, T0, P, A, I, E, W, rstar=RSTAR)
        assert f.shape == (TIMES.size,)
        assert df.shape == (TIMES.size, 7)
        np.testing.assert_allclose(f, flux, atol=1e-14)

    def test_k_gradient_finite_difference(self):
        """The k-gradient matches central finite differences on partial-overlap points."""
        mg = SecondaryEclipseModel(return_grad=True)
        mg.set_data(TIMES)
        f, df = mg.evaluate(K, T0, P, A, I, E, W, rstar=RSTAR)

        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        eps = 1e-7
        fp = m.evaluate(K + eps, T0, P, A, I, E, W, rstar=RSTAR)
        fm = m.evaluate(K - eps, T0, P, A, I, E, W, rstar=RSTAR)
        fd = (fp - fm) / (2 * eps)

        baseline = pi * K**2
        mask = (f > 1e-6) & (f < baseline - 1e-6)
        assert mask.any()
        np.testing.assert_allclose(df[mask, 0], fd[mask], rtol=1e-3, atol=1e-8)
