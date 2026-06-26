"""Gradient-layout tests for UniformDiskModel and SecondaryEclipseModel.

Like the RoadRunnerModel, these models must return a Jacobian whose columns
match the parameters as passed to ``evaluate()`` (order ``[k, t0, p, a, i, e, w]``;
no LD coefficients here). A passband-dependent ``k`` expands to one column per
passband and an epoch-dependent ``t0`` / orbital parameter expands to one column
per epoch, with each data point's derivative landing only in its own column.
"""

import unittest

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from numpy import array, linspace, pi, concatenate, zeros, full
from math import radians

from pytransit.models.udmodel import UniformDiskModel
from pytransit.models.semodel import SecondaryEclipseModel


# --------------------------------------------------------------------------- #
# UniformDiskModel — Numba backend
# --------------------------------------------------------------------------- #
class TestUDNumbaPerPassband(unittest.TestCase):
    """Two passbands, single epoch; k passed per passband."""

    def setUp(self):
        self.npb, self.ntc, self.nor = 2, 1, 1
        t = linspace(-0.12, 0.12, 150)
        self.times = concatenate([t, t])
        self.lcids = concatenate([zeros(t.size, 'int'), full(t.size, 1, 'int')])
        self.pbids = array([0, 1])
        self.model = UniformDiskModel(backend='numba', return_grad=True)
        self.model.set_data(self.times, self.lcids, self.pbids)
        self.pb_of_point = self.pbids[self.lcids]

    def _eval(self, k):
        return self.model.evaluate(k, 0.0, 2.5, 8.0, radians(88.0), 0.0, pi / 2)

    def test_k_expands_per_passband(self):
        k = array([0.10, 0.12])
        eps = 1e-6
        flux0, dflux0 = self._eval(k)
        # [k0 k1 | t0 | p a i e w] = 8 columns
        self.assertEqual(dflux0.shape, (self.times.size, 8))
        for ipb in range(self.npb):
            kp = k.copy(); kp[ipb] += eps
            km = k.copy(); km[ipb] -= eps
            fd = (self._eval(kp)[0] - self._eval(km)[0]) / (2 * eps)
            anal = dflux0[:, ipb]
            in_transit = flux0 < -1e-6
            same = (self.pb_of_point == ipb) & in_transit
            self.assertTrue(same.any())
            np.testing.assert_allclose(anal[same], fd[same], rtol=1e-3, atol=1e-8)
            np.testing.assert_allclose(anal[self.pb_of_point != ipb], 0.0, atol=1e-12)


class TestUDNumbaPerEpoch(unittest.TestCase):
    """Two epochs with orbit variations; t0 and i passed per epoch."""

    backend = 'numba'

    def setUp(self):
        p = 2.5
        t = linspace(-0.12, 0.12, 150)
        self.times = concatenate([t, p + t])
        self.lcids = concatenate([zeros(t.size, 'int'), full(t.size, 1, 'int')])
        self.epids = array([0, 1])
        self.model = UniformDiskModel(backend=self.backend, return_grad=True)
        self.model.set_data(self.times, self.lcids, epids=self.epids,
                            include_orbit_variations=True)
        # x = [k, t0_0, t0_1, p, a, i0, i1, e, w]
        self.x = array([0.10, 0.0, 2.5, 2.5, 8.0,
                        radians(88.0), radians(87.5), 0.05, pi / 2])
        self.ep_of_point = self.epids[self.lcids]

    def _eval(self, x):
        return self.model.evaluate(x[0], x[[1, 2]], x[3], x[4], x[[5, 6]], x[7], x[8])

    def test_width_matches_input(self):
        _, dflux = self._eval(self.x)
        # [k | t0_0 t0_1 | p | a | i0 i1 | e | w] = 9 columns
        self.assertEqual(dflux.shape, (self.times.size, 9))

    def test_all_columns_vs_finite_differences(self):
        eps = 1e-6
        flux0, dflux0 = self._eval(self.x)
        in_transit = np.asarray(flux0) < -1e-6
        self.assertTrue(in_transit.any())
        for c in range(self.x.size):
            xp = self.x.copy(); xp[c] += eps
            xm = self.x.copy(); xm[c] -= eps
            fd = np.asarray(self._eval(xp)[0]) - np.asarray(self._eval(xm)[0])
            fd = fd / (2 * eps)
            np.testing.assert_allclose(np.asarray(dflux0)[in_transit, c], fd[in_transit],
                                       rtol=2e-3, atol=1e-7, err_msg=f"column {c}")

    def test_t0_and_i_scattered_by_epoch(self):
        _, dflux0 = self._eval(self.x)
        dflux0 = np.asarray(dflux0)
        for ep, (t0_col, i_col) in enumerate([(1, 5), (2, 6)]):
            other = self.ep_of_point != ep
            np.testing.assert_allclose(dflux0[other, t0_col], 0.0, atol=1e-12)
            np.testing.assert_allclose(dflux0[other, i_col], 0.0, atol=1e-12)


class TestUDJaxPerEpoch(TestUDNumbaPerEpoch):
    """Same per-epoch checks for the JAX backend."""
    backend = 'jax'


class TestUDSharedOrbit(unittest.TestCase):
    """Two epochs, no orbit variations: t0 expands, orbital params shared."""

    def setUp(self):
        p = 2.5
        t = linspace(-0.12, 0.12, 150)
        self.times = concatenate([t, p + t])
        self.lcids = concatenate([zeros(t.size, 'int'), full(t.size, 1, 'int')])
        self.epids = array([0, 1])
        self.model = UniformDiskModel(backend='numba', return_grad=True)
        self.model.set_data(self.times, self.lcids, epids=self.epids,
                            include_orbit_variations=False)

    def test_layout(self):
        flux, dflux = self.model.evaluate(0.1, array([0.0, 2.5]), 2.5, 8.0,
                                          radians(88.0), 0.05, pi / 2)
        # [k | t0_0 t0_1 | p | a | i | e | w] = 8 columns
        self.assertEqual(dflux.shape, (self.times.size, 8))


class TestUDBackwardCompatible(unittest.TestCase):
    def test_width_seven(self):
        times = linspace(-0.12, 0.12, 150)
        m = UniformDiskModel(backend='numba', return_grad=True)
        m.set_data(times)
        _, dflux = m.evaluate(0.1, 0.0, 2.5, 8.0, radians(88.0), 0.05, pi / 2)
        self.assertEqual(dflux.shape, (times.size, 7))


# --------------------------------------------------------------------------- #
# SecondaryEclipseModel — Numba backend
# --------------------------------------------------------------------------- #
class TestSEPerEpoch(unittest.TestCase):
    """Two epochs with orbit variations; t0 and i passed per epoch."""

    def setUp(self):
        p = 2.5
        ec = 1.25                                   # ~secondary eclipse phase (e small, w=pi/2)
        t = linspace(-0.25, 0.25, 200)
        self.times = concatenate([ec + t, p + ec + t])
        self.lcids = concatenate([zeros(t.size, 'int'), full(t.size, 1, 'int')])
        self.epids = array([0, 1])
        self.model = SecondaryEclipseModel(backend='numba', return_grad=True)
        self.model.set_data(self.times, self.lcids, epids=self.epids,
                            include_orbit_variations=True)
        # x = [k, t0_0, t0_1, p, a, i0, i1, e, w]
        self.x = array([0.10, 0.0, 2.5, 2.5, 8.0,
                        radians(89.0), radians(88.8), 0.05, pi / 2])
        self.ep_of_point = self.epids[self.lcids]

    def _eval(self, x):
        return self.model.evaluate(x[0], x[[1, 2]], x[3], x[4], x[[5, 6]], x[7], x[8])

    def test_width_matches_input(self):
        _, dflux = self._eval(self.x)
        self.assertEqual(dflux.shape, (self.times.size, 9))

    def test_all_columns_vs_finite_differences(self):
        eps = 1e-6
        flux0, dflux0 = self._eval(self.x)
        baseline = pi * self.x[0] ** 2
        in_eclipse = flux0 < baseline - 1e-9
        self.assertTrue(in_eclipse.any())
        for c in range(self.x.size):
            xp = self.x.copy(); xp[c] += eps
            xm = self.x.copy(); xm[c] -= eps
            fd = (self._eval(xp)[0] - self._eval(xm)[0]) / (2 * eps)
            # k (col 0) varies the out-of-eclipse baseline too -> check everywhere;
            # all other columns are eclipse-shape derivatives -> check in eclipse.
            mask = np.ones_like(flux0, bool) if c == 0 else in_eclipse
            np.testing.assert_allclose(dflux0[mask, c], fd[mask],
                                       rtol=2e-3, atol=1e-7, err_msg=f"column {c}")

    def test_t0_and_i_scattered_by_epoch(self):
        _, dflux0 = self._eval(self.x)
        for ep, (t0_col, i_col) in enumerate([(1, 5), (2, 6)]):
            other = self.ep_of_point != ep
            np.testing.assert_allclose(dflux0[other, t0_col], 0.0, atol=1e-12)
            np.testing.assert_allclose(dflux0[other, i_col], 0.0, atol=1e-12)


class TestSEBackwardCompatible(unittest.TestCase):
    def test_width_seven(self):
        times = 1.25 + linspace(-0.25, 0.25, 200)
        m = SecondaryEclipseModel(backend='numba', return_grad=True)
        m.set_data(times)
        _, dflux = m.evaluate(0.1, 0.0, 2.5, 8.0, radians(89.0), 0.05, pi / 2)
        self.assertEqual(dflux.shape, (times.size, 7))


if __name__ == '__main__':
    unittest.main()
