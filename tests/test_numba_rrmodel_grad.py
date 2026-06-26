"""Tests for the RoadRunnerModel Numba gradients.

The returned Jacobian is laid out to *match the parameters as passed* to
``evaluate()``: one column per input parameter element, in the order
``[k, t0, p, a, i, e, w, ldc]``. A parameter passed as a scalar contributes a
single shared column; a parameter passed per passband (k) or per epoch
(t0, orbital params) is scattered into one column per passband/epoch, with each
data point's derivative landing only in its own passband's / epoch's column.
"""

import unittest

import numpy as np
from numpy import array, linspace, pi, concatenate, zeros, full

from pytransit.models.rrmodel import RoadRunnerModel


class TestRRGradHeterogeneousPassbands(unittest.TestCase):
    """Two passbands, single epoch; k and ldc passed per passband."""

    def setUp(self):
        self.npb = 2
        self.ntc = 1
        self.nor = 1
        self.nldc = 2  # quadratic

        t = linspace(-0.06, 0.06, 200)
        self.times = concatenate([t, t])
        self.lcids = concatenate([zeros(t.size, 'int'), full(t.size, 1, 'int')])
        self.pbids = array([0, 1])
        self.epids = array([0, 0])

        self.k = array([0.10, 0.12])           # per-passband -> 2 columns
        self.t0, self.p, self.a = 0.0, 3.0, 6.0
        self.i, self.e, self.w = 0.5 * pi, 0.0, 0.0
        self.ldc = array([[0.30, 0.20], [0.55, 0.10]])

        self.model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        self.model.set_data(self.times, self.lcids, self.pbids, self.epids)

        # Column layout: [k0 k1 | t0 | p a i e w | ld(pb0) ld(pb1)]
        self.ld_off = self.npb + self.ntc + 5 * self.nor   # = 8
        self.width = self.ld_off + self.npb * self.nldc     # = 12
        self.pb_of_point = self.pbids[self.lcids]

    def _eval(self, k=None, ldc=None):
        return self.model.evaluate(self.k if k is None else k, self.t0, self.p, self.a,
                                   self.i, self.e, self.w,
                                   ldc=self.ldc if ldc is None else ldc)

    def test_output_width(self):
        _, dflux = self._eval()
        self.assertEqual(dflux.shape, (self.times.size, self.width))

    def test_k_is_per_passband(self):
        eps = 1e-6
        flux0, dflux0 = self._eval()
        for ipb in range(self.npb):
            kp = self.k.copy(); kp[ipb] += eps
            km = self.k.copy(); km[ipb] -= eps
            fd = (self._eval(k=kp)[0] - self._eval(k=km)[0]) / (2 * eps)

            anal = dflux0[:, ipb]                      # column for k of passband ipb
            in_transit = flux0 < 0.9999
            same = (self.pb_of_point == ipb) & in_transit
            other = (self.pb_of_point != ipb)
            self.assertTrue(same.any())
            # k gradient uses the dldm/dk ~= 0 approximation (loose tol, as in tsmodel).
            np.testing.assert_allclose(anal[same], fd[same], rtol=0.1, atol=1e-5,
                                       err_msg=f"k gradient band {ipb}")
            # k of band ipb must not affect the other band's points.
            np.testing.assert_allclose(anal[other], 0.0, atol=1e-12)

    def test_ld_is_per_passband(self):
        eps = 1e-6
        flux0, dflux0 = self._eval()
        for ipb in range(self.npb):
            for j in range(self.nldc):
                lp = self.ldc.copy(); lp[ipb, j] += eps
                lm = self.ldc.copy(); lm[ipb, j] -= eps
                fd = (self._eval(ldc=lp)[0] - self._eval(ldc=lm)[0]) / (2 * eps)

                slot = self.ld_off + ipb * self.nldc + j
                anal = dflux0[:, slot]
                in_transit = flux0 < 0.9999
                same = (self.pb_of_point == ipb) & in_transit
                other = (self.pb_of_point != ipb)
                np.testing.assert_allclose(anal[same], fd[same], rtol=1e-3, atol=1e-7,
                                           err_msg=f"ld band {ipb} coeff {j}")
                np.testing.assert_allclose(anal[other], 0.0, atol=1e-12)


class TestRRGradPerEpoch(unittest.TestCase):
    """Two epochs with orbit variations; mirrors the requested example."""

    def setUp(self):
        self.nldc = 2
        n = 200
        t1 = linspace(-0.06, 0.06, n)
        self.times = concatenate([t1, 3.0 + t1])
        self.lcids = concatenate([zeros(n, 'int'), full(n, 1, 'int')])
        self.epids = array([0, 1])           # lc0 -> epoch 0, lc1 -> epoch 1

        self.model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        self.model.set_data(self.times, self.lcids, epids=self.epids,
                            include_orbit_variations=True)

        # x = [k, t0_0, t0_1, p, a, i0, i1, e, w, ld0, ld1]
        self.x = array([0.10, -0.001, 3.001, 3.0, 6.0,
                        0.5 * pi, 0.5 * pi - 0.01, 0.05, 0.1, 0.30, 0.20])
        self.ep_of_point = self.epids[self.lcids]

    def _eval(self, x):
        return self.model.evaluate(x[0], x[[1, 2]], x[3], x[4], x[[5, 6]], x[7], x[8],
                                   ldc=x[[9, 10]])

    def test_matches_input_arity(self):
        flux, dflux = self._eval(self.x)
        # [k | t0_0 t0_1 | p | a | i0 i1 | e | w | ld0 ld1] = 11 columns
        self.assertEqual(dflux.shape, (self.times.size, 11))

    def test_all_columns_vs_finite_differences(self):
        eps = 1e-6
        flux0, dflux0 = self._eval(self.x)
        in_transit = flux0 < 0.9999
        self.assertTrue(in_transit.any())

        for c in range(self.x.size):
            xp = self.x.copy(); xp[c] += eps
            xm = self.x.copy(); xm[c] -= eps
            fd = (self._eval(xp)[0] - self._eval(xm)[0]) / (2 * eps)
            # Column 0 is k, which uses the dldm/dk ~= 0 approximation -> looser tol.
            rtol, atol = (0.1, 1e-5) if c == 0 else (2e-3, 1e-6)
            np.testing.assert_allclose(dflux0[in_transit, c], fd[in_transit],
                                       rtol=rtol, atol=atol, err_msg=f"column {c}")

    def test_t0_and_i_are_scattered_by_epoch(self):
        _, dflux0 = self._eval(self.x)
        for ep, (t0_col, i_col) in enumerate([(1, 5), (2, 6)]):
            other = self.ep_of_point != ep
            # epoch-ep parameter columns must be zero for the other epoch's points.
            np.testing.assert_allclose(dflux0[other, t0_col], 0.0, atol=1e-12)
            np.testing.assert_allclose(dflux0[other, i_col], 0.0, atol=1e-12)


class TestRRGradSharedOrbit(unittest.TestCase):
    """Two epochs, NO orbit variations: t0 expands, orbital params stay shared."""

    def setUp(self):
        self.nldc = 2
        n = 200
        t1 = linspace(-0.06, 0.06, n)
        self.times = concatenate([t1, 3.0 + t1])
        self.lcids = concatenate([zeros(n, 'int'), full(n, 1, 'int')])
        self.epids = array([0, 1])

        self.model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        self.model.set_data(self.times, self.lcids, epids=self.epids,
                            include_orbit_variations=False)

    def _eval(self, t0):
        # i passed as a scalar -> single shared column despite two epochs.
        return self.model.evaluate(0.1, t0, 3.0, 6.0, 0.5 * pi, 0.05, 0.1,
                                   ldc=array([0.3, 0.2]))

    def test_t0_expands_orbital_shared(self):
        t0 = array([-0.001, 3.001])
        flux, dflux = self._eval(t0)
        # [k | t0_0 t0_1 | p | a | i | e | w | ld0 ld1] = 10 columns
        self.assertEqual(dflux.shape, (self.times.size, 10))

    def test_t0_columns_scattered(self):
        eps = 1e-6
        t0 = array([-0.001, 3.001])
        flux0, dflux0 = self._eval(t0)
        ep_of_point = self.epids[self.lcids]
        for ep, col in enumerate([1, 2]):
            tp = t0.copy(); tp[ep] += eps
            tm = t0.copy(); tm[ep] -= eps
            fd = (self._eval(tp)[0] - self._eval(tm)[0]) / (2 * eps)
            in_transit = flux0 < 0.9999
            np.testing.assert_allclose(dflux0[in_transit, col], fd[in_transit],
                                       rtol=2e-3, atol=1e-6, err_msg=f"t0 epoch {ep}")
            np.testing.assert_allclose(dflux0[ep_of_point != ep, col], 0.0, atol=1e-12)


class TestRRGradBackwardCompatible(unittest.TestCase):
    """Single passband, single epoch, all scalars -> compact 7+nldc layout."""

    def setUp(self):
        self.nldc = 2
        self.times = linspace(-0.06, 0.06, 200)
        self.model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        self.model.set_data(self.times)

    def _eval(self, k=0.1, a=6.0):
        return self.model.evaluate(k, 0.0, 3.0, a, 0.5 * pi, 0.05, 0.1, ldc=array([0.3, 0.2]))

    def test_width_is_seven_plus_nldc(self):
        _, dflux = self._eval()
        self.assertEqual(dflux.shape, (self.times.size, 7 + self.nldc))

    def test_k_and_a_columns(self):
        eps = 1e-6
        flux0, dflux0 = self._eval()
        in_transit = flux0 < 0.9999
        fd_k = (self._eval(k=0.1 + eps)[0] - self._eval(k=0.1 - eps)[0]) / (2 * eps)
        fd_a = (self._eval(a=6.0 + eps)[0] - self._eval(a=6.0 - eps)[0]) / (2 * eps)
        # k uses the dldm/dk ~= 0 approximation -> looser tol; a is analytic.
        np.testing.assert_allclose(dflux0[in_transit, 0], fd_k[in_transit], rtol=0.1, atol=1e-5)
        np.testing.assert_allclose(dflux0[in_transit, 3], fd_a[in_transit], rtol=1e-3, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
