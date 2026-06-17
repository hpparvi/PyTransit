"""Tests for tsmodel_and_grad: forward values and analytical gradients vs finite differences."""

import unittest
import numpy as np
from numpy import zeros, sqrt, pi, linspace, array

from pytransit.backends.numba.rrmodel import create_z_grid, calculate_weights_3d
from pytransit.backends.numba.limb_darkening.quadratic import ld_quadratic, ldd_quadratic, ldi_quadratic, ldig_quadratic
from pytransit.backends.numba.tsmodel_grad import tsmodel_and_grad


def setup_ld_inputs(mu, ldc_2d, nk, klims, ze, ng):
    """Build ldp, ldg, istar, distar arrays for tsmodel_and_grad."""
    npv, npb, nldc = ldc_2d.shape
    nmu = mu.size

    ldp = zeros((npv, npb, nmu))
    ldg = zeros((npv, npb, 1 + nldc, nmu))
    istar = zeros((npv, npb))
    distar = zeros((npv, npb, nldc))

    for ipv in range(npv):
        for ipb in range(npb):
            pv = ldc_2d[ipv, ipb]
            ldp[ipv, ipb, :] = ld_quadratic(mu, pv)
            ldd = ldd_quadratic(mu, pv)
            ldg[ipv, ipb, :, :] = ldd
            istar[ipv, ipb] = ldi_quadratic(pv)
            distar[ipv, ipb, :] = ldig_quadratic(pv)

    dk, dg, weights = calculate_weights_3d(nk, klims[0], klims[1], ze, ng)
    return ldp, ldg, istar, distar, dk, dg, weights


class TestTSModelAndGrad(unittest.TestCase):

    def setUp(self):
        """Set up a basic transit scenario with 2 passbands."""
        self.npv = 1
        self.npb = 2
        self.nldc = 2  # quadratic LD

        # Orbital parameters
        self.k_vals = array([[0.09, 0.11]])
        self.t0 = array([0.0])
        self.p = array([3.0])
        self.a = array([5.0])
        self.inc = array([87.0 * pi / 180.0])
        self.e = array([0.1])  # Non-zero to allow FD for e
        self.w = array([0.5])

        self.times = linspace(-0.15, 0.15, 300)
        self.nsamples = array([1])
        self.exptimes = array([0.0])

        # LD coefficients: different per band
        self.ldc = array([[[0.3, 0.2], [0.5, 0.1]]])

        # Integration setup
        self.nk = 256
        self.klims = (0.005, 0.5)
        self.ng = 100

        ze, zm = create_z_grid(0.7, 20, 20)
        self.ze = ze
        self.mu = sqrt(1 - zm ** 2)

        self.ldp, self.ldg, self.istar, self.distar, self.dk, self.dg, self.weights = \
            setup_ld_inputs(self.mu, self.ldc, self.nk, self.klims, ze, self.ng)

    def _call(self, k=None, t0=None, p=None, a=None, inc=None, e=None, w=None):
        """Call tsmodel_and_grad with fixed LD inputs."""
        return tsmodel_and_grad(
            self.times,
            k if k is not None else self.k_vals.copy(),
            t0 if t0 is not None else self.t0.copy(),
            p if p is not None else self.p.copy(),
            a if a is not None else self.a.copy(),
            inc if inc is not None else self.inc.copy(),
            e if e is not None else self.e.copy(),
            w if w is not None else self.w.copy(),
            self.nsamples, self.exptimes,
            self.ldp, self.ldg, self.istar, self.distar,
            self.weights, self.dk, self.klims[0], self.klims[1], self.ng, self.dg, self.ze
        )

    def _call_with_ldc(self, ldc):
        """Call tsmodel_and_grad recomputing LD for given coefficients."""
        ldp, ldg, istar, distar, dk, dg, wts = \
            setup_ld_inputs(self.mu, ldc, self.nk, self.klims, self.ze, self.ng)
        return tsmodel_and_grad(
            self.times, self.k_vals.copy(), self.t0.copy(), self.p.copy(),
            self.a.copy(), self.inc.copy(), self.e.copy(), self.w.copy(),
            self.nsamples, self.exptimes,
            ldp, ldg, istar, distar,
            wts, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze
        )

    def test_output_shapes(self):
        flux, dflux = self._call()
        npt = self.times.size
        self.assertEqual(flux.shape, (self.npv, self.npb, npt))
        self.assertEqual(dflux.shape, (self.npv, self.npb, npt, 7 + self.nldc))

    def test_out_of_transit_is_one(self):
        flux, _ = self._call()
        for ipb in range(self.npb):
            self.assertAlmostEqual(flux[0, ipb, 0], 1.0, places=10)
            self.assertAlmostEqual(flux[0, ipb, -1], 1.0, places=10)

    def test_transit_depth(self):
        flux, _ = self._call()
        mid = self.times.size // 2
        for ipb in range(self.npb):
            self.assertLess(flux[0, ipb, mid], 1.0)

    def test_gradient_k_multi_band(self):
        """k derivative with multi-band (kmean approximation, relaxed tolerance)."""
        eps = 1e-7
        flux0, dflux0 = self._call()

        for ipb in range(self.npb):
            kp = self.k_vals.copy(); kp[0, ipb] += eps
            fp, _ = self._call(k=kp)
            km = self.k_vals.copy(); km[0, ipb] -= eps
            fm, _ = self._call(k=km)

            fd = (fp[0, ipb] - fm[0, ipb]) / (2 * eps)
            anal = dflux0[0, ipb, :, 0]
            mask = flux0[0, ipb] < 0.9999
            if mask.any():
                np.testing.assert_allclose(anal[mask], fd[mask], rtol=0.1, atol=1e-8,
                    err_msg=f"k gradient band {ipb}")

    def test_gradient_orbital_params(self):
        """Orbital parameter derivatives against finite differences."""
        eps = 1e-7
        flux0, dflux0 = self._call()

        params = [
            ('t0', 't0', self.t0),
            ('p', 'p', self.p),
            ('a', 'a', self.a),
            ('i', 'inc', self.inc),
            ('e', 'e', self.e),
            ('w', 'w', self.w),
        ]

        for ip, (name, kwarg, arr) in enumerate(params):
            plus = arr.copy(); plus[0] += eps
            fp, _ = self._call(**{kwarg: plus})
            minus = arr.copy(); minus[0] -= eps
            fm, _ = self._call(**{kwarg: minus})

            for ipb in range(self.npb):
                fd = (fp[0, ipb] - fm[0, ipb]) / (2 * eps)
                anal = dflux0[0, ipb, :, ip + 1]
                mask = flux0[0, ipb] < 0.9999
                if mask.any():
                    np.testing.assert_allclose(anal[mask], fd[mask], rtol=1e-3, atol=1e-8,
                        err_msg=f"{name} gradient band {ipb}")

    def test_gradient_ld_coefficients(self):
        """LD coefficient derivatives against finite differences."""
        eps = 1e-7
        flux0, dflux0 = self._call_with_ldc(self.ldc)

        for j in range(self.nldc):
            for ipb in range(self.npb):
                ldc_p = self.ldc.copy(); ldc_p[0, ipb, j] += eps
                fp, _ = self._call_with_ldc(ldc_p)
                ldc_m = self.ldc.copy(); ldc_m[0, ipb, j] -= eps
                fm, _ = self._call_with_ldc(ldc_m)

                fd = (fp[0, ipb] - fm[0, ipb]) / (2 * eps)
                anal = dflux0[0, ipb, :, 7 + j]
                mask = flux0[0, ipb] < 0.9999
                if mask.any():
                    np.testing.assert_allclose(anal[mask], fd[mask], rtol=1e-3, atol=1e-8,
                        err_msg=f"LD coeff {j} band {ipb}")

    def test_eccentric_orbit(self):
        flux, dflux = self._call(e=array([0.3]), w=array([0.5]))
        mid = self.times.size // 2
        self.assertFalse(np.any(np.isnan(flux)))
        for ipb in range(self.npb):
            self.assertLess(flux[0, ipb, mid], 1.0)

    def test_invalid_params_return_nan(self):
        flux, dflux = self._call(a=array([0.5]))
        self.assertTrue(np.all(np.isnan(flux)))
        self.assertTrue(np.all(np.isnan(dflux)))

    def test_gradient_k_single_band(self):
        """k derivative with single band (kmean == k, relaxed for dldm/dk ≈ 0)."""
        k = array([[0.1]])
        ldc = array([[[0.3, 0.2]]])
        ldp, ldg, istar, distar, dk, dg, wts = setup_ld_inputs(
            self.mu, ldc, self.nk, self.klims, self.ze, self.ng)

        def call_1b(k_val):
            return tsmodel_and_grad(
                self.times, array([[k_val]]), self.t0, self.p, self.a, self.inc, self.e, self.w,
                self.nsamples, self.exptimes, ldp, ldg, istar, distar,
                wts, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze)

        eps = 1e-7
        flux0, dflux0 = call_1b(0.1)
        fp, _ = call_1b(0.1 + eps)
        fm, _ = call_1b(0.1 - eps)

        fd = (fp[0, 0] - fm[0, 0]) / (2 * eps)
        anal = dflux0[0, 0, :, 0]
        mask = flux0[0, 0] < 0.9999
        if mask.any():
            # dldm/dk ≈ 0 approximation still affects near-limb points
            np.testing.assert_allclose(anal[mask], fd[mask], rtol=0.1, atol=1e-8,
                err_msg="Single-band k gradient")


if __name__ == '__main__':
    unittest.main()
