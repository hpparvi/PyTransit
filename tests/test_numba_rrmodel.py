"""The numba RoadRunner model against exact integration of the limb-darkened disk."""
import unittest

import numpy as np
from numpy import array, linspace, pi, arccos, sqrt, abs as npabs
from scipy.integrate import quad
from meepmeep.numba2d import solve2d, sep_c

from pytransit.models.rrmodel import RoadRunnerModel
from pytransit.backends.numba.ccintersection import ccia
from pytransit.backends.numba.limb_darkening import ld_quadratic, ldi_quadratic

LDC = array([0.3, 0.1])
P, A = 3.0, 6.0


def exact_mean_intensity(k, b):
    def th(z):
        if b < 1e-12:
            return 2 * pi if z < k else 0.0
        if z <= k - b:
            return 2 * pi
        if z < b - k or z > b + k:
            return 0.0
        return 2 * arccos(min(1.0, max(-1.0, (z * z + b * b - k * k) / (2 * z * b))))
    lo, hi = max(0.0, b - k), min(1.0, b + k)
    pts = [x for x in (b - k, b + k, k - b) if lo < x < hi] or None
    num = quad(lambda z: th(z) * z * ld_quadratic(array([sqrt(1 - z * z)]), LDC)[0], lo, hi, points=pts, limit=400, epsrel=1e-12, epsabs=0)[0]
    den = quad(lambda z: th(z) * z, lo, hi, points=pts, limit=400, epsrel=1e-12, epsabs=0)[0]
    return num / den


def reference_flux(z, k):
    istar = ldi_quadratic(LDC)
    return array([1.0 if zi >= 1 + k else (istar - exact_mean_intensity(k, zi) * ccia(1.0, k, zi)) / istar for zi in z])


def inclination(b):
    return arccos(b / A)


class TestFluxAccuracy(unittest.TestCase):
    """Worst-case error over impact parameters 0, 0.7 and 0.9 against nested adaptive quadrature."""
    tolerances_ppm = {0.02: 0.5, 0.05: 1.0, 0.1: 2.5, 0.2: 5.0, 0.3: 10.0}

    def test_defaults_against_exact_integration(self):
        times = linspace(-0.09, 0.09, 61)
        model = RoadRunnerModel(backend='numba')
        model.set_data(times)
        for k, tol in self.tolerances_ppm.items():
            worst = 0.0
            for b in (0.0, 0.7, 0.9):
                inc = inclination(b)
                xyc = solve2d(0.0, P, A, inc, 0.0, 0.0)
                z = array([sep_c(t, xyc) for t in times])
                flux = model.evaluate(k, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC)
                worst = max(worst, 1e6 * npabs(flux - reference_flux(z, k)).max())
            self.assertLess(worst, tol, f'k = {k}: {worst:.2f} ppm')

    def test_more_nodes_reduce_the_error(self):
        times = linspace(-0.09, 0.09, 61)
        inc = inclination(0.7)
        xyc = solve2d(0.0, P, A, inc, 0.0, 0.0)
        z = array([sep_c(t, xyc) for t in times])
        errs = []
        for nq, ng in ((8, 100), (16, 400)):
            model = RoadRunnerModel(backend='numba', nq=nq, ng=ng)
            model.set_data(times)
            errs.append(npabs(model.evaluate(0.2, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC) - reference_flux(z, 0.2)).max())
        self.assertLess(errs[1], 0.3 * errs[0])


class TestRadiusRatioDerivative(unittest.TestCase):
    """The k derivative includes the change of the mean intensity under the planet with k."""

    def test_matches_central_differences(self):
        times = linspace(-0.09, 0.09, 121)
        model = RoadRunnerModel(backend='numba', return_grad=True)
        model.set_data(times)
        h = 1e-5
        # The impact parameters avoid 1 - k for the tested k: there the planet sits exactly at the
        # limb contact at mid-transit, where the derivative is discontinuous and a central
        # difference returns the mean of the two sides.
        for b in (0.0, 0.7, 0.93):
            inc = inclination(b)
            for k in (0.05, 0.1, 0.2):
                flux, dflux = model.evaluate(k, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC)
                fp = model.evaluate(k + h, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC)[0]
                fm = model.evaluate(k - h, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC)[0]
                fd = (fp - fm) / (2 * h)
                m = npabs(fd) > 1e-3
                np.testing.assert_allclose(dflux[m, 0], fd[m], rtol=2e-3, atol=1e-5, err_msg=f'b = {b}, k = {k}')


if __name__ == '__main__':
    unittest.main()
