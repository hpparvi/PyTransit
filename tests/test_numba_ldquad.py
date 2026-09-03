"""The quadrature primitives behind the numba RoadRunner model."""
import unittest

import numpy as np
from numpy import pi, linspace, sqrt, zeros, empty, array, arccos, cos, concatenate
from scipy.integrate import quad

from pytransit.backends.numba.ldquad import (quadrature_rules, ldm_nodes, split_point, g_nodes, cubic_coefficients,
                                             split_cubic_coefficients, ldm_lookup, ldm_lookup_and_grad, profile_grid,
                                             profile_at, ldm_table)
from pytransit.backends.numba.ccintersection import ccia
from pytransit.backends.numba.limb_darkening import ld_quadratic

LDC = array([0.3, 0.1])
CASES = [(0.1, 0.0), (0.3, 0.1), (0.1, 0.5), (0.3, 0.4), (0.1, 0.9), (0.3, 0.8), (0.05, 0.9)]


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


class TestNodes(unittest.TestCase):
    def test_factors_integrate_to_the_overlap_area(self):
        for nq, tol in ((8, 1e-6), (16, 1e-12)):
            rules = quadrature_rules(nq)
            for k, g in CASES:
                mu = empty((1, 2 * nq)); wf = zeros((1, 2 * nq)); ldm_nodes(k, array([g]), rules, mu, wf)
                self.assertAlmostEqual(wf.sum() / ccia(1.0, k, g * (1 + k)), 1.0, delta=tol, msg=(nq, k, g))

    def test_mean_intensity_matches_adaptive_quadrature(self):
        rules = quadrature_rules(8)
        for k, g in CASES:
            mu = empty((1, 16)); wf = zeros((1, 16)); ldm_nodes(k, array([g]), rules, mu, wf)
            prof = ld_quadratic(mu[0], LDC)
            self.assertLess(abs((wf[0] * prof).sum() / wf[0].sum() - exact_mean_intensity(k, g * (1 + k))), 1e-4, (k, g))

    def test_tabulated_profile_matches_direct_evaluation(self):
        rules = quadrature_rules(8); gs = g_nodes(0.1, 50); mu = empty((50, 16)); wf = zeros((50, 16)); ldm_nodes(0.1, gs, rules, mu, wf)
        mug, t0, dt = profile_grid(); ldp = ld_quadratic(mug, LDC); tab = empty(50); ldm_table(mu, wf, t0, dt, ldp, tab)
        direct = array([(wf[i] * ld_quadratic(mu[i], LDC)).sum() / wf[i].sum() for i in range(50)])
        # At g = 1 the innermost node lies below the profile table's first mu, where the profile is
        # held constant; the overlap area vanishes there so the table value never reaches the flux.
        np.testing.assert_allclose(tab[:-1], direct[:-1], atol=1e-8)


class TestGTable(unittest.TestCase):
    def test_nodes_split_at_the_limb_contact(self):
        gs = g_nodes(0.1, 100)
        self.assertEqual(gs.size, 100)
        self.assertAlmostEqual(gs[49], split_point(0.1)); self.assertAlmostEqual(gs[50], split_point(0.1))
        self.assertEqual(gs[0], 0.0); self.assertEqual(gs[-1], 1.0)

    def test_cubic_lookup_reproduces_a_cubic_and_its_derivative(self):
        f = lambda x: 1.0 - 0.3 * x + 0.2 * x ** 2 - 0.5 * x ** 3
        df = lambda x: -0.3 + 0.4 * x - 1.5 * x ** 2
        for k in (0.1, 0.3):
            gs = g_nodes(k, 40); gc = split_point(k); n1 = 20
            coef = zeros((38, 4)); split_cubic_coefficients(f(gs), n1, coef)
            for g in linspace(0.0, 0.999, 200):
                v, d = ldm_lookup_and_grad(g, gc, n1, coef)
                self.assertAlmostEqual(v, f(g), delta=1e-12); self.assertAlmostEqual(d, df(g), delta=1e-10)
                self.assertAlmostEqual(ldm_lookup(g, gc, n1, coef), v, delta=1e-15)

    def test_cubic_lookup_is_fourth_order(self):
        f = lambda x: cos(3.0 * x); errs = []
        for ng in (50, 100, 200):
            gs = g_nodes(0.1, ng); gc = split_point(0.1); n1 = ng // 2
            coef = zeros((ng - 2, 4)); split_cubic_coefficients(f(gs), n1, coef)
            errs.append(max(abs(ldm_lookup(g, gc, n1, coef) - f(g)) for g in linspace(0, 0.999, 500)))
        self.assertGreater(errs[0] / errs[1], 8); self.assertGreater(errs[1] / errs[2], 8)


if __name__ == '__main__':
    unittest.main()
