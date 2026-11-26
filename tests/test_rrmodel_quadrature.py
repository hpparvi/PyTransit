#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The RoadRunner mean-intensity quadrature: nodes, the g table, its cubic lookup, and the model's accuracy."""

import warnings

import pytest
from numpy import pi, linspace, sqrt, zeros, empty, abs, allclose, isfinite, arccos, array, cos, sin, concatenate
from scipy.integrate import quad

from pytransit import RoadRunnerModel, QuadraticModel, OblatePlanetModel, TSModel
from pytransit.models.roadrunner.common import (quadrature_rules, ldm_nodes, split_point, g_nodes, table_sizes,
                                                split_cubic_coefficients, ldm_lookup, ldm_lookup_full,
                                                circle_circle_intersection_area as area)
from pytransit.models.limb_darkening import ld_quadratic

LDC = array([0.3, 0.1])


def _exact_mean_intensity(k, b):
    """Adaptive quadrature of the mean intensity under the planet at impact parameter b."""
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


class TestQuadratureNodes:
    @pytest.mark.parametrize('nq,tol', [(8, 1e-06), (16, 1e-12)])
    def test_geometric_factors_integrate_to_the_overlap_area(self, nq, tol):
        """Sum of the factors equals the planet-star overlap area in every regime.

        The integrands are regular after the substitutions rather than polynomial, so the area
        converges at the ordinary Gauss rate; the mean intensity is a ratio in which these errors
        largely cancel and converges much faster (see the next test).
        """
        rules = quadrature_rules(nq)
        for k, g in [(0.1, 0.0), (0.3, 0.1), (0.1, 0.5), (0.3, 0.4), (0.1, 0.9), (0.3, 0.8), (0.05, 0.9)]:
            gs = array([g]); mu = empty((1, 2 * nq)); wf = zeros((1, 2 * nq))
            ldm_nodes(k, gs, rules, mu, wf)
            b = g * (1 + k)
            assert wf.sum() == pytest.approx(area(1.0, k, b), rel=tol), (k, g)

    def test_mean_intensity_matches_adaptive_quadrature(self):
        rules = quadrature_rules(8)
        for k, g in [(0.1, 0.0), (0.3, 0.1), (0.1, 0.5), (0.3, 0.4), (0.1, 0.9), (0.3, 0.8)]:
            gs = array([g]); mu = empty((1, 16)); wf = zeros((1, 16))
            ldm_nodes(k, gs, rules, mu, wf)
            prof = ld_quadratic(mu[0], LDC)
            assert abs((wf[0] * prof).sum() / wf[0].sum() - _exact_mean_intensity(k, g * (1 + k))) < 1e-4, (k, g)

    # A planet larger than half the star can cover the disk centre while it crosses the limb, a
    # geometry that does not exist for k < 0.5. The cases below cover it (b < k and b + k > 1)
    # together with its neighbours.
    LARGE_PLANETS = [(0.6, 0.1), (0.6, 0.3), (0.6, 0.37), (0.6, 0.38), (0.6, 0.8), (0.8, 0.05), (0.8, 0.2),
                     (0.8, 0.44), (0.8, 0.7), (1.0, 0.01), (1.0, 0.3), (1.0, 0.5), (1.0, 0.9)]

    def test_geometric_factors_integrate_to_the_overlap_area_for_large_planets(self):
        rules = quadrature_rules(8)
        for k, g in self.LARGE_PLANETS:
            gs = array([g]); mu = empty((1, 16)); wf = zeros((1, 16))
            ldm_nodes(k, gs, rules, mu, wf)
            assert wf.sum() == pytest.approx(area(1.0, k, g * (1 + k)), rel=2e-4), (k, g)

    def test_mean_intensity_matches_adaptive_quadrature_for_large_planets(self):
        rules = quadrature_rules(8)
        for k, g in self.LARGE_PLANETS:
            gs = array([g]); mu = empty((1, 16)); wf = zeros((1, 16))
            ldm_nodes(k, gs, rules, mu, wf)
            prof = ld_quadratic(mu[0], LDC)
            assert abs((wf[0] * prof).sum() / wf[0].sum() - _exact_mean_intensity(k, g * (1 + k))) < 1e-5, (k, g)

    @pytest.mark.parametrize('k', [0.55, 0.7, 0.9, 1.0])
    @pytest.mark.parametrize('b', [0.0, 0.3, 0.9])
    def test_large_planet_light_curve_matches_adaptive_quadrature(self, k, b):
        # The reference is integrated directly because the analytic quadratic model is itself
        # off by some tens of ppm for a planet this large when its edge passes the disk centre.
        a, p = 40.0, 30.0
        time = linspace(-0.25, 0.25, 60)
        inc = arccos(b / a)
        phase = 2 * pi * time / p
        z = a * sqrt(sin(phase) ** 2 + (cos(inc) * cos(phase)) ** 2)
        star = pi * (1.0 - LDC[0] / 3.0 - LDC[1] / 6.0)
        ref = array([1.0 - _exact_mean_intensity(k, x) * area(1.0, k, x) / star if x < 1.0 + k - 1e-9 else 1.0 for x in z])
        tm = RoadRunnerModel('quadratic')
        tm.set_data(time)
        assert abs(tm.evaluate(k, LDC, 0.0, p, a, inc) - ref).max() < 2e-5

    def test_nodes_lie_on_the_disk(self):
        rules = quadrature_rules(8)
        gs = linspace(0, 1 - 1e-9, 50); mu = empty((50, 16)); wf = zeros((50, 16))
        ldm_nodes(0.2, gs, rules, mu, wf)
        assert ((mu >= 0.0) & (mu <= 1.0)).all()


class TestGTable:
    def test_split_point_is_the_limb_contact(self):
        assert split_point(0.1) == pytest.approx(0.9 / 1.1)

    def test_g_nodes_form_three_pieces_meeting_at_the_contacts(self):
        for k in (0.005, 0.1, 0.5):
            gc = split_point(k)
            for ng in (12, 39, 100):
                gs, n1 = g_nodes(k, ng)
                n1_, n2 = table_sizes(ng)
                assert gs.size == ng and n1 == n1_ and n1 + 2 * n2 == ng and n1 >= 4 and n2 >= 4
                # Full transit: from the limb contact to the disk centre
                assert gs[0] == pytest.approx(gc) and gs[n1 - 1] == pytest.approx(0.0, abs=1e-15)
                # Ingress: from each contact to the middle of the ingress
                gm = 0.5 * (gc + 1.0)
                assert gs[n1] == pytest.approx(gc) and gs[n1 + n2 - 1] == pytest.approx(gm)
                assert gs[n1 + n2] == pytest.approx(1.0) and gs[-1] == pytest.approx(gm)

    def test_nodes_are_uniform_in_the_piece_coordinates(self):
        k, ng = 0.1, 39
        gc = split_point(k)
        gs, n1 = g_nodes(k, ng)
        n2 = table_sizes(ng)[1]
        assert allclose(sqrt(1.0 - (gs[:n1] / gc) ** 2), linspace(0, 1, n1), atol=1e-7)
        assert allclose(sqrt(2.0 * (gs[n1:n1 + n2] - gc) / (1.0 - gc)), linspace(0, 1, n2), atol=1e-7)
        assert allclose(sqrt(2.0 * (1.0 - gs[n1 + n2:]) / (1.0 - gc)), linspace(0, 1, n2), atol=1e-7)

    @staticmethod
    def _lookup_error(f, k, ng, gmin=0.0, gmax=1.0):
        gs, n1 = g_nodes(k, ng); gc = split_point(k)
        coef = zeros((ng - 3, 4))
        split_cubic_coefficients(f(gs), n1, coef)
        return max(abs(ldm_lookup(g, gc, n1, coef) - f(g)) for g in linspace(gmin, gmax, 1001))

    # The mean intensity is an even function of the impact parameter, and the first piece's
    # coordinate, sqrt(1 - (g / gc)**2), is built for even functions; the test functions that
    # cover the first piece are therefore even in g.

    def test_cubic_lookup_is_exact_for_the_lowest_orders(self):
        for k in (0.1, 0.3):
            gc = split_point(k)
            # g**2 is quadratic in the first piece's coordinate, and g in the ingress pieces'
            assert self._lookup_error(lambda x: 1.0 - 0.3 * x ** 2, k, 39, 0.0, gc - 1e-9) < 1e-12
            assert self._lookup_error(lambda x: 1.0 - 0.3 * x, k, 39, gc, 1.0) < 1e-12

    def test_cubic_lookup_resolves_the_contact_square_roots(self):
        # The singular terms the table is built to absorb: a square root at the last contact and
        # a 3/2 power at the limb contact. A 39-node table uniform in g is off by 3e-2 for the
        # square root; in the piece coordinates both terms are polynomials next to their contact.
        k = 0.1; gc = split_point(k)
        assert self._lookup_error(lambda x: sqrt(abs(1.0 - x ** 2)), k, 39) < 1e-4
        assert self._lookup_error(lambda x: abs(x ** 2 - gc ** 2) ** 1.5, k, 39) < 1e-4

    def test_cubic_lookup_is_fourth_order_for_smooth_functions(self):
        errs = [self._lookup_error(lambda x: cos(3.0 * x), 0.1, ng) for ng in (48, 96, 192)]
        assert errs[0] / errs[1] > 8 and errs[1] / errs[2] > 8

    def test_cubic_lookup_is_fourth_order_across_the_contacts(self):
        k = 0.1; gc = split_point(k)
        f = lambda x: cos(3.0 * x) * (1.0 + sqrt(abs(1.0 - x ** 2))) + abs(x ** 2 - gc ** 2) ** 1.5
        errs = [self._lookup_error(f, k, ng) for ng in (48, 96, 192)]
        assert errs[0] / errs[1] > 8 and errs[1] / errs[2] > 8

    def test_squared_distance_lookup_matches_the_general_lookup(self):
        for k in (0.05, 0.3):
            gc = split_point(k)
            gs, n1 = g_nodes(k, 39)
            coef = zeros((36, 4))
            split_cubic_coefficients(cos(2.0 * gs), n1, coef)
            for z in linspace(0.0, (1.0 - k) * (1.0 - 1e-9), 200):
                assert ldm_lookup_full(z * z, 1.0 / (1.0 - k) ** 2, n1, coef) == pytest.approx(
                    ldm_lookup(z / (1.0 + k), gc, n1, coef), abs=1e-12)


class TestModelAccuracy:
    """Defaults (nq=8, ng=39) against the analytic Mandel & Agol model, worst over impact parameters."""
    a, p = 13.0, 4.0
    times = linspace(-0.08, 0.08, 800)

    def _err(self, k, b, **kw):
        ref = QuadraticModel(); ref.set_data(self.times)
        tm = RoadRunnerModel('quadratic', small_planet_limit=kw.pop('small_planet_limit', 0.0), **kw); tm.set_data(self.times)
        i = arccos(b / self.a)
        return 1e6 * abs(tm.evaluate(k, list(LDC), 0.0, self.p, self.a, i) - ref.evaluate(k, list(LDC), 0.0, self.p, self.a, i)).max()

    @pytest.mark.parametrize('k,tol', [(0.02, 0.5), (0.05, 1.0), (0.1, 1.0), (0.2, 3.0), (0.3, 3.0)])
    def test_defaults(self, k, tol):
        assert max(self._err(k, b) for b in (0.0, 0.7, 0.9)) < tol

    def test_more_nodes_help(self):
        assert self._err(0.3, 0.0, ng=200, nq=12) < 0.5 * self._err(0.3, 0.0)

    def test_small_planet_path(self):
        assert self._err(0.005, 0.0, small_planet_limit=0.01) < 1.0


class TestDeprecatedArguments:
    @pytest.mark.parametrize('kw', [dict(nz=40), dict(nzin=20), dict(nzlimb=20), dict(zcut=0.7),
                                    dict(precompute_weights=True), dict(klims=(0.05, 0.2)), dict(nk=128)])
    def test_old_arguments_warn_and_are_ignored(self, kw):
        with pytest.warns(FutureWarning):
            tm = RoadRunnerModel(**kw)
        assert tm.nq == 8 and tm.ng == 39

    def test_new_arguments_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            tm = RoadRunnerModel(nq=12, ng=50)
        assert tm.nq == 12 and tm.ng == 50

    def test_oblate_and_spectroscopy_models_take_nq(self):
        assert OblatePlanetModel(nq=6).nq == 6
        assert TSModel(nq=6).nq == 6
