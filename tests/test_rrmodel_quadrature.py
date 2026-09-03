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
from numpy import pi, linspace, sqrt, zeros, empty, abs, allclose, isfinite, arccos, array, cos, concatenate
from scipy.integrate import quad

from pytransit import RoadRunnerModel, QuadraticModel, OblatePlanetModel, TSModel
from pytransit.models.roadrunner.common import (quadrature_rules, ldm_nodes, split_point, g_nodes,
                                                cubic_coefficients, ldm_lookup,
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

    def test_nodes_lie_on_the_disk(self):
        rules = quadrature_rules(8)
        gs = linspace(0, 1 - 1e-9, 50); mu = empty((50, 16)); wf = zeros((50, 16))
        ldm_nodes(0.2, gs, rules, mu, wf)
        assert ((mu >= 0.0) & (mu <= 1.0)).all()


class TestGTable:
    def test_split_point_is_the_limb_contact(self):
        assert split_point(0.1) == pytest.approx(0.9 / 1.1)

    def test_g_nodes_share_the_split_node_and_have_at_least_four_per_segment(self):
        for k in (0.005, 0.1, 0.5):
            gs, n1 = g_nodes(k, 100)
            assert gs.size == 100
            assert gs[0] == 0.0 and gs[n1 - 1] == pytest.approx(split_point(k)) and gs[n1] == pytest.approx(split_point(k))
            assert n1 >= 4 and gs.size - n1 >= 4

    def test_cubic_lookup_reproduces_a_cubic_exactly(self):
        f = lambda x: 1.0 - 0.3 * x + 0.2 * x ** 2 - 0.5 * x ** 3
        for k in (0.1, 0.3):
            gs, n1 = g_nodes(k, 40); gc = split_point(k)
            coef = concatenate([cubic_coefficients(f(gs[:n1])), cubic_coefficients(f(gs[n1:]))])
            for g in linspace(0.0, 0.999, 200):
                assert abs(ldm_lookup(g, gc, n1, coef) - f(g)) < 1e-12

    def test_cubic_lookup_is_fourth_order(self):
        f = lambda x: cos(3.0 * x)
        errs = []
        for ng in (50, 100, 200):
            gs, n1 = g_nodes(0.1, ng); gc = split_point(0.1)
            coef = concatenate([cubic_coefficients(f(gs[:n1])), cubic_coefficients(f(gs[n1:]))])
            errs.append(max(abs(ldm_lookup(g, gc, n1, coef) - f(g)) for g in linspace(0, 0.999, 500)))
        assert errs[0] / errs[1] > 8 and errs[1] / errs[2] > 8


class TestModelAccuracy:
    """Defaults (nq=8, ng=100) against the analytic Mandel & Agol model, worst over impact parameters."""
    a, p = 13.0, 4.0
    times = linspace(-0.08, 0.08, 800)

    def _err(self, k, b, **kw):
        ref = QuadraticModel(); ref.set_data(self.times)
        tm = RoadRunnerModel('quadratic', small_planet_limit=kw.pop('small_planet_limit', 0.0), **kw); tm.set_data(self.times)
        i = arccos(b / self.a)
        return 1e6 * abs(tm.evaluate(k, list(LDC), 0.0, self.p, self.a, i) - ref.evaluate(k, list(LDC), 0.0, self.p, self.a, i)).max()

    @pytest.mark.parametrize('k,tol', [(0.02, 0.3), (0.05, 1.0), (0.1, 2.5), (0.2, 5.0), (0.3, 10.0)])
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
        assert tm.nq == 8 and tm.ng == 100

    def test_new_arguments_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            tm = RoadRunnerModel(nq=12, ng=50)
        assert tm.nq == 12 and tm.ng == 50

    def test_oblate_and_spectroscopy_models_take_nq(self):
        assert OblatePlanetModel(nq=6).nq == 6
        assert TSModel(nq=6).nq == 6
