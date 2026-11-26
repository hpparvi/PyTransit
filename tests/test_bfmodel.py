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

"""The brute-force model is the reference the other models are tested against, so it is tested
against analytic solutions that share none of its code."""

from math import cos, sin, sqrt, acos

import pytest
from numba import njit
from numpy import pi, linspace, array, arccos, isfinite, mean, concatenate
from scipy.integrate import quad

from meepmeep.backends.numba.newton.newton import xyz_newton_v

from pytransit import BFModel, RoadRunnerModel
from pytransit.models.numba.gdmodel import star_planet_overlap
from pytransit.models.numba.ma_chromosphere_nb import chromosphere_z_s
from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s
from pytransit.models.numba.ma_uniform_nb import uniform_z_s

LDC = [0.4, 0.25]

# Planet-star distances across the whole transit, including both contact points.
def _zs(k):
    return concatenate([linspace(0.0, 1.0 - k, 6), [1.0 - k], linspace(1.0 - k + 1e-3, 1.0 + k - 1e-3, 9), [1.0 + k]])


# The same, but avoiding the exact contact z = 1 - k.
def _zs_off_contact(k):
    return concatenate([linspace(0.01, 1.0 - k - 1e-6, 6), linspace(1.0 - k + 1e-3, 1.0 + k - 1e-3, 9)])


def _ring_reference(z, k, u1, u2):
    """The quadratic transit as an integral over stellar rings, each weighted by its arc inside the planet."""
    def intensity(r):
        mu = sqrt(1.0 - r * r)
        return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2

    def arc(r):
        if r <= k - z:
            return 2.0 * pi
        return 2.0 * acos(min(max((r * r + z * z - k * k) / (2.0 * r * z), -1.0), 1.0))

    lo, hi = max(z - k, 0.0), min(z + k, 1.0)
    points = [k - z] if lo < k - z < hi else None
    blocked = quad(lambda r: intensity(r) * arc(r) * r, lo, hi, epsabs=1e-14, epsrel=1e-13, limit=500,
                   points=points)[0]
    total = 2.0 * pi * quad(lambda r: intensity(r) * r, 0.0, 1.0, epsabs=1e-14, epsrel=1e-13)[0]
    return 1.0 - blocked / total


@njit
def ld_shell(mu, pv):
    """The limb-brightened profile of an optically thin shell."""
    return 1.0 / mu


@pytest.mark.parametrize('k', [0.05, 0.1, 0.3])
def test_uniform_disk_matches_the_analytic_solution(k):
    tm = BFModel('uniform')
    for z in _zs(k):
        assert abs(tm.evaluate_xy(z, 0.0, k, []) - uniform_z_s(z, k, 1.0)) < 1e-10


@pytest.mark.parametrize('k', [0.05, 0.1, 0.3])
def test_quadratic_profile_matches_a_ring_integral(k):
    tm = BFModel('quadratic')
    for z in _zs(k):
        assert abs(tm.evaluate_xy(0.0, z, k, LDC) - _ring_reference(z, k, *LDC)) < 1e-11


@pytest.mark.parametrize('k', [0.05, 0.1, 0.3])
def test_quadratic_profile_matches_mandel_agol(k):
    """The analytic model approximates its elliptic integrals, to about 1e-7 at k = 0.3."""
    tm = BFModel('quadratic')
    for z in _zs_off_contact(k):
        assert abs(tm.evaluate_xy(0.0, z, k, LDC) - eval_quad_z_s(z, k, array(LDC))) < 2e-7


def test_singular_shell_profile_matches_the_chromosphere_model():
    """The shell's brightness diverges at the limb; the integration must still converge."""
    tm = BFModel(ld_shell)
    for z in _zs_off_contact(0.1):
        assert abs(tm.evaluate_xy(z, 0.0, 0.1, [0.0]) - chromosphere_z_s(z, 0.1)) < 1e-8


@pytest.mark.parametrize('fstar,astar', [(0.1, 0.0), (0.1, 0.4), (0.25, 1.2)])
def test_oblate_star_blocks_the_exact_overlap_area(fstar, astar):
    """For a uniform star the blocked fraction is the overlap area over the stellar disk's area."""
    tm = BFModel('uniform')
    k = 0.1
    for x, y in [(0.0, 0.0), (0.0, 0.86), (0.0, 0.93), (0.97, 0.05), (0.62, 0.55), (-0.4, -0.7)]:
        # star_planet_overlap takes an axis-aligned star, so rotate the planet into the star's frame.
        xs, ys = cos(astar) * x + sin(astar) * y, -sin(astar) * x + cos(astar) * y
        expected = 1.0 - star_planet_overlap(xs, ys, k, fstar) / (pi * (1.0 - fstar))
        assert abs(tm.evaluate_xy(x, y, k, [], fstar=fstar, astar=astar) - expected) < 1e-10


def test_oblate_planet_matches_the_opmodel_reference():
    """The scanline reference in the OPModel tests was written independently of this model."""
    from tests.test_opmodel import reference_flux
    tm = BFModel('quadratic')
    k, f, alpha = 0.1, 0.3, 0.7
    for x, y in [(0.0, 0.0), (0.5, 0.3), (0.9, 0.1), (0.95, 0.25), (1.05, 0.0)]:
        assert abs(tm.evaluate_xy(x, y, k, LDC, f=f, alpha=alpha) - reference_flux(x, y, k, f, alpha, *LDC)) < 1e-9


def test_brightness_function_matches_the_radial_profile():
    """A 2D brightness function that encodes the radial profile must give the same transit."""
    u1, u2 = LDC

    def brightness(x, y, pv):
        r2 = x * x + y * y
        mu = sqrt(max(1.0 - r2, 0.0))
        return 1.0 - pv[0] * (1.0 - mu) - pv[1] * (1.0 - mu) ** 2

    tp, tb = BFModel('quadratic'), BFModel(brightness=brightness)
    for x, y in [(0.0, 0.2), (0.6, 0.6), (0.98, 0.0)]:
        kw = dict(k=0.1, ldc=LDC, f=0.2, alpha=0.5)
        assert abs(tp.evaluate_xy(x, y, **kw) - tb.evaluate_xy(x, y, **kw)) < 1e-10


class TestLightCurves:
    time = linspace(-0.12, 0.12, 49)
    orbit = dict(t0=0.0, p=4.0, a=13.0, i=0.49 * pi)

    def test_exact_orbit_places_the_planet_with_the_kepler_solution(self):
        tm = BFModel('quadratic')
        tm.set_data(self.time)
        flux = tm.evaluate(0.1, LDC, e=0.3, w=1.0, **self.orbit)
        x, y, z = xyz_newton_v(self.time, 0.0, 4.0, 13.0, 0.49 * pi, 0.3, 1.0)
        expected = array([tm.evaluate_xy(xx, yy, 0.1, LDC) for xx, yy in zip(x, y)])
        assert abs(flux - expected).max() < 1e-12

    def test_taylor_orbit_matches_a_high_resolution_roadrunner(self):
        """With the same orbit, RoadRunner at nq=16, ng=400 is accurate to well below 1 ppm."""
        tb = BFModel('quadratic', orbit='taylor')
        rr = RoadRunnerModel('quadratic', nq=16, ng=400)
        for tm in (tb, rr):
            tm.set_data(self.time)
        assert abs(tb.evaluate(0.1, LDC, **self.orbit) - rr.evaluate(0.1, LDC, **self.orbit)).max() < 1e-6

    def test_secondary_eclipse_is_not_a_transit(self):
        tm = BFModel('quadratic')
        tm.set_data(self.time + 2.0)          # Centred on the secondary eclipse
        assert (tm.evaluate(0.1, LDC, **self.orbit) == 1.0).all()

    def test_supersampling_averages_the_samples(self):
        tm = BFModel('quadratic')
        tm.set_data(self.time, nsamples=[3], exptimes=[0.01])
        flux = tm.evaluate(0.1, LDC, **self.orbit)
        single = BFModel('quadratic')
        samples = []
        for offset in (-0.01 / 3, 0.0, 0.01 / 3):
            single.set_data(self.time + offset)
            samples.append(single.evaluate(0.1, LDC, **self.orbit))
        assert abs(flux - mean(samples, 0)).max() < 1e-12

    def test_passbands_use_their_own_radius_ratios_and_coefficients(self):
        n = self.time.size
        lcids = (linspace(0, 1, n) > 0.5).astype(int)
        tm = BFModel('quadratic')
        tm.set_data(self.time, lcids=lcids, pbids=[0, 1])
        flux = tm.evaluate([0.1, 0.12], [[0.4, 0.25], [0.2, 0.1]], **self.orbit)
        assert isfinite(flux).all()
        for ipb, (k, ldc) in enumerate([(0.1, [0.4, 0.25]), (0.12, [0.2, 0.1])]):
            single = BFModel('quadratic')
            single.set_data(self.time[lcids == ipb])
            assert abs(flux[lcids == ipb] - single.evaluate(k, ldc, **self.orbit)).max() < 1e-12


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        BFModel('no-such-law')
    with pytest.raises(ValueError):
        BFModel('quadratic', orbit='keplerian')
