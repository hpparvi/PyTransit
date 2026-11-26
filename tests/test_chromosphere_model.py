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

from math import acos, asin, sin

import pytest
from numba import get_num_threads, set_num_threads
from numpy.random import default_rng
from numpy import pi, linspace, isfinite, array, abs, arccos, isinf, nan, full
from scipy.integrate import quad

from pytransit import ChromosphereModel
from pytransit.models.numba.ma_chromosphere_nb import chromosphere_z_s, chromosphere_z_v
from pytransit.orbits.orbits_py import z_newton_s


def _direct_flux(z, k):
    """The shell transit as an integral over rings, each weighted by its arc inside the planet.

    The shell's brightness is 1 / sqrt(1 - r^2); the substitution r = sin(x) removes the singularity.
    """
    def arc(r):
        if r <= k - z:
            return 2.0 * pi
        return 2.0 * acos(min(max((r * r + z * z - k * k) / (2.0 * r * z), -1.0), 1.0))

    lo, hi = max(z - k, 0.0), min(z + k, 1.0)
    if hi <= lo:
        return 1.0
    points = [asin(k - z)] if lo < k - z < hi else None
    blocked = quad(lambda x: arc(sin(x)) * sin(x), asin(lo), asin(hi), epsabs=1e-15, epsrel=1e-14, limit=1000,
                   points=points)[0]
    return 1.0 - blocked / (2.0 * pi)


class TestShellFlux:
    """The analytic shell model against direct integration.

    The looser tolerances are set by the reference integral, whose integrand has a square-root
    endpoint next to the singular points; the model itself agrees with an arbitrary-precision
    integral to rounding there.
    """

    @pytest.mark.parametrize('k', [0.02, 0.1, 0.3, 0.5, 0.7, 0.95])
    def test_matches_direct_integration(self, k):
        for z in linspace(0.0, 1.0 + k, 41):
            assert abs(chromosphere_z_s(z, k) - _direct_flux(z, k)) < 1e-11

    @pytest.mark.parametrize('k', [0.02, 0.1, 0.3, 0.5, 0.7, 0.95])
    @pytest.mark.parametrize('offset', [0.0, 1e-14, -1e-14, 1e-12, -1e-12, 1e-10, -1e-10, 1e-8, -1e-8])
    def test_singular_points(self, k, offset):
        """The planet's edge at the disk centre (z = k) and the limb contact from the inside (z = 1 - k)."""
        for z in (k + offset, 1.0 - k + offset):
            assert abs(chromosphere_z_s(z, k) - _direct_flux(z, k)) < 2e-9

    @pytest.mark.parametrize('k', [1.0, 1.3])
    def test_planet_larger_than_the_star(self, k):
        assert chromosphere_z_s(0.0, k) == 0.0
        assert chromosphere_z_s(k - 1.0, k) == 0.0
        for z in linspace(k - 1.0 + 1e-3, 1.0 + k, 15):
            assert abs(chromosphere_z_s(z, k) - _direct_flux(z, k)) < 2e-9

    def test_unocculted_source(self):
        assert chromosphere_z_s(1.1, 0.1) == 1.0
        assert chromosphere_z_s(1.2, 0.1) == 1.0
        assert chromosphere_z_s(-0.5, 0.1) == 1.0

    def test_array_version_matches_scalar(self):
        zs = linspace(0.0, 1.2, 50)
        assert abs(chromosphere_z_v(zs, 0.1) - array([chromosphere_z_s(z, 0.1) for z in zs])).max() == 0.0


class TestChromosphereModelKernels:
    k, t0, p, a, i = 0.1, 0.0, 4.0, 13.0, arccos(0.3 / 13.0)

    @pytest.mark.parametrize('e,w', [(0.0, 0.0), (0.3, 1.0), (0.6, 4.0)])
    def test_matches_an_exact_orbit(self, e, w):
        times = linspace(-0.12, 0.12, 501)
        tm = ChromosphereModel()
        tm.set_data(times)
        pv = array([self.t0, self.p, self.a, self.i, e, w])
        exact = array([chromosphere_z_s(z_newton_s(t, pv), self.k) for t in times])
        assert abs(tm.evaluate(self.k, self.t0, self.p, self.a, self.i, e, w) - exact).max() < 3e-6

    def test_transits_repeat_with_the_period(self):
        times = linspace(-0.12, 0.12, 201)
        tm = ChromosphereModel()
        tm.set_data(times)
        f0 = tm.evaluate(self.k, self.t0, self.p, self.a, self.i)
        tm.set_data(times + 3 * self.p)
        assert abs(tm.evaluate(self.k, self.t0, self.p, self.a, self.i) - f0).max() < 1e-9

    def test_long_exposures_are_not_clipped(self):
        """An exposure centred outside the transit still collects in-transit flux."""
        times = linspace(-0.4, 0.4, 161)
        nsamples, exptime = 20, 0.3
        tm = ChromosphereModel()
        tm.set_data(times, nsamples=nsamples, exptimes=exptime)
        flux = tm.evaluate(self.k, self.t0, self.p, self.a, self.i)

        offsets = exptime * ((array(range(nsamples)) + 0.5) / nsamples - 0.5)
        tf = ChromosphereModel()
        tf.set_data((times[:, None] + offsets).ravel())
        expected = tf.evaluate(self.k, self.t0, self.p, self.a, self.i).reshape((times.size, nsamples)).mean(1)
        assert abs(flux - expected).max() < 1e-12
        assert (flux < 1.0 - 1e-6).sum() > (tf.evaluate(self.k, self.t0, self.p, self.a, self.i) < 1.0).sum() / nsamples

    @pytest.mark.parametrize('nthreads', [2, 3, 8])
    @pytest.mark.parametrize('npt', [1, 63, 64, 65, 700, 5000, 20011])
    def test_parallel_matches_serial(self, nthreads, npt):
        """Unsorted times over several transits: every point must get the serial model's value,
        whichever thread and block it lands in and whether or not the threads are used."""
        times = default_rng(npt).uniform(-1.0, 9.0, npt)
        current = get_num_threads()
        try:
            fluxes = []
            for nt in (1, nthreads):
                tm = ChromosphereModel(nthreads=nt)
                tm.set_data(times, nsamples=3, exptimes=0.01)
                fluxes.append(tm.evaluate(self.k, self.t0, self.p, self.a, self.i, 0.2, 0.5))
            assert fluxes[0].shape == fluxes[1].shape
            assert abs(fluxes[0] - fluxes[1]).max() < 1e-13
        finally:
            set_num_threads(current)

    def test_parallel_nan_parameters_give_inf(self):
        current = get_num_threads()
        try:
            tm = ChromosphereModel(nthreads=2)
            tm.set_data(linspace(-0.12, 0.12, 5000))
            assert isinf(tm.evaluate(self.k, self.t0, self.p, nan, self.i)).all()
        finally:
            set_num_threads(current)

    def test_population_matches_scalar(self):
        times = linspace(-0.12, 0.12, 300)
        tm = ChromosphereModel()
        tm.set_data(times, nsamples=3, exptimes=0.01)
        ks = array([0.08, 0.1, 0.12])
        flux = tm.evaluate(ks, full(3, self.t0), full(3, self.p), full(3, self.a), full(3, self.i))
        for ipv in range(3):
            assert abs(flux[ipv] - tm.evaluate(ks[ipv], self.t0, self.p, self.a, self.i)).max() == 0.0

    def test_nan_parameters_give_inf(self):
        times = linspace(-0.12, 0.12, 50)
        tm = ChromosphereModel()
        tm.set_data(times)
        assert isinf(tm.evaluate(self.k, self.t0, self.p, nan, self.i)).all()
        assert isinf(tm.evaluate(nan, self.t0, self.p, self.a, self.i)).any()
        flux = tm.evaluate(array([0.1, nan]), full(2, self.t0), full(2, self.p), full(2, self.a), full(2, self.i))
        assert isfinite(flux[0]).all()
        assert isinf(flux[1]).all()


class TestChromosphereModel:
    """Basic behaviour of the optically thin shell model (Schlawin et al., ApJL 722, L75, 2010)."""
    k, t0, p, a, i = 0.1, 0.0, 2.0, 6.0, 0.5 * pi
    times = linspace(-0.15, 0.15, 200)

    def _model(self):
        tm = ChromosphereModel()
        tm.set_data(self.times)
        return tm

    def test_evaluate_returns_finite_flux(self):
        flux = self._model().evaluate(self.k, self.t0, self.p, self.a, self.i)
        assert flux.shape == self.times.shape
        assert isfinite(flux).all()

    def test_evaluate_ps_returns_finite_flux(self):
        flux = self._model().evaluate_ps(self.k, self.t0, self.p, self.a, self.i)
        assert flux.shape == self.times.shape
        assert isfinite(flux).all()

    def test_out_of_transit_flux_is_unity(self):
        flux = self._model().evaluate(self.k, self.t0, self.p, self.a, self.i)
        assert abs(flux[0] - 1.0) < 1e-12
        assert abs(flux[-1] - 1.0) < 1e-12

    def test_transit_is_a_dip(self):
        flux = self._model().evaluate(self.k, self.t0, self.p, self.a, self.i)
        assert flux.min() < 1.0
        assert flux.max() <= 1.0 + 1e-12

    def test_evaluate_ps_matches_evaluate_for_scalar_parameters(self):
        tm = self._model()
        assert abs(tm.evaluate(self.k, self.t0, self.p, self.a, self.i)
                   - tm.evaluate_ps(self.k, self.t0, self.p, self.a, self.i)).max() < 1e-12

    def test_parameter_population_evaluation(self):
        tm = self._model()
        npv = 3
        ks = array([[0.09], [0.10], [0.11]])
        flux = tm.evaluate(ks, array(npv * [self.t0]), array(npv * [self.p]),
                           array(npv * [self.a]), array(npv * [self.i]))
        assert flux.shape == (npv, self.times.size)
        assert isfinite(flux).all()
        # A larger planet blocks more light.
        depths = 1.0 - flux.min(1)
        assert depths[0] < depths[1] < depths[2]

    def test_the_shell_is_limb_brightened(self):
        """An optically thin shell is brightest at the limb, which inverts the transit shape.

        The planet therefore blocks *less* light at mid-transit and *more* near the limb than it
        would crossing a uniform disk of the same total brightness, whose depth is k**2.
        """
        tm = ChromosphereModel()
        times = linspace(-0.06, 0.06, 401)
        tm.set_data(times)
        flux = tm.evaluate(self.k, self.t0, self.p, self.a, self.i)

        uniform_depth = self.k ** 2
        mid_transit_depth = 1.0 - flux[abs(times).argmin()]
        deepest = 1.0 - flux.min()

        assert mid_transit_depth < uniform_depth
        assert deepest > uniform_depth

    def test_eccentric_orbit_evaluates(self):
        flux = self._model().evaluate(self.k, self.t0, self.p, self.a, self.i, e=0.3, w=0.4 * pi)
        assert isfinite(flux).all()
        assert flux.min() < 1.0

    def test_supersampling_evaluates(self):
        tm = ChromosphereModel()
        tm.set_data(self.times, nsamples=10, exptimes=0.01)
        flux = tm.evaluate(self.k, self.t0, self.p, self.a, self.i)
        assert isfinite(flux).all()
        assert flux.min() < 1.0
