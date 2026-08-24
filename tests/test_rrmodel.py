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

from numpy import pi, linspace, array, abs, zeros, full, isfinite

from meepmeep.backends.numba.point2d import solve2d, pos_c

from pytransit import RoadRunnerModel

from .test_opmodel import reference_flux


class TestRoadRunnerModelAccuracy:
    """Compare the RoadRunner model fluxes against nested adaptive quadrature.

    The reference flux integrates the quadratic limb darkening intensity over the exact
    planet-star overlap with nested adaptive quadrature (validated against the analytic
    Mandel & Agol model in ``tests/test_opmodel.py``). The tolerances document the accuracy
    of the model at its default resolution settings: the error is dominated by the linear
    interpolation of the limb darkening weights over the radius ratio grid and grows with
    the radius ratio (increasing `nk` tightens it).
    """
    u1, u2 = 0.4, 0.3
    p, a = 2.0, 6.0
    times = linspace(0.0, 0.058, 12)

    def _reference(self, k, inc):
        xyc = solve2d(0.0, self.p, self.a, inc, 0.0, 0.0)
        return array([reference_flux(*pos_c(t, xyc), k, 0.0, 0.0, self.u1, self.u2) for t in self.times])

    def _model(self, **kwargs):
        tm = RoadRunnerModel()
        tm.set_data(self.times)
        return tm.evaluate(ldc=[self.u1, self.u2], t0=0.0, p=self.p, a=self.a, **kwargs)

    def test_small_planet(self):
        flux = self._model(k=0.03, i=0.5 * pi)
        assert abs(flux - self._reference(0.03, 0.5 * pi)).max() < 3e-6

    def test_small_planet_approximation(self):
        """At and below the small-planet limit the center-intensity approximation must stay sub-ppm."""
        tm = RoadRunnerModel()
        tm.set_data(self.times)
        flux = tm.evaluate(k=0.01, ldc=[self.u1, self.u2], t0=0.0, p=self.p, a=self.a, i=0.5 * pi)
        assert abs(flux - self._reference(0.01, 0.5 * pi)).max() < 1.5e-6
        # the approximation must activate at the limit: results differ from the disabled case
        tm_off = RoadRunnerModel(small_planet_limit=0.0)
        tm_off.set_data(self.times)
        flux_off = tm_off.evaluate(k=0.01, ldc=[self.u1, self.u2], t0=0.0, p=self.p, a=self.a, i=0.5 * pi)
        assert abs(flux - flux_off).max() > 0.0
        assert abs(flux_off - self._reference(0.01, 0.5 * pi)).max() < 1.5e-6

    def test_scalar_evaluation(self):
        """Scalar parameters follow the single-light-curve rr_simple path."""
        flux = self._model(k=0.1, i=0.5 * pi)
        assert abs(flux - self._reference(0.1, 0.5 * pi)).max() < 1.5e-5

    def test_grazing(self):
        flux = self._model(k=0.1, i=1.41)
        assert abs(flux - self._reference(0.1, 1.41)).max() < 1e-5

    def test_large_planet(self):
        """The radius-ratio grid interpolation error dominates at large k."""
        flux = self._model(k=0.3, i=0.5 * pi)
        assert abs(flux - self._reference(0.3, 0.5 * pi)).max() < 1.5e-4

    def test_vector_evaluation(self):
        """Vector parameters follow the heterogeneous rr_full path."""
        tm = RoadRunnerModel()
        tm.set_data(self.times)
        ldc = array([[self.u1, self.u2], [self.u1, self.u2]]).reshape((2, 1, 2))
        flux = tm.evaluate(k=array([[0.1], [0.2]]), ldc=ldc, t0=zeros((2, 1)),
                           p=full(2, self.p), a=full(2, self.a), i=array([0.5 * pi, 1.45]),
                           e=zeros(2), w=zeros(2))
        assert isfinite(flux).all()
        assert abs(flux[0] - self._reference(0.1, 0.5 * pi)).max() < 1.5e-5
        assert abs(flux[1] - self._reference(0.2, 1.45)).max() < 7e-5
