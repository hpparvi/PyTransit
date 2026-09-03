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

import pytest
from numpy import pi, linspace, array, abs, zeros, full, isfinite, allclose

from meepmeep.backends.numba.point2d import solve2d, pos_c

from pytransit import RoadRunnerModel
from pytransit.models.roadrunner.common import radius_ratio_array

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


class TestRadiusRatioArray:
    """The ``(npv, nk)`` normalisation of the radius ratios shared by the Numba and OpenCL models.

    A one-dimensional radius ratio array is ambiguous: it is either a set of passband-dependent
    radius ratios for a single parameter vector, or one radius ratio for each vector of a
    population. The normalisation resolves this from the number of parameter vectors, following
    the convention already used for the limb darkening coefficients.
    """

    def test_scalar(self):
        assert radius_ratio_array(0.1, 1).shape == (1, 1)

    def test_scalar_broadcasts_over_a_population(self):
        k = radius_ratio_array(0.1, 4)
        assert k.shape == (4, 1)
        assert (k == 0.1).all()

    def test_1d_is_per_passband_for_a_single_parameter_vector(self):
        k = radius_ratio_array([0.1, 0.2, 0.3], 1)
        assert k.shape == (1, 3)
        assert allclose(k[0], [0.1, 0.2, 0.3])

    def test_1d_is_per_parameter_vector_for_a_population(self):
        k = radius_ratio_array([0.1, 0.2, 0.3], 3)
        assert k.shape == (3, 1)
        assert allclose(k[:, 0], [0.1, 0.2, 0.3])

    def test_1d_single_element_broadcasts_over_a_population(self):
        k = radius_ratio_array([0.1], 4)
        assert k.shape == (4, 1)
        assert (k == 0.1).all()

    def test_2d_is_passed_through(self):
        k = radius_ratio_array([[0.1, 0.2], [0.3, 0.4]], 2)
        assert k.shape == (2, 2)

    def test_2d_single_row_broadcasts_over_a_population(self):
        k = radius_ratio_array([[0.1, 0.2]], 3)
        assert k.shape == (3, 2)
        assert allclose(k, [[0.1, 0.2]] * 3)

    def test_mismatched_leading_dimension_raises(self):
        with pytest.raises(ValueError):
            radius_ratio_array([0.1, 0.2, 0.3], 5)
        with pytest.raises(ValueError):
            radius_ratio_array([[0.1], [0.2]], 5)

    def test_too_many_dimensions_raises(self):
        with pytest.raises(ValueError):
            radius_ratio_array(zeros((2, 2, 2)), 2)

    def test_result_is_writeable(self):
        """Numba types a read-only array as a distinct type, which would force a recompilation."""
        assert radius_ratio_array([0.1], 4).flags.writeable
        assert radius_ratio_array([[0.1, 0.2]], 3).flags.writeable


class TestRadiusRatioShapes:
    """A 1D radius ratio vector must give one light curve per parameter vector."""
    npv = 4
    times = linspace(-0.05, 0.05, 100)
    ks = linspace(0.08, 0.14, npv)

    def _population(self, k):
        tm = RoadRunnerModel()
        tm.set_data(self.times)
        return tm.evaluate(k, full((self.npv, 2), [0.4, 0.3]), zeros(self.npv), full(self.npv, 2.0),
                           full(self.npv, 6.0), full(self.npv, 0.5 * pi))

    def test_1d_radius_ratio_gives_one_light_curve_per_parameter_vector(self):
        flux = self._population(self.ks)
        assert flux.shape == (self.npv, self.times.size)
        depths = 1.0 - flux.min(axis=1)
        # Each parameter vector must use its own radius ratio, so the depths must increase with k.
        assert (depths[1:] > depths[:-1]).all()
        assert allclose(depths, self.ks ** 2, rtol=0.25)

    def test_1d_and_2d_radius_ratios_agree(self):
        assert allclose(self._population(self.ks), self._population(self.ks.reshape((self.npv, 1))))

    def test_scalar_radius_ratio_broadcasts_over_a_population(self):
        flux = self._population(0.1)
        assert flux.shape == (self.npv, self.times.size)
        assert allclose(flux, flux[0])

    def test_mismatched_radius_ratio_count_raises(self):
        with pytest.raises(ValueError):
            self._population(linspace(0.08, 0.14, self.npv + 2))

