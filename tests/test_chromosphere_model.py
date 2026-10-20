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

from numpy import pi, linspace, isfinite, array, abs

from pytransit import ChromosphereModel


class TestChromosphereModel:
    """Basic behaviour of the optically thin shell model (Schlawin et al., ApJL 722, L75, 2010).

    The model evaluates the projected star-planet distance through the interpolated
    true-anomaly tables, so these tests also cover the tables being built at all.
    """
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
        """The eccentric branch is what actually exercises the true-anomaly interpolation tables."""
        flux = self._model().evaluate(self.k, self.t0, self.p, self.a, self.i, e=0.3, w=0.4 * pi)
        assert isfinite(flux).all()
        assert flux.min() < 1.0

    def test_supersampling_evaluates(self):
        tm = ChromosphereModel()
        tm.set_data(self.times, nsamples=10, exptimes=0.01)
        flux = tm.evaluate(self.k, self.t0, self.p, self.a, self.i)
        assert isfinite(flux).all()
        assert flux.min() < 1.0
