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

"""Population evaluation must accept every documented argument form and give the same answer."""

from numpy import pi, linspace, full, tile, ones, zeros, abs, isfinite

from pytransit import RoadRunnerModel

TIMES = linspace(-0.08, 0.08, 300)
LDC = [0.3, 0.1]
NPV = 10


class TestPopulationArgumentForms:
    def _model(self):
        tm = RoadRunnerModel('quadratic', small_planet_limit=0.0)
        tm.set_data(TIMES)
        return tm

    def test_scalar_eccentricity_defaults_match_explicit_arrays(self):
        tm = self._model()
        ks, o = full((NPV, 1), 0.1), ones(NPV)
        explicit = tm.evaluate(ks, tile(LDC, (NPV, 1)), zeros((NPV, 1)), 4 * o, 13 * o, 0.49 * pi * o,
                               e=zeros(NPV), w=zeros(NPV))
        defaults = tm.evaluate(ks, tile(LDC, (NPV, 1)), zeros((NPV, 1)), 4 * o, 13 * o, 0.49 * pi * o)
        assert isfinite(defaults).all()
        assert abs(defaults - explicit).max() == 0.0

    def test_one_dimensional_t0_matches_two_dimensional(self):
        tm = self._model()
        ks, o = full((NPV, 1), 0.1), ones(NPV)
        t0s = linspace(-0.002, 0.002, NPV)
        two_d = tm.evaluate(ks, tile(LDC, (NPV, 1)), t0s.reshape((NPV, 1)), 4 * o, 13 * o, 0.49 * pi * o,
                            e=zeros(NPV), w=zeros(NPV))
        one_d = tm.evaluate(ks, tile(LDC, (NPV, 1)), t0s, 4 * o, 13 * o, 0.49 * pi * o,
                            e=zeros(NPV), w=zeros(NPV))
        assert abs(one_d - two_d).max() == 0.0

    def test_population_rows_match_scalar_evaluations(self):
        """Every row of a population must equal the same parameters evaluated alone."""
        tm = self._model()
        ks = linspace(0.08, 0.12, NPV)
        t0s = linspace(-0.002, 0.002, NPV)
        o = ones(NPV)
        batch = tm.evaluate(ks.reshape((NPV, 1)), tile(LDC, (NPV, 1)), t0s, 4 * o, 13 * o, 0.49 * pi * o)
        for j in range(NPV):
            single = tm.evaluate(ks[j], LDC, t0s[j], 4.0, 13.0, 0.49 * pi)
            assert abs(batch[j] - single).max() < 1e-12


class TestOtherModelsShareTheFix:
    def test_transmission_spectroscopy_model(self):
        from pytransit import TSModel
        tm = TSModel('quadratic', small_planet_limit=0.0)
        tm.set_data(TIMES)
        npb, o = 3, ones(NPV)
        ks = full((NPV, npb), 0.1); ldc = tile(LDC, (NPV, npb, 1))
        explicit = tm.evaluate(ks, ldc, zeros(NPV), 4 * o, 13 * o, 0.49 * pi * o, e=zeros(NPV), w=zeros(NPV))
        defaults = tm.evaluate(ks, ldc, zeros(NPV), 4 * o, 13 * o, 0.49 * pi * o)
        assert isfinite(defaults).all()
        assert abs(defaults - explicit).max() == 0.0

    def test_oblate_planet_model(self):
        from pytransit import OblatePlanetModel
        tm = OblatePlanetModel('quadratic', small_planet_limit=0.0)
        tm.set_data(TIMES)
        ks, o = full((NPV, 1), 0.1), ones(NPV)
        explicit = tm.evaluate(ks, 0.05 * o, 0.3 * o, tile(LDC, (NPV, 1)), zeros(NPV), 4 * o, 13 * o, 0.49 * pi * o,
                               e=zeros(NPV), w=zeros(NPV))
        defaults = tm.evaluate(ks, 0.05 * o, 0.3 * o, tile(LDC, (NPV, 1)), zeros(NPV), 4 * o, 13 * o, 0.49 * pi * o)
        assert isfinite(defaults).all()
        assert abs(defaults - explicit).max() == 0.0
