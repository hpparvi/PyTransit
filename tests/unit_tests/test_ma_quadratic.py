#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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
import unittest

from numpy import linspace, pi, array, zeros, tile
from numpy.random import randint, seed, uniform, normal

from pytransit import QuadraticModel

class TestQuadraticModel(unittest.TestCase):

    def setUp(self) -> None:
        seed(0)
        self.npt = 20
        self.npv = npv = 5
        self.time = linspace(-0.1, 0.1, self.npt)
        self.lcids = randint(0, 3, size=self.npt)
        self.pbids = [0, 1, 1]

        self.ldc = tile([[0.01, 0.3]], (npv, 2))

        self.radius_ratios = uniform(0.09, 0.11, size=(npv, 2))
        self.zero_epochs = normal(0.0, 0.01, size=npv)
        self.periods = normal(1.0, 0.01, size=npv)
        self.smas = normal(3.0, 0.01, size=npv)
        self.inclinations = uniform(0.49 * pi, 0.5 * pi, size=npv)
        self.eccentricities = uniform(0.0, 0.9, size=npv)
        self.omegas = uniform(0, 2*pi, size=npv)

    def test_init(self):
        QuadraticModel()
        QuadraticModel(interpolate=True)
        QuadraticModel(interpolate=False)

    def test_set_data(self):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(self.time)
        assert tm.npb == 1

        tm.set_data(self.time, lcids=self.lcids)
        assert tm.npb == 1

        tm.set_data(self.time, lcids=self.lcids, pbids=self.pbids)
        assert tm.npb == 2

        tm = QuadraticModel(interpolate=True)
        tm.set_data(self.time)
        assert tm.npb == 1

        tm.set_data(self.time, lcids=self.lcids)
        assert tm.npb == 1

        tm.set_data(self.time, lcids=self.lcids, pbids=self.pbids)
        assert tm.npb == 2

    def test_evaluate_1(self):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(self.time)
        flux = tm.evaluate(self.radius_ratios[0, 0], self.ldc[0,:2], self.zero_epochs[0], self.periods[0], self.smas[0], self.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == self.time.size

        tm = QuadraticModel(interpolate=False)
        tm.set_data(self.time)
        flux = tm.evaluate(self.radius_ratios[0, 0], self.ldc[0,:2], self.zero_epochs[0], self.periods[0], self.smas[0], self.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == self.time.size

    def test_evaluate_2(self):
        tm = QuadraticModel()
        tm.set_data(self.time, self.lcids, self.pbids)
        flux = tm.evaluate(self.radius_ratios[0, 0], self.ldc[0], self.zero_epochs[0], self.periods[0], self.smas[0], self.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == self.time.size

    def test_evaluate_3(self):
        tm = QuadraticModel()
        tm.set_data(self.time, self.lcids, self.pbids)
        flux = tm.evaluate(self.radius_ratios[0], self.ldc[0], self.zero_epochs[0], self.periods[0], self.smas[0], self.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == self.time.size

    def test_evaluate_ps(self):
        tm = QuadraticModel()
        tm.set_data(self.time)
        flux = tm.evaluate(0.1, [0.2, 0.3], 0.0, 1.0, 3.0, 0.5*pi)
        assert flux.ndim == 1
        assert flux.size == self.time.size

    def test_evaluate_pv(self):
        tm = QuadraticModel()
        tm.set_data(self.time)

        pvp = array([[0.12, 0.00, 1.0, 3.0, 0.500*pi, 0.0, 0.0],
                     [0.11, 0.01, 0.9, 2.9, 0.495*pi, 0.0, 0.0]])

        ldc = [[0.1, 0.2],[0.3, 0.1]]
        flux = tm.evaluate_pv(pvp[0], ldc[0])
        assert flux.ndim == 1
        assert flux.size == self.time.size

        ldc = [[0.1, 0.2],[0.3, 0.1]]
        flux = tm.evaluate_pv(pvp, ldc)
        assert flux.ndim == 2
        assert flux.shape == (2, self.time.size)

    # TODO: Set up OpenCL in Travis
    # -----------------------------
    #def test_to_opencl(self):
    #    tm = QuadraticModel()
    #    tm.set_data(self.time)
    #    tm2 = tm.to_opencl()
    #    assert isinstance(tm2, QuadraticModelCL)

if __name__ == '__main__':
    unittest.main()