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

from numpy import pi, array

from pytransit import QuadraticModel


class TestQuadraticModel:

    def test_init(self):
        QuadraticModel()
        QuadraticModel(interpolate=True)
        QuadraticModel(interpolate=False)

    def test_set_data(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids, pbids=model_data.pbids)
        assert tm.npb == 2

        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids, pbids=model_data.pbids)
        assert tm.npb == 2

    def test_evaluate_1i(self, model_data):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.ldc[0, :2], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_1d(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.ldc[0, :2], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_2i(self, model_data):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.ldc[0], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_2d(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.ldc[0], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_3i(self, model_data):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0], model_data.ldc[0], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_3d(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0], model_data.ldc[0], model_data.zero_epochs[0],
                           model_data.periods[0], model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_psi(self, model_data):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time)
        flux = tm.evaluate(0.1, [0.2, 0.3], 0.0, 1.0, 3.0, 0.5 * pi)
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_psd(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time)
        flux = tm.evaluate(0.1, [0.2, 0.3], 0.0, 1.0, 3.0, 0.5 * pi)
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_pvi(self, model_data):
        tm = QuadraticModel(interpolate=True)
        tm.set_data(model_data.time)

        pvp = array([[0.12, 0.00, 1.0, 3.0, 0.500 * pi, 0.0, 0.0],
                     [0.11, 0.01, 0.9, 2.9, 0.495 * pi, 0.0, 0.0]])

        ldc = [[0.1, 0.2], [0.3, 0.1]]
        flux = tm.evaluate_pv(pvp[0], ldc[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

        ldc = [[0.1, 0.2], [0.3, 0.1]]
        flux = tm.evaluate_pv(pvp, ldc)
        assert flux.ndim == 2
        assert flux.shape == (2, model_data.time.size)

    def test_evaluate_pvd(self, model_data):
        tm = QuadraticModel(interpolate=False)
        tm.set_data(model_data.time)

        pvp = array([[0.12, 0.00, 1.0, 3.0, 0.500 * pi, 0.0, 0.0],
                     [0.11, 0.01, 0.9, 2.9, 0.495 * pi, 0.0, 0.0]])

        ldc = [[0.1, 0.2], [0.3, 0.1]]
        flux = tm.evaluate_pv(pvp[0], ldc[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

        ldc = [[0.1, 0.2], [0.3, 0.1]]
        flux = tm.evaluate_pv(pvp, ldc)
        assert flux.ndim == 2
        assert flux.shape == (2, model_data.time.size)

    # TODO: Set up OpenCL in Travis
    # -----------------------------
    # def test_to_opencl(self):
    #    tm = QuadraticModel()
    #    tm.set_data(self.time)
    #    tm2 = tm.to_opencl()
    #    assert isinstance(tm2, QuadraticModelCL)
