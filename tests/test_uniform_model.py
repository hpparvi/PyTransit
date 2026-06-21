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

from pytransit import UniformModel


class TestUniformModel:

    def test_init_transit(self):
        UniformModel()

    def test_init_eclipse(self):
        UniformModel(eclipse=True)

    def test_set_data(self, model_data):
        tm = UniformModel()
        tm.set_data(model_data.time)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids, pbids=model_data.pbids)
        assert tm.npb == 2

        tm = UniformModel()
        tm.set_data(model_data.time)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids)
        assert tm.npb == 1

        tm.set_data(model_data.time, lcids=model_data.lcids, pbids=model_data.pbids)
        assert tm.npb == 2

    def test_evaluate_1(self, model_data):
        tm = UniformModel()
        tm.set_data(model_data.time)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.zero_epochs[0], model_data.periods[0],
                           model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_2(self, model_data):
        tm = UniformModel()
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0, 0], model_data.zero_epochs[0], model_data.periods[0],
                           model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_3(self, model_data):
        tm = UniformModel()
        tm.set_data(model_data.time, model_data.lcids, model_data.pbids)
        flux = tm.evaluate(model_data.radius_ratios[0], model_data.zero_epochs[0], model_data.periods[0],
                           model_data.smas[0], model_data.inclinations[0])
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_ps(self, model_data):
        tm = UniformModel()
        tm.set_data(model_data.time)
        flux = tm.evaluate(0.1, 0.0, 1.0, 3.0, 0.5 * pi)
        assert flux.ndim == 1
        assert flux.size == model_data.time.size

    def test_evaluate_population(self, model_data):
        # Evaluate a population of parameter sets simultaneously through the
        # vectorised evaluate() (which replaced the removed evaluate_pv()). The
        # vector branch is selected because the orbital parameters are arrays.
        tm = UniformModel()
        tm.set_data(model_data.time)

        k  = array([0.12, 0.11])
        t0 = array([0.00, 0.01])
        p  = array([1.0, 0.9])
        a  = array([3.0, 2.9])
        i  = array([0.500 * pi, 0.495 * pi])

        flux = tm.evaluate(k, t0, p, a, i)
        assert flux.ndim == 2
        assert flux.shape == (2, model_data.time.size)
