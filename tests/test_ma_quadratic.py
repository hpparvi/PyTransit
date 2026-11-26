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

import pytest
from numba import get_num_threads, set_num_threads
from numpy import pi, array, arange, arccos, linspace, repeat, isinf, nan, abs
from numpy import zeros
from numpy.random import default_rng

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

    # TODO: Set up OpenCL in Travis
    # -----------------------------
    # def test_to_opencl(self):
    #    tm = QuadraticModel()
    #    tm.set_data(self.time)
    #    tm2 = tm.to_opencl()
    #    assert isinstance(tm2, QuadraticModelCL)


class TestQuadraticModelKernels:
    """The light curve loop around the model: transit window, supersampling, threads and bad parameters."""

    LDC = array([0.3, 0.1])
    P, A = 4.0, 13.0
    INC = arccos(0.3 / 13.0)

    @pytest.mark.parametrize("exptime", [0.02, 0.2])
    def test_supersampling_equals_the_average_over_the_exposure(self, exptime):
        # An exposure can reach into the transit from far outside it when it is long, so the
        # window in which the model is evaluated has to grow with the exposure time.
        ns = 50
        time = linspace(-0.3, 0.3, 61)
        offsets = exptime * ((arange(ns) + 0.5) / ns - 0.5)
        tm = QuadraticModel()
        tm.set_data((time[:, None] + offsets[None, :]).ravel())
        reference = tm.evaluate(0.1, self.LDC, 0.0, self.P, self.A, self.INC).reshape(time.size, ns).mean(1)
        tm.set_data(time, nsamples=ns, exptimes=exptime)
        assert abs(tm.evaluate(0.1, self.LDC, 0.0, self.P, self.A, self.INC) - reference).max() < 1e-12

    def test_the_largest_radius_ratio_sets_the_transit_window(self):
        # The second passband has the larger object, and its transit starts 0.06 d earlier.
        p, a, inc = 100.0, 80.0, 0.5 * pi
        time = linspace(-0.6, 0.6, 400)
        ks = array([0.05, 0.4])
        ldc = array([[0.3, 0.1], [0.3, 0.1]])
        tm = QuadraticModel()
        tm.set_data(time)
        single = tm.evaluate(ks[1], self.LDC, 0.0, p, a, inc)
        tm.set_data(time, lcids=repeat([0, 1], 200), pbids=[0, 1])
        both = tm.evaluate(ks, ldc, 0.0, p, a, inc)
        assert abs(both[200:] - single[200:]).max() < 1e-12
        assert (single[200:] < 1.0).sum() > 50

    @pytest.mark.parametrize("supersampling", [{}, dict(nsamples=5, exptimes=0.01)])
    def test_parallel_model_equals_the_serial_model(self, supersampling):
        nthreads = get_num_threads()
        try:
            time = linspace(-0.2, 0.2, 5000)
            serial, parallel = QuadraticModel(), QuadraticModel(nthreads=2)
            assert parallel.parallel and not serial.parallel
            serial.set_data(time, **supersampling)
            parallel.set_data(time, **supersampling)
            pv = (0.1, self.LDC, 0.0, self.P, self.A, self.INC)
            assert abs(parallel.evaluate(*pv) - serial.evaluate(*pv)).max() < 1e-13
        finally:
            set_num_threads(nthreads)

    @pytest.mark.parametrize("nthreads", [2, 3, 8])
    @pytest.mark.parametrize("npt", [1, 63, 64, 65, 700, 5000, 20011])
    @pytest.mark.parametrize("interpolate", [False, True])
    def test_interleaved_blocks_cover_every_point(self, nthreads, npt, interpolate):
        # Unsorted times over several transits, two light curves with their own passband,
        # radius ratio and supersampling: every point must get the serial model's value,
        # whichever thread and block it lands in and whether or not the threads are used.
        rng = default_rng(npt)
        time = rng.uniform(-1.0, 9.0, npt)
        lcids = (arange(npt) % 3 == 0).astype(int) if npt > 1 else zeros(1, int)
        setup = dict(lcids=lcids, pbids=[0, 1][:lcids.max() + 1], nsamples=[1, 5][:lcids.max() + 1],
                     exptimes=[0.0, 0.02][:lcids.max() + 1])
        k = array([0.1, 0.12])[:lcids.max() + 1]
        ldc = array([0.3, 0.1, 0.5, 0.2])[:2 * (lcids.max() + 1)]
        current = get_num_threads()
        try:
            serial = QuadraticModel(interpolate=interpolate)
            parallel = QuadraticModel(interpolate=interpolate, nthreads=nthreads)
            serial.set_data(time, **setup)
            parallel.set_data(time, **setup)
            pv = (k, ldc, 0.0, self.P, self.A, self.INC)
            flux = parallel.evaluate(*pv)
            assert flux.shape == serial.evaluate(*pv).shape
            assert abs(flux - serial.evaluate(*pv)).max() < 1e-13
        finally:
            set_num_threads(current)

    def test_parallel_model_gives_infinite_flux_for_bad_parameters(self):
        current = get_num_threads()
        try:
            time = linspace(-0.2, 0.2, 5000)
            tm = QuadraticModel(nthreads=2)
            tm.set_data(time)
            assert isinf(tm.evaluate(0.1, self.LDC, 0.0, self.P, nan, self.INC)).all()
            tm = QuadraticModel(interpolate=True, klims=(0.05, 0.15), nthreads=2)
            tm.set_data(time)
            assert isinf(tm.evaluate(0.2, self.LDC, 0.0, self.P, self.A, self.INC)).all()
        finally:
            set_num_threads(current)

    @pytest.mark.parametrize("name", ["k", "ldc", "p", "a", "i"])
    def test_nan_parameter_gives_infinite_flux_in_transit(self, name):
        pv = dict(k=0.1, ldc=self.LDC.copy(), t0=0.0, p=self.P, a=self.A, i=self.INC)
        pv[name] = array([nan, 0.1]) if name == 'ldc' else nan
        tm = QuadraticModel()
        tm.set_data(array([-0.01, 0.0, 0.01]))
        assert isinf(tm.evaluate(**pv)).all()
