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

"""The serial and parallel transmission spectroscopy kernels must give the same light curves."""

import pytest
from numba import config as numba_config
from numpy import pi, linspace, full, tile, ones, zeros, array, stack, empty, isfinite, isnan
from numpy.testing import assert_allclose

from pytransit import RoadRunnerModel, TSModel
from pytransit.models.limb_darkening import evaluate_ld
from pytransit.models.roadrunner.common import g_nodes, ldm_nodes, ldm_table, ldm_table_map, ldm_table_banded

TIMES = linspace(-0.1, 0.1, 500)
NPB = 20
K = linspace(0.095, 0.105, NPB)
LDC = stack([linspace(0.5, 0.2, NPB), linspace(0.2, 0.1, NPB)], axis=1)
ORBIT = dict(t0=0.0, p=2.0, a=6.0, i=0.48 * pi)

needs_threads = pytest.mark.skipif(numba_config.NUMBA_NUM_THREADS < 2,
                                   reason='the parallel kernel needs at least two numba threads')


def _model(nthreads):
    tm = TSModel('quadratic', nthreads=nthreads)
    tm.set_data(TIMES)
    return tm


@needs_threads
class TestParallelKernel:
    def test_uses_the_parallel_kernel(self):
        assert _model(2).parallel

    def test_matches_serial_for_one_parameter_vector(self):
        serial = _model(1).evaluate(K, LDC, **ORBIT)
        parallel = _model(2).evaluate(K, LDC, **ORBIT)
        assert serial.min() < 0.99
        assert_allclose(parallel, serial, rtol=0, atol=1e-12)

    def test_matches_serial_for_a_population(self):
        npv = 5
        o = ones(npv)
        ks = K * linspace(0.9, 1.1, npv)[:, None]
        args = (ks, tile(LDC, (npv, 1, 1)), zeros(npv), 2.0 * o, 6.0 * o, linspace(0.47, 0.5, npv) * pi)
        assert_allclose(_model(2).evaluate(*args), _model(1).evaluate(*args), rtol=0, atol=1e-12)

    def test_matches_serial_with_supersampling(self):
        tm1, tm2 = TSModel('quadratic'), TSModel('quadratic', nthreads=2)
        for tm in (tm1, tm2):
            tm.set_data(TIMES, nsamples=[5], exptimes=[0.01])
        assert_allclose(tm2.evaluate(K, LDC, **ORBIT), tm1.evaluate(K, LDC, **ORBIT), rtol=0, atol=1e-12)


def _roadrunner_per_bin(ks, ldcs):
    rr = RoadRunnerModel('quadratic')
    rr.set_data(TIMES)
    return array([rr.evaluate(k, ldc, **ORBIT) for k, ldc in zip(ks, ldcs)])


@pytest.mark.parametrize('nthreads', [1, pytest.param(2, marks=needs_threads)])
class TestAgainstRoadRunner:
    def test_equal_radius_ratios_reproduce_roadrunner(self, nthreads):
        """With one radius ratio in every bin the shared geometry is exact, so each bin is RoadRunner."""
        flux = _model(nthreads).evaluate(full(NPB, 0.1), LDC, **ORBIT)[0]
        assert_allclose(flux, _roadrunner_per_bin(full(NPB, 0.1), LDC), rtol=0, atol=1e-9)

    def test_shared_geometry_error_for_a_one_percent_spread(self, nthreads):
        """Bins within 1% of the mean radius ratio stay within a few ppm of RoadRunner (6.2 ppm measured)."""
        ks = 0.1 * linspace(0.99, 1.01, NPB)
        flux = _model(nthreads).evaluate(ks, LDC, **ORBIT)[0]
        assert_allclose(flux, _roadrunner_per_bin(ks, LDC), rtol=0, atol=8e-6)

    def test_invalid_parameter_vectors_give_nan(self, nthreads):
        o = ones(2)
        flux = _model(nthreads).evaluate(tile(K, (2, 1)), tile(LDC, (2, 1, 1)), zeros(2), 2.0 * o,
                                         array([6.0, 0.5]), 0.48 * pi * o)
        assert isfinite(flux[0]).all()
        assert isnan(flux[1]).all()


def test_table_map_reproduces_the_mean_intensity_table():
    tm = TSModel('quadratic')
    k, ng = 0.1, tm.ng
    gs, _ = g_nodes(k, ng)
    mu, wf = empty((ng, 2 * tm.nq)), zeros((ng, 2 * tm.nq))
    ldm_nodes(k, gs, tm._rules, mu, wf)

    ldp = evaluate_ld(tm.ldmodel, tm.mu, LDC[None])[0]
    expected = empty((NPB, ng))
    for ipb in range(NPB):
        ldm_table(mu, wf, tm._t0, tm._dt, ldp[ipb], expected[ipb])
    wt, lo, hi = ldm_table_map(mu, wf, tm._t0, tm._dt, ldp.shape[1])
    for ig in range(ng):
        assert (wt[ig, :lo[ig]] == 0).all() and (wt[ig, hi[ig]:] == 0).all()
    for ipb in range(NPB):
        table = empty(ng)
        ldm_table_banded(wt, lo, hi, ldp[ipb], 1.0, table)
        assert_allclose(table, expected[ipb], rtol=1e-13)
