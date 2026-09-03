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

"""Tests for the OpenCL RoadRunner model against the Numba model.

The OpenCL model computes in single precision, and its orbit solver is a separate single
precision implementation, so the fluxes are compared at the ppm level. The mean intensity
tables, which are what the port is about, are compared node by node at single precision.
"""

import pytest
from numpy import (arccos, array, isnan, linspace, pi, repeat, tile, abs as npabs, isfinite, float32, empty,
                   zeros, allclose, array_equal)
from numpy.random import default_rng

from pytransit import RoadRunnerModel
from pytransit.models.roadrunner.common import g_nodes, ldm_nodes, ldm_table, ldm_lookup
from pytransit.models.limb_darkening import evaluate_ld, ld_power_2

cl = pytest.importorskip('pyopencl')

from pytransit.models.roadrunner.rrmodel_cl import RoadRunnerModelCL  # noqa: E402


@pytest.fixture(scope='module')
def clenv():
    try:
        device = cl.get_platforms()[0].get_devices()[0]
    except Exception as e:  # noqa: BLE001 - no usable OpenCL runtime
        pytest.skip(f'No OpenCL device available: {e}')
    ctx = cl.Context([device])
    return ctx, cl.CommandQueue(ctx)


@pytest.fixture(scope='module')
def time():
    return linspace(-0.12, 0.12, 1000)


def models(clenv, time, ldmodel='power-2', **kwargs):
    ctx, queue = clenv
    tm = RoadRunnerModel(ldmodel, **kwargs)
    tm.set_data(time)
    tc = RoadRunnerModelCL(ldmodel, cl_ctx=ctx, cl_queue=queue, **kwargs)
    tc.set_data(time)
    return tm, tc


class TestInit:

    def test_init(self, clenv):
        ctx, queue = clenv
        tm = RoadRunnerModelCL(cl_ctx=ctx, cl_queue=queue)
        assert tm.nq == 8
        assert tm.ng == 100
        tm = RoadRunnerModelCL(cl_ctx=ctx, cl_queue=queue, nq=12, ng=200)
        assert tm.nq == 12
        assert tm.ng == 200

    def test_deprecated_arguments_warn(self, clenv):
        ctx, queue = clenv
        with pytest.warns(FutureWarning):
            RoadRunnerModelCL(cl_ctx=ctx, cl_queue=queue, nz=40)
        with pytest.warns(FutureWarning):
            RoadRunnerModelCL(cl_ctx=ctx, cl_queue=queue, klims=(0.01, 0.2), nk=20)
        tm = RoadRunnerModelCL(cl_ctx=ctx, cl_queue=queue)
        with pytest.warns(FutureWarning):
            tm.init_siwft_arrays(40, 80)
        assert tm.ng == 80


class TestTables:
    """The device-built mean intensity tables against the Numba ones."""

    @pytest.mark.parametrize('k', [0.02, 0.1, 0.3])
    def test_tables_match_numba(self, clenv, time, k):
        tm, tc = models(clenv, time)
        ldc = [0.6, 0.5]
        tc.evaluate(k, ldc, 0.0, 2.0, 4.0, 0.5 * pi)
        gcs, n1s, coef = tc.tables()

        gs, n1 = g_nodes(k, tm.ng)
        nq = tm._rules.shape[2]
        mu = empty((tm.ng, 2 * nq))
        wf = zeros((tm.ng, 2 * nq))
        ldm = empty(tm.ng)
        ldp = evaluate_ld(ld_power_2, tm.mu, array([[ldc]]))[0, 0]
        ldm_nodes(k, gs, tm._rules, mu, wf)
        ldm_table(mu, wf, tm._t0, tm._dt, ldp, ldm)

        assert n1s[0, 0] == n1
        assert gcs[0, 0] == pytest.approx((1 - k) / (1 + k), rel=1e-6)

        # The raw table, and the cubic lookup evaluated between the nodes
        ldm_dev = empty(tc.ng, float32)
        cl.enqueue_copy(tc.queue, ldm_dev, tc._b_ldm)
        tc.queue.finish()
        assert npabs(ldm_dev - ldm).max() < 1e-5

        from pytransit.models.roadrunner.common import split_cubic_coefficients
        coef_ref = zeros((tm.ng - 2, 4))
        split_cubic_coefficients(ldm, n1, coef_ref)
        for g in linspace(0, 1, 2001):
            ref = ldm_lookup(g, gcs[0, 0], n1, coef_ref)
            dev = ldm_lookup(g, float(gcs[0, 0]), int(n1s[0, 0]), coef[0, 0].astype(float))
            assert abs(dev - ref) < 1e-5


class TestFlux:

    @pytest.mark.parametrize('k', [0.02, 0.1, 0.3])
    @pytest.mark.parametrize('b', [0.0, 0.5, 0.8, 1.05])
    def test_single_matches_numba(self, clenv, time, k, b):
        tm, tc = models(clenv, time)
        a = 4.0
        i = arccos(b / a)
        fn = tm.evaluate(k, [0.6, 0.5], 0.0, 2.0, a, i)
        fc = tc.evaluate(k, [0.6, 0.5], 0.0, 2.0, a, i)
        assert fc.shape == fn.shape
        assert isfinite(fc).all()
        # Single precision arithmetic and orbit solver: ppm-level agreement.
        assert npabs(fc - fn).max() < 1.5e-5

    @pytest.mark.parametrize('ldmodel', ['uniform', 'linear', 'quadratic', 'nonlinear', 'power-2'])
    def test_limb_darkening_models(self, clenv, time, ldmodel):
        tm, tc = models(clenv, time, ldmodel)
        ldc = {'uniform': [], 'linear': [0.5], 'quadratic': [0.4, 0.3],
               'nonlinear': [0.5, 0.3, 0.2, 0.1], 'power-2': [0.6, 0.5]}[ldmodel]
        fn = tm.evaluate(0.1, ldc, 0.0, 2.0, 4.0, 0.5 * pi)
        fc = tc.evaluate(0.1, ldc, 0.0, 2.0, 4.0, 0.5 * pi)
        assert npabs(fc - fn).max() < 1e-5

    def test_eccentric_orbit(self, clenv, time):
        tm, tc = models(clenv, time)
        fn = tm.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.49 * pi, 0.3, 0.7)
        fc = tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.49 * pi, 0.3, 0.7)
        assert npabs(fc - fn).max() < 1e-5

    def test_supersampling(self, clenv, time):
        ctx, queue = clenv
        tm = RoadRunnerModel('power-2')
        tm.set_data(time, nsamples=10, exptimes=0.02)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, nsamples=10, exptimes=0.02)
        fn = tm.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)
        fc = tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)
        assert npabs(fc - fn).max() < 1e-5

    def test_population_with_passband_dependent_radius_ratios(self, clenv, time):
        ctx, queue = clenv
        lcids = repeat([0, 1], time.size // 2)
        pbids = [0, 1]
        tm = RoadRunnerModel('power-2')
        tm.set_data(time, lcids, pbids)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, lcids, pbids)

        npv = 40
        rng = default_rng(1)
        k = rng.uniform(0.08, 0.12, (npv, 2))
        t0 = rng.normal(0.0, 0.002, npv)
        p = tile(2.0, npv)
        a = rng.uniform(3.5, 4.5, npv)
        i = arccos(rng.uniform(0.0, 0.8, npv) / a)
        ldc = tile([0.6, 0.5, 0.4, 0.3], (npv, 1))

        fn = tm.evaluate(k, ldc, t0, p, a, i)
        fc = tc.evaluate(k, ldc, t0, p, a, i)
        assert fc.shape == (npv, time.size)
        assert isfinite(fc).all()
        # The impact parameters reach 0.8, where the single precision orbit solver's error in
        # the projected distance shows most in the flux.
        assert npabs(fc - fn).max() < 3e-5

        # Each passband against a single-passband model with that passband's radius ratio
        half = time.size // 2
        for ipb, sl in enumerate((slice(0, half), slice(half, None))):
            tm1 = RoadRunnerModel('power-2')
            tm1.set_data(time[sl])
            f1 = tm1.evaluate(k[0, ipb], ldc[0, 2 * ipb:2 * ipb + 2], t0[0], p[0], a[0], i[0])
            assert npabs(fc[0, sl] - f1).max() < 3e-5

    def test_shared_radius_ratio_population(self, clenv, time):
        ctx, queue = clenv
        lcids = repeat([0, 1], time.size // 2)
        tm = RoadRunnerModel('power-2')
        tm.set_data(time, lcids, [0, 1])
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, lcids, [0, 1])
        npv = 5
        k = tile(0.1, (npv, 1))
        ldc = tile([0.6, 0.5, 0.4, 0.3], (npv, 1))
        fn = tm.evaluate(k, ldc, zeros(npv), tile(2.0, npv), tile(4.0, npv), tile(0.5 * pi, npv))
        fc = tc.evaluate(k, ldc, zeros(npv), tile(2.0, npv), tile(4.0, npv), tile(0.5 * pi, npv))
        assert npabs(fc - fn).max() < 1e-5

    def test_invalid_radius_ratio_shape_raises(self, clenv, time):
        ctx, queue = clenv
        lcids = repeat([0, 1, 2], time.size // 3 + 1)[:time.size]
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, lcids, [0, 1, 2])
        with pytest.raises(ValueError):
            tc.evaluate(tile(0.1, (2, 2)), tile([0.6, 0.5], (2, 3)), zeros(2), tile(2.0, 2), tile(4.0, 2),
                        tile(0.5 * pi, 2))

    def test_invalid_parameters_give_nan(self, clenv, time):
        tm, tc = models(clenv, time)
        assert isnan(tc.evaluate(float('nan'), [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)).all()
        assert isnan(tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 0.9, 0.5 * pi)).all()
        assert isnan(tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi, -0.1, 0.0)).all()
        for k in (0.0, -0.01, 1.5):
            assert isnan(tc.evaluate(k, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)).all()
        # A NaN in one parameter vector must not leak into the others
        npv = 3
        k = tile(0.1, (npv, 1))
        k[1] = float('nan')
        f = tc.evaluate(k, tile([0.6, 0.5], (npv, 1)), zeros(npv), tile(2.0, npv), tile(4.0, npv),
                        tile(0.5 * pi, npv))
        assert isnan(f[1]).all()
        assert isfinite(f[[0, 2]]).all()

    def test_copy_false_returns_none(self, clenv, time):
        tm, tc = models(clenv, time)
        assert tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi, copy=False) is None

    def test_reinit_integration(self, clenv, time):
        tm, tc = models(clenv, time)
        f1 = tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)
        tc.init_integration(12, 200)
        tm.init_integration(12, 200)
        f2 = tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)
        fn = tm.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)
        assert npabs(f2 - fn).max() < 1e-5
        assert npabs(f1 - f2).max() < 1e-5


class TestRadiusRatioShapes:
    """The radius ratio normalisation must agree with the Numba model.

    A one-dimensional radius ratio array used to be read as ``(1, npv)`` by both models, which
    made the Numba model silently return a single light curve and the OpenCL model raise an
    ``IndexError``. Both now read it as one radius ratio per parameter vector.
    """
    npv = 4

    def _population(self, tm, k, time):
        return tm.evaluate(k, tile([0.6, 0.5], (self.npv, 1)), zeros(self.npv), tile(2.0, self.npv),
                           tile(4.0, self.npv), tile(0.5 * pi, self.npv))

    def test_1d_radius_ratio_matches_numba(self, clenv, time):
        tm, tc = models(clenv, time)
        k = linspace(0.08, 0.12, self.npv)
        fn = self._population(tm, k, time)
        fc = self._population(tc, k, time)
        assert fc.shape == (self.npv, time.size) == fn.shape
        assert npabs(fc - fn).max() < 1e-5
        # Each parameter vector must use its own radius ratio.
        depths = 1.0 - fc.min(axis=1)
        assert (depths[1:] > depths[:-1]).all()

    def test_1d_and_2d_radius_ratios_agree(self, clenv, time):
        tm, tc = models(clenv, time)
        k = linspace(0.08, 0.12, self.npv)
        assert allclose(self._population(tc, k, time), self._population(tc, k.reshape((self.npv, 1)), time))

    def test_mismatched_radius_ratio_count_raises(self, clenv, time):
        tm, tc = models(clenv, time)
        k = linspace(0.08, 0.12, self.npv + 2)
        with pytest.raises(ValueError):
            self._population(tc, k, time)
        with pytest.raises(ValueError):
            self._population(tm, k, time)


class TestKernelArgumentCaching:
    """The cached kernel arguments must be invalidated whenever a device buffer is reallocated.

    The kernels are bound once and their arguments set only when they change, so that PyOpenCL
    does not re-marshal every argument on each launch. A buffer that is reallocated without
    invalidating the cached arguments would leave a kernel reading a freed buffer, which gives
    silently wrong fluxes rather than an error, so every reallocation path is exercised here.
    """
    args = (0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)

    def _population(self, tm, npv):
        return tm.evaluate(tile(0.1, (npv, 1)), tile([0.6, 0.5], (npv, 1)), zeros(npv), tile(2.0, npv),
                           tile(4.0, npv), tile(0.5 * pi, npv))

    def test_population_size_changes(self, clenv, time):
        """`_allocate` releases and recreates the per-population buffers."""
        tm, tc = models(clenv, time)
        for npv in (1, 5, 12, 3, 1):
            fc = self._population(tc, npv)
            fn = self._population(tm, npv)
            assert fc.shape == fn.shape
            assert npabs(fc - fn).max() < 1e-5

    def test_set_data_changes(self, clenv, time):
        """`set_data` releases and recreates the time and light curve index buffers."""
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        for npt in (500, 1300, 700):
            t = linspace(-0.12, 0.12, npt)
            tm = RoadRunnerModel('power-2')
            tm.set_data(t)
            tc.set_data(t)
            fc, fn = self._population(tc, 3), self._population(tm, 3)
            assert fc.shape == (3, npt)
            assert npabs(fc - fn).max() < 1e-5

    def test_radius_ratio_count_changes(self, clenv, time):
        """The parameter vector buffer is reallocated when the number of radius ratios changes."""
        ctx, queue = clenv
        lcids = repeat([0, 1], time.size // 2)
        tm = RoadRunnerModel('power-2')
        tm.set_data(time, lcids, [0, 1])
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, lcids, [0, 1])
        npv = 3
        ldc = tile([0.6, 0.5, 0.4, 0.3], (npv, 1))
        orbit = (zeros(npv), tile(2.0, npv), tile(4.0, npv), tile(0.5 * pi, npv))
        for k in (tile(0.1, (npv, 1)), tile([0.1, 0.11], (npv, 1)), tile(0.1, (npv, 1))):
            fc = tc.evaluate(k, ldc, *orbit)
            fn = tm.evaluate(k, ldc, *orbit)
            assert npabs(fc - fn).max() < 3e-5

    def test_init_integration_changes(self, clenv, time):
        """`init_integration` releases and recreates the quadrature rule buffers."""
        tm, tc = models(clenv, time)
        for nq, ng in ((8, 100), (12, 200), (6, 80)):
            tc.init_integration(nq, ng)
            tm.init_integration(nq, ng)
            fc, fn = self._population(tc, 3), self._population(tm, 3)
            assert npabs(fc - fn).max() < 1e-5

    def test_reallocation_invalidates_the_cached_arguments(self, clenv, time):
        """Every reallocation entry point must clear the cached-argument flag.

        `set_data` and `init_integration` also force a reallocation by resetting `npv`, and
        `_allocate` clears the flag itself, so their own invalidation is redundant as long as
        that coupling holds. These assertions guard the explicit invalidation directly, so that
        removing it is caught here rather than becoming a silent staleness bug later.
        """
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)

        tc.set_data(time)
        assert tc._kernel_args_set is False
        tc.evaluate(*self.args)
        assert tc._kernel_args_set is True

        tc.set_data(linspace(-0.1, 0.1, 700))
        assert tc._kernel_args_set is False
        tc.evaluate(*self.args)
        assert tc._kernel_args_set is True

        tc.init_integration(10, 120)
        assert tc._kernel_args_set is False
        tc.evaluate(*self.args)
        assert tc._kernel_args_set is True

        tc._allocate(4)
        assert tc._kernel_args_set is False

    def test_repeated_evaluations_are_stable(self, clenv, time):
        """Identical calls must give bit-identical fluxes."""
        tm, tc = models(clenv, time)
        first = tc.evaluate(*self.args).copy()
        for _ in range(20):
            assert array_equal(tc.evaluate(*self.args), first)

    def test_interleaved_reallocations(self, clenv, time):
        """Reallocation paths interleaved in one model instance must not leave stale arguments."""
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        for npt, npv, nq, ng in ((600, 1, 8, 100), (600, 7, 8, 100), (1100, 7, 8, 100),
                                 (1100, 2, 12, 160), (600, 7, 12, 160), (600, 1, 8, 100)):
            t = linspace(-0.12, 0.12, npt)
            tc.set_data(t)
            tc.init_integration(nq, ng)
            tm = RoadRunnerModel('power-2', nq=nq, ng=ng)
            tm.set_data(t)
            fc, fn = self._population(tc, npv), self._population(tm, npv)
            assert fc.shape == fn.shape
            assert npabs(fc - fn).max() < 1e-5

