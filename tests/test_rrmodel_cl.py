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

import re
from pathlib import Path

import pytest
from numpy import (arccos, array, isnan, linspace, pi, repeat, tile, abs as npabs, isfinite, float32, float64,
                   empty, zeros, allclose, array_equal)
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


@pytest.fixture(scope='module')
def fp64(clenv):
    ctx, _ = clenv
    if not all(d.double_fp_config for d in ctx.devices):
        pytest.skip('The OpenCL device does not support double precision (cl_khr_fp64).')
    return True


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


class TestPrecision:
    """Single and double precision builds of the kernel.

    The kernel's floating point type is a `-DREAL=` build option, so both builds come from the
    same source and the only thing that separates them is the compile-time type. Both backends now
    take the projected distance from the same MeepMeep expansion, so the orbit cancels and the
    only difference left in a double precision build is the mean intensity tables, which agree to
    ~2e-8 regardless of the geometry. The tolerances below are set by that, and they are tight on
    purpose: an orbit that stopped matching would show up here first.
    """
    args = (0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)

    def test_single_is_the_default(self, clenv, time):
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time)
        assert tc.precision == 'single'
        assert tc.dtype is float32
        assert tc.evaluate(*self.args).dtype == float32

    def test_double_returns_float64(self, clenv, time, fp64):
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(time)
        assert tc.dtype is float64
        assert tc.evaluate(*self.args).dtype == float64

    def test_unknown_precision_raises(self, clenv):
        ctx, queue = clenv
        with pytest.raises(ValueError):
            RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='half')

    @pytest.mark.parametrize('ldmodel', ['uniform', 'linear', 'quadratic', 'power-2'])
    def test_double_matches_numba(self, clenv, time, fp64, ldmodel):
        ctx, queue = clenv
        ldc = [0.6, 0.5] if ldmodel in ('quadratic', 'power-2') else [0.6]
        ldc = [] if ldmodel == 'uniform' else ldc
        tm = RoadRunnerModel(ldmodel)
        tm.set_data(time)
        tc = RoadRunnerModelCL(ldmodel, cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(time)
        orbit = (0.1, ldc, 0.0, 2.0, 4.0, 0.5 * pi)
        assert npabs(tc.evaluate(*orbit) - tm.evaluate(*orbit)).max() < 5e-8

    def _errors(self, clenv, t, orbit, ldmodel='power-2'):
        """Maximum deviation from the Numba model for both precisions."""
        ctx, queue = clenv
        tm = RoadRunnerModel(ldmodel)
        tm.set_data(t)
        fn = tm.evaluate(*orbit)
        errs = {}
        for precision in ('single', 'double'):
            tc = RoadRunnerModelCL(ldmodel, cl_ctx=ctx, cl_queue=queue, precision=precision)
            tc.set_data(t)
            errs[precision] = npabs(tc.evaluate(*orbit) - fn).max()
        return errs

    @pytest.mark.parametrize('a,half', [(4.0, 0.12), (20.0, 0.04)])
    def test_double_is_much_closer_to_numba_than_single(self, clenv, fp64, a, half):
        """With the orbit shared, single precision rounding is the only thing left to remove.

        Checked at a short and a long transit: the gain used to depend strongly on the geometry,
        because the Numba model's expansion of the projected distance was not matched on the
        device, and it should not any more.
        """
        errs = self._errors(clenv, linspace(-half, half, 1000),
                            (0.1, [0.6, 0.5], 0.0, 2.0, a, 0.5 * pi))
        assert errs['double'] < errs['single'] / 20.0

    @pytest.mark.parametrize('ldmodel,ldc', [('uniform', []), ('linear', [0.6]),
                                             ('quadratic', [0.6, 0.5]), ('power-2', [0.6, 0.5])])
    def test_double_is_never_worse_than_single(self, clenv, time, fp64, ldmodel, ldc):
        errs = self._errors(clenv, time, (0.1, ldc, 0.0, 2.0, 4.0, 0.5 * pi), ldmodel)
        assert errs['double'] <= errs['single']

    def test_double_population(self, clenv, time, fp64):
        ctx, queue = clenv
        npv = 5
        tm = RoadRunnerModel('power-2')
        tm.set_data(time)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(time)
        k = linspace(0.08, 0.12, npv)
        ldc = tile([0.6, 0.5], (npv, 1))
        orbit = (zeros(npv), tile(2.0, npv), tile(4.0, npv), tile(0.5 * pi, npv))
        fc, fn = tc.evaluate(k, ldc, *orbit), tm.evaluate(k, ldc, *orbit)
        assert fc.shape == (npv, time.size)
        assert npabs(fc - fn).max() < 5e-8

    def test_double_eccentric_orbit(self, clenv, time, fp64):
        ctx, queue = clenv
        tm = RoadRunnerModel('power-2')
        tm.set_data(time)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(time)
        orbit = (0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi, 0.2, 0.4)
        assert npabs(tc.evaluate(*orbit) - tm.evaluate(*orbit)).max() < 5e-8

    @pytest.mark.parametrize('e,w', [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.5, 0.0)])
    def test_eccentric_orbit_solver_is_not_the_bottleneck(self, clenv, fp64, e, w):
        """An eccentric orbit must agree as closely as a circular one.

        Eccentricity enters only through the expansion coefficients, which are solved on the host
        and shared with the Numba model, so it should make no difference at all. It used to: the
        kernel solved Kepler's equation itself, with a convergence threshold that capped the
        projected distance at ~1e-4 R_star and cost up to 89 ppm here.
        """
        ctx, queue = clenv
        t = linspace(-0.02, 0.02, 1000)
        tm = RoadRunnerModel('power-2')
        tm.set_data(t)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(t)
        orbit = (0.1, [0.6, 0.5], 0.0, 2.0, 20.0, 0.5 * pi, e, w)
        assert npabs(tm.evaluate(*orbit) - tc.evaluate(*orbit)).max() < 5e-8

    def test_double_invalid_parameters_give_nan(self, clenv, time, fp64):
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(time)
        assert isnan(tc.evaluate(float('nan'), [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)).all()
        assert isnan(tc.evaluate(0.1, [0.6, 0.5], 0.0, 2.0, 0.9, 0.5 * pi)).all()


class TestKernelSourceIsPrecisionAgnostic:
    """The kernel must contain no single-precision-only spellings.

    Both builds compile the same source with `-DREAL=` set, so a `float` declaration or an
    `f`-suffixed literal that creeps back in would mix types in the double build, and a bare
    literal would silently promote the single build to double arithmetic. None of that shows up
    as a test failure elsewhere, so it is checked at the source level.
    """

    @staticmethod
    def _code_lines():
        source = (Path(__file__).parent.parent / 'pytransit' / 'models' / 'roadrunner' / 'rrmodel.cl').read_text()
        # Strip block comments, which legitimately mention float and single precision.
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
        return source

    def test_no_float_declarations(self):
        assert re.search(r'\bfloat\b', self._code_lines()) is None

    def test_no_single_precision_literals(self):
        assert re.search(r'(?<![\w.])\d*\.?\d+([eE][-+]?\d+)?f\b', self._code_lines()) is None

    def test_no_float_only_constants_or_intrinsics(self):
        code = self._code_lines()
        assert re.search(r'\bM_[A-Z_0-9]+_F\b', code) is None
        assert re.search(r'\b(native|half)_[a-z]+\b', code) is None

    def test_fp64_pragma_is_guarded(self):
        """The guard comes from MeepMeep's `common.cl`, so check the source as the host builds it."""
        from meepmeep.backends.opencl import read_kernel_source
        source = read_kernel_source('point2d.cl') + self._code_lines()
        assert '#ifdef USE_FP64' in source
        assert '#pragma OPENCL EXTENSION cl_khr_fp64 : enable' in source


class TestEpochFolding:
    """Data spanning several epochs, where the expansion is only valid near each transit.

    The projected distance is a Taylor expansion around the transit centre, so away from the
    transit it is a polynomial with no physical meaning and will happily dip below one. The
    kernel must reject those samples by the transit bounding box instead of evaluating it. The
    box is widened by the exposure time, so an exposure time defaulting to one day rather than
    zero stretched it across the far side of a two-day orbit and put a full-depth spurious
    transit there, in both precisions.
    """
    p = 2.0
    orbit = (0.1, [0.6, 0.5], 0.0, 2.0, 4.0, 0.5 * pi)

    def _times(self, nep=3):
        return linspace(-0.12, (nep - 1) * self.p + 0.12, 1500)

    @pytest.mark.parametrize('precision', ['single', 'double'])
    def test_no_spurious_transit_between_epochs(self, clenv, precision):
        ctx, queue = clenv
        t = self._times()
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision=precision)
        tc.set_data(t)
        f = tc.evaluate(*self.orbit)
        # Everything more than a quarter period from a transit centre is out of transit.
        phase = (t + 0.5 * self.p) % self.p - 0.5 * self.p
        assert (f[npabs(phase) > 0.25 * self.p] == 1.0).all()

    def test_multi_epoch_matches_numba(self, clenv, fp64):
        ctx, queue = clenv
        t = self._times()
        tm = RoadRunnerModel('power-2')
        tm.set_data(t)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(t)
        fn, fc = tm.evaluate(*self.orbit), tc.evaluate(*self.orbit)
        assert (fn < 1.0).sum() == (fc < 1.0).sum()
        assert npabs(fn - fc).max() < 5e-8

    def test_supersampled_multi_epoch_matches_numba(self, clenv, fp64):
        """A real exposure time widens the bounding box; the two must still agree."""
        ctx, queue = clenv
        t = self._times()
        tm = RoadRunnerModel('power-2')
        tm.set_data(t, nsamples=10, exptimes=0.02)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue, precision='double')
        tc.set_data(t, nsamples=10, exptimes=0.02)
        assert npabs(tm.evaluate(*self.orbit) - tc.evaluate(*self.orbit)).max() < 5e-8

    def test_scalar_sampling_arguments_are_one_dimensional(self, clenv, time):
        """A scalar would give a zero-dimensional array, which the compiled expansion cannot index."""
        ctx, queue = clenv
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, nsamples=5, exptimes=0.01)
        assert tc.exptimes.ndim == 1
        assert tc.nsamples.ndim == 1

    def test_default_exposure_time_is_zero(self, clenv, time):
        """As in `TransitModel.set_data`; the bounding box depends on it."""
        ctx, queue = clenv
        tm = RoadRunnerModel('power-2')
        tm.set_data(time)
        tc = RoadRunnerModelCL('power-2', cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time)
        assert (tc.exptimes == 0.0).all()
        assert allclose(tc.exptimes, tm.exptimes)

