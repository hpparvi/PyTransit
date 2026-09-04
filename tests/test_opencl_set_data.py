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

"""The `set_data` contract shared by the OpenCL models.

Every OpenCL model used to carry its own copy of `set_data` rather than inheriting one, and the
copies had drifted: all four defaulted the exposure times to one day where `TransitModel` defaults
to zero, which made supersampling without an explicit exposure time spread the samples of one
exposure over a whole day and return a transit several times too shallow, and none of them
validated the light curve or passband indices at all. They now share `OpenCLTransitModel.set_data`.
These tests run against every OpenCL model, so a model that reimplements `set_data` again has to
keep the contract.
"""

import pytest
from numpy import array, linspace, pi, abs as npabs, asarray, repeat, zeros, uint32

from pytransit import QuadraticModel, QPower2Model, UniformModel, RoadRunnerModel

cl = pytest.importorskip('pyopencl')

from pytransit.models.ma_quadratic_cl import QuadraticModelCL          # noqa: E402
from pytransit.models.qpower2_cl import QPower2ModelCL                 # noqa: E402
from pytransit.models.ma_uniform_cl import UniformModelCL              # noqa: E402
from pytransit.models.roadrunner.rrmodel_cl import RoadRunnerModelCL   # noqa: E402

# (label, Numba model factory, OpenCL model factory, limb darkening coefficients)
MODELS = [
    ('quadratic', QuadraticModel, QuadraticModelCL, array([0.4, 0.3])),
    ('qpower2', QPower2Model, QPower2ModelCL, array([0.4, 0.3])),
    ('uniform', UniformModel, UniformModelCL, None),
    ('roadrunner', RoadRunnerModel, RoadRunnerModelCL, array([0.4, 0.3])),
]
IDS = [m[0] for m in MODELS]
ORBIT = dict(k=0.1, t0=0.0, p=2.0, a=4.0, i=0.5 * pi)


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
    return linspace(-0.12, 0.12, 500)


def _evaluate(model, ldc):
    kwargs = dict(ORBIT)
    if ldc is not None:
        kwargs['ldc'] = ldc
    return model.evaluate(**kwargs)


@pytest.mark.parametrize('label,numba_cls,cl_cls,ldc', MODELS, ids=IDS)
class TestSetDataDefaults:

    def test_default_exposure_time_is_zero(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time)
        assert (asarray(tc.exptimes) == 0.0).all()

    def test_supersampling_without_an_exposure_time_matches_numba(self, clenv, time, label,
                                                                  numba_cls, cl_cls, ldc):
        """With a zero exposure time every sample of an exposure falls at the same time.

        The result is the unsupersampled model, and it must not differ from the Numba model:
        a default of one day gave a transit up to five times too shallow.
        """
        ctx, queue = clenv
        tm, tc = numba_cls(), cl_cls(cl_ctx=ctx, cl_queue=queue)
        tm.set_data(time, nsamples=10)
        tc.set_data(time, nsamples=10)
        assert npabs(_evaluate(tm, ldc) - _evaluate(tc, ldc)).max() < 1e-4

    def test_scalar_sampling_arguments_are_one_dimensional(self, clenv, time, label,
                                                           numba_cls, cl_cls, ldc):
        """A scalar would give a zero-dimensional array, which cannot be indexed."""
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, nsamples=5, exptimes=0.01)
        assert asarray(tc.exptimes).ndim == 1
        assert asarray(tc.nsamples).ndim == 1

    def test_explicit_supersampling_matches_numba(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tm, tc = numba_cls(), cl_cls(cl_ctx=ctx, cl_queue=queue)
        tm.set_data(time, nsamples=10, exptimes=0.02)
        tc.set_data(time, nsamples=10, exptimes=0.02)
        assert npabs(_evaluate(tm, ldc) - _evaluate(tc, ldc)).max() < 1e-4


@pytest.mark.parametrize('label,numba_cls,cl_cls,ldc', MODELS, ids=IDS)
class TestSharedSetData:
    """Behaviour the OpenCL models get from sharing `TransitModel.set_data`.

    None of them validated the light curve or passband indices while each carried its own copy,
    so a malformed index array gave silently wrong fluxes or an out-of-bounds device read.
    """

    def test_non_integer_lcids_raise(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        with pytest.raises(ValueError):
            tc.set_data(time, lcids=zeros(time.size, float))

    def test_wrong_lcids_size_raises(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        with pytest.raises(ValueError):
            tc.set_data(time, lcids=zeros(time.size - 1, int))

    def test_wrong_pbids_size_raises(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        with pytest.raises(ValueError):
            tc.set_data(time, lcids=repeat([0, 1], time.size // 2), pbids=array([0]))

    def test_non_contiguous_pbids_raise(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        with pytest.raises(ValueError):
            tc.set_data(time, lcids=repeat([0, 1], time.size // 2), pbids=array([1, 2]))

    def test_epids_are_accepted(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, lcids=repeat([0, 1], time.size // 2), pbids=[0, 1], epids=[0, 1])
        assert (asarray(tc.epids) == [0, 1]).all()

    def test_repeated_set_data_with_the_same_array_is_a_noop(self, clenv, time, label,
                                                             numba_cls, cl_cls, ldc):
        """The base class short-circuits, and the device buffers must then be left alone."""
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        assert tc.set_data(time) is True
        buffer = tc._b_time
        assert tc.set_data(time) is False
        assert tc._b_time is buffer
        assert tc.set_data(time.copy()) is True
        assert tc._b_time is not buffer

    def test_scalar_supersampling_applies_to_every_light_curve(self, clenv, time, label,
                                                               numba_cls, cl_cls, ldc):
        """A single sample count and exposure time apply to every light curve, as documented.

        They used to be left as length-one arrays, which every model indexes per light curve and
        so reads past the end of for all but the first: a `ZeroDivisionError` from a garbage
        sample count in the Numba models, and an out-of-bounds device read in the OpenCL ones.
        """
        ctx, queue = clenv
        lcids = repeat([0, 1], time.size // 2)
        tm, tc = numba_cls(), cl_cls(cl_ctx=ctx, cl_queue=queue)
        tm.set_data(time, lcids, [0, 0], nsamples=10, exptimes=0.02)
        tc.set_data(time, lcids, [0, 0], nsamples=10, exptimes=0.02)
        assert tm.nsamples.size == tm.exptimes.size == 2
        assert tc.nsamples.size == tc.exptimes.size == 2

        # And they must give the same fluxes as the same values spelled out per light curve.
        tm2 = numba_cls()
        tm2.set_data(time, lcids, [0, 0], nsamples=[10, 10], exptimes=[0.02, 0.02])
        assert npabs(_evaluate(tm, ldc) - _evaluate(tm2, ldc)).max() == 0.0
        assert npabs(_evaluate(tm, ldc) - _evaluate(tc, ldc)).max() < 1e-4

    def test_mismatched_supersampling_size_raises(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        lcids = repeat([0, 1], time.size // 2)
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        with pytest.raises(ValueError):
            tc.set_data(time, lcids, [0, 0], nsamples=[10, 10, 10])
        with pytest.raises(ValueError):
            tc.set_data(time, lcids, [0, 0], exptimes=[0.02, 0.02, 0.02])

    def test_arrays_carry_the_device_types(self, clenv, time, label, numba_cls, cl_cls, ldc):
        ctx, queue = clenv
        tc = cl_cls(cl_ctx=ctx, cl_queue=queue)
        tc.set_data(time, nsamples=5, exptimes=0.01)
        assert tc.time.dtype == tc.dtype
        assert tc.exptimes.dtype == tc.dtype
        assert tc.lcids.dtype == uint32
        assert tc.pbids.dtype == uint32
        assert tc.nsamples.dtype == uint32
        assert isinstance(tc.nlc, uint32) and isinstance(tc.npb, uint32)
        assert tc.nptb == time.size

