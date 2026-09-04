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

Every OpenCL model reimplements `set_data` rather than inheriting `TransitModel.set_data`, so
the defaults have to be kept in step by hand. They were not: all four defaulted the exposure
times to one day where the base class defaults to zero, which made supersampling without an
explicit exposure time spread the samples of one exposure over a whole day and return a transit
several times too shallow. The tests below pin the contract for every OpenCL model at once,
because the next model to reimplement `set_data` will get it wrong in the same way.
"""

import pytest
from numpy import array, linspace, pi, abs as npabs, asarray

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
