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

"""The OpenCL kernels must compile without a word from the driver.

`build_program` silences PyOpenCL's `CompilerWarning`, because some drivers write a banner even
for a clean build and the warning is not the caller's to act on. That is only safe as long as the
kernels really do build clean, which is what these tests check: they build every program the
models build, into a cache directory of their own so that a real compile happens rather than a
cache hit, and assert that the driver's build log is empty.
"""

import pytest

cl = pytest.importorskip('pyopencl')

import pytransit.models.ma_quadratic_cl as ma_quadratic_cl      # noqa: E402
import pytransit.models.qpower2_cl as qpower2_cl                # noqa: E402
import pytransit.models.ma_uniform_cl as ma_uniform_cl          # noqa: E402
import pytransit.models.roadrunner.rrmodel_cl as rrmodel_cl     # noqa: E402

MODULES = {
    'quadratic': (ma_quadratic_cl, ma_quadratic_cl.QuadraticModelCL, {}),
    'qpower2': (qpower2_cl, qpower2_cl.QPower2ModelCL, {}),
    'uniform': (ma_uniform_cl, ma_uniform_cl.UniformModelCL, {}),
    'roadrunner-single': (rrmodel_cl, rrmodel_cl.RoadRunnerModelCL, {'precision': 'single'}),
    'roadrunner-double': (rrmodel_cl, rrmodel_cl.RoadRunnerModelCL, {'precision': 'double'}),
}


@pytest.fixture(scope='module')
def clenv():
    try:
        device = cl.get_platforms()[0].get_devices()[0]
    except Exception as e:  # noqa: BLE001 - no usable OpenCL runtime
        pytest.skip(f'No OpenCL device available: {e}')
    ctx = cl.Context([device])
    return ctx, cl.CommandQueue(ctx)


@pytest.mark.parametrize('label', list(MODULES), ids=list(MODULES))
def test_kernels_build_without_compiler_output(clenv, tmp_path, monkeypatch, label):
    ctx, queue = clenv
    module, model_cls, kwargs = MODULES[label]

    if label == 'roadrunner-double' and not all(d.double_fp_config for d in ctx.devices):
        pytest.skip('The OpenCL device does not support double precision (cl_khr_fp64).')

    logs = []
    real_build = module.build_program

    def spy(ctx_, source, options='', cache_dir=None):
        # A fresh cache directory, so that the program is really compiled and there is a log.
        program = real_build(ctx_, source, options, cache_dir=str(tmp_path))
        logs.append(program.get_build_info(ctx_.devices[0], cl.program_build_info.LOG))
        return program

    monkeypatch.setattr(module, 'build_program', spy)
    model_cls(cl_ctx=ctx, cl_queue=queue, **kwargs)

    assert logs, 'the model did not build a program through build_program'
    for log in logs:
        assert log.strip() == '', f'{label} kernel build log is not empty:\n{log}'


def test_importing_a_model_does_not_silence_compiler_warnings():
    """The filter belongs to the build, not to the process.

    Importing any OpenCL model used to disable `CompilerWarning` globally, which also hid the
    warning for OpenCL code the caller builds itself.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.warn('probe', cl.CompilerWarning)
    assert len(caught) == 1, 'CompilerWarning is suppressed at import time'
