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

"""OpenCL context, queue, and program management.

The context and queue are lazily-created module-level singletons shared by
every OpenCL model, so a script building several models does not end up with
several contexts and several copies of the same compiled program. Both can be
overridden per model with the ``cl_ctx`` / ``cl_queue`` constructor arguments.
"""

from importlib.resources import files

import pyopencl as cl
from numpy import float32, float64

__all__ = ['get_context', 'get_queue', 'build_program', 'read_kernel_source',
           'dtype_for_precision', 'build_options']

_CONTEXT = None
_QUEUES: dict = {}
_PROGRAMS: dict = {}


def get_context(cl_ctx=None) -> cl.Context:
    """Return the shared OpenCL context, or `cl_ctx` if one is given."""
    global _CONTEXT
    if cl_ctx is not None:
        return cl_ctx
    if _CONTEXT is None:
        try:
            _CONTEXT = cl.create_some_context(interactive=False)
        except cl.Error as error:
            # pyopencl imports fine without a driver; the failure only surfaces here.
            raise RuntimeError(
                "Could not create an OpenCL context. The OpenCL backend needs a working "
                "OpenCL driver (ICD) for at least one device; install one, or use "
                "backend='numba'.") from error
    return _CONTEXT


def get_queue(ctx: cl.Context, cl_queue=None) -> cl.CommandQueue:
    """Return the shared command queue for `ctx`, or `cl_queue` if one is given."""
    if cl_queue is not None:
        return cl_queue
    queue = _QUEUES.get(ctx)
    if queue is None:
        queue = _QUEUES[ctx] = cl.CommandQueue(ctx)
    return queue


def dtype_for_precision(ctx: cl.Context, precision: str):
    """Map a precision name to a NumPy dtype, checking device support.

    Double precision needs `cl_khr_fp64`. Most consumer GPUs support it but run
    it at a fraction of the single-precision rate, so `precision='single'` is
    offered as an opt-in; see the class docstring in `udmodel.py`.
    """
    if precision == 'double':
        for device in ctx.devices:
            if not device.double_fp_config:
                raise RuntimeError(
                    f"The OpenCL device '{device.name}' does not support double precision "
                    f"(cl_khr_fp64). Use precision='single' instead.")
        return float64
    elif precision == 'single':
        return float32
    else:
        raise ValueError(f"Unknown precision '{precision}', expected 'double' or 'single'.")


def build_options(precision: str) -> str:
    """Preprocessor options selecting the kernel's floating-point type.

    Deliberately does not enable `-cl-fast-relaxed-math`: the OpenCL backend is
    tested for agreement with the Numba reference to ~1e-12, and relaxed math
    breaks that.
    """
    if precision == 'double':
        return '-DREAL=double -DUSE_FP64'
    return '-DREAL=float'


def read_kernel_source(*names: str) -> str:
    """Concatenate the named ``.cl`` files shipped with this package."""
    package = files('pytransit.backends.opencl')
    return '\n'.join(package.joinpath(name).read_text() for name in names)


def build_program(ctx: cl.Context, source: str, options: str) -> cl.Program:
    """Build `source` in `ctx`, caching on (context, source, options)."""
    key = (ctx, source, options)
    program = _PROGRAMS.get(key)
    if program is None:
        program = _PROGRAMS[key] = cl.Program(ctx, source).build(options=options)
    return program
