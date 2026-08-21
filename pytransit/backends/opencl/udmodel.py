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

"""OpenCL host wrapper for the uniform-disk transit model."""

import pyopencl as cl
from numpy import ascontiguousarray, asarray, float64, int32, zeros

from ._host import solve_coefficients, solve_coefficients_d
from .context import (build_options, build_program, dtype_for_precision, get_context,
                      get_queue, read_kernel_source)

__all__ = ['UDModelCL']


class UDModelCL:
    """Uniform-disk transit model evaluated on an OpenCL device.

    Callable with the same signature as `backends.numba.udmodel`, so the model
    class can dispatch to it without special-casing the call itself. The Taylor
    coefficients are solved on the host (see `_host.py`) and only the
    per-datapoint separation, circle-circle intersection, and supersampling loop
    run on the device.

    Parameters
    ----------
    return_grad : bool
        Evaluate the gradient kernel and return ``(flux, dflux)``.
    precision : {'double', 'single'}
        Device floating-point type. Double precision matches the Numba backend
        to ~1e-12 and is the default. Single precision is dramatically faster on
        consumer GPUs, where double-precision throughput is typically a
        sixty-fourth of single, but the kite-area term in the circle-circle
        intersection loses accuracy near the contact points; expect errors of
        order 1e-6 in flux. Note that single precision also changes the *returned*
        dtype to float32, which propagates through the gradient expansion into
        whatever consumes it.
    cl_ctx, cl_queue : optional
        Override the shared context and queue from `context.py`. The queue must be
        in-order: parameter uploads overwrite buffers a previously enqueued kernel
        may still be reading, and `copy=False` deliberately leaves a kernel in
        flight, so out-of-order execution would race.

    Notes
    -----
    Data buffers are created by `set_data` and released and recreated if it is
    called again. Parameter and output buffers are reallocated only when the
    number of parameter vectors changes.

    Times are shifted to a reference epoch on the host before being cast to the
    device dtype; see `set_data`.
    """

    def __init__(self, return_grad: bool = False, precision: str = 'double',
                 cl_ctx=None, cl_queue=None):
        self.return_grad = return_grad
        self.precision = precision

        self.ctx = get_context(cl_ctx)
        self.queue = get_queue(self.ctx, cl_queue)
        if self.queue.properties & cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE:
            raise ValueError("The OpenCL backend requires an in-order command queue, because it "
                             "relies on queue ordering between parameter uploads and kernels.")
        self.dtype = dtype_for_precision(self.ctx, precision)
        self.prg = build_program(self.ctx,
                                 read_kernel_source('common.cl', 'udmodel.cl'),
                                 build_options(precision))

        self.npt: int = 0
        self.nlc: int = 0
        self.npv: int = 0
        self.tref: float = 0.0

        # Device-resident results. These stay valid after an `evaluate(copy=False)`
        # call and are what a device-side likelihood would consume. Note that
        # `_b_dflux` holds the *compact* (npv, npt, 7) gradient; the expansion into
        # per-passband and per-epoch columns is a host-side operation.
        self._b_flux = None
        self._b_dflux = None

        self._b_times = None
        self._b_lcids = None
        self._b_pbids = None
        self._b_epids = None
        self._b_nsamples = None
        self._b_exptimes = None

        self._b_k = None
        self._b_t0 = None
        self._b_p = None
        self._b_xyc = None
        self._b_dxyc = None
        self._b_bt1 = None
        self._b_bt4 = None

    def _release(self, *names) -> None:
        for name in names:
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.release()
                setattr(self, name, None)

    def _upload(self, name: str, array) -> cl.Buffer:
        """Copy `array` into the named device buffer, reallocating only if needed.

        Called once per evaluation for the parameter arrays, so it reuses the
        existing allocation whenever the size is unchanged. Allocating and
        releasing buffers on every call is a significant cost in a sampling loop,
        where the kernel itself takes only a few milliseconds.
        """
        array = ascontiguousarray(array)
        buffer = getattr(self, name)
        if buffer is None or buffer.size != array.nbytes:
            self._release(name)
            buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, array.nbytes)
            setattr(self, name, buffer)
        cl.enqueue_copy(self.queue, buffer, array)
        return buffer

    def set_data(self, times, lcids, pbids, epids, nsamples, exptimes) -> None:
        """Upload the observation times and per-light-curve metadata.

        Times are shifted to a reference epoch at the centre of the data span
        before being cast to the device dtype, and `t0` is shifted by the same
        amount in `__call__`. The model only ever uses `t - t0`, so this changes
        nothing mathematically, but it keeps the device arithmetic away from the
        absolute time origin. That matters enormously in single precision: a
        float32 ULP at BJD 2458325 is 0.25 d, so folding the absolute time would
        quantise a transit into a couple of representable values while raising no
        error. It also slightly improves the double-precision folding.
        """
        times = ascontiguousarray(times, float64)
        if times.size == 0:
            raise ValueError("The time array is empty; the OpenCL model needs at least one point.")

        self.tref = 0.5 * float(times.min() + times.max())
        self.times = ascontiguousarray(times - self.tref, self.dtype)
        self.npt = self.times.size
        self.nlc = asarray(pbids).size

        self._upload('_b_times', self.times)
        self._upload('_b_lcids', ascontiguousarray(lcids, int32))
        self._upload('_b_pbids', ascontiguousarray(pbids, int32))
        self._upload('_b_epids', ascontiguousarray(epids, int32))
        self._upload('_b_nsamples', ascontiguousarray(nsamples, int32))
        self._upload('_b_exptimes', ascontiguousarray(exptimes, self.dtype))

    def _allocate(self, npv: int) -> None:
        """(Re)allocate the output buffers if the required size changed.

        Keyed on the byte size rather than on `npv`, so a `set_data` call that
        changes the number of points invalidates the buffers without having to
        remember to reset any bookkeeping.
        """
        nbytes = npv * self.npt * self.dtype().nbytes
        for name, size in (('_b_flux', nbytes), ('_b_dflux', 7 * nbytes)):
            if name == '_b_dflux' and not self.return_grad:
                continue
            buffer = getattr(self, name)
            if buffer is None or buffer.size != size:
                self._release(name)
                setattr(self, name, cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size))
        self.npv = npv

    def __call__(self, times, k, t0, p, a, i, e, w, lcids, pbids, epids,
                 nsamples, exptimes, npb, nor, copy: bool = True):
        """Evaluate the model.

        The signature matches `backends.numba.udmodel` / `udmodel_grad`. The data
        arrays are taken from the buffers uploaded by `set_data`; the ones passed
        here are accepted for signature compatibility and only size-checked, so
        changing the supersampling or the passband map requires another
        `set_data` call rather than a different argument here.

        With ``copy=False`` nothing is read back and `None` is returned, leaving
        the results in `_b_flux` and `_b_dflux`.
        """
        if self._b_times is None:
            raise RuntimeError("set_data() must be called before evaluating the model.")
        for name, array, expected in (('times', times, self.npt), ('lcids', lcids, self.npt),
                                      ('pbids', pbids, self.nlc), ('epids', epids, self.nlc),
                                      ('nsamples', nsamples, self.nlc),
                                      ('exptimes', exptimes, self.nlc)):
            if asarray(array).size != expected:
                raise ValueError(f"The '{name}' array size ({asarray(array).size}) does not match "
                                 f"the uploaded data ({expected}). Call set_data() again.")

        npv, ntc = k.shape[0], t0.shape[1]
        self._allocate(npv)

        if self.return_grad:
            xyc, dxyc, bt1, bt4 = solve_coefficients_d(k, p, a, i, e, w, nor)
            self._upload('_b_dxyc', asarray(dxyc, self.dtype).reshape(-1))
        else:
            xyc, bt1, bt4 = solve_coefficients(k, p, a, i, e, w, nor)

        self._upload('_b_k', asarray(k, self.dtype).reshape(-1))
        self._upload('_b_t0', asarray(t0 - self.tref, self.dtype).reshape(-1))
        self._upload('_b_p', asarray(p, self.dtype).reshape(-1))
        self._upload('_b_xyc', asarray(xyc, self.dtype).reshape(-1))
        self._upload('_b_bt1', asarray(bt1, self.dtype))
        self._upload('_b_bt4', asarray(bt4, self.dtype))

        shape = (self.npt, npv)  # time index first, so neighbouring items write neighbours
        sizes = (int32(self.npt), int32(npb), int32(ntc), int32(nor))
        data = (self._b_lcids, self._b_pbids, self._b_epids,
                self._b_nsamples, self._b_exptimes)

        if self.return_grad:
            self.prg.udmodel_grad(self.queue, shape, None,
                                  self._b_times, self._b_k, self._b_t0, self._b_p,
                                  self._b_xyc, self._b_dxyc, self._b_bt1, self._b_bt4,
                                  *data, *sizes, self._b_flux, self._b_dflux)
        else:
            self.prg.udmodel(self.queue, shape, None,
                             self._b_times, self._b_k, self._b_t0, self._b_p,
                             self._b_xyc, self._b_bt1, self._b_bt4,
                             *data, *sizes, self._b_flux)

        if not copy:
            return None

        flux = zeros((npv, self.npt), self.dtype)
        cl.enqueue_copy(self.queue, flux, self._b_flux)
        if not self.return_grad:
            return flux

        dflux = zeros((npv, self.npt, 7), self.dtype)
        cl.enqueue_copy(self.queue, dflux, self._b_dflux)
        return flux, dflux
