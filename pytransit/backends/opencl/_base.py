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

"""Shared host-side machinery for the OpenCL transit model backends."""

import pyopencl as cl
from numpy import ascontiguousarray, asarray, float64, int32, zeros

from .context import dtype_for_precision, get_context, get_queue

__all__ = ['CLModel']


class CLModel:
    """Context, buffer, and data handling shared by the OpenCL model backends.

    Subclasses build their own program and implement `__call__` with the same
    signature as the corresponding Numba kernel, so the model classes can
    dispatch to them without special-casing the call.

    Parameters
    ----------
    return_grad : bool
        Evaluate the gradient kernel and return ``(flux, dflux)``.
    precision : {'double', 'single'}
        Device floating-point type. Double precision matches the Numba backends
        closely and is the default. Single precision is dramatically faster on
        consumer GPUs, where double-precision throughput is typically a
        sixty-fourth of single, at the cost of accuracy near the contact points.
        It also changes the *returned* dtype to float32, which propagates through
        the gradient expansion into whatever consumes it.
    cl_ctx, cl_queue : optional
        Override the shared context and queue from `context.py`. The queue must be
        in-order: parameter uploads overwrite buffers a previously enqueued kernel
        may still be reading, and `copy=False` deliberately leaves a kernel in
        flight, so out-of-order execution would race.

    Notes
    -----
    Buffers are reused across evaluations and reallocated only when the required
    size changes; allocating and releasing them per call is a significant cost in
    a sampling loop, where the kernel itself takes a few milliseconds.
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

        self.times = None
        self.npt: int = 0
        self.nlc: int = 0
        self.tref: float = 0.0

        # Device-resident results. These stay valid after an `evaluate(copy=False)`
        # call and are what a device-side likelihood would consume. `_b_dflux` holds
        # the *compact* gradient; the expansion into per-passband and per-epoch
        # columns is a host-side operation.
        self._b_flux = None
        self._b_dflux = None

        self._b_times = None
        self._b_lcids = None
        self._b_pbids = None
        self._b_epids = None
        self._b_nsamples = None
        self._b_exptimes = None

    # -- buffer handling -------------------------------------------------------

    def _release(self, *names) -> None:
        for name in names:
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.release()
                setattr(self, name, None)

    def _upload(self, name: str, array) -> cl.Buffer:
        """Copy `array` into the named device buffer, reallocating only if needed."""
        array = ascontiguousarray(array)
        buffer = getattr(self, name, None)
        if buffer is None or buffer.size != array.nbytes:
            self._check_allocation(name, array.nbytes)
            self._release(name)
            buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, array.nbytes)
            setattr(self, name, buffer)
        cl.enqueue_copy(self.queue, buffer, array)
        return buffer

    def _allocate(self, npv: int, ngrad: int = 0) -> None:
        """(Re)allocate the output buffers if the required size changed.

        Keyed on the byte size rather than on `npv`, so a `set_data` call that
        changes the number of points invalidates the buffers without having to
        remember to reset any bookkeeping. `ngrad` is the compact gradient row
        width, which differs per model and, for RoadRunner, per evaluation.
        """
        nbytes = npv * self.npt * self.dtype().nbytes
        sizes = [('_b_flux', nbytes)]
        if self.return_grad:
            sizes.append(('_b_dflux', ngrad * nbytes))
        for name, size in sizes:
            self._check_allocation(name, size)
            buffer = getattr(self, name)
            if buffer is None or buffer.size != size:
                self._release(name)
                setattr(self, name, cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size))

    def _check_allocation(self, name: str, nbytes: int) -> None:
        """Fail clearly when a buffer exceeds what the device can allocate.

        `cl.Buffer` allocation is lazy, so an oversized request otherwise
        surfaces as a bare MEM_OBJECT_ALLOCATION_FAILURE when the kernel is
        enqueued, naming neither the buffer nor the size. The binding limit is
        the per-buffer ceiling, which is typically a quarter of device memory.
        """
        limit = min(device.max_mem_alloc_size for device in self.ctx.devices)
        if nbytes > limit:
            raise MemoryError(
                f"The OpenCL '{name.removeprefix('_b_')}' buffer would need "
                f"{nbytes / 1e9:.2f} GB, but the device's max_mem_alloc_size is "
                f"{limit / 1e9:.2f} GB. Evaluate fewer parameter vectors at a time, "
                f"split the data, or use precision='single' to halve it.")

    # -- data ------------------------------------------------------------------

    def set_data(self, times, lcids, pbids, epids, nsamples, exptimes) -> None:
        """Upload the observation times and per-light-curve metadata.

        Times are shifted to a reference epoch at the centre of the data span
        before being cast to the device dtype, and `t0` is shifted by the same
        amount when it is uploaded. The models only ever use `t - t0`, so this
        changes nothing mathematically, but it keeps the device arithmetic away
        from the absolute time origin. That matters enormously in single
        precision: a float32 ULP at BJD 2458325 is 0.25 d, so folding the absolute
        time would quantise a transit into a couple of representable values while
        raising no error. It also slightly improves the double-precision folding.
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

    def _check_data(self, times, lcids, pbids, epids, nsamples, exptimes) -> None:
        """Verify the caller's data arrays still match what `set_data` uploaded."""
        if self._b_times is None:
            raise RuntimeError("set_data() must be called before evaluating the model.")
        for name, array, expected in (('times', times, self.npt), ('lcids', lcids, self.npt),
                                      ('pbids', pbids, self.nlc), ('epids', epids, self.nlc),
                                      ('nsamples', nsamples, self.nlc),
                                      ('exptimes', exptimes, self.nlc)):
            if asarray(array).size != expected:
                raise ValueError(f"The '{name}' array size ({asarray(array).size}) does not match "
                                 f"the uploaded data ({expected}). Call set_data() again.")

    # -- results ---------------------------------------------------------------

    def _read_back(self, npv: int, ngrad: int):
        """Copy the flux and, when requested, the compact gradient off the device."""
        flux = zeros((npv, self.npt), self.dtype)
        cl.enqueue_copy(self.queue, flux, self._b_flux)
        if not self.return_grad:
            return flux
        dflux = zeros((npv, self.npt, ngrad), self.dtype)
        cl.enqueue_copy(self.queue, dflux, self._b_dflux)
        return flux, dflux

    @property
    def data_buffers(self):
        """The per-light-curve index buffers, in the order every kernel takes them."""
        return (self._b_lcids, self._b_pbids, self._b_epids,
                self._b_nsamples, self._b_exptimes)
