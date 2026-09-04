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
from numpy import ascontiguousarray, asarray, concatenate, float64, int32, zeros

from .context import (build_options, build_program, dtype_for_precision, get_context,
                      get_queue, read_kernel_source)

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

        # Log-likelihood reduction: built lazily, since most models are only ever
        # evaluated.
        self._reduction = None
        self._reduction_lsize = None
        self._obs_id = None
        self._obs_cache = None
        self._b_obs = None
        self._b_err = None
        self._b_slot_col = None
        self._b_slot_kind = None
        self._b_slot_grp = None
        self._b_lnl = None
        self._slot_key = None

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
                # READ_WRITE, not WRITE_ONLY: the reduction kernel reads these, and
                # reading a WRITE_ONLY buffer inside a kernel is undefined behaviour.
                setattr(self, name, cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size))

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
        self._obs_id = None      # a new time array invalidates any uploaded observations

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

    # -- log-likelihood reduction ---------------------------------------------

    def set_observations(self, obs, err) -> None:
        """Upload the observed flux and its per-point uncertainties.

        Cached on the identity of the two arrays, so a sampler calling
        `loglikelihood` in a loop uploads them once. The cache holds a reference
        to what it uploaded: `id()` is unique only among *live* objects, so
        without that a freed array's address could be recycled by a different
        array and the stale upload would be reused silently.
        """
        key = (id(obs), id(err))
        if key == self._obs_id:
            return
        # Pin the objects the key was taken from, before any dtype conversion:
        # in single precision `ascontiguousarray` returns copies, and pinning
        # those would leave the keyed originals free to be deallocated and their
        # addresses recycled -- which would make this cache hit on a different
        # array and silently reuse a stale upload.
        keyed = (obs, err)
        obs = ascontiguousarray(obs, self.dtype)
        err = ascontiguousarray(err, self.dtype)
        if obs.size != self.npt or err.size != self.npt:
            raise ValueError(f"The observation arrays must have {self.npt} points, got "
                             f"{obs.size} and {err.size}.")
        self._upload('_b_obs', obs)
        self._upload('_b_err', err)
        self._obs_cache = keyed
        self._obs_id = key

    def reduce_loglikelihood(self, jitter: float, baseline: float, slots, ngrad: int, npv: int):
        """Reduce the device-resident flux and Jacobian into log-likelihood sums.

        Must follow an evaluation with ``copy=False``, whose results are still in
        `_b_flux` and `_b_dflux`. Nothing but the reduced sums crosses the bus.

        Parameters
        ----------
        jitter : float
            Additional white-noise term, added in quadrature to the per-point
            uncertainties.
        baseline : float
            Flux baseline. The residual is ``(obs - baseline) - dF``, so a
            normalised light curve (baseline=1.0) matches the model's flux deficit
            convention.
        slots : tuple of ndarray
            ``(col, kind, grp)`` describing each gradient output slot; see
            `reduction.cl`. The first `NSCALAR` slots are the scalar sums and are
            prepended here.
        ngrad : int
            Width of the compact Jacobian rows on the device.

        Returns
        -------
        ndarray
            Shape ``(npv, 5 + n_gradient_slots)``: the five scalar sums followed
            by one reduced gradient column per slot.
        """
        if self._b_flux is None or self._b_dflux is None:
            raise RuntimeError("Evaluate the model with copy=False before reducing it.")
        if self._b_obs is None:
            raise RuntimeError("set_observations() must be called before reducing.")

        # The kernel strides the Jacobian by `ngrad`, which the caller derives
        # independently of the buffer the model kernel filled. A mismatch -- a
        # custom LDModel whose gradient row count disagrees with its coefficient
        # count, or a stale buffer from a differently shaped evaluation -- would
        # read the wrong columns and return plausible nonsense.
        expected = npv * self.npt * ngrad * self.dtype().nbytes
        if self._b_dflux.size != expected:
            raise ValueError(
                f"The gradient row width ({ngrad}) does not match the Jacobian on the device "
                f"({self._b_dflux.size} bytes, expected {expected}). Evaluate and reduce with "
                f"the same parameter shapes.")

        if self._reduction is None:
            self._reduction = build_program(
                self.ctx, read_kernel_source('common.cl', 'reduction.cl'),
                build_options(self.precision))

        col, kind, grp = slots
        nslot = 5 + col.size
        # The slot table depends only on the parameter shapes, so in a sampling
        # loop it is identical every call; re-uploading it would be three
        # avoidable round trips per likelihood evaluation.
        key = (col.tobytes(), kind.tobytes(), grp.tobytes())
        if key != self._slot_key:
            # The five scalar sums occupy the leading slots; their col/kind/grp
            # are unused by the kernel but must be present so the arrays line up.
            pad = zeros(5, int32)
            self._upload('_b_slot_col', concatenate([pad, asarray(col, int32)]))
            self._upload('_b_slot_kind', concatenate([pad, asarray(kind, int32)]))
            self._upload('_b_slot_grp', concatenate([pad, asarray(grp, int32)]))
            self._slot_key = key

        nbytes = npv * nslot * self.dtype().nbytes
        if self._b_lnl is None or self._b_lnl.size != nbytes:
            self._check_allocation('_b_lnl', nbytes)
            self._release('_b_lnl')
            self._b_lnl = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, nbytes)

        lsize = self._reduction_local_size()
        self._reduction.wn_loglikelihood(
            self.queue, (lsize * nslot, npv), (lsize, 1),
            self._b_obs, self._b_err, self._b_flux, self._b_dflux,
            self._b_lcids, self._b_pbids, self._b_epids,
            self._b_slot_col, self._b_slot_kind, self._b_slot_grp,
            int32(self.npt), int32(ngrad), int32(nslot),
            self.dtype(jitter), self.dtype(baseline),
            cl.LocalMemory(lsize * self.dtype().nbytes), self._b_lnl)

        sums = zeros((npv, nslot), self.dtype)
        cl.enqueue_copy(self.queue, sums, self._b_lnl)
        return sums

    def _reduction_local_size(self) -> int:
        """Largest power-of-two work-group the kernel's tree reduction can use.

        Bounded by the *kernel's* limit, not just the device's: register and
        local-memory pressure can push `CL_KERNEL_WORK_GROUP_SIZE` below
        `CL_DEVICE_MAX_WORK_GROUP_SIZE`, and enqueueing a larger group then fails
        with CL_INVALID_WORK_GROUP_SIZE.
        """
        if self._reduction_lsize is None:
            kernel = self._reduction.wn_loglikelihood
            limit = min(min(device.max_work_group_size,
                            kernel.get_work_group_info(
                                cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
                        for device in self.ctx.devices)
            size = 1
            while size * 2 <= min(limit, 256):
                size *= 2
            self._reduction_lsize = size
        return self._reduction_lsize

    @property
    def data_buffers(self):
        """The per-light-curve index buffers, in the order every kernel takes them."""
        return (self._b_lcids, self._b_pbids, self._b_epids,
                self._b_nsamples, self._b_exptimes)
