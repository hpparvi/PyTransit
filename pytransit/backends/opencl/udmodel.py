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
from numpy import asarray, int32

from ._base import CLModel
from ._host import solve_coefficients, solve_coefficients_d
from .context import build_options, build_program, read_kernel_source

__all__ = ['UDModelCL']

NGRAD = 7   # compact gradient row width: [k, t0, p, a, i, e, w]


class UDModelCL(CLModel):
    """Uniform-disk transit model evaluated on an OpenCL device.

    Callable with the same signature as `backends.numba.udmodel`, so the model
    class can dispatch to it without special-casing the call itself. The Taylor
    coefficients are solved on the host (see `_host.py`) and only the
    per-datapoint separation, circle-circle intersection, and supersampling loop
    run on the device.

    See `CLModel` for the constructor arguments and the buffer and time-shift
    handling.

    Notes
    -----
    In single precision, expect flux errors of order 1e-6 from the kite-area term
    in the circle-circle intersection near the contact points.
    """

    def __init__(self, return_grad: bool = False, precision: str = 'double',
                 cl_ctx=None, cl_queue=None):
        super().__init__(return_grad, precision, cl_ctx, cl_queue)
        self.prg = build_program(self.ctx,
                                 read_kernel_source('common.cl', 'udmodel.cl'),
                                 build_options(precision))

        self._b_k = None
        self._b_t0 = None
        self._b_p = None
        self._b_xyc = None
        self._b_dxyc = None
        self._b_bt1 = None
        self._b_bt4 = None

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
        self._check_data(times, lcids, pbids, epids, nsamples, exptimes)

        npv, ntc = k.shape[0], t0.shape[1]
        self._allocate(npv, NGRAD)

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

        if self.return_grad:
            self.prg.udmodel_grad(self.queue, shape, None,
                                  self._b_times, self._b_k, self._b_t0, self._b_p,
                                  self._b_xyc, self._b_dxyc, self._b_bt1, self._b_bt4,
                                  *self.data_buffers, *sizes, self._b_flux, self._b_dflux)
        else:
            self.prg.udmodel(self.queue, shape, None,
                             self._b_times, self._b_k, self._b_t0, self._b_p,
                             self._b_xyc, self._b_bt1, self._b_bt4,
                             *self.data_buffers, *sizes, self._b_flux)

        return None if not copy else self._read_back(npv, NGRAD)
