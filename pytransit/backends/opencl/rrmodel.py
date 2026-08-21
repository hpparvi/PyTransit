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

"""OpenCL host wrapper for the RoadRunner transit model."""

from numpy import asarray, int32

from ._base import CLModel
from ._host import rr_precompute, rr_precompute_d
from .context import build_options, build_program, read_kernel_source

__all__ = ['RRModelCL']


class RRModelCL(CLModel):
    """RoadRunner transit model (Parviainen 2020) evaluated on an OpenCL device.

    Callable with the same signature as `backends.numba.rrmodel`, so
    `RoadRunnerModel` can dispatch to it through the branch it already uses for
    Numba.

    Limb darkening never reaches the device as a function. `RoadRunnerModel`
    tabulates the profile on the host, and this class contracts it against the
    weight table (`_host.rr_precompute`) into the mean intensity under the planet
    disk. The kernels only interpolate that table, so any limb-darkening
    profile -- including a user-supplied `LDModel` -- works without device code.

    See `CLModel` for the constructor arguments and the buffer and time-shift
    handling.

    Notes
    -----
    Single precision is noticeably worse here than for the uniform disk: the
    per-sample flux is ``(ldi - Ip*A)/ldi``, a difference of comparable
    magnitudes, so the subtraction loses significance on top of the usual
    contact-point error.

    The number of limb-darkening coefficients is only known once `ldg` arrives,
    so the gradient program is built on the first gradient evaluation with
    ``-DNLDC`` set, and cached by `context.build_program` thereafter.
    """

    def __init__(self, return_grad: bool = False, precision: str = 'double',
                 cl_ctx=None, cl_queue=None):
        super().__init__(return_grad, precision, cl_ctx, cl_queue)
        self._source = read_kernel_source('common.cl', 'rrmodel.cl')
        self._programs: dict = {}

        self._b_k = None
        self._b_t0 = None
        self._b_p = None
        self._b_xyc = None
        self._b_dxyc = None
        self._b_ldm = None
        self._b_dldm = None
        self._b_ldi = None
        self._b_dldi = None
        self._b_bt1 = None
        self._b_bt4 = None
        self._b_valid = None

    def _program(self, nldc: int):
        """Build (or fetch) the program specialised for `nldc` coefficients."""
        program = self._programs.get(nldc)
        if program is None:
            options = f'{build_options(self.precision)} -DNLDC={nldc}'
            program = self._programs[nldc] = build_program(self.ctx, self._source, options)
        return program

    def __call__(self, times, k, t0, p, a, i, e, w, lcids, pbids, epids,
                 nsamples, exptimes, ldp, ldg, ldi, dldi,
                 weights, dk, kmin, kmax, dg, z_edges, npb, nep, copy: bool = True):
        """Evaluate the model.

        The signature matches `backends.numba.rrmodel` / `rrmodel_grad`. The data
        arrays are taken from the buffers uploaded by `set_data`; the ones passed
        here are only size-checked.

        With ``copy=False`` nothing is read back and `None` is returned, leaving
        the results in `_b_flux` and `_b_dflux`.
        """
        self._check_data(times, lcids, pbids, epids, nsamples, exptimes)

        npv, ntc = k.shape[0], t0.shape[1]
        ng = weights.shape[1]
        nldc = ldg.shape[2] - 1 if self.return_grad else 0
        ngrad = 7 + npb * nldc
        self._allocate(npv, ngrad)

        if self.return_grad:
            ldm, dldm, xyc, dxyc, bt1, bt4, valid = rr_precompute_d(
                k, p, a, i, e, w, ldp, ldg, weights, dk, kmin, kmax, z_edges, ng, nldc, nep)
            self._upload('_b_dxyc', asarray(dxyc, self.dtype).reshape(-1))
            self._upload('_b_dldm', asarray(dldm, self.dtype).reshape(-1))
            self._upload('_b_dldi', asarray(dldi, self.dtype).reshape(-1))
        else:
            ldm, xyc, bt1, bt4, valid = rr_precompute(
                k, p, a, i, e, w, ldp, weights, dk, kmin, kmax, z_edges, ng, nep)

        self._upload('_b_k', asarray(k, self.dtype).reshape(-1))
        self._upload('_b_t0', asarray(t0 - self.tref, self.dtype).reshape(-1))
        self._upload('_b_p', asarray(p, self.dtype).reshape(-1))
        self._upload('_b_xyc', asarray(xyc, self.dtype).reshape(-1))
        self._upload('_b_ldm', asarray(ldm, self.dtype).reshape(-1))
        self._upload('_b_ldi', asarray(ldi, self.dtype).reshape(-1))
        self._upload('_b_bt1', asarray(bt1, self.dtype))
        self._upload('_b_bt4', asarray(bt4, self.dtype))
        self._upload('_b_valid', asarray(valid, int32))

        shape = (self.npt, npv)  # time index first, so neighbouring items write neighbours
        sizes = (int32(self.npt), int32(npb), int32(ntc), int32(nep),
                 int32(ng), self.dtype(dg))
        program = self._program(nldc)

        if self.return_grad:
            program.rrmodel_grad(self.queue, shape, None,
                                 self._b_times, self._b_k, self._b_t0, self._b_p,
                                 self._b_xyc, self._b_dxyc, self._b_ldm, self._b_dldm,
                                 self._b_ldi, self._b_dldi, self._b_bt1, self._b_bt4,
                                 self._b_valid, *self.data_buffers, *sizes,
                                 self._b_flux, self._b_dflux)
        else:
            program.rrmodel(self.queue, shape, None,
                            self._b_times, self._b_k, self._b_t0, self._b_p,
                            self._b_xyc, self._b_ldm, self._b_ldi,
                            self._b_bt1, self._b_bt4, self._b_valid,
                            *self.data_buffers, *sizes, self._b_flux)

        return None if not copy else self._read_back(npv, ngrad)
