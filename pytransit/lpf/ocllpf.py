#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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


import pyopencl as cl
from numba import njit, float64
from numpy import inf, sqrt, zeros, isfinite, where, atleast_2d, int32, uint32

from .loglikelihood.clloglikelihood import CLLogLikelihood
from ..models.ma_quadratic_cl import QuadraticModelCL
from ..orbits.orbits_py import as_from_rhop, i_from_ba
from .lpf import BaseLPF

class OCLBaseLPF(BaseLPF):
    def __init__(self, target: str, passbands: list, times: list = None, fluxes: list = None, errors: list = None,
                 pbids: list = None, covariates: list = None, nsamples: tuple = None, exptimes: tuple = None,
                 wnids: list = None,
                 klims: tuple = (0.01, 0.75), nk: int = 512, nz: int = 512, cl_ctx=None, cl_queue=None, init_data=True,
                 cl_lnl_chunks: int = 1):

        self.cl_ctx = cl_ctx or self.tm.ctx
        self.cl_queue = cl_queue or self.tm.queue
        self.cl_lnl_chunks = cl_lnl_chunks

        tm = QuadraticModelCL(klims=klims, nk=nk, nz=nz, cl_ctx=cl_ctx, cl_queue=cl_queue)
        super().__init__(target, passbands, times, fluxes, errors, pbids, covariates, wnids, tm, nsamples, exptimes,
                         init_data=init_data)

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(CLLogLikelihood(self, cl_lnl_chunks=self.cl_lnl_chunks))

    def transit_model(self, pvp, copy=False):
        pvp = atleast_2d(pvp)
        pvp_t = zeros([pvp.shape[0], 8], "f")
        uv = zeros([pvp.shape[0], 2], "f")
        pvp_t[:, 0:1] = sqrt(pvp[:, self._pid_k2])  # Radius ratio
        pvp_t[:, 1:3] = pvp[:, 0:2]                 # Transit centre and orbital period
        pvp_t[:, 3] = a = as_from_rhop(pvp[:, 2], pvp[:, 1])
        pvp_t[:, 4] = i_from_ba(pvp[:, 3], a)
        a, b = sqrt(pvp[:, self._sl_ld][:, 0]), 2. * pvp[:, self._sl_ld][:, 1]
        uv[:, 0] = a * b
        uv[:, 1] = a * (1. - b)
        flux = self.tm.evaluate_pv(pvp_t, uv, copy=copy)
        return flux if copy else None

    def flux_model(self, pvp):
        return self.transit_model(pvp, copy=True).astype('d')

    def lnlikelihood(self, pvp):
        """Log likelihood for a 1D or 2D array of model parameters.

        Parameters
        ----------
        pvp: ndarray
            Either a 1D parameter vector or a 2D parameter array.

        Returns
        -------
            Log likelihood for the given parameter vector(s).
        """
        pvp = atleast_2d(pvp)
        self.transit_model(pvp, copy=False)
        return self._lnlikelihood_models[0](pvp)
