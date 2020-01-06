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


"""OpenCL implementation of the transit over a uniform disk.
"""

import warnings
from os.path import dirname, join
import pyopencl as cl
from pyopencl import CompilerWarning
from numpy import array, uint32, float32, asarray, zeros, ones, unique, atleast_2d, squeeze, ndarray
from .transitmodel import TransitModel

warnings.filterwarnings('ignore', category=CompilerWarning)


class UniformModelCL(TransitModel):
    """
    Exoplanet transit over a uniform disk.
    """

    def __init__(self, method: str = 'pars', is_secondary: bool = False, cl_ctx=None, cl_queue=None) -> None:
        super().__init__(method, is_secondary)

        # Initialize the OpenCL context and queue
        # ---------------------------------------
        self.ctx = cl_ctx or cl.create_some_context()
        self.queue = cl_queue or cl.CommandQueue(self.ctx)

        # Declare stuff
        # -------------
        self.nptb: int = 0  # Number of datapoints
        self.nlc: int  = 0  # Number of light curves
        self.npb: int  = 0  # Number of passpands

        # Declare the arrays
        # ------------------
        self.flux:     ndarray = None       # Model flux values
        self.time:     ndarray = None       # Mid-exposure times
        self.lcids:    ndarray = None       # Light curve indices
        self.pbids:    ndarray = None       # Passband indices
        self.nsamples: ndarray = None       # Number of samples per light curve
        self.exptimes: ndarray = None       # Exposure times per light curve
        self.pv:       ndarray = array([])  # Parameter vector array

        # Declare the buffers
        # --------------------
        self._b_time = None    # Buffer for the mid-exposure times
        self._b_flux = None    # Buffer for the model flux values
        self._b_pv   = None    # Parameter vector buffer

        # Build the program
        # -----------------
        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'opencl','ma_uniform.cl'),'r').read()).build()


    def set_data(self, time: ndarray, lcids: ndarray = None, pbids: ndarray = None, nsamples: ndarray = None, exptimes: ndarray = None):
        mf = cl.mem_flags

        if self._b_time is not None:
            self._b_time.release()
            self._b_lcids.release()
            self._b_pbids.release()
            self._b_nsamples.release()
            self._b_etimes.release()

        self.nlc = uint32(1 if lcids is None else unique(lcids).size)
        self.npb = uint32(1 if pbids is None else unique(pbids).size)
        self.nptb = time.size

        self.time = asarray(time, dtype='float32')
        self.lcids = zeros(time.size, 'uint32') if lcids is None else asarray(lcids, dtype='uint32')
        self.pbids = zeros(self.nlc, 'uint32') if pbids is None else asarray(pbids, dtype='uint32')
        self.nsamples = ones(self.nlc, 'uint32') if nsamples is None else asarray(nsamples, dtype='uint32')
        self.exptimes = ones(self.nlc, 'float32') if exptimes is None else asarray(exptimes, dtype='float32')

        self._b_time = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.time)
        self._b_lcids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcids)
        self._b_pbids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pbids)
        self._b_nsamples = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.nsamples)
        self._b_etimes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.exptimes)


    def evaluate_ps(self, k, t0, p, a, i, e=0., w=0., copy=True):
        pvp = array([[k, t0, p, a, i, e, w]], float32)
        return self.evaluate_pv(pvp, copy)

    def evaluate_pv(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        self.npv = uint32(pvp.shape[0])
        self.spv = uint32(pvp.shape[1])

        if pvp.size != self.pv.size:
            if self._b_flux is not None:
                self._b_flux.release()
                self._b_pv.release()

            self.pv = zeros(pvp.shape, float32)
            self.flux = zeros((self.npv, self.nptb), float32)

            mf = cl.mem_flags
            self._b_flux = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * self.npv)
            self._b_pv = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)

        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_pv, self.pv)

        self.prg.uniform_eccentric_pop(self.queue, (self.npv, self.nptb), None, self._b_time, self._b_lcids, self._b_pbids,
                                  self._b_pv, self._b_nsamples, self._b_etimes,
                                  self.spv, self.nlc, self.npb, self._b_flux)

        if copy:
            cl.enqueue_copy(self.queue, self.flux, self._b_flux)
            return squeeze(self.flux)
        else:
            return None
