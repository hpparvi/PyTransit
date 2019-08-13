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


"""Mandel-Agol transit model
"""

import numpy as np
import pyopencl as cl
from os.path import dirname, join

import warnings
from pyopencl import CompilerWarning

from numpy import array, uint32, float32, int32, asarray, zeros, ones, unique, atleast_2d, squeeze

from .numba.ma_quadratic_nb import calculate_interpolation_tables
from .transitmodel import TransitModel

warnings.filterwarnings('ignore', category=CompilerWarning)

class QuadraticModelCL(TransitModel):
    """
    Exoplanet transit light curve model by Mandel and Agol (2001).
    """

    def __init__(self, method: str = 'pars', is_secondary: bool = False, klims: tuple = (0.05, 0.25), nk: int = 256, nz: int = 256, cl_ctx=None, cl_queue=None) -> None:
        super().__init__(method, is_secondary)

        self.ctx = cl_ctx or cl.create_some_context()
        self.queue = cl_queue or cl.CommandQueue(self.ctx)

        self.ed,self.le,self.ld,self.kt,self.zt = map(lambda a: np.array(a,dtype=float32,order='C'),
                                                      calculate_interpolation_tables(klims[0],klims[1],nk,nz))
        self.klims = klims
        self.nk    = int32(nk)
        self.nz    = int32(nz)
        self.nptb  = 0
        self.npb   = 0
        self.u     = np.array([])
        self.f     = None
        self.k0, self.k1 = map(float32, self.kt[[0,-1]])
        self.dk = float32(self.kt[1]-self.kt[0])
        self.dz = float32(self.zt[1]-self.zt[0])
        self.pv = array([])

        mf = cl.mem_flags

        # Create the buffers for the Mandel & Agol coefficient arrays
        self._b_ed = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ed)
        self._b_le = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.le)
        self._b_ld = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ld)
        self._b_kt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.kt)
        self._b_zt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.zt)

        self.time  = None
        self.lcids = None
        self.pbids = None
        self.nsamples = None
        self.exptimes = None

        # Declare the buffers for the ld coefficients, time, and flux arrays. These will
        # be initialised when the model is first evaluated, and reinitialised if the
        # array sizes change.
        #
        self._b_u = None       # Limb darkening coefficient buffer
        self._b_time = None       # Time buffer
        self._b_f = None       # Flux buffer
        self._b_p = None       # Parameter vector buffer

        self._time_id = None   # Time array ID

        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'opencl','ma_quadratic.cl'),'r').read()).build()


    def set_data(self, time, lcids=None, pbids=None, nsamples=None, exptimes=None):
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

        #TODO: Fix nsamples and exptimes in cases where a scalar is given.
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


    def evaluate_ps(self, k, ldc, t0, p, a, i, e=0., w=0., copy=True):
        ldc = asarray(ldc, float32)
        pvp = array([[k, t0, p, a, i, e, w]], float32)
        return self.evaluate_pv(pvp, ldc, copy)

    def evaluate_pv(self, pvp, ldc, copy=True):
        pvp = atleast_2d(pvp)
        ldc = atleast_2d(ldc).astype(float32)
        self.npv = uint32(pvp.shape[0])
        self.spv = uint32(pvp.shape[1])

        # Release and reinitialise the GPU buffers if the sizes of the time or
        # limb darkening coefficient arrays change.
        if (ldc.size != self.u.size) or (pvp.size != self.pv.size):
            assert self.npb == ldc.shape[1] // 2

            if self._b_f is not None:
                self._b_f.release()
                self._b_u.release()
                self._b_p.release()

            self.pv = zeros(pvp.shape, float32)
            self.u = zeros((self.npv, 2 * self.npb), float32)
            self.f = zeros((self.npv, self.nptb), float32)

            mf = cl.mem_flags
            self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * self.npv)
            self._b_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.u)
            self._b_p = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)

        # Copy the limb darkening coefficient array to the GPU
        cl.enqueue_copy(self.queue, self._b_u, ldc)

        # Copy the parameter vector to the GPU
        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        self.prg.ma_eccentric_pop(self.queue, (self.npv, self.nptb), None, self._b_time, self._b_lcids, self._b_pbids,
                                  self._b_p, self._b_u,
                                  self._b_ed, self._b_le, self._b_ld, self._b_nsamples, self._b_etimes,
                                  self.k0, self.k1, self.nk, self.nz, self.dk, self.dz,
                                  self.spv, self.nlc, self.npb, self._b_f)

        if copy:
            cl.enqueue_copy(self.queue, self.f, self._b_f)
            return squeeze(self.f)
        else:
            return None


    def evaluate_pv_ttv(self, pvp, ldc, copy=True, tdv=False):
        pvp = atleast_2d(pvp)
        ldc = atleast_2d(ldc).astype(float32)
        self.npv = uint32(pvp.shape[0])
        self.spv = uint32(pvp.shape[1])

        # Release and reinitialise the GPU buffers if the sizes of the time or
        # limb darkening coefficient arrays change.
        if (ldc.size != self.u.size) or (pvp.size != self.pv.size):
            assert self.npb == ldc.shape[1] // 2

            if self._b_f is not None:
                self._b_f.release()
                self._b_u.release()
                self._b_p.release()

            self.pv = zeros(pvp.shape, float32)
            self.u  = zeros((self.npv, 2*self.npb), float32)
            self.f  = zeros((self.npv, self.time.size), float32)

            mf = cl.mem_flags
            self._b_f   = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * self.npv)
            self._b_u   = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.u)
            self._b_p   = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)

        # Copy the limb darkening coefficient array to the GPU
        cl.enqueue_copy(self.queue, self._b_u, ldc)

        # Copy the parameter vector to the GPU
        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        if tdv:
            self.prg.ma_eccentric_pop_tdv(self.queue, (self.npv, self.nptb), None, self._b_time, self._b_lcids,
                                      self._b_pbids,
                                      self._b_p, self._b_u,
                                      self._b_ed, self._b_le, self._b_ld, self._b_nsamples, self._b_etimes,
                                      self.k0, self.k1, self.nk, self.nz, self.dk, self.dz,
                                      self.spv, self.nlc, self.npb, self._b_f)
        else:
            self.prg.ma_eccentric_pop_ttv(self.queue, (self.npv, self.nptb), None,
                                          self._b_time, self._b_lcids, self._b_pbids,
                                          self._b_p, self._b_u,
                                          self._b_ed, self._b_le, self._b_ld,
                                          self._b_nsamples, self._b_etimes,
                                          self.k0, self.k1, self.nk, self.nz, self.dk, self.dz,
                                          self.spv, self.nlc, self.npb,
                                          self._b_f)

        if copy:
            cl.enqueue_copy(self.queue, self.f, self._b_f)
            return self.f
        else:
            return None
