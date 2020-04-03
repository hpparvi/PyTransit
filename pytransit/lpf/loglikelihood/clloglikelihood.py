#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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
from numba import prange, njit
from numpy import atleast_2d, zeros, log, pi, asarray, unique, array, inf, where, isfinite, uint32, float64, int32, \
    arange

from ...param import  LParameter, NormalPrior as NP, UniformPrior as UP

@njit(parallel=False)
def psum2d(a):
    s = a.copy()
    ns = a.shape[0]
    for j in range(ns):
        midpoint = a.shape[1]
        while midpoint > 1:
            is_even = midpoint % 2 == 0
            if not is_even:
                s[j,0] += s[j,midpoint-1]
            midpoint = midpoint // 2
            for i in range(midpoint):
                s[j,i] = s[j,i] + s[j,midpoint+i]
    return s[:,0].astype(float64)


class CLLogLikelihood:
    def __init__(self, lpf, name: str = 'wn', noise_ids=None, sigma=None,
                 cl_ctx = None, cl_queue=None, cl_lnl_chunks: int = 1):
        self.name = name
        self.lpf = lpf
        self.cl_ctx = cl_ctx or self.lpf.tm.ctx
        self.cl_queue = cl_queue or self.lpf.tm.queue
        self.cl_lnl_chunks = cl_lnl_chunks

        # Define the OpenCL buffers
        # -------------------------
        self._b_flux = None
        self._b_err = None
        self._b_lnl2d = None
        self._b_lnl1d = None
        self._b_lcids = None
        self._b_errids = None

        if sigma is None:
            self.free = True
        else:
            self.sigma = asarray(sigma)
            self.free = False

        if noise_ids is not None:
            raise NotImplementedError("CLLoglikelihood cannot be applied to a subset of the data at the moment.")

        if lpf.noise_ids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CLLogLikelihood.')

        self.global_noise_ids = noise_ids if noise_ids is not None else unique(lpf.noise_ids)
        self.mapping = {g:l for g,l in zip(self.global_noise_ids, arange(self.global_noise_ids.size))}

        slices, lnids = [], []
        for nid, sl in zip(lpf.noise_ids, lpf.lcslices):
            if nid in self.global_noise_ids:
                slices.append([sl.start, sl.stop])
                lnids.append(self.mapping[nid])
        self.lcslices = array(slices)
        self.local_pv_noise_ids = array(lnids, int32)

        self.times = lpf.timea
        self.fluxes = lpf.ofluxa

        if self.free:
            self.init_parameters()

        src = """
                   __kernel void lnl2d(const int nlc, __global const float *obs, __global const float *mod, __global const int *lcids,
                   __global const float *err, const int nerr, __global const int *errids, __global float *lnl2d){
                          uint i_tm = get_global_id(1);    // time vector index
                          uint n_tm = get_global_size(1);  // time vector size
                          uint i_pv = get_global_id(0);    // parameter vector index
                          uint n_pv = get_global_size(0);  // parameter vector population size
                          uint gid = i_pv*n_tm + i_tm;     // global linear index
                          uint ierr  = errids[lcids[i_tm]];
                          float e = err[i_pv*nerr + ierr];
                          lnl2d[gid] = -log(e) - 0.5f*log(2*M_PI_F) - 0.5f*pown((obs[i_tm]-mod[gid]) / e, 2);
                    }

                    __kernel void lnl1d(const uint npt, __global float *lnl2d, __global float *lnl1d){
                          uint i_pv = get_global_id(0);    // parameter vector index
                          uint n_pv = get_global_size(0);  // parameter vector population size

                        int i;
                        bool is_even;
                        uint midpoint = npt;
                        __global float *lnl = &lnl2d[i_pv*npt];

                        while(midpoint > 1){
                            is_even = midpoint % 2 == 0;   
                            if (is_even == 0){
                                lnl[0] += lnl[midpoint-1];
                            }
                            midpoint /= 2;

                            for(i=0; i<midpoint; i++){
                                lnl[i] = lnl[i] + lnl[midpoint+i];
                            }
                        }
                        lnl1d[i_pv] = lnl[0];
                    }

                    __kernel void lnl1d_chunked(const uint npt, __global float *lnl2d, __global float *lnl1d){
                        uint ipv = get_global_id(0);    // parameter vector index
                        uint npv = get_global_size(0);  // parameter vector population size
                        uint ibl = get_global_id(1);    // block index
                        uint nbl = get_global_size(1);  // number of blocks
                        uint lnp = npt / nbl;

                        __global float *lnl = &lnl2d[ipv*npt + ibl*lnp];

                        if(ibl == nbl-1){
                            lnp = npt - (ibl*lnp);
                        }

                        bool is_even;
                        uint midpoint = lnp;
                        while(midpoint > 1){
                            is_even = midpoint % 2 == 0;   
                            if (is_even == 0){
                                lnl[0] += lnl[midpoint-1];
                            }
                            midpoint /= 2;

                            for(int i=0; i<midpoint; i++){
                                lnl[i] = lnl[i] + lnl[midpoint+i];
                            }
                        }
                        lnl1d[ipv*nbl + ibl] = lnl[0];
                    }
                """
        self.prg_lnl = cl.Program(self.cl_ctx, src).build()
        self._init_data()

    def init_parameters(self):
        name = self.name
        pgp = [LParameter(f'{name}_loge_{i}', f'{name} log10 sigma {i}', '', UP(-4, 0), bounds=(-inf, inf)) for i in self.global_noise_ids]
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, pgp)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{name}", self.pv_slice)
        setattr(self.lpf, f"_start_{name}", self.pv_start)

    def _init_data(self):
        lpf = self.lpf
        self.nlc = int32(lpf.nlc)
        self.n_noise_blocks = int32(lpf.n_noise_blocks)

        # Initialise the Python arrays
        # ----------------------------
        ninit = 1
        self.timea = lpf.timea.astype('f')
        self.ofluxa = lpf.ofluxa.astype('f')
        self.lnl2d = zeros([ninit, lpf.ofluxa.size], 'f')
        self.lnl1d = zeros([ninit, self.cl_lnl_chunks], 'f')
        self.ferr = zeros([ninit, lpf.nlc])
        self.lcids = lpf.lcids.astype('int32')
        self.pbids = lpf.pbids.astype('int32')
        self.noise_ids = lpf.noise_ids.astype('int32')

        # Release OpenCL buffers if they're initialised
        # ---------------------------------------------
        if self._b_flux:
            self._b_flux.release()
            self._b_err.release()
            self._b_lnl2d.release()
            self._b_lnl1d.release()
            self._b_lcids.release()
            self._b_errids.release()

        # Initialise OpenCL buffers
        # -------------------------
        mf = cl.mem_flags
        self._b_flux = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ofluxa)
        self._b_err = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ferr)
        self._b_lnl2d = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, self.lnl2d.nbytes)
        self._b_lnl1d = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, self.lnl1d.nbytes)
        self._b_lcids = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcids)
        self._b_errids = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.local_pv_noise_ids)

    def _lnl2d(self, pv):
        if self.lnl2d.shape[0] != pv.shape[0] or self.lnl1d.size != pv.shape[0]:
            self.err = zeros([pv.shape[0], self.n_noise_blocks], 'f')
            self._b_err.release()
            self._b_err = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, self.err.nbytes)
            self.lnl2d = zeros([pv.shape[0], self.ofluxa.size], 'f')
            self._b_lnl2d.release()
            self._b_lnl2d = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, self.lnl2d.nbytes)
            self.lnl1d = zeros([pv.shape[0], self.cl_lnl_chunks], 'f')
            if self._b_lnl1d:
                self._b_lnl1d.release()
            self._b_lnl1d = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, self.lnl1d.nbytes)

        err = (10 ** pv[:, self.pv_slice]).astype('f')
        cl.enqueue_copy(self.cl_queue, self._b_err, err)
        self.prg_lnl.lnl2d(self.cl_queue, self.lpf.tm.f.shape, None, self.nlc, self._b_flux, self.lpf.tm._b_f,
                           self._b_lcids, self._b_err, self.n_noise_blocks, self._b_errids, self._b_lnl2d)

    def lnlikelihood_numba(self, pv):
        self._lnl2d(pv)
        cl.enqueue_copy(self.cl_queue, self.lnl2d, self._b_lnl2d)
        lnl = psum2d(self.lnl2d)
        return where(isfinite(lnl), lnl, -inf)

    def lnlikelihood_ocl(self, pv):
        self._lnl2d(pv)
        self.prg_lnl.lnl1d_chunked(self.cl_queue, [self.lnl2d.shape[0], self.cl_lnl_chunks], None,
                                   uint32(self.lnl2d.shape[1]), self._b_lnl2d, self._b_lnl1d)
        cl.enqueue_copy(self.cl_queue, self.lnl1d, self._b_lnl1d)
        lnl = self.lnl1d.astype('d').sum(1)
        return lnl

    def lnlikelihood_numpy(self, pv):
        self._lnl2d(pv)
        cl.enqueue_copy(self.cl_queue, self.lnl2d, self._b_lnl2d)
        lnl = self.lnl2d.astype('d').sum(1)
        return where(isfinite(lnl), lnl, -inf)

    def __call__(self, pvp, model):
        return self.lnlikelihood_ocl(pvp)
