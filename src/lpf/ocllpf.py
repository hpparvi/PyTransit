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

from ..models.ma_quadratic_cl import QuadraticModelCL
from ..orbits.orbits_py import as_from_rhop, i_from_ba
from .lpf import BaseLPF


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


class OCLBaseLPF(BaseLPF):
    def __init__(self, target: str, passbands: list, times: list = None, fluxes: list = None, errors: list = None,
                 pbids: list = None, covariates: list = None, nsamples: tuple = None, exptimes: tuple = None,
                 wnids: list = None,
                 klims: tuple = (0.01, 0.75), nk: int = 512, nz: int = 512, cl_ctx=None, cl_queue=None, init_data=True,
                 cl_lnl_chunks: int = 1):

        self.cl_ctx = cl_ctx or self.tm.ctx
        self.cl_queue = cl_queue or self.tm.queue
        self.cl_lnl_chunks = cl_lnl_chunks

        # Define the OpenCL buffers
        # -------------------------
        self._b_flux = None
        self._b_err = None
        self._b_lnl2d = None
        self._b_lnl1d = None
        self._b_lcids = None
        self._b_errids = None
        self._b_covariates = None

        super().__init__(target, passbands, times, fluxes, errors, pbids, covariates, wnids, None, nsamples, exptimes, init_data=False)

        if init_data:
            self._init_data(times = times, fluxes = fluxes, pbids = pbids, covariates = covariates,
                            errors = errors, wnids = wnids, nsamples = nsamples, exptimes = exptimes)
            self._init_parameters()
            self._init_instrument()

        self.tm = QuadraticModelCL(klims=klims, nk=nk, nz=nz, cl_ctx=cl_ctx, cl_queue=cl_queue)
        self.tm.set_data(self.timea, self.lcids, self.pbids, self.nsamples, self.exptimes)

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
        self.lnlikelihood = self.lnlikelihood_ocl


    def _init_data(self, times, fluxes, pbids, covariates=None, errors=None, wnids = None, nsamples=None, exptimes=None):
        super()._init_data(times, fluxes, pbids, covariates, errors, wnids, nsamples, exptimes)
        self.nlc = int32(self.nlc)
        self.n_noise_blocks = int32(self.n_noise_blocks)

        # Initialise the Python arrays
        # ----------------------------
        self.timea = self.timea.astype('f')
        self.ofluxa = self.ofluxa.astype('f')
        self.lnl2d = zeros([50, self.ofluxa.size], 'f')
        self.lnl1d = zeros([self.lnl2d.shape[0], self.cl_lnl_chunks], 'f')
        self.ferr = zeros([50, self.nlc])
        self.lcids = self.lcids.astype('int32')
        self.pbids = self.pbids.astype('int32')
        self.noise_ids = self.noise_ids.astype('int32')
        if covariates is not None:
            self.cova = self.cova.astype('f')

        # Release OpenCL buffers if they're initialised
        # ---------------------------------------------
        if self._b_flux:
            self._b_flux.release()
            self._b_err.release()
            self._b_lnl2d.release()
            self._b_lnl1d.release()
            self._b_lcids.release()
            self._b_errids.release()

        if self._b_covariates:
            self._b_covariates.release()

        # Initialise OpenCL buffers
        # -------------------------
        mf = cl.mem_flags
        self._b_flux = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ofluxa)
        self._b_err = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ferr)
        self._b_lnl2d = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, self.lnl2d.nbytes)
        self._b_lnl1d = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, self.lnl1d.nbytes)
        self._b_lcids = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcids)
        self._b_errids = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.noise_ids)
        if covariates is not None:
            self._b_covariates = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.cova)

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
        self.transit_model(pv)
        cl.enqueue_copy(self.cl_queue, self._b_err, (10**pv[:, self._sl_err]).astype('f'))
        self.prg_lnl.lnl2d(self.cl_queue, self.tm.f.shape, None, self.nlc, self._b_flux, self.tm._b_f,
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

    def lnprior(self, pv):
        lnpriors = zeros(pv.shape[0])
        for i, p in enumerate(self.ps.priors):
            lnpriors += p.logpdf(pv[:, i])
        return lnpriors + self.additional_priors(pv)

    def lnposterior(self, pv):
        lnp = self.lnlikelihood(pv) + self.lnprior(pv)
        return where(isfinite(lnp), lnp, -inf)
