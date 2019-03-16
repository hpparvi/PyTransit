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
from astropy.stats import sigma_clip

from numba import njit
from numpy import (inf, sqrt, ones, zeros_like, concatenate, diff, log, ones_like, float32 as f32,
                   clip, argsort, any, s_, zeros, arccos, nan, isnan, full, pi, sum, repeat, asarray, ndarray,
                   full_like, isfinite, where, atleast_2d, int32)
from tqdm.auto import tqdm

from pytransit.mandelagol_cl import MandelAgolCL
from pytransit.orbits_py import as_from_rhop, i_from_ba
from pytransit.lpf.lpf import BaseLPF

try:
    import pandas as pd
    with_pandas = True
except ImportError:
    with_pandas = False

try:
    from pyde import DiffEvol
    with_pyde = True
except ImportError:
    with_pyde = False

try:
    from emcee import EnsembleSampler
    with_emcee = True
except ImportError:
    with_emcee = False


class OCLBaseLPF(BaseLPF):
    def __init__(self, target: str, passbands: list, times: list = None, fluxes: list = None,
                 pbids: list = None, nsamples: int = 1, exptime: float = 0.020433598, cl_ctx=None, cl_queue=None):

        self.cl_ctx = cl_ctx or self.tm.ctx
        self.cl_queue = cl_queue or self.tm.queue
        super().__init__(target, passbands, times, fluxes, pbids, None, 1, exptime)

        self.tm = MandelAgolCL(self.npb, supersampling=nsamples, exptime=exptime, klims=(0.01, 0.75), nk=512, nz=512,
                               cl_ctx=cl_ctx, cl_queue=cl_queue)


        src = """
           __kernel void lnl2d(const int nlc, __global const float *obs, __global const float *mod, __global const float *err, __global const int *lcids, __global float *lnl2d){
                  uint i_tm = get_global_id(0);    // time vector index
                  uint n_tm = get_global_size(0);  // time vector size
                  uint i_pv = get_global_id(1);    // parameter vector index
                  uint n_pv = get_global_size(1);  // parameter vector population size
                  uint gid = i_tm*n_pv + i_pv;     // global linear index
                  float e = err[i_pv*nlc + lcids[i_tm]];
                  lnl2d[gid] = -log(e) - 0.5f*log(2*M_PI_F) - 0.5f*pown((obs[i_tm]-mod[gid]) / e, 2);
            }

            __kernel void lnl1d(const uint npt, __global const float *lnl2d, __global float *lnl1d){
                  uint i_pv = get_global_id(0);    // parameter vector index
                  uint n_pv = get_global_size(0);  // parameter vector population size

                  lnl1d[i_pv] = 0.f;
                  for(uint i_tm=0; i_tm<npt; i_tm++){
                    lnl1d[i_pv] += lnl2d[i_pv + i_tm*n_pv];
                  }
            }
        """
        self.prg_lnl = cl.Program(self.cl_ctx, src).build()


    def _init_data(self, times, fluxes, pbids):
        super()._init_data(times, fluxes, pbids)
        self.nlc = int32(self.nlc)
        # Initialise the Python arrays
        # ----------------------------
        self.timea = self.timea.astype('f')
        self.ofluxa = self.ofluxa.astype('f')
        self.lnl2d = zeros([self.ofluxa.size, 50], 'f')
        self.ferr = zeros([50, self.nlc])
        self.lcida = self.lcida.astype('int32')

        # Initialise OpenCL buffers
        # -------------------------
        mf = cl.mem_flags
        self._b_flux = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ofluxa)
        self._b_err = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ferr)
        self._b_lnl2d = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, self.lnl2d.nbytes)
        self._b_lcids = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcida)


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
        flux = self.tm.evaluate_t_pv2d(self.timea, pvp_t, uv, copy=copy)
        return flux.T if copy else None

    def flux_model(self, pvp):
        return self.transit_model(pvp, copy=True).astype('d')

    def lnlikelihood(self, pv):
        if self.lnl2d.shape[1] != pv.shape[0]:
            self.err = zeros([pv.shape[0], self.nlc], 'f')
            self._b_err.release()
            self._b_err = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, self.err.nbytes)
            self.lnl2d = zeros([self.ofluxa.size, pv.shape[0]], 'f')
            self._b_lnl2d.release()
            self._b_lnl2d = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, self.lnl2d.nbytes)
        self.transit_model(pv)
        cl.enqueue_copy(self.cl_queue, self._b_err, (10 ** pv[:, self._sl_err]).astype('f'))
        self.prg_lnl.lnl2d(self.cl_queue, self.tm.f.shape, None, self.nlc, self._b_flux, self.tm._b_f,
                           self._b_err, self._b_lcids, self._b_lnl2d)
        cl.enqueue_copy(self.cl_queue, self.lnl2d, self._b_lnl2d)
        lnl = self.lnl2d.astype('d').sum(0)
        return where(isfinite(lnl), lnl, -inf)

    def lnprior(self, pv):
        lnpriors = zeros(pv.shape[0])
        for i, p in enumerate(self.ps.priors):
            lnpriors += p.logpdf(pv[:, i])
        return lnpriors

    def lnposterior(self, pv):
        return self.lnlikelihood(pv) + self.lnprior(pv)

    def optimize_global(self, niter=200, npop=50, population=None, label='Global optimisation', leave=False):
        if not with_pyde:
            raise ImportError("PyDE not installed.")

        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True, vfun=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave):
            pass

    def sample_mcmc(self, niter=500, thin=5, label='MCMC sampling', reset=False, leave=True):
        if not with_emcee:
            raise ImportError('Emcee not installed.')
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior, vectorize=True)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:, -1, :].copy()
        if reset:
            self.sampler.reset()
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label, leave=False):
            pass


    def remove_outliers(self, sigma=5):
        fmodel = self.flux_model(self.de.minimum_location)[0]
        times, fluxes, pbids = [], [], []
        for i in range(len(self.times)):
            res = self.fluxes[i] - fmodel[i]
            mask = ~sigma_clip(res, sigma=sigma).mask
            times.append(self.times[i][mask])
            fluxes.append(self.fluxes[i][mask])
        self._init_data(times, fluxes, self.pbids)