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

import scipy.ndimage as ndi
import pyopencl as cl

from matplotlib.pyplot import subplots, setp
from numpy import sqrt, array, inf, int, s_, percentile, median, mean, round, zeros, isfinite, where, atleast_2d, ceil

try:
    import seaborn as sb
    with_seaborn = True
except ImportError:
    with_seaborn = False

from pytransit.param.parameter import GParameter, LParameter
from pytransit.param.parameter import UniformPrior as UP, NormalPrior as NP
from pytransit.lpf.ocllpf import OCLBaseLPF
from pytransit.orbits_py import as_from_rhop, i_from_ba


def plot_estimates(x, p, ax, bwidth=0.8):
    ax.bar(x, p[4, :] - p[3, :], bwidth, p[3, :], alpha=0.25, fc='b')
    ax.bar(x, p[2, :] - p[1, :], bwidth, p[1, :], alpha=0.25, fc='b')
    [ax.plot((xx - 0.47 * bwidth, xx + 0.47 * bwidth), (pp[[0, 0]]), 'k') for xx, pp in zip(x, p.T)]


class OCLTTVLPF(OCLBaseLPF):
    def __init__(self, target: str, zero_epoch: float, period: float, passbands: list,
                 times: list = None, fluxes: list = None, pbids: list = None,
                 nsamples: int = 1, exptime: float = 0.020433598, cl_ctx=None, cl_queue=None):

        self.zero_epoch = zero_epoch
        self.period = period
        self._tc_prior_percentile = 1.

        super().__init__(target, passbands, times, fluxes, pbids, nsamples, exptime, cl_ctx, cl_queue)

        src = """
            __kernel void lnl2d(const int nlc, __global const float *obs, __global const float *mod, __global const float *err, __global const int *lcids, __global float *lnl2d){
                   uint i_tm = get_global_id(1);    // time vector index
                   uint n_tm = get_global_size(1);  // time vector size
                   uint i_pv = get_global_id(0);    // parameter vector index
                   uint n_pv = get_global_size(0);  // parameter vector population size
                   uint gid = i_pv*n_tm + i_tm;     // global linear index
                   float e = err[i_pv];
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
            
                prefetch(lnl, lnp);
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

    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.

        The orbit part of the parameter vector will be [rho b tc_0 tc_1 ... tc_n].

        """

        # Basic orbital parameters
        # ------------------------
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
                  GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]

        # Transit centers
        # ---------------

        def create_tc_prior(t, f, p=5):
            m = f > percentile(f, p)
            m = ~ndi.binary_erosion(m, iterations=6, border_value=1)
            return NP(t[m].mean(), 0.25 * t[m].ptp())

        self.tnumber = round((array([t.mean() for t in self.times]) - self.zero_epoch) / self.period).astype(int)
        for t, f, tn in zip(self.times, self.fluxes, self.tnumber):
            prior = create_tc_prior(t, f, self._tc_prior_percentile)
            porbit.append(GParameter(f'tc_{tn:d}', f'transit_centre_{tn:d}', 'd', prior, (-inf, inf)))

        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]

    def _init_p_noise(self):
        """Initialise a single average white noise parameter shared with all the light curves.

        Initialise a single average white noise estimate for all the light curves. This
        overrides the standard behaviour where each light curve has a separate
        white noise estimate. This class is meant mainly for base-spaced observations
        where the separate transits are cut from long continuous light curves, and the
        noise properties don't change much.
        """
        pns = [LParameter('log_err', 'log_error', '', UP(-8, -0), bounds=(-8, -0))]
        self.ps.add_lightcurve_block('log_err', 1, 1, pns)
        self._sl_err = self.ps.blocks[-1].slice
        self._start_err = self.ps.blocks[-1].start

    def optimize_times(self, window):
        times, fluxes, pbids = [], [], []
        tcp = self.ps[self._sl_tc]
        for i in range(self.nlc):
            tc = tcp[i].prior.mean
            mask = abs(self.times[i] - tc) < 0.5*window/24.
            times.append(self.times[i][mask])
            fluxes.append(self.fluxes[i][mask])
        self._init_data(times, fluxes, self.pbids)

    def lnlikelihood(self, pv):
        if self.lnl2d.shape[1] != pv.shape[0]:
            self.err = zeros(pv.shape[0], 'f')
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

    def transit_model(self, pvp, copy=False):
        pvp = atleast_2d(pvp)
        pvp_cl = zeros([pvp.shape[0], 6 + self.nlc], "f")  # k tc p a i e w
        uv = zeros([pvp.shape[0], 2], "f")
        tc_end = 1 + self.nlc
        pvp_cl[:, 0:1] = sqrt(pvp[:, self._pid_k2])  # Radius ratio
        pvp_cl[:, 1:tc_end] = pvp[:, self._sl_tc]  # Transit centre and orbital period
        pvp_cl[:, tc_end + 0] = self.period
        pvp_cl[:, tc_end + 1] = a = as_from_rhop(pvp[:, 0], self.period)
        pvp_cl[:, tc_end + 2] = i_from_ba(pvp[:, 1], a)
        a, b = sqrt(pvp[:, self._sl_ld][:, 0]), 2. * pvp[:, self._sl_ld][:, 1]
        uv[:, 0] = a * b
        uv[:, 1] = a * (1. - b)
        flux = self.tm.evaluate_t_pv2d_ttv(self.timea, pvp_cl, uv, self.lcida, self.nlc, copy=copy)
        return flux.T if copy else None

    def posterior_period(self, burn: int = 0, thin: int = 1) -> float:
        df = self.posterior_samples(burn, thin, False)
        tccols = [c for c in df.columns if 'tc' in c]
        tcs = median(df[tccols], 0)
        return mean((tcs[1:] - tcs[0]) / (self.tnumber[1:] - self.tnumber[0]))

    def plot_ttvs(self, burn=0, thin=1, axs=None, figsize=None, bwidth=0.8, fmt='h', windows=None):
        assert fmt in ('d', 'h', 'min')
        multiplier = {'d': 1, 'h': 24, 'min': 1440}
        ncol = 1 if windows is None else len(windows)
        fig, axs = (None, axs) if axs is not None else subplots(1, ncol, figsize=figsize, sharey=True)
        df = self.posterior_samples(burn, thin)
        tccols = [c for c in df.columns if 'tc' in c]
        tcs = median(df[tccols], 0)
        period = mean((tcs[1:] - tcs[0]) / (self.tnumber[1:] - self.tnumber[0]))
        tc_linear = self.zero_epoch + self.tnumber * period
        p = multiplier[fmt] * percentile(df[tccols] - tc_linear, [50, 16, 84, 0.5, 99.5], 0)
        setp(axs, ylabel='Transit center - linear prediction [{}]'.format(fmt), xlabel='Transit number')
        if windows is None:
            plot_estimates(self.tnumber, p, axs, bwidth)
            if with_seaborn:
                sb.despine(ax=axs, offset=15)
        else:
            setp(axs[1:], ylabel='')
            for ax, w in zip(axs, windows):
                m = (self.tnumber > w[0]) & (self.tnumber < w[1])
                plot_estimates(self.tnumber[m], p[:, m], ax, bwidth)
                setp(ax, xlim=w)
                if with_seaborn:
                    sb.despine(ax=ax, offset=15)
        if fig:
            fig.tight_layout()
        return axs


    def plot_transits(self, ncols=4, figsize=(13, 11), remove=(), ylim=None):
        nt = len(self.times)
        nrows = int(ceil(nt / ncols))
        fig, axs = subplots(nrows, ncols, figsize=figsize, sharey=True, gridspec_kw=dict(hspace=0.01, wspace=0.01))

        if self.de is not None:
            pv = self.de.minimum_location
            fmodel = self.flux_model(pv).ravel()
        else:
            pv = None
            fmodel = None

        for i in range(nt):
            ax = axs.flat[i]

            # The light curve itself
            # ----------------------
            lc = '0.75' if i in remove else 'k'
            ax.plot(self.times[i], self.fluxes[i], '.', c=lc)

            # The fitted model, if available
            # ------------------------------
            if fmodel is not None:
                m = self.lcida == i
                ax.plot(self.times[i], fmodel[m], 'w', lw=4)
                ax.plot(self.times[i], fmodel[m], 'k', lw=1)

            # Transit centre prior
            # --------------------
            p = self.ps[self._start_tc + i].prior
            [ax.axvspan(p.mean - s * p.std, p.mean + s * p.std, alpha=0.15) for s in (3, 2, 1)]

            if self.de is not None:
                ax.axvline(pv[self._start_tc + i], c='k', zorder=-2)

            # The transit index
            # -----------------
            ax.text(0.05, 0.95, i, ha='left', va='top', transform=ax.transAxes)
            ax.set_xlim(self.times[i][[0, -1]])

        if ylim is not None:
            setp(axs, ylim=ylim)

        for ax in axs.flat[nt:]:
            ax.set_visible(False)

        setp(axs, xticks=[])
        fig.tight_layout()
