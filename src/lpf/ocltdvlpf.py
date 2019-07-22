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
import seaborn as sb

from matplotlib.pyplot import subplots, setp
from numpy import sqrt, array, inf, int, s_, percentile, median, mean, round, zeros, isfinite, where, atleast_2d, ceil, \
    newaxis

from ..param.parameter import GParameter, LParameter
from ..param.parameter import UniformPrior as UP, NormalPrior as NP
from .oclttvlpf import OCLTTVLPF, plot_estimates
from ..orbits.orbits_py import as_from_rhop, i_from_ba, p_from_dkaiews

with_seaborn = True


class OCLTDVLPF(OCLTTVLPF):
    def __init__(self, target: str, zero_epoch: float, period: float, duration_prior: tuple,
                 passbands: list, times: list = None, fluxes: list = None, errors: list = None, pbids: list = None,
                 nsamples: int = 1, exptime: float = 0.020433598, cl_ctx=None, cl_queue=None):
        self.t14_prior = duration_prior
        super().__init__(target, zero_epoch, period, passbands, times, fluxes, errors, pbids, nsamples, exptime, cl_ctx,
                         cl_queue)

    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.
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

        # Transit durations
        # -----------------
        for tn in self.tnumber:
            porbit.append(GParameter(f't14_{tn:d}', f'duration_{tn:d}', 'd', NP(*self.t14_prior), (0, inf)))

        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]
        self._start_td = 2 + self.nlc
        self._sl_td = s_[self._start_td:self._start_td + self.nlc]

    def transit_model(self, pvp, copy=False):
        pvp = atleast_2d(pvp)
        pvp_cl = zeros([pvp.shape[0], 5 + 2 * self.nlc], "f")  # k [tc] [p] a i e w
        uv = zeros([pvp.shape[0], 2], "f")
        tc_end = 1 + self.nlc
        td_end = 1 + 2 * self.nlc
        pvp_cl[:, 0:1] = k = sqrt(pvp[:, self._pid_k2])  # Radius ratio
        pvp_cl[:, 1:tc_end] = pvp[:, self._sl_tc]  # Transit centres
        pvp_cl[:, td_end + 0] = a = as_from_rhop(pvp[:, 0], self.period)
        pvp_cl[:, td_end + 1] = i = i_from_ba(pvp[:, 1], a)
        pvp_cl[:, tc_end:td_end] = p_from_dkaiews(pvp[:, self._sl_td], k, a[:, newaxis], i[:, newaxis], 0., 0.,
                                                  1)  # Orbital periods
        a, b = sqrt(pvp[:, self._sl_ld][:, 0]), 2. * pvp[:, self._sl_ld][:, 1]
        uv[:, 0] = a * b
        uv[:, 1] = a * (1. - b)
        flux = self.tm.evaluate_t_pv2d_ttv(self.timea, pvp_cl, uv, self.lcids, self.nlc, copy=copy, tdv=True)
        return flux.T if copy else None


    def plot_tdvs(self, burn=0, thin=1, axs=None, figsize=None, bwidth=0.8, fmt='h', windows=None):
        assert fmt in ('d', 'h', 'min')
        multiplier = {'d': 1, 'h': 24, 'min': 1440}
        ncol = 1 if windows is None else len(windows)
        fig, axs = (None, axs) if axs is not None else subplots(1, ncol, figsize=figsize, sharey=True)
        df = self.posterior_samples(burn, thin)
        dcols = [c for c in df.columns if 't14_' in c]
        p = multiplier[fmt] * df[dcols].quantile([0.50, 0.16, 0.84, 0.005, 0.995]).values
        setp(axs, ylabel='Transit duration [{}]'.format(fmt), xlabel='Transit number')

        if windows is None:
            plot_estimates(self.tnumber, p, axs)
        else:
            setp(axs[1:], ylabel='')
            for ax, w in zip(axs, windows):
                m = (self.tnumber > w[0]) & (self.tnumber < w[1])
                plot_estimates(self.tnumber[m], p[:, m], ax)
                setp(ax, xlim=w)
                if with_seaborn:
                    sb.despine(ax=ax, offset=15)

        if fig:
            fig.tight_layout()

        return axs
