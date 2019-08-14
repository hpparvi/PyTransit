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

import seaborn as sb
from matplotlib.pyplot import subplots, setp
from numpy import pi, sign, cos, sqrt, sin, array, arccos, inf, round, int, s_, percentile, concatenate, median, mean, \
    arange, poly1d, polyfit

from numba import njit, prange
from .lpf import BaseLPF
from ..models.transitmodel import TransitModel
from ..param.parameter import ParameterSet, PParameter, GParameter
from ..param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from ..orbits.orbits_py import as_from_rhop

with_seaborn = True

@njit("f8[:](f8[:], f8, f8, f8, f8[:], i8[:])", cache=False, parallel=False)
def z_circular_ttv(t, p, a, i, tc, tcid):
    cosph = cos(2*pi * (t - tc[tcid]) / p)
    z = sign(cosph) * a * sqrt(1.0 - cosph * cosph * sin(i) ** 2)
    return z


def plot_estimates(x, p, ax, bwidth=0.8):
    ax.bar(x, p[4, :] - p[3, :], bwidth, p[3, :], alpha=0.25, fc='b')
    ax.bar(x, p[2, :] - p[1, :], bwidth, p[1, :], alpha=0.25, fc='b')
    [ax.plot((xx - 0.47 * bwidth, xx + 0.47 * bwidth), (pp[[0, 0]]), 'k') for xx, pp in zip(x, p.T)]

class TTVLPF(BaseLPF):
    """Log posterior function for TTV estimation.

    A log posterior function for TTV estimation. Each light curve represents a single transit, and
    is given a separate free transit centre parameter. The average orbital period and (one) transit
    zero epoch are assumed as known.

    Notes: The number of parameters can grow large with Kepler short-period planets.

    """
    def __init__(self, target: str, zero_epoch: float, period: float, tc_sigma: float, passbands: list,
                 times: list = None, fluxes: list = None, pbids: list = None, tm: TransitModel = None,
                 nsamples: int = 1, exptime: float = 0.020433598):
        self.zero_epoch = zero_epoch
        self.period = period
        self.tc_sigma = tc_sigma
        super().__init__(target, passbands, times, fluxes, pbids, tm, nsamples, exptime)

    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.
        """
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 25.0), (0, inf)),
                  GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        s = self.tc_sigma
        self.tnumber = round((array([t.mean() for t in self.times]) - self.zero_epoch) / self.period).astype(int)
        tcs = self.period * self.tnumber + self.zero_epoch
        for tc, tn in zip(tcs, self.tnumber):
            porbit.append(GParameter(f'tc_{tn:d}', f'transit_centre_{tn:d}', 'd', N(tc, s), (-inf, inf)))
        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]

    def optimize_times(self, window):
        times, fluxes, pbids = [], [], []
        tcp = self.ps[self._sl_tc]
        for i in range(self.nlc):
            tc = tcp[i].prior.mean
            mask = abs(self.times[i] - tc) < 0.5*window/24.
            times.append(self.times[i][mask])
            fluxes.append(self.fluxes[i][mask])
        self._init_data(times, fluxes, self.pbids)

    def _compute_z(self, pv):
        a = as_from_rhop(pv[0], self.period)
        if a < 1.:
            return None
        else:
            i = arccos(pv[1] / a)
            tc = pv[self._sl_tc]
            return z_circular_ttv(self.timea, self.period, a, i, tc, self.lcida)

    def plot_light_curve(self, ax=None, figsize=None, time=False):
        fig, ax = (None, ax) if ax is not None else subplots(figsize=figsize)
        time = self.timea_orig if time else arange(self.timea_orig.size)
        ax.plot(time, concatenate(self.fluxes))
        ax.plot(time, concatenate(self.flux_model(self.de.minimum_location)))
        fig.tight_layout()
        return ax

    def posterior_period(self, burn: int = 0, thin: int = 1) -> float:
        df = self.posterior_samples(burn, thin, derived_parameters=False)
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
        lineph = poly1d(polyfit(self.tnumber, tcs, 1))
        tc_linear = lineph(self.tnumber)
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
