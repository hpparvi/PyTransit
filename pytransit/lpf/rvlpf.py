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

from matplotlib.pyplot import subplots
from numba import njit
from numpy import zeros, log, pi, inf, atleast_2d, zeros_like, arange, arctan2, cos, squeeze, median, linspace, \
    percentile, argsort, sum
from numpy.random.mtrand import permutation

from pytransit.utils.misc import fold
from pytransit.param import GParameter, UniformPrior as UP, NormalPrior as NP
from pytransit.orbits.orbits_py import ta_newton_v


@njit(cache=False)
def lnlike_normal(o, m, e):
    npv = m.shape[0]
    lnl = zeros(npv)
    for ipv in range(npv):
        lnl[ipv] = -sum(log(e)) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m[ipv])**2/e**2)
    return lnl


class RVModel:
    """Radial velocity mixin class to allow for customised joint modelling of transits and RVs.
    """

    def __init__(self):
        self.rv_times = None
        self.rv_values = None
        self.rv_errors = None

        self._read_rv_data()

        self.ps.thaw()
        self._init_p_rv()
        self.ps.freeze()

    def _read_rv_data(self):
        raise NotImplementedError

    def _init_p_rv(self):
        ps = self.ps
        prv = [GParameter('rv_sys', 'systemic_velocity', 'm/s', NP(0.0, 0.1), (-inf, inf))]
        for i in range(1, self.nplanets + 1):
            prv.append(GParameter(f'rv_k_{i}', f'rv_semiamplitude_{i}', 'm/s', UP(0.0, 1.0), (0, inf)))
        ps.add_global_block('rv', prv)
        self._start_rv = ps.blocks[-1].start
        self._sl_rv = ps.blocks[-1].slice

    def rv_model(self, pvp, times=None, planets=None, add_sv=True):
        times = self.rv_times if times is None else times
        pvp = atleast_2d(pvp)
        pvp_orbits = pvp[:, self._sl_pl]
        pvp_rvs = pvp[:, self._sl_rv]

        rvs = zeros((pvp.shape[0], times.size))
        tas = zeros_like(rvs)

        planets = planets if planets is not None else arange(self.nplanets)

        for ipl in planets:
            pv = pvp_orbits[:, ipl * 6:(1 + ipl) * 6]
            tc, p = pv[:, 1], pv[:, 2]
            e = pv[:, 4] ** 2 + pv[:, 5] ** 2
            w = arctan2(pv[:, 5], pv[:, 4])
            for ipv in range(pv.shape[0]):
                tas[ipv, :] = ta_newton_v(times, t0=tc[ipv], p=p[ipv], e=e[ipv], w=w[ipv])
                rvs[ipv] += pvp_rvs[ipv, 1 + ipl] * (cos(w[ipv] + tas[ipv]) + e[ipv] * cos(w[ipv]))
        if add_sv:
            rvs += pvp_rvs[:, 0:1]
        return squeeze(rvs)

    def lnlikelihood_rv(self, pv):
        return lnlike_normal(self.rv_values, self.rv_model(pv), self.rv_errors)

    def plot_rv_vs_time(self, method='de', pv=None, nsamples: int = 200, ntimes: int = 500, axs=None):

        if axs is None:
            fig, axs = subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)})
        else:
            fig, axs = None, axs

        if pv is None:
            if method == 'de':
                if self.de is None:
                    raise ValueError("The global optimizer hasn't been initialized.")
                pvp = None
                pv = self.de.minimum_location
            elif method == 'mcmc':
                if self.sampler is None:
                    raise ValueError("The sampler hasn't been initialized.")
                df = self.posterior_samples(derived_parameters=False)
                pvp = permutation(df.values)[:nsamples, :]
                pv = median(pvp, 0)
        else:
            if pv.ndim == 1:
                pvp = None
                pv = pv
            else:
                pvp = permutation(pv)[:nsamples, :]
                pv = median(pvp, 0)

        rv_sys = pv[self._start_rv]
        rv_time = linspace(self.rv_times.min() - 1, self.rv_times.max() + 1, num=ntimes)

        if pvp is None:
            rv_model = self.rv_model(pv, rv_time) - rv_sys
            rv_model_limits = None
        else:
            rv_percentiles = percentile(self.rv_model(pvp, rv_time), [50, 16, 84, 2.5, 97.5], 0) - rv_sys
            rv_model = rv_percentiles[0]
            rv_model_limits = rv_percentiles[1:]

        if rv_model_limits is not None:
            axs[0].fill_between(rv_time, rv_model_limits[2], rv_model_limits[3], facecolor='blue', alpha=0.25)
            axs[0].fill_between(rv_time, rv_model_limits[0], rv_model_limits[1], facecolor='darkblue', alpha=0.5)

        axs[0].plot(rv_time, rv_model, 'k', lw=1)
        axs[0].errorbar(self.rv_times, self.rv_values - rv_sys, self.rv_errors, fmt='ok')
        axs[1].errorbar(self.rv_times, self.rv_values - self.rv_model(pv), self.rv_errors, fmt='ok')

        if fig is not None:
            fig.tight_layout()

        return fig

    def plot_rv_vs_phase(self, planet: int, method='de', pv=None, nsamples: int = 200, ntimes: int = 500, axs=None):
        if axs is None:
            fig, axs = subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)})
        else:
            fig, axs = None, axs

        if pv is None:
            if method == 'de':
                if self.de is None:
                    raise ValueError("The global optimizer hasn't been initialized.")
                pvp = None
                pv = self.de.minimum_location
            elif method == 'mcmc':
                if self.sampler is None:
                    raise ValueError("The sampler hasn't been initialized.")
                df = self.posterior_samples(derived_parameters=False)
                pvp = permutation(df.values)[:nsamples, :]
                pv = median(pvp, 0)
        else:
            if pv.ndim == 1:
                pvp = None
                pv = pv
            else:
                pvp = permutation(pv)[:nsamples, :]
                pv = median(pvp, 0)

        rv_sys = pv[self._start_rv]
        rv_time = linspace(self.rv_times.min() - 1, self.rv_times.max() + 1, num=ntimes)

        all_planets = set(range(self.nplanets))
        other_planets = all_planets.difference([planet])

        if pvp is None:
            rv_model = self.rv_model(pv, rv_time, [planet]) - rv_sys
            rv_others = self.rv_model(pv, planets=other_planets, add_sv=False)
            rv_model_limits = None
        else:
            rv_percentiles = percentile(self.rv_model(pvp, rv_time, [planet]), [50, 16, 84, 2.5, 97.5], 0) - rv_sys
            rv_model = rv_percentiles[0]
            rv_model_limits = rv_percentiles[1:]
            rv_others = median(self.rv_model(pvp, planets=other_planets, add_sv=False), 0)

        period = pv[self.ps.names.index(f'p_{planet + 1}')]
        tc = pv[self.ps.names.index(f'tc_{planet + 1}')]

        phase = (fold(self.rv_times, period, tc, 0.5) - 0.5) * period
        phase_model = (fold(rv_time, period, tc, 0.5) - 0.5) * period
        msids = argsort(phase_model)

        if pvp is not None:
            axs[0].fill_between(phase_model[msids], rv_model_limits[2, msids], rv_model_limits[3, msids],
                                facecolor='blue',
                                alpha=0.15)
            axs[0].fill_between(phase_model[msids], rv_model_limits[0, msids], rv_model_limits[1, msids],
                                facecolor='darkblue',
                                alpha=0.25)
        axs[0].errorbar(phase, self.rv_values - rv_others - rv_sys, self.rv_errors, fmt='ok')
        axs[0].plot(phase_model[msids], rv_model[msids], 'k')
        axs[1].errorbar(phase, self.rv_values - self.rv_model(pv), self.rv_errors, fmt='ok')

        if fig is not None:
            fig.tight_layout()
        return fig