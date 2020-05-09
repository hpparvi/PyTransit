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
from typing import Iterable, Optional

from matplotlib.pyplot import subplots
from numba import njit
from numpy import zeros, log, pi, inf, atleast_2d, arange, arctan2, cos, squeeze, median, linspace, \
    percentile, argsort, sum, concatenate, full, sqrt
from numpy.random.mtrand import permutation

from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.utils.misc import fold
from pytransit.param import GParameter, UniformPrior as UP, NormalPrior as NP, ParameterSet
from pytransit.orbits.orbits_py import ta_newton_v


@njit(cache=False)
def lnlike_normal(o, m, e):
    npv = m.shape[0]
    lnl = zeros(npv)
    for ipv in range(npv):
        lnl[ipv] = -sum(log(e[ipv])) - 0.5*o.size*log(2.*pi) - 0.5*sum((o-m[ipv])**2/e[ipv]**2)
    return lnl


class RVModel:
    """Radial velocity model.
    """

    def __init__(self, lpf: LogPosteriorFunction, nplanets: int,
                 times, rvs: Iterable, rves: Iterable, rvis: Iterable,
                 tref: Optional[float] = None):
        self.lpf = lpf

        if hasattr(lpf, 'nplanets'):
            assert lpf.nplanets == nplanets
        self.nplanets = nplanets

        if hasattr(lpf, '_tref'):
            assert tref is None
            self._tref = lpf._tref
        else:
            self._tref = lpf._tref = tref if tref is not None else 0.0

        self.times = None  # Mid-measurement times
        self.rvs = None  # RV values
        self.rves = None  # RV errors
        self.rvis = None  # RV instrument indices

        self._timea = None
        self._rva = None
        self._rvea = None
        self._rv_ids = None

        self.setup_data(times, rvs, rves, rvis)

        if hasattr(lpf, 'ps') and lpf.ps is not None:
            self.ps = lpf.ps
        else:
            self._init_parameters()
            lpf.ps = self.ps

        self._init_p_rv()

    def setup_data(self, times: Iterable, rvs: Iterable, rves: Iterable, rvis: Iterable):
        self.times = times
        self.rvs = rvs
        self.rves = rves
        self.rvis = rvis

        self._timea = concatenate(times) - self._tref
        self._rva = concatenate(rvs)
        self._rvea = concatenate(rves)
        self._rv_ids = concatenate([full(c.size, i) for i, c in enumerate(self.times)])

    def _init_parameters(self):
        self.ps = ps = ParameterSet([])
        pp = []
        for i in range(1, self.nplanets + 1):
            pp.extend([
                GParameter(f'tc_{i}', f'zero_epoch_{i}', 'd', NP(0.0, 0.1), (-inf, inf)),
                GParameter(f'p_{i}', f'period_{i}', 'd', NP(1.0, 1e-5), (0, inf)),
                GParameter(f'secw_{i}', f'sqrt(e)_cos(w)_{i}', 'rad', UP(-1.0, 1.0), (-1, 1)),
                GParameter(f'sesw_{i}', f'sqrt(e)_sin(w)_{i}', '', UP(-1.0, 1.0), (-1, 1)),
            ])
        ps.add_global_block('planets', pp)
        self._start_pl = ps.blocks[-1].start
        self._sl_pl = ps.blocks[-1].slice

    def _init_p_rv(self):
        self.ps.thaw()
        ps = self.ps
        prv = [GParameter(f'rv_shift_{self.rvis[i]}', 'systemic_velocity', 'm/s', NP(0.0, 0.1), (-inf, inf)) for i in
               range(len(self.times))]
        ps.add_global_block('rv_shifts', prv)
        self._start_rvs = ps.blocks[-1].start
        self._sl_rvs = ps.blocks[-1].slice

        prv = [GParameter(f'rv_err_{self.rvis[i]}', 'additional rv error', 'm/s', UP(0.0, 1.0), (-inf, inf)) for i in
               range(len(self.times))]
        ps.add_global_block('rv_errors', prv)
        self._start_rv_err = ps.blocks[-1].start
        self._sl_rv_err = ps.blocks[-1].slice

        prv = []
        for i in range(1, self.nplanets + 1):
            prv.append(GParameter(f'rv_k_{i}', f'rv_semiamplitude_{i}', 'm/s', UP(0.0, 1.0), (0, inf)))
        ps.add_global_block('rv_semiamplitudes', prv)
        self._start_rvk = ps.blocks[-1].start
        self._sl_rvk = ps.blocks[-1].slice
        self.ps.freeze()

        pnames = "rv_k_{} tc_{} p_{} secw_{} sesw_{}".split()
        self.pids = zeros((self.nplanets, 5), 'int')
        for ipl in range(self.nplanets):
            for ip, p in enumerate(pnames):
                name = p.format(ipl + 1)
                self.pids[ipl, ip] = self.ps.names.index(name)

    def rv_shifts(self, pvp):
        pvp = atleast_2d(pvp)
        return squeeze(pvp[:, self._sl_rvs][:, self._rv_ids])

    def rv_model(self, pvp, times=None, planets=None, add_sv=True):
        times = self._timea if times is None else times - self._tref
        pvp = atleast_2d(pvp)
        rvs = zeros((pvp.shape[0], times.size))

        planets = planets if planets is not None else arange(self.nplanets)
        for ipl in planets:
            pv = pvp[:, self.pids[ipl]]
            tc = pv[:, 1] - self._tref
            p = pv[:, 2]
            e = pv[:, 3] ** 2 + pv[:, 4] ** 2
            w = arctan2(pv[:, 4], pv[:, 3])
            for ipv in range(pv.shape[0]):
                ta = ta_newton_v(times, t0=tc[ipv], p=p[ipv], e=e[ipv], w=w[ipv])
                rvs[ipv] += pv[ipv, 0] * (cos(w[ipv] + ta) + e[ipv] * cos(w[ipv]))
        if add_sv:
            rvs += self.rv_shifts(pvp)
        return squeeze(rvs)

    def lnlikelihood(self, pvp):
        pvp = atleast_2d(pvp)
        errors = sqrt(self._rvea ** 2 + pvp[:, self._sl_rv_err][:, self._rv_ids] ** 2)
        return lnlike_normal(self._rva, self.rv_model(pvp), errors)

    def plot_rv_vs_time(self, method='de', pv=None, nsamples: int = 200, ntimes: int = 500, axs=None):

        if axs is None:
            fig, axs = subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)}, sharex='all')
        else:
            fig, axs = None, axs

        if pv is None:
            if method == 'de':
                if self.lpf.de is None:
                    raise ValueError("The global optimizer hasn't been initialized.")
                pvp = None
                pv = self.lpf.de.minimum_location
            elif method == 'mcmc':
                if self.lpf.sampler is None:
                    raise ValueError("The sampler hasn't been initialized.")
                df = self.lpf.posterior_samples()
                pvp = permutation(df.values)[:nsamples, :]
                pv = median(pvp, 0)
        else:
            if pv.ndim == 1:
                pvp = None
                pv = pv
            else:
                pvp = permutation(pv)[:nsamples, :]
                pv = median(pvp, 0)

        rv_time = linspace(self._timea.min() - 1, self._timea.max() + 1, num=ntimes) + self._tref

        if pvp is None:
            rv_model = self.rv_model(pv, rv_time, add_sv=False)
            rv_model_limits = None
        else:
            rv_percentiles = percentile(self.rv_model(pvp, rv_time, add_sv=False), [50, 16, 84, 2.5, 97.5], 0)
            rv_model = rv_percentiles[0]
            rv_model_limits = rv_percentiles[1:]

        if rv_model_limits is not None:
            axs[0].fill_between(rv_time, rv_model_limits[2], rv_model_limits[3], facecolor='blue', alpha=0.25)
            axs[0].fill_between(rv_time, rv_model_limits[0], rv_model_limits[1], facecolor='darkblue', alpha=0.5)

        axs[0].plot(rv_time, rv_model, 'k', lw=1)
        axs[0].errorbar(self._timea + self._tref, self._rva + self.rv_shifts(pv), self._rvea, fmt='ok')
        axs[1].errorbar(self._timea + self._tref, self._rva - self.rv_model(pv), self._rvea, fmt='ok')

        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_rv_vs_phase(self, planet: int, method='de', pv=None, nsamples: int = 200, ntimes: int = 500, axs=None):
        if axs is None:
            fig, axs = subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)}, sharex='all')
        else:
            fig, axs = None, axs

        if pv is None:
            if method == 'de':
                if self.lpf.de is None:
                    raise ValueError("The global optimizer hasn't been initialized.")
                pvp = None
                pv = self.lpf.de.minimum_location
            elif method == 'mcmc':
                if self.lpf.sampler is None:
                    raise ValueError("The sampler hasn't been initialized.")
                df = self.lpf.posterior_samples()
                pvp = permutation(df.values)[:nsamples, :]
                pv = median(pvp, 0)
        else:
            if pv.ndim == 1:
                pvp = None
                pv = pv
            else:
                pvp = permutation(pv)[:nsamples, :]
                pv = median(pvp, 0)

        rv_time = linspace(self._timea.min() - 1, self._timea.max() + 1, num=ntimes)

        all_planets = set(range(self.nplanets))
        other_planets = all_planets.difference([planet])

        if pvp is None:
            rv_model = self.rv_model(pv, rv_time + self._tref, [planet], add_sv=False)
            rv_others = self.rv_model(pv, planets=other_planets, add_sv=False)
            rv_model_limits = None
        else:
            rv_percentiles = percentile(self.rv_model(pvp, rv_time + self._tref, [planet], add_sv=False),
                                        [50, 16, 84, 2.5, 97.5], 0)
            rv_model = rv_percentiles[0]
            rv_model_limits = rv_percentiles[1:]
            rv_others = median(self.rv_model(pvp, planets=other_planets, add_sv=False), 0)

        period = pv[self.ps.names.index(f'p_{planet + 1}')]
        tc = pv[self.ps.names.index(f'tc_{planet + 1}')] - self._tref

        phase = (fold(self._timea, period, tc, 0.5) - 0.5) * period
        phase_model = (fold(rv_time, period, tc, 0.5) - 0.5) * period
        msids = argsort(phase_model)

        if pvp is not None:
            axs[0].fill_between(phase_model[msids], rv_model_limits[2, msids], rv_model_limits[3, msids],
                                facecolor='blue',
                                alpha=0.15)
            axs[0].fill_between(phase_model[msids], rv_model_limits[0, msids], rv_model_limits[1, msids],
                                facecolor='darkblue',
                                alpha=0.25)

        axs[0].errorbar(phase, self._rva - rv_others + self.rv_shifts(pv), self._rvea, fmt='ok')
        axs[0].plot(phase_model[msids], rv_model[msids], 'k')
        axs[1].errorbar(phase, self._rva - self.rv_model(pv), self._rvea, fmt='ok')

        axs[0].autoscale(axis='x', tight=True)
        if fig is not None:
            fig.tight_layout()
        return fig
