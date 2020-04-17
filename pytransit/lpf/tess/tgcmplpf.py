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
#  GNU General Public License for more details.s
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List

from numba import njit
from matplotlib.pyplot import subplots, setp
from numpy import arange, zeros, inf, atleast_2d, squeeze, argsort, ndarray, \
    floor, sqrt, ones, ones_like, array, argmin, arctan2, percentile, median, ceil, where
from numpy.random.mtrand import permutation
from uncertainties import ufloat

from pytransit.lpf.cntlpf import map_ldc, contaminate
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch, as_from_rhop, i_from_ba
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, GParameter
from pytransit.utils.misc import fold
from .tgclpf import BaseTGCLPF

@njit(fastmath=True)
def map_pv(pv, pids, ipl):
    pv = atleast_2d(pv)
    pp = pv[:, pids[ipl]]
    pvt = zeros((pv.shape[0], 7))
    pvt[:, 0] = sqrt(pp[:, 0])                    # 0 - radius ratio
    pvt[:, 1] = pp[:, 1]                          # 1 - zero epoch
    pvt[:, 2] = pp[:, 2]                          # 2 - period
    pvt[:, 3] = as_from_rhop(pp[:, 4], pp[:, 2])  # 3 - scaled semi-major axis
    pvt[:, 4] = i_from_ba(pp[:, 3], pvt[:, 3])    # 4 - inclination
    pvt[:, 5] = pp[:,5]**2 + pp[:,6]**2           # 5 - eccentricity
    pvt[:, 6] = arctan2(pp[:,6], pp[:,5])         # 6 - argument of periastron
    return pvt


class TGCMPLPF(BaseTGCLPF):

    def __init__(self, name: str, nplanets: int, zero_epochs: List, periods: List,
                 nsamples: int = 2, bldur: float = 0.1, trdur: float = 0.04, use_pdc: bool = False):

        for i, t0 in enumerate(zero_epochs):
            if isinstance(t0, float):
                zero_epochs[i] = ufloat(t0, 1e-3)

        for i, pr in enumerate(periods):
            if isinstance(pr, float):
                periods[i] = ufloat(pr, 1e-5)

        self.zero_epochs = zero_epochs
        self.periods = periods

        self.nplanets = nplanets
        self.nsamples = nsamples
        self.bldur = bldur
        self.trdur = trdur
        self.use_pdc = use_pdc

        self._stess = None
        self._etess = None

        super().__init__(name)

        for i in range(self.nplanets):
            self.set_prior(f'zero_epoch_{i+1}', NP(zero_epochs[i].n, 3 * zero_epochs[i].s))
            self.set_prior(f'period_{i+1}', NP(periods[i].n, 3 * periods[i].s))

        self.pids = zeros((self.nplanets, 7), 'int')
        for ipl in range(self.nplanets):
            pnames = f"k2_true_{ipl+1} tc_{ipl+1} p_{ipl+1} b_{ipl+1} rho secw_{ipl+1} sesw_{ipl+1}".split()
            for ip, pname in enumerate(pnames):
                self.pids[ipl, ip] = self.ps.names.index(pname)

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        ps = self.ps
        pcn = [GParameter('ctess', 'tess_contamination', '', UP(0.0, 0.99), (0, 1)),
               GParameter('cgref', 'ground_based_contamination', '', UP(0.0, 0.99), bounds=(0, 1)),
               GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000))]
        ps.add_global_block('contamination', pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._start_cn = ps.blocks[-1].start
        self._sl_cn = ps.blocks[-1].slice

        pp = []
        for i in range(1, self.nplanets + 1):
            pp.extend([
                GParameter(f'k2_true_{i}', f'true_area_ratio_{i}', 'A_s', UP(0.02 ** 2, 0.1 ** 2),
                           (0.02 ** 2, 0.1 ** 2)),
                GParameter(f'tc_{i}', f'zero_epoch_{i}', 'd', NP(0.0, 0.1), (-inf, inf)),
                GParameter(f'p_{i}', f'period_{i}', 'd', NP(1.0, 1e-5), (0, inf)),
                GParameter(f'b_{i}', f'impact_parameter_{i}', 'R_s', UP(0.0, 1.0), (0, inf)),
                GParameter(f'secw_{i}', f'sqrt_e_cos_w_{i}', '', NP(0.0, 1e-8), (-1, 1)),
                GParameter(f'sesw_{i}', f'sqrt_e_sin_w_{i}', '', NP(0.0, 1e-8), (-1, 1))
            ])
        ps.add_global_block('planets', pp)
        self._start_pl = ps.blocks[-1].start
        self._sl_pl = ps.blocks[-1].slice

    def create_pv_population(self, npop: int = 50) -> ndarray:
        return self.ps.sample_from_prior(npop)

    def transit_model(self, pvp, copy=True, planets=None):
        pvp = atleast_2d(pvp)
        flux = ones([pvp.shape[0], self.timea.size])
        ldc = map_ldc(pvp[:, self._sl_ld])
        planets = planets if planets is not None else arange(self.nplanets)
        for i in planets:
            pvpl = map_pv(pvp, self.pids, i)
            pvpl[:,1] -= self._tref
            flux += self.tm.evaluate_pv(pvpl, ldc, copy) - 1.

        pvc = pvp[:, self._sl_cn]
        cnt = zeros((pvp.shape[0], self.npb))
        cnt[:, 0] = pvc[:, 0]

        for i, pv in enumerate(pvc):
            if (2500 < pv[2] < 12000) and (2500 < pv[3] < 12000):
                cnt[i, 1:] = self.cm.contamination(pv[1], pv[2], pv[3])
            else:
                cnt[i, 1:] = -inf
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def plot_gb_transits(self, method='de', pv: ndarray = None, remove_baseline: bool = True,
                         figsize: tuple = (14, 2), axes=None, ncol: int = 4,
                         xlim: tuple = None, ylim: tuple = None, nsamples: int = 200):

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

        if pvp is None:
            if remove_baseline:
                fobs = self.ofluxa / squeeze(self.baseline(pv))
                fmodel = squeeze(self.transit_model(pv))
                fbasel = ones_like(self.ofluxa)
            else:
                fobs = self.ofluxa
                fmodel = squeeze(self.flux_model(pv))
                fbasel = squeeze(self.baseline(pv))
            fmodel_limits = None
        else:
            if remove_baseline:
                fobs = self.ofluxa / squeeze(self.baseline(pv))
                fmodels = percentile(self.transit_model(pvp), [50, 16, 84, 2.5, 97.5], 0)
                fbasel = ones_like(self.ofluxa)
            else:
                fobs = self.ofluxa
                fmodels = percentile(self.flux_model(pvp), [50, 16, 84, 2.5, 97.5], 0)
                fbasel = median(self.baseline(pvp), 0)
            fmodel = fmodels[0]
            fmodel_limits = fmodels[1:]

        tcids = [self.ps.names.index(f'tc_{i + 1}') for i in range(self.nplanets)]
        prids = [self.ps.names.index(f'p_{i + 1}') for i in range(self.nplanets)]

        t0s = pv[tcids]
        prs = pv[prids]

        tcs = array([t.mean() for t in self.times[self._stess:]])
        tds = array([abs(fold(tcs, prs[i], t0s[i], 0.5) - 0.5) for i in range(self.nplanets)])
        pids = argmin(tds, 0)

        nlc = self.nlc - self._stess
        nrow = int(ceil(nlc / ncol))

        if axes is None:
            fig, axs = subplots(nrow, ncol, figsize=figsize, sharex='all', sharey='all', squeeze=False)
        else:
            fig, axs = None, axes

        [ax.autoscale(enable=True, axis='x', tight=True) for ax in axs.flat]

        etess = self._stess
        for iax, i in enumerate(range(self.nlc - etess)):
            ax = axs.flat[iax]
            sl = self.lcslices[etess + i]
            t = self.times[etess + i]
            e = epoch(t.mean(), t0s[pids[i]], prs[pids[i]])
            tc = t0s[pids[i]] + e * prs[pids[i]]
            tt = 24 * (t - tc)

            if fmodel_limits is not None:
                ax.fill_between(tt, fmodel_limits[2, sl], fmodel_limits[3, sl], facecolor='blue', alpha=0.15)
                ax.fill_between(tt, fmodel_limits[0, sl], fmodel_limits[1, sl], facecolor='darkblue', alpha=0.25)
            ax.plot(tt, fobs[sl], 'k.', alpha=0.2)
            ax.plot(tt, fmodel[sl], 'k')

        setp(axs, xlim=xlim, ylim=ylim)
        setp(axs[-1, :], xlabel='Time - T$_c$ [h]')
        setp(axs[:, 0], ylabel='Normalised flux')
        fig.tight_layout()
        return fig

    def plot_folded_planets(self, passband: str, method: str = 'de', bwidth: float = 10, axs=None, pv: ndarray = None,
                            nsamples: int = 100, limp=(2.5, 97.5, 16, 84), limc: str = 'darkblue', lima: float = 0.15,
                            ylines=None):
        from pytransit.lpf.tesslpf import downsample_time

        if axs is None:
            fig, axs = subplots(1, self.nplanets, sharey='all')
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

        is_pb = self.pbids == self.passbands.index(passband)
        pbmask = zeros(self.timea.size, 'bool')
        for sl, cpb in zip(self.lcslices, is_pb):
            if cpb:
                pbmask[sl] = 1

        tcids = [self.ps.names.index(f'tc_{i + 1}') for i in range(self.nplanets)]
        prids = [self.ps.names.index(f'p_{i + 1}') for i in range(self.nplanets)]
        t0s = pv[tcids]
        prs = pv[prids]

        for ipl in range(self.nplanets):
            planets = set(arange(self.nplanets))
            planets.remove(ipl)
            if pvp is None:
                mflux = squeeze(self.transit_model(pv, planets=[ipl]))[pbmask]
                rflux = squeeze(self.transit_model(pv, planets=planets))[pbmask]
                mflim = None
                fbline = self.baseline(pv)[pbmask]
            else:
                mfluxes = self.transit_model(pvp, planets=[ipl])[:, pbmask]
                rfluxes = self.transit_model(pvp, planets=planets)[:, pbmask]
                fblines = self.baseline(pvp)[:, pbmask]
                mflux = median(mfluxes, 0)
                mflim = percentile(mfluxes, limp, 0)
                rflux = median(rfluxes, 0)
                fbline = median(fblines, 0)

            oflux = self.ofluxa[pbmask] / rflux / fbline
            phase = (fold(self.timea[pbmask], prs[ipl], t0s[ipl], 0.5) - 0.5) * prs[ipl]
            m = abs(phase) < 0.5 * self.bldur
            sids = argsort(phase[m])
            if m.sum() > 0:
                phase, mflux, oflux = phase[m][sids], mflux[m][sids], oflux[m][sids]
                bp, bf, be = downsample_time(phase, oflux, bwidth / 24 / 60)
                if mflim is not None:
                    for il in range(mflim.shape[0] // 2):
                        axs[ipl].fill_between(phase, mflim[2 * il, m][sids], mflim[2 * il + 1, m][sids], fc=limc,
                                              alpha=lima)
                axs[ipl].errorbar(bp, bf, be, fmt='ok')
                axs[ipl].plot(phase, oflux, '.', alpha=0.25)
                axs[ipl].plot(phase, mflux, 'k')

                if ylines is not None:
                    axs[ipl].fill_between(phase, mflux, 1, fc='w', zorder=-99)
                    for yl in ylines:
                        axs[ipl].axhline(yl, lw=1, ls='--', c='k', alpha=0.5, zorder=-100)

        setp(axs, xlim=(-0.5 * self.bldur, 0.5 * self.bldur), ylim=(0.996, 1.004))
        if fig is not None:
            fig.tight_layout()
        return fig