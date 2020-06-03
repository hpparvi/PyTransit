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
from matplotlib.pyplot import subplots, setp
from numba import njit
from numpy import inf, repeat, atleast_2d, sqrt, arctan2, squeeze, ndarray, argsort, array, unique, arange
from seaborn import despine

from .loglikelihood.fmcloglikelihood import FrozenMultiCeleriteLogLikelihood
from .lpf import BaseLPF
from .. import UniformModel
from .tesslpf import downsample_time
from ..param import GParameter, PParameter, UniformPrior as UP, NormalPrior as NP
from ..orbits import as_from_rhop, i_from_ba, i_from_baew, eclipse_phase, d_from_pkaiews, epoch
from ..utils.misc import fold


@njit
def set_depths(f, pvp, pbids, lcids, ik2, sfr):
    f = atleast_2d(f)
    pvp = atleast_2d(pvp)
    npt = f.shape[1]
    for i in range(npt):
        ipb = pbids[lcids[i]]
        f[:, i] = 1 + (f[:, i] - 1.) * pvp[:, sfr + ipb]
    return f


class EclipseLPF(BaseLPF):

    def _init_p_limb_darkening(self):
        pass

    def _init_p_orbit(self):
        porbit = [
            GParameter('tc', 'zero epoch', 'd', NP(0.0, 0.1), (-inf, inf)),
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact parameter', 'R_s', UP(0.0, 1.0), (0, 1)),
            GParameter('secw', 'sqrt(e) cos(w)', '', UP(-0.5, 0.5), (-1, 1)),
            GParameter('sesw', 'sqrt(e) sin(w)', '', UP(-0.5, 0.5), (-1, 1))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """
        pk2 = [PParameter('k2', 'area ratio', 'A_s', UP(0.01 ** 2, 0.2 ** 2), (0, inf))]
        self.ps.add_global_block('k2', pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, 1)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice
        self._ix_k2 = self._start_k2

        pfr = [PParameter(f'fr_{pb}', 'Flux ratio', '', UP(0, 1), (0, 1)) for pb in self.passbands]
        self.ps.add_passband_block('fr', len(self.passbands), 1, pfr)
        self._pid_fr = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_fr = self.ps.blocks[-1].start
        self._sl_fr = self.ps.blocks[-1].slice

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(FrozenMultiCeleriteLogLikelihood(self))

    def _post_initialisation(self):
        super()._post_initialisation()
        self.tm = UniformModel(eclipse=True)
        self.tm.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        zero_epoch = pv[:, 0] - self._tref
        period = pv[:, 1]
        smaxis = as_from_rhop(pv[:, 2], period)
        inclination = i_from_ba(pv[:, 3], smaxis)
        radius_ratio = sqrt(pv[:, 6:7])
        eccentricity = pv[:, 4] ** 2 + pv[:, 5] ** 2
        omega = arctan2(pv[:, 5], pv[:, 4])
        fmod = self.tm.evaluate(radius_ratio, zero_epoch, period, smaxis, inclination, eccentricity, omega)
        fmod = set_depths(fmod, pv, self.pbids, self.lcids, self._start_k2, self._start_fr)
        return squeeze(fmod)

    def create_pv_population(self, npop):
        return self.ps.sample_from_prior(npop)

    def plot_light_curves(self, pv=None, figsize=None, remove_baseline: bool = False):
        if pv is None:
            if self.de is not None:
                pv = self.de.minimum_location
            else:
                pv = self.ps.mean_pv

        tc, p, rho, b, secw, sesw, k2 = pv[:7]
        a = as_from_rhop(rho, p)
        e = secw ** 2 + sesw ** 2
        w = arctan2(sesw, secw)
        i = i_from_baew(b, a, e, w)
        ec = tc + eclipse_phase(p, i, e, w)
        t14 = d_from_pkaiews(p, sqrt(k2), a, i, e, w, -1)
        eps = array([epoch(t.mean(), ec, p) for t in self.times])
        uep = unique(eps)
        nep = uep.size
        npb = self.npb

        fig, axs = subplots(nep, npb, sharey='all', sharex='all', figsize=figsize)
        emap = {e: ied for e, ied in zip(uep, arange(nep))}
        fmodel = self.flux_model(self.de.minimum_location)
        bline = self._lnlikelihood_models[0].predict_baseline(pv)

        for ilc in range(self.nlc):
            iep = emap[eps[ilc]]
            ipb = self.pbids[ilc]
            ax = axs[iep, ipb]
            time = 24 * (self.times[ilc] - (ec + eps[ilc] * p))
            ax.plot(time, self.fluxes[ilc])
            if remove_baseline:
                ax.plot(time, fmodel[self.lcslices[ilc]], 'k')
            else:
                ax.plot(time, fmodel[self.lcslices[ilc]] + bline[self.lcslices[ilc]] - 1, 'k')

            ax.axvspan(-24 * 0.5 * t14, 24 * 0.5 * t14, alpha=0.25)
        setp(axs[-1], xlabel='Time - T$_c$ [h]')
        setp(axs[:, 0], ylabel='Normalised flux')
        fig.tight_layout()
        return fig

    def plot_folded_transit(self, solution: str = 'de', pv: ndarray = None, binwidth: float = 1,
                            plot_model: bool = True, plot_unbinned: bool = True, plot_binned: bool = True,
                            xlim: tuple = None, ylim: tuple = None, ax=None, figsize: tuple = None, malpha=0.1):

        # TODO: Doesn't take the passband information into account yet -> FIX!

        if pv is None:
            if solution.lower() == 'local':
                pv = self._local_minimization.x
            elif solution.lower() in ('de', 'global'):
                pv = self.de.minimum_location
            elif solution.lower() in ('mcmc', 'mc'):
                pv = self.posterior_samples(derived_parameters=False).median().values
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        ax.autoscale(enable=True, axis='x', tight=True)

        t = self.timea
        fo = self.ofluxa
        fm = squeeze(self.transit_model(pv))
        bl = squeeze(self.baseline(pv))

        phase = 24 * pv[1] * (fold(t, pv[1], pv[0], 0.0) - 0.5)
        sids = argsort(phase)
        phase = phase[sids]
        if plot_unbinned:
            ax.plot(phase, (fo / bl)[sids], 'k.', alpha=malpha, ms=2)
        if plot_binned:
            bp, bf, be = downsample_time(phase, (fo / bl)[sids], binwidth / 60)
            ax.errorbar(bp, bf, be, fmt='ko', ms=3)
        if plot_model:
            ax.plot(phase, fm[sids], 'k')
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - T$_c$ [h]', ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_flux_ratio_posteriors(self, figsize=None):
        df = self.posterior_samples(derived_parameters=False)
        frcols = [c for c in df.columns if 'fr' in c]
        fig, axs = subplots(1, len(frcols), sharex='all', sharey='all', figsize=figsize)
        for i, c in enumerate(frcols):
            axs[i].hist(df[c])
            axs[i].set_xlabel(f"{c.strip('_s').split('_')[-1]} flux ratio")
        setp(axs, yticks=[])
        despine(fig, offset=10)
        fig.tight_layout()
        return fig