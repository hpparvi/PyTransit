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

from pathlib import Path

import pandas as pd
from matplotlib.pyplot import subplots, setp
from numpy import arange, concatenate, zeros, inf, atleast_2d, where, repeat, squeeze, argsort, isfinite, ndarray, \
    floor, ceil, percentile, median
from numpy.random import uniform, permutation

from pytransit import LinearModelBaseline
from pytransit.contamination import SMContamination, Instrument
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.lpf.cntlpf import PhysContLPF, map_pv_pclpf, map_ldc, contaminate
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, GParameter, PParameter
from pytransit.utils.misc import fold


class BaseTGCLPF(PhysContLPF):
    """Log posterior function that combined TESS light curve with ground-based transit light curves.

    Example log posterior function to model a TESS light curve jointly with with ground-based light curves observed in
    arbitrary passbands. The TESS light curve is assumed to contain an unknown amount of light contamination, and the
    ground-based observations are assumed to all be contaminated by an identical source, but the contamination follows
    a physical passband-dependent model.

    **Note:** This LPF is meant to be inherited by an LPF that implements the `read_data` method.
    """

    def __init__(self, name: str, use_ldtk: bool = False):
        self.result_dir = Path('.')
        self._stess = None
        self._ntess = None

        times, fluxes, pbnames, pbs, wns, covs = self.read_data()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = arange(len(times))
        tref = floor(concatenate(times).min())

        self.wns = wns
        PhysContLPF.__init__(self, name, passbands=pbnames, times=times, fluxes=fluxes, pbids=pbids, wnids=wnids,
                             covariates=covs, tref=tref)

    def read_data(self):
        """Read in the TESS light curve and the ground-based data.

        Read in the TESS light curve and the ground-based data. This method needs to be implemented by the class inheriting
        the base ´TESSGBLPF`.
        """
        raise NotImplementedError

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.01**2, 0.30**2), (0., inf))]
        pcn = [GParameter('k2_true', 'true_area_ratio', 'As', UP(0.01**2, 0.75**2), bounds=(1e-8, inf)),
               GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('k2_app_tess', 'tess_apparent_area_ratio', 'A_s', UP(0.01**2, 0.30**2), (0., inf))]
        ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        ps.add_global_block('contamination', pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._sl_cn = ps.blocks[-1].slice

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def create_pv_population(self, npop: int = 50) -> ndarray:
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 5] = pvp_trial[:, 4]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 5] = pvp_trial[:, 4] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True):
        df = super().posterior_samples(burn, thin, derived_parameters)
        if derived_parameters:
            df['ctess'] = 1 - df.k2_app_tess / df.k2_true
        return df

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.add_prior(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
        self.add_prior(lambda pv: where(pv[:, 8] < pv[:, 5], 0, -inf))

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        pvt = map_pv_pclpf(pvp)
        pvt[:,1] -= self._tref
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        cnref = 1. - pvp[:, 4] / pvp[:, 5]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, 6], pvp[:, 7])
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def plot_folded_tess_transit(self, solution: str = 'de', pv: ndarray = None, binwidth: float = 1,
                                 plot_model: bool = True, plot_unbinned: bool = True, plot_binned: bool = True,
                                 xlim: tuple = None, ylim: tuple = None, ax=None, figsize: tuple = None):

        if pv is None:
            if solution.lower() == 'local':
                pv = self._local_minimization.x
            elif solution.lower() in ('de', 'global'):
                pv = self.de.minimum_location
            elif solution.lower() in ('mcmc', 'mc'):
                pv = self.posterior_samples().median().values
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        ax.autoscale(enable=True, axis='x', tight=True)

        etess = self._ntess
        t = self.timea[:etess]
        fo = self.ofluxa[:etess]
        fm = squeeze(self.transit_model(pv))[:etess]
        bl = squeeze(self.baseline(pv))[:etess]

        phase = 24 * pv[1] * (fold(t, pv[1], pv[0], 0.5) - 0.5)
        sids = argsort(phase)
        phase = phase[sids]
        bp, bf, be = downsample_time(phase, (fo / bl)[sids], binwidth / 60)
        if plot_unbinned:
            ax.plot(phase, (fo / bl)[sids], 'k.', alpha=0.1, ms=2)
        if plot_binned:
            ax.errorbar(bp, bf, be, fmt='ko', ms=3)
        if plot_model:
            ax.plot(phase, fm[sids], 'k')
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - T$_c$ [h]', ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_gb_transits(self, solution: str = 'de', pv: ndarray = None, figsize: tuple = None, axes=None, ncol: int = 4,
                         xlim: tuple = None, ylim: tuple = None, remove_baseline: bool = True, n_samples: int = 1500):

        solution = solution.lower()
        samples = None

        if pv is None:
            if solution == 'local':
                pv = self._local_minimization.x
            elif solution in ('de', 'global'):
                solution = 'global'
                pv = self.de.minimum_location
            elif solution in ('mcmc', 'mc'):
                solution = 'mcmc'
                samples = self.posterior_samples(derived_parameters=False)
                samples = permutation(samples.values)[:n_samples]
                pv = median(samples, 0)
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        nlc = self.nlc - self._stess
        nrow = int(ceil(nlc / ncol))

        if axes is None:
            fig, axs = subplots(nrow, ncol, figsize=figsize, sharex='all', sharey='all', squeeze=False)
        else:
            fig, axs = None, axes

        [ax.autoscale(enable=True, axis='x', tight=True) for ax in axs.flat]

        if remove_baseline:
            if solution == 'mcmc':
                fbasel = median(self.baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.transit_model(samples), [50, 0.5, 99.5], axis=0)
            else:
                fbasel = squeeze(self.baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.transit_model(pv)), None, None
            fobs = self.ofluxa / fbasel
        else:
            if solution == 'mcmc':
                fbasel = median(self.baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.flux_model(samples), [50, 1, 99], axis=0)
            else:
                fbasel = squeeze(self.baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.flux_model(pv)), None, None
            fobs = self.ofluxa

        etess = self._stess
        t0, p = pv[[0, 1]]

        for i in range(nlc):
            ax = axs.flat[i]
            sl = self.lcslices[etess + i]
            t = self.times[etess + i]
            e = epoch(t.mean(), t0, p)
            tc = t0 + e * p
            tt = 24 * (t - tc)
            ax.plot(tt, fobs[sl], 'k.', alpha=0.2)
            ax.plot(tt, fmodel[sl], 'k')

            if solution == 'mcmc':
                ax.fill_between(tt, fmodm[sl], fmodp[sl], zorder=-100, alpha=0.2, fc='k')

            if not remove_baseline:
                ax.plot(tt, fbasel[sl], 'k--', alpha=0.2)

        setp(axs, xlim=xlim, ylim=ylim)
        setp(axs[-1, :], xlabel='Time - T$_c$ [h]')
        setp(axs[:, 0], ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig

