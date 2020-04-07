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

import astropy.units as u

from pathlib import Path

from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from corner import corner
from matplotlib.pyplot import setp
from matplotlib.pyplot import subplots
from numba import njit
from numpy import zeros, squeeze, ceil, arange, digitize, full, nan, \
    sqrt, percentile, isfinite, floor, argsort, ones_like, atleast_2d, median, ndarray, unique, nanmedian
from numpy.random import permutation

from .loglikelihood import CeleriteLogLikelihood
from .lpf import BaseLPF
from .. import TransitModel
from ..orbits import epoch
from ..utils.keplerlc import KeplerLC
from ..utils.misc import fold

try:
    from ldtk import tess
    with_ldtk = True
except ImportError:
    with_ldtk = False

@njit
def downsample_time(time, vals, inttime=1.):
    duration = time.max() - time.min()
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = time[0] + bins * inttime
    bids = digitize(time, edges) - 1
    bt, bv, be = full(nbins, nan), zeros(nbins), zeros(nbins)
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i] = vals[bmask].mean()
            if bmask.sum() > 2:
                be[i] = vals[bmask].std() / sqrt(bmask.sum())
            else:
                be[i] = nan
    m = isfinite(be)
    return bt[m], bv[m], be[m]


class TESSLPF(BaseLPF):
    bjdrefi = 2457000

    def __init__(self, name: str, dfile: Path = None, tic: int = None, zero_epoch: float = None, period: float = None,
                 nsamples: int = 2, trdur: float = 0.125, bldur: float = 0.3, use_pdc=True,
                 split_transits=True, separate_noise=False, tm: TransitModel = None, minpt=10):

        self.zero_epoch = zero_epoch
        self.period = period

        if tic is not None:
            from lightkurve import search_lightcurvefile
            print("Searching for TESS light curves")
            lcf = search_lightcurvefile(tic, mission='TESS')
            print(f"Found {len(lcf)} TESS light curves")
            print(f"Downloading TESS light curves")
            lcs = lcf.download_all()
            if use_pdc:
                ts = lcs.PDCSAP_FLUX.stitch().normalize().to_timeseries()
            else:
                ts = lcs.SAP_FLUX.stitch().normalize().to_timeseries()
        elif dfile is not None:
            ts = TimeSeries.read(dfile, format='tess.fits')
            if use_pdc:
                ts = ts['time', 'pdcsap_flux', 'pdcsap_flux_err']
            else:
                ts = ts['time', 'sap_flux', 'sap_flux_err']
            ts.rename_columns(ts.colnames, 'time flux flux_err'.split())
            m = ~isfinite(ts['flux'])
            self.normalization = (sigma_clipped_stats(ts['flux'], mask=m)[1]).value
            ts['flux_err'] /= self.normalization
            ts['flux'] /= self.normalization
            ts['flux'].mask = m

        else:
            raise NotImplementedError("Need to give either a TIC or a SPOC light curve file")

        tref = floor(ts.time.jd.min())

        self.period = period = period if isinstance(period, u.Quantity) else u.d * period
        self.zero_epoch = zero_epoch = zero_epoch if isinstance(zero_epoch, Time) else Time(zero_epoch, format='jd',
                                                                                            scale='tdb')

        ts_folded = ts.fold(period, zero_epoch)
        mwindow = abs(ts_folded.time.jd) < 0.5 * bldur
        mint = abs(ts_folded.time.jd) < 0.5 * trdur
        moot = mwindow & ~mint

        self.transit_duration = trdur
        self.baseline_duration = bldur

        bm = ~ts['flux'].mask & mwindow

        if split_transits:
            ep = epoch(ts.time.jd, zero_epoch.jd, period)
            ep -= ep.min()

            times, fluxes = [], []
            for e in unique(ep):
                m = bm & (ep == e)
                if m.sum() >= minpt:
                    times.append(ts.time.jd[m].astype('d'))
                    try:
                        fluxes.append(ts['flux'].data.data[m].astype('d'))
                    except AttributeError:
                        fluxes.append(ts['flux'].value[m].astype('d'))

            pbids = len(times) * [0]
        else:
            times, fluxes = [ts.time.jd[bm]], [ts['flux'].data.data[bm].astype('d')]
            pbids = [0]

        wnids = arange(len(times)) if separate_noise else None
        BaseLPF.__init__(self, name, ['TESS'], times=times, fluxes=fluxes, pbids=pbids,
                         nsamples=nsamples, exptimes=[0.00139], wnids=wnids, tref=tref, tm=tm)

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(CeleriteLogLikelihood(self))

    def add_ldtk_prior(teff, logg, z):
        if with_ldtk:
            super().add_ldtk_prior(teff, logg, z, passbands=(tess,))
        else:
            raise ImportError('Could not import LDTk, cannot add an LDTk prior.')

    def plot_individual_transits(self, solution: str = 'de', pv: ndarray = None, ncols: int = 2, n_samples: int = 100,
                                 xlim: tuple = None, ylim: tuple = None, axs=None, figsize: tuple = None,
                                 remove_baseline: bool = False):

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

        t0 = floor(self.times[0].min())
        nrows = int(ceil(self.nlc / ncols))

        if axs is None:
            fig, axs = subplots(nrows, ncols, figsize=figsize, sharey=True, constrained_layout=True)
        else:
            fig, axs = None, axs

        [ax.autoscale(enable=True, axis='x', tight=True) for ax in axs.flat]

        def baseline(pvp):
            pvp = atleast_2d(pvp)
            bl = zeros((pvp.shape[0], self.ofluxa.size))
            for i, pv in enumerate(pvp):
                bl[i] = self._lnlikelihood_models[0].predict_baseline(pv)
            return bl

        if remove_baseline:
            if solution == 'mcmc':
                fbasel = median(baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.transit_model(samples), [50, 0.5, 99.5], axis=0)
            else:
                fbasel = squeeze(baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.transit_model(pv)), None, None
            fobs = self.ofluxa / fbasel
        else:
            if solution == 'mcmc':
                fbasel = median(baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.flux_model(samples), [50, 1, 99], axis=0)
            else:
                fbasel = squeeze(baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.flux_model(pv)), None, None
            fobs = self.ofluxa

        t0, p = pv[[0, 1]]

        for i, sl in enumerate(self.lcslices):
            ax = axs.flat[i]
            t = self.times[i]
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
        return fig

    def plot_folded_transit(self, method='de', figsize=(13, 6), ylim=(0.9975, 1.002), xlim=None, binwidth=8,
                            remove_baseline: bool = False):
        if method == 'de':
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, 0.5)
        binwidth = binwidth / 24 / 60
        sids = argsort(phase)

        tm = self.transit_model(pv)

        if remove_baseline:
            gp = self._lnlikelihood_models[0]
            bl = squeeze(gp.predict_baseline(pv))
        else:
            bl = ones_like(self.ofluxa)

        bp, bfo, beo = downsample_time(phase[sids], (self.ofluxa / bl)[sids], binwidth)

        fig, ax = subplots(figsize=figsize)
        ax.plot(phase - 0.5 * p, self.ofluxa / bl, '.', alpha=0.15)
        ax.errorbar(bp - 0.5 * p, bfo, beo, fmt='ko')
        ax.plot(phase[sids] - 0.5 * p, tm[sids], 'k')
        xlim = xlim if xlim is not None else 1.01 * (bp[isfinite(bp)][[0, -1]] - 0.5 * p)
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - Tc [d]', ylabel='Normalised flux')
        fig.tight_layout()

    def plot_basic_posteriors(self):
        df = self.posterior_samples()
        corner(df['tc p rho b k'.split()],
               labels='Zero epoch, Period, Stellar density, impact parameter, radius ratio'.split(', '))