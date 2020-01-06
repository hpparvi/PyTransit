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

from pathlib import Path

from astropy.table import Table
from corner import corner
from matplotlib.pyplot import setp
from matplotlib.pyplot import subplots
from numba import njit
from numpy import zeros, squeeze, ceil, arange, digitize, full, nan, \
    sqrt, percentile, isfinite, floor, argsort
from numpy.random import permutation

from .lpf import BaseLPF
from ..utils.keplerlc import KeplerLC
from ..utils.misc import fold


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
                 nsamples: int = 2, trdur: float = 0.125, bldur: float = 0.3, use_pdc=False,
                 split_transits=True, separate_noise=False):

        if tic is not None:
            from lightkurve import search_lightcurvefile
            lcf = search_lightcurvefile(tic, mission='TESS')
            lc = lcf.download_all()
            if use_pdc:
                lc = lc.PDCSAP_FLUX.stitch().normalize()
            else:
                lc = lc.SAP_FLUX.stitch().normalize()
            time, flux = lc.time.astype('d'), lc.flux.astype('d')

        elif dfile is not None:
            tb = Table.read(dfile)
            self.zero_epoch = zero_epoch - self.bjdrefi
            df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
            time, flux = df.TIME.values, df.PDCSAP_FLUX.values if use_pdc else df.SAP_FLUX.values

        time += self.bjdrefi

        if split_transits:
            self.zero_epoch = zero_epoch - self.bjdrefi
            self.period = period
            self.transit_duration = trdur
            self.baseline_duration = bldur
            self.lc = lc = KeplerLC(time, flux, zeros(time.size), zero_epoch, period, trdur, bldur)
            times, fluxes = lc.time_per_transit, lc.normalized_flux_per_transit
            pbids = lc.nt * [0]
        else:
            times, fluxes = [time], [flux]
            pbids = [0]
            self.zero_epoch = None
            self.period = None
            self.transit_duration = None
            self.baseline_duration = None

        wnids = arange(len(times)) if separate_noise else None

        BaseLPF.__init__(self, name, ['TESS'], times=times, fluxes=fluxes, pbids=pbids,
                         nsamples=nsamples, exptimes=[0.00139], wnids=wnids)


    def plot_individual_transits(self, ncols: int = 2, figsize=(14, 8)):
        df = self.posterior_samples(derived_parameters=False)
        pvp = permutation(df.values)[:5000]
        pv = df.median()
        tmodels = self.flux_model(pvp)
        mm = percentile(tmodels, [50, 16, 84, 0.5, 99.5], 0)

        t0 = floor(self.times[0].min())
        nrows = int(ceil(self.nlc / ncols))
        fig, axs = subplots(nrows, ncols, figsize=figsize, sharey=True, constrained_layout=True)
        for i, sl in enumerate(self.lcslices):
            axs.flat[i].plot(self.times[i] - t0, self.fluxes[i], drawstyle='steps-mid', alpha=0.5)
            axs.flat[i].plot(self.times[i] - t0, mm[0][sl], 'k')
            axs.flat[i].fill_between(self.times[i] - t0, mm[1][sl], mm[2][sl], alpha=0.75, facecolor='orangered')
            setp(axs[:, 0], ylabel='Normalized flux')
            setp(axs[-1, :], xlabel=f'Time - {self.bjdrefi + t0:.0f} [days]')
        return fig

    def plot_folded_transit(self, method='de', figsize=(13, 6), ylim=(0.9975, 1.002), xlim=None, binwidth=8):
        if method == 'de':
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, 0.5)
        binwidth = binwidth / 24 / 60
        sids = argsort(phase)

        tm = self.transit_model(pv)
        bl = squeeze(self.baseline(pv))

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
        corner(df['tc pr rho b k'.split()],
               labels='Zero epoch, Period, Stellar density, impact parameter, radius ratio'.split(', '))