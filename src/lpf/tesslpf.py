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
from numba import njit, prange
from numpy import atleast_2d, zeros, log, concatenate, pi, transpose, sum, squeeze, ceil, arange, digitize, full, nan, \
    sqrt, percentile, isfinite, floor, argsort
from numpy.polynomial.legendre import legvander
from numpy.random import uniform, permutation
from matplotlib.pyplot import subplots
from matplotlib.pyplot import setp

from .lpf import BaseLPF
from ..param.parameter import LParameter, UniformPrior as UP, NormalPrior as NP
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
    def __init__(self, name: str, dfile: Path, zero_epoch: float, period: float, nsamples: int = 5,
                 trdur: float = 0.125, bldur: float = 0.3, nlegendre: int = 2):

        tb = Table.read(dfile)
        self.bjdrefi = tb.meta['BJDREFI']
        self.zero_epoch = zero_epoch - self.bjdrefi

        df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
        self.lc = lc = KeplerLC(df.TIME.values, df.SAP_FLUX.values, zeros(df.shape[0]),
                                self.zero_epoch, period, trdur, bldur)

        self.nlegendre = nlegendre
        super().__init__(name, ['TESS'],
                         times=lc.time_per_transit, fluxes=lc.normalized_flux_per_transit,
                         pbids=lc.nt * [0], nsamples=nsamples, exptimes=[0.00139])

        self.mtimes = [t - t.mean() for t in self.times]
        self.windows = window = concatenate(self.mtimes).ptp()
        self.mtimes = [t / window for t in self.mtimes]
        self.legs = [legvander(t, self.nlegendre) for t in self.mtimes]
        self.ofluxa = self.ofluxa.astype('d')

    def _init_p_baseline(self):
        """Baseline parameter initialisation.
        """
        bls = []
        for i in range(self.nlc):
            bls.append(LParameter(f'bli_{i}', f'bl_intercept_{i}', '', NP(1.0, 0.01), bounds=(0.95, 1.05)))
            for ipoly in range(1, self.nlegendre + 1):
                bls.append(
                    LParameter(f'bls_{i}_{ipoly}', f'bl_c_{i}_{ipoly}', '', NP(0.0, 0.001), bounds=(-0.1, 0.1)))
        self.ps.add_lightcurve_block('baseline', self.nlegendre + 1, self.nlc, bls)
        self._sl_bl = self.ps.blocks[-1].slice
        self._start_bl = self.ps.blocks[-1].start

    def baseline(self, pvp):
        """Multiplicative baseline"""
        pvp = atleast_2d(pvp)
        fbl = zeros((pvp.shape[0], self.timea.size))
        bl = pvp[:, self._sl_bl]
        for itr, sl in enumerate(self.lcslices):
            fbl[:, sl] = bl[:, itr * (self.nlegendre + 1):(itr + 1) * (self.nlegendre + 1)] @ self.legs[itr].T
        return fbl

    def flux_model(self, pvp):
        return squeeze(self.transit_model(pvp) * self.baseline(pvp))

    def plot_individual_transits(self, ncols: int = 2, figsize=(14, 8)):
        df = self.posterior_samples(include_ldc=True)
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

        bp, bfo, beo = downsample_time(phase, self.ofluxa / bl, binwidth)

        fig, ax = subplots(figsize=figsize)
        ax.plot(phase - 0.5 * p, self.ofluxa / bl, '.', alpha=0.15)
        ax.errorbar(bp - 0.5 * p, bfo, beo, fmt='ko')
        ax.plot(phase[sids] - 0.5 * p, tm[sids], 'k')
        xlim = xlim if xlim is not None else 1.01 * (bp[isfinite(bp)][[0, -1]] - 0.5 * p)
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - Tc [d]', ylabel='Normalised flux')
        fig.tight_layout()

    def plot_basic_posteriors(self):
        df = self.posterior_samples()
        df['k'] = sqrt(df.k2)
        df.drop('k2', axis=1, inplace=True)
        corner(df['tc pr rho b k'.split()],
               labels='Zero epoch, Period, Stellar density, impact parameter, radius ratio'.split(', '))