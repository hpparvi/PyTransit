#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2021  Hannu Parviainen
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
from typing import Union, List, Tuple, Optional, Iterable

import arviz as az
import xarray as xa
import numpy as np

from matplotlib.pyplot import subplots, setp
from numba import njit
from numpy import inf, squeeze, atleast_2d, sqrt, zeros_like, \
    ones, arange, argsort, arctan2, ndarray, median, percentile, ceil
from numpy.random import permutation

from .tesslpf import downsample_time
from .. import BaseLPF, TransitModel
from ..orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from ..param import UniformPrior as UP, NormalPrior as NP, GParameter

np.seterr(invalid='ignore')


@njit(fastmath=True, cache=False)
def map_ldc(ldc):
    ldc = atleast_2d(ldc)
    uv = zeros_like(ldc)
    a, b = sqrt(ldc[:, 0::2]), 2. * ldc[:, 1::2]
    uv[:, 0::2] = a * b
    uv[:, 1::2] = a * (1. - b)
    return uv


class TransitAnalysis(BaseLPF):
    """Class for exoplanet transit modelling.
    """
    def __init__(self, name: str,
                 passbands: Iterable[str],
                 times: Union[Iterable[ndarray], ndarray],
                 fluxes: Union[Iterable[ndarray], ndarray],
                 errors: Optional[Union[Iterable[ndarray], ndarray]] = None,
                 pbids: Optional[Union[Iterable[ndarray], ndarray]] = None,
                 covariates: Optional[Union[Iterable[ndarray], ndarray]] = None,
                 wnids: Optional[Union[Iterable[ndarray], ndarray]] = None,
                 nplanets: int = 1,
                 tm: Optional[TransitModel] = None,
                 nsamples: Union[Iterable[int], int] = 1,
                 exptimes: Union[Iterable[float], float] = 0.,
                 init_data: bool = True,
                 result_dir: Optional[Path] = None,
                 tref: float = 0.0,
                 lnlikelihood: str = 'wn'):

        self.nplanets = nplanets
        super().__init__(name, passbands, times, fluxes, errors, pbids, covariates, wnids, tm,
                         nsamples, exptimes, init_data, result_dir, tref, lnlikelihood)

    def _init_p_orbit(self):
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        ps = self.ps
        pp = []
        for i in range(1, self.nplanets + 1):
            pp.extend([
                GParameter(f'tc_{i}', f'zero_epoch_{i}', '', NP(0.0, 0.1), (-inf, inf)),
                GParameter(f'p_{i}', f'period_{i}', 'd', NP(1.0, 1e-5), (0, inf)),
                GParameter(f'b_{i}', f'impact_parameter_{i}', 'R_s', UP(0.0, 1.0), (0, inf)),
                GParameter(f'k2_{i}', f'area_ratio_{i}', 'A_s', UP(0.02**2, 0.2**2), (0, inf)),
                GParameter(f'secw_{i}', f'sqrt(e) cos(w) {i}', '', NP(0.0, 1e-5), (-1, 1)),
                GParameter(f'sesw_{i}', f'sqrt(e) sin(w) {i}', '', NP(0.0, 1e-5), (-1, 1)),
            ])
        ps.add_global_block('planets', pp)
        self._start_pl = ps.blocks[-1].start
        self._sl_pl = ps.blocks[-1].slice

    def transit_model(self, pv, copy=True, planets=None):
        pv = atleast_2d(pv)
        flux = ones([pv.shape[0], self.timea.size])
        ldc = map_ldc(pv[:, self._sl_ld])
        planets = planets if planets is not None else arange(self.nplanets)
        for ipl in planets:
            ist = 6*ipl
            t0 = pv[:, 1 + ist]
            p  = pv[:, 2 + ist]
            k = sqrt(pv[:, 4 + ist : 5 + ist])
            aor = as_from_rhop(pv[:, 0], p)
            inc = i_from_ba(pv[:, 3 + ist], aor)
            ecc = pv[:, 5 + ist]**2 + pv[:, 6 + ist]**2
            w = arctan2(pv[:, 6 + ist], pv[:, 5 + ist])
            flux += self.tm.evaluate(k, ldc, t0, p, aor, inc, ecc, w, copy) - 1.
        return squeeze(flux)

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def posterior_samples(self):
        dd = az.from_emcee(self.sampler, var_names=self.ps.names)
        ds = xa.Dataset()
        pst = dd.posterior
        c = pst.rho.coords
        DA = xa.DataArray
        for i in range(1, self.nplanets + 1):
            p = pst[f'p_{i}'].values
            ds[f'k_{i}'] = k = DA(sqrt(pst[f'k2_{i}']), coords=c)
            ds[f'a_{i}'] = a = DA(as_from_rhop(pst.rho.values, p), coords=c)
            ds[f'e_{i}'] = e = DA(pst[f'secw_{i}']**2 + pst[f'sesw_{i}']**2, coords=c)
            ds[f'w_{i}'] = w = DA(arctan2(pst[f'sesw_{i}'], pst[f'secw_{i}']), coords=c)
            ds[f'i_{i}'] = inc = DA(i_from_baew(pst[f'b_{i}'].values, a.values, e.values, w.values), coords=c)
            ds[f't14_{i}']   = DA(d_from_pkaiews(p, k.values, a.values, inc.values, 0., 0., 1, kind=14), coords=c)
            ds[f't23_{i}']   = DA(d_from_pkaiews(p, k.values, a.values, inc.values, 0., 0., 1, kind=23), coords=c)
        dd.add_groups({'derived_parameters': ds})
        return dd

    def plot_light_curves(self, method='de', ncol: int = 3, width: Optional[float] = None, planet: int = 1,
                          max_samples: int = 1000, figsize=None, data_alpha=0.5, ylim=None):

        solutions = 'best fit de posterior mc mcmc'.split()
        if method not in solutions:
            raise ValueError(f'The "method" needs to be one of {solutions}')

        if width is None:
            if self.nlc == 1:
                width = 24 * self.timea.ptp()
            else:
                width = 2.0

        ncol = min(ncol, self.nlc)
        nrow = int(ceil(self.nlc / ncol))
        tid, pid = self.ps.find_pid(f"tc_{planet}"), self.ps.find_pid(f"p_{planet}")
        if method in ('mcmc', 'mc', 'posterior'):
            pvp = self.posterior_samples().posterior.to_array().values.copy().T.reshape([-1, len(self.ps)])
            t0, p = median(pvp[:, tid]), median(pvp[:, pid])
            fmodel = self.flux_model(permutation(pvp)[:max_samples])
            fmperc = percentile(fmodel, [50, 16, 84, 2.5, 97.5], 0)
        elif method in ('de', 'fit', 'best'):
            pv = self.de.minimum_location
            fmodel = squeeze(self.flux_model(pv))
            t0, p = pv[tid], pv[pid]
            fmperc = None
        else:
            raise ValueError

        fig, axs = subplots(nrow, ncol, figsize=figsize, constrained_layout=True, sharey='all', sharex='all',
                            squeeze=False)
        for i in range(self.nlc):
            ax = axs.flat[i]
            e = epoch(self.times[i].mean(), t0, p)
            tc = t0 + e * p
            time = self.times[i] - tc

            ax.plot(time, self.fluxes[i], '.', alpha=data_alpha)

            if method in ('de', 'fit', 'best'):
                ax.plot(time, fmodel[self.lcslices[i]], 'w', lw=4)
                ax.plot(time, fmodel[self.lcslices[i]], 'k', lw=1)
            else:
                ax.fill_between(time, *fmperc[3:5, self.lcslices[i]], alpha=0.15)
                ax.fill_between(time, *fmperc[1:3, self.lcslices[i]], alpha=0.25)
                ax.plot(time, fmperc[0, self.lcslices[i]])

            setp(ax, xlabel=f'Time - T$_c$ [d]', xlim=(-width / 2 / 24, width / 2 / 24))
        setp(axs[:, 0], ylabel='Normalised flux')

        if ylim is not None:
            setp(axs, ylim=ylim)

        for ax in axs.flat[self.nlc:]:
            ax.remove()
        return fig

    def plot_folded_planets(self, bin_width: float = 10.0, window_width: float = 3.0, figsize: Optional[Tuple] = None):
        fig, axs = subplots(1, self.nplanets, figsize=figsize, sharey='all')
        pv = self.de.minimum_location.copy()
        for ipl in range(self.nplanets):
            planets = set(arange(self.nplanets))
            planets.remove(ipl)
            t0 = pv[1 + 6 * ipl]
            p = pv[2 + 6 * ipl]
            mflux = self.transit_model(pv, planets=[ipl])
            rflux = self.transit_model(pv, planets=planets)
            oflux = self.ofluxa / rflux
            phase = fold(self.timea, p, t0)
            m = abs(phase) < 0.5 * window_width / 24
            sids = argsort(phase[m])
            phase, mflux, oflux = phase[m][sids], mflux[m][sids], oflux[m][sids]
            bp, bf, be = downsample_time(phase, oflux, bin_width / 24 / 60)
            axs[ipl].errorbar(bp, bf, be, fmt='ok')
            axs[ipl].plot(phase, oflux, '.', alpha=0.25)
            axs[ipl].plot(phase, mflux, 'k')
        setp(axs, xlim=(-0.5 * window_width / 24, 0.5 * window_width / 24), ylim=(0.996, 1.004), xlabel='Time - T$_0$ [d]')
        setp(axs[0], ylabel='Normalised flux')
        fig.tight_layout()
        return fig
