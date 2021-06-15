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
from typing import Union, List, Tuple, Optional

import numpy as np
from matplotlib.pyplot import subplots, setp
from numba import njit
from numpy import inf, squeeze, atleast_2d, sqrt, zeros_like, \
    ones, arange, argsort, arctan2

from .tesslpf import downsample_time
from .. import BaseLPF, TransitModel
from ..orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews
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


class MPTLPF(BaseLPF):
    """Transit LPF for multiplanet systems.
    """
    def __init__(self, name: str, nplanets: int, passbands: List[str],
                 times: List = None, fluxes: List = None, errors: List = None,
                 pbids: List = None, covariates: List = None, wnids: List = None,
                 tm: TransitModel = None,
                 nsamples: Union[List[int], int] = 1,
                 exptimes: Union[List[float], float] = 0.,
                 init_data=True, result_dir: Path = None,
                 tref: float = 0.0, lnlikelihood: str = 'wn'):

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
                GParameter(f'k2_{i}', f'area_ratio_{i}', 'A_s', UP(0.02 ** 2, 0.1 ** 2), (0, inf)),
                GParameter(f'secw_{i}', f'sqrt(e) cos(w) {i}', '', UP(-1.0, 1.0), (-1, 1)),
                GParameter(f'sesw_{i}', f'sqrt(e) sin(w) {i}', '', UP(-1.0, 1.0), (-1, 1)),
            ])
        ps.add_global_block('planets', pp)
        self._start_pl = ps.blocks[-1].start
        self._sl_pl = ps.blocks[-1].slice

    def transit_model(self, pv, copy=True, planets=None):
        pv = atleast_2d(pv)
        flux = ones([pv.shape[0], self.timea.size])
        ldc = map_ldc(pv[:, self._sl_ld])
        planets = planets if planets is not None else arange(self.nplanets)
        for i in planets:
            pp = pv[:, 1 + 6 * i: 7 + 6 * i]
            t0 = pp[:, 0]
            p = pp[:, 1]
            k = sqrt(pp[:, 3])
            aos = as_from_rhop(pv[:, 0], p)
            inc = i_from_ba(pp[:, 2], aos)
            ecc = pp[:, 4] ** 2 + pp[:, 5] ** 2
            w = arctan2(pp[:, 5], pp[:, 4])
            flux += self.tm.evaluate(k, ldc, t0, p, aos, inc, ecc, w, copy) - 1.
        return squeeze(flux)

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True):
        df = super(MPTLPF, self).posterior_samples(burn=burn, thin=thin, derived_parameters=False)
        if derived_parameters:
            for i in range(1, self.nplanets + 1):
                p = df[f'p_{i}'].values
                k = df[f'k_{i}'] = sqrt(df[f'k2_{i}'])
                e = df[f'e_{i}'] = df[f'secw_{i}'] ** 2 + df[f'sesw_{i}'] ** 2
                w = df[f'w_{i}'] = arctan2(df[f'sesw_{i}'], df[f'secw_{i}'])
                a = df[f'a_{i}'] = as_from_rhop(df.rho.values, p)
                inc = df[f'inc_{i}'] = i_from_baew(df[f'b_{i}'].values, a, e.values, w.values)
                df[f't14_{i}'] = d_from_pkaiews(p, k.values, a, inc, e.values, w.values, 1, kind=14)
                df[f't23_{i}'] = d_from_pkaiews(p, k.values, a, inc, e.values, w.values, 1, kind=23)
        return df

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
