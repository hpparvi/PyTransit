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

"""Contaminated TESS LPF

This module contains the log posterior function (LPF) for a TESS light curve with possible third-light
contamination from an unresolved source inside the photometry aperture.
"""
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, List

from astropy.table import Table
from matplotlib.pyplot import subplots, setp
from numba import njit
from astropy.io.fits import getval
from numpy import repeat, inf, where, newaxis, squeeze, atleast_2d, isfinite, concatenate, zeros, sqrt, zeros_like, \
    ones, median, any, arange, argsort
from numpy.random.mtrand import uniform
from pytransit import BaseLPF
from pytransit.lpf.tesslpf import TESSLPF, downsample_time
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.param import UniformPrior as UP, NormalPrior as NP, PParameter, GParameter
from ldtk import tess
from pytransit.utils.misc import fold
from scipy.ndimage import label
from uncertainties import ufloat, UFloat


@njit(fastmath=True)
def map_pv(pv, i):
    pv = atleast_2d(pv)
    pp = pv[:,2+4*i:6+4*i]
    pvt = zeros((pv.shape[0], 7))
    pvt[:,0]   = sqrt(pp[:,3])
    pvt[:,1:3] = pp[:,0:2]
    pvt[:,  3] = as_from_rhop(pv[:,0], pp[:,1])
    pvt[:,  4] = i_from_ba(pp[:,2], pvt[:,3])
    return pvt


@njit(fastmath=True, cache=False)
def map_ldc(ldc):
    ldc = atleast_2d(ldc)
    uv = zeros_like(ldc)
    a, b = sqrt(ldc[:,0::2]), 2.*ldc[:,1::2]
    uv[:,0::2] = a * b
    uv[:,1::2] = a * (1. - b)
    return uv

class CMPTESSLPF(BaseLPF):
    """Contaminated TESS LPF for multiplanet systems

    This class implements a log posterior function for a TESS light curve that allows for unknown flux contamination.
    The amount of flux contamination is not constrained.
    """
    def __init__(self, name: str, fname: Union[Path, str], nplanets: int, zero_epochs: List, periods: List,
                 nsamples: int = 1, bldur: float = 0.1, trdur: float = 0.04, use_pdc: bool = False):

        self._fname = fname
        self.nplanets = nplanets

        for i,t0 in enumerate(zero_epochs):
            if isinstance(t0, float):
                zero_epochs[i] = ufloat(t0, 1e-3)

        for i, pr in enumerate(periods):
            if isinstance(pr, float):
                periods[i] = ufloat(pr, 1e-5)

        self.zero_epochs = zero_epochs
        self.periods = periods
        self.bjdrefi = getval(fname, 'bjdrefi', 1)
        self.bldur = bldur
        self.trdur = trdur

        times, fluxes, pbnames, pbs, wns, covs = self.read_data()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = arange(len(times))
        self.wns = wns

        BaseLPF.__init__(self, name, passbands=pbnames, times=times, fluxes=fluxes, pbids=pbids, wnids=wnids,
                         covariates=covs, nsamples=nsamples, exptimes=[0.00139])

    def read_data(self):
        raise NotImplementedError

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [GParameter('rho', 'stellar_density',  'g/cm^3', UP(0.1, 25.0), (0,    inf))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        ps = self.ps
        pp = [GParameter('ctess', 'tess_contamination', '', UP(0.0, 0.99), (0, 1))]
        for i in range(1,self.nplanets+1):
            pp.extend([
                GParameter(f'tc_{i}', f'zero_epoch_{i}',       'd', NP(0.0, 0.1), (-inf, inf)),
                GParameter(f'p_{i}',  f'period_{i}',           'd', NP(1.0, 1e-5), (0, inf)),
                GParameter(f'b_{i}',  f'impact_parameter_{i}', 'R_s', UP(0.0, 1.0), (0, inf)),
                GParameter(f'k2_true_{i}', f'true_area_ratio_{i}', 'A_s', UP(0.02**2, 0.1**2), (0.02**2, 0.1**2))
            ])
        ps.add_global_block('planets', pp)
        self._start_pl = ps.blocks[-1].start
        self._sl_pl = ps.blocks[-1].slice

    def transit_model(self, pv, copy=True, planets=None):
        pv = atleast_2d(pv)
        flux = ones([pv.shape[0],self.timea.size])
        ldc = map_ldc(pv[:,self._sl_ld])
        planets = planets if planets is not None else arange(self.nplanets)
        for i in planets:
            pvp = map_pv(pv, i)
            flux += self.tm.evaluate_pv(pvp, ldc, copy) - 1.
        cnt = pv[:,1]
        return squeeze(cnt[:, newaxis] + (1. - cnt[:, newaxis]) * flux)

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def plot_folded_planets(self, bwidth: float = 10):
        fig, axs = subplots(1, self.nplanets, figsize=(13, 6), sharey=True)
        pv = self.de.minimum_location.copy()

        for ipl in range(self.nplanets):
            planets = set(arange(self.nplanets))
            planets.remove(ipl)
            mflux = self.transit_model(pv, planets=[ipl])
            rflux = self.transit_model(pv, planets=planets)
            oflux = self.ofluxa / rflux
            phase = (fold(self.timea, pv[3 + 4 * ipl], pv[2 + 4 * ipl], 0.5) - 0.5) * pv[3 + 4 * ipl]
            m = abs(phase) < 0.5 * self.bldur
            sids = argsort(phase[m])
            phase, mflux, oflux = phase[m][sids], mflux[m][sids], oflux[m][sids]
            bp, bf, be = downsample_time(phase, oflux, bwidth / 24 / 60)
            axs[ipl].errorbar(bp, bf, be, fmt='ok')
            axs[ipl].plot(phase, oflux, '.', alpha=0.25)
            axs[ipl].plot(phase, mflux, 'k')

        setp(axs, xlim=(-0.5 * self.bldur, 0.5 * self.bldur), ylim=(0.996, 1.004))
        fig.tight_layout()
        return fig