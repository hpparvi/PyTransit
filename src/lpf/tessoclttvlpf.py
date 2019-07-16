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
from numba import njit, prange
from numpy import atleast_2d, zeros, log, concatenate, pi, transpose, sum, compress, ones
from uncertainties import UFloat, ufloat

from .oclttvlpf import OCLTTVLPF
from ..utils.keplerlc import KeplerLC
from .baselines.legendrebaseline import  LegendreBaseline

@njit(parallel=True, cache=False, fastmath=True)
def lnlike_normal_v(o, m, e):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        lnl[i] = -npt*log(e[i, 0]) - 0.5*log(2*pi) - 0.5*sum(((o-m[i, :])/e[i ,0])**2)
    return lnl


class TESSCLTTVLPF(LegendreBaseline, OCLTTVLPF):
    def __init__(self, name: str, dfile: Path, zero_epoch: float, period: float, nsamples: int = 10,
                 trdur: float = 0.125, bldur: float = 0.3, nlegendre: int = 2, ctx = None, queue = None):

        zero_epoch = zero_epoch if isinstance(zero_epoch, UFloat) else ufloat(zero_epoch, 1e-5)
        period = period if isinstance(period, UFloat) else ufloat(period, 1e-7)

        tb = Table.read(dfile)
        self.bjdrefi = tb.meta['BJDREFI']
        zero_epoch = zero_epoch - self.bjdrefi

        df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
        self.lc = lc = KeplerLC(df.TIME.values, df.SAP_FLUX.values, zeros(df.shape[0]),
                                zero_epoch.n, period.n, trdur, bldur)

        LegendreBaseline.__init__(self, nlegendre)
        OCLTTVLPF.__init__(self, name, zero_epoch, period, ['TESS'],
                         times=lc.time_per_transit, fluxes=lc.normalized_flux_per_transit,
                         pbids=zeros(lc.nt, 'int'), nsamples=nsamples, exptimes=[0.00139],
                         cl_ctx=ctx, cl_queue=queue)

        self.lnlikelihood = self.lnlikelihood_nb

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        return pvp

    def flux_model(self, pvp):
        tmodel = transpose(self.transit_model(pvp, copy=True)).copy()
        return tmodel * self.baseline(pvp)

    def lnlikelihood_nb(self, pvp):
        fmodel = self.flux_model(pvp).astype('d')
        err = 10**atleast_2d(pvp)[:, self._sl_err]
        return lnlike_normal_v(self.ofluxa, fmodel, err)
