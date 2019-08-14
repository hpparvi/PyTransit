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

from numpy import atleast_2d, zeros, concatenate, ones, inf
from numpy.polynomial.legendre import legvander
from numba import njit, prange

from ...param import LParameter, NormalPrior as NP

@njit(parallel=True, fastmath=True)
def lbaseline(ltimes, lcids, pv, deg):
    pv = atleast_2d(pv)
    npv = pv.shape[0]
    npt = ltimes.size
    npl = deg + 1
    bl = zeros((npv, npt))
    for ipt in prange(npt):
        leg = ones(npl)
        ilc = lcids[ipt]
        leg[1] = ltimes[ipt]
        for iln in range(npl):
            for ipv in range(npv):
                bl[ipv, ipt] += pv[ipv, npl*ilc + iln] * leg[iln]
            if iln > 0 and npl > 1:
                leg[iln+1] = ((2*iln + 1)*ltimes[ipt]*leg[iln] - iln*leg[iln-1] ) / (iln+1)
    return bl


class LegendreBaseline:
    def __init__(self, nlegendre):
        self.nlegendre = nlegendre
        self.times = []
        self.timea = None
        self.lcslices = None
        self.nlc = 0
        self.ps = None

    def _init_p_baseline(self):
        """Baseline parameter initialisation.
        """

        self._baseline_times = [(t - t.mean()) / t.ptp() for t in self.times]
        self._baseline_timea = concatenate(self._baseline_times)

        self.legendre_poly = [legvander(t, self.nlegendre) for t in self._baseline_times]
        self.ofluxa = self.ofluxa.astype('d')

        bls = []
        for i, tn in enumerate(range(self.nlc)):
            fstd = self.fluxes[i].std()
            bls.append(LParameter(f'bli_{tn}', f'bl_intercept_{tn}', '', NP(1.0, fstd), bounds=(-inf, inf)))
            for ipoly in range(1, self.nlegendre + 1):
                bls.append(
                    LParameter(f'bls_{tn}_{ipoly}', f'bl_c_{tn}_{ipoly}', '', NP(0.0, fstd), bounds=(-inf, inf)))
        self.ps.add_lightcurve_block('baseline', self.nlegendre + 1, self.nlc, bls)
        self._sl_bl = self.ps.blocks[-1].slice
        self._start_bl = self.ps.blocks[-1].start

    def baseline(self, pvp):
        """Multiplicative baseline"""
        pvp = atleast_2d(pvp)
        return lbaseline(self._baseline_timea, self.lcids, pvp[:,self._sl_bl], self.nlegendre)

