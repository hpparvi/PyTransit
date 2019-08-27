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

from numba import prange, njit
from numpy import atleast_2d, zeros, dot, array, concatenate, squeeze, r_, inf

from ...param import LParameter, NormalPrior as NP

@njit(parallel=True)
def linear_model(pvp, timea, lcids, cstart, ncov, cova):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = timea.size
    bl = zeros((npv, npt))
    for ipv in prange(npv):
        ii = 0
        for ipt in range(npt):
            ilc = lcids[ipt]  # Current light curve index
            cst = cstart[ilc]  # Start of the coefficients in the parameter vector
            nc = ncov[ilc]  # Number of covariates for the current light curve
            bl[ipv, ipt] = pvp[ipv, cst] + dot(pvp[ipv, cst + 1:cst + nc], cova[ii:ii + nc - 1])
            ii += nc
    return bl


class LinearModelBaseline:

    def _init_p_baseline(self):
        """Baseline parameter initialisation.
        """
        self.ncov = array([c.shape[1] for c in self.covariates])  # Number of covariates per light curve
        self.cova = concatenate([c.ravel() for c in self.covariates])  # Flattened covariate vector

        bls = []
        for i, tn in enumerate(range(self.nlc)):
            fstd = self.fluxes[i].std()
            bls.append(LParameter(f'bli_{tn}', f'bl_intercept_{tn}', '', NP(1.0, fstd), bounds=(-inf, inf)))
            for ipoly in range(1, self.ncov[i] + 1):
                bls.append(
                    LParameter(f'bls_{tn}_{ipoly}', f'bl_c_{tn}_{ipoly}', '', NP(0.0, fstd), bounds=(-inf, inf)))
        self.ps.add_global_block('baseline', bls)
        self._sl_bl = self.ps.blocks[-1].slice
        self._start_bl = self.ps.blocks[-1].start
        self._bl_coeff_start = r_[[0], self.ncov + 1].cumsum()  # Parameter vector start index for the coefficients

    def baseline(self, pvp):
        pvp = atleast_2d(pvp)
        return squeeze(
            linear_model(pvp[:, self._sl_bl], self.timea, self.lcids, self._bl_coeff_start, self.ncov, self.cova))
