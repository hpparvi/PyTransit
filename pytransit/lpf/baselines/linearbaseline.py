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

from typing import Optional
from numba import prange, njit
from numpy import atleast_2d, zeros, dot, array, concatenate, squeeze, r_, inf, unique, full, ndarray, ones

from ...param import LParameter, NormalPrior as NP

@njit(parallel=True)
def linear_model(pvp, cids, cstart, ncov, cova):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = cids.size
    bl = zeros((npv, npt))
    for ipv in prange(npv):
        ii = 0
        for ipt in range(npt):
            ilc = cids[ipt]  # Current local light curve index
            cst = cstart[ilc]  # Start of the coefficients in the parameter vector
            nc = ncov[ilc]  # Number of covariates for the current light curve
            bl[ipv, ipt] = pvp[ipv, cst] + dot(pvp[ipv, cst + 1:cst + nc], cova[ii:ii + nc - 1])
            ii += nc
    return bl


class LinearModelBaseline:
    def __init__(self, lpf, name: str = 'lm', lcids=None):
        self.name = name
        self.lpf = lpf

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising LinearModelBaseline.')

        self.init_data(lcids)
        self.init_parameters()

    def init_data(self, lcids = None):
        self.lcids = lcids if lcids is not None else unique(self.lpf.lcids)
        self.nlc = self.lcids.size
        self.ncov = array(
            [self.lpf.covariates[lcid].shape[1] for lcid in self.lcids])  # Number of covariates per light curve
        self.cova = concatenate([self.lpf.covariates[lcid].ravel() for lcid in self.lcids])  # Flattened covariate vector
        self.cids = concatenate([full((self.lpf.lcids == lcid).sum(), i) for i, lcid in enumerate(self.lcids)])
        self._bl_coeff_start = r_[[0], self.ncov + 1].cumsum()  # Parameter vector start index for the coefficients

        self.mask = zeros(self.lpf.lcids.size, bool)
        for lcid in self.lcids:
            self.mask[self.lpf.lcslices[lcid]] = 1

        if hasattr(self.lpf, 'ins'):
            self.ins = [self.lpf.ins[lcid] for lcid in self.lcids]
        else:
            self.ins = self.nlc * ['']

        if hasattr(self.lpf, 'piis'):
            self.piis = [self.lpf.piis[lcid] for lcid in self.lcids]
        else:
            self.piis = zeros(self.nlc, int)

    def init_parameters(self):
        """Baseline parameter initialisation.
        """
        bls = []
        for i, tn in enumerate(range(self.nlc)):
            ins = self.ins[i] + '_'
            pii = self.piis[i]

            fstd = self.lpf.fluxes[i].std()
            bls.append(LParameter(f'{self.name}_i_{ins}{pii}', f'{self.name}_intercept_{ins}{pii}', '', NP(1.0, fstd), bounds=(-inf, inf)))
            for ipoly in range(1, self.ncov[i] + 1):
                bls.append(
                    LParameter(f'{self.name}_s_{ins}{pii}_{ipoly}', f'{self.name}_c_{ins}{pii}_{ipoly}', '', NP(0.0, fstd), bounds=(-inf, inf)))
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, bls)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{self.name}", self.pv_slice)
        setattr(self.lpf, f"_start_{self.name}", self.pv_start)

    def __call__(self, pvp, bl: Optional[ndarray] = None):
        pvp = atleast_2d(pvp)
        if bl is None:
            bl = ones((pvp.shape[0], self.lpf.timea.size))
        else:
            bl = atleast_2d(bl)
        bl[:, self.mask] += linear_model(pvp[:, self.pv_slice], self.cids, self._bl_coeff_start, self.ncov, self.cova) - 1.
        return squeeze(bl)
