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
from numba import prange, njit
from numpy import atleast_2d, zeros, log, pi, asarray, unique, array, inf, arange

from ...param import  LParameter, NormalPrior as NP, UniformPrior as UP


@njit(parallel=True, cache=False, fastmath=True)
def lnlike_normal(o, m, e, slices, nids):
    m = atleast_2d(m)
    slices = atleast_2d(slices)

    npv = m.shape[0]
    nsl = slices.shape[0]
    lnl = zeros(npv)
    for i in prange(npv):
        for isl in range(nsl):
            _e = e[i, nids[isl]]
            for j in range(slices[isl, 0], slices[isl, 1]):
                lnl[i] += -log(_e) - 0.5 * log(2 * pi) - 0.5 * ((o[j] - m[i, j]) / _e) ** 2
    return lnl

class WNLogLikelihood:
    def __init__(self, lpf, name: str = 'wn', noise_ids=None, sigma=None):
        self.name = name
        self.lpf = lpf

        if sigma is None:
            self.free = True
        else:
            self.sigma = asarray(sigma)
            self.free = False

        if lpf.noise_ids is None:
            raise ValueError('The LPF data needs to be initialised before initialising WNLogLikelihood.')

        self.global_noise_ids = noise_ids if noise_ids is not None else unique(lpf.noise_ids)
        self.mapping = {g:l for g,l in zip(self.global_noise_ids, arange(self.global_noise_ids.size))}

        slices, lnids = [], []
        for nid, sl in zip(lpf.noise_ids, lpf.lcslices):
            if nid in self.global_noise_ids:
                slices.append([sl.start, sl.stop])
                lnids.append(self.mapping[nid])
        self.lcslices = array(slices)
        self.local_pv_noise_ids = array(lnids)

        self.times = lpf.timea
        self.fluxes = lpf.ofluxa

        if self.free:
            self.init_parameters()

    def init_parameters(self):
        name = self.name
        pgp = [LParameter(f'{name}_loge_{i}', f'{name} log10 sigma {i}', '', UP(-4, 0), bounds=(-inf, inf)) for i in self.global_noise_ids]
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, pgp)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{name}", self.pv_slice)
        setattr(self.lpf, f"_start_{name}", self.pv_start)

    def __call__(self, pvp, model):
        e = 10 ** atleast_2d(pvp)[:, self.pv_slice]
        return lnlike_normal(self.fluxes, model, e, self.lcslices, self.local_pv_noise_ids)