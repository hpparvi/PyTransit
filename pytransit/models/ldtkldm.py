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

from typing import Tuple, Optional, Union
from pathlib import Path

from numba import njit
from numpy import zeros, interp, pi, ndarray
from numpy.random import randint

from .ldmodel import LDModel

try:
    from ldtk import LDPSetCreator
except ImportError:
    class LDPSetCreator:
        def __init__(self, *nargs, **kwargs):
            raise ModuleNotFoundError('LDTkLDModel requires LDTk.')

@njit
def ntrapz(x, y):
    npt = x.size
    ii = 0.0
    for i in range(1, npt):
        ii += (x[i] - x[i - 1]) * 0.5 * (y[i] + y[i - 1])
    return ii


@njit
def eval_ldm(emu, mu, z, ldps, npv, nsamples):
    npb = ldps.shape[0]
    ldp = zeros((npv, npb, emu.size))
    ldi = zeros((npv, npb))
    iis = randint(0, nsamples, size=npv)

    for ipv in range(npv):
        for ipb in range(npb):
            ldp[ipv, ipb] = interp(emu, mu, ldps[ipb, iis[ipv]])
            ldi[ipv, ipb] = -2 * pi * ntrapz(z, z * ldps[ipb, iis[ipv]])
    return ldp, ldi


@njit
def eval_ldm_frozen(emu, mu, z, mldps, npv):
    npb = mldps.shape[0]
    ldp = zeros((npv, npb, emu.size))
    ldi = zeros((npv, npb))

    for ipb in range(npb):
        ldp[0, ipb] = interp(emu, mu, mldps[ipb])
        ldi[0, ipb] = -2 * pi * ntrapz(z, z * mldps[ipb])

    for i in range(1, npv):
        ldp[i] = ldp[0]
        ldi[i] = ldi[0]

    return ldp, ldi


class LDTkLDModel(LDModel):
    def __init__(self, pbs: Tuple, teff: Tuple[float, float], logg: Tuple[float, float], z: Tuple[float, float],
                 samples: int = 500, frozen: bool = False, cache: Optional[Union[str, Path]] = None, lowres: bool = True):
        super().__init__()
        self._sc = LDPSetCreator(teff, logg, z, pbs, cache=cache, dataset='vis_lowres')
        self._ps = self._sc.create_profiles(samples)
        self._i = 0

        self.npb = len(pbs)
        self.nsamples = samples
        self.frozen = frozen
        self.z = self._ps._z
        self.mu = self._ps._mu
        self.profiles = self._ps._ldps
        self.mean_profiles = self._ps.profile_averages

    def thaw(self):
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def __call__(self, mu: ndarray, x: Optional[ndarray] = None) -> Tuple[ndarray, ndarray]:
        npv = 1 if x is None else x.shape[0]
        self._i = i = randint(0, self.nsamples)
        if self.frozen:
            return eval_ldm_frozen(mu, self.mu, self.z, self.mean_profiles, npv)
        else:
            return eval_ldm(mu, self.mu, self.z, self.profiles, npv, self.nsamples)

    def _evaluate(self, mu: ndarray, x:ndarray) -> ndarray:
        raise NotImplementedError

    def _integrate(self, x: ndarray) -> float:
        raise NotImplementedError

class LDTkLD(LDTkLDModel):
    ...