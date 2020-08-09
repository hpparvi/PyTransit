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

from typing import Tuple
from numpy import linspace, trapz, sqrt, ndarray, zeros, pi


class LDModel:
    def __init__(self, niz: int = 200):
        self._int_z = linspace(0, 1, niz)
        self._int_mu = sqrt(1 - self._int_z ** 2)

    def __call__(self, mu: ndarray, x: ndarray) -> Tuple[ndarray, ndarray]:
        return self._evaluate(mu, x), self._integrate(x)

    def _evaluate(self, mu: ndarray, x:ndarray) -> ndarray:
        raise NotImplementedError

    def _integrate(self, x: ndarray) -> ndarray:
        npv = x.shape[0]
        npb = x.shape[1]
        ldi = zeros((npv, npb))
        for ipv in range(npv):
            for ipb in range(npb):
                ldi[ipv,ipb] = 2. * pi * trapz(self._int_z * self(self._int_mu, x), self._int_z)
        return ldi