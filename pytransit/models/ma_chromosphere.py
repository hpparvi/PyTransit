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


from numpy import ndarray, array, squeeze

from .numba.ma_chromosphere_nb import chromosphere_model
from .transitmodel import TransitModel

__all__ = ['ChromosphereModel']

class ChromosphereModel(TransitModel):
    def __init__(self, method: str = 'pars'):
        super().__init__(method, False)

    def evaluate_ps(self, k: float, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
        pvp = array([[k, t0, p, a, i, e, w]])
        flux = chromosphere_model(self.time, pvp, self.lcids, self.nsamples, self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray) -> ndarray:
        flux = chromosphere_model(self.time, pvp, self.lcids, self.nsamples, self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)
