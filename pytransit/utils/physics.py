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
import warnings

from numpy import exp
from scipy.constants import c, k, h
from numba import jit

warnings.warn("the pytransits.utils.physics module is deprecated and will be removed in PyTransit 2.0",
               FutureWarning, stacklevel=2)

@jit
def planck(T, wl):
    """Radiance as a function or black-body temperature and wavelength.

    Parameters
    ----------

      T   : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      B   : Radiance
    """
    return 2*h*c**2 / wl**5 / (exp(h*c / (wl*k*T)) - 1.)

@jit
def planck_ratio(T1, T2, wl):
    """Ratio of the two black-body object radiances

    Parameters
    ----------

      T1  : Temperature  [K]
      T2  : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      rB   : Radiance ratio
    """
    return  (exp(h*c / (wl*k*T2)) - 1.) / (exp(h*c / (wl*k*T1)) - 1.)