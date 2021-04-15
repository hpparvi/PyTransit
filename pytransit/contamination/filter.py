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

from numpy import array, ones_like, zeros_like, diff, arange, linspace, ndarray, where, ones
from scipy.interpolate import interp1d


class Filter:
    def __init__(self, name: str):
        self.name: str = name
        self.bbox: ndarray = array([250, 1000], dtype='d')

    def __call__(self, wl):
        raise NotImplementedError

    def sample(self, n: Optional[int] = 100):
        raise NotImplementedError


class DeltaFilter(Filter):
    def __init__(self, name: str, wl: float):
        super().__init__(name)
        self.wl: float = wl
        self.bbox = array([wl-1e-5, wl+1e-5])

    def __call__(self, wl):
        return where(abs(wl - self.wl) < 1e-5, 1.0, 0.0)

    def sample(self, n: Optional[int] = 100):
        return array(self.wl), array(1.0)


class ClearFilter(Filter):
    """Constant unity transmission.
    """

    def __init__(self, name: str):
        raise NotImplementedError("CleanFilter has been removed, please use a wide BoxcarFilter instead.")


class BoxcarFilter(Filter):
    """Filter with a transmission of 1 inside the minimum and maximum wavelengths and 0 outside.
    """

    def __init__(self, name, wl_min, wl_max):
        """
        Parameters
        ----------
        :param name: passband name
        :param wl_min: minimum wavelength
        :param wl_max: maximum wavelength
        """
        super().__init__(name)
        self.bbox = array([wl_min, wl_max], dtype='d')

    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl >= self.bbox[0]) & (wl <= self.bbox[1])] = 1.
        return w

    def sample(self, n: Optional[int] = 100):
        return linspace(*self.bbox, num=n), ones(n)


class TabulatedFilter(Filter):
    """Interpolated tabulated filter.
    """

    def __init__(self, name, wl, tm):
        """
        Parameters
        ----------

        name  : string      passband name
        wl    : array_like  a list of wavelengths
        tm    : array_like  a list of transmission values
        """
        super().__init__(name)
        self.wl = array(wl)
        self.tm = array(tm)
        self.bbox = array([self.wl.min(), self.wl.max()])
        assert self.wl.size == self.tm.size, "The wavelength and transmission arrays must be of same size"
        assert all(diff(self.wl) > 0.), "Wavelength array must be monotonously increasing"
        assert all((self.tm >= 0.0) & (self.tm <= 1.0)), "Transmission must always be between 0.0 and 1.0"
        self._ip = interp1d(self.wl, self.tm, kind='cubic')

    def __call__(self, wl):
        return self._ip(wl)

    def sample(self, n: Optional[int] = 100):
        return self.wl, self.tm


sdss_g = BoxcarFilter("g'", 400, 550) #: SDSS G filter
sdss_r = BoxcarFilter("r'", 570, 690)
sdss_i = BoxcarFilter("i'", 710, 790)
sdss_z = BoxcarFilter("z'", 810, 900)
kepler = TabulatedFilter('kepler',
                         arange(350, 960, 25),
                         array([0.000, 0.001, 0.000, 0.056, 0.465, 0.536, 0.624, 0.663,
                                0.681, 0.715, 0.713, 0.696, 0.670, 0.649, 0.616, 0.574,
                                0.541, 0.490, 0.468, 0.400, 0.332, 0.279, 0.020, 0.000,
                                0.000]))

__all__ = 'Filter TabulatedFilter BoxcarFilter sdss_g sdss_r sdss_i sdss_z kepler'.split()