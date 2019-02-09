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

from numpy import asarray, arange, newaxis, zeros
from numba import njit

@njit
def average_samples_1d(a, npt, nsamples):
    t = zeros(npt)
    for i in range(npt):
        t[i] = a[i*nsamples:(i+1)*nsamples].mean()
    return t

class SuperSampler(object):
    def __init__(self, nsamples, exptime):
        assert nsamples > 0
        assert exptime > 0.

        self.nsamples = nss = nsamples
        self.exptime  = etm = exptime
        self.sample_positions = self.create_sample_positions(nsamples, exptime)
        
        self._cache_id = None
        self._cache_hash = None
        self._time_original = None
        self._time_supersampled = None
        self._npt_original = None
        
        
    def create_sample_positions(self, nss, etm):
        return etm * ((arange(1, nss+1, dtype='d') - 0.5)/nss - 0.5)

    
    def sample(self, time):
        if (id(time) != self._cache_id) or (hash(time[0]) != self._cache_hash):
            self._cache_id = id(time)
            self._cache_hash = hash(time[0])
            self._time_original = time = asarray(time)
            self._time_supersampled = (time[:, newaxis] + self.sample_positions).ravel()
            self._npt_original = time.size
        return self._time_supersampled
    
    
    def average(self, values):
        values = asarray(values)
        if values.ndim == 1:
            return average_samples_1d(values, self._npt_original, self.nsamples)
        else:
            return values.reshape((self._npt_original, self.nsamples, -1)).mean(1)
