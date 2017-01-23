## PyTransit
## Copyright (C) 2010--2015  Hannu Parviainen
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from math import fabs
import numpy as np
from .orbits_f import orbits as of
from .utils_f import utils as uf
from .supersampler import SuperSampler
from .orbits import Orbit

class TransitModel(object):
    """Exoplanet transit light curve model 
    """
    def __init__(self, nldc=2, nthr=0, interpolate=False, supersampling=1, exptime=0.020433598, eclipse=False):
        self.nldc = int(nldc)
        self.nthr = int(nthr)
        self.time = None
        self.eclipse = bool(eclipse)
        self.sampler = SuperSampler(supersampling, exptime, nthr=self.nthr)
        self.orbit = Orbit(nthr=self.nthr)
        
    def __call__(self, z, k, u, c=0., update=True):
        raise NotImplementedError


    def evaluate(self, t, k, u, t0, p, a, i, e=0., w=0., c=0., update=True, interpolate_z=False):
        """Evaluates the transit model given a time array and necessary parameters.

        Args:
            t: Array of time values [d].
            k: Planet to star radius ratio.
            u: Array of limb darkening coefficients.
            c: Contamination factor (fraction of third light), optional.
            t0: Zero epoch.
            p: Orbital period.
            a: Scaled semi-major axis
            i: Orbital inclination
            e: Eccentricity
            w: Argument of periastron.
            c: Contamination factor. 

        Returns:
            An array of model flux values for each time sample.
        """

        u   = np.asfortranarray(u)
        npb = 1 if u.ndim == 1 else u.shape[0]

        ## Check if we have multiple radius ratio (k) values, approximate the k with their
        ## mean if yes, and calculate the area ratio factors.
        ## 
        if isinstance(k, np.ndarray):
            _k = k.mean()
            kf = (k/_k)**2
        else:
            _k = k
            kf = 1.

        z = self._calculate_z(t, t0, p, a, i, e, w)
        supersampled_flux = self.__call__(z, k, u=u, c=c, update=update)
        averaged_flux = self.sampler.average(supersampled_flux)
 
        return kf*(averaged_flux-1.)+1.


    def _calculate_z(self, t, t0, p, a, i, e=0, w=0):
        z = self.orbit.projected_distance(self.sampler.sample(t), t0, p, a, i, e, w)
        return z if not self.eclipse else -z
