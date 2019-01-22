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

import numpy as np
from numpy import atleast_2d, array, ones
from .supersampler import SuperSampler
from .orbits import Orbit
from .limb_darkening import UniformLD, LinearLD, QuadraticLD, TriangularQLD

def _extract_time_transit(time, k, tc, p, a, i, e, w, td_factor=1.1):
    td = of.duration_eccentric_w(p, k, a, i, e, w, 1)
    folded_time = (time - tc + 0.5*p) % p - 0.5*p
    mask = np.abs(folded_time) < td_factor*0.5*td
    return time[mask], mask

def _extract_time_eclipse(time, k, tc, p, a, i, e, w, td_factor=1.1):
    td  = of.duration_eccentric_w(p, k, a, i, e, w, 1)
    tc += of.eclipse_shift_ex(p, i, e, w)
    folded_time = (time - tc + 0.5*p) % p - 0.5*p
    mask = np.abs(folded_time) < td_factor*0.5*td
    return time[mask], mask

class TransitModel(object):
    """Exoplanet transit light curve model 
    """
    native_ld_par = None 
    
    def __init__(self, nldc=2, nthr=1, interpolate=False, supersampling=1, exptime=0.020433598, eclipse=False,
                 orbit=None, LDPar=QuadraticLD, optimize=False):
        assert isinstance(nldc, int)
        assert isinstance(nthr, int)
        assert isinstance(supersampling, int)
        assert supersampling > 0
        assert exptime > 0.

        self.nldc = int(nldc)
        self.nthr = int(nthr)
        self.time = None
        self.eclipse = bool(eclipse)
        self.optimize = optimize

        # Initialize the supersampler
        # ---------------------------
        self.sampler = SuperSampler(supersampling, exptime, nthr=self.nthr)

        # Initialize the orbit
        # --------------------
        if not orbit:
            self.orbit = Orbit(nthr=self.nthr)
        else:
            if isinstance(orbit, Orbit):
                self.orbit = orbit
            elif isinstance(orbit, str):
                self.orbit = Orbit(method=orbit, nthr=self.nthr)
            else:
                raise NotImplementedError

        # Initialize the limb darkening mapping
        # -------------------------------------
        self.ldmap = LDPar()
        
    def __call__(self, z, k, u, c=0., update=True):
        raise NotImplementedError


    def evaluate(self, t, k, u, t0, p, a, i, e=0., w=0., c=0.):
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

        u = atleast_2d(u)
        npb = u.shape[0]

        if self.optimize:
            if not self.eclipse:
                tt, tm = _extract_time_transit(t, k, t0, p, a, i, e, w, 1.1)
            else:
                tt, tm = _extract_time_eclipse(t, k, t0, p, a, i, e, w, 1.1)
        else:
            tt, tm = t, np.ones_like(t, dtype=np.bool)

        # Check if we have multiple radius ratio (k) values, approximate the k with their
        # mean if yes, and calculate the area ratio factors.
        #
        if isinstance(k, np.ndarray):
            _k = k.mean()
            kf = (k/_k)**2
        else:
            _k = k
            kf = 1.

        z = self._calculate_z(tt, t0, p, a, i, e, w)
        supersampled_flux = self.__call__(z, k, u=u, c=c)

        averaged_flux = ones((t.size, npb))
        averaged_flux[tm,:] = self.sampler.average(supersampled_flux)

        return kf*(averaged_flux-1.)+1.


    def _calculate_z(self, t, t0, p, a, i, e=0, w=0):
        z = self.orbit.projected_distance(self.sampler.sample(t), t0, p, a, i, e, w)
        return z if not self.eclipse else -z
