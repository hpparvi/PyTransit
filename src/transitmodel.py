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

import numpy as np
from numpy import atleast_2d, ones, ndarray
from .supersampler import SuperSampler
from .orbits import Orbit, extract_time_eclipse, extract_time_transit

class TransitModel(object):
    """Exoplanet transit light curve model 
    """

    methods = 'pars', 'pv1d', 'pv2d'

    def __init__(self, npb: int = 1, eccentric: bool = False, constant_k: bool = True,
                 contamination: bool = False, orbit: Orbit = None, optimize: bool = False,
                 eclipse: bool = False, method: str = 'pars') -> None:
        """

        Parameters
        ----------
        npb
        eccentric
        constant_k
        contamination
        orbit
        optimize
        eclipse
        method
        """
        assert method in self.methods
        assert npb >= 1
        self.npb = npb
        self.eccentric = eccentric
        self.constant_k = constant_k
        self.contamination = contamination
        self.optimize = optimize
        self.eclipse = eclipse
        self.time = None

        if method == 'pars':
            self.__call__ = self.evaluate_t
        elif method == 'pv1d':
            self.__call__ = self.evaluate_t_pv1d
        elif method == 'pv2d':
            self.__call__ = self.evaluate_t_pv2d

        # Initialize the orbit
        # --------------------
        if not orbit:
            self.orbit = Orbit(nthr=1)
        else:
            if isinstance(orbit, Orbit):
                self.orbit = orbit
            elif isinstance(orbit, str):
                self.orbit = Orbit(method=orbit, nthr=1)
            else:
                raise NotImplementedError

    def evaluate_t(self, t: ndarray, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0., c: float = 0.) -> ndarray:
        raise NotImplementedError

    def evaluate_t_pv1d(self, t: ndarray, pv: ndarray) -> ndarray:
        raise NotImplementedError

    def evaluate_t_pv2d(self, t: ndarray, pv: ndarray) -> ndarray:
        raise NotImplementedError

    def evaluate_z(self, z: ndarray, k: float, ldc: ndarray, c: float = 0.) -> ndarray:
        raise NotImplementedError

    def evaluate_z_pv1d(self, z: ndarray, pv: ndarray) -> ndarray:
        raise NotImplementedError

    def evaluate_z_pv2d(self, z: ndarray, pv: ndarray) -> ndarray:
        raise NotImplementedError


class SuperSampledTransitModel(TransitModel):
    def __init__(self, npb: int = 1, eccentric: bool = False, constant_k: bool = True,
                 contamination: bool = False, orbit: Orbit = None, optimize: bool = False,
                 eclipse: bool = False, method: str = 'pars', sampler: SuperSampler = None,
                 supersampling: int = 1, exptime: float = 0.020433598) -> None:

        super().__init__(npb, eccentric, constant_k, contamination, orbit, optimize, eclipse, method)

        assert isinstance(supersampling, int)
        assert supersampling > 0
        assert exptime > 0.

        # Initialize the supersampler
        # ---------------------------
        if sampler:
            self.sampler = sampler
        else:
            self.sampler = SuperSampler(supersampling, exptime)


    def evaluate_t(self, t: ndarray, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0., c: float = 0.) -> ndarray:
        """Evaluates the transit model given a time array and necessary parameters.

        Args:
            t: Array of time values [d].
            k: Planet to star radius ratio.
            ldc: Array of limb darkening coefficients.
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

        ldc = atleast_2d(ldc)
        npb = ldc.shape[0]

        if self.optimize:
            if not self.eclipse:
                tt, tm = extract_time_transit(t, k, t0, p, a, i, e, w, 1.1)
            else:
                tt, tm = extract_time_eclipse(t, k, t0, p, a, i, e, w, 1.1)
        else:
            tt, tm = t, np.ones_like(t, dtype=np.bool)

        # Check if we have multiple radius ratio (k) values, approximate the k with their
        # mean if yes, and calculate the area ratio factors.
        #
        if isinstance(k, ndarray):
            _k = k.mean()
            kf = (k/_k)**2
        else:
            _k = k
            kf = 1.

        z = self._calculate_z(tt, t0, p, a, i, e, w)
        supersampled_flux = self.evaluate_z(z, k, ldc, c)

        averaged_flux = ones((t.size, npb))
        averaged_flux[tm,:] = self.sampler.average(supersampled_flux)
        return kf*(averaged_flux-1.)+1.


    def _calculate_z(self, t: ndarray, t0: float, p: float, a: float, i: float, e: float = 0, w: float = 0) -> ndarray:
        z = self.orbit.projected_distance(self.sampler.sample(t), t0, p, a, i, e, w)
        return z if not self.eclipse else -z
