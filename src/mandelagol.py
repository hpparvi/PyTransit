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

from numpy import atleast_2d, atleast_1d

import pytransit.ma_quadratic_nb as maq
import pytransit.ma_uniform_nb as mau
from .transitmodel import SuperSampledTransitModel
from .orbits import Orbit
from .supersampler import SuperSampler

class MandelAgol(SuperSampledTransitModel):
    """Quadratic Mandel-Agol transit model (ApJ 580, L171-L175 2002).

    This class wraps the Numba implementations of the
        - uniform
        - quadratic
        - chromosphere
    Mandel & Agol transit models.
    """

    models = 'uniform quadratic interpolated_quadratic chromosphere'.split()
    
    def __init__(self, npb: int = 1, eccentric: bool = False, constant_k: bool = True,
                 contamination: bool = False, orbit: Orbit = None, optimize: bool = False,
                 eclipse: bool = False, method: str = 'pars',
                 sampler: SuperSampler = None, supersampling: int = 1, exptime: float = 0.020433598,
                 interpolate: bool = True, klims: tuple = (0.01, 0.25), nk: int = 256, nz: int = 256,
                 model: str = 'quadratic', **kwargs):
        """Initialise the model.

        Args:
            interpolate: If True, evaluates the model using interpolation (default = True).
            supersampling: Number of subsamples to calculate for each light curve point (default=0).
            exptime: Integration time for a single exposure, used in supersampling default=(0.02).
            eclipse: If True, evaluates the model for eclipses. If false, eclipses are filtered out (default = False). 
            klims: Minimum and maximum radius ratio if interpolation is used as (kmin,kmax).
            nk: Interpolation table resolution in k.
            nz: Interpolation table resolution in z.
        """
        assert model in self.models, 'Unknown model {}'.format(model)

        super().__init__(npb, eccentric, constant_k, contamination, orbit, optimize, eclipse,
                         method, sampler, supersampling, exptime)
        self.interpolate = interpolate

        if model == 'chromosphere':
            self.evaluate_z = self._eval_chromosphere
        elif model == 'uniform':
            self.evaluate_z = self._eval_uniform
        else:
            if model == 'interpolated_quadratic' or self.interpolate:
                self.ed, self.le, self.ld, self.kt, self.zt = maq.calculate_interpolation_tables(klims[0], klims[1], nk, nz)
                self.klims = klims
                self.nk = nk
                self.nz = nz
            self.evaluate_z = self._eval_quadratic


    def _eval_uniform(self, z, k, u=None, c=0.0):
        """Wraps the Numba implementation of a transit over a uniform disk

            Args:
                z: Array of normalised projected distances.
                k: Planet to star radius ratio.
                u: Array of limb darkening coefficients, not used.
                c: Array of contamination values as [c1, c2, ... c_npb].
                update: Not used.

            Returns:
                An array of model flux values for each z.
        """
        return mau.eval_uniform(z, k, c)

    
    def _eval_chromosphere(self, z, k, u=None, c=0.0):
        """Wraps the Numba implementation of a transit over a chromosphere

            Args:
                z: Array of normalised projected distances.
                k: Planet to star radius ratio.
                c: Array of contamination values as [c1, c2, ... c_npb].

            Returns:
                An array of model flux values for each z.
        """
        return maq.eval_chromosphere(z, k, c)

    
    def _eval_quadratic(self, z, k, u, c=0.0):
        """Wraps the Numba implementation of the quadratic Mandel-Agol model

          Args:
                z: Array of normalised projected distances
                k: Planet to star radius ratio
                u: Array of limb darkening coefficients
                c: scalar or array of contamination values as [c1, c2, ... c_npb]
        """
        u = atleast_2d(u)
        c = atleast_1d(c)

        if self.interpolate:
            return maq.eval_quad_ip(z, k, u, c, self.ed, self.ld, self.le, self.kt, self.zt)
        else:
            return maq.eval_quad(z, k, u, c)[0]


    
class MAChromosphere(MandelAgol):
    def __init__(self, npb: int = 1, eccentric: bool = False, constant_k: bool = True, contamination: bool = False,
                 orbit: Orbit = None, optimize: bool = False, eclipse: bool = False,
                 sampler: SuperSampler = None, supersampling: int = 1, exptime: float = 0.020433598):
        super().__init__(npb, eccentric, constant_k, contamination, orbit, optimize, eclipse, sampler,
                         model='chromosphere', supersampling=supersampling, exptime=exptime)

    def evaluate_t(self, t, k, t0, p, a, i, e=0., w=0., c=0.):
        return super().evaluate_t(t, k, [], t0, p, a, i, e, w, c)



class MAUniform(MandelAgol):
    def __init__(self, npb: int = 1, eccentric: bool = False, constant_k: bool = True, contamination: bool = False,
                 orbit: Orbit = None, optimize: bool = False, eclipse: bool = False,
                 sampler: SuperSampler = None, supersampling: int = 1, exptime: float = 0.020433598):
        super().__init__(npb, eccentric, constant_k, contamination, orbit, optimize, eclipse, sampler,
                         model='uniform', supersampling=supersampling, exptime=exptime)
