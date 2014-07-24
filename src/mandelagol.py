"""Mandel-Agol transit model


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

import numpy as np

from mandelagol_f import mandelagol as ma
from orbits_f import orbits as of
from tm import TransitModel

class MandelAgol(TransitModel):
    """
    Exoplanet transit light curve model by XX

    :param npol: (optional)

    :param nldc: (optional)
        Number of limb darkening coefficients (1 = linear limb darkening, 2 = quadratic)

    :param nthr: (optional)
        Number of threads (default = number of cores)

    :param  lerp: (optional)
        Switch telling if linear interpolation be used (default = False).

    :param supersampling: (optional)
        Number of subsamples to calculate for each light curve point

    :param exptime: (optional)
        Integration time for a single exposure, used in supersampling


    Examples
    --------

    Basic case::

      m = MandelAgol() # Initialize the model, use quadratic limb darkening law and all available cores
      I = m(z,k,u)     # Evaluate the model for projected distance z, radius ratio k, and limb darkening coefficients u
      
    Use linear interpolation::

      m = MandelAgol(lerp=True) # Initialize the model
      I = m(z,k,u)           # Evaluate the model

    Use linear interpolation, two different sets of z::

      m  = MandelAgol(lerp=True)   # Initialize the model
      I1 = m(z1,k,u)               # Evaluate the model for z1, update the interpolation table
      I2 = m(z2,k,u, update=False) # Evaluate the model for z2, don't update the interpolation table
    """
    def __init__(self, nldc=2, nthr=0, lerp=False, supersampling=0, exptime=0.020433598):
        if not (nldc == 0 or nldc == 2):
            raise NotImplementedError('Only the uniform and quadratic Mandel-Agol models are currently supported.')
        super(MandelAgol, self).__init__(nldc, nthr, lerp, supersampling, exptime)
    
        if nldc == 0:
            self._eval_nolerp = self._eval_nolerp_uniform
        else:
            self._eval_nolerp = self._eval_nolerp_quadratic

        self._eval = self._eval_lerp if lerp else self._eval_nolerp


    def _eval_nolerp_quadratic(self, z, k, u, c, update):
        return ma.eval_quad(z, k, u, c, self.nthr)

    def _eval_nolerp_uniform(self, z, k, u, c, update):
        return ma.eval_uniform(z, k, c, self.nthr)


    def _eval_lerp(self, z, k, u, c, update):
        return ma.eval_lerp(z, k, u, c, self.nthr, update, *self._coeff_arr)


    def evaluate(self, t, k, u, t0, p, a, i, e=0., w=0., c=0., update=True, lerp_z=False):
        """Evaluates the transit model for the given parameters.

        :param t:
            Array of time values

        :param k:
            Radius ratio

        :param u:
            Quadratic limb darkening coefficients [u1,u2]

        :param t0:
            Zero epoch

        :param p:
            Orbital period

        :param a:
            Scaled semi-major axis

        :param i:
            Inclination

        :param e: (optional, default=0)
            Eccentricity

        :param w: (optional, default=0)
            Argument of periastron

        :param c: (optional, default=0)
            Contamination factor ``c``
        """
            
        z = self._calculate_z(t, t0, p, a, i, e, w, lerp_z)

        flux = self.__call__(z, k, u, c, update)

        if self.ss:
            flux = flux.reshape((self.npt, self.nss)).mean(1)

        return flux
