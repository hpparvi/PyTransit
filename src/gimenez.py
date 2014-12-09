"""Gimenez transit model

   A package offering an easy access to the Fortran implementation of the
   transit model by A. Gimenez (A&A 450, 1231--1237, 2006). The Fortran code is
   adapted from the original implementation at http://thor.ieec.uab.es/LRVCode,
   and includes several optimisations that make it several orders faster for
   light curves with thousands to millions of datapoints.


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

import numpy as np

from math import fabs
from gimenez_f import gimenez as g
from orbits_f import orbits as of
from tm import TransitModel

class Gimenez(TransitModel):
    """
    Exoplanet transit light curve model by A. Gimenez (A&A 450, 1231--1237, 2006).

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

    """
    def __init__(self, npol=100, nldc=2, nthr=0, lerp=False, supersampling=0, exptime=0.020433598):
        super(Gimenez,self).__init__(nldc,nthr,lerp,supersampling,exptime)
        self._eval = self._eval_lerp if lerp else self._eval_nolerp
        self._coeff_arr = g.init_arrays(npol, nldc)
        self.npol = npol


    def __call__(self, z, k, u, c=0., b=1e-8, update=True):
        """
        Evaluate the model

        :param z:
            Array of normalised projected distances
        
        :param k:
            Planet to star radius ratio
        
        :param u:
            Array of limb darkening coefficients
        
        :param c:
            Contamination factor (fraction of third light)
            
        :param b: (optional)
            Not used, ignore for now.

        :param update: (optional)
        """

        u = np.reshape(u, [-1, self.nldc]).T
        c = np.ones(u.shape[1])*c
        flux = self._eval(z, k, u, c, b, update)
        return flux if u.shape[1] > 1 else flux.ravel()


    def _eval_nolerp(self, z, k, u, c, b, update):
        return g.eval(z, k, u, c, self.nthr, *self._coeff_arr)


    def _eval_lerp(self, z, k, u, c, b, update):
        return g.eval_lerp(z, k, u, b, c, self.nthr, update, *self._coeff_arr)


    def evaluate(self, t, k, u, t0, p, a, i, e=0., w=0., c=0., update=True, lerp_z=False):
        """Evaluates the transit model for the given parameters.

        :param t:
            Array of time values

        :param k:
            Radius ratio(s), can be a single float or an array of values for each passband

        :param u:
            Either 1D (nldc or npb*nldc) or 2D (npb,nldc) array of limb darkening coefficients (ldcs).
            If 2D, the model returns ``npb`` light curves, each corresponding to an ldc set
            described by a row of the ldc array.

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
            Contamination factor(s) ``c`` as a float or an array with ``c`` for each passband
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
            
        z = self._calculate_z(t, t0, p, a, i, e, w, lerp_z)
        flux = self.__call__(z, _k, u, c, update)

        if self.ss:
            if npb == 1:
                flux = flux.reshape((self.npt, self.nss)).mean(1)
            else:
                flux = flux.reshape((self.npt, self.nss, npb)).mean(1)

        return kf*(flux-1.)+1.
