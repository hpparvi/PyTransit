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

    :param  interpolate: (optional)
        Switch telling if linear interpolation be used (default = False).

    :param supersampling: (optional)
        Number of subsamples to calculate for each light curve point

    :param exptime: (optional)
        Integration time for a single exposure, used in supersampling

    """
    def __init__(self, npol=100, nldc=2, nthr=0, interpolate=False, supersampling=0, exptime=0.020433598, eclipse=False):
        super(Gimenez,self).__init__(nldc,nthr,interpolate,supersampling,exptime,eclipse)
        self._eval = self._eval_interpolate if interpolate else self._eval_nointerpolate
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


    def _eval_nointerpolate(self, z, k, u, c, b, update):
        return g.eval(z, k, u, c, self.nthr, *self._coeff_arr)


    def _eval_interpolate(self, z, k, u, c, b, update):
        return g.eval_lerp(z, k, u, b, c, self.nthr, update, *self._coeff_arr)
