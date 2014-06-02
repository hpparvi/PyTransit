"""Gimenez transit model

   A package offering an easy access to the Fortran implementation of the
   transit model by A. Gimenez (A&A 450, 1231--1237, 2006). The Fortran code is
   adapted from the original implementation at http://thor.ieec.uab.es/LRVCode,
   and includes several optimisations that make it several orders faster for
   light curves with thousands to millions of datapoints.


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

from math import fabs
import numpy as np
from math import fabs

from gimenez_f import gimenez as g
from orbits_f import orbits as of

class Gimenez(object):
    """
    Exoplanet transit light curve model by A. Gimenez (A&A 450, 1231--1237, 2006).

    :param npol: (optional)

    :param nldc: (optional)
        Number of limb darkening coefficients (1 = linear limb darkening, 2 = quadratic)

    :param nthr: (optional)
        Number of threads (default = number of cores)

    :param  lerp: (optional)
        Switch telling if linear interpolation be used (default = False).


    Examples
    --------

    Basic case::

      m = Gimenez() # Initialize the model, use quadratic limb darkening law and all available cores
      I = m(z,k,u)  # Evaluate the model for projected distance z, radius ratio k, and limb darkening coefficients u
      
    Use linear interpolation::

      m = Gimenez(lerp=True) # Initialize the model
      I = m(z,k,u)           # Evaluate the model

    Use linear interpolation, two different sets of z::

      m  = Gimenez(lerp=True)      # Initialize the model
      I1 = m(z1,k,u)               # Evaluate the model for z1, update the interpolation table
      I2 = m(z2,k,u, update=False) # Evaluate the model for z2, don't update the interpolation table
    """
    def __init__(self, npol=100, nldc=2, nthr=0, lerp=False, supersampling=0, exptime=0.020433598):
        self._coeff_arr = g.init_arrays(npol, nldc)
        self.npol = npol
        self.nldc = nldc
        self.nthr = nthr

        self.ss  = bool(supersampling)
        self.nss = supersampling
        self.exp = exptime

        self.time = None

        self._eval = self._eval_lerp if lerp else self._eval_nolerp


    def __call__(self, z, k, u, c=0., b=1e-8, update=True):
        """Evaluate the model

        :param z:
            Array of normalised projected distances
        
        :param k:
            Planet to star radius ratio
        
        :param u:
            Array of limb darkening coefficients
        
        :param c:
            Contamination factor (fraction of third light)
            
        :param b: (optional)
        :param update: (optional)
        """

        u = np.reshape(u, [-1, self.nldc]).T
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
            Array of limb darkening coefficients
        
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
            Contamination factor
        """

        u = np.asfortranarray(u)

        ## Calculate the supersampling time array if not cached
        ##
        if t is not self.time:
            self.time = t
            self._time = np.asarray(t)
            self.npt = self._time.size

            if self.ss:
                self.dt = self.exp / float(self.nss)
                self._time = np.array([[tt + (-1)**(iss%2)*(0.5*self.dt + iss//2*self.dt)
                                        for iss in range(self.nss)] for tt in self._time]).ravel()

        ## Calculate the normalised projected distance
        ##
        if lerp_z:
            z = of.z_eccentric_ip(self._time, t0, p, a, i, e, w, nthreads=self.nthr, update=True)
        else:
            if fabs(e) < 0.01:
                z = of.z_circular(self._time, t0, p, a, i, nthreads=self.nthr)
            elif fabs(e) < 0.2:
                z = of.z_eccentric_ps3(self._time, t0, p, a, i, e, w, nthreads=self.nthr)
            else:
                z = of.z_eccentric_newton(self._time, t0, p, a, i, e, w, nthreads=self.nthr)

        ## Check if we have multiple radius ratio (k) values, approximate the k with their
        ## mean if yes, and calculate the area ratio factors.
        ## 
        if isinstance(k, np.ndarray):
            _k = k.mean()
            kf = (k/_k)**2
        else:
            _k = k
            kf = 1.
            
        flux = self.__call__(z, _k, u, c, update)

        if self.ss:
            flux = flux.reshape((self.npt, self.nss, u.shape[0])).mean(1)

        flux = kf*(flux-1.)+1.

        return flux
        

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as pl

    z = np.linspace(1e-7,1.3,1000)
    for nldc,ldc,ls in zip([0,1,2], [[], [0], [0.3,0]], ['-','--',':']):
        pl.plot(Gimenez(nldc=nldc, lerp=True)(z, 0.1, ldc), ls=ls,  c='0.0')
    pl.ylim(0.988, 1.001)
    pl.show()
