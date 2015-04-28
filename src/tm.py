"""


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

from math import fabs
import numpy as np
from orbits_f import orbits as of

class TransitModel(object):
    """
    Exoplanet transit light curve model 

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
    def __init__(self, nldc=2, nthr=0, lerp=False, supersampling=0, exptime=0.020433598, eclipse=False):
        self.nldc = nldc
        self.nthr = nthr
        self.ss  = bool(supersampling)
        self.nss = int(supersampling)
        self.exp = exptime
        self.time = None
        self.eclipse = eclipse

    def __call__(self, z, k, u, c=0., update=True):
        """Evaluate the model

        :param z:
            Array of normalised projected distances
        
        :param k:
            Planet to star radius ratio
        
        :param u:
            Array of limb darkening coefficients
        
        :param c:
            Contamination factor (fraction of third light)
            
        :param update: (optional)
        """
        return self._eval(z, k, u, c, update)


    def _eval_nolerp(self, z, k, u, c, update):
        raise NotImplementedError()

    def _eval_lerp(self, z, k, u, c, update):
        raise NotImplementedError

    def evaluate(self, t, k, u, t0, p, a, i, e=0., w=0., c=0., update=True, lerp_z=False):
        raise NotImplementedError()

    def _calculate_z(self, t, t0, p, a, i, e=0, w=0, lerp_z=False):
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
                if self.eclipse:
                    z *= -1.
            elif fabs(e) < 0.2:
                z = of.z_eccentric_ps3(self._time, t0, p, a, i, e, w, nthreads=self.nthr)
            else:
                z = of.z_eccentric_newton(self._time, t0, p, a, i, e, w, nthreads=self.nthr)

        return z
