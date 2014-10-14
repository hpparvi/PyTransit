"""Mandel-Agol transit model (ApJ 580, L171–L175 2002).

.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

import numpy as np

from mandelagol_f import mandelagol as ma
from orbits_f import orbits as of
from tm import TransitModel

class MandelAgol(TransitModel):
    """
    Exoplanet transit light curve model by Mandel and Agol.

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
    def __init__(self, nldc=2, nthr=0, lerp=False, supersampling=0, exptime=0.020433598, klims=(0.07,0.13), nk=128, nz=256):
        if not (nldc == 0 or nldc == 2):
            raise NotImplementedError('Only the uniform and quadratic Mandel-Agol models are currently supported.')
        super(MandelAgol, self).__init__(nldc, nthr, lerp, supersampling, exptime)
        self.lerp = lerp
            
        ## Uniform stellar disk
        if nldc == 0:
            self._eval = self._eval_uniform

        ## Quadratic limb darkening
        else:
            if self.lerp:
                self.ed,self.le,self.ld,self.kt,self.zt = ma.calculate_interpolation_tables(klims[0],klims[1],nk,nz,4)
                self.klims = klims
                self.nk = nk
                self.nz = nz

            self._eval = self._eval_quadratic


    def _eval_uniform(self, z, k, u, c, update=True):
        """Wraps the Fortran implementation of a transit over a uniform disk

           :param z: 
               Array of normalised projected distances

           :param k: 
               Planet to star radius ratio

           :param u:
                Not used

           :param c: 
                Array of contamination values as [c1, c2, ... c_npb]

           :param update: 
                Not used
        """
        return ma.eval_uniform(z, k, c, self.nthr)


    def _eval_quadratic(self, z, k, u, c, update=True):
        """Wraps the Fortran implementation of the quadratic Mandel-Agol model

           :param z: 
               Array of normalised projected distances

           :param k: 
               Planet to star radius ratio

           :param u:
                Array of limb darkening coefficients arranged as [u1, v1, u2, v2, ... u_npb, v_npb]

           :param c: 
                Array of contamination values as [c1, c2, ... c_npb]

           :param update: 
                Not used
        """
        u = np.asarray(u).ravel()
        npb = len(u)//2
        if not isinstance(c, np.ndarray) or c.size != npb:
            c = c*np.ones(npb)
            
        if self.lerp:
            return ma.eval_quad_bilerp(z,k,u,c,self.nthr, self.ed,self.ld,self.le,self.kt,self.zt)
        else:
            return ma.eval_quad(z, k, u, c, self.nthr)


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
            
        """
        flux = self._eval(z, k, u, c, update)
        return flux if np.asarray(u).size > 2 else flux.ravel()


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
        flux = self.__call__(z, k, u, c, update)

        if self.ss:
            flux = flux.reshape((self.npt, self.nss)).mean(1)

        return kf*(flux-1.)+1.
