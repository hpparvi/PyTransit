"""Mandel-Agol transit model


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

import numpy as np
import pyopencl as cl
from os.path import dirname, join

from mandelagol_f import mandelagol as ma
from orbits_f import orbits as of
from tm import TransitModel

class MandelAgolCL(TransitModel):
    """
    Exoplanet transit light curve model by Mandel and Agol (2001).

    :param nldc: (optional)
        Number of limb darkening coefficients (1 = linear limb darkening, 2 = quadratic)

    :param supersampling: (optional)
        Number of subsamples to calculate for each light curve point

    :param exptime: (optional)
        Integration time for a single exposure, used in supersampling


    Examples
    --------

    Basic case::

      m = MandelAgolCL() # Initialize the model, use quadratic limb darkening law and all available cores
      I = m(z,k,u)       # Evaluate the model for projected distance z, radius ratio k, and limb darkening coefficients u
      
    """
    def __init__(self, supersampling=0, exptime=0.020433598, klims=(0.07,0.13), nk=128, nz=256, **kwargs):
        super(MandelAgolCL, self).__init__(2, 0, True, supersampling, exptime)

        self.ctx = kwargs.get('cl_ctx', cl.create_some_context())
        self.queue = kwargs.get('cl_queue', cl.CommandQueue(self.ctx))

        self.ed,self.le,self.ld,self.kt,self.zt = map(lambda a: np.array(a,dtype=np.float32,order='C'),
                                                      ma.calculate_interpolation_tables(klims[0],klims[1],nk,nz,4))
        self.klims = klims
        self.nk    = np.int32(nk)
        self.nz    = np.int32(nz)
        self.nptb  = 0
        self.npb   = 0
        self.u     = None 
        self.f     = None
        self.k0, self.k1 = map(np.float32, self.kt[[0,-1]])
        self.dk = np.float32(self.kt[1]-self.kt[0])
        self.dz = np.float32(self.zt[1]-self.zt[0])

        mf = cl.mem_flags
        self._b_ed = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ed)
        self._b_le = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.le)
        self._b_ld = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ld)
        self._b_kt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.kt)
        self._b_zt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.zt)

        self._b_u = None
        self._b_z = None
        self._b_f = None

        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'ma_lerp.cl'),'r').read()).build()


    def _eval(self, z, k, u, *nargs, **kwargs):
        z = np.asarray(z, np.float32)
        u = np.asarray(u, np.float32, order='C').T

        if (z.size != self.nptb) or (u.shape[1] != self.npb):
            if self._b_z is not None:
                self._b_z.release()
                self._b_f.release()
                self._b_u.release()

            self.npb = 1 if u.ndim == 1 else u.shape[1]
            self.nptb = z.size

            self.u = np.zeros((2,self.npb), np.float32)
            self.f = np.zeros((z.size, self.npb), np.float32)

            mf = cl.mem_flags    
            self._b_z = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
            self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.f.nbytes)
            self._b_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.u)

        cl.enqueue_copy(self.queue, self._b_z, z)
        cl.enqueue_copy(self.queue, self._b_u, u)

        self.prg.ma(self.queue, self.f.shape, None, self._b_z, self._b_u,
                    self._b_ed, self._b_le, self._b_ld,
                    np.float32(k), self.k0, self.k1,
                    self.nk, self.nz, self.dk, self.dz, self._b_f)

        cl.enqueue_copy(self.queue, self.f, self._b_f)
        return self.f


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

        #u = np.reshape(u, [-1, self.nldc]).T
        #c = np.ones(u.shape[1])*c
        flux = self._eval(z, k, u, c, update)
        return flux if np.asarray(u).ndim > 1 else flux.ravel()

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
