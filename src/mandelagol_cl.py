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


"""Mandel-Agol transit model


.. moduleauthor:: Hannu Parviainen <hannu.parviainen@astro.ox.ac.uk>
"""

import numpy as np
import pyopencl as cl
from os.path import dirname, join

import pytransit.mandelagol_py as ma
from pytransit.tm import TransitModel

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
    def __init__(self, supersampling=1, exptime=0.020433598, eclipse=False, klims=(0.05,0.25), nk=256, nz=256, **kwargs):
        super(MandelAgolCL, self).__init__(nldc=2, nthr=0, interpolate=True, supersampling=1, exptime=1,
                                           eclipse=eclipse, klims=klims, nk=nk, nz=nz)

        super().__init__(nldc=2, nthr=0, interpolate=True, supersampling=1, exptime=1,
                         eclipse=eclipse, klims=klims, nk=nk, nz=nz)

        self.ctx = kwargs.get('cl_ctx', cl.create_some_context())
        self.queue = kwargs.get('cl_queue', cl.CommandQueue(self.ctx))

        self.ed,self.le,self.ld,self.kt,self.zt = map(lambda a: np.array(a,dtype=np.float32,order='C'),
                                                      ma.calculate_interpolation_tables(klims[0],klims[1],nk,nz))
        self.klims = klims
        self.nk    = np.int32(nk)
        self.nz    = np.int32(nz)
        self.nptb  = 0
        self.npb   = 0
        self.u     = np.array([])
        self.f     = None
        self.k0, self.k1 = map(np.float32, self.kt[[0,-1]])
        self.dk = np.float32(self.kt[1]-self.kt[0])
        self.dz = np.float32(self.zt[1]-self.zt[0])
        self.nss = int32(supersampling)
        self.etime = float32(exptime)

        mf = cl.mem_flags

        # Create the buffers for the Mandel & Agol coefficient arrays
        self._b_ed = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ed)
        self._b_le = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.le)
        self._b_ld = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ld)
        self._b_kt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.kt)
        self._b_zt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.zt)

        # Declare the buffers for the ld coefficients, time, and flux arrays. These will
        # be initialised when the model is first evaluated, and reinitialised if the
        # array sizes change.
        #
        self._b_u = None       # Limb darkening coefficient buffer
        self._b_t = None       # Time buffer
        self._b_f = None       # Flux buffer
        self._b_p = None       # Parameter vector buffer

        self._time_id = None   # Time array ID

        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'ma_lerp.cl'),'r').read()).build()
        #self.prg = cl.Program(self.ctx, open('../src/ma_lerp.cl','r').read()).build()


    def _eval(self, t, k, u, t0, p, a, i, e=0., w=0., c=0.):
        u = np.array(u, np.float32, order='C').T

        # Release and reinitialise the GPU buffers if the sizes of the time or
        # limb darkening coefficient arrays change.
        if (t.size != self.nptb) or (u.size != self.u.size):
            if self._b_t is not None:
                self._b_t.release()
                self._b_f.release()
                self._b_u.release()
                self._b_p.release()

            self.npb = 1 if u.ndim == 1 else u.shape[1]
            self.nptb = t.size

            self.u = np.zeros((2,self.npb), np.float32)
            self.f = np.zeros((t.size, self.npb), np.float32)
            self.pv = np.zeros(7, np.float32)

            mf = cl.mem_flags
            self._b_t = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t)
            self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, t.nbytes)
            self._b_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.u)
            self._b_p = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)

        # Copy the time array to the GPU if it has been changed
        if id(t) != self._time_id:
            cl.enqueue_copy(self.queue, self._b_t, t)
            self._time_id = id(t)

        # Copy the limb darkening coefficient array to the GPU
        cl.enqueue_copy(self.queue, self._b_u, u)

        # Copy the parameter vector to the GPU
        self.pv[:] = array([k, t0, p, a, i, e, w], dtype=float32)
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        self.prg.ma_eccentric(self.queue, t.shape, None, self._b_t, self._b_p, self._b_u,
                    self._b_ed, self._b_le, self._b_ld, self.nss, self.etime,
                    self.k0, self.k1, self.nk, self.nz, self.dk, self.dz, self._b_f)
        cl.enqueue_copy(self.queue, self.f, self._b_f)
        return self.f


    def _eval_pop(self, t, pvp, u):
        u = np.array(u, np.float32, order='C').T
        self.npv = uint32(pvp.shape[0])
        self.spv = uint32(pvp.shape[1])

        # Release and reinitialise the GPU buffers if the sizes of the time or
        # limb darkening coefficient arrays change.
        if (t.size != self.nptb) or (u.size != self.u.size):
            if self._b_t is not None:
                self._b_t.release()
                self._b_f.release()
                self._b_u.release()
                self._b_p.release()

            self.npb = 1 if u.ndim == 1 else u.shape[1]
            self.nptb = t.size

            self.u = np.zeros((2, self.npb), np.float32)
            self.f = np.zeros((t.size, self.npv), np.float32)
            self.pv = np.zeros(pvp.shape, np.float32)

            mf = cl.mem_flags
            self._b_t = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t)
            self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, t.nbytes * self.npv)
            self._b_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.u)
            self._b_p = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)

        # Copy the time array to the GPU if it has been changed
        if id(t) != self._time_id:
            cl.enqueue_copy(self.queue, self._b_t, t)
            self._time_id = id(t)

        # Copy the limb darkening coefficient array to the GPU
        cl.enqueue_copy(self.queue, self._b_u, u)

        # Copy the parameter vector to the GPU
        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        self.prg.ma_eccentric_pop(self.queue, (t.size, self.npv), None, self._b_t, self._b_p, self._b_u,
                                  self._b_ed, self._b_le, self._b_ld, self.nss, self.etime,
                                  self.k0, self.k1, self.nk, self.nz, self.dk, self.dz, self.spv, self._b_f)
        cl.enqueue_copy(self.queue, self.f, self._b_f)
        return self.f


    def __call__(self, z, k, u, c=0., b=1e-8, update=True, **kwargs):
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
        flux = self._eval(z, k, u, c, update, **kwargs)
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
