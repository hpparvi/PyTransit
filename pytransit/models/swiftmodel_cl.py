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
from typing import Optional, Union, Callable, Tuple

import numpy as np
import pyopencl as cl
from os.path import dirname, join

import warnings
from pyopencl import CompilerWarning

from numpy import array, uint32, float32, int32, asarray, zeros, ones, unique, atleast_2d, squeeze, ndarray, \
    concatenate, empty, linspace, diff, trapz
from .ldmodel import LDModel

from pytransit.models.numba.rrmodel import create_z_grid

from .transitmodel import TransitModel
from .numba.ldmodels import *

warnings.filterwarnings('ignore', category=CompilerWarning)

__all__ = ['SwiftModelCL']

class SwiftModelCL(TransitModel):
    ldmodels = {'uniform': (ld_uniform, ldi_uniform),
                'linear': (ld_linear, ldi_linear),
                'quadratic': (ld_quadratic, ldi_quadratic),
                'quadratic-tri': (ld_quadratic_tri, ldi_quadratic_tri),
                'nonlinear': ld_nonlinear,
                'general': ld_general,
                'square_root': ld_square_root,
                'logarithmic': ld_logarithmic,
                'exponential': ld_exponential,
                'power-2': ld_power_2,
                'power-2-pm': ld_power_2_pm}

    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 interpolate: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut=0.7, ng: int = 50, parallel: bool = False,
                 small_planet_limit: float = 0.05, cl_ctx=None, cl_queue=None) -> None:
        super().__init__()

        self.ctx = cl_ctx or cl.create_some_context()
        self.queue = cl_queue or cl.CommandQueue(self.ctx)

        self.splimit = small_planet_limit

        # Set up the limb darkening model
        # --------------------------------
        if isinstance(ldmodel, str):
            try:
                if isinstance(self.ldmodels[ldmodel], tuple):
                    self.ldmodel = self.ldmodels[ldmodel][0]
                    self.ldmmean = self.ldmodels[ldmodel][1]
                else:
                    self.ldmodel = self.ldmodels[ldmodel]
                    self.ldmmean = None
            except KeyError:
                print(
                    f"Unknown limb darkening model: {ldmodel}. Choose from [{', '.join(self.ldmodels.keys())}] or supply a callable function.")
                raise
        elif isinstance(ldmodel, LDModel):
            self.ldmodel = ldmodel
            self.ldmmean = ldmodel._integrate
        elif callable(ldmodel):
            self.ldmodel = ldmodel
            self.ldmmean = None
        elif isinstance(ldmodel, tuple) and callable(ldmodel[0]) and callable(ldmodel[1]):
            self.ldmodel = ldmodel[0]
            self.ldmmean = ldmodel[1]
        else:
            raise NotImplementedError

        self._ldmu = linspace(1, 0, 200)
        self._ldz = sqrt(1 - self._ldmu ** 2)

        # Set the basic variable
        # ----------------------
        self.klims = klims
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut

        self.npv = None
        self.nptb  = 0
        self.npb   = 0
        self.f     = None
        self.pv = array([])

        self.time  = None
        self.lcids = None
        self.pbids = None
        self.nsamples = None
        self.exptimes = None

        self.ze = None
        self.gs = None
        self.dg = None

        # Declare the buffers for the swift model arrays
        self._b_ze = None
        self._b_gs = None
        self._b_weights = None
        self._b_istar = None
        self._b_ldp = None
        self._b_ldw = None
        self._b_ks = None

        # Declare the buffers for the ld coefficients, time, and flux arrays. These will
        # be initialised when the model is first evaluated, and reinitialised if the
        # array sizes change.
        #
        self._b_time = None    # Time buffer
        self._b_f = None       # Flux buffer
        self._b_p = None       # Parameter vector buffer

        self._time_id = None   # Time array ID

        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'opencl','swiftmodel.cl'),'r').read()).build()

        self.init_siwft_arrays(self.zcut, self.ng, self.nzin, self.nzlimb)

    def init_siwft_arrays(self, zcut: float = 0.7, ng: int = 50, nzin: int = 30, nzlimb: int = 30):
        mf = cl.mem_flags

        self.ze, self.zm = create_z_grid(zcut, nzin, nzlimb)
        self.mu = sqrt(1-self.zm**2).astype('float32')
        self.ze = self.ze.astype('float32')
        self.nz = int32(self.ze.size)
        self.ng = int32(ng)

        self.gs = linspace(0, 0.9999, ng).astype('float32')
        self.dg = float32(diff(self.gs)[0])

        if self._b_ze is not None:
            self._b_ze.release()
            self._b_gs.releare()

        self._b_ze = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ze)
        self._b_gs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.gs)

    def set_data(self, time, lcids=None, pbids=None, nsamples=None, exptimes=None):
        mf = cl.mem_flags

        if self._b_time is not None:
            self._b_time.release()
            self._b_lcids.release()
            self._b_pbids.release()
            self._b_nsamples.release()
            self._b_etimes.release()

        self.nlc = uint32(1 if lcids is None else unique(lcids).size)
        self.npb = uint32(1 if pbids is None else unique(pbids).size)
        self.nptb = time.size

        self.time = asarray(time, dtype='float32')
        self.lcids = zeros(time.size, 'uint32') if lcids is None else asarray(lcids, dtype='uint32')
        self.pbids = zeros(self.nlc, 'uint32') if pbids is None else asarray(pbids, dtype='uint32')
        self.nsamples = ones(self.nlc, 'uint32') if nsamples is None else asarray(nsamples, dtype='uint32')
        self.exptimes = ones(self.nlc, 'float32') if exptimes is None else asarray(exptimes, dtype='float32')

        self._b_time = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.time)
        self._b_lcids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcids)
        self._b_pbids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pbids)
        self._b_nsamples = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.nsamples)
        self._b_etimes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.exptimes)


    def evaluate(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Optional[Union[float, ndarray]] = None,
                 w: Optional[Union[float, ndarray]] = None, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array.
        ldc
            Limb darkening coefficients as a 1D or 2D array.
        t0
            Transit center(s) as a float or a 1D vector.
        p
            Orbital period(s) as a float or a 1D vector.
        a
            Orbital semi-major axis (axes) divided by the stellar radius as a float or a 1D vector.
        i
            Orbital inclination(s) as a float or a 1D vector.
        e : optional
            Orbital eccentricity as a float or a 1D vector.
        w : optional
            Argument of periastron as a float or a 1D vector.

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously. In
        the first case, the orbital parameters should all be given as floats. In the second case, the orbital parameters
        should be given as a 1D array-like.

        Returns
        -------
        ndarray
            Modelled flux either as a 1D or 2D ndarray.
        """
        npv = 1 if isinstance(t0, float) else len(t0)
        k = asarray(k)

        if k.size == 1:
            nk = 1
        elif npv == 1:
            nk = k.size
        else:
            nk = k.shape[1]

        if e is None:
            e, w = 0.0, 0.0

        pvp = empty((npv, nk + 6), dtype=float32)
        pvp[:, :nk] = k
        pvp[:, nk] = t0
        pvp[:, nk + 1] = p
        pvp[:, nk + 2] = a
        pvp[:, nk + 3] = i
        pvp[:, nk + 4] = e
        pvp[:, nk + 5] = w

        return self.evaluate_pv(pvp, ldc, copy)

    def evaluate_ps(self, k, ldc, t0, p, a, i, e=0., w=0., copy=True) -> ndarray:
        """Evaluate the transit model for a set of scalar parameters.

           Parameters
           ----------
           k : array-like
               Radius ratio(s) either as a single float or an 1D array.
           ldc
             Limb darkening coefficients as a 1D or 2D array.
           t0 : float
               Transit center as a float.
           p : float
               Orbital period as a float.
           a : float
               Orbital semi-major axis divided by the stellar radius as a float.
           i : float
               Orbital inclination(s) as a float.
           e : float, optional
               Orbital eccentricity as a float.
           w : float, optional
               Argument of periastron as a float.

           Notes
           -----
           This version of the `evaluate` method is optimized for calculating a single transit model (such as when using a
           local optimizer). If you want to evaluate the model for a large number of parameters simultaneously, use either
           `evaluate` or `evaluate_pv`.

           Returns
           -------
           ndarray
               Modelled flux as a 1D ndarray.
           """
        if isinstance(k, float):
            pv = array([[k, t0, p, a, i, e, w]], float32)
        else:
            pv = concatenate([k, [t0, p, a, i, e, w]]).astype(float32)
        return self.evaluate_pv(pv, ldc, copy)

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a 2D parameter array.

           Parameters
           ----------
           pvp
               Parameter array with a shape `(npv, npar)` where `npv` is the number of parameter vectors, and each row
               contains a set of parameters `[k, t0, p, a, i, e, w]`. The radius ratios can also be given per passband,
               in which case the row should be structured as `[k_0, k_1, k_2, ..., k_npb, t0, p, a, i, e, w]`.

           Notes
           -----
           This version of the `evaluate` method is optimized for calculating several models in parallel, such as when
           using *emcee* for MCMC sampling.

           Returns
           -------
           ndarray
               Modelled flux either as a 1D or 2D ndarray.
           """
        pvp = atleast_2d(pvp)
        ldc = asarray(ldc)
        nk = pvp.shape[1] - 6

        # Release and reinitialise the GPU buffers if the parameter vector size changes
        if self.npv != pvp.shape[0]:
            self.npv = uint32(pvp.shape[0])
            self.spv = uint32(pvp.shape[1])

            if self._b_f is not None:
                self._b_f.release()
                self._b_p.release()

            if self._b_weights is not None:
                self._b_weights.release()
                self._b_ldp.release()
                self._b_ldw.release()
                self._b_istar.release()
                self._b_ks.release()

            self.pv = zeros(pvp.shape, float32)
            self.f = zeros((self.npv, self.nptb), float32)

            mf = cl.mem_flags
            self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * self.npv)
            self._b_p = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)
            self._b_weights = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.npv * self.ng * self.nz * float32().nbytes)
            self._b_ldp = cl.Buffer(self.ctx, mf.READ_ONLY, self.npv*self.npb*self.ng*float32().nbytes)
            self._b_ldw = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.npv*self.npb*self.ng*float32().nbytes)
            self._b_istar = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.npv*self.npb*float32().nbytes)
            self._b_ks = cl.Buffer(self.ctx, mf.READ_ONLY, self.npv*float32().nbytes)

        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((self.npv, self.npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(self.npv):
                    for ipb in range(self.npb):
                        istar[ipv, ipb] = 2 * pi * trapz(self._ldz * ldpi[ipv,ipb], self._ldz)

        # Copy the limb darkening profiles and their integrals to the GPU
        cl.enqueue_copy(self.queue, self._b_ldp, ldp.astype('float32'))
        cl.enqueue_copy(self.queue, self._b_istar, istar.astype('float32'))
        cl.enqueue_copy(self.queue, self._b_ks, pvp[:, :nk].mean(1).astype('float32'))

        # Copy the parameter vector to the GPU
        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        self.prg.calculate_weights(self.queue, (self.npv, self.ng, self.nz), None, self._b_ks, self._b_ze, self._b_gs, self._b_weights)
        self.prg.calculate_ldw(self.queue, (self.npv, self.npb, self.ng), None, self.nz, self._b_ldp, self._b_weights, self._b_ldw)

        self.prg.swift_pop(self.queue, (self.npv, self.nptb), None, self._b_time, self._b_istar, self._b_ldw, self.ng,
                           self.dg, self._b_lcids, self._b_pbids, self._b_p, self._b_nsamples, self._b_etimes,
                           self.spv, self.nlc, self.npb, self._b_f)

        if copy:
            cl.enqueue_copy(self.queue, self.f, self._b_f)
            return squeeze(self.f)
        else:
            return None