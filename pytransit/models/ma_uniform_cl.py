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


"""OpenCL implementation of the transit over a uniform disk (Mandel & Agol, ApJ 580, L171-L175 2002).
"""

import warnings
from os.path import dirname, join
from pathlib import Path
from typing import Union, Optional
from pkg_resources import resource_filename

import pyopencl as cl
from pyopencl import CompilerWarning
from numpy import array, uint32, float32, asarray, zeros, ones, unique, atleast_2d, squeeze, ndarray, empty, concatenate
from .transitmodel import TransitModel
from ..orbits.taylor_z import vajs_from_paiew_v

warnings.filterwarnings('ignore', category=CompilerWarning)


class UniformModelCL(TransitModel):
    """Exoplanet transit over a uniform disk (Mandel & Agol, ApJ 580, L171-L175 2002).
    """

    def __init__(self, cl_ctx=None, cl_queue=None) -> None:
        super().__init__()

        # Initialize the OpenCL context and queue
        # ---------------------------------------
        self.ctx = cl_ctx or cl.create_some_context()
        self.queue = cl_queue or cl.CommandQueue(self.ctx)

        # Declare stuff
        # -------------
        self.nptb: int = 0  # Number of datapoints
        self.nlc: int  = 0  # Number of light curves
        self.npb: int  = 0  # Number of passpands

        # Declare the arrays
        # ------------------
        self.flux:     ndarray = None       # Model flux values
        self.time:     ndarray = None       # Mid-exposure times
        self.lcids:    ndarray = None       # Light curve indices
        self.pbids:    ndarray = None       # Passband indices
        self.nsamples: ndarray = None       # Number of samples per light curve
        self.exptimes: ndarray = None       # Exposure times per light curve
        self.vajs:     ndarray = None
        self.pv:       ndarray = array([])  # Parameter vector array

        # Declare the buffers
        # --------------------
        self._b_time = None    # Buffer for the mid-exposure times
        self._b_flux = None    # Buffer for the model flux values
        self._b_pv   = None    # Parameter vector buffer
        self._b_vajs = None

        # Build the program
        # -----------------
        rd = Path(resource_filename('pytransit', 'models/opencl'))
        po = rd / 'orbits.cl'
        pm = rd / 'ma_uniform.cl'
        self.prg = cl.Program(self.ctx, po.read_text() + pm.read_text()).build()


    def set_data(self, time: ndarray, lcids: ndarray = None, pbids: ndarray = None, nsamples: ndarray = None, exptimes: ndarray = None):
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

    def evaluate(self, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Optional[Union[float, ndarray]] = None,
                 w: Optional[Union[float, ndarray]] = None, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array.
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
        return self.evaluate_pv(pvp, copy)

    def evaluate_ps(self, k, t0, p, a, i, e=0., w=0., copy=True):
        """Evaluate the transit model for a set of scalar parameters.

        Parameters
        ----------
        k : array-like
            Radius ratio(s) either as a single float or an 1D array.
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
        return self.evaluate_pv(pv, copy)

    def evaluate_pv(self, pvp, copy=True):
        """Evaluate the transit model for 2D parameter array.

        Parameters
        ----------
        pvp
            Parameter array with a shape `(npv, npar)` where `npv` is the number of parameter vectors, and each row
            contains a set of parameters `[k, t0, p, a, i, e, w]`. The radius ratios can also be given per passband,
            in which case the row should be structured as `[k_0, k_1, k_2, ..., k_npb, t0, p, a, i, e, w]`.
        ldc
            Limb darkening coefficient array with shape `(npv, 2*npb)`, where `npv` is the number of parameter vectors
            and `npb` is the number of passbands.

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
        self.npv = uint32(pvp.shape[0])
        self.spv = uint32(pvp.shape[1])

        if pvp.size != self.pv.size:
            if self._b_flux is not None:
                self._b_flux.release()
                self._b_pv.release()
                self._b_vajs.release()

            self.pv = zeros(pvp.shape, float32)
            self.flux = zeros((self.npv, self.nptb), float32)
            self.vajs = zeros((self.npv, 9), float32)

            mf = cl.mem_flags
            self._b_flux = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * self.npv)
            self._b_pv = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pv)
            self._b_vajs = cl.Buffer(self.ctx, mf.READ_WRITE, float32().nbytes*self.npv*9)

        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_pv, self.pv)

        self.prg.vajs_from_paiew_v(self.queue, (self.npv, ), None, self.spv, self._b_pv, self._b_vajs)
        self.prg.uniform_eccentric_pop(self.queue, (self.npv, self.nptb), None, self._b_time, self._b_lcids, self._b_pbids,
                                  self._b_pv, self._b_nsamples, self._b_etimes, self._b_vajs,
                                  self.spv, self.nlc, self.npb, self._b_flux)

        if copy:
            cl.enqueue_copy(self.queue, self.flux, self._b_flux)
            return squeeze(self.flux)
        else:
            return None
