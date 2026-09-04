#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
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
from os.path import dirname, join
from warnings import warn, filterwarnings

import pyopencl as cl
from pyopencl import CompilerWarning

from numpy import (array, uint32, float32, float64, int32, asarray, ascontiguousarray, zeros, ones,
                   unique, atleast_1d, atleast_2d, squeeze, ndarray,
                   concatenate, empty, linspace, sqrt, pi, isnan, isscalar, trapezoid)

from ..ldmodel import LDModel
from ..limb_darkening import (ld_uniform, ldi_uniform, ld_linear, ldi_linear, ld_quadratic, ldi_quadratic,
                              ld_quadratic_tri, ldi_quadratic_tri, ld_nonlinear, ldi_nonlinear, ld_general, ldi_general,
                              ld_square_root, ldi_square_root, ld_logarithmic, ldi_logarithmic,
                              ld_exponential, ldi_exponential, ld_power_2, ldi_power_2, ld_power_2_pm, ldi_power_2_pm,
                              evaluate_ld, evaluate_ldi)
from ..transitmodel import TransitModel
from .._deprecation import deprecated_evaluation_method
from numba import njit
from meepmeep.backends.numba.point2d import solve2d, bounding_box
from meepmeep.backends.opencl import read_kernel_source, build_options

from .common import quadrature_rules, profile_grid, radius_ratio_array, CUBIC_MATRICES

filterwarnings('ignore', category=CompilerWarning)

__all__ = ['RoadRunnerModelCL']


@njit(cache=True)
def _expansion_arrays(valid, p, a, i, e, w, k0, exptimes):
    """Taylor series coefficients for the sky position, and the transit bounding boxes.

    A port of the orbit half of `model_full.rr_precompute`, so that both backends evaluate the
    same expansion. Compiled because the loop over the parameter vectors is otherwise a Python
    loop over `solve2d`, whose dispatch overhead dominates the whole evaluation for a large
    population: ~6 ms for a thousand parameter vectors, against ~7 ms for everything else.
    """
    npv, nlc = p.size, exptimes.size
    xyc = zeros((npv, 2, 5))
    bbs = zeros((npv, nlc, 2))
    for ipv in range(npv):
        if not valid[ipv]:
            continue
        xyc[ipv] = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        bt1, bt4 = bounding_box(k0[ipv], xyc[ipv])
        for ilc in range(nlc):
            bbs[ipv, ilc, 0] = bt1 - (0.003 + exptimes[ilc])
            bbs[ipv, ilc, 1] = bt4 + (0.003 + exptimes[ilc])
    return xyc, bbs


def _dtype_for_precision(ctx, precision: str):
    """Map a precision name to a NumPy dtype, checking that the device supports it."""
    if precision == 'double':
        for device in ctx.devices:
            if not device.double_fp_config:
                raise RuntimeError(f"The OpenCL device '{device.name}' does not support double "
                                   f"precision (cl_khr_fp64). Use precision='single' instead.")
        return float64
    elif precision == 'single':
        return float32
    else:
        raise ValueError(f"Unknown precision '{precision}', expected 'single' or 'double'.")


class RoadRunnerModelCL(TransitModel):
    """OpenCL implementation of the RoadRunner transit model (Parviainen, MNRAS 499, 1633, 2020).

    A GPU implementation of :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` with the
    same accuracy settings, `nq` and `ng`. The limb darkening profile is evaluated on the host on
    the same fixed grid of mu as in the Numba model and uploaded to the device, which integrates
    it over the planet's footprint by the same geometry-matched Gauss quadrature, tabulates the
    mean intensity under the planet against the grazing parameter, and reads the table with a
    cubic lookup during the evaluation. Any radially symmetric limb darkening model the Numba
    model accepts works here as well.

    The model computes in single precision; see :doc:`/guide/opencl` for what that implies.

    This class is not exported at the package top level; import it from its module::

        from pytransit.models.roadrunner.rrmodel_cl import RoadRunnerModelCL
    """
    ldmodels = {'uniform': (ld_uniform, ldi_uniform),
                'linear': (ld_linear, ldi_linear),
                'quadratic': (ld_quadratic, ldi_quadratic),
                'quadratic-tri': (ld_quadratic_tri, ldi_quadratic_tri),
                'nonlinear': (ld_nonlinear, ldi_nonlinear),
                'general': (ld_general, ldi_general),
                'square_root': (ld_square_root, ldi_square_root),
                'logarithmic': (ld_logarithmic, ldi_logarithmic),
                'exponential': (ld_exponential, ldi_exponential),
                'power-2': (ld_power_2, ldi_power_2),
                'power-2-pm': (ld_power_2_pm, ldi_power_2_pm)}

    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 interpolate: Optional[bool] = None, klims: Optional[tuple] = None, nk: Optional[int] = None,
                 nzin: Optional[int] = None, nzlimb: Optional[int] = None, zcut: Optional[float] = None,
                 ng: int = 100, parallel: bool = False, small_planet_limit: float = 0.05, cl_ctx=None,
                 cl_queue=None, nz: Optional[int] = None, nq: int = 8,
                 precision: str = 'single') -> None:
        """The OpenCL RoadRunner transit model.

        Parameters
        ----------
        ldmodel
            Limb darkening model: either the name of a built-in model, a callable returning the
            stellar intensity profile as a function of mu, a tuple of callables returning the
            profile and its integral over the stellar disk, or an ``LDModel`` instance.
        nq : int, optional
            Number of quadrature nodes per segment used to integrate the intensity profile over
            the planet's footprint.
        ng : int, optional
            Number of grazing parameter nodes in the mean intensity table, split at the limb
            contact and interpolated with cubics.
        precision : str, optional
            Device floating point type, ``'single'`` (the default) or ``'double'``. Double
            precision needs `cl_khr_fp64`, which most GPUs support but run at a fraction of the
            single precision rate -- typically a sixty-fourth on consumer NVIDIA cards -- so it
            is an opt-in for accuracy rather than a better default. It also changes the dtype of
            the returned flux to float64. The precision is fixed for the lifetime of the model,
            because the kernel's floating point type is a compile-time build option.
        interpolate, klims, nk : optional
            Deprecated and ignored: the mean intensity under the planet is always computed for the
            radius ratio being evaluated, so there is no weight table to precompute.
        nz, nzin, nzlimb, zcut : optional
            Deprecated and ignored: the stellar disk is no longer discretised into annuli.
        parallel, small_planet_limit
            Accepted for interface compatibility with the Numba model; unused here.
        cl_ctx, cl_queue : optional
            OpenCL context and command queue. Created with ``cl.create_some_context()`` if
            omitted.
        """
        super().__init__()

        if interpolate is not None or klims is not None or nk is not None:
            warn("The 'interpolate', 'klims' and 'nk' arguments are no longer used and will be removed "
                 "in the future: the mean intensity under the planet is always computed for the radius "
                 "ratio being evaluated.", FutureWarning)
        if nz is not None or nzin is not None or nzlimb is not None or zcut is not None:
            warn("The 'nz', 'nzin', 'nzlimb' and 'zcut' arguments are no longer used and will be removed in "
                 "the future: the stellar disk is no longer discretised into annuli. The quadrature "
                 "resolution is set by 'nq'.", FutureWarning)

        self.ctx = cl_ctx or cl.create_some_context()
        self.queue = cl_queue or cl.CommandQueue(self.ctx)

        # The kernel's floating point type is a build option, so the precision is fixed for the
        # lifetime of the model: changing it would mean rebuilding the program and rebinding the
        # kernels. It also sets the dtype of every host array and of the returned flux.
        self.precision = precision
        self.dtype = _dtype_for_precision(self.ctx, precision)

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

        # Numerical disk integration of a profile without an analytic integral
        self._ldmu = linspace(1, 0, 200)
        self._ldz = sqrt(1 - self._ldmu ** 2)

        # Discretisation
        # --------------
        self.nq: int = nq
        self.ng: int = ng
        self.mu = None            # The mu grid the intensity profile is tabulated on
        self._t0 = 0.0
        self._dt = 0.0
        self._rules = None
        self._b_rules = None      # Quadrature rules on the device
        self._b_cm = None         # Cubic stencil matrices on the device

        self.npv = None
        self.nptb = 0
        self.npb = 0
        self.f = None
        self.pv = array([])

        self.time = None
        self.lcids = None
        self.pbids = None
        self.nsamples = None
        self.exptimes = None

        # Declare the per-population buffers. These are initialised when the model is first
        # evaluated, and reinitialised if the population size changes.
        self._b_ks = None        # Radius ratios per passband
        self._b_ldp = None       # Intensity profiles
        self._b_istar = None     # Disk-integrated intensities
        self._b_ldm = None       # Mean intensity tables
        self._b_gcs = None       # Limb contacts, the table split points
        self._b_n1s = None       # First segment sizes of the tables
        self._b_coef = None      # Split cubic coefficients of the tables
        self._b_valid = None     # Parameter vector validity flags
        self._b_xyc = None       # Taylor series coefficients for the (x, y) position
        self._b_bbs = None       # Transit bounding boxes per (pv, lc)
        self._b_f = None         # Flux buffer
        self._b_p = None         # Parameter vector buffer

        self._b_time = None
        self._time_id = None

        # MeepMeep's device functions are prepended to the model source: `sep_c2` is the twin of
        # the `sep_c` the Numba model uses, so both backends evaluate the same expansion. Its
        # `common.cl` also supplies the fp64 pragma and the shared constants, and its build
        # options use the same `-DREAL=` convention.
        source = read_kernel_source('point2d.cl') + open(join(dirname(__file__), 'rrmodel.cl')).read()
        self.prg = cl.Program(self.ctx, source)
        self.prg.build(options=build_options(precision))

        # Bind the kernels once. Every `Program.__getattr__` builds a new Kernel object and
        # regenerates its invoker, which consults PyOpenCL's on-disk (SQLite) cache, so looking
        # the kernels up per evaluation would dominate the cost of small models.
        self._k_ldm = self.prg.calculate_ldm
        self._k_coefficients = self.prg.calculate_coefficients
        self._k_flux = self.prg.rr_flux
        self._kernel_args_set = False

        self.init_integration(nq, ng)

    def init_integration(self, nq: int, ng: int) -> None:
        """Set the quadrature resolution and the mean intensity table size.

        Called by the initialiser, and useful afterwards for changing the model's accuracy without
        creating a new model. The arguments have the same meaning as in the initialiser.

        Parameters
        ----------
        nq : int
            Number of quadrature nodes per segment.
        ng : int
            Number of grazing parameter nodes in the mean intensity table.
        """
        mf = cl.mem_flags
        self.nq = int(nq)
        self.ng = int(ng)
        self._rules = quadrature_rules(nq)
        self.mu, self._t0, self._dt = profile_grid()
        self.nmu = self.mu.size

        if self._b_rules is not None:
            self._b_rules.release()
            self._b_cm.release()
        self._b_rules = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                  hostbuf=self._rules.astype(self.dtype).ravel())
        self._kernel_args_set = False
        self._b_cm = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=CUBIC_MATRICES.astype(self.dtype).ravel())

        # The table buffers depend on ng, so force their reallocation on the next evaluation.
        self.npv = None

    def init_siwft_arrays(self, nz: int = 40, ng: int = 50):
        """Deprecated: use `init_integration`."""
        warn("'init_siwft_arrays' has been replaced by 'init_integration(nq, ng)' and will be removed in the "
             "future. The stellar disk is no longer discretised into annuli, so 'nz' is ignored.", FutureWarning)
        self.init_integration(self.nq, ng)

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

        self.time = asarray(time, dtype=self.dtype)
        self.lcids = zeros(time.size, 'uint32') if lcids is None else asarray(lcids, dtype='uint32')
        self.pbids = zeros(self.nlc, 'uint32') if pbids is None else asarray(pbids, dtype='uint32')
        # `atleast_1d`, as in `TransitModel.set_data`: a scalar exposure time would otherwise give
        # a zero-dimensional array, which the compiled expansion loop cannot index.
        self.nsamples = (ones(self.nlc, 'uint32') if nsamples is None
                         else atleast_1d(asarray(nsamples, dtype='uint32')))
        # Zero, as in `TransitModel.set_data`, not one: with `nsamples` of 1 the supersampling
        # offset is exactly zero either way, but the exposure time widens the transit bounding
        # box, and a default of one day would stretch it over the far side of a short orbit.
        self.exptimes = (zeros(self.nlc, self.dtype) if exptimes is None
                         else atleast_1d(asarray(exptimes, dtype=self.dtype)))

        self._kernel_args_set = False
        self._b_time = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.time)
        self._b_lcids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lcids)
        self._b_pbids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pbids)
        self._b_nsamples = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.nsamples)
        self._b_etimes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.exptimes)

        # The passband count enters the buffer sizes, so force their reallocation.
        self.npv = None

    def evaluate(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Optional[Union[float, ndarray]] = None,
                 w: Optional[Union[float, ndarray]] = None, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array. A 1D vector is
            read as the radius ratios per passband when evaluating a single parameter vector,
            and as one radius ratio per parameter vector when evaluating a population. Give a
            population several radius ratios per parameter vector as an ``(npv, nk)`` array.
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
        copy : optional
            Copy the fluxes back from the device. With ``copy=False`` the fluxes are left in the
            device buffer ``_b_f`` and ``None`` is returned.

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
        npv = 1 if isscalar(t0) else len(t0)
        k = radius_ratio_array(k, npv)
        nk = k.shape[1]

        if e is None:
            e, w = 0.0, 0.0

        pvp = empty((npv, nk + 6), dtype=self.dtype)
        pvp[:, :nk] = k
        pvp[:, nk] = t0
        pvp[:, nk + 1] = p
        pvp[:, nk + 2] = a
        pvp[:, nk + 3] = i
        pvp[:, nk + 4] = e
        pvp[:, nk + 5] = w

        return self._evaluate_pv(pvp, ldc, copy)

    @deprecated_evaluation_method()
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
            pv = array([[k, t0, p, a, i, e, w]], self.dtype)
        else:
            pv = concatenate([k, [t0, p, a, i, e, w]]).astype(self.dtype)
        return self._evaluate_pv(pv, ldc, copy)

    @deprecated_evaluation_method()
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
        return self._evaluate_pv(pvp, ldc, copy)

    def _allocate(self, npv: int) -> None:
        """(Re)allocate the per-population device buffers for `npv` parameter vectors."""
        mf = cl.mem_flags
        nb = self.dtype().nbytes
        npb, ng = int(self.npb), self.ng

        if self._b_f is not None:
            for name in ('_b_f', '_b_p', '_b_ks', '_b_ldp', '_b_istar', '_b_ldm', '_b_gcs', '_b_n1s',
                         '_b_coef', '_b_valid', '_b_xyc', '_b_bbs'):
                getattr(self, name).release()

        self.npv = uint32(npv)
        self.f = zeros((npv, self.nptb), self.dtype)
        self._kernel_args_set = False
        self._b_f = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.time.nbytes * npv)
        self._b_p = None
        self._b_ks = cl.Buffer(self.ctx, mf.READ_ONLY, npv * npb * nb)
        self._b_ldp = cl.Buffer(self.ctx, mf.READ_ONLY, npv * npb * self.nmu * nb)
        self._b_istar = cl.Buffer(self.ctx, mf.READ_ONLY, npv * npb * nb)
        self._b_ldm = cl.Buffer(self.ctx, mf.READ_WRITE, npv * npb * ng * nb)
        self._b_gcs = cl.Buffer(self.ctx, mf.READ_WRITE, npv * npb * nb)
        self._b_n1s = cl.Buffer(self.ctx, mf.READ_WRITE, npv * npb * int32().nbytes)
        self._b_coef = cl.Buffer(self.ctx, mf.READ_WRITE, npv * npb * (ng - 2) * 4 * nb)
        self._b_valid = cl.Buffer(self.ctx, mf.READ_ONLY, npv * int32().nbytes)
        self._b_xyc = cl.Buffer(self.ctx, mf.READ_ONLY, npv * 10 * nb)
        self._b_bbs = cl.Buffer(self.ctx, mf.READ_ONLY, npv * int(self.nlc) * 2 * nb)

    def _set_kernel_args(self) -> None:
        """Bind the kernel arguments that do not change between evaluations.

        Only the global sizes vary from call to call, so setting the arguments once and
        enqueuing with `cl.enqueue_nd_range_kernel` avoids PyOpenCL re-marshalling every
        argument on each launch. The arguments are invalidated whenever a buffer is
        reallocated, which happens in `init_integration`, `set_data`, `_allocate`, and when
        the shape of the parameter vector array changes.
        """
        self._k_ldm.set_args(self._b_ks, self._b_ldp, self._b_rules,
                             self.dtype(self._t0), self.dtype(self._dt), int32(self.nmu), int32(self.nq),
                             self._b_gcs, self._b_n1s, self._b_ldm)
        self._k_coefficients.set_args(self._b_ldm, self._b_n1s, self._b_cm, int32(self.ng), self._b_coef)
        self._k_flux.set_args(self._b_time, self._b_ks, self._b_istar, self._b_gcs, self._b_n1s,
                              self._b_coef, self._b_valid, self._b_xyc, self._b_bbs, int32(self.ng),
                              self._b_lcids, self._b_pbids, self._b_p, self._b_nsamples, self._b_etimes,
                              self.spv, self.nlc, self.npb, self._b_f)
        self._kernel_args_set = True

    def _evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: bool = True) -> ndarray:
        # Implementation shared with the supported `evaluate` method, so that calling
        # `evaluate` does not raise the deprecation warning.
        mf = cl.mem_flags
        pvp = atleast_2d(asarray(pvp, dtype=self.dtype))
        npv = pvp.shape[0]
        npb = int(self.npb)
        nk = pvp.shape[1] - 6

        if nk != 1 and nk != npb:
            raise ValueError('Radius ratios should be given either as an [npv, 1] or [npv, npb] array.')

        if self.npv != npv:
            self._allocate(npv)

        # The parameter vector buffer depends on the number of radius ratios as well.
        if self._b_p is None or self.pv.shape != pvp.shape:
            if self._b_p is not None:
                self._b_p.release()
            self.pv = zeros(pvp.shape, self.dtype)
            self.spv = uint32(pvp.shape[1])
            self._b_p = cl.Buffer(self.ctx, mf.READ_ONLY, self.pv.nbytes)
            self._kernel_args_set = False

        # Normalise the limb darkening coefficients to a 3D array with a shape (npv, npb, nldc),
        # as in the Numba model.
        ldc = atleast_2d(ldc)
        if ldc.ndim == 2:
            ldc = ldc.reshape((npv, npb, -1))

        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((npv, npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(npb):
                        istar[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        ldp = ldp.reshape((npv, npb, self.nmu))
        istar = istar.reshape((npv, npb))

        # Radius ratios per passband, and the parameter vector validity as in the Numba model
        ks = empty((npv, npb), self.dtype)
        ks[:, :] = pvp[:, :nk]
        a, e = pvp[:, nk + 2], pvp[:, nk + 4]
        valid = ~(isnan(a) | (a <= 1.0) | (e < 0.0) | isnan(ldp[:, 0, 0])) & ((ks > 0.0) & (ks <= 1.0)).all(1)

        # Taylor series expansion of the sky position and the transit bounding box, computed as
        # `model_full.rr_precompute` computes them so that both backends evaluate the same orbit.
        # The solvers and the expansion point placement stay on the host by MeepMeep's contract;
        # the device only evaluates the polynomial.
        orb = [ascontiguousarray(pvp[:, nk + j], float64) for j in range(1, 6)]
        xyc, bbs = _expansion_arrays(valid, *orb, ascontiguousarray(ks[:, 0], float64),
                                     asarray(self.exptimes, dtype=float64))

        cl.enqueue_copy(self.queue, self._b_ks, ks)
        cl.enqueue_copy(self.queue, self._b_ldp, ldp.astype(self.dtype))
        cl.enqueue_copy(self.queue, self._b_istar, istar.astype(self.dtype))
        cl.enqueue_copy(self.queue, self._b_valid, valid.astype(int32))
        cl.enqueue_copy(self.queue, self._b_xyc, ascontiguousarray(xyc, self.dtype).ravel())
        cl.enqueue_copy(self.queue, self._b_bbs, ascontiguousarray(bbs, self.dtype).ravel())

        self.pv[:] = pvp
        cl.enqueue_copy(self.queue, self._b_p, self.pv)

        if not self._kernel_args_set:
            self._set_kernel_args()

        # Tabulate the mean intensity under the planet and fit the cubics
        cl.enqueue_nd_range_kernel(self.queue, self._k_ldm, (npv, npb, self.ng), None)
        cl.enqueue_nd_range_kernel(self.queue, self._k_coefficients, (npv, npb, self.ng - 2), None)

        # Evaluate the model
        cl.enqueue_nd_range_kernel(self.queue, self._k_flux, (npv, self.nptb), None)

        if copy:
            cl.enqueue_copy(self.queue, self.f, self._b_f)
            return squeeze(self.f)
        else:
            return None

    def tables(self) -> Tuple[ndarray, ndarray, ndarray]:
        """Read the mean intensity tables of the last evaluation back from the device.

        Returns
        -------
        gcs : ndarray
            Limb contacts (the table split points) with a shape ``(npv, npb)``.
        n1s : ndarray
            Number of nodes in the first segment of each table, shape ``(npv, npb)``.
        coef : ndarray
            Split cubic coefficients of the tables, shape ``(npv, npb, ng - 2, 4)``.
        """
        if self.npv is None:
            raise ValueError('The model has not been evaluated yet.')
        npv, npb = int(self.npv), int(self.npb)
        gcs = empty((npv, npb), self.dtype)
        n1s = empty((npv, npb), int32)
        coef = empty((npv, npb, self.ng - 2, 4), self.dtype)
        cl.enqueue_copy(self.queue, gcs, self._b_gcs)
        cl.enqueue_copy(self.queue, n1s, self._b_n1s)
        cl.enqueue_copy(self.queue, coef, self._b_coef)
        self.queue.finish()
        return gcs, n1s, coef
