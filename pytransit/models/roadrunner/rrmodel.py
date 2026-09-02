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
from typing import Tuple, Callable, Union, List, Optional
from warnings import warn

from numba import config as numba_config, get_num_threads, njit, set_num_threads
from numpy import ndarray, linspace, isscalar, unique, atleast_1d, squeeze, atleast_2d, sqrt, zeros, pi
from scipy.integrate import trapezoid

from ..ldmodel import LDModel
from ..limb_darkening import (ld_uniform, ldi_uniform, ld_linear, ldi_linear, ld_quadratic, ldi_quadratic,
                              ld_quadratic_tri, ldi_quadratic_tri, ld_nonlinear, ldi_nonlinear, ld_general, ldi_general,
                              ld_square_root, ldi_square_root, ld_logarithmic, ldi_logarithmic,
                              ld_exponential, ldi_exponential, ld_power_2, ldi_power_2, ld_power_2_pm, ldi_power_2_pm,
                              evaluate_ld, evaluate_ldi)
from ..transitmodel import TransitModel

from .common import create_z_grid, calculate_weights_3d
from .model_full import rr_full
from .model_simple import rr_simple

__all__ = ['RoadRunnerModel']


class RoadRunnerModel(TransitModel):
    """The RoadRunner transit model (Parviainen, MNRAS 499, 1633, 2020).

    RoadRunner is PyTransit's recommended general-purpose transit model. Unlike the classical
    models, which are analytic solutions derived for one specific limb darkening law,
    RoadRunner separates the *geometry* of the transit from the *stellar intensity profile*.
    The planet-star overlap geometry is solved numerically once and tabulated, and the limb
    darkening enters only as a profile sampled on a fixed grid of normalized distances from the
    disk center. This has two consequences:

    - **Any radially symmetric limb darkening model works.** Besides the eleven built-in
      profiles listed in `ldmodels`, the model accepts a plain Python callable, a pair of
      callables giving the profile and its disk integral, or an
      :class:`~pytransit.models.ldmodel.LDModel` instance backed by a stellar atmosphere
      grid. Switching from a quadratic law to a numerically tabulated one costs nothing in
      accuracy or, largely, in speed.
    - **The evaluation cost is nearly independent of the limb darkening law.** A four-parameter
      non-linear law is about as fast as a linear one.

    Accuracy is set by the discretization parameters `nzin`, `nzlimb`, `zcut`, and `ng`. The
    defaults give sub-ppm accuracy for typical transit geometries.

    Attributes
    ----------
    ldmodels : dict
        The built-in limb darkening models, keyed by name: ``'uniform'``, ``'linear'``,
        ``'quadratic'``, ``'quadratic-tri'``, ``'nonlinear'``, ``'general'``, ``'square_root'``,
        ``'logarithmic'``, ``'exponential'``, ``'power-2'``, and ``'power-2-pm'``. Each value is
        a ``(profile, disk_integral)`` pair of Numba-compiled functions.

    Examples
    --------
    ::

        from pytransit import RoadRunnerModel

        tm = RoadRunnerModel('power-2')
        tm.set_data(time)
        flux = tm.evaluate(k=0.1, ldc=[0.6, 0.5], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

    See Also
    --------
    TransmissionSpectroscopyModel : RoadRunner specialised for spectroscopic time series.
    OblatePlanetModel : RoadRunner for planets with an elliptical sky projection.
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
                 precompute_weights: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 100,
                 nthreads: int = 1, small_planet_limit: float = 0.01, **kwargs):
        """The RoadRunner transit model by Parviainen (2020).

        Parameters
        ----------
        precompute_weights : bool, optional
            Precompute a 3D weight table for radius ratio values set by `klims`.
        klims : tuple, optional
            Radius ratio limits (kmin, kmax) for the precomputed weight table.
        nk : int, optional
            Radius ratio grid size for the precomputed weight table.
        nzin : int, optional
            Normalized distance grid size for the inner disk.
        nzlimb : int, optional
            Normalized distance grid size for the limb.
        zcut: float, optional
            Normalized distance that separates the stellar disk into an inner disk and limb.
        ng : int, optional
            Size of the grazing value table.
        nthreads: int, optional
            Number of threads to use for the model computation. Values above one enable the
            parallel model version and set the numba thread count. Note that the numba thread
            count is process-global, so the model created last defines the thread count for all
            models, and the value cannot exceed numba's launch-time maximum (NUMBA_NUM_THREADS).
        small_planet_limit: float, optional
            The radius ratio limit at and below which the single-light-curve model uses a small
            planet approximation: the mean stellar intensity blocked by the planet is
            approximated by the intensity at the planet's center, and the limb darkening
            weighting is skipped. The approximation error grows roughly quadratically with the
            radius ratio (below 1 ppm at the default limit of 0.01, but ~100 ppm at k = 0.05),
            so raise the limit only if speed matters more than ppm-level accuracy. Set to None
            or 0.0 to disable.
        """
        super().__init__()

        if 'interpolate' in kwargs:
            warn("The 'interpolate' argument has been replaced by 'precompute_weights' and will be removed in the future.", FutureWarning)
            self.interpolate: bool = kwargs.get('interpolate', False)
        else:
            self.interpolate: bool = precompute_weights

        if 'parallel' in kwargs:
            warn("The 'parallel' argument has been replaced by 'nthreads' and will be removed in the future.", FutureWarning)
            self.nthreads: int = get_num_threads()
            self.parallel: bool = True
        else:
            self.nthreads: int = min(nthreads, numba_config.NUMBA_NUM_THREADS)
            self.parallel = self.nthreads > 1
            if self.parallel:
                set_num_threads(self.nthreads)

        self.splimit: float | None = small_planet_limit

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

        # Set the basic variable
        # ----------------------
        self.klims = klims
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut

        # Declare the basic arrays
        # ------------------------
        self.ze = None
        self.zm = None
        self.mu = None
        self.dk = None
        self.dg = None
        self.weights = None

        self._ldmu = linspace(1, 0, 200)
        self._ldz = sqrt(1 - self._ldmu ** 2)

        self.init_integration(nzin, nzlimb, zcut, ng, nk)

    def set_data(self, time: Union[ndarray, List],
                 lcids: Optional[Union[ndarray, List]] = None,
                 pbids: Optional[Union[ndarray, List]] = None,
                 nsamples: Optional[Union[ndarray, List]] = None,
                 exptimes: Optional[Union[ndarray, List]] = None,
                 epids: Optional[Union[ndarray, List]] = None) -> None:
        super().set_data(time, lcids, pbids, nsamples, exptimes, epids)
        self.nep = unique(self.epids).size

    def init_integration(self, nzin, nzlimb, zcut, ng, nk):
        """Rebuild the stellar disk discretisation and the limb darkening weight tables.

        Called by the initialiser, and useful afterwards for changing the model's accuracy without
        creating a new model. The arguments have the same meaning as in the initialiser.

        Parameters
        ----------
        nzin : int
            Number of discretisation nodes covering the inner stellar disk.
        nzlimb : int
            Number of discretisation nodes covering the stellar limb.
        zcut : float
            Normalised distance separating the inner disk from the limb.
        ng : int
            Size of the grazing value table.
        nk : int
            Radius ratio grid size for the precomputed weight table.
        """
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut
        self.ze, self.zm = create_z_grid(zcut, nzin, nzlimb)
        self.mu = sqrt(1 - self.zm ** 2)
        self.dk, self.dg, self.weights = calculate_weights_3d(nk, self.klims[0], self.klims[1], self.ze, ng)

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List],
                 t0: Union[float, ndarray], p: Union[float, ndarray], a: Union[float, ndarray],
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0, w: Union[float, ndarray] = 0.0,
                 copy: bool = True) -> ndarray:
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

        npv = 1 if isscalar(p) else p.size
        ldc = atleast_2d(ldc)
        if ldc.ndim == 2:
            # Normalize the limb darkening coefficients to a 3D array with a shape
            # (npv, npb, nldc). A 2D array is interpreted either as (npb, nldc) for
            # a single parameter vector or as (npv, npb*nldc) when npv > 1.
            ldc = ldc.reshape((npv, self.npb, -1))

        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((npv, self.npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(self.npb):
                        istar[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        k, t0, p, a, i, e, w = (atleast_2d(k), atleast_2d(t0), atleast_1d(p), atleast_1d(a),
                                atleast_1d(i), atleast_1d(e), atleast_1d(w))

        if self.nlc > 1 or k.shape[0] > 1:
            return squeeze(rr_full(self.time, k, t0, p, a, i, e, w, self.parallel, self.nlc, self.npb, self.nep,
                                   self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes,
                                   ldp, istar, self.weights, self.dk, self.klims[0], self.klims[1], self.dg, self.ze))
        else:
            splimit = self.splimit if self.splimit is not None else 0.0
            return rr_simple(self.time, k[0, 0], t0[0, 0], p[0], a[0], i[0], e[0], w[0], self.parallel, splimit,
                             self.nsamples[0], self.exptimes[0],
                             ldp[0, 0, :], istar[0, 0], self.weights, self.dk, self.klims[0], self.klims[1], self.dg,
                             self.ze, self.zm)
