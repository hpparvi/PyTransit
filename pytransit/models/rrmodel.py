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
import timeit
from typing import Tuple, Callable, Union, List, Optional

from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros, asarray, linspace, sqrt, pi, ones, log, exp, \
    tile, full
from scipy.integrate import trapz

from .ldmodel import LDModel
from .numba.ldmodels import *
from .numba.rrmodel import rrmodel_direct_s, rrmodel_interpolated_s, \
    rrmodel_direct_s_simple, rrmodel_direct_v, \
    rrmodel_interpolated_v

from .numba.rrmodel import create_z_grid, calculate_weights_3d
from .transitmodel import TransitModel

__all__ = ['RoadRunnerModel']


class RoadRunnerModel(TransitModel):
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
                 small_planet_limit: float = 0.05):
        """The RoadRunner transit model by Parviainen (2020).

        Parameters
        ----------
        interpolate : bool, optional
            Use the interpolation method presented in Parviainen (2015) if true.
        klims : tuple, optional
            Radius ratio limits (kmin, kmax) for the interpolated model.
        nk : int, optional
            Radius ratio grid size for the interpolated model.
        nz : int, optional
            Normalized distance grid size for the interpolated model.
        """
        super().__init__()
        self.interpolate = interpolate
        self.parallel = parallel
        self.is_simple = False
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

        self._m_direct_s = None
        self._m_direct_v = None
        self._m_interp_s = None
        self._m_interp_v = None

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
        self.set_methods()

    def set_methods(self):
        if self.npb == 1 and all(self.nsamples == 1) and all(self.epids == 0):
            self.is_simple = True
        else:
            self.is_simple = False

    def init_integration(self, nzin, nzlimb, zcut, ng, nk=None):
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut
        self.ze, self.zm = create_z_grid(zcut, nzin, nzlimb)
        self.mu = sqrt(1 - self.zm ** 2)
        if self.interpolate:
            self.dk, self.dg, self.weights = calculate_weights_3d(nk, self.klims[0], self.klims[1], self.ze, ng)


    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List], t0: Union[float, ndarray],
                 p: Union[float, ndarray],
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

        # Scalar parameters branch
        # ------------------------
        if isinstance(p, float):
            if e is None:
                e, w = 0.0, 0.0
            return self.evaluate_ps(k, ldc, t0, p, a, i, e, w, copy)

        # Parameter population branch
        # ---------------------------
        else:
            k, t0, p, a, i = asarray(k), asarray(t0), asarray(p), asarray(a), asarray(i)

            if k.ndim == 1:
                k = k.reshape((k.size, 1))

            if t0.ndim == 1:
                t0 = t0.reshape((t0.size, 1))

            npv = p.size
            if e is None:
                e, w = zeros(npv), zeros(npv)

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
                            istar[ipv, ipb] = 2 * pi * trapz(self._ldz * ldpi[ipv, ipb], self._ldz)

            if self.interpolate:
                flux = rrmodel_interpolated_v(self.time, k, t0, p, a, i, e, w, ldp, istar, self.weights, self.dk,
                                       self.klims[0], self.dg, self.lcids, self.pbids, self.epids, self.nsamples,
                                       self.exptimes, self.npb, self.parallel)
            else:
                flux = rrmodel_direct_v(self.time, k, t0, p, a, i, e, w, ldp, istar, self.ze, self.ng,
                                       self.lcids, self.pbids, self.epids, self.nsamples,
                                       self.exptimes, self.npb, self.parallel)
        return squeeze(flux)

    def evaluate_ps(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: float, a: float, i: float,
                    e: float = 0.0, w: float = 0.0, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar parameters.

        Parameters
        ----------
        k : array-like
            Radius ratio(s) either as a single float or an 1D array.
        ldc : array-like
            Limb darkening coefficients as a 1D array.
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

        k = asarray(k)
        ldc = asarray(ldc)
        t0 = asarray(t0)

        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)
            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((1,self.npb))
                if ldc.ndim == 1:
                    ldc = ldc.reshape((1, 1, -1))
                elif ldc.ndim == 2:
                    ldc = ldc.reshape((1, ldc.shape[1], -1))

                for ipb in range(self.npb):
                    istar[0,ipb] = 2 * pi * trapz(self._ldz * self.ldmodel(self._ldmu, ldc[0,ipb]), self._ldz)

        if self.interpolate:
            flux = rrmodel_interpolated_s(self.time, k, t0, p, a, i, e, w, ldp, istar, self.weights, self.zm,
                                    self.dk, self.klims[0], self.dg, self.splimit,
                                    self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes, self.parallel)
        else:
            flux = rrmodel_direct_s(self.time, k, t0, p, a, i, e, w, ldp, istar, self.ze, self.zm, self.ng, self.splimit,
                                    self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes, self.parallel)

        return squeeze(flux)
