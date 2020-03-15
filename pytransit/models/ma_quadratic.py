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
from typing import Union

from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros, asarray

from .numba.ma_quadratic_nb import quadratic_model_interpolated, calculate_interpolation_tables, \
    quadratic_model_interpolated_v, quadratic_model_interpolated_s, quadratic_model_direct_v, quadratic_model_direct_s
from .transitmodel import TransitModel

__all__ = ['QuadraticModel']

class QuadraticModel(TransitModel):
    """Mandel-Agol transit model with quadratic limb darkening (ApJ 580, L171-L175 2002).
    """

    def __init__(self, method: str = 'pars', is_secondary: bool = False,
                 interpolate: bool = True, klims: tuple = (0.01, 0.25), nk: int = 256, nz: int = 256):
        """Initialise the model.

        Args:
            is_secondary: If True, evaluates the model for eclipses. If false, eclipses are filtered out (default = False).
            interpolate: If True, evaluates the model using interpolation (default = True).
            klims: Minimum and maximum radius ratio if interpolation is used as (kmin,kmax).
            nk: Interpolation table resolution in k.
            nz: Interpolation table resolution in z.
        """
        super().__init__(method, is_secondary)
        self.interpolate = interpolate

        # Interpolation tables for the model components
        # ---------------------------------------------
        if interpolate:
            self.ed, self.le, self.ld, self.kt, self.zt = calculate_interpolation_tables(klims[0], klims[1], nk, nz)
            self.klims = klims
            self.nk = nk
            self.nz = nz

    def evaluate_ps(self, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0., copy: bool = True) -> ndarray:
        assert self.time is not None, "Need to set the data before calling the transit model."
        ldc = atleast_2d(ldc)
        assert ldc.shape[1] == 2*self.npb, "The quadratic model needs two limb darkening coefficients per passband"
        pvp = array([[k, t0, p, a, i, e, w]])
        if self.interpolate:
            flux = quadratic_model_interpolated(self.time, pvp, ldc, self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                self.kt, self.zt)
        else:
            raise NotImplementedError
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: bool = True) -> ndarray:
        assert self.time is not None, "Need to set the data before calling the transit model."
        ldc = atleast_2d(ldc)
        assert ldc.shape[1] == 2*self.npb, "The quadratic model needs two limb darkening coefficients per passband"
        if self.interpolate:
            flux = quadratic_model_interpolated(self.time, pvp, ldc, self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                self.kt, self.zt)
        else:
            raise NotImplementedError
        return squeeze(flux)

    def evaluate(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = None,
                 w: Union[float, ndarray] = None,
                 copy: bool = True) -> ndarray:
        """Evaluates the transit model for a set of scalar or vector parameters.

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
        e
            Orbital eccentricity as a float or a 1D vector.
        w
            Argument of periastron as a float or a 1D vector.
        copy

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously.
        The orbital parameters can be given either as a float or a 1D array-like (preferably ndarray for optimal speed.)

        Returns
        -------

        """

        ldc = atleast_2d(ldc)
        npv = ldc.shape[0]

        # Scalar parameters branch
        # ------------------------
        if isinstance(t0, float):
            k = asarray(k)
            if e is None:
                e, w = 0., 0.

            if not (isinstance(p, float) and isinstance(a, float) and isinstance(i, float)
                    and isinstance(e, float) and isinstance(w, float)):
                raise ValueError("All the orbital parameters need to be scalar if `t0` is scalar.")

            if self.interpolate:
                flux = quadratic_model_interpolated_s(self.time, k, t0, p, a, i, e, w, ldc,
                                                      self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                      self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                      self.kt, self.zt)
            else:
                flux = quadratic_model_direct_s(self.time, k, t0, p, a, i, e, w, ldc,
                                                self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                self._es, self._ms, self._tae)

        # Parameter population branch
        # ---------------------------
        else:
            if e is None:
                e, w = zeros(npv), zeros(npv)
            if self.interpolate:
                flux = quadratic_model_interpolated_v(self.time, k, t0, p, a, i, e, w, ldc,
                                                      self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                      self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                      self.kt, self.zt)
            else:
                flux = quadratic_model_direct_v(self.time, k, t0, p, a, i, e, w, ldc,
                                                self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                self._es, self._ms, self._tae)
        return squeeze(flux)
