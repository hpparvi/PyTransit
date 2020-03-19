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
from typing import Union, Optional

from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros, asarray

from .numba.ma_quadratic_nb import quadratic_model_interpolated_pv, calculate_interpolation_tables, \
    quadratic_model_interpolated_v, quadratic_model_interpolated_s, quadratic_model_direct_v, quadratic_model_direct_s, \
    quadratic_model_direct_pv
from .transitmodel import TransitModel

__all__ = ['QuadraticModel']

class QuadraticModel(TransitModel):
    """Transit model with quadratic limb darkening (ApJ 580, L171-L175 2002).
    """

    def __init__(self, method: str = 'pars', is_secondary: bool = False,
                 interpolate: bool = True, klims: tuple = (0.005, 0.5), nk: int = 256, nz: int = 256):
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

    def evaluate(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Optional[Union[float, ndarray]] = None,
                 w: Optional[Union[float, ndarray]] = None, copy: Optional[bool] = True) -> ndarray:
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
        e
            Orbital eccentricity as a float or a 1D vector.
        w
            Argument of periastron as a float or a 1D vector.

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously.
        The orbital parameters can be given either as a float or a 1D array-like (preferably ndarray for optimal speed.)

        Returns
        -------
        Transit model
        """

        # Scalar parameters branch
        # ------------------------
        if isinstance(t0, float):
            return self.evaluate_ps(k, ldc, t0, p, a, i, e, w, copy)

        # Parameter population branch
        # ---------------------------
        else:
            ldc = atleast_2d(ldc)
            npv = ldc.shape[0]
            if e is None:
                e, w = zeros(npv), zeros(npv)
            if self.interpolate:
                flux = quadratic_model_interpolated_v(self.time, k, t0, p, a, i, e, w, ldc,
                                                      self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb,
                                                      self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                      self.kt, self.zt)
            else:
                flux = quadratic_model_direct_v(self.time, k, t0, p, a, i, e, w, ldc,
                                                self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb,
                                                self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_ps(self, k: Union[float, ndarray], ldc: ndarray, t0: float, p: float, a: float, i: float,
                    e: Optional[float] = None, w: Optional[float] = None, copy: Optional[bool] = True) -> ndarray:
        """Evaluate the transit model for a set of scalar parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float or an 1D array.
        ldc
            Limb darkening coefficients as a 1D array.
        t0
            Transit center as a float.
        p
            Orbital period as a float.
        a
            Orbital semi-major axis divided by the stellar radius as a float.
        i
            Orbital inclination(s) as a float.
        e
            Orbital eccentricity as a float.
        w
            Argument of periastron as a float.

        Notes
        -----
        This version of the `evaluate` method is optimized for calculating a single transit model (such as when using a
        local optimizer). If you want to evaluate the model for a large number of parameters simultaneously, use either
        `evaluate` or `evaluate_pv`.

        Returns
        -------
        Model flux
        """

        ldc = asarray(ldc)
        k = asarray(k)
        if e is None:
            e, w = 0., 0.

        if self.time is None:
            raise ValueError("Need to set the data before calling the transit model.")
        if ldc.size != 2 * self.npb:
            raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

        if self.interpolate:
            flux = quadratic_model_interpolated_s(self.time, k, t0, p, a, i, e, w, ldc,
                                                  self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb,
                                                  self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                  self.kt, self.zt)
        else:
            flux = quadratic_model_direct_s(self.time, k, t0, p, a, i, e, w, ldc,
                                            self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb,
                                            self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: Optional[bool] = True) -> ndarray:
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

        """

        ldc = asarray(ldc)
        pvp = asarray(pvp)

        if self.time is None:
            raise ValueError("Need to set the data before calling the transit model.")

        if self.interpolate:
            flux = quadratic_model_interpolated_pv(self.time, pvp, ldc, self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                   self.npb, self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                   self.kt, self.zt)
        else:
            flux = quadratic_model_direct_pv(self.time, pvp, ldc, self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                   self.npb, self._es, self._ms, self._tae)
        return squeeze(flux)
