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
from typing import Union, Optional, List

from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros, asarray

from .numba.ma_quadratic_nb import quadratic_model_interpolated_pv, calculate_interpolation_tables, \
    quadratic_model_interpolated_v, quadratic_model_interpolated_s, quadratic_model_direct_v, quadratic_model_direct_s, \
    quadratic_model_direct_pv
from .transitmodel import TransitModel

__all__ = ['QuadraticModel']

class QuadraticModel(TransitModel):
    """Transit model with quadratic limb darkening (Mandel & Agol, ApJ 580, L171-L175 2002).
    """

    def __init__(self, interpolate: bool = True, klims: tuple = (0.005, 0.5), nk: int = 256, nz: int = 256):
        """Transit model with quadratic limb darkening (Mandel & Agol, ApJ 580, L171-L175 2002).

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

        # Interpolation tables for the model components
        # ---------------------------------------------
        if interpolate:
            self.ed, self.le, self.ld, self.kt, self.zt = calculate_interpolation_tables(klims[0], klims[1], nk, nz)
            self.klims = klims
            self.nk = nk
            self.nz = nz
        else:
            self.ed, self.le, self.ld, self.kt, self.zt = None, None, None, None, None
            self.klims, self.nk, self.nz = None, None, None

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List], t0: Union[float, ndarray], p: Union[float, ndarray],
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
        if isinstance(t0, float):
            if e is None:
                e, w = 0.0, 0.0
            return self.evaluate_ps(k, ldc, t0, p, a, i, e, w, copy)

        # Parameter population branch
        # ---------------------------
        else:
            ldc = atleast_2d(ldc)
            ldc, k, t0, p, a, i = asarray(ldc), asarray(k), asarray(t0), asarray(p), asarray(a), asarray(i)
            npv = t0.size
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

        ldc = asarray(ldc)
        k = asarray(k)

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

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a 2D parameter array.

        Parameters
        ----------
        pvp: ndarray
            Parameter array with a shape `(npv, npar)` where `npv` is the number of parameter vectors, and each row
            contains a set of parameters `[k, t0, p, a, i, e, w]`. The radius ratios can also be given per passband,
            in which case the row should be structured as `[k_0, k_1, k_2, ..., k_npb, t0, p, a, i, e, w]`.
        ldc: ndarray
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

    def to_opencl(self):
        """Creates an OpenCL clone (`QuadraticModelCL`) of the transit model.

        Returns
        -------
        QuadraticModelCL
        """
        from .ma_quadratic_cl import QuadraticModelCL
        tm = QuadraticModelCL(klims=self.klims)
        tm.set_data(self.time, self.lcids, self.pbids, self.nsamples, self.exptimes)
        return tm