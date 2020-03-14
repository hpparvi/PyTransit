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

from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros

from .numba.ma_quadratic_nb import quadratic_model_interpolated, calculate_interpolation_tables, \
    quadratic_model_interpolated_v
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
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = None, w: Union[float, ndarray] = None,
                 copy: bool = True) -> ndarray:
        t0, p, a, i = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i)
        k, ldc = atleast_2d(k), atleast_2d(ldc)

        if e is None:
            e = zeros(p.size)
            w = zeros(p.size)
        else:
            e, w = atleast_1d(e), atleast_1d(w)

        if self.interpolate:
            flux = quadratic_model_interpolated_v(self.time, k, t0, p, a, i, e, w, ldc,
                                                  self.lcids, self.pbids, self.nsamples, self.exptimes,
                                                  self._es, self._ms, self._tae, self.ed, self.ld, self.le,
                                                  self.kt, self.zt)
        else:
            raise NotImplementedError
            # flux = quadratic_model_direct_v(self.time, k, t0, p, a, i, e, w, ldc,
            #                                    self.lcids, self.pbids, self.nsamples, self.exptimes,
            #                               self._es, self._ms, self._tae)
        return squeeze(flux)