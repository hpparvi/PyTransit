# -*- coding: utf-8 -*-
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

"""QPower2 transit model.

This module offers the QPower2 class implementing the transit model by Maxted & Gill (A&A, 622, A33, 2019).
"""
from typing import Union, Optional

from numpy import squeeze, ndarray, array, asarray, zeros

from .numba.qpower2_nb import qpower2_model_pv, qpower2_model_s, qpower2_model_v
from .transitmodel import TransitModel

__all__ = ['QPower2Model']

class QPower2Model(TransitModel):
    """QPower2 transit model by Maxted & Gill (A&A, 622, A33, 2019).
    """

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

        k = asarray(k)

        # Scalar parameters branch
        # ------------------------
        if isinstance(t0, float):
            if e is None:
                e, w = 0., 0.

            flux = qpower2_model_s(self.time, k, ldc, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes)
        # Parameter population branch
        # ---------------------------
        else:
            npv = t0.size
            if e is None:
                e, w = zeros(npv), zeros(npv)

            if k.ndim == 1:
                k = k.reshape((k.size,1))

            flux = qpower2_model_v(self.time, k, ldc, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes)
        return squeeze(flux)

    def evaluate_ps(self, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
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
        assert self.time is not None, "Need to set the data before calling the transit model."
        k = asarray(k)
        flux = qpower2_model_s(self.time, k, ldc, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes)
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray) -> ndarray:
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
        assert self.time is not None, "Need to set the data before calling the transit model."
        flux = qpower2_model_pv(self.time, pvp, ldc, self.lcids, self.pbids, self.nsamples, self.exptimes)
        return squeeze(flux)

