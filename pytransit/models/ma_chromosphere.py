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
from typing import Union

from numpy import ndarray, array, squeeze, asarray, zeros

from .numba.ma_chromosphere_nb import chromosphere_model_pv, chromosphere_model_v, chromosphere_model_s
from .transitmodel import TransitModel

__all__ = ['ChromosphereModel']

class ChromosphereModel(TransitModel):

    def evaluate(self, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = None, w: Union[float, ndarray] = None,
                 copy: bool = True) -> ndarray:
        """Evaluates the uniform transit model for a set of scalar or vector parameters.

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
        Transit model
        """

        k = asarray(k)

        # Scalar parameters branch
        # ------------------------
        if isinstance(t0, float):
            if e is None:
                e, w = 0., 0.

            flux = chromosphere_model_s(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples,
                                   self.exptimes, self._es, self._ms, self._tae)
        # Parameter population branch
        # ---------------------------
        else:
            npv = t0.size
            if e is None:
                e, w = zeros(npv), zeros(npv)

            if k.ndim == 1:
                k = k.reshape((k.size,1))

            flux = chromosphere_model_v(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples,
                                   self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_ps(self, k: float, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
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
        if self.time is None:
            raise ValueError("Need to set the data before calling the transit model.")

        k = asarray(k)
        flux = chromosphere_model_s(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes,
                               self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray) -> ndarray:
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
        flux = chromosphere_model_pv(self.time, pvp, self.lcids, self.pbids, self.nsamples, self.exptimes, self._es,
                                self._ms, self._tae)
        return squeeze(flux)
