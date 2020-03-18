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

from numpy import ndarray, array, squeeze, zeros, atleast_2d, asarray

from .transitmodel import TransitModel
from .numba.ma_uniform_nb import uniform_model_v, uniform_model_s, uniform_model_pv

__all__ = ['UniformModel']

class UniformModel(TransitModel):

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

            flux = uniform_model_s(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples,
                                   self.exptimes, self._es, self._ms, self._tae)
        # Parameter population branch
        # ---------------------------
        else:
            npv = t0.size
            if e is None:
                e, w = zeros(npv), zeros(npv)

            flux = uniform_model_v(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples,
                                   self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_ps(self, k: float, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
        assert self.time is not None, "Need to set the data before calling the transit model."
        pvp = array([[k, t0, p, a, i, e, w]])
        flux = uniform_model_pv(self.time, pvp, self.lcids, self.nsamples, self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)

    def evaluate_pv(self, pvp: ndarray) -> ndarray:
        assert self.time is not None, "Need to set the data before calling the transit model."
        flux = uniform_model_pv(self.time, pvp, self.lcids, self.nsamples, self.exptimes, self._es, self._ms, self._tae)
        return squeeze(flux)
