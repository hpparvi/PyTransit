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
from typing import Union, List, Optional

from numpy import ndarray, atleast_1d

from ..transitmodel import TransitModel
from .model_ecspec import esmodel

__all__ = ['EclipseSpectroscopyModel']


class EclipseSpectroscopyModel(TransitModel):

    def __init__(self, parallel: bool = False):
        self.parallel = parallel
        self.model = njit(parallel=parallel, fastmath=False)(esmodel)
        super().__init__()

    def evaluate(self, f: ndarray, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = 0.0,
                 w: Union[float, ndarray] = 0.0) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        f
            Flux ratios either as a 1D vector or 2D array
        k
            Radius ratio either as a single float or a 1D vector.
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
            Modelled flux either as a 3D ndarray.
        """
        k, t0, p, a, i, e, w = map(atleast_1d, (k, t0, p, a, i, e, w))
        return self.model(self.time, k, t0, p, a, i, e, w, f, self.nsamples[0], self.exptimes[0])

    def __call__(self, f: ndarray, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = 0.0,
                 w: Union[float, ndarray] = 0.0) -> ndarray:
        return self.evaluate(f, k, t0, p, a, i, e, w)
