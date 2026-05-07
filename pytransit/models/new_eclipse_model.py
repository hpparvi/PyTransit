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

from dask.array import atleast_1d
from numpy import squeeze, zeros
from numpy.typing import NDArray
from .roadrunner.model_eclipse import eclipse_model
from .transitmodel import TransitModel

__all__ = ['EclipseModel']

npfloat = Union[float, NDArray]


class EclipseModel(TransitModel):

    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, k: npfloat, t0: npfloat, p: npfloat, a: npfloat, i: npfloat, e: npfloat = None, w: npfloat = None,
                 rstar: npfloat = 1.0, copy: bool = True) -> NDArray:
        """Evaluates a multiplicative secondary eclipse model for a set of scalar or vector parameters.

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
        rstar
            Stellar radius in R_sun as a float or a 1D vector.

        Returns
        -------
        Multiplicative eclipse model
        """

        k = atleast_1d(k)
        t0 = atleast_1d(t0)
        p = atleast_1d(p)
        a = atleast_1d(a)
        i = atleast_1d(i)
        rstar = atleast_1d(rstar)
        e = zeros(1) if e is None else atleast_1d(e)
        w = zeros(1) if w is None else atleast_1d(w)

        return squeeze(eclipse_model(self.time, k, t0, p, a, i, e, w, rstar, self.lcids, self.pbids, self.nsamples, self.exptimes))
