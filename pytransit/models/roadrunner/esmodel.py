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
from numba import njit

from ..transitmodel import TransitModel
from .model_ecspec import esmodel

__all__ = ['EclipseSpectroscopyModel']


class EclipseSpectroscopyModel(TransitModel):
    """Secondary eclipse model specialised for spectroscopic time series.

    The eclipse counterpart of :class:`TransmissionSpectroscopyModel`. It models the
    occultation of the planet by the star for many wavelength bins that share a single event,
    computing the eclipse geometry once and reusing it across all bins.

    The quantity of interest in eclipse spectroscopy is the wavelength-dependent planet-star
    flux ratio, so it is a *parameter* here rather than something folded into the depth: the
    per-bin flux ratios are passed as the first argument `f` of `evaluate`.

    The model also applies the light travel time correction between the transit and the
    secondary eclipse, which shifts the eclipse by roughly ``2 a R_star / c`` (about 40 s for a
    hot Jupiter). The correction needs a physical stellar radius, given through the `rstar`
    argument of `evaluate` in solar radii.

    Parameters
    ----------
    parallel : bool, optional
        Compile the model with Numba's parallel backend. Worth enabling for large numbers of
        wavelength bins.

    Examples
    --------
    ::

        from pytransit import ESModel

        em = ESModel()
        em.set_data(time)
        flux = em.evaluate(f=fr_per_bin, k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi, rstar=1.2)
    """

    def __init__(self, parallel: bool = False):
        self.parallel = parallel
        self.model = njit(parallel=parallel, fastmath=False)(esmodel)
        super().__init__()

    def evaluate(self, f: ndarray, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = 0.0,
                 w: Union[float, ndarray] = 0.0, rstar: Union[float, ndarray] = 1.0) -> ndarray:
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
        rstar : optional
            Stellar radius in solar radii, used to compute the light travel time
            correction between transit and secondary eclipse. Defaults to 1.0.

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
        k, t0, p, a, i, e, w, rstar = map(atleast_1d, (k, t0, p, a, i, e, w, rstar))
        if rstar.size == 1 and k.size > 1:
            rstar = rstar.repeat(k.size)
        return self.model(self.time, k, t0, p, a, i, e, w, rstar, f, self.nsamples[0], self.exptimes[0])

    def __call__(self, f: ndarray, k: Union[float, ndarray], t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = 0.0,
                 w: Union[float, ndarray] = 0.0, rstar: Union[float, ndarray] = 1.0) -> ndarray:
        return self.evaluate(f, k, t0, p, a, i, e, w, rstar)
