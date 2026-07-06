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
from typing import Tuple, Callable, Union, List, Optional

from numpy import ndarray, linspace, isscalar, unique, atleast_1d
from scipy.integrate import trapezoid

from ..ldmodel import LDModel
from ..numba.ldmodels import *

from .opmodel_full import opmodel
from .rrmodel import RoadRunnerModel

__all__ = ['OblatePlanetModel']


class OblatePlanetModel(RoadRunnerModel):
    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 precompute_weights: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 100, nlines: int = 100,
                 nthreads: int = 1, small_planet_limit: float = 0.05, exact_areas: bool = False,
                 exact_ld: bool = False, nannuli: int = 20, **kwargs):
        """
        Parameters
        ----------
        nlines
            Number of scanlines used by the θ-sampled scanline planet-star intersection area model.
        exact_areas
            Calculate the planet-star intersection areas using the exact analytic algorithm instead of
            the θ-sampled scanline approximation. The exact algorithm is accurate to machine precision
            but several times slower to evaluate. Can be overridden per call in `evaluate`.
        exact_ld
            Integrate the limb darkening over the planet's exact elliptical footprint (a Stieltjes sum
            over `nannuli` stellar annuli with the annulus areas from the ellipse-disk intersection
            routine) instead of using the circular-footprint mean intensity approximation. Reduces the
            model error from tens of ppm to below the ppm level for strongly oblate planets at roughly
            25 times the computational cost. When combined with `exact_areas`, the annulus areas are
            calculated with the exact analytic algorithm (slower, but free of the scanline resolution
            floor). Can be overridden per call in `evaluate`.
        nannuli
            Number of stellar annuli used by the exact-footprint limb darkening integration.
        """
        super().__init__(ldmodel, precompute_weights, klims, nk, nzin, nzlimb, zcut, ng, nthreads, small_planet_limit, **kwargs)
        self.nlines = nlines
        self.exact_areas = exact_areas
        self.exact_ld = exact_ld
        self.nannuli = nannuli

    def evaluate(self, k: Union[float, ndarray], f: Union[float, ndarray], alpha: Union[float, ndarray],
                 ldc: Union[ndarray, List],
                 t0: Union[float, ndarray], p: Union[float, ndarray], a: Union[float, ndarray],
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0, w: Union[float, ndarray] = 0.0,
                 copy: bool = True, exact_areas: Optional[bool] = None, exact_ld: Optional[bool] = None) -> ndarray:
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
        exact_areas : optional
            Calculate the planet-star intersection areas using the exact analytic algorithm instead of
            the θ-sampled scanline approximation. If None (default), uses the value given in the model
            initializer.
        exact_ld : optional
            Integrate the limb darkening over the planet's exact elliptical footprint instead of using
            the circular-footprint mean intensity approximation. If None (default), uses the value
            given in the model initializer.

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

        npv = 1 if isscalar(p) else p.size
        ldc = atleast_1d(ldc)

        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((npv, self.npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(self.npb):
                        istar[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        exact_areas = self.exact_areas if exact_areas is None else exact_areas
        exact_ld = self.exact_ld if exact_ld is None else exact_ld
        flux = opmodel(self.time, k, f, alpha, t0, p, a, i, e, w, self.parallel,
                       self.nlc, self.npb, self.nep, self.nlines,
                       self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes,
                       ldp, istar, self.weights, self.dk, self.klims[0], self.klims[1], self.dg, self.ze,
                       self.mu, exact_areas, exact_ld, self.nannuli)

        return flux
