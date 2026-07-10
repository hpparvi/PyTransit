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

from numpy import ndarray, linspace, isscalar, unique, atleast_1d, atleast_2d, zeros, pi
from scipy.integrate import trapezoid

from ..ldmodel import LDModel
from ..limb_darkening import evaluate_ld, evaluate_ldi

from .opmodel_full import opmodel
from .rrmodel import RoadRunnerModel

__all__ = ['OblatePlanetModel']


class OblatePlanetModel(RoadRunnerModel):
    """Transit model for an oblate planet.

    A RoadRunner-type transit model (Parviainen, 2020) for a planet whose sky projection is an
    ellipse with a semi-major axis ``k`` (in units of the stellar radius), flattening ``f``, and
    projected obliquity ``alpha``, rather than a circle. Planetary oblateness imprints signals of
    tens of ppm on the transit light curve (Seager & Hui, 2002; Barnes & Fortney, 2003), so the
    model offers several accuracy levels that can be chosen to match the precision needs:

    1. The default model calculates the planet-star intersection areas with a θ-sampled scanline
       method and approximates the limb darkening blocked by the planet with the mean intensity
       over the footprint of an area-equivalent circular planet. The model error is below the
       ppm level for a spherical planet but can reach tens of ppm for strongly oblate planets
       in grazing geometries.
    2. ``exact_areas`` replaces the scanline intersection areas with an analytic ellipse-circle
       intersection area algorithm that is accurate to machine precision. This removes the
       geometric discretization error but keeps the mean-intensity limb darkening approximation.
    3. ``exact_ld`` integrates the limb darkening over the planet's exact elliptical footprint,
       which reduces the model error below the ppm level for all tested geometries at roughly
       25 times the computational cost of the default model. Combined with ``exact_areas``, the
       integration is also free of the scanline resolution floor.

    The accuracy levels are validated against direct numerical integration of the limb-darkened
    stellar disk in the test suite (``tests/test_opmodel.py``).

    The model follows the standard PyTransit API: initialize, call ``set_data`` to set the
    observation times (and optionally the light curve, passband, and epoch indices), and call
    ``evaluate`` to calculate the model fluxes. Any radially symmetric limb darkening model is
    supported.
    """

    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 precompute_weights: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 100, nlines: int = 100,
                 nthreads: int = 1, small_planet_limit: float = 0.05, exact_areas: bool = False,
                 exact_ld: bool = False, nannuli: int = 20, **kwargs):
        """Initialize the oblate planet transit model.

        Parameters
        ----------
        ldmodel
            Limb darkening model: either the name of a built-in model ('uniform', 'linear',
            'quadratic', 'quadratic-tri', 'nonlinear', 'general', 'square_root', 'logarithmic',
            'exponential', 'power-2', or 'power-2-pm'), a callable returning the stellar
            intensity profile as a function of µ, a tuple of callables returning the intensity
            profile and its integral over the stellar disk, or an ``LDModel`` instance.
        precompute_weights
            Precompute a 3D limb darkening weight table for the radius ratio range set by
            `klims`. Speeds up repeated evaluations at the cost of initialization time.
        klims
            Radius ratio limits (kmin, kmax) for the precomputed weight table.
        nk
            Radius ratio grid size for the precomputed weight table.
        nzin
            Number of limb darkening profile discretization nodes covering the inner stellar disk
            (normalized distances from 0 to `zcut`). Together with `nzlimb`, this sets the
            resolution of the tabulated intensity profile used by all the model versions,
            including the profile interpolation in the exact-footprint limb darkening mode.
        nzlimb
            Number of limb darkening profile discretization nodes covering the stellar limb
            (normalized distances from `zcut` to 1), spaced uniformly in µ.
        zcut
            Normalized distance that separates the stellar disk into an inner disk and the limb.
        ng
            Size of the grazing value table used by the mean limb darkening interpolation.
        nlines
            Number of scanlines used by the θ-sampled scanline planet-star intersection area
            calculation. The scanline discretization error decreases as ``nlines**-2`` in
            ordinary transit geometries and as ``nlines**-1.5`` in grazing geometries.
        nthreads
            Number of threads to use for the model computation. Values above one enable the
            parallel model version and set the numba thread count. The numba thread count is
            process-global, so the model created last defines the thread count for all models,
            and the value cannot exceed numba's launch-time maximum (NUMBA_NUM_THREADS).
        small_planet_limit
            The radius ratio limit below which to use a small planet approximation.
        exact_areas
            Calculate the planet-star intersection areas using the exact analytic algorithm
            instead of the θ-sampled scanline approximation. The exact algorithm is accurate to
            machine precision but several times slower to evaluate. Can be overridden per call
            in `evaluate`.
        exact_ld
            Integrate the limb darkening over the planet's exact elliptical footprint (a
            Stieltjes sum over `nannuli` stellar annuli with the annulus areas from the
            ellipse-disk intersection routine) instead of using the circular-footprint mean
            intensity approximation. Reduces the model error from tens of ppm to below the ppm
            level for strongly oblate planets at roughly 25 times the computational cost. When
            combined with `exact_areas`, the annulus areas are calculated with the exact
            analytic algorithm (slower, but free of the scanline resolution floor). Can be
            overridden per call in `evaluate`.
        nannuli
            Number of stellar annuli used by the exact-footprint limb darkening integration.
            The annulus discretization error decreases as ``nannuli**-2``.
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
            Radius ratio(s) either as a single float, 1D vector, or 2D array. The radius ratio
            gives the planet's projected semi-major axis in units of the stellar radius. A 2D
            array should have a shape [npv, 1] or [npv, npb], where npv is the number of
            parameter vectors and npb the number of passbands, and allows for passband-dependent
            radius ratios.
        f
            Flattening(s) of the planet's projection as a float or a 1D vector. The flattening,
            f = (a - b) / a, where a and b are the projected semi-major and semi-minor axes,
            ranges from 0 (spherical planet) to values below 1.
        alpha
            Projected obliquity (obliquities) of the planet in radians as a float or a 1D
            vector. The angle is measured from the x-axis of the sky plane (the direction of
            orbital motion at mid-transit) to the projected semi-major axis of the planet.
        ldc
            Limb darkening coefficients as a 1D or 2D array. For multiple parameter vectors with
            multiple passbands, the coefficients should be given as a 3D array with a shape
            [npv, npb, ncoef].
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
        copy : optional
            Kept for compatibility with the PyTransit transit model API; currently unused.
        exact_areas : optional
            Calculate the planet-star intersection areas using the exact analytic algorithm instead of
            the θ-sampled scanline approximation. If None (default), uses the value given in the model
            initializer.
        exact_ld : optional
            Integrate the limb darkening over the planet's exact elliptical footprint instead of using
            the circular-footprint mean intensity approximation. If None (default), uses the value
            given in the model initializer.

        Returns
        -------
        ndarray
            Modelled flux either as a 1D or 2D ndarray with a shape [npt] or [npv, npt], where
            npt is the number of time samples given in `set_data`.

        Raises
        ------
        ValueError
            If the radius ratios are given as a 2D array with a shape other than [npv, 1] or
            [npv, npb].

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters
        simultaneously. In the first case, the orbital parameters should all be given as floats. In
        the second case, the orbital parameters should be given as a 1D array-like, and the fluxes
        are calculated for all the parameter vectors in one call (parallelized over the numba
        threads if the model was initialized with nthreads > 1).

        The `exact_areas` and `exact_ld` arguments select the model accuracy level per call
        without touching the model defaults: for example, an MCMC run can sample with the fast
        default model and the posterior can be validated by re-evaluating the samples with
        ``exact_ld=True``.
        """

        npv = 1 if isscalar(p) else p.size
        ldc = atleast_2d(ldc)
        if ldc.ndim == 2:
            # Normalize the limb darkening coefficients to a 3D array with a shape
            # (npv, npb, nldc). A 2D array is interpreted either as (npb, nldc) for
            # a single parameter vector or as (npv, npb*nldc) when npv > 1.
            ldc = ldc.reshape((npv, self.npb, -1))

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
