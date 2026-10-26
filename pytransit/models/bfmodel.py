#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
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

"""A brute-force transit model by direct numerical integration, for testing the other models."""

from math import asin, cos, sin, sqrt, floor
from typing import Callable, Optional, Union, Literal

from numpy import ndarray, array, asarray, atleast_1d, atleast_2d, zeros, ones, arange, pi, linspace
from scipy.integrate import quad
from scipy.optimize import brentq

from meepmeep.backends.numba.newton.newton import xyz_newton_v
from meepmeep.backends.numba.point2d import solve2d, pos_c

from .transitmodel import TransitModel

__all__ = ['BFModel']


def _chord(y: float, cx: float, cy: float, a: float, f: float, angle: float):
    """The x-range an ellipse covers on the horizontal line at height y, or None if it misses it.

    The ellipse is centred at (cx, cy), has a semi-major axis `a` at `angle` from the x-axis, and
    a flattening `f`, so that its semi-minor axis is a(1 - f).
    """
    b = a * (1.0 - f)
    c, s = cos(angle), sin(angle)
    v = y - cy
    qa = c * c / (a * a) + s * s / (b * b)
    qb = 2.0 * v * c * s * (1.0 / (a * a) - 1.0 / (b * b))
    qc = v * v * (s * s / (a * a) + c * c / (b * b)) - 1.0
    d = qb * qb - 4.0 * qa * qc
    if d <= 0.0:
        return None
    sd = sqrt(d)
    return cx + (-qb - sd) / (2.0 * qa), cx + (-qb + sd) / (2.0 * qa)


def _half_height(a: float, f: float, angle: float) -> float:
    """The half-extent along y of an ellipse with a semi-major axis `a` at `angle` and flattening `f`."""
    b = a * (1.0 - f)
    return sqrt((a * sin(angle)) ** 2 + (b * cos(angle)) ** 2)


def _crossings(px: float, py: float, k: float, f: float, alpha: float, fstar: float, astar: float) -> list:
    """The y-coordinates where the planet's outline crosses the stellar limb.

    The scanline integrand has a corner at each of them, where the ends of the covered interval
    switch from following one outline to following the other, so the outer integration is split
    there. The crossings are found by scanning the planet's outline for sign changes of the star's
    implicit equation and refining each with Brent's method.
    """
    ca, sa = cos(alpha), sin(alpha)
    cs, ss = cos(astar), sin(astar)
    b = k * (1.0 - f)

    def point(t):
        u, v = k * cos(t), b * sin(t)
        return px + ca * u - sa * v, py + sa * u + ca * v

    def inside(t):
        x, y = point(t)
        xs, ys = cs * x + ss * y, (-ss * x + cs * y) / (1.0 - fstar)
        return xs * xs + ys * ys - 1.0

    ts = linspace(0.0, 2.0 * pi, 1441)
    gs = [inside(t) for t in ts]
    ys = []
    for j in range(ts.size - 1):
        if (gs[j] > 0.0) != (gs[j + 1] > 0.0):
            ys.append(point(brentq(inside, ts[j], ts[j + 1], xtol=1e-15))[1])
    return ys


class BFModel(TransitModel):
    """Brute-force transit model that integrates the stellar brightness numerically.

    The model integrates the stellar surface brightness over the exact region the planet covers
    with nested adaptive quadrature, so it makes no approximation beyond the requested integration
    tolerance. It is slow, by design: its purpose is to serve as a reference for testing the other
    transit models, not to be fitted to data.

    Both the planet and the star can be oblate. The planet is an ellipse with the semi-major axis
    `k`, flattening `f`, and orientation `alpha`, as in
    :class:`~pytransit.models.roadrunner.opmodel.OblatePlanetModel`. The stellar disk is an ellipse
    with an equatorial radius of one, flattening `fstar`, and orientation `astar`. Both angles are
    measured from the sky-plane x-axis, the direction of the planet's motion at mid-transit, to the
    ellipse's semi-major axis.

    The stellar brightness is given either as a radial limb darkening profile or as a function of
    the sky-plane coordinates. A radial profile ``f(mu, pv)`` takes the cosine of the viewing angle,
    with ``mu = sqrt(1 - r**2)`` where `r` is the elliptical radius, the distance from the stellar
    centre in units of the stellar disk's radius in that direction. A brightness function
    ``g(x, y, pv)`` takes the sky-plane coordinates in units of the stellar equatorial radius, and
    can describe any brightness distribution, such as a gravity-darkened star.

    The planet's position comes from an exact solution of Kepler's equation by default, so the
    model can also be used to test the orbit approximations of the other models. With
    ``orbit='taylor'`` it uses the same Taylor series expansion of the orbit as RoadRunner and the
    models derived from it, which isolates the error of the flux integration.
    """

    def __init__(self, profile: Union[str, Callable] = 'quadratic', brightness: Optional[Callable] = None,
                 orbit: Literal['exact', 'taylor'] = 'exact', epsabs: float = 1e-12, epsrel: float = 1e-10) -> None:
        """
        Parameters
        ----------
        profile
            The radial limb darkening profile: the name of a built-in law (see
            ``RoadRunnerModel.ldmodels``) or a callable ``f(mu, pv)``. Ignored if `brightness` is
            given.
        brightness
            A callable ``g(x, y, pv)`` returning the stellar surface brightness at the sky-plane
            coordinates (x, y), in units of the stellar equatorial radius. Replaces `profile`.
        orbit
            ``'exact'`` to solve Kepler's equation for the planet's position, or ``'taylor'`` to use
            the Taylor series expansion of the orbit around mid-transit that RoadRunner uses.
        epsabs, epsrel
            The absolute and relative tolerances of the numerical integration.
        """
        super().__init__()
        from .roadrunner.rrmodel import RoadRunnerModel

        if brightness is not None:
            self._profile = None
        elif isinstance(profile, str):
            if profile not in RoadRunnerModel.ldmodels:
                raise ValueError(f"Unknown limb darkening law '{profile}'. Use one of "
                                 f"{', '.join(RoadRunnerModel.ldmodels)}, or a callable.")
            self._profile = RoadRunnerModel.ldmodels[profile][0]
        elif callable(profile):
            self._profile = profile
        else:
            raise ValueError("The profile must be the name of a limb darkening law or a callable.")

        if orbit not in ('exact', 'taylor'):
            raise ValueError("The orbit must be either 'exact' or 'taylor'.")

        self._brightness = brightness
        self.orbit = orbit
        self.epsabs = epsabs
        self.epsrel = epsrel

    def _profile_at(self, mu: float, pv: ndarray) -> float:
        # The profiles are only guaranteed to accept arrays: some built-in laws reject a scalar mu.
        return self._profile(array([mu]), pv)[0]

    def _scanline_integrand(self, phi: float, y: float, m: float, w: float, sqa: float, pv: ndarray) -> float:
        """The stellar brightness on a scanline at x = m + w sin(phi), times the Jacobian w cos(phi).

        On a scanline, 1 - r**2 is a quadratic in x that vanishes where the scanline meets the stellar
        limb, at m - w and m + w. With x = m + w sin(phi) it becomes exactly (sqa w cos(phi))**2, so mu
        follows without the cancellation of sqrt(1 - r**2) near the limb, and the Jacobian cancels an
        inverse square root singularity of the profile there, such as that of an optically thin shell.
        """
        cphi = cos(phi)
        if self._brightness is not None:
            return self._brightness(m + w * sin(phi), y, pv) * w * cphi
        return self._profile_at(min(sqa * w * cphi, 1.0), pv) * w * cphi

    def _total_flux(self, pv: ndarray, fstar: float, astar: float) -> float:
        """The flux of the whole stellar disk."""
        if self._brightness is None:
            # Substituting r = sin(t) removes an inverse square root singularity at the limb, such as
            # that of an optically thin shell, and the ellipse's area scales the circle's by (1 - f).
            integral = quad(lambda t: self._profile_at(cos(t), pv) * sin(t) * cos(t), 0.0, 0.5 * pi,
                            epsabs=self.epsabs, epsrel=self.epsrel, limit=200)[0]
            return 2.0 * pi * (1.0 - fstar) * integral
        else:
            return self._integrate(None, pv, fstar, astar)

    def _integrate(self, planet: Optional[tuple], pv: ndarray, fstar: float, astar: float) -> float:
        """Integrate the brightness over the stellar disk, or over its overlap with the planet.

        The region is cut into horizontal scanlines. On each one the planet and the star cover an
        interval whose ends follow analytically from the two ellipses, and the brightness is
        integrated over their intersection.
        """
        ca, sa = cos(astar), sin(astar)
        hs = _half_height(1.0, fstar, astar)
        if planet is None:
            ylo, yhi = -hs, hs
        else:
            px, py, k, f, alpha = planet
            hp = _half_height(k, f, alpha)
            ylo, yhi = max(py - hp, -hs), min(py + hp, hs)
            if yhi <= ylo:
                return 0.0

        # The coefficient of x**2 in the squared elliptical radius, r**2.
        sqa = sqrt(ca * ca + (sa / (1.0 - fstar)) ** 2)

        def scanline(y):
            cs = _chord(y, 0.0, 0.0, 1.0, fstar, astar)
            if cs is None:
                return 0.0
            lo, hi = cs
            m, w = 0.5 * (lo + hi), 0.5 * (hi - lo)
            if planet is not None:
                cp = _chord(y, px, py, k, f, alpha)
                if cp is None:
                    return 0.0
                lo, hi = max(lo, cp[0]), min(hi, cp[1])
            if hi <= lo:
                return 0.0
            plo = asin(min(max((lo - m) / w, -1.0), 1.0))
            phi = asin(min(max((hi - m) / w, -1.0), 1.0))
            return quad(self._scanline_integrand, plo, phi, args=(y, m, w, sqa, pv),
                        epsabs=self.epsabs, epsrel=self.epsrel, limit=200)[0]

        points = None
        if planet is not None:
            points = [y for y in _crossings(px, py, k, f, alpha, fstar, astar) if ylo < y < yhi] or None
        return quad(scanline, ylo, yhi, epsabs=self.epsabs, epsrel=self.epsrel, limit=500, points=points)[0]

    def _flux(self, x: float, y: float, k: float, pv: ndarray, f: float, alpha: float, fstar: float,
              astar: float, total: float) -> float:
        # The star lies inside the unit circle and the planet inside a circle of radius k.
        if sqrt(x * x + y * y) >= 1.0 + k:
            return 1.0
        return 1.0 - self._integrate((x, y, k, f, alpha), pv, fstar, astar) / total

    def evaluate_xy(self, x: Union[float, ndarray], y: Union[float, ndarray], k: float, ldc: Union[ndarray, list],
                    f: float = 0.0, alpha: float = 0.0, fstar: float = 0.0, astar: float = 0.0):
        """Evaluate the flux for the planet centred at given sky-plane positions.

        Useful for testing the flux integration of another model without involving its orbit.

        Parameters
        ----------
        x, y
            The planet's sky-plane position(s) in units of the stellar equatorial radius, as floats
            or arrays.
        k
            The planet's radius ratio, its projected semi-major axis for an oblate planet.
        ldc
            The parameters given to the limb darkening profile or the brightness function.
        f, alpha
            The planet's flattening and orientation [rad].
        fstar, astar
            The stellar disk's flattening and orientation [rad].

        Returns
        -------
        float or ndarray
            The flux, normalised to one outside the transit.
        """
        pv = asarray(ldc, float)
        total = self._total_flux(pv, fstar, astar)
        if isinstance(x, ndarray) or isinstance(y, ndarray):
            xs, ys = (atleast_1d(asarray(v, float)) for v in (x, y))
            xs, ys = xs + 0.0 * ys, ys + 0.0 * xs
            return asarray([self._flux(xx, yy, k, pv, f, alpha, fstar, astar, total) for xx, yy in zip(xs, ys)])
        return self._flux(float(x), float(y), k, pv, f, alpha, fstar, astar, total)

    def _positions(self, times: ndarray, t0: float, p: float, a: float, i: float, e: float, w: float):
        """The planet's sky-plane positions, and whether it is in front of the star."""
        if self.orbit == 'exact':
            x, y, z = xyz_newton_v(times, t0, p, a, i, e, w)
            return x, y, z > 0.0
        c = solve2d(0.0, p, a, i, e, w)
        x, y = zeros(times.size), zeros(times.size)
        for j, t in enumerate(times):
            tc = t - (t0 + floor((t - t0 + 0.5 * p) / p) * p)
            x[j], y[j] = pos_c(tc, c)
        return x, y, ones(times.size, bool)

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, list], t0: float, p: float, a: float,
                 i: float, e: float = 0.0, w: float = 0.0, f: float = 0.0, alpha: float = 0.0,
                 fstar: float = 0.0, astar: float = 0.0, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a single set of parameters.

        Parameters
        ----------
        k
            Radius ratio as a float, or one per passband as a 1D array. For an oblate planet, the
            projected semi-major axis in units of the stellar equatorial radius.
        ldc
            The parameters of the limb darkening profile or the brightness function, as a 1D array
            for a single passband or a 2D array with one row per passband.
        t0, p, a, i, e, w
            Transit centre, period, scaled semi-major axis, inclination [rad], eccentricity, and
            argument of periastron [rad].
        f, alpha
            The planet's flattening and orientation [rad].
        fstar, astar
            The stellar disk's flattening and orientation [rad].
        copy
            Kept for compatibility with the transit model API; currently unused.

        Returns
        -------
        ndarray
            The flux, normalised to one outside the transit, as a 1D array.
        """
        if self.time is None:
            raise ValueError("Need to set the data before evaluating the model.")
        k = atleast_1d(asarray(k, float))
        ldc = atleast_2d(asarray(ldc, float))
        if k.size not in (1, self.npb):
            raise ValueError(f"Give either one radius ratio or one per passband ({self.npb}).")
        if ldc.shape[0] not in (1, self.npb):
            raise ValueError(f"Give the limb darkening parameters either once or once per passband ({self.npb}).")

        totals = [self._total_flux(ldc[min(ipb, ldc.shape[0] - 1)], fstar, astar) for ipb in range(self.npb)]

        flux = zeros(self.npt)
        for j in range(self.npt):
            ilc = self.lcids[j]
            ipb = self.pbids[ilc]
            _k = k[min(ipb, k.size - 1)]
            pv = ldc[min(ipb, ldc.shape[0] - 1)]
            ns = self.nsamples[ilc]
            times = self.time[j] + self.exptimes[ilc] * ((arange(1, ns + 1) - 0.5) / ns - 0.5)
            xs, ys, front = self._positions(times, t0, p, a, i, e, w)
            for x, y, infront in zip(xs, ys, front):
                flux[j] += self._flux(x, y, _k, pv, f, alpha, fstar, astar, totals[ipb]) if infront else 1.0
            flux[j] /= ns
        return flux
