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

"""Planet properties from radial velocity observations."""

import astropy.units as u

from astropy.units import Quantity
from astropy.constants import G
from numpy import pi, sin, sqrt


def mp_from_kiepms(k, i, e, p, mstar):
    """Planet mass from the radial velocity semi-amplitude.

    Calculates the planet mass as ``mp = K (P / 2 pi G)**(1/3) mstar**(2/3) sqrt(1 - e**2) / sin(i)``.

    Parameters
    ----------
    k : float, ndarray, or Quantity
        Radial velocity semi-amplitude [m/s].
    i : float, ndarray, or Quantity
        Orbital inclination [rad].
    e : float or ndarray
        Eccentricity.
    p : float, ndarray, or Quantity
        Orbital period [d].
    mstar : float, ndarray, or Quantity
        Stellar mass [M_Sun].

    Returns
    -------
    float or ndarray
        Planet mass [M_Jup].

    Notes
    -----
    The relation assumes that the planet's mass is negligible compared to the star's, which
    underestimates the planet mass by a fraction of about ``(2/3) mp / mstar``.
    """
    k = k.to(u.m/u.s) if isinstance(k, Quantity) else k * u.m/u.s
    i = i.to(u.rad) if isinstance(i, Quantity) else i*u.rad
    p = p.to(u.s) if isinstance(p, Quantity) else (p*u.day).to(u.s)
    mstar = mstar.to(u.kg) if isinstance(mstar, Quantity) else (mstar*u.M_sun).to(u.kg)
    return (k * (p/(2*pi*G)) ** (1/3) * mstar ** (2 / 3) / sin(i) * sqrt((1 - e ** 2))).to(u.M_jup).value


def surface_gravity(k, i, e, p, ar):
    """Planet surface gravity from the radial velocity semi-amplitude and transit parameters.

    Calculates the planet's surface gravity as ``g = (2 pi / P) sqrt(1 - e**2) K / ((Rp/a)**2 sin(i))``
    following Southworth et al. (2007). The relation needs no knowledge of the stellar mass.

    Parameters
    ----------
    k : float, ndarray, or Quantity
        Radial velocity semi-amplitude [m/s].
    i : float, ndarray, or Quantity
        Orbital inclination [rad].
    e : float or ndarray
        Eccentricity.
    p : float, ndarray, or Quantity
        Orbital period [d].
    ar : float or ndarray
        The squared ratio of the planet radius and the semi-major axis, ``(Rp/a)**2``. With the
        radius ratio `k` and the scaled semi-major axis `a` of the transit model, this is
        ``(k / a)**2``.

    Returns
    -------
    float or ndarray
        Surface gravity [m/s^2].

    References
    ----------
    Southworth, J., Wheatley, P. J. & Sams, G. A method for the direct determination of the surface
    gravities of transiting extrasolar planets. MNRAS 379, L11 (2007).
    """
    k = k.to(u.m / u.s) if isinstance(k, Quantity) else k * u.m / u.s
    i = i.to(u.rad) if isinstance(i, Quantity) else i * u.rad
    p = p.to(u.s) if isinstance(p, Quantity) else (p * u.day).to(u.s)
    return (2 * pi / p * sqrt(1 - e ** 2) * k / (ar * sin(i))).value
