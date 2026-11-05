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

"""Physical quantities for modelling phase curves and secondary eclipses.

The functions estimate the planet's equilibrium temperature, its reflected and thermal flux, and
the amplitudes of the Doppler beaming and ellipsoidal variation signals of the host star.

Several functions return a planet-star flux ratio *per projected area element*, that is, the
ratio of the planet's and the star's surface brightnesses. The planet-star flux ratio follows by
multiplying it by the planet-star area ratio, ``k**2``.
"""

from typing import Union

import astropy.units as u
from numba import njit
from numpy import ndarray, pi, atleast_1d, zeros, exp, diff, log, sqrt, nan, vstack, tile, linspace, cos, sin, sum, \
    isfinite
from scipy.constants import c,h,k,G
from scipy.optimize import brentq

from ..stars import create_husser2013_interpolator, create_bt_settl_interpolator

NPType = Union[float,ndarray]

mj2kg = u.M_jup.to(u.kg)
ms2kg = u.M_sun.to(u.kg)
d2s = u.day.to(u.s)

def equilibrium_temperature(tstar: NPType, a: NPType, f: NPType, ab: NPType) -> NPType:
    """Planetary equilibrium temperature.

    Calculates the equilibrium temperature of a planet as ``Teq = tstar sqrt(1/a) (f (1 - ab))**(1/4)``,
    following López-Morales & Seager (2007).

    Parameters
    ----------
    tstar
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    f
        Heat redistribution factor: 1/4 for a uniform redistribution over the whole planet, 1/2 for
        a uniform redistribution over the dayside, and 2/3 for no redistribution.
    ab
        Bond albedo.

    Returns
    -------
    float or ndarray
        Equilibrium temperature [K].
    """
    return tstar * sqrt(1 / a) * (f * (1 - ab)) ** 0.25


# Thermal emission
# ================

def planck(t: NPType, l: NPType) -> NPType:
    """Spectral radiance of a black body per unit wavelength.

    Parameters
    ----------
    t
        Black body temperature [K].
    l
        Wavelength [m].

    Returns
    -------
    float or ndarray
        Spectral radiance [W m^-3 sr^-1].
    """
    return 2 * h * c ** 2 / l ** 5 / (exp(h * c / (l * k * t)) - 1)

@njit
def summed_planck(teff, wl, tm):
    """Black body spectral radiance summed over a wavelength grid with weights.

    Parameters
    ----------
    teff : float or ndarray
        Black body temperature(s) [K].
    wl : ndarray
        Wavelength grid [m].
    tm : ndarray
        Weight of each wavelength, such as the passband transmission.

    Returns
    -------
    ndarray
        The weighted sum of the spectral radiance for each temperature.
    """
    teff = atleast_1d(teff)
    flux = zeros(teff.shape[0])
    for i in range(flux.size):
        flux[i] = sum(tm*(2*h*c**2 / wl**5 / (exp(h*c / (wl*k*teff[i])) - 1.)))
    return flux


def emission(tp: NPType, tstar: NPType, k: NPType, flt) -> NPType:
    """Passband-integrated planet-star flux ratio from thermal emission.

    Approximates both the planet and the star as black bodies, and integrates their spectra over
    the passband weighted by the number of photons, as appropriate for a photon-counting detector.

    Parameters
    ----------
    tp
        Temperature of the planet [K].
    tstar
        Effective temperature of the star [K].
    k
        Planet-star radius ratio.
    flt : Filter
        Passband transmission as a :class:`~pytransit.contamination.filter.Filter`.

    Returns
    -------
    float or ndarray
        Planet-star flux ratio.
    """
    wl_nm = linspace(*flt.bbox, 100)
    wl = 1e-9 * wl_nm
    tm = flt(wl_nm) * wl
    tp, tstar = atleast_1d(tp).astype('d'), atleast_1d(tstar).astype('d')
    return k**2 * (summed_planck(tp, wl, tm) / summed_planck(tstar, wl, tm))


# Doppler boosting
# ================

def doppler_beaming_factor(teff: float, flt, dataset: str = 'BT-Settl') -> float:
    """Photon-weighted passband-integrated Doppler beaming factor.

    Calculates the Doppler beaming factor ``B = 5 + d ln F / d ln l`` from a model spectrum, and
    averages it over the passband weighted by the number of photons [Bloemen2011]_ [Claret2020]_.

    Parameters
    ----------
    teff
        Effective temperature of the star [K].
    flt : Filter
        Passband transmission as a :class:`~pytransit.contamination.filter.Filter`.
    dataset
        Stellar spectrum library, either ``'BT-Settl'`` or ``'Husser2013'``.

    Returns
    -------
    float
        Photon-weighted passband-integrated Doppler beaming factor.

    References
    ----------
    .. [Bloemen2011] Bloemen, S. et al. Kepler observations of the beaming binary KPD 1946+4340. MNRAS 410, 1787
       (2011).
    .. [Claret2020] Claret, A. et al. Doppler beaming factors for white dwarfs, main sequence stars, and giant stars.
       A&A 641, A157 (2020).
    """
    if dataset == 'Husser2013':
        ip = create_husser2013_interpolator()
    else:
        ip = create_bt_settl_interpolator()

    wl_nm = ip.grid[1][1:-1]
    wl = 1e-9 * wl_nm
    fl = ip(vstack((tile(teff, wl_nm.size), wl_nm)).T)
    b = 5. + diff(log(fl)) / diff(log(wl))
    w1 = flt(wl_nm)[1:] * wl[1:] * fl[1:]
    w2 = flt(wl_nm)[:-1] * wl[:-1] * fl[:-1]
    w = 0.5 * (w1 + w2)
    m = isfinite(b)
    return sum(w[m] * b[m]) / sum(w[m])


def doppler_beaming_amplitude(mp: NPType, ms: NPType, period: NPType, b: NPType) -> NPType:
    """Amplitude of the Doppler beaming signal.

    Calculates the relative amplitude of the Doppler beaming signal, ``A = B K / c``, following
    Loeb & Gaudi [Loeb2003]_. The photon-weighted passband-integrated Doppler beaming factor `B`
    (Bloemen et al. 2011, [Barclay2012]_) needs to be calculated beforehand for the star and the
    instrument using :func:`doppler_beaming_factor`.

    Parameters
    ----------
    mp
        Planet mass [M_Jup].
    ms
        Stellar mass [M_Sun].
    period
        Orbital period [d].
    b
        Photon-weighted passband-integrated Doppler beaming factor.

    Returns
    -------
    float or ndarray
        Relative amplitude of the Doppler beaming signal.

    Notes
    -----
    The radial velocity semi-amplitude `K` is calculated for a circular edge-on orbit assuming
    that the planet's mass is negligible compared to the star's. Multiply the amplitude by
    ``sin(i)`` for an inclined orbit.

    References
    ----------
    .. [Loeb2003] Loeb, A. & Gaudi, B. S. Periodic Flux Variability of Stars due to the Reflex Doppler Effect Induced by
       Planetary Companions. ApJ 588, L117 (2003).
    .. [Barclay2012] Barclay, T. et al. Photometrically derived masses and radii of the planet and star in the TrES-2
       system. ApJ 761, 53 (2012).
    """
    return b / c *(2*pi*G/(d2s*period))**(1/3) * ((mp*mj2kg)/(ms*ms2kg)**(2/3))

# Ellipsoidal variations
# ======================

def ellipsoidal_variation(f: NPType, theta: NPType, mp: float, ms: float, a: float, i: float, e: float, u: float, g: float):
    """Ellipsoidal variation signal. Not implemented yet.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError
    #return ellipsoidal_variation_amplitude(mp, ms, a, i, u, g)

def ellipsoidal_variation_signal(f: NPType, theta: NPType, e: float) -> NPType:
    """Shape of the ellipsoidal variation signal.

    Calculates the ellipsoidal variation signal normalised to unit amplitude at the distance of
    the semi-major axis. The amplitude scales with the inverse cube of the star-planet distance,
    which gives the eccentricity-dependent factor.

    Parameters
    ----------
    f
        True anomaly [rad].
    theta
        Angle between the line of sight and the star-planet direction [rad].
    e
        Eccentricity.

    Returns
    -------
    float or ndarray
        Normalised ellipsoidal variation signal. Multiply it by
        :func:`ellipsoidal_variation_amplitude` to get the relative flux variation.
    """
    return -((1 + e * cos(f)) / (1-e**2))**3 * cos(2*theta)


def ellipsoidal_variation_amplitude(mp: NPType, ms: NPType, a: NPType, i: NPType, u: NPType, g: NPType) -> NPType:
    """Amplitude of the ellipsoidal variation signal.

    Calculates the relative amplitude of the ellipsoidal variation signal as
    ``A = alpha (mp / ms) a**-3 sin(i)**2``, where ``alpha = 0.15 (15 + u)(1 + g) / (3 - u)``
    [Morris1985]_, following the approach described by Lillo-Box et al. [Lillo-Box2014]_.

    Parameters
    ----------
    mp
        Planet mass [M_Jup].
    ms
        Stellar mass [M_Sun].
    a
        Semi-major axis divided by the stellar radius.
    i
        Orbital inclination [rad].
    u
        Linear limb darkening coefficient.
    g
        Gravity darkening coefficient.

    Returns
    -------
    float or ndarray
        Relative amplitude of the ellipsoidal variation signal.

    References
    ----------
    .. [Morris1985] Morris, S. L. The ellipsoidal variable stars. ApJ 295, 143 (1985).
    .. [Lillo-Box2014] Lillo-Box, J. et al. Kepler-91b: a planet at the end of its life. A&A 562, A109 (2014).
    """
    ae = 0.15 * (15 + u) * (1 + g) / (3 - u)
    return ae * (mp*mj2kg)/(ms*ms2kg) * a**-3 * sin(i)**2


# Reflected light and thermal emission
# ====================================

def reflected_fr(a: NPType, ab: NPType, r: NPType = 2/3) -> NPType:
    """Reflected light flux ratio per projected area element.

    Calculates the planet-star surface brightness ratio from reflected light at full phase,
    ``ag / a**2``, where the geometric albedo is ``ag = r ab``.

    Parameters
    ----------
    a
        Semi-major axis divided by the stellar radius.
    ab
        Bond albedo.
    r
        Inverse of the phase integral, the ratio of the geometric and Bond albedos. The default
        2/3 corresponds to a Lambertian sphere.

    Returns
    -------
    float or ndarray
        Reflected light flux ratio per projected area element.
    """
    return r * ab / a ** 2


def thermal_fr(Ts: NPType, a: NPType, f: NPType, A: NPType, l: NPType, Ti: NPType = 0) -> NPType:
    """Thermal flux ratio per projected area element.

    Calculates the planet-star surface brightness ratio at a single wavelength from thermal
    emission, approximating both the planet and the star as black bodies. The planet's
    temperature combines its equilibrium temperature with its internal temperature as
    ``T = (Teq**4 + Ti**4)**(1/4)``.

    Parameters
    ----------
    Ts
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    f
        Heat redistribution factor, see :func:`equilibrium_temperature`.
    A
        Bond albedo.
    l
        Wavelength [m].
    Ti
        Internal temperature of the planet [K].

    Returns
    -------
    float or ndarray
        Thermal flux ratio per projected area element.
    """
    tp = (equilibrium_temperature(Ts, a, f, A)**4 + Ti**4)**0.25
    return planck(tp, l) / planck(Ts, l)

def flux_ratio(tstar: NPType, a: NPType, f: NPType, ab: NPType, l: NPType, r: NPType = 2/3, ti: NPType = 0) -> NPType:
    """Total flux ratio per projected area element.

    Calculates the sum of the reflected light and thermal flux ratios, see :func:`reflected_fr`
    and :func:`thermal_fr`.

    Parameters
    ----------
    tstar
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    f
        Heat redistribution factor, see :func:`equilibrium_temperature`.
    ab
        Bond albedo.
    l
        Wavelength [m].
    r
        Inverse of the phase integral, the ratio of the geometric and Bond albedos.
    ti
        Internal temperature of the planet [K].

    Returns
    -------
    float or ndarray
        Total flux ratio per projected area element.
    """
    return reflected_fr(a, ab, r) + thermal_fr(tstar, a, f, ab, l, ti)


def solve_teq(fr: float, tstar: float, a: float, ab: float, l: float, r: float = 2/3, ti: float = 0) -> float:
    """Solve the planet's equilibrium temperature from a flux ratio.

    Finds the equilibrium temperature between 5 K and the stellar temperature for which the
    reflected light and thermal flux ratios sum to the given flux ratio.

    Parameters
    ----------
    fr
        Flux ratio per projected area element.
    tstar
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    ab
        Bond albedo.
    l
        Wavelength [m].
    r
        Inverse of the phase integral, the ratio of the geometric and Bond albedos.
    ti
        Internal temperature of the planet [K].

    Returns
    -------
    float
        Equilibrium temperature [K], or NaN if no solution is found.
    """
    Bs = planck(tstar, l)
    try:
        return brentq(lambda Teq: reflected_fr(a, ab, r) + planck((Teq**4 + ti**4)**0.25, l) / Bs - fr, 5, tstar)
    except ValueError:
        return nan


def solve_ab(fr: float, tstar: float, a: float, f: float, l: float, r: float = 2/3, ti: float = 0) -> float:
    """Solve the Bond albedo from a flux ratio.

    Finds the Bond albedo between 0 and 0.3 for which the reflected light and thermal flux ratios
    sum to the given flux ratio.

    Parameters
    ----------
    fr
        Flux ratio per projected area element.
    tstar
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    f
        Heat redistribution factor, see :func:`equilibrium_temperature`.
    l
        Wavelength [m].
    r
        Inverse of the phase integral, the ratio of the geometric and Bond albedos.
    ti
        Internal temperature of the planet [K].

    Returns
    -------
    float
        Bond albedo, or NaN if no solution is found.
    """
    try:
        return brentq(lambda ab: reflected_fr(a, ab, r) + thermal_fr(tstar, a, f, ab, l, ti) - fr, 0, 0.3)
    except ValueError:
        return nan


def solve_redistribution(fr: float, tstar: float, a: float, ab: float, l: float, r: float = 2/3,
                         ti: float = 0) -> float:
    """Solve the heat redistribution factor from a flux ratio.

    Solves the equilibrium temperature with :func:`solve_teq`, and then finds the redistribution
    factor between 1/4 (uniform redistribution) and 2/3 (no redistribution) that gives it.

    Parameters
    ----------
    fr
        Flux ratio per projected area element.
    tstar
        Effective stellar temperature [K].
    a
        Semi-major axis divided by the stellar radius.
    ab
        Bond albedo.
    l
        Wavelength [m].
    r
        Inverse of the phase integral, the ratio of the geometric and Bond albedos.
    ti
        Internal temperature of the planet [K].

    Returns
    -------
    float
        Heat redistribution factor, or NaN if no solution is found.
    """
    teq = solve_teq(fr, tstar, a, ab, l, r, ti)
    try:
        return brentq(lambda f: equilibrium_temperature(tstar, a, f, ab) - teq, 0.25, 2/3)
    except ValueError:
        return nan
