#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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
from pathlib import Path
from typing import Union

import pandas as pd
import astropy.units as u
from numba import njit
from numpy import ndarray, pi, atleast_1d, zeros, exp, diff, log, sqrt, nan, vstack, tile, linspace, cos, sin, sum
from scipy.constants import c,h,k,G
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

#from ..contamination.filter import Filter

NPType = Union[float,ndarray]

mj2kg = u.M_jup.to(u.kg)
ms2kg = u.M_sun.to(u.kg)
d2s = u.day.to(u.s)

def equilibrium_temperature(tstar: NPType, a: NPType, f: NPType, ab: NPType) -> NPType:
    """Planetary equilibrium temperature [K].

    Parameters
    ----------
    tstar
        Effective stellar temperature  [K]
    a
        Scaled semi-major axis [Rsun]
    f
        Redistribution factor
    ab
        Bond albedo

    Returns
    -------
    Teq : float or ndarray
        Equilibrium temperature [K]
    """
    return tstar * sqrt(1 / a) * (f * (1 - ab)) ** 0.25


# Thermal emission
# ================

def planck(t: NPType, l: NPType) -> NPType:
    """Radiance of a black body as a function of wavelength.

    Parameters
    ----------
    t
        Black body temperature [K]
    l
        Wavelength [m]

    Returns
    -------
    L  : float or ndarray
        Back body radiance [W m^-2 sr^-1]
    """
    return 2 * h * c ** 2 / l ** 5 / (exp(h * c / (l * k * t)) - 1)

@njit
def summed_planck(teff, wl, tm):
    teff = atleast_1d(teff)
    flux = zeros(teff.shape[0])
    for i in range(flux.size):
        flux[i] = sum(tm*(2*h*c**2 / wl**5 / (exp(h*c / (wl*k*teff[i])) - 1.)))
    return flux


def emission(tp: NPType, tstar: NPType, k: NPType, flt) -> NPType:
    """Thermal emission from the planet.

    Parameters
    ----------
    tp : float or ndarray
        Equilibrium temperature of the planet [K]
    tstar : float or ndarray
        Effective temperature of the star [K]
    k : float or ndarray
        Planet-star radius ratio
    flt: Filter
        Passband transmission

    Returns
    -------
    float or ndarray

    References
    """
    wl = linspace(flt.wl_min, flt.wl_max, 100)
    tm = flt(wl)
    return k**2 * (summed_planck(tp, wl, tm) / summed_planck(tstar, wl, tm))


# Doppler boosting
# ================

def doppler_boosting_alpha(teff: float, flt):
    """The photon weighted bandpass-integrated boosting factor.

    Parameters
    ----------
    teff
        Effective temperature of the star [K]
    flt
        Passband transmission

    Returns
    -------
    float
        The photon weighted bandpass-integrated boosting factor.
    """
    spfile = Path(__file__).parent.parent / 'contamination' / 'data' / 'spectra.h5'
    spectra: pd.DataFrame = pd.read_hdf(spfile)
    wl_nm = spectra.index.values
    wl = wl_nm * 1e-9

    ip = RegularGridInterpolator((wl_nm, spectra.columns.values), spectra.values)
    fl = ip(vstack((wl_nm, tile(teff, wl_nm.size))).T)

    b = 5. + diff(log(fl)) / diff(log(wl))
    w = flt(wl_nm)[1:] * wl[1:] * fl[1:]
    return sum(w * b) / sum(w)


def doppler_boosting_amplitude(mp: NPType, ms: NPType, period: NPType, alpha: NPType) -> NPType:
    """The amplitude of the doppler boosting signal.

    Calculates the amplitude of the doppler boosting (beaming, reflex doppler effect) signal following the approach
    described by Loeb & Gaudi in [Loeb2003]_ . Note that you need to pre-calculate the photon-weighted bandpass-integrated
    boosting factor (alpha) [Bloemen2010]_ [Barclay2012]_ for the star and the instrument using ``doppler_boosting_alpha``.

    Parameters
    ----------
    mp : float or ndarray
        Planetary mass [MJup]
    ms : float or ndarray
        Stellar mass [MSun]
    period : float or ndarray
        Orbital period [d]
    alpha : float or ndarray
        Photon-weighted bandpass-integrated boosting factor

    Returns
    -------
    float or ndarray
        Doppler boosting signal amplitude

    References
    ----------
    .. [Loeb2003] Loeb, A. & Gaudi, B. S. Periodic Flux Variability of Stars due to the Reflex Doppler Effect Induced by
           Planetary Companions. Astrophys. J. 588, L117–L120 (2003).
    .. [Bloemen2010] Bloemen, S. et al. Kepler observations of the beaming binary KPD 1946+4340. MNRAS 410, (2010).
    .. [Barclay2012] Barclay, T. et al. PHOTOMETRICALLY DERIVED MASSES AND RADII OF THE PLANET AND STAR IN THE TrES-2
           SYSTEM. AspJ 761, 53 (2012).
    """
    return alpha / c *(2*pi*G/(d2s*period))**(1/3) * ((mp*mj2kg)/(ms*ms2kg)**(2/3))

# Ellipsoidal variations
# ======================

def ellipsoidal_variation(f: NPType, theta: NPType, mp: float, ms: float, a: float, i: float, e: float, u: float, g: float):
    raise NotImplementedError
    #return ellipsoidal_variation_amplitude(mp, ms, a, i, u, g)

def ellipsoidal_variation_signal(f: NPType, theta: NPType, e: float) -> NPType:
    """

    Parameters
    ----------
    f
        True anomaly [rad]
    theta
        Angle between the line-of-sight and the star-planet direction
    e
        Eccentricity

    Returns
    -------

    """
    return -((1 + e * cos(f)) / (1-e**2))**3 * cos(2*theta)


def ellipsoidal_variation_amplitude(mp: NPType, ms: NPType, a: NPType, i: NPType, u: NPType, g: NPType) -> NPType:
    """The amplitude of the ellipsoidal variation signal.

    Calculates the amplitude of the ellipsoidal variation signal following the approach described by
    Lillo-Box et al. in [Lillo-Box2014]_, page 11.

    Parameters
    ----------
    mp : float or ndarray
        Planetary mass [MJup]
    ms : float or ndarray
        Stellar mass [MSun]
    a : float or ndarray
        Semi-major axis of the orbit divided by the stellar radius
    i : float or ndarray
        Orbital inclination [rad]
    u : float or ndarray
        Linear limb darkening coefficient
    g : float or ndarray
        Gravity darkening coefficient

    Returns
    -------
    ev_amplitude: float or ndarray
        The amplitude of the ellipsoidal variation signal

    References
    ----------
    .. [Lillo-Box2014] Lillo-Box, J. et al. Kepler-91b: a planet at the end of its life. A&A 562, A109 (2014).

    """
    ae = 0.15 * (15 + u) * (1 + g) / (3 - g)
    return ae * (mp*mj2kg)/(ms*ms2kg) * a**-3 * sin(i)**2


def reflected_fr(a: NPType, ab: NPType, r: NPType = 1.5) -> NPType:
    """Reflected flux ratio per projected area element.

    Parameters
    ----------
    a
        Scaled semi-major axis [Rsun]
    ab
        Bond albedo
    r
        Inverse of the phase integral

    Returns
    -------
    fr : float
        Reflected flux ratio
    """
    return r * ab / a ** 2

def flux_ratio(tstar: NPType, a: NPType, f: NPType, ab: NPType, l: NPType, r: NPType = 1.5, ti: NPType = 0) -> NPType:
    """Total flux ratio per projected area element.

    Parameters
    ----------
    tstar
        Effective stellar temperature [K]
    a
        Scaled semi-major axis [Rs]
    f
        Redistribution factor
    ab
        Bond albedo
    l
        Wavelength [m]
    r
        Inverse of the phase integral
    ti
        Temperature [K]

    Returns
    -------
    fr: float
        Total flux ratio
    """
    return reflected_fr(a, ab, r) + thermal_fr(tstar, a, f, ab, l, ti)


def solve_teq(fr, tstar, a, ab, l, r=1.5, ti=0):
    """Solve the equilibrium temperature.

    Parameters
    ----------
    fr
        Flux ratio
    tstar
        Effective stellar temperature [K]
    a
        Scaled semi-major axis [Rs]
    ab
        Bond albedo
    l
        Wavelength [m]
    r
        Inverse of the phase integral
    ti
        Temperature [K]

    Returns
    -------
    Teq : float or ndarray
        Equilibrium temperature

    """
    Bs = planck(tstar, l)
    try:
        return brentq(lambda Teq: reflected_fr(a, ab, r) + planck(Teq + ti, l) / Bs - fr, 5, tstar)
    except ValueError:
        return nan


def solve_ab(fr, tstar, a, f, l, r=1.5, ti=0):
    """Solve the Bond albedo.

    Parameters
    ----------
    fr  :
        Flux ratio                    [-]
    tstar  :
        Effective stellar temperature [K]
    a   :
        Scaled semi-major axis        [Rs]
    A   :
        Bond albedo                   [-]
    l   :
        Wavelength                    [m]
    r   :
        Inverse of the phase integral [-]
    ti  :
        Temperature                   [K]

    Returns
    -------
      A : Bond albedo
    """
    try:
        return brentq(lambda ab: reflected_fr(a, ab, r) + thermal_fr(tstar, a, f, ab, l, ti) - fr, 0, 0.3)
    except ValueError:
        return nan


def solve_redistribution(fr, tstar, a, ab, l):
    """Solve the redistribution factor.

    Parameters
    ----------
    fr  :
        Flux ratio                    [-]
    tstar  :
        Effective stellar temperature [K]
    a   :
        Scaled semi-major axis        [Rs]
    ab   :
        Bond albedo                   [-]
    l   :
        Wavelength                    [m]
    r   :
        Inverse of the phase integral [-]
    Ti  : t
        Temperature                   [K]

    Returns
    -------
      f : Redistribution factor
    """
    Teqs = solve_teq(fr, tstar, l)
    return brentq(lambda f: equilibrium_temperature(tstar, a, f, ab) - Teqs, 0.25, 15)
