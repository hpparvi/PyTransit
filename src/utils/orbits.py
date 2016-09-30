from __future__ import division

import numpy as np
from scipy.constants import G, pi

d_h = 24.
d_m = 60 * d_h
d_s = 60 * d_m

au,   au_e             = 1.496e11, 0.0       
msun, msun_e           = 1.9891e30, 0.0      
rsun, rsun_e           = 0.5*1.392684e9, 0.0

def p_from_am(a=1., ms=1.):
    """Orbital period from the semi-major axis and stellar mass.

    Parameters
    ----------

      a    : semi-major axis [AU]
      ms   : stellar mass    [M_Sun]

    Returns
    -------

      p    : Orbital period  [d]
    """
    return np.sqrt((4*pi**2*(a*au)**3)/(G*ms*msun)) / d_s

    
def a_from_mp(ms, period):
    """Semi-major axis from the stellar mass and planet's orbital period.

    Parameters
    ----------

      ms     : stellar mass    [M_Sun]
      period : orbital period  [d]

    Returns
    -------

      a : semi-major axis [AU]
    """
    return ((G*(ms*msun)*(period*d_s)**2)/(4*pi**2))**(1/3)/au


def as_from_rhop(rho, period):
    """Scaled semi-major axis from the stellar density and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]

    Returns
    -------

      as : scaled semi-major axis [R_star]
    """
    return (G/(3*pi))**(1/3)*((period*d_s)**2 * 1e3*rho)**(1/3)


def a_from_rhoprs(rho, period, rstar):
    """Semi-major axis from the stellar density, stellar radius, and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]
      rstar  : stellar radius  [R_Sun]

    Returns
    -------

      a : semi-major axis [AU]
    """
    return as_from_rhop(rho,period)*rstar*rsun/au


def af_transit(e,w):
    """Calculates the -- factor during the transit"""
    return (1.0-e**2)/(1.0 + e*np.sin(w))


def i_from_baew(b,a,e,w):
    """Orbital inclination from the impact parameter, scaled semi-major axis, eccentricity and argument of periastron

    Parameters
    ----------

      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]
      e  : eccentricity           [-]
      w  : argument of periastron [rad]

    Returns
    -------

      i  : inclination            [rad]
    """
    return np.arccos(b / (a*af_transit(e, w)))
