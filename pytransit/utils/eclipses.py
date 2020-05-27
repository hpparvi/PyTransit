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

"""Methods for modelling of secondary eclipses

  .. deprecated:: 2.0.0
    ``pytransit.utils.eclipses`` will be removed in PyTransit 2.1, it is replaced by ``pytransit.utils.phasecurves``.

"""
import warnings
from numpy import sqrt, exp, NaN

from scipy.constants import k,h,c
from scipy.optimize import brentq

warnings.warn("the pytransits.utils.eclipses module is deprecated and will be removed in PyTransit 2.1",
               FutureWarning, stacklevel=2)


def Teq(Ts, a, f, A):
    """Equilibruim temperature of a planet.

    Parameters
    ----------

      Ts : Effective stellar temperature  [K]
      a  : Scaled semi-major axis         [Rs]
      f  : Redistribution factor          [-]
      A  : Bond albedo                    [-]

    Returns
    -------

      Teq : Equilibrium temperature       [K]
    """
    return Ts*sqrt(1/a)*(f*(1-A))**0.25


def Planck(T, l):
    """Radiance of a black body as a function of wavelength.

    Parameters
    ----------

      T  : Black body temperature [K]
      l  : Wavelength             [m]

    Returns
    -------

      L  : Back body radiance [W m^-2 sr^-1]
    """
    return 2*h*c**2/l**5 / (exp(h*c/(l*k*T)) - 1)


def reflected_fr(a, A, r=1.5):
    """Reflected flux ratio per projected area element.

    Parameters
    ----------

      a  : Scaled semi-major axis         [Rs]
      A  : Bond albedo                    [-]
      r  : Inverse of the phase integral  [-]

    Returns
    -------

      fr : Reflected flux ratio           [-]
    """
    return r*A/a**2


def thermal_fr(Ts, a, f, A, l, Ti=0):
    """Thermal flux ratio per projected area element.
    
    Parameters
    ----------

      Ts  : Effective stellar temperature [K]
      a   : Scaled semi-major axis        [Rs]
      f   : Redistribution factor         [-]
      A   : Bond albedo                   [-]
      l   : Wavelength                    [m]
      Ti  : temperature                   [K]

    Returns
    -------

      fr: Thermal flux ratio              [-]
    """
    return Planck(Teq(Ts, a, f, A)+Ti, l) / Planck(Ts, l)


def flux_ratio(Ts, a, f, A, l, r=1.5, Ti=0):
    """Total flux ratio per projected area element.

    Parameters
    ----------

      Ts  : Effective stellar temperature [K]
      a   : Scaled semi-major axis        [Rs]
      f   : Redistribution factor         [-]
      A   : Bond albedo                   [-]
      l   : Wavelength                    [m]
      r   : Inverse of the phase integral [-]
      Ti  : temperature                   [K]

    Returns
    -------

      fr: Total flux ratio                [-]
    """
    return reflected_fr(a, A, r) + thermal_fr(Ts, a, f, A, l, Ti)


def solve_Teq(fr, Ts, a, A, l, r=1.5, Ti=0):
    """Solve the equilibrium temperature.

    Parameters
    ----------

      fr  : Flux ratio                    [-]
      Ts  : Effective stellar temperature [K]
      a   : Scaled semi-major axis        [Rs]
      A   : Bond albedo                   [-]
      l   : Wavelength                    [m]
      r   : Inverse of the phase integral [-]
      Ti  : temperature                   [K]

    Returns
    -------

      Teq : Equilibrium temperature

    """
    Bs = Planck(Ts, l)
    try:
        return brentq(lambda Teq: reflected_fr(a, A, r) + Planck(Teq+Ti, l)/Bs - fr, 5, Ts)
    except ValueError:
        return NaN


def solve_A(fr, Ts, a, f, l, r=1.5, Ti=0):
    """Solve the Bond albedo.

    Parameters
    ----------

      fr  : Flux ratio                    [-]
      Ts  : Effective stellar temperature [K]
      a   : Scaled semi-major axis        [Rs]
      A   : Bond albedo                   [-]
      l   : Wavelength                    [m]
      r   : Inverse of the phase integral [-]
      Ti  : temperature                   [K]

    Returns
    -------

      A : Bond albedo
    """
    try:
        return brentq(lambda A: reflected_fr(a, A, r) + thermal_fr(Ts, a, f, A, l, Ti) - fr, 0, 0.3)
    except ValueError:
        return NaN

def solve_redistribution(fr, Ts, a, A, l):
    """Solve the redistribution factor.

    Parameters
    ----------

      fr  : Flux ratio                    [-]
      Ts  : Effective stellar temperature [K]
      a   : Scaled semi-major axis        [Rs]
      A   : Bond albedo                   [-]
      l   : Wavelength                    [m]
      r   : Inverse of the phase integral [-]
      Ti  : temperature                   [K]

    Returns
    -------

      f : Redistribution factor
    """
    Teqs = solve_Teq(fr, Ts, l)
    return brentq(lambda f: Teq(Ts, a, f, A) - Teqs, 0.25, 15)
