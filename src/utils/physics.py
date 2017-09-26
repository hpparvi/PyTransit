from numpy import exp
from scipy.constants import c, k, h
from numba import jit

@jit
def planck(T, wl):
    """Radiance as a function or black-body temperature and wavelength.

    Parameters
    ----------

      T   : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      B   : Radiance
    """
    return 2*h*c**2 / wl**5 / (exp(h*c / (wl*k*T)) - 1.)

@jit
def planck_ratio(T1, T2, wl):
    """Ratio of the two black-body object radiances

    Parameters
    ----------

      T1  : Temperature  [K]
      T2  : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      rB   : Radiance ratio
    """
    return  (exp(h*c / (wl*k*T2)) - 1.) / (exp(h*c / (wl*k*T1)) - 1.)