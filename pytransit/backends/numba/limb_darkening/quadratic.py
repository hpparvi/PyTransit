from math import pi

from numba import njit
from numpy import zeros


@njit(fastmath=True)
def ld_quadratic(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic(pv):
    return 2 * pi * 1 / 12 * (-2 * pv[0] - pv[1] + 6)

@njit(fastmath=True)
def ldd_quadratic(mu, pv):
    """Quadratic limb darkening model derivatives.

    Quadratic limb darkening model derivatives as an array
    [di/dmu, di/da, di/db].

    Multiply di/dmu by -z/sqrt(1-z**2) to get di/dz"""
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0] + 2*pv[1] - 2*pv[1]*mu
    ldd[1] = mu - 1.0
    ldd[2] = -(1.0 - mu)**2
    return ldd