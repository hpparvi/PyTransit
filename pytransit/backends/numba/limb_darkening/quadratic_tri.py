from math import pi, sqrt

from numba import njit
from numpy import zeros

@njit(fastmath=True)
def ld_quadratic_tri(mu, pv):
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u * (1. - mu) - v * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic_tri(pv):
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 2 * pi * 1 / 12 * (-2 * u - v + 6)
