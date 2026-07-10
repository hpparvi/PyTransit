from math import pi

from numba import njit
from numpy import log


@njit(fastmath=True)
def ld_logarithmic(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * mu * log(mu)


@njit(fastmath=True)
def ldi_logarithmic(pv):
    return 2 * pi * (0.5 - pv[0] / 6 + pv[1] / 9)
