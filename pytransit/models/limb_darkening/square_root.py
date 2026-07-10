from math import pi

from numba import njit
from numpy import sqrt


@njit(fastmath=True)
def ld_square_root(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - sqrt(mu))


@njit(fastmath=True)
def ldi_square_root(pv):
    return 2 * pi * (0.5 - pv[0] / 6 - pv[1] / 10)
