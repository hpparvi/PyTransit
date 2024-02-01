from math import pi

from numba import njit
from numpy import zeros


@njit(fastmath=True)
def ld_linear(mu, pv):
    return 1. - pv[0] * (1. - mu)


@njit(fastmath=True)
def ldi_linear(pv):
    return 2 * pi * 1 / 6 * (3 - 2 * pv[0])


@njit(fastmath=True)
def ldd_linear(mu, pv):
    ldd = zeros((2, mu.size))
    ldd[0] = pv[0]
    ldd[1] = mu - 1.0
    return ldd