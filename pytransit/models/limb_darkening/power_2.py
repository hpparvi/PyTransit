from math import pi

from numba import njit
from numpy import zeros, log, log2


@njit(fastmath=True)
def ld_power_2(mu, pv):
    return 1. - pv[0] * (1. - mu ** pv[1])


@njit
def ldi_power_2(pv):
    return 2 * pi * (0.5 - 0.5 * pv[0] + pv[0] / (pv[1] + 2.0))


@njit(fastmath=True)
def ldd_power_2(mu, pv):
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0]*pv[1]*mu**(pv[1]-1.0)
    ldd[1] = mu**pv[1] - 1.0
    ldd[2] = pv[0]*mu**pv[1] * log(mu)
    return ldd


@njit(fastmath=True)
def ld_power_2_pm(mu, pv):
    c = 1 - pv[0] + pv[1]
    a = log2(c/pv[1])
    return 1. - c * (1. - mu**a)


@njit
def ldi_power_2_pm(pv):
    c = 1 - pv[0] + pv[1]
    a = log2(c/pv[1])
    return 2 * pi * (0.5 - 0.5 * c + c / (a + 2.0))
