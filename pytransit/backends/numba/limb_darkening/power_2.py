from math import pi, sqrt, gamma

from numba import njit
from numpy import zeros, log


@njit(fastmath=True)
def ld_power_2(mu, pv):
    return 1. - pv[0] * (1. - mu ** pv[1])


@njit
def ldi_power_2(mu, pv):
    return 2 * pi * sqrt(pi) * pv[0] * gamma(0.5*pv[1] + 1.0) / (2*gamma(0.5*(pv[1]+3.0))) - pv[0] + 1


@njit(fastmath=True)
def ldd_power_2(mu, pv):
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0]*pv[1]*mu**(pv[1]-1.0)
    ldd[1] = mu**pv[1] - 1.0
    ldd[2] = pv[0]*mu**pv[1] * log(mu)
    return ldd