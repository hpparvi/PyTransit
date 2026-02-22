from numba import njit
from numpy import zeros, log, pi


@njit(fastmath=True)
def ld_power_2(mu, pv):
    return 1. - pv[0] * (1. - mu ** pv[1])


@njit(fastmath=True)
def ldi_power_2(pv):
    return pi * (1.0 - pv[0] * pv[1] / (pv[1] + 2.0))


@njit(fastmath=True)
def ldig_power_2(pv):
    c, a = pv[0], pv[1]
    g = zeros(2)
    g[0] = -pi * a / (a + 2.0)
    g[1] = -2.0 * pi * c / (a + 2.0) ** 2
    return g


@njit(fastmath=True)
def ldd_power_2(mu, pv):
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0]*pv[1]*mu**(pv[1]-1.0)
    ldd[1] = mu**pv[1] - 1.0
    ldd[2] = pv[0]*mu**pv[1] * log(mu)
    return ldd