from numba import njit
from numpy import zeros, pi


@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    return 1. - pv[0] * (1. - mu**0.5) - pv[1] * (1. - mu) - pv[2] * (1. - mu**1.5) - pv[3] * (1. - mu ** 2)


@njit(fastmath=True)
def ldi_nonlinear(pv):
    return 2 * pi * (0.5 - pv[0] / 10.0 - pv[1] / 6.0 - 3.0 * pv[2] / 14.0 - pv[3] / 4.0)


@njit(fastmath=True)
def ldig_nonlinear(pv):
    g = zeros(4)
    g[0] = -pi / 5.0
    g[1] = -pi / 3.0
    g[2] = -3.0 * pi / 7.0
    g[3] = -pi / 2.0
    return g


@njit(fastmath=True)
def ldd_nonlinear(mu, pv):
    ldd = zeros((5, mu.size))
    ldd[0] = 0.5 * pv[0] * mu**(-0.5) + pv[1] + 1.5 * pv[2] * mu**0.5 + 2.0 * pv[3] * mu
    ldd[1] = mu**0.5 - 1.0
    ldd[2] = mu - 1.0
    ldd[3] = mu ** 1.5 - 1.0
    ldd[4] = mu ** 2 - 1.0
    return ldd