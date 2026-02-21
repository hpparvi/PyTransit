from numba import njit
from numpy import zeros, pi


@njit(fastmath=True)
def ld_general(mu, pv):
    ldp = zeros(mu.size)
    for i in range(pv.size):
        ldp += pv[i] * (1.0 - mu ** (i + 1))
    return ldp


@njit(fastmath=True)
def ldi_general(pv):
    s = 0.0
    for i in range(pv.size):
        s += pv[i] * (i + 1.0) / (2.0 * (i + 3.0))
    return 2 * pi * s


@njit(fastmath=True)
def ldd_general(mu, pv):
    n = pv.size
    ldd = zeros((1 + n, mu.size))
    for i in range(n):
        ldd[0] -= pv[i] * (i + 1.0) * mu ** i
        ldd[1 + i] = 1.0 - mu ** (i + 1)
    return ldd