from numba import njit
from numpy import zeros


@njit(fastmath=True)
def ld_general(mu, pv):
    ldp = zeros(mu.size)
    for i in range(pv.size):
        ldp += pv[i] * (1.0 - mu ** (i + 1))
    return ldp