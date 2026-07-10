from math import pi

from numba import njit
from numpy import ones


@njit(fastmath=True)
def ld_general(mu, pv):
    """Giménez (2006) general limb darkening model: I(mu) = 1 - sum c_i (1 - mu^(i+1))."""
    ldp = ones(mu.size)
    for i in range(pv.size):
        ldp -= pv[i] * (1.0 - mu ** (i + 1))
    return ldp


@njit(fastmath=True)
def ldi_general(pv):
    istar = 0.5
    for i in range(pv.size):
        istar -= pv[i] * (0.5 - 1.0 / (i + 3))
    return 2 * pi * istar
