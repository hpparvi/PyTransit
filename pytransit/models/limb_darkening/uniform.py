from math import pi

from numba import njit
from numpy import ones


@njit(fastmath=True)
def ld_uniform(mu, pv):
    return ones(mu.size)


@njit(fastmath=True)
def ldi_uniform(pv):
    return pi
