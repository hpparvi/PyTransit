from math import pi, sqrt

from numba import njit


@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    return 1. - pv[0] * (1. - sqrt(mu)) - pv[1] * (1. - mu) - pv[2] * (1. - mu**1.5) - pv[3] * (1. - mu ** 2)