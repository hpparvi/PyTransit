from math import pi

from numba import njit


@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    return 1. - pv[0] * (1. - mu**0.5) - pv[1] * (1. - mu) - pv[2] * (1. - mu**1.5) - pv[3] * (1. - mu ** 2)


@njit(fastmath=True)
def ldi_nonlinear(pv):
    return 2 * pi * (0.5 - pv[0] / 10 - pv[1] / 6 - 3 * pv[2] / 14 - pv[3] / 4)
