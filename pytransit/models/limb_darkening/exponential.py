from math import pi

from numba import njit
from numpy import exp


@njit(fastmath=True)
def ld_exponential(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] / (1. - exp(mu))


@njit(fastmath=True)
def ldi_exponential(pv):
    # -int_0^1 mu/(1-exp(mu)) dmu = pi^2/6 + ln(1-1/e) - Li2(1/e)
    return 2 * pi * (0.5 - pv[0] / 6 + 0.7775046341122480 * pv[1])
