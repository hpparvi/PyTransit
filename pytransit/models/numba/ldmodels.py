#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from numba import njit
from numpy import ones, pi, sqrt, log, exp, zeros, power


@njit(fastmath=True)
def ld_uniform(mu, pv):
    return ones(mu.size)


@njit(fastmath=True)
def ldi_uniform(pv):
    return pi


@njit(fastmath=True)
def ld_linear(mu, pv):
    return 1. - pv[0] * (1. - mu)


@njit(fastmath=True)
def ldi_linear(pv):
    return 2 * pi * 1 / 6 * (3 - 2 * pv[0])


@njit(fastmath=True)
def ld_quadratic(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic(pv):
    return 2 * pi * 1 / 12 * (-2 * pv[0] - pv[1] + 6)


@njit(fastmath=True)
def ld_quadratic_tri(mu, pv):
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u * (1. - mu) - v * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic_tri(pv):
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 2 * pi * 1 / 12 * (-2 * u - v + 6)


@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    return 1. - pv[0] * (1. - sqrt(mu)) - pv[1] * (1. - mu) - pv[2] * (1. - power(mu, 1.5)) - pv[3] * (1. - mu ** 2)


@njit(fastmath=True)
def ld_general(mu, pv):
    ldp = zeros(mu.size)
    for i in range(pv.size):
        ldp += pv[i] * (1.0 - mu ** (i + 1))
    return ldp


@njit(fastmath=True)
def ld_square_root(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - sqrt(mu))


@njit(fastmath=True)
def ld_logarithmic(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] * mu * log(mu)


@njit(fastmath=True)
def ld_exponential(mu, pv):
    return 1. - pv[0] * (1. - mu) - pv[1] / (1. - exp(mu))


@njit(fastmath=True)
def ld_power_2(mu, pv):
    return 1. - pv[0] * (1. - mu ** pv[1])

@njit(fastmath=True)
def ld_power_2_pm(mu, pv):
    c = 1 - pv[0] + pv[1]
    a = log(c/pv[1])
    return 1. - c * (1. - mu**a)

@njit
def evaluate_ld(ldm, mu, pvo):
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo

    npv = pv.shape[0]
    npb = pv.shape[1]
    ldp = zeros((npv, npb, mu.size))
    for ipv in range(npv):
        for ipb in range(npb):
            ldp[ipv, ipb, :] = ldm(mu, pv[ipv, ipb])
    return ldp


@njit
def evaluate_ldi(ldi, pvo):
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo

    npv = pv.shape[0]
    npb = pv.shape[1]
    istar = zeros((npv, npb))
    for ipv in range(npv):
        for ipb in range(npb):
            istar[ipv, ipb] = ldi(pv[ipv, ipb])
    return istar