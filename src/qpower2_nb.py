# -*- coding: utf-8 -*-

#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

"""QPower2 transit model.

This module implements the qpower2 transit model by Maxted & Gill (ArXIV:1812.01606).
"""

from numpy import sqrt, pi, arccos , ones_like
from numba import njit, prange

@njit
def q1n(z, p, c, alpha, g, I_0):
    s = 1-z**2
    c0 = (1-c+c*s**g)
    c2 = 0.5*alpha*c*s**(g-2)*((alpha-1)*z**2-1)
    return 1-I_0*pi*p**2*(c0 + 0.25*p**2*c2 - 0.125*alpha*c*p**2*s**(g-1))

@njit
def q2n(z, p, c, alpha, g, I_0):
    d = (z**2 - p**2 + 1)/(2*z)
    ra = 0.5*(z-p+d)
    rb = 0.5*(1+d)
    sa = 1-ra**2
    sb = 1-rb**2
    q = (z-d)/p
    w2 = p**2-(d-z)**2
    w = sqrt(w2)
    b0 = 1 - c + c*sa**g
    b1 = -alpha*c*ra*sa**(g-1)
    b2 = 0.5*alpha*c*sa**(g-2)*((alpha-1)*ra**2-1)
    a0 = b0 + b1*(z-ra) + b2*(z-ra)**2
    a1 = b1+2*b2*(z-ra)
    aq = arccos(q)
    J1 = ( (a0*(d-z)-(2/3)*a1*w2 + 0.25*b2*(d-z)*(2*(d-z)**2-p**2))*w + (a0*p**2 + 0.25*b2*p**4)*aq )
    J2 = alpha*c*sa**(g-1)*p**4*(0.125*aq + (1/12)*q*(q**2-2.5)*sqrt(1.-q**2) )
    d0 = 1 - c + c*sb**g
    d1 = -alpha*c*rb*sb**(g-1)
    K1 = ((d0-rb*d1)*arccos(d) + ((rb*d+(2/3)*(1-d**2))*d1 - d*d0)*sqrt(1-d**2))
    K2 = (1/3)*c*alpha*sb**(g+0.5)*(1-d)
    return 1. - I_0*(J1 - J2 + K1 - K2)


@njit(parallel=True)
def qpower2(z, k, u, c):
    I_0 = (u[1] + 2) / (pi * (u[1] - u[0] * u[1] + 2))
    g = 0.5 * u[1]
    f = ones_like(z)
    for i in prange(z.size):
        if z[i] >= 0.0:

            # full transit branch
            # -------------------
            if z[i] <= 1.0 - k:
                f[i] = q1n(z[i], k, u[0], u[1], g, I_0)

            # ingress and egress branch
            # -------------------------
            elif abs(z[i] - 1.0) < k:
                f[i] = q2n(z[i], k, u[0], u[1], g, I_0)

    return c + (1.0 - c) * f

