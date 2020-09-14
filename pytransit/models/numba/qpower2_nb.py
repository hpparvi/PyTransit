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

This module implements the qpower2 transit model by Maxted & Gill (A&A, 622, A33, 2019).
"""

from numpy import any, sqrt, pi, arccos, ones_like, atleast_2d, zeros, atleast_1d, nan, copysign, fmax, floor
from numba import njit, prange

from ...orbits.taylor_z import vajs_from_paiew, z_taylor_st


@njit(fastmath=True)
def q1n(z, p, c, alpha, g, I_0):
    s = 1-z**2
    c0 = (1-c+c*s**g)
    c2 = 0.5*alpha*c*s**(g-2)*((alpha-1)*z**2-1)
    return 1-I_0*pi*p**2*(c0 + 0.25*p**2*c2 - 0.125*alpha*c*p**2*s**(g-1))


@njit(fastmath=True)
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
    J1 = (a0*(d-z)-(2/3)*a1*w2 + 0.25*b2*(d-z)*(2*(d-z)**2-p**2))*w + (a0*p**2 + 0.25*b2*p**4)*aq
    J2 = alpha*c*sa**(g-1)*p**4*(0.125*aq + (1/12)*q*(q**2-2.5)*sqrt(1.-q**2) )
    d0 = 1 - c + c*sb**g
    d1 = -alpha*c*rb*sb**(g-1)
    K1 = ((d0-rb*d1)*arccos(d) + ((rb*d+(2/3)*(1-d**2))*d1 - d*d0)*sqrt(1-d**2))
    K2 = (1/3)*c*alpha*sb**(g+0.5)*(1-d)
    return 1. - I_0*(J1 - J2 + K1 - K2)


@njit(fastmath=False)
def qpower2_z_s(z, k, u):
    if k > 1.0 or any(u < 0.0):
        return nan
    if (copysign(1.0, z) < 0.0) or z >= 1.0 + k:
        flux = 1.0
    else:
        I_0 = (u[1] + 2) / (pi * (u[1] - u[0] * u[1] + 2))
        g = 0.5 * u[1]
        if z <= 1.0 - k:
            flux = q1n(z, k, u[0], u[1], g, I_0)
        elif abs(z - 1.0) < k:
            flux = q2n(z, k, u[0], u[1], g, I_0)
    return max(flux, 0.0)


@njit(parallel=True, fastmath=True)
def qpower2_model_v(t, k, ldc, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes):
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
    k = atleast_2d(k)
    ldc = atleast_2d(ldc)
    npv = k.shape[0]
    npt = t.size
    flux = zeros((npv, npt))
    for ipv in prange(npv):
        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        half_window_width = fmax(0.125, (2.0 + k[0, 0])/vx)

        for j in range(npt):
            epoch = floor((t[j] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = t[j] - (t0[ipv] + epoch * p[ipv])
            if abs(tc) > half_window_width:
                flux[ipv, j] = 1.0
            else:
                ilc = lcids[j]
                ipb = pbids[ilc]

                if k.shape[1] == 1:
                    _k = k[ipv, 0]
                else:
                    _k = k[ipv, ipb]

                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                    if z > 1.0 + _k:
                        flux[ipv, j] += 1.
                    else:
                        flux[ipv, j] += qpower2_z_s(z, _k, ldc[ipv, 2*ipb:2*(ipb+1)])
                flux[ipv, j] /= nsamples[ilc]
    return flux


@njit(parallel=False, fastmath=True)
def qpower2_model_s(t, k, ldc, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes):
    k = atleast_1d(k)
    ldc = atleast_1d(ldc)
    npt = t.size

    y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
    half_window_width = fmax(0.125, (2.0 + k[0]) / vx)

    flux = zeros(npt)
    for j in range(npt):
        epoch = floor((t[j] - t0 + 0.5 * p) / p)
        tc = t[j] - (t0 + epoch * p)
        if abs(tc) > half_window_width:
            flux[j] = 1.0
        else:
            ilc = lcids[j]
            ipb = pbids[ilc]
            _k = k[0] if k.size == 1 else k[ipb]

            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                if z > 1.0 + _k:
                    flux[j] += 1.
                else:
                    flux[j] += qpower2_z_s(z, _k, ldc[2*ipb:2*(ipb+1)])
            flux[j] /= nsamples[ilc]
    return flux


@njit(parallel=True, fastmath=True)
def qpower2_model_pv(t, pvp, ldc, lcids, pbids, nsamples, exptimes):
    pvp = atleast_2d(pvp)
    ldc = atleast_2d(ldc)
    npv = pvp.shape[0]
    nk = pvp.shape[1] - 6
    npt = t.size
    flux = zeros((npv, npt))

    for ipv in prange(npv):
        t0, p, a, i, e, w = pvp[ipv, nk:]
        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
        half_window_width = fmax(0.125, (2 + pvp[ipv, 0])/vx)
        for j in range(npt):
            epoch = floor((t[j] - t0 + 0.5*p)/p)
            tc = t[j] - (t0 + epoch*p)
            if abs(tc) > half_window_width:
                flux[ipv, j] = 1.0
            else:
                ilc = lcids[j]
                ipb = pbids[ilc]

                if nk == 1:
                    k = pvp[ipv, 0]
                else:
                    if ipb < nk:
                        k = pvp[ipv, ipb]
                    else:
                        k = nan

                for isample in range(1,nsamples[ilc]+1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                    if z < 0.0 or z > 1.0+k:
                        flux[ipv, j] += 1.
                    else:
                        flux[ipv, j] += qpower2_z_s(z, k, ldc[ipv, 2*ipb:2*(ipb+1)])
                flux[ipv, j] /= nsamples[ilc]
    return flux
