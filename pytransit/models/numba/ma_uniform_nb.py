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

from numba import njit, prange
from numpy import pi, sqrt, arccos, abs, zeros_like, sign, sin, cos, abs, atleast_2d, zeros, atleast_1d, isnan, inf, \
    nan, copysign, fmax, floor
from ...orbits.taylor_z import vajs_from_paiew, z_taylor_st, vajs_from_paiew_eclipse

TWO_PI = 2.0 * pi
HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi


@njit(cache=False, fastmath=True)
def uniform_z_v(zs, k, zsign=1.0):
    flux = zeros_like(zs)

    for i in range(len(zs)):
        z = zs[i] * zsign

        # Out of transit
        # --------------
        if (copysign(1.0, z) < 0.0) or z >= 1.0 + k:
            flux[i] = 1.0

        # Full transit by a larger object
        # -------------------------------
        elif k > 1.0 and z < k - 1.0:
            flux[i] = 0.0

        # Full transit
        # ------------
        elif z <= 1.0 - k:
            flux[i] = 1.0 - k * k

        # Ingress and egress
        # ------------------
        else:
            kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
            kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
            lambdae = k * k * kap0 + kap1
            lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
            flux[i] = 1.0 - lambdae

    return flux


@njit(fastmath=True)
def uniform_z_s(z, k, zsign):
    z *= zsign

    # Out of transit
    # --------------
    if (copysign(1.0, z) < 0.0) or z >= 1.0 + k:
        flux = 1.0

    # Full transit by a larger object
    # -------------------------------
    elif k > 1.0 and z < k - 1.0:
        flux = 0.0

    # Full transit
    # ------------
    elif z <= 1.0 - k:
        flux = 1.0 - k * k

    # Ingress and egress
    # ------------------
    else:
        kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
        kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
        lambdae = k * k * kap0 + kap1
        lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
        flux = 1.0 - lambdae

    return flux


@njit(parallel=True, fastmath=True)
def uniform_model_v(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, zsign):
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
    k = atleast_2d(k)

    npv = k.shape[0]
    npt = t.size
    flux = zeros((npv, npt))
    for ipv in prange(npv):
        if zsign >= 0:
            y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
            half_window_width = fmax(0.125, (2.0 + k[0, 0])/vx)
            et = 0.0
        else:
            et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
            half_window_width = fmax(0.125, (2.0 + k[0, 0]) / (-vx))

        for j in range(npt):
            epoch = floor((t[j] - t0[ipv] - et + 0.5 * p[ipv]) / p[ipv])
            tc = t[j] - (t0[ipv] + et + epoch * p[ipv])
            if abs(tc) > half_window_width:
                flux[ipv, j] = 1.0
            else:
                ilc = lcids[j]
                ipb = pbids[ilc]

                if a[ipv] < 1.0 or e[ipv] > 0.94:
                    flux[ipv, j] = nan
                    continue

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
                        flux[ipv, j] += uniform_z_s(z, _k, 1.0)
                flux[ipv, j] /= nsamples[ilc]
    return flux


@njit(parallel=False, fastmath=True)
def uniform_model_s(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, zsign):
    k = atleast_1d(k)
    npt = t.size
    flux = zeros(npt)

    if a < 1.0:
        flux[:] = nan
        return flux

    if zsign >= 0:
        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
        half_window_width = fmax(0.125, (2.0 + k[0]) / vx)
        et = 0.0
    else:
        et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p, a, i, e, w)
        half_window_width = fmax(0.125, (2.0 + k[0]) / (-vx))

    for j in range(npt):
        epoch = floor((t[j] - t0 - et + 0.5 * p) / p)
        tc = t[j] - (t0 + et + epoch * p)
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
                    flux[j] += uniform_z_s(z, _k, 1.0)
            flux[j] /= nsamples[ilc]
    return flux


@njit(parallel=True, fastmath=True)
def uniform_model_pv(t, pvp, lcids, pbids, nsamples, exptimes, zsign):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = t.size
    nk = pvp.shape[1] - 6

    flux = zeros((npv, npt))
    for ipv in range(npv):
        t0, p, a, i, e, w = pvp[ipv, nk:]

        if zsign >= 0:
            y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
            half_window_width = fmax(0.125, (2 + pvp[ipv, 0])/vx)
            et = 0.0
        else:
            et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p, a, i, e, w)
            half_window_width = fmax(0.125, (2.0 + pvp[ipv, 0]) / (-vx))

        for j in prange(npt):
            epoch = floor((t[j] - t0 - et + 0.5 * p) / p)
            tc = t[j] - (t0 + et + epoch * p)
            if abs(tc) > half_window_width:
                flux[ipv, j] = 1.0
            else:
                ilc = lcids[j]
                ipb = pbids[ilc]

                if a < 1.0 or e > 0.94:
                    flux[ipv, j] = nan
                    continue

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
                    if z > 1.0+k:
                        flux[ipv, j] += 1.
                    else:
                        flux[ipv, j] += uniform_z_s(z, k, 1.0)
                flux[ipv, j] /= nsamples[ilc]
    return flux
