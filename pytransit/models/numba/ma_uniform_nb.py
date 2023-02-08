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
from math import fabs

from numba import njit, prange
from numpy import pi, sqrt, arccos, abs, zeros_like, sign, sin, cos, abs, atleast_2d, zeros, atleast_1d, isnan, inf, \
    nan, copysign, fmax, floor

from ...orbits.taylor_z import vajs_from_paiew, z_taylor_st, vajs_from_paiew_eclipse, t14

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


@njit(fastmath=True)
def folded_time(t, t0, p):
    epoch = floor((t - t0 + 0.5 * p) / p)
    return t - (t0 + epoch * p)


@njit
def dfdz(z, r1, r2):
    """Circle-circle intersection derivative givena scalar z."""
    if r1 < z - r2:
        return 0.0
    elif r1 >= z + r2:
        return 0.0
    elif z - r2 <= -r1:
        return 0.0
    else:
        a = z**2 + r2**2 - r1**2
        b = z**2 + r1**2 - r2**2
        t1 = - r2**2 * (1/r2 - a / (2*r2*z**2))  /  sqrt(1 - a**2 / (4*r2**2*z**2))
        t2 = - r1**2 * (1/r1 - b / (2*r1*z**2))  /  sqrt(1 - b**2 / (4*r1**2*z**2))
        t3 = z*(r1**2 + r2**2 - z**2) / sqrt((-z + r2 + r1) * (z + r2 - r1) * (z - r2 + r1) * (z + r2 + r1))
        return (t1 + t2 - t3) / pi


@njit
def xuniform_model_v(times, k, t0, p, a, i, e, w, with_derivatives):
    npt = times.size
    flux = zeros(npt)
    dflux = zeros((6, npt)) if with_derivatives else None
    ds = zeros(6)

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux, dflux

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1.)

    # ---------------------------------------------------------------------
    # Solve the Taylor series coefficients for the planet's (x, y) location
    # and its derivatives if they're requested.
    # ---------------------------------------------------------------------
    cf = solve_xy_p5s(0.0, p, a, i, e, w)
    if with_derivatives:
        dcf = xy_derivative_coeffs(diffs(p, a, i, e, w, 1e-4), 1e-4, cf)
    else:
        dcf = None

    # -----------------------------------------------
    # Calculate the transit model and its derivatives
    # -----------------------------------------------
    for j in range(npt):
        t = folded_time(times[j], t0, p)
        if fabs(t) > half_window_width:
            flux[j] = 1.0
        else:
            x, y, d = xyd_t15s(t, cf)
            if d <= 1.0 + k:
                flux[j] += uniform_z_s(d, k, 1.0)
                if with_derivatives:
                    ds = pd_derivatives_s(t, x, y, dcf, ds)
                    dflux[:, j] += -dfdz(d, k, 1.0) * ds
            else:
                flux[j] += 1.
    return flux, dflux


@njit(parallel=True, fastmath=False)
def uniform_model_v(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, zsign):
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
    k = atleast_2d(k)

    npv = k.shape[0]
    npt = t.size
    flux = zeros((npv, npt))
    for ipv in prange(npv):
        if a[ipv] <= 1.0 or e[ipv] >= 0.96:
            flux[ipv, :] = nan
            continue

        if zsign >= 0:
            y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
            et = 0.0
            half_window_width = 0.025 + 0.5 * t14(k[ipv, 0], y0, vx, vy, ax, ay, jx, jy, sx, sy)
        else:
            et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
            half_window_width = 0.025 - 0.5 * t14(k[ipv, 0], y0, vx, vy, ax, ay, jx, jy, sx, sy)

        for j in range(npt):
            epoch = floor((t[j] - t0[ipv] - et + 0.5 * p[ipv]) / p[ipv])
            tc = t[j] - (t0[ipv] + et + epoch * p[ipv])
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
                        flux[ipv, j] += uniform_z_s(z, _k, 1.0)
                flux[ipv, j] /= nsamples[ilc]
    return flux


@njit(parallel=False, fastmath=False)
def uniform_model_s(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, zsign):
    k = atleast_1d(k)
    npt = t.size
    flux = zeros(npt)

    if a <= 1.0:
        flux[:] = nan
        return flux

    if zsign >= 0:
        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
        et = 0.0
        half_window_width = 0.025 + 0.5 * t14(k[0], y0, vx, vy, ax, ay, jx, jy, sx, sy)
    else:
        et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p, a, i, e, w)
        half_window_width = 0.025 - 0.5 * t14(k[0], y0, vx, vy, ax, ay, jx, jy, sx, sy)

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


@njit(parallel=True, fastmath=False)
def uniform_model_pv(t, pvp, lcids, pbids, nsamples, exptimes, zsign):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = t.size
    nk = pvp.shape[1] - 6

    flux = zeros((npv, npt))
    for ipv in range(npv):
        t0, p, a, i, e, w = pvp[ipv, nk:]
        if a <= 1.0:
            flux[ipv, :] = nan
            continue

        if zsign >= 0:
            y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
            et = 0.0
            half_window_width = 0.025 + 0.5 * t14(pvp[ipv, 0], y0, vx, vy, ax, ay, jx, jy, sx, sy)
        else:
            et, y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew_eclipse(p, a, i, e, w)
            half_window_width = 0.025 - 0.5 * t14(pvp[ipv, 0], y0, vx, vy, ax, ay, jx, jy, sx, sy)

        for j in prange(npt):
            epoch = floor((t[j] - t0 - et + 0.5 * p) / p)
            tc = t[j] - (t0 + et + epoch * p)
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
                    if z > 1.0+k:
                        flux[ipv, j] += 1.
                    else:
                        flux[ipv, j] += uniform_z_s(z, k, 1.0)
                flux[ipv, j] /= nsamples[ilc]
    return flux
