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
    nan, copysign
from ...orbits.orbits_py import z_ip_s

TWO_PI = 2.0 * pi
HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi


@njit(cache=False, fastmath=True)
def uniform_z_v(zs, k, zsign=1.0):
    flux = zeros_like(zs)

    if abs(k - 0.5) < 1e-3:
        k = 0.5

    for i in range(len(zs)):
        z = zs[i] * zsign
        if z < 0.0 or z > 1.0 + k:
            flux[i] = 1.0
        elif k > 1.0 and z < k - 1.0:
            flux[i] = 0.0
        elif z > abs(1.0 - k) and z < 1.0 + k:
            kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
            kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
            lambdae = k * k * kap0 + kap1
            lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
            flux[i] = 1.0 - lambdae
        elif z < 1.0 - k:
            flux[i] = 1.0 - k * k
    return flux


@njit(fastmath=True)
def uniform_z_s(z, k, zsign=1.0):
    if abs(k - 0.5) < 1e-3:
        k = 0.5

    z *= zsign
    if z < 0.0 or z > 1.0 + k:
        flux = 1.0
    elif k > 1.0 and z < k - 1.0:
        flux = 0.0
    elif z > abs(1.0 - k) and z < 1.0 + k:
        kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
        kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
        lambdae = k * k * kap0 + kap1
        lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
        flux = 1.0 - lambdae
    elif z < 1.0 - k:
        flux = 1.0 - k * k
    return flux


@njit(parallel=True, fastmath=True)
def uniform_model_v(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, es, ms, tae, zsign=1.0):
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
    k = atleast_2d(k)

    npv = k.shape[0]
    npt = t.size
    flux = zeros((npv, npt))
    for j in prange(npt):
        for ipv in range(npv):
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
                z = z_ip_s(t[j] + time_offset, t0[ipv], p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], es, ms, tae)
                if z > 1.0 + _k:
                    flux[ipv, j] += 1.
                else:
                    flux[ipv, j] += uniform_z_s(z, _k, zsign)
            flux[ipv, j] /= nsamples[ilc]
    return flux


@njit(parallel=True, fastmath=True)
def uniform_model_s(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes, es, ms, tae, zsign=1.0):
    k = atleast_1d(k)
    npt = t.size
    flux = zeros(npt)

    if a < 1.0 or e > 0.94:
        flux[:] = nan
        return flux

    for j in prange(npt):
        ilc = lcids[j]
        ipb = pbids[ilc]
        _k = k[0] if k.size == 1 else k[ipb]

        for isample in range(1, nsamples[ilc] + 1):
            time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
            z = z_ip_s(t[j] + time_offset, t0, p, a, i, e, w, es, ms, tae)
            if z > 1.0 + _k:
                flux[j] += 1.
            else:
                flux[j] += uniform_z_s(z, _k, zsign)
        flux[j] /= nsamples[ilc]
    return flux


@njit(parallel=True, fastmath=True)
def uniform_model_pv(t, pvp, lcids, pbids, nsamples, exptimes, es, ms, tae, zsign=1.0):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = t.size
    nk = pvp.shape[1] - 6

    flux = zeros((npv, npt))
    for j in prange(npt):
        for ipv in range(npv):
            t0, p, a, i, e, w = pvp[ipv,nk:]
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
                z = z_ip_s(t[j]+time_offset, t0, p, a, i, e, w, es, ms, tae)
                if z > 1.0+k:
                    flux[ipv, j] += 1.
                else:
                    flux[ipv, j] += uniform_z_s(z, k, zsign)
            flux[ipv, j] /= nsamples[ilc]
    return flux
