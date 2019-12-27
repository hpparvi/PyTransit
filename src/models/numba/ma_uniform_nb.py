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
from numpy import pi, sqrt, arccos, abs, zeros_like, sign, sin, cos, abs, atleast_2d, zeros
from ...orbits.orbits_py import z_ip_s

TWO_PI = 2.0 * pi
HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi


@njit(cache=False, fastmath=True)
def uniform_z_v(zs, k):
    flux = zeros_like(zs)

    if abs(k - 0.5) < 1e-3:
        k = 0.5

    for i in range(len(zs)):
        z = zs[i]
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
def uniform_z_s(z, k):
    if abs(k - 0.5) < 1e-3:
        k = 0.5

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
def uniform_model(t, pvp, lcids, nsamples, exptimes, es, ms, tae):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    npt = t.size
    flux = zeros((npv, npt))
    for ipv in range(npv):
        k, t0, p, a, i, e, w = pvp[ipv,:]
        for j in prange(npt):
            lci = lcids[j]
            for isample in range(1,nsamples[lci]+1):
                time_offset = exptimes[lci] * ((isample - 0.5) / nsamples[lci] - 0.5)
                z = z_ip_s(t[j]+time_offset, t0, p, a, i, e, w, es, ms, tae)
                if z > 1.0+k:
                    flux[ipv, j] += 1.
                else:
                    flux[ipv, j] += uniform_z_s(z, k)
            flux[ipv, j] /= nsamples[lci]
    return flux