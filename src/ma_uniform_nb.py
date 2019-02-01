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

from numba import njit, prange
from numpy import pi, sqrt, arccos, abs, zeros_like, sign, sin, cos, abs

TWO_PI = 2.0 * pi
HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi


@njit(["f4[:](f4[:],f4,f4)", "f8[:](f8[:],f8,f8)"], cache=False, fastmath=True)
def eval_uniform_z(zs, k, c):
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
        flux[i] = c + (1.0 - c) * flux[i]
    return flux


@njit(["f8[:](f8[:], f8[:], f8, f8, i8, f8)"], cache=False, parallel=True, fastmath=True)
def eval_uniform_circular_t_p(ts, opv, k, c, nss, exptime):
    t0, p, a, i, e, w = opv
    fluxes = zeros_like(ts)

    if abs(k - 0.5) < 1e-3:
        k = 0.5

    for it in prange(len(ts)):
        flux = 0.0
        for iss in range(nss):
            # Compute z
            # ---------
            toffset = exptime * ((iss - 0.5) / nss - 0.5)
            cosph = cos(TWO_PI * (ts[it] + toffset - t0) / p)
            z = sign(cosph) * a * sqrt(1.0 - cosph * cosph * sin(i) ** 2)

            # Compute flux
            # ------------
            if z < 0.0 or z > 1.0 + k:
                flux += 1.0
            elif k > 1.0 and z < k - 1.0:
                flux += 0.0
            elif z > abs(1.0 - k) and z < 1.0 + k:
                kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
                kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
                lambdae = k * k * kap0 + kap1
                lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
                flux += 1.0 - lambdae
            elif z < 1.0 - k:
                flux += 1.0 - k * k
        fluxes[it] = flux

    fluxes /= float(nss)
    fluxes = c + (1.0 - c) * fluxes
    return fluxes