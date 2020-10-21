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
from typing import Tuple

from numba import njit, prange
from numpy import pi, sqrt, arccos, abs, log, zeros, linspace, array, atleast_2d, floor, inf, isnan, atleast_1d, ndarray, nan, copysign, \
    fmax, any

from ...orbits.taylor_z import vajs_from_paiew, z_taylor_st

HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi


@njit(cache=False)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353

    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""

    kc = sqrt(1.0 - k * k)
    e = kc
    p = sqrt(n + 1.0)
    ip = 1.0 / p
    m0 = 1.0
    c = 1.0
    d = 1.0 / p

    for nit in range(1000):
        f = c
        c = d / p + c
        g = e / p
        d = 2.0 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0

        if (abs(1.0 - kc / g) > 1e-8):
            kc = 2.0 * sqrt(e)
            e = kc * m0
        else:
            return HALF_PI * (c * m0 + d) / (m0 * (m0 + p))
    return 0.0


@njit(cache=False)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390

    m1 = 1.0 - k * k
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1.0 / m1)
    return ee1 + ee2


@njit(cache=False)
def ellk(k):
    a0 = 1.386294361120
    a1 = 0.096663442590
    a2 = 0.035900923830
    a3 = 0.037425637130
    a4 = 0.014511962120
    b0 = 0.50
    b1 = 0.124985935970
    b2 = 0.068802485760
    b3 = 0.033283553460
    b4 = 0.004417870120

    m1 = 1.0 - k * k
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1)
    return ek1 - ek2


def calculate_interpolation_tables(kmin: float = 0.05, kmax: float = 0.2, nk: int = 512, nz:int = 512) -> Tuple:
    zs = linspace(0, 1 + 1.001 * kmax, nz)
    ks = linspace(kmin, kmax, nk)

    ld = zeros((nk, nz))
    le = zeros((nk, nz))
    ed = zeros((nk, nz))

    for ik, k in enumerate(ks):
        _, ld[ik, :], le[ik, :], ed[ik, :] = eval_quad_z_v(zs, k, array([[0.0, 0.0]]))

    return ed, le, ld, ks, zs


@njit(cache=False, parallel=False)
def eval_quad_z_v(zs, k, u: ndarray):
    """Evaluates the transit model for an array of normalized distances.

    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u: 2D array
        Limb darkening coefficients

    Returns
    -------
    Transit model evaluated at `z`.

    """
    if abs(k - 0.5) < 1.0e-4:
        k = 0.5

    npt = len(zs)
    npb = u.shape[0]

    k2 = k ** 2
    omega = zeros(npb)
    flux = zeros((npt, npb))
    le = zeros(npt)
    ld = zeros(npt)
    ed = zeros(npt)

    for i in range(npb):
        omega[i] = 1.0 - u[i,0] / 3.0 - u[i,1] / 6.0

    for i in prange(npt):
        z = zs[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # The source is unocculted
        if z > 1.0 + k or z < 0.0:
            flux[i, :] = 1.0
            le[i] = 0.0
            ld[i] = 0.0
            ed[i] = 0.0
            continue

        # The source is completely occulted
        elif (k >= 1.0 and z <= k - 1.0):
            flux[i, :] = 0.0
            le[i] = 1.0
            ld[i] = 1.0
            ed[i] = 1.0
            continue

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z ** 2

        # The source is partially occulted and the occulting object crosses the limb
        # Equation (26):
        if z >= abs(1.0 - k) and z <= 1.0 + k:
            kap1 = arccos(min((1.0 - k2 + z2) / (2.0 * z), 1.0))
            kap0 = arccos(min((k2 + z2 - 1.0) / (2.0 * k * z), 1.0))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4.0 * z2 - (1.0 + z2 - k2) ** 2, 0.0))) * INV_PI

        # The occulting object transits the source star (but doesn't completely cover it):
        if z <= 1.0 - k:
            le[i] = k2

        # The edge of the occulting star lies at the origin- special expressions in this case:
        if abs(z - k) < 1.e-4 * (z + k):
            # ! Table 3, Case V.:
            if (k == 0.5):
                ld[i] = 1.0 / 3.0 - 4.0 * INV_PI / 9.0
                ed[i] = 3.0 / 32.0
            elif (z > 0.5):
                q = 0.50 / k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = 1.0 / 3.0 + 16.0 * k / 9.0 * INV_PI * (2.0 * k2 - 1.0) * Ek - (
                                                                                       32.0 * k ** 4 - 20.0 * k2 + 3.0) / 9.0 * INV_PI / k * Kk
                ed[i] = 1.0 / 2.0 * INV_PI * (
                    kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)))
            elif (z < 0.5):
                # ! Table 3, Case VI.:
                q = 2.0 * k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = 1.0 / 3.0 + 2.0 / 9.0 * INV_PI * (4.0 * (2.0 * k2 - 1.0) * Ek + (1.0 - 4.0 * k2) * Kk)
                ed[i] = k2 / 2.0 * (k2 + 2.0 * z2)

        # The occulting star partly occults the source and crosses the limb:
        # Table 3, Case III:
        if ((z > 0.5 + abs(k - 0.5) and z < 1.0 + k) or (k > 0.50 and z > abs(1.0 - k) and z < k)):
            q = sqrt((1.0 - (k - z) ** 2) / 4.0 / z / k)
            Kk = ellk(q)
            Ek = ellec(q)
            n = 1.0 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld[i] = 1.0 / 9.0 * INV_PI / sqrt(k * z) * (
                ((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk + 4.0 * k * z * (
                    z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld[i] = ld[i] + 2.0 / 3.0
            ed[i] = 1.0 / 2.0 * INV_PI * (
                kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)))

        # The occulting star transits the source:
        # Table 3, Case IV.:
        if k <= 1.0 and z < (1.0 - k):
            q = sqrt((x2 - x1) / (1.0 - x1))
            Kk = ellk(q)
            Ek = ellec(q)
            n = x2 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld[i] = 2.0 / 9.0 * INV_PI / sqrt(1.0 - x1) * (
                (1.0 - 5.0 * z2 + k2 + x3 * x3) * Kk + (1.0 - x1) * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld[i] = ld[i] + 2.0 / 3.0
            if (abs(k + z - 1.0) < 1.e-4):
                ld[i] = 2.0 / 3.0 * INV_PI * arccos(1.0 - 2.0 * k) - 4.0 / 9.0 * INV_PI * sqrt(k * (1.0 - k)) * (
                    3.0 + 2.0 * k - 8.0 * k2)
            ed[i] = k2 / 2.0 * (k2 + 2.0 * z2)

        for j in range(npb):
            flux[i, j] = 1.0 - ((1.0 - u[j,0] - 2.0 * u[j,1]) * le[i] + (u[j,0] + 2.0 * u[j,1]) * ld[i] + u[j,1] * ed[i]) / omega[j]

    return flux, ld, le, ed


@njit(cache=False, parallel=False, fastmath=True)
def eval_quad_z_s(z: float, k: float, u: ndarray):
    """Evaluates the transit model for scalar normalized distance.

    Parameters
    ----------
    z: float
        Normalized distance
    k: float
        Planet-star radius ratio
    u: 1D array
        Limb darkening coefficients

    Returns
    -------
    Transit model evaluated at `z`.

    """
    if abs(k - 0.5) < 1.0e-4:
        k = 0.5

    k2 = k ** 2
    omega = 1.0 - u[0] / 3.0 - u[1] / 6.0

    if abs(z - k) < 1e-6:
        z += 1e-6

    # The source is unocculted
    if z > 1.0 + k or (copysign(1, z) < 0.0):
        return 1.0

    # The source is completely occulted
    elif (k >= 1.0 and z <= k - 1.0):
        return 0.0

    z2 = z ** 2
    x1 = (k - z) ** 2
    x2 = (k + z) ** 2
    x3 = k ** 2 - z ** 2

    # LE
    # --
    # Case I: The occulting object fully inside the disk of the source
    if z <= 1.0 - k:
        le = k2

    # Case II: ingress and egress
    elif z >= abs(1.0 - k) and z <= 1.0 + k:
        kap1 = arccos(min((1.0 - k2 + z2) / (2.0 * z), 1.0))
        kap0 = arccos(min((k2 + z2 - 1.0) / (2.0 * k * z), 1.0))
        le = k2 * kap0 + kap1
        le = (le - 0.5 * sqrt(max(4.0 * z2 - (1.0 + z2 - k2) ** 2, 0.0))) * INV_PI

    # LD and ED
    # ---------
    is_edge_at_origin = abs(z - k) < 1.e-4 * (z + k)
    is_full_transit = k <= 1.0 and z < (1.0 - k)

    # Case 0: The edge of the occulting object lies at the origin
    if is_edge_at_origin:
        if (k == 0.5):
            ld = 1.0 / 3.0 - 4.0 * INV_PI / 9.0
            ed = 3.0 / 32.0

        elif (z > 0.5):
            q = 0.50 / k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = 1.0 / 3.0 + 16.0 * k / 9.0 * INV_PI * (2.0 * k2 - 1.0) * Ek - (
                        32.0 * k ** 4 - 20.0 * k2 + 3.0) / 9.0 * INV_PI / k * Kk
            ed = 1.0 / 2.0 * INV_PI * (kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt(
                (1.0 - x1) * (x2 - 1.0)))

        elif (z < 0.5):
            q = 2.0 * k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = 1.0 / 3.0 + 2.0 / 9.0 * INV_PI * (4.0 * (2.0 * k2 - 1.0) * Ek + (1.0 - 4.0 * k2) * Kk)
            ed = k2 / 2.0 * (k2 + 2.0 * z2)
    else:
        # Case I: The occulting object fully inside the disk of the source
        if is_full_transit:
            q = sqrt((x2 - x1) / (1.0 - x1))
            Kk = ellk(q)
            Ek = ellec(q)
            n = x2 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld = 2.0 / 9.0 * INV_PI / sqrt(1.0 - x1) * ((1.0 - 5.0 * z2 + k2 + x3 * x3) * Kk + (1.0 - x1) * (
                        z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld = ld + 2.0 / 3.0
            if (abs(k + z - 1.0) < 1.e-4):
                ld = 2.0 / 3.0 * INV_PI * arccos(1.0 - 2.0 * k) - 4.0 / 9.0 * INV_PI * sqrt(k * (1.0 - k)) * (
                            3.0 + 2.0 * k - 8.0 * k2)
            ed = k2 / 2.0 * (k2 + 2.0 * z2)

        # Case II: ingress and egress
        else:
            q = sqrt((1.0 - (k - z) ** 2) / 4.0 / z / k)
            Kk = ellk(q)
            Ek = ellec(q)
            n = 1.0 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld = 1.0 / 9.0 * INV_PI / sqrt(k * z) * (
                    ((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk + 4.0 * k * z * (
                    z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld = ld + 2.0 / 3.0
            ed = 1.0 / 2.0 * INV_PI * (kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt(
                (1.0 - x1) * (x2 - 1.0)))

    flux = 1.0 - ((1.0 - u[0] - 2.0 * u[1]) * le + (u[0] + 2.0 * u[1]) * ld + u[1] * ed) / omega
    return flux


@njit(cache=False, parallel=False, fastmath=True)
def eval_quad_ip(zs, k, u, c, edt, ldt, let, kt, zt):
    npb = u.shape[0]
    flux = zeros((len(zs), npb))
    omega = zeros(npb)
    dk = kt[1] - kt[0]
    dz = zt[1] - zt[0]

    for i in range(npb):
        omega[i] = 1.0 - u[i, 0] / 3.0 - u[i, 1] / 6.0

    ik = int(floor((k - kt[0]) / dk))
    ak1 = (k - kt[ik]) / dk
    ak2 = 1.0 - ak1

    ed2 = edt[ik:ik + 2, :]
    ld2 = ldt[ik:ik + 2, :]
    le2 = let[ik:ik + 2, :]

    for i in prange(len(zs)):
        z = zs[i]
        if (z >= 1.0 + k) or (copysign(1, z) < 0.0):
            flux[i, :] = 1.0
        else:
            iz = int(floor((z - zt[0]) / dz))
            az1 = (z - zt[iz]) / dz
            az2 = 1.0 - az1

            ed = (ed2[0, iz] * ak2 * az2
                  + ed2[1, iz] * ak1 * az2
                  + ed2[0, iz + 1] * ak2 * az1
                  + ed2[1, iz + 1] * ak1 * az1)

            ld = (ld2[0, iz] * ak2 * az2
                  + ld2[1, iz] * ak1 * az2
                  + ld2[0, iz + 1] * ak2 * az1
                  + ld2[1, iz + 1] * ak1 * az1)

            le = (le2[0, iz] * ak2 * az2
                  + le2[1, iz] * ak1 * az2
                  + le2[0, iz + 1] * ak2 * az1
                  + le2[1, iz + 1] * ak1 * az1)

            for j in range(npb):
                flux[i, j] = 1.0 - ((1.0 - u[j, 0] - 2.0 * u[j, 1]) * le + (u[j, 0] + 2.0 * u[j, 1]) * ld + u[j, 1] * ed) / omega[j]
                flux[i, j] = c[j] + (1.0 - c[j]) * flux[i, j]

    return flux


@njit(cache=False, parallel=False, fastmath=True)
def eval_quad_ip(zs, k, u, edt, ldt, let, kt, zt):
    npb = u.shape[0]
    flux = zeros((len(zs), npb))
    omega = zeros(npb)
    dk = kt[1] - kt[0]
    dz = zt[1] - zt[0]

    for i in range(npb):
        omega[i] = 1.0 - u[i, 0] / 3.0 - u[i, 1] / 6.0

    ik = int(floor((k - kt[0]) / dk))
    ak1 = (k - kt[ik]) / dk
    ak2 = 1.0 - ak1

    ed2 = edt[ik:ik + 2, :]
    ld2 = ldt[ik:ik + 2, :]
    le2 = let[ik:ik + 2, :]

    for i in prange(len(zs)):
        z = zs[i]
        if (z >= 1.0 + k) or (copysign(1, z) < 0.0):
            flux[i, :] = 1.0
        else:
            iz = int(floor((z - zt[0]) / dz))
            az1 = (z - zt[iz]) / dz
            az2 = 1.0 - az1

            ed = (ed2[0, iz] * ak2 * az2
                  + ed2[1, iz] * ak1 * az2
                  + ed2[0, iz + 1] * ak2 * az1
                  + ed2[1, iz + 1] * ak1 * az1)

            ld = (ld2[0, iz] * ak2 * az2
                  + ld2[1, iz] * ak1 * az2
                  + ld2[0, iz + 1] * ak2 * az1
                  + ld2[1, iz + 1] * ak1 * az1)

            le = (le2[0, iz] * ak2 * az2
                  + le2[1, iz] * ak1 * az2
                  + le2[0, iz + 1] * ak2 * az1
                  + le2[1, iz + 1] * ak1 * az1)

            for j in range(npb):
                flux[i, j] = 1.0 - ((1.0 - u[j, 0] - 2.0 * u[j, 1]) * le + (u[j, 0] + 2.0 * u[j, 1]) * ld + u[j, 1] * ed) / omega[j]

    return flux


@njit(cache=False, parallel=False, fastmath=True)
def eval_quad_ip_mp(zs, pbi, ks, u, edt, ldt, let, kt, zt):
    npb = u.shape[0]
    flux = zeros(zs.size)
    omega = zeros(npb)
    dk = kt[1] - kt[0]
    dz = zt[1] - zt[0]

    for i in range(npb):
        omega[i] = 1.0 - u[i, 0] / 3.0 - u[i, 1] / 6.0

    j = -1
    for i in prange(zs.size):
        k = ks[pbi[i]]
        z = zs[i]

        if pbi[i] != j:
            ik = int(floor((k - kt[0]) / dk))
            ak1 = (k - kt[ik]) / dk
            ak2 = 1.0 - ak1
            ed2 = edt[ik:ik + 2, :]
            ld2 = ldt[ik:ik + 2, :]
            le2 = let[ik:ik + 2, :]

        j = pbi[i]

        if (z >= 1.0 + k) or (copysign(1, z) < 0.0):
            flux[i] = 1.0
        else:
            iz = int(floor((z - zt[0]) / dz))
            az1 = (z - zt[iz]) / dz
            az2 = 1.0 - az1

            ed = (ed2[0, iz] * ak2 * az2
                  + ed2[1, iz] * ak1 * az2
                  + ed2[0, iz + 1] * ak2 * az1
                  + ed2[1, iz + 1] * ak1 * az1)

            ld = (ld2[0, iz] * ak2 * az2
                  + ld2[1, iz] * ak1 * az2
                  + ld2[0, iz + 1] * ak2 * az1
                  + ld2[1, iz + 1] * ak1 * az1)

            le = (le2[0, iz] * ak2 * az2
                  + le2[1, iz] * ak1 * az2
                  + le2[0, iz + 1] * ak2 * az1
                  + le2[1, iz + 1] * ak1 * az1)

            flux[i] = 1.0 - ((1.0 - u[j, 0] - 2.0 * u[j, 1]) * le + (u[j, 0] + 2.0 * u[j, 1]) * ld + u[j, 1] * ed) / omega[j]
    return flux


@njit(cache=False, fastmath=False)
def quadratic_interpolated_z_s(z, k, u, edt, ldt, let, kt, zt):
    dk = kt[1] - kt[0]
    dz = zt[1] - zt[0]

    omega = 1.0 - u[0] / 3.0 - u[1] / 6.0

    ik = int(floor((k - kt[0]) / dk))
    ak1 = (k - kt[ik]) / dk
    ak2 = 1.0 - ak1

    if (z >= 1.0 + k) or (copysign(1, z) < 0.0):
        flux = 1.0
    else:
        iz = int(floor((z - zt[0]) / dz))
        az1 = (z - zt[iz]) / dz
        az2 = 1.0 - az1
        ed = (edt[ik,     iz    ] * ak2 * az2
            + edt[ik + 1, iz    ] * ak1 * az2
            + edt[ik,     iz + 1] * ak2 * az1
            + edt[ik + 1, iz + 1] * ak1 * az1)

        ld = (ldt[ik,     iz    ] * ak2 * az2
            + ldt[ik + 1, iz    ] * ak1 * az2
            + ldt[ik,     iz + 1] * ak2 * az1
            + ldt[ik + 1, iz + 1] * ak1 * az1)

        le = (let[ik,     iz    ] * ak2 * az2
            + let[ik + 1, iz    ] * ak1 * az2
            + let[ik,     iz + 1] * ak2 * az1
            + let[ik + 1, iz + 1] * ak1 * az1)

        flux = 1.0 - ((1.0 - u[0] - 2.0 * u[1]) * le + (u[0] + 2.0 * u[1]) * ld + u[1] * ed) / omega

    return flux


# Quadratic model for vector parameters
# -------------------------------------
@njit(parallel=True, fastmath=False)
def quadratic_model_v(t, k, t0, p, a, i, e, w, ldc, lcids, pbids, epids, nsamples, exptimes, npb,  edt, ldt, let, kt, zt, interpolate):
    p, a, i, e, w = atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
    ldc = atleast_2d(ldc)

    npv = k.shape[0]
    nk = k.shape[1]
    npt = t.size

    if ldc.shape[1] != 2*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

    if epids.max() != t0.shape[1] - 1:
        raise ValueError("The number of transit centers must equal to the number of individual epoch IDs.")

    flux = zeros((npv, npt))
    for ipv in prange(npv):

        if interpolate and (any(k[ipv] < kt[0]) or any(k[ipv] > kt[-1])):
            flux[ipv, :] = inf
            continue

        if any(isnan(k[ipv])) or isnan(a[ipv]):
            flux[ipv, :] = inf
            continue

        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        half_window_width = fmax(0.125, (2.0 + k[0,0]) / vx)

        for j in range(npt):
            ilc = lcids[j]
            iep = epids[ilc]

            epoch = floor((t[j] - t0[ipv, iep] + 0.5 * p[ipv]) / p[ipv])
            tc = t[j] - (t0[ipv, iep] + epoch * p[ipv])

            if abs(tc) > half_window_width:
                flux[ipv, j] = 1.0
            else:
                ipb = pbids[ilc]

                if nk == 1:
                    _k = k[ipv, 0]
                else:
                    _k = k[ipv, ipb]

                ld = ldc[ipv, 2 * ipb:2 * (ipb + 1)]
                if isnan(ld[0]) or isnan(ld[1]):
                    flux[ipv, j] = inf
                else:
                    for isample in range(1, nsamples[ilc] + 1):
                        time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                        z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                        if z > 1.0 + _k:
                            flux[ipv, j] += 1.
                        else:
                            if interpolate:
                                flux[ipv, j] += quadratic_interpolated_z_s(z, _k, ld, edt, ldt, let, kt, zt)
                            else:
                                flux[ipv, j] += eval_quad_z_s(z, _k, ld)
                    flux[ipv, j] /= nsamples[ilc]
    return flux


# Quadratic model for scalar parameters
# -------------------------------------
@njit(parallel=False, fastmath=True)
def quadratic_model_s(t, k, t0, p, a, i, e, w, ldc, lcids, pbids, epids, nsamples, exptimes, npb, edt, ldt, let, kt, zt, interpolate):
    ldc = atleast_1d(ldc)
    k = atleast_1d(k)
    t0 = atleast_1d(t0)

    if ldc.size != 2*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

    if epids.max() != t0.size - 1:
        raise ValueError("The number of transit centers must equal to the number of individual epoch IDs.")

    y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
    half_window_width = fmax(0.125, (2.0 + k[0]) / vx)

    npt = t.size
    flux = zeros(npt)

    if interpolate and (any(k < kt[0]) or any(k > kt[-1])):
        flux[:] = inf
        return flux

    for j in range(npt):
        ilc = lcids[j]
        iep = epids[ilc]

        epoch = floor((t[j] - t0[iep] + 0.5 * p) / p)
        tc = t[j] - (t0[iep] + epoch * p)
        if abs(tc) > half_window_width:
            flux[j] = 1.0
        else:
            ipb = pbids[ilc]

            if k.size == 1:
                _k = k[0]
            else:
                _k = k[ipb]

            ld = ldc[2 * ipb:2 * (ipb + 1)]
            if isnan(_k) or isnan(a) or isnan(ld[0]) or isnan(ld[1]):
                flux[j] = inf

            else:
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                    if z > 1.0 + _k:
                        flux[j] += 1.
                    else:
                        if interpolate:
                            flux[j] += quadratic_interpolated_z_s(z, _k, ld, edt, ldt, let, kt, zt)
                        else:
                            flux[j] += eval_quad_z_s(z, _k, ld)
                flux[j] /= nsamples[ilc]
    return flux


# Quadratic model for parameter array
# -----------------------------------
@njit(parallel=True, fastmath=False)
def quadratic_model_pv(t, pvp, ldc, lcids, pbids, nsamples, exptimes, npb, edt, ldt, let, kt, zt, interpolate):
    pvp = atleast_2d(pvp)
    ldc = atleast_2d(ldc)

    if ldc.shape[1] != 2*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")
    if ldc.shape[0] != pvp.shape[0]:
        raise ValueError(
            'The parameter array and the limb darkening coefficient array have mismatching shapes. The first dimension must match.')

    npv = pvp.shape[0]
    nk = pvp.shape[1] - 6
    npt = t.size
    flux = zeros((npv, npt))

    for ipv in prange(npv):
        t0, p, a, i, e, w = pvp[ipv, nk:]
        y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)
        half_window_width = fmax(0.125, (2 + pvp[ipv, 0]) / vx)

        if interpolate and (any(pvp[ipv, :nk] < kt[0]) or any(pvp[ipv, :nk] > kt[-1])):
            flux[ipv, :] = inf
            continue

        if any(isnan(pvp[ipv, :nk])) or isnan(a):
            flux[ipv, :] = inf
            continue

        for j in prange(npt):
            epoch = floor((t[j] - t0 + 0.5 * p) / p)
            tc = t[j] - (t0 + epoch * p)
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

                ld = ldc[ipv, 2 * ipb:2 * (ipb + 1)]
                if isnan(ld[0]) or isnan(ld[1]):
                    flux[ipv, j] = inf
                    continue

                for isample in range(1, nsamples[ilc]+1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = z_taylor_st(tc + time_offset, y0, vx, vy, ax, ay, jx, jy, sx, sy)
                    if z > 1.0+k:
                        flux[ipv, j] += 1.
                    else:
                        if interpolate:
                            flux[ipv, j] += quadratic_interpolated_z_s(z, k, ld, edt, ldt, let, kt, zt)
                        else:
                            flux[ipv, j] += eval_quad_z_s(z, k, ld)
                flux[ipv, j] /= nsamples[ilc]
    return flux
