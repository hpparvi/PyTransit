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

from meepmeep.backends.numba.point2d import solve2d, sep_c, bounding_box

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


# The Mandel & Agol (2002) expressions are singular at three coincidences, where the general
# expressions divide zero by zero or take the logarithm of zero, and each has a closed form of its
# own. The closed form is used inside a window around the coincidence, and since it is exact only
# at the coincidence itself, the windows are as narrow as the arithmetic of the general expressions
# allows: the flux error they leave is below 0.01 ppm.

EDGE_TOL = 1e-8    #: |z - k| below which the occulter's edge is taken to pass through the disk centre
LIMB_TOL = 1e-10   #: |z - (1 - k)| below which the occulter is taken to touch the limb from the inside
HALF_TOL = 1e-10   #: |k - 0.5| below which an occulter with its edge at the disk centre also touches the limb


@njit(cache=False, fastmath=True)
def quad_terms(z: float, k: float):
    """The three terms of the Mandel & Agol (2002) quadratic model for a non-negative distance.

    Parameters
    ----------
    z: float
        Normalized distance, z >= 0
    k: float
        Planet-star radius ratio

    Returns
    -------
    le, ld, ed: float
        The uniform-disk term and the two limb darkening terms. All three are zero for an
        unocculted source and one for a completely occulted source.
    """
    # The source is unocculted
    if z > 1.0 + k:
        return 0.0, 0.0, 0.0

    # The source is completely occulted
    if k >= 1.0 and z <= k - 1.0:
        return 1.0, 1.0, 1.0

    k2 = k * k
    z2 = z * z
    x1 = (k - z) ** 2
    x2 = (k + z) ** 2
    x3 = k2 - z2

    # LE and ED
    # ---------
    # The occulting object is fully inside the disk of the source
    if z <= 1.0 - k:
        le = k2
        ed = 0.5 * k2 * (k2 + 2.0 * z2)

    # Ingress and egress
    else:
        kap1 = arccos(max(min((1.0 - k2 + z2) / (2.0 * z), 1.0), -1.0))
        kap0 = arccos(max(min((k2 + z2 - 1.0) / (2.0 * k * z), 1.0), -1.0))
        sq = sqrt(max((1.0 - x1) * (x2 - 1.0), 0.0))
        le = (k2 * kap0 + kap1 - 0.5 * sq) * INV_PI
        ed = 0.5 * INV_PI * (kap1 + k2 * (k2 + 2.0 * z2) * kap0 - 0.25 * (1.0 + 5.0 * k2 + z2) * sq)

    # LD
    # --
    # The edge of the occulting object lies at the origin
    if abs(z - k) < EDGE_TOL:
        if abs(k - 0.5) < HALF_TOL:
            ld = 1.0 / 3.0 - 4.0 * INV_PI / 9.0
        elif k > 0.5:
            q = 0.5 / k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = (1.0 / 3.0 + 16.0 * k / 9.0 * INV_PI * (2.0 * k2 - 1.0) * Ek
                  - (32.0 * k2 * k2 - 20.0 * k2 + 3.0) / 9.0 * INV_PI / k * Kk)
        else:
            q = 2.0 * k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = 1.0 / 3.0 + 2.0 / 9.0 * INV_PI * (4.0 * (2.0 * k2 - 1.0) * Ek + (1.0 - 4.0 * k2) * Kk)

    # The occulting object touches the limb of the source from the inside
    elif abs(z + k - 1.0) < LIMB_TOL:
        ld = (2.0 / 3.0 * INV_PI * arccos(max(1.0 - 2.0 * k, -1.0))
              - 4.0 / 9.0 * INV_PI * sqrt(max(k * (1.0 - k), 0.0)) * (3.0 + 2.0 * k - 8.0 * k2))

    # The occulting object is fully inside the disk of the source
    elif z < 1.0 - k:
        q = sqrt((x2 - x1) / (1.0 - x1))
        Kk = ellk(q)
        Ek = ellec(q)
        Pk = ellpicb(x2 / x1 - 1.0, q)
        ld = 2.0 / 9.0 * INV_PI / sqrt(1.0 - x1) * ((1.0 - 5.0 * z2 + k2 + x3 * x3) * Kk
                                                    + (1.0 - x1) * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
        if z < k:
            ld += 2.0 / 3.0

    # Ingress and egress
    else:
        q = sqrt((1.0 - x1) / (4.0 * z * k))
        Kk = ellk(q)
        Ek = ellec(q)
        Pk = ellpicb(1.0 / x1 - 1.0, q)
        ld = 1.0 / 9.0 * INV_PI / sqrt(k * z) * (((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk
                                                 + 4.0 * k * z * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
        if z < k:
            ld += 2.0 / 3.0

    return le, ld, ed


@njit(cache=False, fastmath=True)
def quad_flux(z: float, k: float, u0: float, u1: float) -> float:
    """The quadratic transit model for a scalar distance and the two limb darkening coefficients.

    A negative distance, including a negative zero, marks an unocculted source.
    """
    if z > 1.0 + k or (copysign(1.0, z) < 0.0):
        return 1.0
    if k >= 1.0 and z <= k - 1.0:
        return 0.0
    le, ld, ed = quad_terms(z, k)
    omega = 1.0 - u0 / 3.0 - u1 / 6.0
    return 1.0 - ((1.0 - u0 - 2.0 * u1) * le + (u0 + 2.0 * u1) * ld + u1 * ed) / omega


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
    npt = len(zs)
    npb = u.shape[0]

    omega = zeros(npb)
    flux = zeros((npt, npb))
    le = zeros(npt)
    ld = zeros(npt)
    ed = zeros(npt)

    for i in range(npb):
        omega[i] = 1.0 - u[i,0] / 3.0 - u[i,1] / 6.0

    for i in prange(npt):
        z = zs[i]

        # The source is unocculted
        if z > 1.0 + k or z < 0.0:
            flux[i, :] = 1.0
            continue

        le[i], ld[i], ed[i] = quad_terms(z, k)

        # The source is completely occulted
        if k >= 1.0 and z <= k - 1.0:
            flux[i, :] = 0.0
            continue

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
    return quad_flux(z, k, u[0], u[1])


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

        c = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        bt1, bt4 = bounding_box(k[ipv, 0], c)
        bt1 -= 0.025
        bt4 += 0.025
        inv_p = 1.0 / p[ipv]

        for j in range(npt):
            ilc = lcids[j]
            iep = epids[ilc]

            epoch = floor((t[j] - t0[ipv, iep] + 0.5 * p[ipv]) * inv_p)
            tc = t[j] - (t0[ipv, iep] + epoch * p[ipv])

            if not (bt1 <= tc <= bt4):
                flux[ipv, j] = 1.0
            else:
                ipb = pbids[ilc]

                if nk == 1:
                    _k = k[ipv, 0]
                else:
                    _k = k[ipv, ipb]

                u0 = ldc[ipv, 2 * ipb]
                u1 = ldc[ipv, 2 * ipb + 1]
                if isnan(u0) or isnan(u1):
                    flux[ipv, j] = inf
                elif nsamples[ilc] == 1 and not interpolate:
                    z = sep_c(tc, c)
                    flux[ipv, j] = 1.0 if z > 1.0 + _k else quad_flux(z, _k, u0, u1)
                else:
                    ld = ldc[ipv, 2 * ipb:2 * (ipb + 1)]
                    inv_ns = 1.0 / nsamples[ilc]
                    for isample in range(1, nsamples[ilc] + 1):
                        time_offset = exptimes[ilc] * ((isample - 0.5) * inv_ns - 0.5)
                        z = sep_c(tc + time_offset, c)
                        if z > 1.0 + _k:
                            flux[ipv, j] += 1.
                        else:
                            if interpolate:
                                flux[ipv, j] += quadratic_interpolated_z_s(z, _k, ld, edt, ldt, let, kt, zt)
                            else:
                                flux[ipv, j] += quad_flux(z, _k, u0, u1)
                    flux[ipv, j] *= inv_ns
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

    c = solve2d(0.0, p, a, i, e, w)
    bt1, bt4 = bounding_box(k[0], c)
    bt1 -= 0.025
    bt4 += 0.025
    inv_p = 1.0 / p

    npt = t.size
    flux = zeros(npt)

    if interpolate and (any(k < kt[0]) or any(k > kt[-1])):
        flux[:] = inf
        return flux

    for j in range(npt):
        ilc = lcids[j]
        iep = epids[ilc]

        epoch = floor((t[j] - t0[iep] + 0.5 * p) * inv_p)
        tc = t[j] - (t0[iep] + epoch * p)
        if not (bt1 <= tc <= bt4):
            flux[j] = 1.0
        else:
            ipb = pbids[ilc]

            if k.size == 1:
                _k = k[0]
            else:
                _k = k[ipb]

            u0 = ldc[2 * ipb]
            u1 = ldc[2 * ipb + 1]
            if isnan(_k) or isnan(a) or isnan(u0) or isnan(u1):
                flux[j] = inf

            elif nsamples[ilc] == 1 and not interpolate:
                z = sep_c(tc, c)
                flux[j] = 1.0 if z > 1.0 + _k else quad_flux(z, _k, u0, u1)

            else:
                ld = ldc[2 * ipb:2 * (ipb + 1)]
                inv_ns = 1.0 / nsamples[ilc]
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) * inv_ns - 0.5)
                    z = sep_c(tc + time_offset, c)
                    if z > 1.0 + _k:
                        flux[j] += 1.
                    else:
                        if interpolate:
                            flux[j] += quadratic_interpolated_z_s(z, _k, ld, edt, ldt, let, kt, zt)
                        else:
                            flux[j] += quad_flux(z, _k, u0, u1)
                flux[j] *= inv_ns
    return flux
