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
from typing import Tuple

from numba import njit, prange, get_num_threads
from numpy import pi, sqrt, arccos, abs, zeros, linspace, array, atleast_2d, floor, inf, isnan, atleast_1d, ndarray, copysign, \
    any, full

from meepmeep.backends.numba.point2d import solve2d, sep_c, bounding_box

HALF_PI = 0.5 * pi
INV_PI = 1 / pi

# Complete elliptic integrals
# ---------------------------
# The complete elliptic integrals of the first, second and third kind are all special cases of
# Bulirsch's general complete integral (Bulirsch 1965, Numerische Mathematik 7, 78 and 353; 1969,
# Numerische Mathematik 13, 305),
#
#     cel(kc, p, a, b) = int_0^(pi/2) (a cos^2 x + b sin^2 x) / (cos^2 x + p sin^2 x)
#                                     / sqrt(cos^2 x + kc^2 sin^2 x) dx,
#
# with kc^2 = 1 - k^2: K(k) = cel(kc, 1, 1, 1), E(k) = cel(kc, 1, 1, kc^2) and
# Pi(n, k) = cel(kc, 1 + n, 1, 1). The iteration that evaluates cel follows the arithmetic-geometric
# mean sequence of (1, kc), which depends on k alone, so the integrals a model term needs are
# iterated together and share the sequence and its square roots. The convergence is quadratic and
# the results are accurate to a few times the machine precision.

ELL_TOL = 1e-8     #: Relative convergence limit of the mean sequence, about the square root of the accuracy reached
ELL_MAXITER = 40   #: The sequence converges in four to six steps unless k is within rounding of one


@njit(cache=False, fastmath=True)
def ellke(k: float):
    """Complete elliptic integrals of the first and second kind, K(k) and E(k).

    Both diverge or lose their meaning at k = 1; the iteration then stops at `ELL_MAXITER` and
    returns its last, finite, iterate.
    """
    kc2 = 1.0 - k * k
    kc = sqrt(kc2)
    e = kc
    em = 1.0
    q = 1.0
    a2 = 1.0
    b2 = 1.0
    a3 = 1.0
    b3 = kc2
    for nit in range(ELL_MAXITER):
        iq = 1.0 / q
        g2 = e * iq
        f = a2
        a2 += b2 * iq
        b2 = 2.0 * (b2 + f * g2)
        f = a3
        a3 += b3 * iq
        b3 = 2.0 * (b3 + f * g2)
        q += g2
        g = em
        em += kc
        if abs(g - kc) > ELL_TOL * g:
            kc = 2.0 * sqrt(e)
            e = kc * em
        else:
            break
    s = HALF_PI / (em * (em + q))
    return s * (b2 + a2 * em), s * (b3 + a3 * em)


@njit(cache=False, fastmath=True)
def ellkepi(n: float, k: float):
    """Complete elliptic integrals of the first, second and third kind, K(k), E(k) and Pi(n, k).

    The integral of the third kind follows the sign convention of Mandel & Agol (2002),
    ``Pi(n, k) = int_0^(pi/2) dx / ((1 + n sin^2 x) sqrt(1 - k^2 sin^2 x))``, and needs n > -1.
    See `ellke` for the behaviour at k = 1.
    """
    kc2 = 1.0 - k * k
    kc = sqrt(kc2)
    e = kc
    em = 1.0
    p = sqrt(1.0 + n)
    a1 = 1.0
    b1 = 1.0 / p
    q = 1.0
    a2 = 1.0
    b2 = 1.0
    a3 = 1.0
    b3 = kc2
    for nit in range(ELL_MAXITER):
        ip = 1.0 / p
        iq = 1.0 / q
        g1 = e * ip
        g2 = e * iq
        f = a1
        a1 += b1 * ip
        b1 = 2.0 * (b1 + f * g1)
        p += g1
        f = a2
        a2 += b2 * iq
        b2 = 2.0 * (b2 + f * g2)
        f = a3
        a3 += b3 * iq
        b3 = 2.0 * (b3 + f * g2)
        q += g2
        g = em
        em += kc
        if abs(g - kc) > ELL_TOL * g:
            kc = 2.0 * sqrt(e)
            e = kc * em
        else:
            break
    s = HALF_PI / em
    sq = s / (em + q)
    return sq * (b2 + a2 * em), sq * (b3 + a3 * em), s * (b1 + a1 * em) / (em + p)


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
            Kk, Ek = ellke(0.5 / k)
            ld = (1.0 / 3.0 + 16.0 * k / 9.0 * INV_PI * (2.0 * k2 - 1.0) * Ek
                  - (32.0 * k2 * k2 - 20.0 * k2 + 3.0) / 9.0 * INV_PI / k * Kk)
        else:
            Kk, Ek = ellke(2.0 * k)
            ld = 1.0 / 3.0 + 2.0 / 9.0 * INV_PI * (4.0 * (2.0 * k2 - 1.0) * Ek + (1.0 - 4.0 * k2) * Kk)

    # The occulting object touches the limb of the source from the inside
    elif abs(z + k - 1.0) < LIMB_TOL:
        ld = (2.0 / 3.0 * INV_PI * arccos(max(1.0 - 2.0 * k, -1.0))
              - 4.0 / 9.0 * INV_PI * sqrt(max(k * (1.0 - k), 0.0)) * (3.0 + 2.0 * k - 8.0 * k2))

    # The occulting object is fully inside the disk of the source
    elif z < 1.0 - k:
        ix1 = 1.0 / x1
        rs = 1.0 / sqrt(1.0 - x1)
        Kk, Ek, Pk = ellkepi(x2 * ix1 - 1.0, sqrt(x2 - x1) * rs)
        ld = 2.0 / 9.0 * INV_PI * rs * ((1.0 - 5.0 * z2 + k2 + x3 * x3) * Kk
                                        + (1.0 - x1) * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 * ix1 * Pk)
        if z < k:
            ld += 2.0 / 3.0

    # Ingress and egress
    else:
        ix1 = 1.0 / x1
        rkz = 1.0 / sqrt(k * z)
        Kk, Ek, Pk = ellkepi(ix1 - 1.0, 0.5 * sqrt(1.0 - x1) * rkz)
        ld = 1.0 / 9.0 * INV_PI * rkz * (((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk
                                         + 4.0 * k * z * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 * ix1 * Pk)
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

    for i in range(npt):
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


# Interpolated model
# ------------------
# The three model terms are tabulated on a regular (k, z) grid and interpolated bilinearly. The
# position of a radius ratio in the grid is the same for every point of a light curve, so it is
# found once per passband with `interpolation_weights`.

@njit(cache=False)
def interpolation_weights(k: float, kt: ndarray):
    """The index of the radius ratio table interval holding `k` and the position of `k` inside it."""
    x = (k - kt[0]) / (kt[1] - kt[0])
    ik = min(max(int(floor(x)), 0), kt.size - 2)
    return ik, x - ik


@njit(cache=False, fastmath=False)
def quadratic_interpolated_z_c(z, k, ik, ak1, cle, cld, ced, edt, ldt, let, z0, inv_dz):
    """The interpolated model for a scalar distance, a located radius ratio and the weights of `flux_coefficients`."""
    if (z >= 1.0 + k) or (copysign(1, z) < 0.0):
        return 1.0

    ak2 = 1.0 - ak1
    x = (z - z0) * inv_dz
    iz = int(floor(x))
    az1 = x - iz
    az2 = 1.0 - az1
    w00 = ak2 * az2
    w10 = ak1 * az2
    w01 = ak2 * az1
    w11 = ak1 * az1
    ed = edt[ik, iz] * w00 + edt[ik + 1, iz] * w10 + edt[ik, iz + 1] * w01 + edt[ik + 1, iz + 1] * w11
    ld = ldt[ik, iz] * w00 + ldt[ik + 1, iz] * w10 + ldt[ik, iz + 1] * w01 + ldt[ik + 1, iz + 1] * w11
    le = let[ik, iz] * w00 + let[ik + 1, iz] * w10 + let[ik, iz + 1] * w01 + let[ik + 1, iz + 1] * w11
    return 1.0 - (cle * le + cld * ld + ced * ed)


@njit(cache=False, fastmath=False)
def quadratic_interpolated_z_s(z, k, u, edt, ldt, let, kt, zt):
    """The interpolated model for a scalar distance and a pair of limb darkening coefficients."""
    ik, ak1 = interpolation_weights(k, kt)
    inv_omega = 1.0 / (1.0 - u[0] / 3.0 - u[1] / 6.0)
    return quadratic_interpolated_z_c(z, k, ik, ak1, (1.0 - u[0] - 2.0 * u[1]) * inv_omega, (u[0] + 2.0 * u[1]) * inv_omega,
                                      u[1] * inv_omega, edt, ldt, let, zt[0], 1.0 / (zt[1] - zt[0]))


@njit(cache=False)
def passband_interpolation_weights(k: ndarray, npb: int, kt: ndarray, interpolate: bool) -> ndarray:
    """`interpolation_weights` for the radius ratio of each passband, as an ``(npb, 2)`` array."""
    kw = zeros((npb, 2))
    if interpolate:
        for ipb in range(npb):
            _k = k[0] if k.size == 1 else k[ipb]
            if not isnan(_k):
                kw[ipb, 0], kw[ipb, 1] = interpolation_weights(_k, kt)
    return kw


# Light curve kernels
# -------------------
# Everything that depends only on the parameters or on the light curve setup is computed once per
# call: the limb darkening enters through three coefficients per passband, and the supersampling
# through a step, a first offset and a reciprocal per light curve.

#: Margin added to both sides of the transit bounding box [d], on top of half the longest exposure.
BBOX_MARGIN = 0.003


@njit(cache=False)
def flux_coefficients(ldc: ndarray, npb: int) -> ndarray:
    """The weights of the three model terms for each passband, as an ``(npb, 3)`` array.

    The flux is ``1 - (c[0] * le + c[1] * ld + c[2] * ed)``. A NaN limb darkening coefficient
    gives NaN weights.
    """
    cf = zeros((npb, 3))
    for ipb in range(npb):
        u0 = ldc[2 * ipb]
        u1 = ldc[2 * ipb + 1]
        inv_omega = 1.0 / (1.0 - u0 / 3.0 - u1 / 6.0)
        cf[ipb, 0] = (1.0 - u0 - 2.0 * u1) * inv_omega
        cf[ipb, 1] = (u0 + 2.0 * u1) * inv_omega
        cf[ipb, 2] = u1 * inv_omega
    return cf


@njit(cache=False)
def supersampling_steps(nsamples: ndarray, exptimes: ndarray) -> ndarray:
    """The supersample step, the offset of the first supersample and 1 / nsamples for each light curve."""
    ss = zeros((nsamples.size, 3))
    for ilc in range(nsamples.size):
        ss[ilc, 2] = 1.0 / nsamples[ilc]
        ss[ilc, 0] = exptimes[ilc] * ss[ilc, 2]
        ss[ilc, 1] = 0.5 * (ss[ilc, 0] - exptimes[ilc])
    return ss


@njit(cache=False)
def bounding_box_padding(nsamples: ndarray, exptimes: ndarray) -> float:
    """How far outside the transit an exposure can be centred and still reach into it [d]."""
    pad = 0.0
    for ilc in range(nsamples.size):
        if nsamples[ilc] > 1:
            pad = max(pad, 0.5 * exptimes[ilc])
    return BBOX_MARGIN + pad


@njit(cache=False)
def largest_radius_ratio(k: ndarray) -> float:
    """The largest radius ratio, ignoring NaNs: the one that sets the transit bounding box."""
    kmax = 0.0
    for i in range(k.size):
        if k[i] > kmax:
            kmax = k[i]
    return kmax


@njit(cache=False, fastmath=True)
def quad_flux_c(z: float, k: float, cle: float, cld: float, ced: float) -> float:
    """The quadratic transit model for a non-negative distance and the weights of `flux_coefficients`."""
    if k >= 1.0 and z <= k - 1.0:
        return 0.0
    le, ld, ed = quad_terms(z, k)
    return 1.0 - (cle * le + cld * ld + ced * ed)


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

    ss = supersampling_steps(nsamples, exptimes)
    pad = bounding_box_padding(nsamples, exptimes)
    z0 = zt[0]
    inv_dz = 1.0 / (zt[1] - zt[0]) if interpolate else 0.0

    flux = zeros((npv, npt))
    for ipv in prange(npv):

        if interpolate and (any(k[ipv] < kt[0]) or any(k[ipv] > kt[-1])):
            flux[ipv, :] = inf
            continue

        if any(isnan(k[ipv])) or isnan(p[ipv]) or isnan(a[ipv]) or isnan(i[ipv]) or isnan(e[ipv]) or isnan(w[ipv]):
            flux[ipv, :] = inf
            continue

        c = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        bt1, bt4 = bounding_box(largest_radius_ratio(k[ipv]), c)
        bt1 -= pad
        bt4 += pad
        inv_p = 1.0 / p[ipv]
        cf = flux_coefficients(ldc[ipv], npb)
        kw = passband_interpolation_weights(k[ipv], npb, kt, interpolate)

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

                cle, cld, ced = cf[ipb, 0], cf[ipb, 1], cf[ipb, 2]
                if isnan(cle):
                    flux[ipv, j] = inf
                elif nsamples[ilc] == 1 and not interpolate:
                    z = sep_c(tc, c)
                    flux[ipv, j] = 1.0 if z > 1.0 + _k else quad_flux_c(z, _k, cle, cld, ced)
                else:
                    ik, ak1 = int(kw[ipb, 0]), kw[ipb, 1]
                    dt, t_first = ss[ilc, 0], ss[ilc, 1]
                    fsum = 0.0
                    for isample in range(nsamples[ilc]):
                        z = sep_c(tc + t_first + isample * dt, c)
                        if z > 1.0 + _k:
                            fsum += 1.0
                        elif interpolate:
                            fsum += quadratic_interpolated_z_c(z, _k, ik, ak1, cle, cld, ced, edt, ldt, let, z0, inv_dz)
                        else:
                            fsum += quad_flux_c(z, _k, cle, cld, ced)
                    flux[ipv, j] = fsum * ss[ilc, 2]
    return flux


# Quadratic model for scalar parameters
# -------------------------------------
# The parallel version deals the light curve out to the threads in blocks of `PARALLEL_BLOCK`
# points, thread i taking blocks i, i + nthreads, ... A contiguous chunk per thread, which is what
# a plain prange over the points gives, leaves most threads with only out-of-transit points, since
# the expensive in-transit points lie next to each other.

PARALLEL_BLOCK = 64       #: Number of consecutive points a thread evaluates before skipping to its next block
PARALLEL_MIN_WORK = 600   #: Estimated work below which the parallel version runs serially, in in-transit model evaluations
PARALLEL_PROBES = 256     #: Number of points probed to estimate the work


# Numba turns whole-array expressions inside a parallel function into parallel loops of their own,
# each with the cost of starting the threads, so the argument checks live in serial helpers.

@njit(cache=False)
def parallel_block_layout(npt: int, parallel: bool):
    """The block size, the number of blocks and the number of threads for a light curve of npt points."""
    if not parallel:
        return max(npt, 1), 1, 1
    nblocks = (npt + PARALLEL_BLOCK - 1) // PARALLEL_BLOCK
    return PARALLEL_BLOCK, nblocks, min(get_num_threads(), nblocks)


@njit(cache=False)
def too_short_for_threads(t: ndarray, t0: ndarray, p: float, bt1: float, bt4: float, lcids: ndarray, epids: ndarray,
                          nsamples: ndarray) -> bool:
    """True for a light curve that is faster to evaluate serially than to start the threads for.

    Starting the threads costs about as much as a few hundred in-transit model evaluations. The
    number of evaluations is estimated from every n-th point, counting an out-of-transit point
    as a sixteenth of an evaluation.
    """
    npt = t.size
    if npt >= 16 * PARALLEL_MIN_WORK:
        return False
    if npt * nsamples.max() < PARALLEL_MIN_WORK:
        return True
    step = max(1, npt // PARALLEL_PROBES)
    inv_p = 1.0 / p
    work = 0.0
    for j in range(0, npt, step):
        ilc = lcids[j]
        tz = t0[epids[ilc]]
        tc = t[j] - (tz + floor((t[j] - tz + 0.5 * p) * inv_p) * p)
        if bt1 <= tc <= bt4:
            work += nsamples[ilc]
        else:
            work += 0.0625
    return work * step < PARALLEL_MIN_WORK


@njit(cache=False)
def check_quadratic_arguments(ldc: ndarray, t0: ndarray, epids: ndarray, npb: int):
    if ldc.size != 2*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")
    if epids.max() != t0.size - 1:
        raise ValueError("The number of transit centers must equal to the number of individual epoch IDs.")


@njit(cache=False)
def infinite_flux(npt: int) -> ndarray:
    """The light curve of a parameter vector the model cannot be evaluated for."""
    return full(npt, inf)


@njit(cache=False)
def outside_table(k: ndarray, kt: ndarray) -> bool:
    """True if any radius ratio falls outside the interpolation table."""
    for i in range(k.size):
        if k[i] < kt[0] or k[i] > kt[-1]:
            return True
    return False


def _create_quadratic_model_s(parallel: bool, serial_model=None):
    def quadratic_model_s(t, k, t0, p, a, i, e, w, ldc, lcids, pbids, epids, nsamples, exptimes, npb, edt, ldt, let, kt, zt, interpolate):
        """The quadratic model for a single parameter vector.

        Compiled both in serial and in parallel; in the parallel version the time samples are
        distributed over the numba threads in interleaved blocks.
        """
        ldc = atleast_1d(ldc)
        k = atleast_1d(k)
        t0 = atleast_1d(t0)

        npt = t.size
        check_quadratic_arguments(ldc, t0, epids, npb)

        # A NaN orbital parameter cannot be solved for an orbit, and the table covers only its radius ratios.
        if isnan(p) or isnan(a) or isnan(i) or isnan(e) or isnan(w) or (interpolate and outside_table(k, kt)):
            return infinite_flux(npt)

        c = solve2d(0.0, p, a, i, e, w)
        pad = bounding_box_padding(nsamples, exptimes)
        bt1, bt4 = bounding_box(largest_radius_ratio(k), c)
        bt1 -= pad
        bt4 += pad

        if parallel and too_short_for_threads(t, t0, p, bt1, bt4, lcids, epids, nsamples):
            return serial_model(t, k, t0, p, a, i, e, w, ldc, lcids, pbids, epids, nsamples, exptimes, npb, edt, ldt, let, kt, zt, interpolate)

        inv_p = 1.0 / p
        cf = flux_coefficients(ldc, npb)
        ss = supersampling_steps(nsamples, exptimes)
        kw = passband_interpolation_weights(k, npb, kt, interpolate)
        z0 = zt[0]
        inv_dz = 1.0 / (zt[1] - zt[0]) if interpolate else 0.0

        flux = zeros(npt)

        block, nblocks, nthreads = parallel_block_layout(npt, parallel)
        for ithread in prange(nthreads):
            for iblock in range(ithread, nblocks, nthreads):
                for j in range(iblock * block, min((iblock + 1) * block, npt)):
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

                        cle, cld, ced = cf[ipb, 0], cf[ipb, 1], cf[ipb, 2]
                        if isnan(_k) or isnan(cle):
                            flux[j] = inf

                        elif nsamples[ilc] == 1 and not interpolate:
                            z = sep_c(tc, c)
                            flux[j] = 1.0 if z > 1.0 + _k else quad_flux_c(z, _k, cle, cld, ced)

                        else:
                            ik, ak1 = int(kw[ipb, 0]), kw[ipb, 1]
                            dt, t_first = ss[ilc, 0], ss[ilc, 1]
                            fsum = 0.0
                            for isample in range(nsamples[ilc]):
                                z = sep_c(tc + t_first + isample * dt, c)
                                if z > 1.0 + _k:
                                    fsum += 1.0
                                elif interpolate:
                                    fsum += quadratic_interpolated_z_c(z, _k, ik, ak1, cle, cld, ced, edt, ldt, let, z0, inv_dz)
                                else:
                                    fsum += quad_flux_c(z, _k, cle, cld, ced)
                            flux[j] = fsum * ss[ilc, 2]
        return flux
    return njit(parallel=parallel, fastmath=False)(quadratic_model_s)


quadratic_model_s = _create_quadratic_model_s(False)
quadratic_model_s_parallel = _create_quadratic_model_s(True, quadratic_model_s)
