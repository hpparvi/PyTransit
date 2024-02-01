from math import fabs, floor
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, full, nan

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc
from meepmeep.utils import d_from_pkaiews

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .common import circle_circle_intersection_area_kite as ccia


@njit
def rr_simple(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
              parallelize: bool, nlc: int, npb: int, nep: int,
              lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
              ldp: ndarray, istar: ndarray,
              weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Simplified RoadRunner model for a single homogeneous light curve."""

    if parallelize:
        return rr_simple_parallel(times, k, t0, p, a, i, e, w, nlc, npb, nep,
                                  lcids, pbids, epids, nsamples, exptimes,
                                  ldp, istar, weights, dk, kmin, kmax, dg, z_edges)
    else:
        return rr_simple_serial(times, k, t0, p, a, i, e, w, nlc, npb, nep,
                                lcids, pbids, epids, nsamples, exptimes,
                                ldp, istar, weights, dk, kmin, kmax, dg, z_edges)


@njit(parallel=False, fastmath=False)
def rr_simple_serial(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
                     nlc: int, npb: int, nep: int,
                     lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
                     ldp: ndarray, istar: ndarray,
                     weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Simplified RoadRunner model for a single homogeneous light curve."""

    npt = times.size
    ng = weights.shape[1]
    _istar = istar[0, 0]

    ldm = zeros(ng)  # Limb darkening means
    xyc = zeros((2, 5))  # Taylor series coefficients for the (x, y) position

    if isnan(a) or (a <= 1.0) or (e < 0.0) or (isnan(ldp[0, 0])):
        return full(npt, nan)

    # ----------------------------------#
    # Calculate the limb darkening mean #
    # ----------------------------------#
    if kmin <= k <= kmax:
        ik = int(floor((k - kmin) / dk))
        ak = (k - kmin - ik * dk) / dk
        ldm[:] = (1.0 - ak) * dot(weights[ik], ldp[0, 0]) + ak * dot(weights[ik + 1], ldp[0, 0])
    else:
        _, _, wg = calculate_weights_2d(k, z_edges, ng)
        ldm[:] = dot(wg, ldp[0, 0])

    # -----------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit #
    # -----------------------------------------------------#
    xyc[:, :] = solve_xy_p5s(0.0, p, a, i, e, w)

    # --------------------------------#
    # Calculate the half-window width #
    # --------------------------------#
    hww = 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1, 14)
    hww = 0.0015 + exptimes + hww

    # --------------------------#
    # Calculate the light curve #
    # --------------------------#
    flux = zeros(npt)
    for ipt in prange(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if fabs(tc) > hww:
            flux[ipt] = 1.0
        else:
            for isample in range(1, nsamples + 1):
                time_offset = exptimes * ((isample - 0.5) / nsamples - 0.5)
                z = pd_t15sc(tc + time_offset, xyc)
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + k), dg, ldm)
                aplanet = ccia(1.0, k, z)[0]
                flux[ipt] += (_istar - iplanet * aplanet) / _istar
            flux[ipt] /= nsamples
    return flux


@njit(parallel=True, fastmath=False)
def rr_simple_parallel(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
                       nlc: int, npb: int, nep: int,
                       lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
                       ldp: ndarray, istar: ndarray,
                       weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Simplified RoadRunner model for a single homogeneous light curve."""

    npt = times.size
    ng = weights.shape[1]
    _istar = istar[0, 0]

    ldm = zeros(ng)  # Limb darkening means
    xyc = zeros((2, 5))  # Taylor series coefficients for the (x, y) position

    if isnan(a) or (a <= 1.0) or (e < 0.0) or (isnan(ldp[0, 0])):
        return full(npt, nan)

    # ----------------------------------#
    # Calculate the limb darkening mean #
    # ----------------------------------#
    if kmin <= k <= kmax:
        ik = int(floor((k - kmin) / dk))
        ak = (k - kmin - ik * dk) / dk
        ldm[:] = (1.0 - ak) * dot(weights[ik], ldp[0, 0]) + ak * dot(weights[ik + 1], ldp[0, 0])
    else:
        _, _, wg = calculate_weights_2d(k, z_edges, ng)
        ldm[:] = dot(wg, ldp[0, 0])

    # -----------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit #
    # -----------------------------------------------------#
    xyc[:, :] = solve_xy_p5s(0.0, p, a, i, e, w)

    # --------------------------------#
    # Calculate the half-window width #
    # --------------------------------#
    hww = 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1, 14)
    hww = 0.0015 + exptimes + hww

    # --------------------------#
    # Calculate the light curve #
    # --------------------------#
    flux = zeros(npt)
    for ipt in prange(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if fabs(tc) > hww:
            flux[ipt] = 1.0
        else:
            for isample in range(1, nsamples + 1):
                time_offset = exptimes * ((isample - 0.5) / nsamples - 0.5)
                z = pd_t15sc(tc + time_offset, xyc)
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + k), dg, ldm)
                aplanet = ccia(1.0, k, z)[0]
                flux[ipt] += (_istar - iplanet * aplanet) / _istar
            flux[ipt] /= nsamples
    return flux
