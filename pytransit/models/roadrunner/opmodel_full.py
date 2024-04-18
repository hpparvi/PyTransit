from math import fabs, floor, sqrt
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, nan, ones, full, linspace, squeeze, atleast_2d, atleast_1d

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc, xy_t15sc
from meepmeep.utils import d_from_pkaiews

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .ecintersection import create_ellipse, ellipse_circle_intersection_area as ecia

def opmodel(times, k, f, alpha, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep, npl,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, weights, dk, kmin, kmax, dg, z_edges):

    k, f, alpha, t0, p, a, i, e, w = (atleast_2d(k), atleast_1d(f), atleast_1d(alpha), atleast_2d(t0), atleast_1d(p),
                                      atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w))

    return squeeze(op_full(times, k, f, alpha, t0, p, a, i, e, w, parallelize, nlc, npb, nep, npl,
                   lcids, pbids, epids, nsamples, exptimes,
                   ldp, istar, weights, dk, kmin, kmax, dg, z_edges))


@njit
def op_full(times: ndarray, k: ndarray, f: ndarray, alpha: ndarray,
            t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int, npl: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Full RoadRunner model for heterogeneous light curves."""

    #if parallelize:
    #    return op_full_parallel(times, k, f, alpha, t0, p, a, i, e, w, nlc, npb, nep, npl,
    #                          lcids, pbids, epids, nsamples, exptimes,
    #                          ldp, istar, weights, dk, kmin, kmax, dg, z_edges)
    #else:
    return op_full_serial(times, k, f, alpha, t0, p, a, i, e, w, nlc, npb, nep, npl,
                          lcids, pbids, epids, nsamples, exptimes,
                          ldp, istar, weights, dk, kmin, kmax, dg, z_edges)


@njit(parallel=False, fastmath=False)
def op_full_serial(times: ndarray, k: ndarray, f: ndarray, alpha: ndarray,
                   t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            nlc: int, npb: int, nep: int, npl: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Full RoadRunner model for heterogeneous light curves."""
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]

    if k.shape[1] > 1 and k.shape[1] != npb:
        raise ValueError('Radius ratios should be given either as an [npv, 1] or [npv, npb] array.')

    _exptimes = zeros(nlc)
    _exptimes[:] = exptimes
    _nsamples = zeros(nlc)
    _nsamples[:] = nsamples

    # Copy the radius ratios
    # ----------------------
    if k.shape[0] == npv and k.shape[1] == npb:
        ks = k
    else:
        ks = zeros((npv, npb))
        ks[:, :] = k[:, 0:npb]

    pv_is_good = full(npv, True)
    ldm = zeros((npv, npb, ng))  # Limb darkening means
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    hwws = zeros((npv, npb))     # Half-window widths [d]

    exs = zeros((npv, npl, 2))   # Ellipse model scanline x-coordinates
    eys = zeros((npv, npl))      # Elilpse model scanline y-coordinates

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0) or (isnan(ldp[ipv, 0, 0])):
            pv_is_good[ipv] = False
            continue

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        if kmin <= ks[ipv, 0] <= kmax:
            ik = int(floor((ks[ipv, 0] - kmin) / dk))
            ak = (ks[ipv, 0] - kmin - ik * dk) / dk
            for ipb in range(npb):
                ldm[ipv, ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb]) + ak * dot(weights[ik + 1], ldp[ipv, ipb])
        else:
            _, _, wg = calculate_weights_2d(ks[ipv, 0], z_edges, ng)
            for ipb in range(npb):
                ldm[ipv, ipb, :] = dot(wg, ldp[ipv, ipb])

        # ------------------------------------------------------#
        # Calculate the Taylor series expansions for the orbits #
        # ------------------------------------------------------#
        xyc[ipv, :, :] = solve_xy_p5s(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # ---------------------------------#
        # Calculate the half-window widths #
        # ---------------------------------#
        hww = 0.5 * d_from_pkaiews(p[ipv], ks[ipv, 0], a[ipv], i[ipv], e[ipv], w[ipv], 1, 14)
        for ilc in range(nlc):
            hwws[ipv, ilc] = 0.0015 + _exptimes[ilc] + hww

        # ----------------------------------
        # Create the ellipse (x, y) points #
        # ----------------------------------
        _y, _x = create_ellipse(npl, ks[ipv,0], f[ipv], alpha[ipv])
        exs[ipv, :, :] = _x
        eys[ipv, :] = _y

    # ---------------------------#
    # Calculate the light curves #
    # ---------------------------#
    flux = zeros((npv, npt))
    for j in prange(npv * npt):
        ipv = j // npt
        ipt = j % npt

        if not pv_is_good[ipv]:
            flux[ipv, ipt] = nan
            continue

        ilc = lcids[ipt]
        ipb = pbids[ilc]
        iep = epids[ilc]

        epoch = floor((times[ipt] - t0[ipv, iep] + 0.5 * p[ipv]) / p[ipv])
        tc = times[ipt] - (t0[ipv, iep] + epoch * p[ipv])
        if fabs(tc) > hwws[ipv, ilc]:
            flux[ipv, ipt] = 1.0
        else:
            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                cx, cy = xy_t15sc(tc + time_offset, xyc[ipv])
                z = sqrt(cx*cx + cy*cy)
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + ks[ipv, ipb]), dg, ldm[ipv, ipb])
                aplanet = ecia(cx, cy, z, ks[ipv, ipb], f[ipv], exs[ipv,:,:], eys[ipv,:])
                flux[ipv, ipt] += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
            flux[ipv, ipt] /= nsamples[ilc]
    return flux

# @njit(parallel=True, fastmath=False)
# def op_full_parallel(times: ndarray, k: ndarray, f: ndarray, alpha: ndarray,
#                    t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
#             nlc: int, npb: int, nep: int, npl: int,
#             lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
#             ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
#     """Full RoadRunner model for heterogeneous light curves."""
#     npv = k.shape[0]
#     npt = times.size
#     ng = weights.shape[1]
#
#     if k.shape[1] > 1 and k.shape[1] != npb:
#         raise ValueError('Radius ratios should be given either as an [npv, 1] or [npv, npb] array.')
#
#     _exptimes = zeros(nlc)
#     _exptimes[:] = exptimes
#     _nsamples = zeros(nlc)
#     _nsamples[:] = nsamples
#
#     # Copy the radius ratios
#     # ----------------------
#     if k.shape[0] == npv and k.shape[1] == npb:
#         ks = k
#     else:
#         ks = zeros((npv, npb))
#         ks[:, :] = k[:, 0:npb]
#
#     pv_is_good = full(npv, True)
#     ldm = zeros((npv, npb, ng))  # Limb darkening means
#     xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
#     hwws = zeros((npv, npb))     # Half-window widths [d]
#
#     exs = zeros((npv, npv, 2))   # Ellipse model scanline x-coordinates
#     eys = zeros((npv, npv))      # Elilpse model scanline y-coordinates
#
#     for ipv in range(npv):
#         if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0) or (isnan(ldp[ipv, 0, 0])):
#             pv_is_good[ipv] = False
#             continue
#
#         # -----------------------------------#
#         # Calculate the limb darkening means #
#         # -----------------------------------#
#         if kmin <= ks[ipv, 0] <= kmax:
#             ik = int(floor((ks[ipv, 0] - kmin) / dk))
#             ak = (ks[ipv, 0] - kmin - ik * dk) / dk
#             for ipb in range(npb):
#                 ldm[ipv, ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb]) + ak * dot(weights[ik + 1], ldp[ipv, ipb])
#         else:
#             _, _, wg = calculate_weights_2d(ks[ipv, 0], z_edges, ng)
#             for ipb in range(npb):
#                 ldm[ipv, ipb, :] = dot(wg, ldp[ipv, ipb])
#
#         # ------------------------------------------------------#
#         # Calculate the Taylor series expansions for the orbits #
#         # ------------------------------------------------------#
#         xyc[ipv, :, :] = solve_xy_p5s(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
#
#         # ---------------------------------#
#         # Calculate the half-window widths #
#         # ---------------------------------#
#         hww = 0.5 * d_from_pkaiews(p[ipv], ks[ipv, 0], a[ipv], i[ipv], e[ipv], w[ipv], 1, 14)
#         for ilc in range(nlc):
#             hwws[ipv, ilc] = 0.0015 + _exptimes[ilc] + hww
#
#         # ----------------------------------
#         # Create the ellipse (x, y) points #
#         # ----------------------------------
#         _y, _x = create_ellipse(npl, ks[ipv,0], f[ipv], alpha[ipv])
#         exs[ipv, :, :] = _x
#         eys[ipv, :] = _y
#
#     # ---------------------------#
#     # Calculate the light curves #
#     # ---------------------------#
#     flux = zeros((npv, npt))
#     for j in prange(npv * npt):
#         ipv = j // npt
#         ipt = j % npt
#
#         if not pv_is_good[ipv]:
#             flux[ipv, ipt] = nan
#             continue
#
#         ilc = lcids[ipt]
#         ipb = pbids[ilc]
#         iep = epids[ilc]
#
#         epoch = floor((times[ipt] - t0[ipv, iep] + 0.5 * p[ipv]) / p[ipv])
#         tc = times[ipt] - (t0[ipv, iep] + epoch * p[ipv])
#         if fabs(tc) > hwws[ipv, ilc]:
#             flux[ipv, ipt] = 1.0
#         else:
#             for isample in range(1, nsamples[ilc] + 1):
#                 time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
#                 z = pd_t15sc(tc + time_offset, xyc[ipv])
#                 iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + ks[ipv, ipb]), dg, ldm[ipv, ipb])
#                 aplanet = ecia(z, ks[ipv, ipb], f[ipv], exs[ipv,:,:], eys[ipv,:])
#                 flux[ipv, ipt] += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
#             flux[ipv, ipt] /= nsamples[ilc]
#     return flux
