from math import fabs, floor
from numba import njit, prange
from numpy import zeros, dot, ndarray

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc
from meepmeep.utils import d_from_pkaiews

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .common import circle_circle_intersection_area_kite as ccia

@njit
def rr_full(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Full RoadRunner model for heterogeneous light curves."""

    if parallelize:
        return rr_full_parallel(times, k, t0, p, a, i, e, w, nlc, npb, nep,
                              lcids, pbids, epids, nsamples, exptimes,
                              ldp, istar, weights, dk, kmin, kmax, dg, z_edges)
    else:
        return rr_full_serial(times, k, t0, p, a, i, e, w, nlc, npb, nep,
                              lcids, pbids, epids, nsamples, exptimes,
                              ldp, istar, weights, dk, kmin, kmax, dg, z_edges)


@njit(parallel=False, fastmath=False)
def rr_full_serial(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            nlc: int, npb: int, nep: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Full RoadRunner model for heterogeneous light curves."""
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]

    _exptimes = zeros(nlc)
    _exptimes[:] = exptimes
    _nsamples = zeros(nlc)
    _nsamples[:] = nsamples

    # Copy the radius ratios
    # ----------------------
    ks = zeros((npv, npb))
    if npv == 1:
        ks[:] = k
    else:
        if k.shape[1] == 1:
            for ipv in range(npv):
                ks[ipv, :] = k[ipv]
        else:
            ks[:, :] = k

    ldm = zeros((npv, npb, ng))  # Limb darkening means
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    hwws = zeros((npv, npb))     # Half-window widths [d]

    for ipv in range(npv):
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

    # ---------------------------#
    # Calculate the light curves #
    # ---------------------------#
    flux = zeros((npv, npt))
    for j in prange(npv * npt):
        ipv = j // npt
        ipt = j % npt
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
                z = pd_t15sc(tc + time_offset, xyc[ipv])
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + ks[ipv, ipb]), dg, ldm[ipv, ipb])
                aplanet = ccia(1.0, ks[ipv, ipb], z)[0]
                flux[ipv, ipt] += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
            flux[ipv, ipt] /= nsamples[ilc]
    return flux

@njit(parallel=True, fastmath=False)
def rr_full_parallel(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            nlc: int, npb: int, nep: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Full RoadRunner model for heterogeneous light curves."""
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]

    _exptimes = zeros(nlc)
    _exptimes[:] = exptimes
    _nsamples = zeros(nlc)
    _nsamples[:] = nsamples

    # Copy the radius ratios
    # ----------------------
    ks = zeros((npv, npb))
    if npv == 1:
        ks[:] = k
    else:
        if k.shape[1] == 1:
            for ipv in range(npv):
                ks[ipv, :] = k[ipv]
        else:
            ks[:, :] = k

    ldm = zeros((npv, npb, ng))  # Limb darkening means
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    hwws = zeros((npv, npb))     # Half-window widths [d]

    for ipv in range(npv):
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

    # ---------------------------#
    # Calculate the light curves #
    # ---------------------------#
    flux = zeros((npv, npt))
    for j in prange(npv * npt):
        ipv = j // npt
        ipt = j % npt
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
                z = pd_t15sc(tc + time_offset, xyc[ipv])
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + ks[ipv, ipb]), dg, ldm[ipv, ipb])
                aplanet = ccia(1.0, ks[ipv, ipb], z)[0]
                flux[ipv, ipt] += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
            flux[ipv, ipt] /= nsamples[ilc]
    return flux