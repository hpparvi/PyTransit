from math import floor, sqrt

from meepmeep.backends.numba.point2d import pos_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, nan, full, squeeze, atleast_2d, atleast_1d

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .ecintersection import (create_ellipse_theta, ellipse_circle_intersection_area_theta,
                             ellipse_circle_intersection_area_exact)


def opmodel(times, k, f, alpha, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep, npl,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, weights, dk, kmin, kmax, dg, z_edges, exact_areas=False):

    k, f, alpha, t0, p, a, i, e, w = (atleast_2d(k), atleast_1d(f), atleast_1d(alpha), atleast_2d(t0), atleast_1d(p),
                                      atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w))

    return squeeze(op_full(times, k, f, alpha, t0, p, a, i, e, w, parallelize, nlc, npb, nep, npl,
                   lcids, pbids, epids, nsamples, exptimes,
                   ldp, istar, weights, dk, kmin, kmax, dg, z_edges, exact_areas))


@njit
def op_full(times: ndarray, k: ndarray, f: ndarray, alpha: ndarray,
            t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int, npl: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray,
            exact_areas: bool = False):
    """Full oblate planet model for heterogeneous light curves.

    The evaluation is split into a serial per-parameter-vector precompute stage and a flux stage
    parallelized over all (parameter vector, time sample) pairs. Only the flux stage is compiled
    with ``parallel=True``: compiling the precompute stage in parallel would turn its many small
    array operations into per-iteration thread-pool launches, which costs far more than it saves.
    """
    ks, pv_is_good, ldm, xyc, bbs, exs, eys, ews = op_precompute(
        k, f, alpha, p, a, i, e, w, nlc, npb, npl, exptimes,
        ldp, weights, dk, kmin, kmax, z_edges, exact_areas)

    if parallelize:
        return op_flux_parallel(times, f, alpha, t0, p, ks, pv_is_good, ldm, xyc, bbs, exs, eys, ews,
                                lcids, pbids, epids, nsamples, exptimes, istar, dg, exact_areas)
    else:
        return op_flux_serial(times, f, alpha, t0, p, ks, pv_is_good, ldm, xyc, bbs, exs, eys, ews,
                              lcids, pbids, epids, nsamples, exptimes, istar, dg, exact_areas)


@njit(parallel=False)
def op_precompute(k: ndarray, f: ndarray, alpha: ndarray, p: ndarray, a: ndarray, i: ndarray,
                  e: ndarray, w: ndarray, nlc: int, npb: int, npl: int, exptimes: ndarray,
                  ldp: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float,
                  z_edges: ndarray, exact_areas: bool):
    """Precompute the per-parameter-vector quantities needed by the flux stage.

    For each parameter vector: the mean limb darkening profiles, the Taylor series expansion
    coefficients for the planet position, the transit bounding boxes, and (when not using exact
    intersection areas) the θ-sampled ellipse scanline grids per passband.
    """
    npv = k.shape[0]
    ng = weights.shape[1]

    if k.shape[1] > 1 and k.shape[1] != npb:
        raise ValueError('Radius ratios should be given either as an [npv, 1] or [npv, npb] array.')

    # Copy the radius ratios
    # ----------------------
    if k.shape[1] == npb:
        ks = k
    else:
        ks = zeros((npv, npb))
        ks[:, :] = k[:, 0:npb]

    pv_is_good = full(npv, True)
    ldm = zeros((npv, npb, ng))  # Limb darkening means
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    bbs = zeros((npv, nlc, 2))   # Bounding boxes per (pv, lc)

    # θ-sampled ellipse scanline grids per (pv, pb), used when exact_areas is False
    if exact_areas:
        exs = zeros((1, 1, 1, 2))
        eys = zeros((1, 1, 1))
        ews = zeros((1, 1, 1))
    else:
        exs = zeros((npv, npb, npl, 2))  # Scanline x-coordinates
        eys = zeros((npv, npb, npl))     # Scanline y-coordinates
        ews = zeros((npv, npb, npl))     # Scanline quadrature weights

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
        xyc[ipv, :, :] = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # -----------------------------#
        # Calculate the bounding boxes #
        # -----------------------------#
        bt1, bt4 = bounding_box(ks[ipv, 0], xyc[ipv])
        for ilc in range(nlc):
            bbs[ipv, ilc, 0] = bt1 - (0.0015 + exptimes[ilc])
            bbs[ipv, ilc, 1] = bt4 + (0.0015 + exptimes[ilc])

        # -------------------------------------------#
        # Create the ellipse scanline (x, y) points  #
        # -------------------------------------------#
        if not exact_areas:
            for ipb in range(npb):
                _y, _x, _w = create_ellipse_theta(npl, ks[ipv, ipb], f[ipv], alpha[ipv])
                exs[ipv, ipb, :, :] = _x
                eys[ipv, ipb, :] = _y
                ews[ipv, ipb, :] = _w

    return ks, pv_is_good, ldm, xyc, bbs, exs, eys, ews


def _op_flux(times: ndarray, f: ndarray, alpha: ndarray, t0: ndarray, p: ndarray,
             ks: ndarray, pv_is_good: ndarray, ldm: ndarray, xyc: ndarray, bbs: ndarray,
             exs: ndarray, eys: ndarray, ews: ndarray,
             lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
             istar: ndarray, dg: float, exact_areas: bool):
    """Calculate the model fluxes for all (parameter vector, time sample) pairs.

    Compiled both in serial and in parallel; in the parallel version the flat loop over
    the (parameter vector, time sample) pairs is distributed over the numba threads.
    """
    npv = ks.shape[0]
    npt = times.size
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
        if not (bbs[ipv, ilc, 0] <= tc <= bbs[ipv, ilc, 1]):
            flux[ipv, ipt] = 1.0
        else:
            fsum = 0.0
            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                cx, cy = pos_c(tc + time_offset, xyc[ipv])
                z = sqrt(cx*cx + cy*cy)
                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + ks[ipv, ipb]), dg, ldm[ipv, ipb])
                if exact_areas:
                    aplanet = ellipse_circle_intersection_area_exact(cx, cy, z, ks[ipv, ipb], f[ipv], alpha[ipv])
                else:
                    aplanet = ellipse_circle_intersection_area_theta(cx, cy, z, ks[ipv, ipb], f[ipv],
                                                                     exs[ipv, ipb], eys[ipv, ipb], ews[ipv, ipb])
                fsum += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
            flux[ipv, ipt] = fsum / nsamples[ilc]
    return flux


op_flux_serial = njit(parallel=False)(_op_flux)
op_flux_parallel = njit(parallel=True)(_op_flux)
