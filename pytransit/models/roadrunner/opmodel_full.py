from math import floor, sqrt

from meepmeep.backends.numba.point2d import pos_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, nan, full, squeeze, atleast_2d, atleast_1d, empty, int64

from .common import (population_arrays, g_nodes, ldm_nodes, ldm_table, split_cubic_coefficients,
                     split_point, ldm_lookup, profile_at)
from .ecintersection import (create_ellipse_theta, ellipse_circle_intersection_area_theta,
                             ellipse_circle_intersection_area_exact, ellipse_disk_intersection_area_theta)


@njit
def op_ld_blocked(cx: float, cy: float, z: float, k: float, f: float, alpha: float,
                  xs: ndarray, ys: ndarray, ws: ndarray, pt0: float, pdt: float, ldpn: ndarray,
                  nannuli: int, exact_areas: bool) -> float:
    """Flux blocked by the oblate planet, with the limb darkening integrated over the exact footprint.

    The blocked flux is calculated as a Stieltjes sum over stellar annuli spanning the radial
    range covered by the planet, with the annulus areas calculated as differences of
    ellipse-disk intersection areas and the annulus nodes placed uniformly in μ so that the
    intensity profile stays smooth at the limb. This removes the circular-footprint mean
    intensity approximation used by the standard model version.
    """
    rlo = max(z - k, 0.0)
    rhi = min(z + k, 1.0)
    mu_hi = sqrt(max(0.0, 1.0 - rlo * rlo))
    mu_lo = sqrt(max(0.0, 1.0 - rhi * rhi))
    dmu = (mu_hi - mu_lo) / nannuli

    if exact_areas:
        a_prev = rhi * rhi * ellipse_circle_intersection_area_exact(cx / rhi, cy / rhi, z / rhi, k / rhi, f, alpha)
    else:
        a_prev = ellipse_disk_intersection_area_theta(cx, cy, z, k, f, xs, ys, ws, rhi)

    blocked = 0.0
    for g in range(nannuli):
        mu1 = mu_lo + (g + 1) * dmu
        r = sqrt(max(0.0, 1.0 - mu1 * mu1))
        if r <= 0.0:
            a_next = 0.0
        elif exact_areas:
            a_next = r * r * ellipse_circle_intersection_area_exact(cx / r, cy / r, z / r, k / r, f, alpha)
        else:
            a_next = ellipse_disk_intersection_area_theta(cx, cy, z, k, f, xs, ys, ws, r)
        mmid = mu_lo + (g + 0.5) * dmu
        blocked += (a_prev - a_next) * profile_at(mmid, pt0, pdt, ldpn)
        a_prev = a_next
    return blocked


def opmodel(times, k, f, alpha, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep, npl,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, pt0, pdt, rules, ng,
            exact_areas=False, exact_ld=False, nannuli=20):

    k = atleast_2d(k)
    t0, p, a, i, e, w = population_arrays(t0, p, a, i, e, w, npv=k.shape[0])
    f, alpha = (full(k.shape[0], atleast_1d(x)[0]) if atleast_1d(x).size == 1 else atleast_1d(x) for x in (f, alpha))

    return squeeze(op_full(times, k, f, alpha, t0, p, a, i, e, w, parallelize, nlc, npb, nep, npl,
                   lcids, pbids, epids, nsamples, exptimes,
                   ldp, istar, pt0, pdt, rules, ng, exact_areas, exact_ld, nannuli))


@njit
def op_full(times: ndarray, k: ndarray, f: ndarray, alpha: ndarray,
            t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int, npl: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int,
            exact_areas: bool = False, exact_ld: bool = False, nannuli: int = 20):
    """Full oblate planet model for heterogeneous light curves.

    The evaluation is split into a serial per-parameter-vector precompute stage and a flux stage
    parallelized over all (parameter vector, time sample) pairs. Only the flux stage is compiled
    with ``parallel=True``: compiling the precompute stage in parallel would turn its many small
    array operations into per-iteration thread-pool launches, which costs far more than it saves.
    """
    ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews = op_precompute(
        k, f, alpha, p, a, i, e, w, nlc, npb, npl, exptimes,
        ldp, pt0, pdt, rules, ng, exact_areas)

    if parallelize:
        return op_flux_parallel(times, f, alpha, t0, p, ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews,
                                lcids, pbids, epids, nsamples, exptimes, ldp, istar, pt0, pdt,
                                exact_areas, exact_ld, nannuli)
    else:
        return op_flux_serial(times, f, alpha, t0, p, ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews,
                              lcids, pbids, epids, nsamples, exptimes, ldp, istar, pt0, pdt,
                              exact_areas, exact_ld, nannuli)


@njit(parallel=False)
def op_precompute(k: ndarray, f: ndarray, alpha: ndarray, p: ndarray, a: ndarray, i: ndarray,
                  e: ndarray, w: ndarray, nlc: int, npb: int, npl: int, exptimes: ndarray,
                  ldp: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int, exact_areas: bool):
    """Precompute the per-parameter-vector quantities needed by the flux stage.

    For each parameter vector: the mean limb darkening profiles, the Taylor series expansion
    coefficients for the planet position, the transit bounding boxes, and (when not using exact
    intersection areas) the θ-sampled ellipse scanline grids per passband.
    """
    npv = k.shape[0]
    nq = rules.shape[2]

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
    klds = zeros((npv, npb))     # LD-equivalent circular planet radii
    gcs = zeros(npv)                     # Split point of the mean intensity table, per pv
    n1s = zeros(npv, int64)              # Nodes in the first table segment, per pv
    coef = zeros((npv, npb, ng - 2, 4))  # Split cubic coefficients of the mean intensity tables
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    ldm = empty(ng)
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
        # The mean limb darkening is evaluated for an area-equivalent circular planet with a
        # radius k*sqrt(1-f): using the ellipse's semi-major axis would overestimate the
        # intensity-sampling footprint and bias the blocked flux (see the accuracy tests in
        # tests/test_opmodel.py).
        for ipb in range(npb):
            klds[ipv, ipb] = ks[ipv, ipb] * sqrt(1.0 - f[ipv])
        gs, n1 = g_nodes(klds[ipv, 0], ng)
        gcs[ipv] = split_point(klds[ipv, 0])
        n1s[ipv] = n1
        ldm_nodes(klds[ipv, 0], gs, rules, mu, wf)
        for ipb in range(npb):
            ldm_table(mu, wf, pt0, pdt, ldp[ipv, ipb], ldm)
            split_cubic_coefficients(ldm, n1, coef[ipv, ipb])

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

    return ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews


def _op_flux(times: ndarray, f: ndarray, alpha: ndarray, t0: ndarray, p: ndarray,
             ks: ndarray, klds: ndarray, pv_is_good: ndarray, gcs: ndarray, n1s: ndarray, coef: ndarray, xyc: ndarray, bbs: ndarray,
             exs: ndarray, eys: ndarray, ews: ndarray,
             lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
             ldp: ndarray, istar: ndarray, pt0: float, pdt: float,
             exact_areas: bool, exact_ld: bool, nannuli: int):
    """Calculate the model fluxes for all (parameter vector, time sample) pairs.

    Compiled both in serial and in parallel; in the parallel version the flat loop over
    the (parameter vector, time sample) pairs is distributed over the numba threads.
    """
    # The precomputed arrays are copied locally because numba generates measurably (~35%)
    # faster code for the flux loop with locally allocated arrays than with array arguments.
    ks = ks.copy()
    klds = klds.copy()
    pv_is_good = pv_is_good.copy()
    coef = coef.copy()
    xyc = xyc.copy()
    bbs = bbs.copy()
    exs = exs.copy()
    eys = eys.copy()
    ews = ews.copy()

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
                if exact_ld:
                    if z >= 1.0 + ks[ipv, ipb]:
                        blocked = 0.0
                    else:
                        blocked = op_ld_blocked(cx, cy, z, ks[ipv, ipb], f[ipv], alpha[ipv],
                                                exs[ipv, ipb], eys[ipv, ipb], ews[ipv, ipb],
                                                pt0, pdt, ldp[ipv, ipb], nannuli, exact_areas)
                    fsum += (istar[ipv, ipb] - blocked) / istar[ipv, ipb]
                else:
                    iplanet = ldm_lookup(z / (1.0 + klds[ipv, ipb]), gcs[ipv], n1s[ipv], coef[ipv, ipb])
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
