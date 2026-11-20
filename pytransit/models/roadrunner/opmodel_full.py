from math import floor, sqrt

from meepmeep.backends.numba.point2d import pos_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, ndarray, isnan, nan, full, squeeze, atleast_2d, atleast_1d, empty, int64

from .common import (population_arrays, g_nodes, ldm_nodes, ldm_table, split_cubic_coefficients,
                     split_point, ldm_lookup, profile_at,
                     valid_radius_ratios)
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

    The evaluation is split into a per-parameter-vector precompute stage and a flux stage over
    all (parameter vector, time sample) pairs. Both are compiled in serial and in parallel. The
    parallel precompute distributes the parameter vectors over the numba threads and keeps
    everything inside one parameter vector serial, so its many small array operations never
    launch the thread pool on their own.
    """
    if parallelize:
        ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews = op_precompute_parallel(
            k, f, alpha, p, a, i, e, w, nlc, npb, npl, exptimes,
            ldp, pt0, pdt, rules, ng, exact_areas)
        return op_flux_parallel(times, f, alpha, t0, p, ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews,
                                lcids, pbids, epids, nsamples, exptimes, ldp, istar, pt0, pdt,
                                exact_areas, exact_ld, nannuli)
    else:
        ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews = op_precompute(
            k, f, alpha, p, a, i, e, w, nlc, npb, npl, exptimes,
            ldp, pt0, pdt, rules, ng, exact_areas)
        return op_flux_serial(times, f, alpha, t0, p, ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews,
                              lcids, pbids, epids, nsamples, exptimes, ldp, istar, pt0, pdt,
                              exact_areas, exact_ld, nannuli)


@njit
def op_precompute_pv(k: ndarray, f: float, alpha: float, p: float, a: float, i: float, e: float, w: float,
                     exptimes: ndarray, ldp: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int,
                     exact_areas: bool, klds: ndarray, gcs: ndarray, n1s: ndarray, coef: ndarray,
                     xyc: ndarray, bbs: ndarray, exs: ndarray, eys: ndarray, ews: ndarray) -> bool:
    """Precompute the quantities of a single parameter vector.

    Fills the vector's rows of the output arrays and returns whether the vector is valid. The
    work arrays are allocated here so that the function can be called from several threads at
    once. The scanline arrays are left untouched when the exact intersection areas are used.
    """
    if isnan(a) or (a <= 1.0) or (e < 0.0) or isnan(ldp[0, 0]) or not valid_radius_ratios(k):
        return False

    npb = k.size
    nq = rules.shape[2]
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    ldm = empty(ng)

    # -----------------------------------#
    # Calculate the limb darkening means #
    # -----------------------------------#
    # The mean limb darkening is evaluated for an area-equivalent circular planet with a
    # radius k*sqrt(1-f): using the ellipse's semi-major axis would overestimate the
    # intensity-sampling footprint and bias the blocked flux (see the accuracy tests in
    # tests/test_opmodel.py).
    # The quadrature nodes depend on the radius ratio, so they are rebuilt whenever it
    # differs from the previous passband's.
    for ipb in range(npb):
        klds[ipb] = k[ipb] * sqrt(1.0 - f)
        if ipb == 0 or klds[ipb] != klds[ipb - 1]:
            gs, n1 = g_nodes(klds[ipb], ng)
            ldm_nodes(klds[ipb], gs, rules, mu, wf)
        gcs[ipb] = split_point(klds[ipb])
        n1s[ipb] = n1
        ldm_table(mu, wf, pt0, pdt, ldp[ipb], ldm)
        split_cubic_coefficients(ldm, n1, coef[ipb])

    # ------------------------------------------------------#
    # Calculate the Taylor series expansions for the orbits #
    # ------------------------------------------------------#
    xyc[:, :] = solve2d(0.0, p, a, i, e, w)

    # -----------------------------#
    # Calculate the bounding boxes #
    # -----------------------------#
    bt1, bt4 = bounding_box(k[0], xyc)
    for ilc in range(exptimes.size):
        bbs[ilc, 0] = bt1 - (0.0015 + exptimes[ilc])
        bbs[ilc, 1] = bt4 + (0.0015 + exptimes[ilc])

    # -------------------------------------------#
    # Create the ellipse scanline (x, y) points  #
    # -------------------------------------------#
    if not exact_areas:
        for ipb in range(npb):
            _y, _x, _w = create_ellipse_theta(eys.shape[1], k[ipb], f, alpha)
            exs[ipb, :, :] = _x
            eys[ipb, :] = _y
            ews[ipb, :] = _w
    return True


def _op_precompute(k: ndarray, f: ndarray, alpha: ndarray, p: ndarray, a: ndarray, i: ndarray,
                   e: ndarray, w: ndarray, nlc: int, npb: int, npl: int, exptimes: ndarray,
                   ldp: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int, exact_areas: bool):
    """Precompute the per-parameter-vector quantities needed by the flux stage.

    For each parameter vector: the mean limb darkening profiles, the Taylor series expansion
    coefficients for the planet position, the transit bounding boxes, and (when not using exact
    intersection areas) the θ-sampled ellipse scanline grids per passband.

    Compiled both in serial and in parallel; in the parallel version the loop over the parameter
    vectors is distributed over the numba threads.
    """
    npv = k.shape[0]

    if k.shape[1] > 1 and k.shape[1] != npb:
        raise ValueError('Radius ratios should be given either as an [npv, 1] or [npv, npb] array.')

    # Copy the radius ratios
    # ----------------------
    if k.shape[1] == npb:
        ks = k
    else:
        ks = zeros((npv, npb))
        for ipb in range(npb):
            ks[:, ipb] = k[:, 0]

    pv_is_good = full(npv, True)
    klds = zeros((npv, npb))             # LD-equivalent circular planet radii
    gcs = zeros((npv, npb))              # Limb contact point of the mean intensity table, per (pv, pb)
    n1s = zeros((npv, npb), int64)       # Nodes in the first table piece, per (pv, pb)
    coef = zeros((npv, npb, ng - 3, 4))  # Cubic coefficients of the mean intensity tables
    xyc = zeros((npv, 2, 5))             # Taylor series coefficients for the (x, y) position
    bbs = zeros((npv, nlc, 2))           # Bounding boxes per (pv, lc)

    # θ-sampled ellipse scanline grids per (pv, pb), used when exact_areas is False. With exact
    # areas they are single-line placeholders, kept at the full (pv, pb) shape so that the flux
    # stage can index them the same way in both modes.
    if exact_areas:
        exs = zeros((npv, npb, 1, 2))
        eys = zeros((npv, npb, 1))
        ews = zeros((npv, npb, 1))
    else:
        exs = zeros((npv, npb, npl, 2))  # Scanline x-coordinates
        eys = zeros((npv, npb, npl))     # Scanline y-coordinates
        ews = zeros((npv, npb, npl))     # Scanline quadrature weights

    for ipv in prange(npv):
        pv_is_good[ipv] = op_precompute_pv(ks[ipv], f[ipv], alpha[ipv], p[ipv], a[ipv], i[ipv], e[ipv], w[ipv],
                                           exptimes[:nlc], ldp[ipv], pt0, pdt, rules, ng, exact_areas,
                                           klds[ipv], gcs[ipv], n1s[ipv], coef[ipv], xyc[ipv], bbs[ipv],
                                           exs[ipv], eys[ipv], ews[ipv])
    return ks, klds, pv_is_good, gcs, n1s, coef, xyc, bbs, exs, eys, ews


op_precompute = njit(parallel=False)(_op_precompute)
op_precompute_parallel = njit(parallel=True)(_op_precompute)


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
                    iplanet = ldm_lookup(z / (1.0 + klds[ipv, ipb]), gcs[ipv, ipb], n1s[ipv, ipb], coef[ipv, ipb])
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
