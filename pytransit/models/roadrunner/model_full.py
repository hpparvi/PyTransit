from meepmeep.backends.numba.point2d import pos_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, ndarray, isnan, nan, full, floor, empty, sqrt, pi

from .common import (g_nodes, table_sizes, ldm_nodes, ldm_table, split_cubic_coefficients, split_point, ldm_lookup,
                     ldm_lookup_full,
                     valid_radius_ratios)
from .common import circle_circle_intersection_area_kite as ccia


@njit
def rr_full(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int):
    """Full RoadRunner model for heterogeneous light curves.

    The evaluation is split into a per-parameter-vector precompute stage and a flux stage over
    all (parameter vector, time sample) pairs. Both are compiled in serial and in parallel. The
    parallel precompute distributes the parameter vectors over the numba threads and keeps
    everything inside one parameter vector serial, so its many small array operations never
    launch the thread pool on their own.
    """
    if parallelize:
        ks, pv_is_good, gcs, coef, xyc, bbs = rr_precompute_parallel(k, p, a, i, e, w, nlc, npb, exptimes,
                                                                     ldp, pt0, pdt, rules, ng)
        return rr_flux_parallel(times, t0, p, ks, pv_is_good, gcs, coef, xyc, bbs,
                                lcids, pbids, epids, nsamples, exptimes, istar)
    else:
        ks, pv_is_good, gcs, coef, xyc, bbs = rr_precompute(k, p, a, i, e, w, nlc, npb, exptimes,
                                                            ldp, pt0, pdt, rules, ng)
        return rr_flux_serial(times, t0, p, ks, pv_is_good, gcs, coef, xyc, bbs,
                              lcids, pbids, epids, nsamples, exptimes, istar)


@njit
def rr_precompute_pv(k: ndarray, p: float, a: float, i: float, e: float, w: float,
                     exptimes: ndarray, ldp: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int,
                     gcs: ndarray, coef: ndarray, xyc: ndarray, bbs: ndarray) -> bool:
    """Precompute the quantities of a single parameter vector.

    Fills the vector's rows of `gcs`, `coef`, `xyc` and `bbs`, and returns whether the vector
    is valid. The work arrays are allocated here so that the function can be called from
    several threads at once.
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
    # The quadrature nodes depend on the radius ratio, so they are rebuilt whenever it
    # differs from the previous passband's.
    for ipb in range(npb):
        if ipb == 0 or k[ipb] != k[ipb - 1]:
            gs, n1 = g_nodes(k[ipb], ng)
            ldm_nodes(k[ipb], gs, rules, mu, wf)
        gcs[ipb] = split_point(k[ipb])
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
        bbs[ilc, 0] = bt1 - (0.003 + exptimes[ilc])
        bbs[ilc, 1] = bt4 + (0.003 + exptimes[ilc])
    return True


def _rr_precompute(k: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   nlc: int, npb: int, exptimes: ndarray, ldp: ndarray,
                   pt0: float, pdt: float, rules: ndarray, ng: int):
    """Precompute the per-parameter-vector quantities needed by the flux stage.

    For each parameter vector: the cubic coefficients of the mean intensity tables, the Taylor
    series expansion coefficients for the planet position, and the transit bounding boxes.

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
    gcs = zeros((npv, npb))              # Limb contact point of the mean intensity table, per (pv, pb)
    coef = zeros((npv, npb, ng - 3, 4))  # Cubic coefficients of the mean intensity tables
    xyc = zeros((npv, 2, 5))             # Taylor series coefficients for the (x, y) position
    bbs = zeros((npv, nlc, 2))           # Bounding boxes per (pv, lc)

    for ipv in prange(npv):
        pv_is_good[ipv] = rr_precompute_pv(ks[ipv], p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], exptimes[:nlc],
                                           ldp[ipv], pt0, pdt, rules, ng,
                                           gcs[ipv], coef[ipv], xyc[ipv], bbs[ipv])
    return ks, pv_is_good, gcs, coef, xyc, bbs


rr_precompute = njit(parallel=False)(_rr_precompute)
rr_precompute_parallel = njit(parallel=True)(_rr_precompute)


def _rr_flux(times: ndarray, t0: ndarray, p: ndarray,
             ks: ndarray, pv_is_good: ndarray, gcs: ndarray, coef: ndarray, xyc: ndarray, bbs: ndarray,
             lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
             istar: ndarray):
    """Calculate the model fluxes for all (parameter vector, time sample) pairs.

    Compiled both in serial and in parallel; in the parallel version the flat loop over
    the (parameter vector, time sample) pairs is distributed over the numba threads.
    """
    # The precomputed arrays are copied locally because numba generates measurably (~35%)
    # faster code for the flux loop with locally allocated arrays than with array arguments.
    ks = ks.copy()
    pv_is_good = pv_is_good.copy()
    coef = coef.copy()
    xyc = xyc.copy()
    bbs = bbs.copy()
    istar = istar.copy()

    npv = ks.shape[0]
    npt = times.size
    nlc = nsamples.size
    n1 = table_sizes(coef.shape[2] + 3)[0]

    # The divisions are hoisted out of the loop as reciprocals.
    inv_p = 1.0 / p
    inv_k1 = 1.0 / (1.0 + ks)
    inv_istar = 1.0 / istar
    inv_ns = 1.0 / nsamples
    dts = exptimes[:nlc] * inv_ns                 # Supersample spacing, per light curve
    t_first = 0.5 * (dts - exptimes[:nlc])        # Offset of the first supersample from the exposure centre

    # A planet fully on the disk blocks its whole area, and its mean intensity table is laid out
    # in a coordinate that follows from the squared projected distance, so those samples need
    # neither the distance itself nor the overlap area calculation.
    z2cs = (1.0 - ks) ** 2
    inv_z2cs = zeros(ks.shape)
    for ipv in range(npv):
        for ipb in range(ks.shape[1]):
            if z2cs[ipv, ipb] > 0.0:
                inv_z2cs[ipv, ipb] = 1.0 / z2cs[ipv, ipb]

    flux = zeros((npv, npt))
    for j in prange(npv * npt):
        ipv = j // npt
        ipt = j - ipv * npt

        if not pv_is_good[ipv]:
            flux[ipv, ipt] = nan
            continue

        ilc = lcids[ipt]
        ipb = pbids[ilc]
        iep = epids[ilc]

        epoch = floor((times[ipt] - t0[ipv, iep]) * inv_p[ipv] + 0.5)
        tc = times[ipt] - (t0[ipv, iep] + epoch * p[ipv])
        if not (bbs[ipv, ilc, 0] <= tc <= bbs[ipv, ilc, 1]):
            flux[ipv, ipt] = 1.0
        else:
            k = ks[ipv, ipb]
            z2c = z2cs[ipv, ipb]
            area_full = pi * k * k
            fsum = 0.0
            for isample in range(nsamples[ilc]):
                px, py = pos_c(tc + t_first[ilc] + isample * dts[ilc], xyc[ipv])
                z2 = px * px + py * py
                if z2 < z2c:
                    fsum += ldm_lookup_full(z2, inv_z2cs[ipv, ipb], n1, coef[ipv, ipb]) * area_full
                else:
                    z = sqrt(z2)
                    fsum += ldm_lookup(z * inv_k1[ipv, ipb], gcs[ipv, ipb], n1, coef[ipv, ipb]) * ccia(1.0, k, z)[0]
            flux[ipv, ipt] = 1.0 - fsum * inv_istar[ipv, ipb] * inv_ns[ilc]
    return flux


rr_flux_serial = njit(parallel=False)(_rr_flux)
rr_flux_parallel = njit(parallel=True)(_rr_flux)
