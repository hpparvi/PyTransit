from meepmeep.backends.numba.point2d import sep_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, nan, full, floor, empty, int64

from .common import (g_nodes, ldm_nodes, ldm_table, split_cubic_coefficients, split_point, ldm_lookup,
                     valid_radius_ratios)
from .common import circle_circle_intersection_area_kite as ccia


@njit
def rr_full(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            parallelize: bool, nlc: int, npb: int, nep: int,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, istar: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int):
    """Full RoadRunner model for heterogeneous light curves.

    The evaluation is split into a serial per-parameter-vector precompute stage and a flux stage
    parallelized over all (parameter vector, time sample) pairs. Only the flux stage is compiled
    with ``parallel=True``: compiling the precompute stage in parallel would turn its many small
    array operations into per-iteration thread-pool launches, which costs far more than it saves.
    """
    ks, pv_is_good, gcs, n1s, coef, xyc, bbs = rr_precompute(k, p, a, i, e, w, nlc, npb, exptimes,
                                                             ldp, pt0, pdt, rules, ng)

    if parallelize:
        return rr_flux_parallel(times, t0, p, ks, pv_is_good, gcs, n1s, coef, xyc, bbs,
                                lcids, pbids, epids, nsamples, exptimes, istar)
    else:
        return rr_flux_serial(times, t0, p, ks, pv_is_good, gcs, n1s, coef, xyc, bbs,
                              lcids, pbids, epids, nsamples, exptimes, istar)


@njit(parallel=False)
def rr_precompute(k: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                  nlc: int, npb: int, exptimes: ndarray, ldp: ndarray,
                  pt0: float, pdt: float, rules: ndarray, ng: int):
    """Precompute the per-parameter-vector quantities needed by the flux stage.

    For each parameter vector: the mean limb darkening profiles, the Taylor series expansion
    coefficients for the planet position, and the transit bounding boxes.
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
    gcs = zeros((npv, npb))            # Split point of the mean intensity table, per (pv, pb)
    n1s = zeros((npv, npb), int64)     # Nodes in the first segment of the table, per (pv, pb)
    coef = zeros((npv, npb, ng - 2, 4))  # Split cubic coefficients of the mean intensity table
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    ldm = empty(ng)
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    bbs = zeros((npv, nlc, 2))   # Bounding boxes per (pv, lc)

    for ipv in range(npv):
        if (isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0) or (isnan(ldp[ipv, 0, 0]))
                or not valid_radius_ratios(ks[ipv])):
            pv_is_good[ipv] = False
            continue

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        # The quadrature nodes depend on the radius ratio, so they are rebuilt whenever it
        # differs from the previous passband's.
        for ipb in range(npb):
            if ipb == 0 or ks[ipv, ipb] != ks[ipv, ipb - 1]:
                gs, n1 = g_nodes(ks[ipv, ipb], ng)
                ldm_nodes(ks[ipv, ipb], gs, rules, mu, wf)
            gcs[ipv, ipb] = split_point(ks[ipv, ipb])
            n1s[ipv, ipb] = n1
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
            bbs[ipv, ilc, 0] = bt1 - (0.003 + exptimes[ilc])
            bbs[ipv, ilc, 1] = bt4 + (0.003 + exptimes[ilc])

    return ks, pv_is_good, gcs, n1s, coef, xyc, bbs


def _rr_flux(times: ndarray, t0: ndarray, p: ndarray,
             ks: ndarray, pv_is_good: ndarray, gcs: ndarray, n1s: ndarray, coef: ndarray, xyc: ndarray, bbs: ndarray,
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
                z = sep_c(tc + time_offset, xyc[ipv])
                iplanet = ldm_lookup(z / (1.0 + ks[ipv, ipb]), gcs[ipv, ipb], n1s[ipv, ipb], coef[ipv, ipb])
                aplanet = ccia(1.0, ks[ipv, ipb], z)[0]
                fsum += (istar[ipv, ipb] - iplanet * aplanet) / istar[ipv, ipb]
            flux[ipv, ipt] = fsum / nsamples[ilc]
    return flux


rr_flux_serial = njit(parallel=False)(_rr_flux)
rr_flux_parallel = njit(parallel=True)(_rr_flux)
