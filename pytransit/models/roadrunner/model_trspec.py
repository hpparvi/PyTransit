from math import fabs, floor

from meepmeep.backends.numba.point2d import sep_c, solve2d, bounding_box
from numba import njit, prange, get_num_threads, set_num_threads
from numpy import zeros, dot, ndarray, isnan, nan, mean, floor, fabs, max, empty

from .common import (g_nodes, ldm_nodes, ldm_table, split_cubic_coefficients, split_point, ldm_lookup)
from .common import circle_circle_intersection_area_kite as ccia


@njit(parallel=False, fastmath=False)
def tsmodel_serial(times: ndarray,
                   k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
                   pt0: float, pdt: float, rules: ndarray, ng: int) -> ndarray:
    if k.ndim != 2:
        raise ValueError(" The radius ratios must be given as a 2D array with shape (npv, npb)")

    if ldp.ndim != 3:
        raise ValueError("The limb darkening profiles must be given as a 3D array with shape (npv, npb, nmu)")

    if k.shape[1] != ldp.shape[1]:
        raise ValueError("The transmission spectrum transit model requires that the number or radius ratios and the number of passbands match.")

    npt = times.size
    npv = k.shape[0]
    npb = k.shape[1]


    flux = zeros((npv, npb, npt))  # Model flux
    nq = rules.shape[2]
    coef = zeros((npb, ng - 2, 4))   # Split cubic coefficients of the mean intensity tables
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    ldm = empty(ng)
    xyc = zeros((2, 5))            # Taylor series coefficients for the (x, y) position

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0):
            flux[ipv, :, :] = nan
            continue

        kmean = mean(k[ipv])
        kmax = max(k[ipv])
        afac = k[ipv] ** 2 / kmean ** 2

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        gs, n1 = g_nodes(kmean, ng)
        gc = split_point(kmean)
        ldm_nodes(kmean, gs, rules, mu, wf)
        for ipb in range(npb):
            ldm_table(mu, wf, pt0, pdt, ldp[ipv, ipb, :], ldm)
            split_cubic_coefficients(ldm, n1, coef[ipb])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        xyc[:, :] = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # ---------------------------#
        # Calculate the bounding box #
        # ---------------------------#
        bt1, bt4 = bounding_box(kmean, xyc)
        bt1 -= 0.0015 + exptimes[0]
        bt4 += 0.0015 + exptimes[0]

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in range(npt):
            epoch = floor((times[ipt] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (t0[ipv] + epoch * p[ipv])
            if not (bt1 <= tc <= bt4):
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples[0] + 1):
                    time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                    z = sep_c(tc + time_offset, xyc)
                    ap0, kappa = ccia(1.0, kmean, z)
                    dadk = 2.0*kmean*kappa
                    if z <= 1.0 - kmax:
                        for ipb in range(npb):
                            iplanet = ldm_lookup(z / (1.0 + kmean), gc, n1, coef[ipb])
                            flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * ap0 * afac[ipb]) / istar[ipv, ipb]
                    else:
                        for ipb in range(npb):
                            iplanet = ldm_lookup(z / (1.0 + kmean), gc, n1, coef[ipb])
                            flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * (ap0 + (k[ipv, ipb]-kmean)*dadk)) / istar[ipv, ipb]
                flux[ipv, :, ipt] /= nsamples[0]
    return flux


@njit(parallel=True, fastmath=False)
def tsmodel_parallel(times: ndarray,
                   k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
                   pt0: float, pdt: float, rules: ndarray, ng: int,
                   nthreads: int) -> ndarray:

    nthreads_current = get_num_threads()
    set_num_threads(nthreads)

    if k.ndim != 2:
        raise ValueError(" The radius ratios must be given as a 2D array with shape (npv, npb)")

    if ldp.ndim != 3:
        raise ValueError("The limb darkening profiles must be given as a 3D array with shape (npv, npb, nmu)")

    if k.shape[1] != ldp.shape[1]:
        raise ValueError("The transmission spectrum transit model requires that the number or radius ratios and the number of passbands match.")

    npt = times.size
    npv = k.shape[0]
    npb = k.shape[1]

    flux = zeros((npv, npb, npt))  # Model flux
    nq = rules.shape[2]
    coef = zeros((npb, ng - 2, 4))   # Split cubic coefficients of the mean intensity tables
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    ldm = empty(ng)
    xyc = zeros((2, 5))            # Taylor series coefficients for the (x, y) position

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0):
            flux[ipv, :, :] = nan
            continue

        kmean = mean(k[ipv])
        kmax = max(k[ipv])
        afac = k[ipv] ** 2 / kmean ** 2

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        gs, n1 = g_nodes(kmean, ng)
        gc = split_point(kmean)
        ldm_nodes(kmean, gs, rules, mu, wf)
        for ipb in range(npb):
            ldm_table(mu, wf, pt0, pdt, ldp[ipv, ipb, :], ldm)
            split_cubic_coefficients(ldm, n1, coef[ipb])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        xyc[:, :] = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # ---------------------------#
        # Calculate the bounding box #
        # ---------------------------#
        bt1, bt4 = bounding_box(kmean, xyc)
        bt1 -= 0.0015 + exptimes[0]
        bt4 += 0.0015 + exptimes[0]

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in prange(npt):
            epoch = floor((times[ipt] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (t0[ipv] + epoch * p[ipv])
            if not (bt1 <= tc <= bt4):
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples[0] + 1):
                    time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                    z = sep_c(tc + time_offset, xyc)
                    ap0, kappa = ccia(1.0, kmean, z)
                    dadk = 2.0*kmean*kappa
                    for ipb in range(npb):
                        if z <= 1.0 - kmax:
                            for ipb in range(npb):
                                iplanet = ldm_lookup(z / (1.0 + kmean), gc, n1, coef[ipb])
                                flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * ap0 * afac[ipb]) / istar[ipv, ipb]
                        else:
                            for ipb in range(npb):
                                iplanet = ldm_lookup(z / (1.0 + kmean), gc, n1, coef[ipb])
                                flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * (
                                            ap0 + (k[ipv, ipb] - kmean) * dadk)) / istar[ipv, ipb]
                flux[ipv, :, ipt] /= nsamples[0]
    set_num_threads(nthreads_current)
    return flux