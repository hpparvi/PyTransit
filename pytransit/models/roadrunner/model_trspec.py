from math import floor

from meepmeep.backends.numba.point2d import sep_c, solve2d, bounding_box
from numba import njit, prange, get_num_threads, set_num_threads
from numpy import zeros, ndarray, isnan, nan, mean, max, empty, int64

from .common import (g_nodes, ldm_nodes, ldm_table_map, ldm_table_banded, split_cubic_coefficients, split_point,
                     ldm_interval, valid_radius_ratios)
from .common import circle_circle_intersection_area_kite as ccia


@njit(fastmath=False)
def transit_samples(times: ndarray, t0: float, p: float, xyc: ndarray, k: float, kmax: float,
                    gc: float, n1: int, nc: int, nsamples: int, exptime: float,
                    interval: ndarray, ipts: ndarray, basis: ndarray, area: ndarray) -> int:
    """The geometry of every exposure sample that overlaps the star, shared by all the wavelength bins.

    For each overlapping sample, stores its exposure index, the mean intensity table interval it
    falls in, the cubic basis ``(a, a**2, a**3)`` at its position in that interval, and the three
    overlap area terms the bins combine as ``area[0] + area[1] * (k / k_mean)**2 + area[2] * (k - k_mean)``.
    The area terms are divided by the number of samples, so the bins can sum the samples directly.

    Returns the number of overlapping samples.
    """
    bt1, bt4 = bounding_box(k, xyc)
    bt1 -= 0.0015 + exptime
    bt4 += 0.0015 + exptime

    n = 0
    for ipt in range(times.size):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if not (bt1 <= tc <= bt4):
            continue
        for isample in range(1, nsamples + 1):
            z = sep_c(tc + exptime * ((isample - 0.5) / nsamples - 0.5), xyc)
            ap0, kappa = ccia(1.0, k, z)
            dadk = 2.0 * k * kappa
            if ap0 == 0.0 and dadk == 0.0:
                continue
            i, a = ldm_interval(z / (1.0 + k), gc, n1, nc)
            ipts[n] = ipt
            interval[n] = i
            basis[n, 0] = a
            basis[n, 1] = a * a
            basis[n, 2] = a * a * a

            # The area is exact while the planet is fully on the disk, and linear in k - k_mean across the limb.
            if z <= 1.0 - kmax:
                area[n, 0] = 0.0
                area[n, 1] = ap0 / nsamples
                area[n, 2] = 0.0
            else:
                area[n, 0] = ap0 / nsamples
                area[n, 1] = 0.0
                area[n, 2] = dadk / nsamples
            n += 1
    return n


def _tsmodel(times: ndarray,
             k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
             nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
             pt0: float, pdt: float, rules: ndarray, ng: int) -> ndarray:
    """The transmission spectroscopy model. Compiled both serially and in parallel over the wavelength bins."""
    if k.ndim != 2:
        raise ValueError(" The radius ratios must be given as a 2D array with shape (npv, npb)")

    if ldp.ndim != 3:
        raise ValueError("The limb darkening profiles must be given as a 3D array with shape (npv, npb, nmu)")

    if k.shape[1] != ldp.shape[1]:
        raise ValueError("The transmission spectrum transit model requires that the number or radius ratios and the number of passbands match.")

    npt = times.size
    npv = k.shape[0]
    npb = k.shape[1]
    nmu = ldp.shape[2]
    ns = nsamples[0]
    nc = ng - 2                      # Number of intervals in the split mean intensity table

    flux = empty((npv, npb, npt))  # Model flux
    nq = rules.shape[2]
    tables = empty((npb, ng))      # Mean intensity tables
    coef = empty((npb, nc, 4))     # Split cubic coefficients of the mean intensity tables
    mu = empty((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))

    # Geometry of the overlapping exposure samples
    ipts = empty(npt * ns, dtype=int64)
    interval = empty(npt * ns, dtype=int64)
    basis = empty((npt * ns, 3))
    area = empty((npt * ns, 3))

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0) or not valid_radius_ratios(k[ipv]):
            flux[ipv, :, :] = nan
            continue

        kmean = mean(k[ipv])
        kmax = max(k[ipv])
        afac = k[ipv] ** 2 / kmean ** 2
        dk = k[ipv] - kmean

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        # The table is linear in the profile, so the map from a profile to its table is built once
        # and applied to every bin. Dividing by the stellar intensity here saves a division per bin
        # and exposure later.
        gs, n1 = g_nodes(kmean, ng)
        gc = split_point(kmean)
        ldm_nodes(kmean, gs, rules, mu, wf)
        wt, lo, hi = ldm_table_map(mu, wf, pt0, pdt, nmu)
        for ipb in prange(npb):
            ldm_table_banded(wt, lo, hi, ldp[ipv, ipb], 1.0 / istar[ipv, ipb], tables[ipb])
            split_cubic_coefficients(tables[ipb], n1, coef[ipb])

        # ------------------------------------------#
        # Calculate the geometry shared by the bins #
        # ------------------------------------------#
        xyc = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        nsm = transit_samples(times, t0[ipv], p[ipv], xyc, kmean, kmax, gc, n1, nc, ns, exptimes[0],
                              interval, ipts, basis, area)

        # ---------------------------#
        # Calculate the light curves #
        # ---------------------------#
        for ipb in prange(npb):
            f = flux[ipv, ipb]
            c = coef[ipb]
            f[:] = 1.0
            for s in range(nsm):
                ci = c[interval[s]]
                iplanet = ci[0] + basis[s, 0] * ci[1] + basis[s, 1] * ci[2] + basis[s, 2] * ci[3]
                f[ipts[s]] -= iplanet * (area[s, 0] + area[s, 1] * afac[ipb] + area[s, 2] * dk[ipb])
    return flux


tsmodel_serial = njit(parallel=False, fastmath=False)(_tsmodel)
_tsmodel_parallel = njit(parallel=True, fastmath=False)(_tsmodel)


@njit
def tsmodel_parallel(times: ndarray,
                     k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                     nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
                     pt0: float, pdt: float, rules: ndarray, ng: int,
                     nthreads: int) -> ndarray:
    nthreads_current = get_num_threads()
    set_num_threads(nthreads)
    flux = _tsmodel_parallel(times, k, t0, p, a, i, e, w, nsamples, exptimes, ldp, istar, pt0, pdt, rules, ng)
    set_num_threads(nthreads_current)
    return flux
