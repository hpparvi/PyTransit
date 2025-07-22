from math import fabs, floor
from numba import njit, prange, get_num_threads, set_num_threads
from numpy import zeros, dot, ndarray, isnan, full, nan, mean, floor, fabs, max

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc
from meepmeep.utils import d_from_pkaiews

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .common import circle_circle_intersection_area_kite as ccia


@njit(parallel=False, fastmath=False)
def tsmodel_serial(times: ndarray,
                   k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
                   weights: ndarray, dk: float, kmin: float, kmax: float, ng: int, dg: float, z_edges: ndarray) -> ndarray:
    if k.ndim != 2:
        raise ValueError(" The radius ratios must be given as a 2D array with shape (npv, npb)")

    if ldp.ndim != 3:
        raise ValueError("The limb darkening profiles must be given as a 3D array with shape (npv, npb, nmu)")

    if k.shape[1] != ldp.shape[1]:
        raise ValueError("The transmission spectrum transit model requires that the number or radius ratios and the number of passbands match.")

    npt = times.size
    npv = k.shape[0]
    npb = k.shape[1]

    if weights is not None:
        ng = weights.shape[1]

    flux = zeros((npv, npb, npt))  # Model flux
    ldm = zeros((npb, ng))         # Limb darkening means
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
        if weights is not None and kmin <= kmean <= kmax:
            ik = int(floor((kmean - kmin) / dk))
            ak = (kmean - kmin - ik * dk) / dk
            for ipb in range(npb):
                ldm[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb, :]) + ak * dot(weights[ik + 1], ldp[ipv, ipb, :])
        else:
            _, dg, wg = calculate_weights_2d(kmean, z_edges, ng)
            for ipb in range(npb):
                ldm[ipb, :] = dot(wg, ldp[ipv, ipb, :])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        xyc[:, :] = solve_xy_p5s(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # --------------------------------#
        # Calculate the half-window width #
        # --------------------------------#
        hww = 0.5 * d_from_pkaiews(p[ipv], kmean, a[ipv], i[ipv], e[ipv], w[ipv], 1, 14)
        hww = 0.0015 + exptimes[0] + hww

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in range(npt):
            epoch = floor((times[ipt] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (t0[ipv] + epoch * p[ipv])
            if fabs(tc) > hww:
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples[0] + 1):
                    time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                    z = pd_t15sc(tc + time_offset, xyc)
                    ap0, kappa = ccia(1.0, kmean, z)
                    dadk = 2.0*kmean*kappa
                    if z <= 1.0 - kmax:
                        for ipb in range(npb):
                            iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + kmean), dg, ldm[ipb])
                            flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * ap0 * afac[ipb]) / istar[ipv, ipb]
                    else:
                        for ipb in range(npb):
                            iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + kmean), dg, ldm[ipb])
                            flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * (ap0 + (k[ipv, ipb]-kmean)*dadk)) / istar[ipv, ipb]
                flux[ipv, :, ipt] /= nsamples[0]
    return flux


@njit(parallel=True, fastmath=False)
def tsmodel_parallel(times: ndarray,
                   k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   nsamples: ndarray, exptimes: ndarray, ldp: ndarray, istar: ndarray,
                   weights: ndarray, dk: float, kmin: float, kmax: float, ng: int, dg: float, z_edges: ndarray,
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
    if weights is not None:
        ng = weights.shape[1]

    flux = zeros((npv, npb, npt))  # Model flux
    ldm = zeros((npb, ng))         # Limb darkening means
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
        if weights is not None and kmin <= kmean <= kmax:
            ik = int(floor((kmean - kmin) / dk))
            ak = (kmean - kmin - ik * dk) / dk
            for ipb in range(npb):
                ldm[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb, :]) + ak * dot(weights[ik + 1],
                                                                                         ldp[ipv, ipb, :])
        else:
            dk, dg, wg = calculate_weights_2d(kmean, z_edges, ng)
            for ipb in prange(npb):
                ldm[ipb, :] = dot(wg, ldp[ipv, ipb, :])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        xyc[:, :] = solve_xy_p5s(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # --------------------------------#
        # Calculate the half-window width #
        # --------------------------------#
        hww = 0.5 * d_from_pkaiews(p[ipv], kmean, a[ipv], i[ipv], e[ipv], w[ipv], 1, 14)
        hww = 0.0015 + exptimes[0] + hww

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in prange(npt):
            epoch = floor((times[ipt] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (t0[ipv] + epoch * p[ipv])
            if fabs(tc) > hww:
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples[0] + 1):
                    time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                    z = pd_t15sc(tc + time_offset, xyc)
                    ap0, kappa = ccia(1.0, kmean, z)
                    dadk = 2.0*kmean*kappa
                    for ipb in range(npb):
                        if z <= 1.0 - kmax:
                            for ipb in range(npb):
                                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + kmean), dg, ldm[ipb])
                                flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * ap0 * afac[ipb]) / istar[ipv, ipb]
                        else:
                            for ipb in range(npb):
                                iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + kmean), dg, ldm[ipb])
                                flux[ipv, ipb, ipt] += (istar[ipv, ipb] - iplanet * (
                                            ap0 + (k[ipv, ipb] - kmean) * dadk)) / istar[ipv, ipb]
                flux[ipv, :, ipt] /= nsamples[0]
    set_num_threads(nthreads_current)
    return flux