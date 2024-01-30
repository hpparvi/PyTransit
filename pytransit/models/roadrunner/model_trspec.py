from math import fabs, floor
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, full, nan, mean

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc
from meepmeep.utils import d_from_pkaiews

from .common import calculate_weights_2d, interpolate_mean_limb_darkening_s
from .common import circle_circle_intersection_area_kite as ccia


@njit(parallel=False, fastmath=False)
def rr_simple_serial(times: ndarray, k: ndarray, t0: float, p: float, a: float, i: float, e: float, w: float,
                     nsamples: int, exptimes: float, ldp: ndarray, istar: ndarray,
                     weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    npt = times.size
    npb = ldp.shape[1]
    ng = weights.shape[1]

    if k.size != npb:
        raise ValueError("The number or radius ratios must match the number of passbands.")

    if isnan(a) or (a <= 1.0) or (e < 0.0):
        return full((npb, npt), nan)

    kmean = mean(k)
    afac = k ** 2 / kmean ** 2

    ldm = zeros((npb, ng))  # Limb darkening means
    xyc = zeros((2, 5))  # Taylor series coefficients for the (x, y) position

    # ----------------------------------#
    # Calculate the limb darkening mean #
    # ----------------------------------#
    if kmin <= kmean <= kmax:
        ik = int(floor((kmean - kmin) / dk))
        ak = (kmean - kmin - ik * dk) / dk
        for ipb in range(npb):
            ldm[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[0, ipb, :]) + ak * dot(weights[ik + 1], ldp[0, ipb, :])
    else:
        _, _, wg = calculate_weights_2d(kmean, z_edges, ng)
        for ipb in range(npb):
            ldm[ipb, :] = dot(wg, ldp[0, ipb, :])

    # -----------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit #
    # -----------------------------------------------------#
    xyc[:, :] = solve_xy_p5s(0.0, p, a, i, e, w)

    # --------------------------------#
    # Calculate the half-window width #
    # --------------------------------#
    hww = 0.5 * d_from_pkaiews(p, kmean, a, i, e, w, 1, 14)
    hww = 0.0015 + exptimes + hww

    # --------------------------#
    # Calculate the light curve #
    # --------------------------#
    flux = zeros((npb, npt))
    for ipt in range(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if fabs(tc) > hww:
            flux[:, ipt] = 1.0
        else:
            for isample in range(1, nsamples + 1):
                time_offset = exptimes * ((isample - 0.5) / nsamples - 0.5)
                z = pd_t15sc(tc + time_offset, xyc)
                aplanet = ccia(1.0, kmean, z)[0]
                for ipb in range(npb):
                    iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + kmean), dg, ldm[ipb])
                    flux[ipb, ipt] += (istar[0, ipb] - iplanet * aplanet * afac[ipb]) / istar[0, ipb]
            flux[:, ipt] /= nsamples
    return flux
