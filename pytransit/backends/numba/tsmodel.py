from meepmeep.numba2d import solve2d, sep_c, bounding_box
from numba import njit, prange
from numpy import ndarray, zeros, nan, floor, isnan, dot, mean

from .ccintersection import ccia_and_k0
from .rrmodel import calculate_weights_2d, interpolate_mean_limb_darkening


@njit(fastmath=False)
def tsmodel(times: ndarray,
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

        # Pre-compute reciprocals to replace divisions in the hot loop
        inv_nsamples = 1.0 / nsamples[0]
        inv_istar = zeros(npb)
        for ipb in range(npb):
            inv_istar[ipb] = 1.0 / istar[ipv, ipb]

        # -----------------------------------#
        # Calculate the limb darkening means #
        # -----------------------------------#
        if weights is not None and kmin <= kmean <= kmax:
            ik = int(floor((kmean - kmin) / dk))
            ak = (kmean - kmin - ik * dk) / dk
            for ipb in prange(npb):
                ldm[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb, :]) + ak * dot(weights[ik + 1], ldp[ipv, ipb, :])
        else:
            _, dg, wg = calculate_weights_2d(kmean, z_edges, ng)
            for ipb in prange(npb):
                ldm[ipb, :] = dot(wg, ldp[ipv, ipb, :])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        xyc[:, :] = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # --------------------------------#
        # Calculate the half-window width #
        # --------------------------------#
        bt1, bt4 = bounding_box(k[0,0], xyc)
        bt1 -= 0.003 + exptimes[0]
        bt4 += 0.003 + exptimes[0]

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
                    ap0, k0 = ccia_and_k0(1.0, kmean, z)
                    dadk = 2.0*kmean*k0
                    if z <= 1.0 - kmax:
                        for ipb in range(npb):
                            iplanet = interpolate_mean_limb_darkening(z / (1.0 + kmean), dg, ldm[ipb])
                            flux[ipv, ipb, ipt] += (1.0 - iplanet * ap0 * afac[ipb] * inv_istar[ipb]) * inv_nsamples
                    else:
                        for ipb in range(npb):
                            iplanet = interpolate_mean_limb_darkening(z / (1.0 + kmean), dg, ldm[ipb])
                            flux[ipv, ipb, ipt] += (1.0 - iplanet * (ap0 + (k[ipv, ipb]-kmean)*dadk) * inv_istar[ipb]) * inv_nsamples
    return flux


