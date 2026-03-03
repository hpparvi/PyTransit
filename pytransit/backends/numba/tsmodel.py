from meepmeep.backends.numba.ts2d import solve_xy_p5, pd_t15c, solve_xy_p5_d, pd_t15c_d
from meepmeep.backends.numba.ts2d.position import bounding_box
from numba import njit, prange
from numpy import ndarray, zeros, sqrt, linspace, nan, floor, isnan, full, dot, any, fabs, mean

from .ccintersection import ccia_and_k0, ccia_and_grad
from .rrmodel import calculate_weights_2d, interpolate_mean_limb_darkening, interpolate_mean_limb_darkening_and_grad

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
        xyc[:, :] = solve_xy_p5(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

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
                    z = pd_t15c(tc + time_offset, xyc)
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


@njit(fastmath=False)
def tsmodel_and_grad(times: ndarray,
                     k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                     nsamples: ndarray, exptimes: ndarray,
                     ldp: ndarray, ldg: ndarray, istar: ndarray, distar: ndarray,
                     weights: ndarray, dk: float, kmin: float, kmax: float, ng: int, dg: float, z_edges: ndarray):
    """Transmission spectrum transit model with analytical gradients.

    Parameters
    ----------
    times : ndarray
        Observation times, shape (npt,).
    k : ndarray
        Radius ratios, shape (npv, npb).
    t0, p, a, i, e, w : ndarray
        Orbital parameters, each shape (npv,).
    nsamples : ndarray
        Number of supersamples.
    exptimes : ndarray
        Exposure times.
    ldp : ndarray
        Limb darkening profiles, shape (npv, npb, nmu).
    ldg : ndarray
        LD profile + per-coefficient derivatives, shape (npv, npb, 1+nldc, nmu).
    istar : ndarray
        Disk-integrated intensities, shape (npv, npb).
    distar : ndarray
        Derivative of istar w.r.t. each LD coefficient, shape (npv, npb, nldc).
    weights : ndarray
        Pre-computed weight table.
    dk : float
        k step size in weight table.
    kmin : float
        Minimum k in weight table.
    kmax : float
        Maximum k in weight table.
    ng : int
        Grazing parameter resolution.
    dg : float
        Grazing parameter step size.
    z_edges : ndarray
        Radial zone edges.

    Returns
    -------
    flux : ndarray, shape (npv, npb, npt)
        Model flux.
    dflux : ndarray, shape (npv, npb, npt, 7 + nldc)
        Derivatives w.r.t. [k_ipb, t0, p, a, i, e, w, c_0, ..., c_{nldc-1}].
    """
    if k.ndim != 2:
        raise ValueError("The radius ratios must be given as a 2D array with shape (npv, npb)")
    if ldp.ndim != 3:
        raise ValueError("The limb darkening profiles must be given as a 3D array with shape (npv, npb, nmu)")
    if k.shape[1] != ldp.shape[1]:
        raise ValueError("The number of radius ratios and passbands must match.")

    npt = times.size
    npv = k.shape[0]
    npb = k.shape[1]
    nldc = ldg.shape[2] - 1

    if weights is not None:
        ng = weights.shape[1]

    flux = zeros((npv, npb, npt))
    dflux = zeros((npv, npb, npt, 7 + nldc))
    ldm = zeros((npb, ng))
    dldm_dc = zeros((npb, nldc, ng))

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0):
            flux[ipv, :, :] = nan
            dflux[ipv, :, :, :] = nan
            continue

        kmean = mean(k[ipv])
        kmax_band = 0.0
        for ipb in range(npb):
            if k[ipv, ipb] > kmax_band:
                kmax_band = k[ipv, ipb]
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
                for j in range(nldc):
                    dldm_dc[ipb, j, :] = (1.0 - ak) * dot(weights[ik], ldg[ipv, ipb, j + 1, :]) + ak * dot(weights[ik + 1], ldg[ipv, ipb, j + 1, :])
        else:
            _, dg, wg = calculate_weights_2d(kmean, z_edges, ng)
            for ipb in prange(npb):
                ldm[ipb, :] = dot(wg, ldp[ipv, ipb, :])
                for j in range(nldc):
                    dldm_dc[ipb, j, :] = dot(wg, ldg[ipv, ipb, j + 1, :])

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        cf, dcf = solve_xy_p5_d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])

        # --------------------------------#
        # Calculate the half-window width #
        # --------------------------------#
        bt1, bt4 = bounding_box(kmax_band, cf)
        bt1 -= 0.003 + exptimes[0]
        bt4 += 0.003 + exptimes[0]

        # ----------------------------------#
        # Calculate the light curve & grads #
        # ----------------------------------#
        for ipt in prange(npt):
            epoch = floor((times[ipt] - t0[ipv] + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (t0[ipv] + epoch * p[ipv])
            if not (bt1 <= tc <= bt4):
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples[0] + 1):
                    time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                    t_eval = tc + time_offset

                    z, dz = pd_t15c_d(t_eval, cf, dcf)

                    if z <= 1.0 - kmax_band:
                        # -------------------------------------------------#
                        # Inner case: planet fully inside the stellar disk #
                        # -------------------------------------------------#
                        ap0, (dadk_mean, dadz_mean) = ccia_and_grad(1.0, kmean, z)

                        for ipb in range(npb):
                            inv_is = inv_istar[ipb]
                            g = z / (1.0 + kmean)
                            iplanet, dIp_dg = interpolate_mean_limb_darkening_and_grad(g, dg, ldm[ipb])
                            aeff = ap0 * afac[ipb]

                            flux[ipv, ipb, ipt] += (1.0 - iplanet * aeff * inv_is) * inv_nsamples

                            # k derivative: d(aeff)/dk_ipb = ap0 * 2*k[ipb] / kmean^2
                            daeff_dk = ap0 * 2.0 * k[ipv, ipb] / (kmean * kmean)
                            dflux[ipv, ipb, ipt, 0] += -(iplanet * daeff_dk) * inv_is * inv_nsamples

                            # z derivatives for orbital parameters
                            dIp_dz = dIp_dg / (1.0 + kmean)
                            daeff_dz = dadz_mean * afac[ipb]
                            dfdz = (dIp_dz * aeff + iplanet * daeff_dz) * inv_is * inv_nsamples

                            # t0 derivative: dz/dt0 = -dz[0]
                            dflux[ipv, ipb, ipt, 1] += dfdz * dz[0]
                            # p, a, i, e, w derivatives
                            for ip in range(1, 6):
                                dflux[ipv, ipb, ipt, ip + 1] += -dfdz * dz[ip]

                            # LD coefficient derivatives
                            for j in range(nldc):
                                dIp_dcj = interpolate_mean_limb_darkening(g, dg, dldm_dc[ipb, j])
                                dflux[ipv, ipb, ipt, 7 + j] += -aeff * dIp_dcj * inv_is * inv_nsamples
                    else:
                        # -------------------------------------------------#
                        # Edge case: planet may cross the stellar limb     #
                        # -------------------------------------------------#
                        for ipb in range(npb):
                            inv_is = inv_istar[ipb]
                            kb = k[ipv, ipb]
                            g = z / (1.0 + kb)
                            iplanet, dIp_dg = interpolate_mean_limb_darkening_and_grad(g, dg, ldm[ipb])
                            aplanet, (dadk, dadz) = ccia_and_grad(1.0, kb, z)

                            flux[ipv, ipb, ipt] += (1.0 - iplanet * aplanet * inv_is) * inv_nsamples

                            # k derivative
                            dIp_dk = dIp_dg * (-z / (1.0 + kb) ** 2)
                            dflux[ipv, ipb, ipt, 0] += -(dIp_dk * aplanet + iplanet * dadk) * inv_is * inv_nsamples

                            # z derivatives for orbital parameters
                            dIp_dz = dIp_dg / (1.0 + kb)
                            dfdz = (dIp_dz * aplanet + iplanet * dadz) * inv_is * inv_nsamples

                            # t0 derivative
                            dflux[ipv, ipb, ipt, 1] += dfdz * dz[0]
                            # p, a, i, e, w derivatives
                            for ip in range(1, 6):
                                dflux[ipv, ipb, ipt, ip + 1] += -dfdz * dz[ip]

                            # LD coefficient derivatives
                            for j in range(nldc):
                                dIp_dcj = interpolate_mean_limb_darkening(g, dg, dldm_dc[ipb, j])
                                dflux[ipv, ipb, ipt, 7 + j] += -aplanet * dIp_dcj * inv_is * inv_nsamples

                # Add the distar contribution for LD coefficients
                for ipb in range(npb):
                    inv_is = inv_istar[ipb]
                    for j in range(nldc):
                        dflux[ipv, ipb, ipt, 7 + j] += (1.0 - flux[ipv, ipb, ipt]) * distar[ipv, ipb, j] * inv_is

    return flux, dflux