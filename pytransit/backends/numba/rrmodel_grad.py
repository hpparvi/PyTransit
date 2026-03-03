from meepmeep.backends.numba.ts2d import solve_xy_p5_d, pd_t15c_d
from meepmeep.backends.numba.ts2d.position import bounding_box
from numpy import ndarray, isnan, any, full, nan, zeros, floor, dot

from .ccintersection import ccia_and_grad
from .rrmodel import calculate_weights_2d, interpolate_mean_limb_darkening_and_grad, interpolate_mean_limb_darkening


def rr_simple_and_grad(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
                       nsamples: ndarray, exptimes: ndarray, ldp: ndarray, ldg: ndarray, ldi: float, dldi: ndarray,
                       weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Simplified RoadRunner model with analytical gradients for a single homogeneous light curve.

    Parameters
    ----------
    times : ndarray
        Observation times.
    k : float
        Radius ratio.
    t0 : float
        Mid-transit time.
    p : float
        Orbital period.
    a : float
        Semi-major axis in stellar radii.
    i : float
        Orbital inclination [rad].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [rad].
    nsamples : ndarray
        Number of supersamples per point.
    exptimes : ndarray
        Exposure times.
    ldp : ndarray
        Limb darkening profile, shape (1, 1, nmu).
    ldg : ndarray
        LD profile and its derivatives, shape (1+nldc, nmu).
        Row 0: dI/dmu, rows 1..: dI/dc_j.
    ldi : float
        Disk-integrated intensity.
    dldi : ndarray
        Derivative of ldi w.r.t. each LD coefficient, shape (nldc,).
    weights : ndarray
        3D weight table, shape (nk, ng, nmu).
    dk : float
        k step size in weight table.
    kmin : float
        Minimum k in weight table.
    kmax : float
        Maximum k in weight table.
    dg : float
        Grazing parameter step size.
    z_edges : ndarray
        Radial zone edges for weight computation.

    Returns
    -------
    flux : ndarray, shape (npt,)
        Model flux.
    dflux : ndarray, shape (npt, 7 + nldc)
        Derivatives w.r.t. [k, t0, p, a, i, e, w, c_0, c_1, ...].
    """
    npt = times.size
    ng = weights.shape[1]
    nldc = ldg.shape[0] - 1
    ldi_val = ldi[0, 0]

    if isnan(a) or (a <= 1.0) or (e < 0.0) or any(isnan(ldp[0])):
        return full(npt, nan), full((npt, 7 + nldc), nan)

    # ------------------------------------------ #
    # Calculate the limb darkening mean (ldm)    #
    # ------------------------------------------ #
    ldm = zeros(ng)

    if kmin <= k <= kmax:
        ik = int(floor((k - kmin) / dk))
        ak = (k - kmin - ik * dk) / dk
        ldm[:] = (1.0 - ak) * dot(weights[ik], ldp[0, 0]) + ak * dot(weights[ik + 1], ldp[0, 0])
    else:
        _, _, wg = calculate_weights_2d(k, z_edges, ng)
        ldm[:] = dot(wg, ldp[0, 0])

    # ------------------------------------------ #
    # LD coefficient derivatives: dldm_dc        #
    # dldm_dc[j] = dot(W, ldg[j+1])             #
    # ------------------------------------------ #
    dldm_dc = zeros((nldc, ng))
    if kmin <= k <= kmax:
        ik = int(floor((k - kmin) / dk))
        ak = (k - kmin - ik * dk) / dk
        for j in range(nldc):
            dldm_dc[j, :] = (1.0 - ak) * dot(weights[ik], ldg[j + 1]) + ak * dot(weights[ik + 1], ldg[j + 1])
    else:
        for j in range(nldc):
            dldm_dc[j, :] = dot(wg, ldg[j + 1])

    # ------------------------------------------------------- #
    # Taylor series expansions for the orbit with derivatives  #
    # ------------------------------------------------------- #
    cf, dcf = solve_xy_p5_d(0.0, p, a, i, e, w)

    # --------------- #
    # Bounding box    #
    # --------------- #
    bt1, bt4 = bounding_box(k, cf)
    bt1 -= 0.003 + exptimes[0]
    bt4 += 0.003 + exptimes[0]

    # ---------------------------------- #
    # Calculate the light curve & grads  #
    # ---------------------------------- #
    flux = zeros(npt)
    dflux = zeros((npt, 7 + nldc))

    for ipt in range(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if not (bt1 <= tc <= bt4):
            flux[ipt] = 1.0
        else:
            for isample in range(1, nsamples[0] + 1):
                time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                t_eval = tc + time_offset

                z, dz = pd_t15c_d(t_eval, cf, dcf)
                aplanet, (dadk, dadz) = ccia_and_grad(1.0, k, z)
                g = z / (1.0 + k)
                iplanet, dIp_dg = interpolate_mean_limb_darkening_and_grad(g, dg, ldm)

                flux[ipt] += (ldi_val - iplanet * aplanet) / ldi_val

                # dI_p/dz = (dI_p/dg) / (1+k)
                dIp_dz = dIp_dg / (1.0 + k)

                # --- k derivative (dldm/dk ≈ 0) ---
                dIp_dk = dIp_dg * (-z / (1.0 + k) ** 2)
                dflux[ipt, 0] += -(dIp_dk * aplanet + iplanet * dadk) / ldi_val

                # --- t0 derivative ---
                # dz/dt0 = -dz/dphase = -dz[0], so dflux/dt0 = +(chain) * dz[0] / ldi
                dflux[ipt, 1] += (dIp_dz * aplanet + iplanet * dadz) * dz[0] / ldi_val

                # --- p, a, i, e, w derivatives (indices 1..5 in dz) ---
                for ip in range(1, 6):
                    dflux[ipt, ip + 1] += -(dIp_dz * aplanet + iplanet * dadz) * dz[ip] / ldi_val

                # --- LD coefficient derivatives ---
                for j in range(nldc):
                    dIp_dcj = interpolate_mean_limb_darkening(g, dg, dldm_dc[j])
                    dflux[ipt, 7 + j] += -aplanet * dIp_dcj / ldi_val

            flux[ipt] /= nsamples[0]
            for ip in range(7 + nldc):
                dflux[ipt, ip] /= nsamples[0]

            # Add the dldi contribution for LD coefficients
            for j in range(nldc):
                dflux[ipt, 7 + j] += (1.0 - flux[ipt]) * dldi[j] / ldi_val

    return flux, dflux
