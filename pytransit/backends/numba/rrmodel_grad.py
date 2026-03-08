from meepmeep.backends.numba.ts2d import solve_xy_p5_d, pd_t15c_d
from meepmeep.backends.numba.ts2d.position import bounding_box
from numba import prange
from numpy import ndarray, isnan, any, full, nan, zeros, floor, dot

from .ccintersection import ccia_and_grad
from .rrmodel import calculate_weights_2d, interpolate_mean_limb_darkening_and_grad, interpolate_mean_limb_darkening
from ._utils import _folded_time


def rrmodel_grad(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                 lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
                 ldp: ndarray, ldg: ndarray, ldi: ndarray, dldi: ndarray,
                 weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray,
                 npb: int, nep: int):
    """RoadRunner model with analytical gradients supporting heterogeneous light curves.

    Parameters
    ----------
    times : ndarray
        Observation times, shape (npt,).
    k : ndarray
        Radius ratio, shape (npv, npb).
    t0 : ndarray
        Mid-transit time, shape (npv, ntc).
    p : ndarray
        Orbital period, shape (npv, nep).
    a : ndarray
        Semi-major axis in stellar radii, shape (npv, nep).
    i : ndarray
        Orbital inclination [rad], shape (npv, nep).
    e : ndarray
        Eccentricity, shape (npv, nep).
    w : ndarray
        Argument of periastron [rad], shape (npv, nep).
    lcids : ndarray
        Light curve index for each time stamp.
    pbids : ndarray
        Passband index for each light curve.
    epids : ndarray
        Epoch index for each light curve.
    nsamples : ndarray
        Number of supersamples per light curve.
    exptimes : ndarray
        Exposure time per light curve.
    ldp : ndarray
        Limb darkening profiles, shape (npv, npb, nmu).
    ldg : ndarray
        LD profile and its derivatives, shape (1+nldc, nmu).
        Row 0: dI/dmu, rows 1..: dI/dc_j.
    ldi : ndarray
        Disk-integrated intensity, shape (npv, npb).
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
    npb : int
        Number of passbands.
    nep : int
        Number of epochs.

    Returns
    -------
    flux : ndarray, shape (npv, npt)
        Model flux.
    dflux : ndarray, shape (npv, npt, 7 + nldc)
        Derivatives w.r.t. [k, t0, p, a, i, e, w, c_0, c_1, ...].
    """
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]
    nldc = ldg.shape[0] - 1

    flux = zeros((npv, npt))
    dflux = zeros((npv, npt, 7 + nldc))

    for ipv in range(npv):
        if isnan(a[ipv, 0]) or (a[ipv, 0] <= 1.0) or (e[ipv, 0] < 0.0) or any(isnan(ldp[ipv, 0])):
            flux[ipv, :] = nan
            dflux[ipv, :, :] = nan
            continue

        ldi_val = ldi[ipv, 0]

        # Pre-compute LD means per passband
        ldm_all = zeros((npb, ng))
        for ipb in range(npb):
            kv = k[ipv, ipb]
            if kmin <= kv <= kmax:
                ik = int(floor((kv - kmin) / dk))
                ak = (kv - kmin - ik * dk) / dk
                ldm_all[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb]) + ak * dot(weights[ik + 1], ldp[ipv, ipb])
            else:
                _, _, wg = calculate_weights_2d(kv, z_edges, ng)
                ldm_all[ipb, :] = dot(wg, ldp[ipv, ipb])

        # LD coefficient derivatives: dldm_dc[j] = dot(W, ldg[j+1])
        dldm_dc = zeros((nldc, ng))
        kv = k[ipv, 0]
        if kmin <= kv <= kmax:
            ik = int(floor((kv - kmin) / dk))
            ak = (kv - kmin - ik * dk) / dk
            for j in range(nldc):
                dldm_dc[j, :] = (1.0 - ak) * dot(weights[ik], ldg[j + 1]) + ak * dot(weights[ik + 1], ldg[j + 1])
        else:
            _, _, wg = calculate_weights_2d(kv, z_edges, ng)
            for j in range(nldc):
                dldm_dc[j, :] = dot(wg, ldg[j + 1])

        # Pre-compute orbital coefficients and derivatives per epoch
        xyc = zeros((nep, 2, 5))
        dxyc = zeros((nep, 6, 2, 5))
        for iep in range(nep):
            xyc[iep], dxyc[iep] = solve_xy_p5_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        # Bounding box (using first epoch)
        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

        # Calculate the light curve & grads
        for ipt in prange(npt):
            ilc = lcids[ipt]
            ipb = pbids[ilc]
            itc = epids[ilc]
            if nep > 1:
                iep = epids[ilc]
            else:
                iep = 0

            t = _folded_time(times[ipt], t0[ipv, itc], p[ipv, iep])
            if not ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                flux[ipv, ipt] = 1.0
            else:
                kpb = k[ipv, ipb]
                ldi_pb = ldi[ipv, ipb]

                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    t_eval = t + time_offset

                    z, dz = pd_t15c_d(t_eval, xyc[iep], dxyc[iep])
                    aplanet, (dadk, dadz) = ccia_and_grad(1.0, kpb, z)
                    g = z / (1.0 + kpb)
                    iplanet, dIp_dg = interpolate_mean_limb_darkening_and_grad(g, dg, ldm_all[ipb])

                    flux[ipv, ipt] += (ldi_pb - iplanet * aplanet) / ldi_pb

                    # dI_p/dz = (dI_p/dg) / (1+k)
                    dIp_dz = dIp_dg / (1.0 + kpb)

                    # --- k derivative (dldm/dk ≈ 0) ---
                    dIp_dk = dIp_dg * (-z / (1.0 + kpb) ** 2)
                    dflux[ipv, ipt, 0] += -(dIp_dk * aplanet + iplanet * dadk) / ldi_pb

                    # --- t0 derivative ---
                    dflux[ipv, ipt, 1] += (dIp_dz * aplanet + iplanet * dadz) * dz[0] / ldi_pb

                    # --- p, a, i, e, w derivatives (indices 1..5 in dz) ---
                    for ip in range(1, 6):
                        dflux[ipv, ipt, ip + 1] += -(dIp_dz * aplanet + iplanet * dadz) * dz[ip] / ldi_pb

                    # --- LD coefficient derivatives ---
                    for j in range(nldc):
                        dIp_dcj = interpolate_mean_limb_darkening(g, dg, dldm_dc[j])
                        dflux[ipv, ipt, 7 + j] += -aplanet * dIp_dcj / ldi_pb

                flux[ipv, ipt] /= nsamples[ilc]
                for ip in range(7 + nldc):
                    dflux[ipv, ipt, ip] /= nsamples[ilc]

                # Add the dldi contribution for LD coefficients
                for j in range(nldc):
                    dflux[ipv, ipt, 7 + j] += (1.0 - flux[ipv, ipt]) * dldi[j] / ldi_val

    return flux, dflux
