from meepmeep.numba2d import solve2d_d, sep_cd, bounding_box
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
        Per-passband LD profile and its derivatives, shape (npv, npb, 1+nldc, nmu).
        Row 0: dI/dmu, rows 1..: dI/dc_j.
    ldi : ndarray
        Disk-integrated intensity, shape (npv, npb).
    dldi : ndarray
        Derivative of ldi w.r.t. each LD coefficient, shape (npv, npb, nldc).
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
    dflux : ndarray, shape (npv, npt, 7 + npb*nldc)
        Derivatives w.r.t. [k, t0, p, a, i, e, w] followed by one nldc-wide
        block of LD-coefficient derivatives per passband. The LD derivatives of
        a data point are non-zero only within its own passband's block; the
        derivative of coefficient j of passband ipb sits at slot 7 + ipb*nldc + j.
    """
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]
    nldc = ldg.shape[2] - 1

    flux = zeros((npv, npt))
    dflux = zeros((npv, npt, 7 + npb * nldc))

    for ipv in range(npv):
        if isnan(a[ipv, 0]) or (a[ipv, 0] <= 1.0) or (e[ipv, 0] < 0.0) or any(isnan(ldp[ipv, 0])):
            flux[ipv, :] = nan
            dflux[ipv, :, :] = nan
            continue

        # Pre-compute LD means and their LD-coefficient derivatives per passband.
        # Each passband uses its own radius ratio and limb darkening profile, so
        # dldm_dc[ipb, j] = dot(W(k_ipb), dI_ipb/dc_j).
        ldm_all = zeros((npb, ng))
        dldm_dc = zeros((npb, nldc, ng))
        for ipb in range(npb):
            kv = k[ipv, ipb]
            if kmin <= kv <= kmax:
                ik = int(floor((kv - kmin) / dk))
                ak = (kv - kmin - ik * dk) / dk
                ldm_all[ipb, :] = (1.0 - ak) * dot(weights[ik], ldp[ipv, ipb]) + ak * dot(weights[ik + 1], ldp[ipv, ipb])
                for j in range(nldc):
                    dldm_dc[ipb, j, :] = (1.0 - ak) * dot(weights[ik], ldg[ipv, ipb, j + 1]) + ak * dot(weights[ik + 1], ldg[ipv, ipb, j + 1])
            else:
                _, _, wg = calculate_weights_2d(kv, z_edges, ng)
                ldm_all[ipb, :] = dot(wg, ldp[ipv, ipb])
                for j in range(nldc):
                    dldm_dc[ipb, j, :] = dot(wg, ldg[ipv, ipb, j + 1])

        # Pre-compute orbital coefficients and derivatives per epoch
        xyc = zeros((nep, 2, 5))
        dxyc = zeros((nep, 7, 2, 5))
        for iep in range(nep):
            xyc[iep], dxyc[iep] = solve2d_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

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

            pv = p[ipv, iep]
            # Integer transit epoch, matching the folding in `_folded_time`. The
            # folded time t_eval = t - t0 - epoch*p depends on the period through
            # this -epoch*p term, which the period derivative must account for.
            epoch = floor((times[ipt] - t0[ipv, itc] + 0.5 * pv) / pv)
            t = _folded_time(times[ipt], t0[ipv, itc], pv)
            if not ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                flux[ipv, ipt] = 1.0
            else:
                kpb = k[ipv, ipb]
                ldi_pb = ldi[ipv, ipb]

                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    t_eval = t + time_offset

                    z, dz = sep_cd(t_eval, xyc[iep], dxyc[iep])
                    aplanet, (dadk, dadz) = ccia_and_grad(1.0, kpb, z)
                    g = z / (1.0 + kpb)
                    iplanet, dIp_dg = interpolate_mean_limb_darkening_and_grad(g, dg, ldm_all[ipb])

                    flux[ipv, ipt] += (ldi_pb - iplanet * aplanet) / ldi_pb

                    # dI_p/dz = (dI_p/dg) / (1+k)
                    dIp_dz = dIp_dg / (1.0 + kpb)

                    # --- k derivative (dldm/dk ≈ 0) ---
                    dIp_dk = dIp_dg * (-z / (1.0 + kpb) ** 2)
                    dflux[ipv, ipt, 0] += -(dIp_dk * aplanet + iplanet * dadk) / ldi_pb

                    # --- t0, p, a, i, e, w derivatives ---
                    # dz is now the 7-element (tc, p, a, i, e, w, lan) gradient from
                    # sep_cd; slot 0 is the proper transit-centre derivative, so all
                    # six orbital terms share the same sign (lan, slot 6, is unused).
                    dflux_dz = -(dIp_dz * aplanet + iplanet * dadz) / ldi_pb
                    for ip in range(6):
                        dflux[ipv, ipt, ip + 1] += dflux_dz * dz[ip]

                    # Period folding correction. The separation derivatives dz are
                    # taken at fixed folded time, but the folded time itself depends
                    # on the period via -epoch*p. Since d_flux/d_t_eval = -d_flux/d_t0,
                    # the total period derivative gains epoch times the transit-centre
                    # term (dz[0]); this vanishes for epoch 0 and grows with epoch.
                    dflux[ipv, ipt, 2] += epoch * dflux_dz * dz[0]

                    # --- LD coefficient derivatives (only this point's passband) ---
                    for j in range(nldc):
                        dIp_dcj = interpolate_mean_limb_darkening(g, dg, dldm_dc[ipb, j])
                        dflux[ipv, ipt, 7 + ipb * nldc + j] += -aplanet * dIp_dcj / ldi_pb

                flux[ipv, ipt] /= nsamples[ilc]
                for ip in range(7 + npb * nldc):
                    dflux[ipv, ipt, ip] /= nsamples[ilc]

                # Add the dldi contribution for this passband's LD coefficients
                for j in range(nldc):
                    dflux[ipv, ipt, 7 + ipb * nldc + j] += (1.0 - flux[ipv, ipt]) * dldi[ipv, ipb, j] / ldi_pb

    return flux, dflux
