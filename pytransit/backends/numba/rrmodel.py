from meepmeep.backends.numba.ts2d import solve_xy_p5, pd_t15c
from meepmeep.backends.numba.ts2d.position import bounding_box
from numba import njit, prange
from numpy import ndarray, zeros, sqrt, linspace, nan, floor, isnan, full, dot, any

from .ccintersection import ccia
from ._utils import _folded_time


@njit
def create_z_grid(zcut: float, nin: int, nedge: int):
    mucut = sqrt(1.0 - zcut ** 2)
    dz = zcut / nin
    dmu = mucut / nedge

    z_edges = zeros(nin + nedge)
    z_means = zeros(nin + nedge)

    for i in range(nin - 1):
        z_edges[i] = (i + 1) * dz

    for i in range(nedge + 1):
        z_edges[-i - 1] = sqrt(1 - (i * dmu) ** 2)

    for i in range(nin + nedge - 1):
        z_means[i + 1] = 0.5 * (z_edges[i] + z_edges[i + 1])

    return z_edges, z_means


@njit
def calculate_weights_2d(k: float, ze: ndarray, ng: int):
    """Calculate a 2D limb darkening weight array.

    Parameters
    ----------
    k: float
        Radius ratio
    ng: int
        Grazing parameter resolution
    nmu: int
        Mu resolution

    Returns
    -------

    """
    gs = linspace(0, 1 - 1e-7, ng)
    nz = ze.size
    weights = zeros((ng, nz))

    for ig in range(ng):
        b = gs[ig] * (1.0 + k)
        a0 = ccia(ze[0], k, b)
        weights[ig, 0] = a0
        s = weights[ig, 0]
        for i in range(1, nz):
            a1 = ccia(ze[i], k, b)
            weights[ig, i] = a1 - a0
            a0 = a1
            s += weights[ig, i]
        for i in range(nz):
            weights[ig, i] /= s
    return gs, gs[1] - gs[0], weights


@njit
def calculate_weights_3d(nk: int, k0: float, k1: float, ze: ndarray, ng: int):
    """Calculate a 3D limb darkening weight array.

    Parameters
    ----------
    k: float
        Radius ratio
    ng: int
        Grazing parameter resolution
    nmu: int
        Mu resolution

    Returns
    -------

    """
    ks = linspace(k0, k1, nk)
    gs = linspace(0., 1. - 1e-7, ng)
    nz = ze.size
    weights = zeros((nk, ng, nz))

    for ik in range(nk):
        for ig in range(ng):
            b = gs[ig] * (1.0 + ks[ik])
            a0 = ccia(ze[0], ks[ik], b)
            weights[ik, ig, 0] = a0
            s = weights[ik, ig, 0]
            for i in range(1, nz):
                a1 = ccia(ze[i], ks[ik], b)
                weights[ik, ig, i] = a1 - a0
                a0 = a1
                s += weights[ik, ig, i]
            for i in range(nz):
                weights[ik, ig, i] /= s
    return (k1-k0)/nk, gs[1] - gs[0], weights


@njit(fastmath=True)
def interpolate_mean_limb_darkening(g, dg, lda):
    if g < 0.0:
        return nan
    if g > 1.0:
        return 0.0
    i = int(floor(g / dg))
    a = (g - i*dg) / dg
    return (1.0 - a) * lda[i] + a * lda[i + 1]


@njit(fastmath=True)
def interpolate_mean_limb_darkening_and_grad(g, dg, lda):
    if g < 0.0:
        return nan, 0.0
    if g > 1.0:
        return 0.0, 0.0
    ig = int(floor(g / dg))
    a = (g - ig * dg) / dg
    value = (1.0 - a) * lda[ig] + a * lda[ig + 1]
    dvalue = (lda[ig + 1] - lda[ig]) / dg
    return value, dvalue


def rrmodel(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray,
            ldp: ndarray, ldg: ndarray, ldi: ndarray, dldi: ndarray,
            weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray,
            npb: int, nep: int):
    """RoadRunner transit model supporting heterogeneous light curves.

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
        LD profile derivatives (unused in forward model).
    ldi : ndarray
        Disk-integrated intensity, shape (npv, npb).
    dldi : ndarray
        Derivative of ldi (unused in forward model).
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
    """
    npv = k.shape[0]
    npt = times.size
    ng = weights.shape[1]

    flux = zeros((npv, npt))

    for ipv in range(npv):
        if isnan(a[ipv, 0]) or (a[ipv, 0] <= 1.0) or (e[ipv, 0] < 0.0) or any(isnan(ldp[ipv, 0])):
            flux[ipv, :] = nan
            continue

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

        # Pre-compute orbital coefficients per epoch
        xyc = zeros((nep, 2, 5))
        for iep in range(nep):
            xyc[iep, :, :] = solve_xy_p5(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        # Bounding box (using first epoch)
        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

        # Calculate the light curve
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
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = pd_t15c(t + time_offset, xyc[iep])
                    iplanet = interpolate_mean_limb_darkening(z / (1.0 + k[ipv, ipb]), dg, ldm_all[ipb])
                    aplanet = ccia(1.0, k[ipv, ipb], z)
                    flux[ipv, ipt] += (ldi[ipv, ipb] - iplanet * aplanet) / ldi[ipv, ipb]
                flux[ipv, ipt] /= nsamples[ilc]
    return flux
