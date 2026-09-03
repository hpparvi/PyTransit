from meepmeep.numba2d import solve2d, sep_c, bounding_box
from numba import njit, prange
from numpy import ndarray, zeros, sqrt, linspace, nan, floor, isnan, full, dot, any

from .ccintersection import ccia
from ._utils import _folded_time
from .ldquad import rr_tables, ldm_lookup


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
    return (k1 - k0) / (nk - 1), gs[1] - gs[0], weights


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
            rules: ndarray, pt0: float, pdt: float, ng: int, npb: int, nep: int):
    """RoadRunner transit model supporting heterogeneous light curves.

    The mean intensity under the planet is computed for each parameter vector and passband by
    the geometry-matched quadrature in `ldquad`, tabulated against the grazing parameter and
    read with a per-interval cubic.

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
        Limb darkening profiles on the `ldquad.profile_grid` mu grid, shape (npv, npb, nmu).
    ldg : ndarray
        LD profile derivatives (unused in forward model).
    ldi : ndarray
        Disk-integrated intensity, shape (npv, npb).
    dldi : ndarray
        Derivative of ldi (unused in forward model).
    rules : ndarray
        Quadrature rules from `ldquad.quadrature_rules`, shape (2, 2, nq).
    pt0 : float
        First sqrt(mu) node of the profile grid.
    pdt : float
        The sqrt(mu) spacing of the profile grid.
    ng : int
        Number of grazing parameter nodes.
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
    n1 = ng // 2
    flux = zeros((npv, npt))

    # Mean intensity under the planet as a split cubic table in g, per passband
    gcs = zeros(npb)
    coef = zeros((npb, ng - 2, 4))

    for ipv in range(npv):
        if isnan(a[ipv, 0]) or (a[ipv, 0] <= 1.0) or (e[ipv, 0] < 0.0) or any(isnan(ldp[ipv, 0])):
            flux[ipv, :] = nan
            continue

        rr_tables(k[ipv], ldp[ipv], rules, pt0, pdt, ng, gcs, coef)

        xyc = zeros((nep, 2, 5))
        for iep in range(nep):
            xyc[iep, :, :] = solve2d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

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
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = sep_c(t + time_offset, xyc[iep])
                    if z >= 1.0 + kpb:
                        flux[ipv, ipt] += 1.0
                    else:
                        iplanet = ldm_lookup(z / (1.0 + kpb), gcs[ipb], n1, coef[ipb])
                        aplanet = ccia(1.0, kpb, z)
                        flux[ipv, ipt] += (ldi[ipv, ipb] - iplanet * aplanet) / ldi[ipv, ipb]
                flux[ipv, ipt] /= nsamples[ilc]
    return flux
