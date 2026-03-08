from meepmeep.backends.numba.ts2d import solve_xy_p5, pd_t15c
from meepmeep.backends.numba.ts2d.position import bounding_box
from numba import njit
from numpy import ndarray, zeros, sqrt, linspace, nan, floor, isnan, full, dot, any

from .ccintersection import ccia


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


def rrmodel(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
            nsamples: ndarray, exptimes: ndarray, ldp: ndarray, ldg: ndarray, ldi: float, dldi: ndarray,
            weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray):
    """Simplified RoadRunner model for a single homogeneous light curve."""

    npt = times.size
    ng = weights.shape[1]

    ldm = zeros(ng)  # Limb darkening means
    xyc = zeros((2, 5))  # Taylor series coefficients for the (x, y) position

    if isnan(a) or (a <= 1.0) or (e < 0.0) or any(isnan(ldp[0])):
        return full(npt, nan)

    # ----------------------------------#
    # Calculate the limb darkening mean #
    # ----------------------------------#
    if kmin <= k <= kmax:
        ik = int(floor((k - kmin) / dk))
        ak = (k - kmin - ik * dk) / dk
        ldm[:] = (1.0 - ak) * dot(weights[ik], ldp[0,0]) + ak * dot(weights[ik + 1], ldp[0,0])
    else:
        _, _, wg = calculate_weights_2d(k, z_edges, ng)
        ldm[:] = dot(wg, ldp[0,0])

    # -----------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit #
    # -----------------------------------------------------#
    xyc[:, :] = solve_xy_p5(0.0, p, a, i, e, w)

    # ---------------------------#
    # Calculate the bounding box #
    # ---------------------------#
    bt1, bt4 = bounding_box(k, xyc)
    bt1 -= 0.003 + exptimes[0]
    bt4 += 0.003 + exptimes[0]

    # --------------------------#
    # Calculate the light curve #
    # --------------------------#
    flux = zeros(npt)
    for ipt in range(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if not (bt1 <= tc <= bt4):
            flux[ipt] = 1.0
        else:
            for isample in range(1, nsamples[0] + 1):
                time_offset = exptimes[0] * ((isample - 0.5) / nsamples[0] - 0.5)
                z = pd_t15c(tc + time_offset, xyc)
                iplanet = interpolate_mean_limb_darkening(z / (1.0 + k), dg, ldm)
                aplanet = ccia(1.0, k, z)
                flux[ipt] += (ldi[0,0] - iplanet * aplanet) / ldi[0,0]
            flux[ipt] /= nsamples[0]
    return flux
