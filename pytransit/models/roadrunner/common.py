from numba import njit
from numpy import sqrt, arctan2, pi, nan, zeros, floor, arccos, linspace, ndarray


@njit
def tsort(r1, r2, b):
    if r1 > r2:
        if r1 > b:
            x = r1
            if r2 > b:
                y = r2
                z = b
            else:
                y = b
                z = r2
        else:
            x = b
            y = r1
            z = r2
    else:
        if r2 > b:
            x = r2
            if r1 > b:
                y = r1
                z = b
            else:
                y = b
                z = r1
        else:
            x = b
            y = r2
            z = r1
    return x, y, z


@njit
def circle_circle_intersection_area(r1, r2, b):
    """Area of the intersection of two circles.
    """
    if r1 < b - r2:
        return 0.0
    elif r1 >= b + r2:
        return pi * r2 ** 2
    elif b - r2 <= -r1:
        return pi * r1 ** 2
    else:
        return (r2 ** 2 * arccos((b ** 2 + r2 ** 2 - r1 ** 2) / (2 * b * r2)) +
                r1 ** 2 * arccos((b ** 2 + r1 ** 2 - r2 ** 2) / (2 * b * r1)) -
                0.5 * sqrt((-b + r2 + r1) * (b + r2 - r1) * (b - r2 + r1) * (b + r2 + r1)))


@njit
def circle_circle_intersection_area_kite(r1, r2, b):
    """Circle-circle intersection routine adapted from Agol et al. (2020)

    Circle-circle intersection routine adapted from Agol et al. (2020). The only
    major change is that the radius of the first circle is also a a free parameter.
    """
    if r1 + r2 <= b:
        return 0.0, 0.0
    elif abs(r1 - r2) < b and b <= r1 + r2:
        x, y, z = tsort(r1, r2, b)
        a_kite = 0.5 * sqrt((x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z)))
        k0 = arctan2(2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b)
        k1 = arctan2(2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b)
        a_lens = r1 * r1 * k1 + r2 * r2 * k0 - a_kite
        return a_lens, k0
    elif b <= r1 - r2:
        return pi * r2 ** 2, pi
    elif b <= r2 - r1:
        return pi * r1 ** 2, 0.0
    else:
        return nan, nan


@njit
def circle_circle_intersection_area_kite_v(r1, r2, b):
    n = r1.size
    a = zeros(n)
    k0 = zeros(n)
    for i in range(n):
        a[i], k0[i] = circle_circle_intersection_area_kite(r1[i], r2[i], b[i])
    return a, k0


@njit
def cciad_s(z, dz, r1, r2):
    """Circle-circle intersection area derivative with respect to z."""
    if r1 < z - r2:
        return 0.0
    elif r1 >= z + r2:
        return 0.0
    elif z - r2 <= -r1:
        return 0.0
    else:
        a = z**2 + r2**2 - r1**2
        b = z**2 + r1**2 - r2**2
        t1 = - r2**2*(1/r2 - a/(2*r2*z**2))/sqrt(1 - a**2/(4*r2**2*z**2))
        t2 = - r1**2*(1/r1 - b/(2*r1*z**2))/sqrt(1 - b**2/(4*r1**2*z**2))
        t3 = z*(r1**2 + r2**2 - z**2)/sqrt((-z + r2 + r1)*(z + r2 - r1)*(z - r2 + r1)*(z + r2 + r1))
        return dz*(t1 + t2 - t3)


@njit
def dfdk(k, b, k0, lda, dg, ist):
    if b < 1.0+k-1e-5:
        g =  b / (1.0+k)
        ig = int(floor(g / dg))
        ag = g - ig*dg
        l = (1.0-ag)*lda[ig] + ag*lda[ig+1]
        return -2.0*k*k0*l/ist
    else:
        return 0.0


@njit
def dfdb(k, b, a, ak, lda, dg, ist):
    if b < 0.005 or b >= 1.0+k-1e-5:
        return 0.0
    else:
        g = b / (1.0+k)
        ig = int(floor(g / dg))
        ag = g - ig*dg
        l1 = lda[ig]
        l2 = lda[ig+1]
        l = (1.-ag)*l1 + ag*l2
        dldb = -(l2-l1) / (dg * (1+k))
        return 2 * ak * l / (b * ist) + dldb * a / ist


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
        a0 = circle_circle_intersection_area(ze[0], k, b)
        weights[ig, 0] = a0
        s = weights[ig, 0]
        for i in range(1, nz):
            a1 = circle_circle_intersection_area(ze[i], k, b)
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
            a0 = circle_circle_intersection_area(ze[0], ks[ik], b)
            weights[ik, ig, 0] = a0
            s = weights[ik, ig, 0]
            for i in range(1, nz):
                a1 = circle_circle_intersection_area(ze[i], ks[ik], b)
                weights[ik, ig, i] = a1 - a0
                a0 = a1
                s += weights[ik, ig, i]
            for i in range(nz):
                weights[ik, ig, i] /= s
    return (k1-k0)/nk, gs[1] - gs[0], weights

@njit(fastmath=True)
def interpolate_mean_limb_darkening_s(g, dg, lda):
    if g < 0.0:
        return nan
    if g > 1.0:
        return 0.0
    i = int(floor(g / dg))
    a = (g - i*dg) / dg
    return (1.0 - a) * lda[i] + a * lda[i + 1]

@njit(fastmath=True)
def interpolate_mean_limb_darkening_v(gs, dg, lda):
    r = zeros(gs.size)
    for i in range(gs.size):
        r[i] = interpolate_mean_limb_darkening_s(gs[i], dg, lda)
    return r