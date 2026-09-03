from numba import njit
from typing import Optional

from numpy import (sqrt, sin, arctan2, pi, nan, zeros, floor, arccos, linspace, ndarray, atleast_1d,
                   atleast_2d, asarray, full, empty, concatenate, array, broadcast_to)
from numpy.polynomial import Polynomial
from scipy.special import roots_jacobi, roots_legendre


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
def create_z_grid(nz: int):
    """Discretise the stellar disk into annuli for the RoadRunner weight tables.

    The annulus edges are uniform in the angle between the surface normal and the line of
    sight, z = sin(gamma), which packs the annuli towards the limb, where the intensity profile
    steepens, while keeping their width finite at the disk centre, where the planet itself sets
    the resolution needed. Each annulus is sampled at its area-weighted mean mu, which makes
    the sampling exact for any intensity profile linear in mu, the dominant term of every
    limb darkening law.

    Parameters
    ----------
    nz : int
        Number of annuli.

    Returns
    -------
    z_edges : ndarray
        Outer edge of each annulus, the last one being 1.
    z_means : ndarray
        Normalised distance at which each annulus samples the intensity profile.
    """
    z_edges = sin(linspace(0.0, 0.5 * pi, nz + 1)[1:])
    z_inner = zeros(nz)
    z_inner[1:] = z_edges[:-1]
    mu_outer = sqrt(1.0 - z_inner ** 2)
    mu_inner = sqrt(1.0 - z_edges ** 2)
    mu_mean = (2.0 / 3.0) * (mu_outer ** 3 - mu_inner ** 3) / (mu_outer ** 2 - mu_inner ** 2)
    return z_edges, sqrt(1.0 - mu_mean ** 2)


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
    # The table nodes are `linspace(k0, k1, nk)`, so their spacing is (k1 - k0) / (nk - 1); the
    # consumers index the table with this spacing, and the last node must stay inside it.
    return (k1 - k0) / (nk - 1), gs[1] - gs[0], weights

@njit
def weight_table_index(k: float, kmin: float, dk: float, nk: int):
    """Locate a radius ratio in the weight table for linear interpolation.

    Returns the index of the lower bracketing node and the interpolation weight of the upper
    one. The upper table limit is included: a radius ratio on the last node interpolates fully
    onto it instead of indexing past the end of the table.
    """
    ik = int(floor((k - kmin) / dk))
    if ik >= nk - 1:
        return nk - 2, 1.0
    return ik, (k - kmin - ik * dk) / dk


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

@njit(fastmath=True)
def interpolate_limb_darkening_s(z, zm, ldp):
    """Interpolate a tabulated limb darkening profile at a normalized distance z.

    Linear interpolation of the limb darkening profile over the (non-uniform) profile node
    grid `zm`, clamped to the innermost node inside the central annulus and to the last node
    beyond the edge of the grid. Used by the small-planet
    approximation, where the mean intensity blocked by the planet is approximated by the
    stellar intensity at the planet's center.
    """
    if z < 0.0:
        return nan
    if z >= zm[-1]:
        return ldp[-1]
    if z <= zm[0]:
        return ldp[0]
    i = zm.size // 2
    if z > zm[i]:
        while z > zm[i + 1]:
            i += 1
    else:
        while z < zm[i]:
            i -= 1
    a = (z - zm[i]) / (zm[i + 1] - zm[i])
    return (1.0 - a) * ldp[i] + a * ldp[i + 1]


def population_arrays(t0, p, a, i, e, w, epochs: bool = True, npv: Optional[int] = None):
    """Broadcast the orbital parameters of a population to the shapes the kernels index.

    The kernels index the orbital parameters per parameter vector, and the zero epoch per
    parameter vector and epoch, so scalars and one-dimensional zero epochs are expanded here
    instead of being read past their ends in compiled code.

    Parameters
    ----------
    t0, p, a, i, e, w
        The orbital parameters as scalars or arrays.
    epochs : bool, optional
        Return the zero epochs as a 2D ``(npv, nep)`` array rather than a 1D ``(npv,)`` one.
    npv : int, optional
        Number of parameter vectors. Defaults to the size of `p`; the transmission spectroscopy
        model sets it from the radius ratio array instead.

    Returns
    -------
    tuple of ndarray
        ``(t0, p, a, i, e, w)`` with `p`, `a`, `i`, `e` and `w` as ``(npv,)`` arrays.
    """
    p, a, i, e, w = (atleast_1d(x).astype(float) for x in (p, a, i, e, w))
    npv = p.size if npv is None else npv
    p, a, i, e, w = (full(npv, x[0]) if (x.size == 1 and npv > 1) else x for x in (p, a, i, e, w))
    t0 = asarray(t0, dtype=float)
    if epochs:
        t0 = t0.reshape((npv, -1)) if npv > 1 else atleast_2d(t0)
    else:
        t0 = atleast_1d(t0)
    return t0, p, a, i, e, w


def radius_ratio_array(k, npv: int) -> ndarray:
    """Normalise the radius ratio(s) into the ``(npv, nk)`` array the kernels index.

    The kernels index the radius ratios per parameter vector and passband, so the many
    shapes `evaluate` accepts are expanded into a single canonical one here. The rule
    mirrors the one used for the limb darkening coefficients: a one-dimensional array is
    read as ``(nk,)`` for a single parameter vector, and as ``(npv,)`` -- one radius ratio
    per parameter vector -- for a population. Pass an explicit ``(npv, nk)`` array to give
    a population several radius ratios per parameter vector.

    Parameters
    ----------
    k
        Radius ratio(s) as a scalar, a 1D array, or a 2D ``(npv, nk)`` array.
    npv : int
        Number of parameter vectors, taken from the orbital parameters.

    Returns
    -------
    ndarray
        The radius ratios as a ``(npv, nk)`` array.

    Raises
    ------
    ValueError
        If `k` has more than two dimensions, or if its leading dimension matches neither
        `npv` nor one.
    """
    k = atleast_1d(k).astype(float)

    if k.ndim == 1:
        # A 1D array is ambiguous: it is a set of radius ratios for a single parameter
        # vector, or one radius ratio for each vector of a population.
        k = k.reshape((1, -1)) if npv == 1 else k.reshape((-1, 1))
    elif k.ndim > 2:
        raise ValueError(f"The radius ratio array can be at most two-dimensional, got {k.ndim} dimensions.")

    if k.shape[0] == 1 and npv > 1:
        k = broadcast_to(k, (npv, k.shape[1])).copy()
    elif k.shape[0] != npv:
        raise ValueError(f"Expected the radius ratios for {npv} parameter vectors, got an array with "
                         f"a leading dimension of {k.shape[0]}. Give the radius ratios either as a "
                         f"({npv}, nk) array or as a single radius ratio per parameter vector.")
    return k


# ----------------------------------------------------------------------------------------------
# Mean intensity under the planet by quadrature
# ----------------------------------------------------------------------------------------------
#
# The mean stellar intensity under the planet at grazing parameter g = b / (1 + k) is
#
#     ldm(g) = int I(z) theta(z) z dz / int theta(z) z dz,
#
# where theta(z) is the angular extent of the planet disk at stellar radius z. Wherever the
# integrand has a square-root endpoint -- theta vanishing like sqrt at a contact, or the profile
# behaving like sqrt(1 - z) at the limb -- the quadrature is matched to it, either by a Gauss-Jacobi
# rule with the square root in its weight function or by substituting z = z0 +/- s**2, which turns
# the square root into a regular function of s:
#
# - planet fully on the disk: z = (b + k) - s**2 makes the limb end regular (both theta's zero and
#   the profile's sqrt), and Gauss-Jacobi absorbs theta's sqrt zero at z = b - k;
# - star centre inside the planet: theta is 2 pi from z = 0 to z = k - b (regular, Gauss-Legendre
#   in mu), then z = (k - b) + s**2 makes that end regular and Gauss-Jacobi absorbs the zero at
#   z = b + k;
# - planet crossing the limb: integrated in mu, where every built-in law is polynomial, so the limb
#   end is regular and Gauss-Jacobi absorbs the sqrt zero at the inner end.
#
# The profile is read from a table on a fixed grid, so a profile source only ever has to be
# evaluated on that grid.

def quadrature_rules(nq: int) -> ndarray:
    """The two fixed quadrature rules on [-1, 1] used by `ldm_nodes`, as a ``(2, 2, nq)`` array.

    ``rules[0]`` is Gauss-Legendre and ``rules[1]`` is Gauss-Jacobi with the weight function
    ``(1 - t)**0.5`` folded into the weights, for integrands with a square-root zero at t = 1.
    Each holds the nodes in row 0 and the weights in row 1.
    """
    t_gl, w_gl = roots_legendre(nq)
    t_gj, w_gj = roots_jacobi(nq, 0.5, 0.0)
    w_gj = w_gj / sqrt(1.0 - t_gj)
    return array([[t_gl, w_gl], [t_gj, w_gj]])


@njit(fastmath=True)
def planet_angular_extent(z: float, b: float, k: float) -> float:
    """Angular extent of the planet disk at stellar radius z for a planet at separation b."""
    if b < 1e-12:
        return 2.0 * pi if z < k else 0.0
    if z <= k - b:
        return 2.0 * pi
    if z < b - k or z > b + k:
        return 0.0
    c = (z * z + b * b - k * k) / (2.0 * z * b)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return 2.0 * arccos(c)


@njit(fastmath=True)
def ldm_nodes(k: float, gs: ndarray, rules: ndarray, mu: ndarray, wf: ndarray) -> None:
    """Quadrature nodes (as mu) and geometric factors for the mean intensity under the planet.

    Fills ``mu[ig, q]`` and ``wf[ig, q]`` for every grazing parameter in `gs` so that the mean
    intensity is ``sum(wf * I(mu)) / sum(wf)``. The factors include the quadrature weight,
    the planet's angular extent and the Jacobian of the integration variable, and they sum to
    the planet-star overlap area. Unused columns are zero-weighted.
    """
    nq = rules.shape[2]
    for ig in range(gs.size):
        # At g = 1 the overlap vanishes; evaluate the geometry just inside so the mean is defined.
        b = min(gs[ig], 1.0 - 1e-9) * (1.0 + k)
        for q in range(2 * nq):
            wf[ig, q] = 0.0
            mu[ig, q] = 1.0
        if b + k <= 1.0:
            if b < k:
                # z in [0, k - b]: theta = 2 pi. Integrated in mu with Gauss-Legendre.
                mu_mid = sqrt(1.0 - (k - b) ** 2)
                c, h = 0.5 * (mu_mid + 1.0), 0.5 * (1.0 - mu_mid)
                for q in range(nq):
                    m = c + h * rules[0, 0, q]
                    z = sqrt(1.0 - m * m)
                    mu[ig, q] = m
                    wf[ig, q] = rules[0, 1, q] * planet_angular_extent(z, b, k) * m * h
                # z in [k - b, b + k] with z = (k - b) + s**2: regular at s = 0, sqrt zero at the end.
                sm = sqrt(2.0 * b)
                c = h = 0.5 * sm
                for q in range(nq):
                    sq = c + h * rules[1, 0, q]
                    z = k - b + sq * sq
                    mu[ig, nq + q] = sqrt(1.0 - z * z)
                    wf[ig, nq + q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * z * 2.0 * sq * h
            else:
                # z in [b - k, b + k] with z = (b + k) - s**2: the limb end is regular, and the
                # sqrt zero of theta at z = b - k sits at s_max, where the Jacobi weight takes it.
                sm = sqrt(2.0 * k)
                c = h = 0.5 * sm
                for q in range(nq):
                    sq = c + h * rules[1, 0, q]
                    z = b + k - sq * sq
                    mu[ig, q] = sqrt(1.0 - z * z)
                    wf[ig, q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * z * 2.0 * sq * h
        else:
            # z in [b - k, 1]: integrated in mu from the limb, where the profile is regular, to
            # mu_lo, where theta has its sqrt zero.
            mu_lo = sqrt(1.0 - (b - k) ** 2)
            c = h = 0.5 * mu_lo
            for q in range(nq):
                m = c + h * rules[1, 0, q]
                z = sqrt(1.0 - m * m)
                mu[ig, q] = m
                wf[ig, q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * m * h


@njit
def valid_radius_ratios(k: ndarray) -> bool:
    """Whether every radius ratio is in (0, 1], the range the mean intensity tables can be built for.

    A NaN, non-positive or above-one radius ratio would leave the quadrature with an empty or
    inverted footprint, so a parameter vector holding one is treated as invalid and evaluates to
    NaN fluxes, as one with a bad semi-major axis or eccentricity does.
    """
    for i in range(k.size):
        if not (0.0 < k[i] <= 1.0):
            return False
    return True


@njit(fastmath=True)
def split_point(k: float) -> float:
    """The grazing parameter at which the planet first touches the stellar limb, (1 - k) / (1 + k).

    The mean intensity under the planet has a kink here, so the g table is split at it.
    """
    return (1.0 - k) / (1.0 + k)


@njit
def g_nodes(k: float, ng: int):
    """Grazing parameter nodes for the mean intensity table: two uniform segments meeting at the limb contact.

    Returns the nodes and the number of nodes in the first segment. The segments share the
    split node, and each has at least four nodes so that a cubic can be fitted in it.
    """
    gc = split_point(k)
    n1 = int(round(ng * gc))
    if n1 < 4:
        n1 = 4
    if ng - n1 < 4:
        n1 = ng - 4
    gs = empty(ng)
    gs[:n1] = linspace(0.0, gc, n1)
    gs[n1:] = linspace(gc, 1.0, ng - n1)
    return gs, n1


@njit(fastmath=True)
def profile_at(m: float, t0: float, dt: float, ldp: ndarray) -> float:
    """Cubic interpolation of a profile tabulated on the grid ``mu = (t0 + dt * arange(n))**2``.

    The table is uniform in ``sqrt(mu)`` rather than in mu, which keeps the interpolation
    accurate for laws with fractional powers of mu near the limb. Below the first node the
    profile is held at its first value.
    """
    n = ldp.size
    x = (sqrt(m) - t0) / dt
    if x <= 0.0:
        return ldp[0]
    if x >= n - 1:
        return ldp[n - 1]
    i = int(x) - 1
    if i < 0:
        i = 0
    elif i > n - 4:
        i = n - 4
    u = x - i
    return (-(u - 1) * (u - 2) * (u - 3) / 6.0 * ldp[i] + u * (u - 2) * (u - 3) / 2.0 * ldp[i + 1]
            - u * (u - 1) * (u - 3) / 2.0 * ldp[i + 2] + u * (u - 1) * (u - 2) / 6.0 * ldp[i + 3])


def profile_grid(nmu: int = 200, t0: float = 0.01):
    """The fixed mu grid the profiles are tabulated on: uniform in sqrt(mu) from `t0` to 1.

    Returns the mu values, the first sqrt(mu) node and the sqrt(mu) spacing. The grid starts
    slightly above mu = 0 because some laws are singular exactly at the limb.
    """
    ts = linspace(t0, 1.0, nmu)
    return ts ** 2, t0, ts[1] - ts[0]


@njit(fastmath=True)
def ldm_table(mu: ndarray, wf: ndarray, t0: float, dt: float, ldp: ndarray, out: ndarray) -> None:
    """The mean intensity under the planet at every g, from the nodes and a tabulated profile."""
    for ig in range(mu.shape[0]):
        num = 0.0
        den = 0.0
        for q in range(mu.shape[1]):
            w = wf[ig, q]
            if w != 0.0:
                num += w * profile_at(mu[ig, q], t0, dt, ldp)
                den += w
        out[ig] = num / den


def _cubic_matrices() -> ndarray:
    """Maps from four stencil values to the cubic's coefficients in the local interval variable.

    ``M[s]`` serves the interval whose left node sits at position ``s`` in a four-node stencil
    at unit spacing, for ``s`` in 0, 1, 2.
    """
    M = zeros((3, 4, 4))
    for s in range(3):
        for j in range(4):
            lag = Polynomial([1.0])
            for m in range(4):
                if m != j:
                    lag = lag * Polynomial([-m, 1.0]) / (j - m)
            shifted = lag(Polynomial([s, 1.0]))          # u = s + a
            M[s, :, j] = shifted.coef[:4] if shifted.coef.size >= 4 else concatenate([shifted.coef, zeros(4 - shifted.coef.size)])
    return M


CUBIC_MATRICES = _cubic_matrices()


@njit(fastmath=True)
def cubic_coefficients(tab: ndarray) -> ndarray:
    """Per-interval cubic polynomial coefficients for a table on a uniform grid.

    Interval ``i`` covers nodes ``i`` and ``i + 1``; its value at local position ``a`` in [0, 1)
    is ``c[i, 0] + a * (c[i, 1] + a * (c[i, 2] + a * c[i, 3]))``. The cubic is the four-point
    Lagrange interpolant on a stencil shifted inwards at the ends, so the interpolation is fourth
    order everywhere including the first and last intervals.
    """
    n = tab.size
    c = zeros((n - 1, 4))
    for i in range(n - 1):
        j = i - 1
        if j < 0:
            j = 0
        elif j > n - 4:
            j = n - 4
        s = i - j
        for r in range(4):
            c[i, r] = (CUBIC_MATRICES[s, r, 0] * tab[j] + CUBIC_MATRICES[s, r, 1] * tab[j + 1]
                       + CUBIC_MATRICES[s, r, 2] * tab[j + 2] + CUBIC_MATRICES[s, r, 3] * tab[j + 3])
    return c


@njit(fastmath=True)
def ldm_lookup(g: float, gc: float, n1: int, coef: ndarray) -> float:
    """The mean intensity under the planet at grazing parameter g from the split cubic table.

    `coef` holds the interval coefficients of both segments back to back: ``n1 - 1`` intervals
    covering [0, gc] and the rest covering [gc, 1].
    """
    if g < gc:
        x = g / gc * (n1 - 1)
        i = int(x)
        if i > n1 - 2:
            i = n1 - 2
    else:
        n2 = coef.shape[0] - (n1 - 1) + 1
        x = (g - gc) / (1.0 - gc) * (n2 - 1)
        i = int(x)
        if i > n2 - 2:
            i = n2 - 2
        x += n1 - 1
        i += n1 - 1
    a = x - i
    return coef[i, 0] + a * (coef[i, 1] + a * (coef[i, 2] + a * coef[i, 3]))



@njit(fastmath=True)
def split_cubic_coefficients(tab: ndarray, n1: int, out: ndarray) -> None:
    """Cubic coefficients of a g table split at node ``n1 - 1``, written back to back into `out`.

    `out` has ``tab.size - 2`` rows: ``n1 - 1`` intervals for the first segment followed by the
    second segment's intervals, which is the layout `ldm_lookup` reads.
    """
    # The split node is stored twice, once as the last node of the first segment and once as
    # the first node of the second, so the second segment starts at index n1.
    out[:n1 - 1] = cubic_coefficients(tab[:n1])
    out[n1 - 1:] = cubic_coefficients(tab[n1:])
