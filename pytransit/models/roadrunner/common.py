from numba import njit
from typing import Optional

from numpy import (sqrt, sin, cos, arctan2, pi, nan, zeros, floor, arccos, linspace, ndarray, atleast_1d,
                   atleast_2d, asarray, full, empty, concatenate, array, broadcast_to, int64)
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
# rule with the square root in its weight function or by a substitution that turns the square root
# into a regular function: z = z0 +/- s**2 for one end, and z = z0 + (z1 - z0) sin(phi)**2 for both
# ends at once.
#
# - planet fully on the disk: z = (b + k) - s**2 makes the limb end regular (both theta's zero and
#   the profile's sqrt), and Gauss-Jacobi absorbs theta's sqrt zero at z = b - k;
# - star centre inside the planet: theta is 2 pi from z = 0 to z = k - b (regular, Gauss-Legendre
#   in mu), and the rest, from z = k - b to z = b + k, is integrated in phi with Gauss-Legendre;
# - planet crossing the limb: from z = |b - k| to the limb in phi with Gauss-Legendre, which is
#   regular both where theta has its square root and where the profile has its own at the limb.
#   A planet larger than half the star can cover the star centre at the same time, and the part
#   from z = 0 to z = k - b is then added as in the previous case.
#
# The Jacobi rule can only take a square root that falls to zero, while theta approaches 2 pi, not
# zero, at z = k - b when the planet covers the star centre; hence the substitution there.
#
# The profile is read from a table on a fixed grid, so a profile source only ever has to be
# evaluated on that grid.

def quadrature_rules(nq: int) -> ndarray:
    """The fixed quadrature rules on [-1, 1] used by `ldm_nodes`, as a ``(3, 2, nq)`` array.

    ``rules[0]`` is Gauss-Legendre and ``rules[1]`` is Gauss-Jacobi with the weight function
    ``(1 - t)**0.5`` folded into the weights, for integrands with a square-root zero at t = 1.
    Each holds the nodes in row 0 and the weights in row 1. ``rules[2]`` holds the sine and the
    cosine of the Gauss-Legendre nodes mapped to phi in [0, pi/2], for the integrals taken in
    ``z = z0 + (z1 - z0) sin(phi)**2``.
    """
    t_gl, w_gl = roots_legendre(nq)
    t_gj, w_gj = roots_jacobi(nq, 0.5, 0.0)
    w_gj = w_gj / sqrt(1.0 - t_gj)
    phi = 0.25 * pi * (1.0 + t_gl)
    return array([[t_gl, w_gl], [t_gj, w_gj], [sin(phi), cos(phi)]])


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
                # z in [k - b, b + k] with z = (k - b) + 2 b sin(phi)**2: theta leaves 2 pi and
                # reaches zero as a square root of the distance to each end, and both are regular
                # in phi.
                h = 0.25 * pi
                for q in range(nq):
                    sp, cp = rules[2, 0, q], rules[2, 1, q]
                    z = k - b + 2.0 * b * sp * sp
                    mu[ig, nq + q] = sqrt(1.0 - z * z)
                    wf[ig, nq + q] = rules[0, 1, q] * planet_angular_extent(z, b, k) * z * 4.0 * b * sp * cp * h
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
            # The planet crosses the limb. A planet larger than half the star can cover the disk
            # centre at the same time, and theta = 2 pi for z in [0, k - b]; this part is integrated
            # in mu with Gauss-Legendre.
            zi = abs(b - k)
            if b < k:
                mu_mid = sqrt(1.0 - zi * zi)
                c, h = 0.5 * (mu_mid + 1.0), 0.5 * (1.0 - mu_mid)
                for q in range(nq):
                    m = c + h * rules[0, 0, q]
                    mu[ig, nq + q] = m
                    wf[ig, nq + q] = rules[0, 1, q] * 2.0 * pi * m * h
            # z in [|b - k|, 1] with z = zi + (1 - zi) sin(phi)**2. Theta leaves its value at zi (zero,
            # or 2 pi for b < k) as a square root of z - zi, and the profile meets the limb as a
            # square root of 1 - z; both are regular in phi, so Gauss-Legendre applies.
            h = 0.25 * pi
            for q in range(nq):
                sp, cp = rules[2, 0, q], rules[2, 1, q]
                z = zi + (1.0 - zi) * sp * sp
                mu[ig, q] = cp * sqrt((1.0 - zi) * (1.0 + z))
                wf[ig, q] = rules[0, 1, q] * planet_angular_extent(z, b, k) * z * 2.0 * (1.0 - zi) * sp * cp * h


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

    The mean intensity under the planet has a square-root singularity here, so the g table is
    split at it.
    """
    return (1.0 - k) / (1.0 + k)


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


@njit(fastmath=True)
def ldm_table_map(mu: ndarray, wf: ndarray, t0: float, dt: float, nmu: int):
    """The linear map from a tabulated profile to the mean intensity table.

    `ldm_table` is linear in the profile, and its nodes, weights, and interpolation stencils depend
    only on the planet, so the map can be built once and applied to any number of profiles with
    `ldm_table_banded`. This is what lets the spectroscopy model share the work between its
    wavelength bins.

    Each table node only reads the profile across the planet, so its row of the map is nonzero in a
    narrow band. Returns the ``(ng, nmu)`` map and, for every row, the first and one-past-the-last
    nonzero column.
    """
    ng, nn = mu.shape
    wt = zeros((ng, nmu))
    lo = empty(ng, dtype=int64)
    hi = empty(ng, dtype=int64)
    for ig in range(ng):
        den = 0.0
        for q in range(nn):
            den += wf[ig, q]
        for q in range(nn):
            w = wf[ig, q]
            if w == 0.0:
                continue
            w /= den
            x = (sqrt(mu[ig, q]) - t0) / dt
            if x <= 0.0:
                wt[ig, 0] += w
            elif x >= nmu - 1:
                wt[ig, nmu - 1] += w
            else:
                i = int(x) - 1
                if i < 0:
                    i = 0
                elif i > nmu - 4:
                    i = nmu - 4
                u = x - i
                wt[ig, i] -= w * (u - 1) * (u - 2) * (u - 3) / 6.0
                wt[ig, i + 1] += w * u * (u - 2) * (u - 3) / 2.0
                wt[ig, i + 2] -= w * u * (u - 1) * (u - 3) / 2.0
                wt[ig, i + 3] += w * u * (u - 1) * (u - 2) / 6.0

        lo[ig] = 0
        while lo[ig] < nmu - 1 and wt[ig, lo[ig]] == 0.0:
            lo[ig] += 1
        hi[ig] = nmu
        while hi[ig] > lo[ig] + 1 and wt[ig, hi[ig] - 1] == 0.0:
            hi[ig] -= 1
    return wt, lo, hi


@njit(fastmath=True)
def ldm_table_banded(wt: ndarray, lo: ndarray, hi: ndarray, ldp: ndarray, scale: float, out: ndarray) -> None:
    """The mean intensity table of one profile from the banded map of `ldm_table_map`, times `scale`."""
    for ig in range(wt.shape[0]):
        # Slicing the band first lets the loop index start from zero, which numba needs to drop its
        # negative index check and vectorise the sum.
        w = wt[ig, lo[ig]:hi[ig]]
        p = ldp[lo[ig]:hi[ig]]
        s = 0.0
        for j in range(w.size):
            s += w[j] * p[j]
        out[ig] = scale * s


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
    c = zeros((tab.size - 1, 4))
    cubic_coefficients_into(tab, c)
    return c


@njit(fastmath=True)
def cubic_coefficients_into(tab: ndarray, out: ndarray) -> None:
    """`cubic_coefficients` written into `out`, which has ``tab.size - 1`` rows."""
    n = tab.size
    for i in range(n - 1):
        j = i - 1
        if j < 0:
            j = 0
        elif j > n - 4:
            j = n - 4
        s = i - j
        for r in range(4):
            out[i, r] = (CUBIC_MATRICES[s, r, 0] * tab[j] + CUBIC_MATRICES[s, r, 1] * tab[j + 1]
                         + CUBIC_MATRICES[s, r, 2] * tab[j + 2] + CUBIC_MATRICES[s, r, 3] * tab[j + 3])


# ----------------------------------------------------------------------------------------------
# Mean intensity table in contact-regularised coordinates
# ----------------------------------------------------------------------------------------------
#
# The mean intensity under the planet behaves like a square root of the distance to the two limb
# contacts, g = gc (the planet touches the limb from the inside) and g = 1 (from the outside). A
# cubic on a grid uniform in g cannot follow a square root, which limits such a table to roughly
# first-order convergence. The table is therefore split into three pieces, each uniform in a
# coordinate that grows like the square root of the distance to its contact,
#
# - piece 1, g in [0, gc]:    s = sqrt(1 - (g / gc)**2),
# - piece 2, x in [0, 1/2]:   s = sqrt(2 x),          x = (g - gc) / (1 - gc),
# - piece 3, x in [1/2, 1]:   s = sqrt(2 (1 - x)),
#
# with s running from 0 at the contact to 1 at the far end of the piece. The mean intensity is a
# smooth function of s in each piece, so the cubic interpolation is fourth order everywhere.
#
# The first piece uses the square of g / gc rather than g / gc itself for two reasons. The mean
# intensity is an even function of the impact parameter, so it is a smooth function of g**2 at the
# disk centre, and 1 - (g / gc)**2 = (1 - g / gc)(1 + g / gc) still vanishes linearly at the
# contact; the coordinate therefore suits both ends of the piece. And since g / gc = z / (1 - k)
# for a planet at projected distance z, the coordinate follows from z**2 directly, which lets a
# flux loop look up a full-transit sample with `ldm_lookup_full` without taking the square root
# that turns z**2 into z.

@njit
def table_sizes(ng: int):
    """Number of nodes in the full-transit piece and in each of the two ingress pieces of an `ng`-node table."""
    n2 = ng // 3
    return ng - 2 * n2, n2


@njit
def g_nodes(k: float, ng: int):
    """Grazing parameter nodes of the contact-regularised mean intensity table.

    Returns the nodes and the number of nodes in the first piece. The nodes of the three pieces
    are stored back to back, each piece ordered from its contact outwards, so the nodes are not
    monotonic in g. Each piece needs at least four nodes for a cubic, so `ng` must be at least 12.
    """
    gc = split_point(k)
    n1, n2 = table_sizes(ng)
    gs = empty(ng)
    for j in range(n1):
        s = j / (n1 - 1)
        gs[j] = gc * sqrt(1.0 - s * s)
    for j in range(n2):
        s = j / (n2 - 1)
        gs[n1 + j] = gc + (1.0 - gc) * 0.5 * s * s
        gs[n1 + n2 + j] = gc + (1.0 - gc) * (1.0 - 0.5 * s * s)
    return gs, n1


@njit(fastmath=True)
def split_cubic_coefficients(tab: ndarray, n1: int, out: ndarray) -> None:
    """Cubic coefficients of the three table pieces, written back to back into `out`.

    `out` has ``tab.size - 3`` rows: ``n1 - 1`` intervals for the full-transit piece followed by
    the intervals of the two ingress pieces, which is the layout `ldm_lookup` reads.
    """
    n2 = (tab.size - n1) // 2
    cubic_coefficients_into(tab[:n1], out[:n1 - 1])
    cubic_coefficients_into(tab[n1:n1 + n2], out[n1 - 1:n1 + n2 - 2])
    cubic_coefficients_into(tab[n1 + n2:], out[n1 + n2 - 2:])


@njit(fastmath=True)
def ldm_interval(g: float, gc: float, n1: int, nc: int):
    """The interval of the mean intensity table holding grazing parameter g, and g's position in it.

    `nc` is the number of intervals in the table. Returns the interval index and the local
    position in [0, 1] that `ldm_lookup` evaluates the interval's cubic at.
    """
    if g < gc:
        r = g / gc
        u = sqrt(1.0 - r * r) * (n1 - 1)
        m = n1 - 2
        i0 = 0
    else:
        m2 = (nc - n1 + 1) // 2          # Intervals in each ingress piece
        x = (g - gc) / (1.0 - gc)
        if x < 0.5:
            u = sqrt(2.0 * x) * m2
            i0 = n1 - 1
        else:
            # Beyond the last contact the overlap area is zero, so the value only has to be finite.
            u = sqrt(2.0 * max(1.0 - x, 0.0)) * m2
            i0 = n1 - 1 + m2
        m = m2 - 1
    i = int(u)
    if i > m:
        i = m
    return i0 + i, u - i


@njit(fastmath=True)
def ldm_lookup(g: float, gc: float, n1: int, coef: ndarray) -> float:
    """The mean intensity under the planet at grazing parameter g from the contact-regularised table.

    `coef` holds the interval coefficients of the three pieces back to back, as written by
    `split_cubic_coefficients`.
    """
    i, a = ldm_interval(g, gc, n1, coef.shape[0])
    return coef[i, 0] + a * (coef[i, 1] + a * (coef[i, 2] + a * coef[i, 3]))


@njit(fastmath=True)
def ldm_lookup_full(z2: float, inv_z2c: float, n1: int, coef: ndarray) -> float:
    """The mean intensity under a planet fully on the stellar disk, from its squared projected distance.

    Equal to `ldm_lookup` for a planet at projected distance z < 1 - k, but takes ``z2 = z**2``
    and ``inv_z2c = 1 / (1 - k)**2``: the first table piece is uniform in
    ``sqrt(1 - (g / gc)**2) = sqrt(1 - z2 * inv_z2c)``, so the lookup needs neither z nor g.
    """
    u = sqrt(max(1.0 - z2 * inv_z2c, 0.0)) * (n1 - 1)
    i = int(u)
    if i > n1 - 2:
        i = n1 - 2
    a = u - i
    return coef[i, 0] + a * (coef[i, 1] + a * (coef[i, 2] + a * coef[i, 3]))
