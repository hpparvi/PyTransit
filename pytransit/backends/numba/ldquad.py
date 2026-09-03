"""Mean stellar intensity under the planet by geometry-matched Gauss quadrature.

The mean intensity under the planet at grazing parameter g = b / (1 + k) is

    ldm(g) = int I(z) theta(z) z dz / int theta(z) z dz,

where theta(z) is the angular extent of the planet disk at stellar radius z. Wherever the
integrand has a square-root endpoint -- theta vanishing like sqrt at a contact, or the profile
behaving like sqrt(1 - z) at the limb -- the quadrature is matched to it, either by a Gauss-Jacobi
rule with the square root in its weight function or by substituting z = z0 +/- s**2, which turns
the square root into a regular function of s:

- planet fully on the disk: z = (b + k) - s**2 makes the limb end regular (both theta's zero and
  the profile's sqrt), and Gauss-Jacobi absorbs theta's sqrt zero at z = b - k;
- star centre inside the planet: theta is 2 pi from z = 0 to z = k - b (regular, Gauss-Legendre
  in mu), then z = (k - b) + s**2 makes that end regular and Gauss-Jacobi absorbs the zero at
  z = b + k;
- planet crossing the limb: integrated in mu, where every built-in law is polynomial, so the limb
  end is regular and Gauss-Jacobi absorbs the sqrt zero at the inner end.

The profile is read from a table on a fixed grid uniform in sqrt(mu), so a profile source only
ever has to be evaluated on that grid. The mean intensity is tabulated against g in two uniform
segments split at the limb contact, where the exact function has a kink, and each segment is
stored as per-interval cubic coefficients so that the lookup and its derivative are one index
and a Horner evaluation.
"""
from math import floor

from numba import njit
from numpy import sqrt, pi, arccos, zeros, linspace, ndarray, array, empty, isnan, isfinite, nan
from numpy.polynomial import Polynomial
from scipy.special import roots_jacobi, roots_legendre

# Fast-math flags for the functions that must see NaNs: everything except the no-NaN and
# no-infinity assumptions, under which LLVM folds the `isnan` guards away.
_FASTMATH_NAN_SAFE = {'contract', 'reassoc', 'arcp', 'afn', 'nsz'}

__all__ = ['quadrature_rules', 'planet_angular_extent', 'ldm_nodes', 'split_point', 'g_nodes',
           'profile_grid', 'profile_at', 'ldm_table', 'cubic_coefficients', 'split_cubic_coefficients',
           'ldm_lookup', 'ldm_lookup_and_grad', 'rr_tables', 'rr_tables_grad']


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
                mu_mid = sqrt(1.0 - (k - b) ** 2)
                c, h = 0.5 * (mu_mid + 1.0), 0.5 * (1.0 - mu_mid)
                for q in range(nq):
                    m = c + h * rules[0, 0, q]
                    z = sqrt(1.0 - m * m)
                    mu[ig, q] = m
                    wf[ig, q] = rules[0, 1, q] * planet_angular_extent(z, b, k) * m * h
                sm = sqrt(2.0 * b)
                c = h = 0.5 * sm
                for q in range(nq):
                    sq = c + h * rules[1, 0, q]
                    z = k - b + sq * sq
                    mu[ig, nq + q] = sqrt(1.0 - z * z)
                    wf[ig, nq + q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * z * 2.0 * sq * h
            else:
                sm = sqrt(2.0 * k)
                c = h = 0.5 * sm
                for q in range(nq):
                    sq = c + h * rules[1, 0, q]
                    z = b + k - sq * sq
                    mu[ig, q] = sqrt(1.0 - z * z)
                    wf[ig, q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * z * 2.0 * sq * h
        else:
            mu_lo = sqrt(1.0 - (b - k) ** 2)
            c = h = 0.5 * mu_lo
            for q in range(nq):
                m = c + h * rules[1, 0, q]
                z = sqrt(1.0 - m * m)
                mu[ig, q] = m
                wf[ig, q] = rules[1, 1, q] * planet_angular_extent(z, b, k) * m * h


@njit(fastmath=_FASTMATH_NAN_SAFE)
def split_point(k: float) -> float:
    """The grazing parameter at which the planet first touches the limb, (1 - k) / (1 + k)."""
    return (1.0 - k) / (1.0 + k)


@njit
def g_nodes(k: float, ng: int) -> ndarray:
    """Grazing parameter nodes: two uniform segments of ``ng // 2`` nodes meeting at the limb contact.

    Both segments have a fixed node count so that the table shapes are independent of k, which
    the JAX backend needs under ``jit``; the split node is stored once per segment.
    """
    gc = split_point(k)
    n1 = ng // 2
    gs = empty(ng)
    gs[:n1] = linspace(0.0, gc, n1)
    gs[n1:] = linspace(gc, 1.0, ng - n1)
    return gs


@njit(fastmath=_FASTMATH_NAN_SAFE)
def profile_at(m: float, t0: float, dt: float, ldp: ndarray) -> float:
    """Cubic interpolation of a profile tabulated on the grid ``mu = (t0 + dt * arange(n))**2``.

    Below the first node the profile is held at its first value.
    """
    n = ldp.size
    x = (sqrt(m) - t0) / dt
    if isnan(x):
        return nan
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
    """Maps from four stencil values to the cubic's coefficients in the local interval variable."""
    M = zeros((3, 4, 4))
    for s in range(3):
        for j in range(4):
            lag = Polynomial([1.0])
            for m in range(4):
                if m != j:
                    lag = lag * Polynomial([-m, 1.0]) / (j - m)
            coef = lag(Polynomial([s, 1.0])).coef
            M[s, :coef.size, j] = coef[:4]
    return M


CUBIC_MATRICES = _cubic_matrices()


@njit(fastmath=True)
def cubic_coefficients(tab: ndarray) -> ndarray:
    """Per-interval cubic coefficients for a table on a uniform grid.

    Interval ``i`` covers nodes ``i`` and ``i + 1``; its value at local position ``a`` in [0, 1)
    is ``c[i, 0] + a * (c[i, 1] + a * (c[i, 2] + a * c[i, 3]))``. The cubic is the four-point
    Lagrange interpolant on a stencil shifted inwards at the ends.
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
def split_cubic_coefficients(tab: ndarray, n1: int, out: ndarray) -> None:
    """Cubic coefficients of a g table split at node ``n1 - 1``, written back to back into `out`.

    `out` has ``tab.size - 2`` rows: ``n1 - 1`` intervals for the first segment followed by the
    second segment's intervals. The split node is stored once per segment.
    """
    out[:n1 - 1] = cubic_coefficients(tab[:n1])
    out[n1 - 1:] = cubic_coefficients(tab[n1:])


@njit(fastmath=_FASTMATH_NAN_SAFE)
def _locate(g: float, gc: float, n1: int, nint: int):
    """Interval index and local coordinate of g in the split table, and the interval width."""
    if g < gc:
        x = g / gc * (n1 - 1)
        i = int(x)
        if i > n1 - 2:
            i = n1 - 2
        return i, x - i, gc / (n1 - 1)
    n2 = nint - (n1 - 1) + 1
    x = (g - gc) / (1.0 - gc) * (n2 - 1)
    i = int(x)
    if i > n2 - 2:
        i = n2 - 2
    return i + n1 - 1, x - i, (1.0 - gc) / (n2 - 1)


@njit(fastmath=_FASTMATH_NAN_SAFE)
def ldm_lookup(g: float, gc: float, n1: int, coef: ndarray) -> float:
    """The mean intensity under the planet at grazing parameter g from the split cubic table.

    A NaN g (from a NaN radius ratio) gives NaN; without the guard ``int(nan)`` is undefined
    and would index the table anywhere.
    """
    if isnan(g) or isnan(gc):
        return nan
    i, a, _ = _locate(g, gc, n1, coef.shape[0])
    return coef[i, 0] + a * (coef[i, 1] + a * (coef[i, 2] + a * coef[i, 3]))


@njit(fastmath=_FASTMATH_NAN_SAFE)
def ldm_lookup_and_grad(g: float, gc: float, n1: int, coef: ndarray):
    """The mean intensity under the planet and its derivative with respect to g (NaN for NaN g)."""
    if isnan(g) or isnan(gc):
        return nan, nan
    i, a, h = _locate(g, gc, n1, coef.shape[0])
    value = coef[i, 0] + a * (coef[i, 1] + a * (coef[i, 2] + a * coef[i, 3]))
    deriv = (coef[i, 1] + a * (2.0 * coef[i, 2] + 3.0 * a * coef[i, 3])) / h
    return value, deriv


@njit
def rr_tables(k: ndarray, ldp: ndarray, rules: ndarray, pt0: float, pdt: float, ng: int,
              gcs: ndarray, coef: ndarray) -> None:
    """Split cubic tables of the mean intensity under the planet for one parameter vector.

    Fills ``gcs[ipb]`` with the limb contact and ``coef[ipb]`` (``(ng - 2, 4)``) with the
    table for every passband; passbands sharing a radius ratio share the quadrature nodes.
    All the backends build their tables with this function, so their tables are identical.
    """
    nq = rules.shape[2]
    n1 = ng // 2
    mu = zeros((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    tab = zeros(ng)
    for ipb in range(k.size):
        kv = k[ipb]
        if not isfinite(kv):
            # A NaN radius ratio (routine from a sampler) yields NaN tables and a NaN flux.
            gcs[ipb] = nan
            coef[ipb, :, :] = nan
            continue
        gcs[ipb] = split_point(kv)
        if ipb == 0 or kv != k[ipb - 1]:
            ldm_nodes(kv, g_nodes(kv, ng), rules, mu, wf)
        ldm_table(mu, wf, pt0, pdt, ldp[ipb], tab)
        split_cubic_coefficients(tab, n1, coef[ipb])


@njit
def rr_tables_grad(k: ndarray, ldp: ndarray, ldg: ndarray, rules: ndarray, pt0: float, pdt: float, ng: int,
                   gcs: ndarray, coef: ndarray, kcoef: ndarray, dcoef: ndarray) -> None:
    """`rr_tables` plus the tables the gradient needs, for one parameter vector.

    ``kcoef[ipb]`` holds the total derivative of the mean intensity with respect to the
    radius ratio along the node trajectory ``g_nodes(k)``, by a forward difference on k with
    the same nodes on both sides of the limb contact (the derivative at fixed g jumps there);
    the kernels remove the node motion at lookup time. ``dcoef[ipb, j]`` holds the derivative
    with respect to limb darkening coefficient j, from ``ldg[ipb, j + 1]``.
    """
    nq = rules.shape[2]
    n1 = ng // 2
    nldc = ldg.shape[1] - 1
    mu = zeros((ng, 2 * nq))
    wf = zeros((ng, 2 * nq))
    mup = zeros((ng, 2 * nq))
    wfp = zeros((ng, 2 * nq))
    tab = zeros(ng)
    tabp = zeros(ng)
    tabk = zeros(ng)
    h = 1e-5
    for ipb in range(k.size):
        kv = k[ipb]
        if not isfinite(kv):
            gcs[ipb] = nan
            coef[ipb, :, :] = nan
            kcoef[ipb, :, :] = nan
            dcoef[ipb, :, :, :] = nan
            continue
        gcs[ipb] = split_point(kv)
        if ipb == 0 or kv != k[ipb - 1]:
            ldm_nodes(kv, g_nodes(kv, ng), rules, mu, wf)
            ldm_nodes(kv + h, g_nodes(kv + h, ng), rules, mup, wfp)
        ldm_table(mu, wf, pt0, pdt, ldp[ipb], tab)
        split_cubic_coefficients(tab, n1, coef[ipb])
        ldm_table(mup, wfp, pt0, pdt, ldp[ipb], tabp)
        tabk[:] = (tabp - tab) / h
        split_cubic_coefficients(tabk, n1, kcoef[ipb])
        for j in range(nldc):
            ldm_table(mu, wf, pt0, pdt, ldg[ipb, j + 1], tab)
            split_cubic_coefficients(tab, n1, dcoef[ipb, j])
