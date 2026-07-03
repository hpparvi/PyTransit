from math import cos, sin, sqrt, isfinite, nan, fabs, atan2

from numpy import ndarray, pi, full, arange, empty, cos as ncos, sin as nsin
from numba import njit

TWO_PI = 2.0 * pi

# Fixed scan grid used by ellipse_circle_intersection_area for isolating the
# roots of g(t) = A cos²t + B cos t + C sin t + D. The trigonometric tables are
# precomputed so that evaluating g on the grid needs no per-call trig calls.
NSCAN = 64
_TSCAN = arange(NSCAN + 1) * (TWO_PI / NSCAN)
_CSCAN = ncos(_TSCAN)
_SSCAN = nsin(_TSCAN)
_C2SCAN = _CSCAN * _CSCAN

@njit
def rotated_ellipse_bbox(a: float, f: float):
    """
    Parameters
    ----------
    a : float
        The angle (in radians) by which the ellipse is rotated.
    f : float
        The eccentricity of the ellipse, with a value between 0 and 1.

    Returns
    -------
    hw : float
        The half-width of the bounding box of the rotated ellipse.
    hh : float
        The half-height of the bounding box of the rotated ellipse.

    """
    r1, r2 = 1.0, 1.0-f
    ux = r1 * cos(a)
    uy = r1 * sin(a)
    vx = r2 * cos(a + pi/2)
    vy = r2 * sin(a + pi/2)
    hw = sqrt(ux*ux + vx*vx)
    hh = sqrt(uy*uy + vy*vy)
    return hw, hh

@njit
def rotated_ellipse_x(y: float, a: float, f: float) -> tuple[float, float]:
    """Calculate the x coordinates for a rotated ellipse given the y coordinate and the flattening factor.

    Parameters
    ----------
    y
        The y-coordinate of the point on the ellipse.
    a
        The angle (in radians) of rotation for the ellipse.
    f
        The flattening parameter of the ellipse.

    Returns
    -------
    tuple[float, float]
        The ellipse x-coordinates for the given y coordinate.
    """
    ca, sa = cos(a), sin(a)
    b = 1.0 - f
    d = b**2*ca**2 - y**2*sa**4 - 2*y**2*sa**2*ca**2 - y**2*ca**4 + sa**2
    if d >= 0.0:
        xl = (y*(-sa*ca + sa*ca/b**2) - sqrt(d)/b) / (ca**2 + sa**2/b**2)
        xr = (y*(-sa*ca + sa*ca/b**2) + sqrt(d)/b) / (ca**2 + sa**2/b**2)
        return xl, xr
    else:
        return nan, nan


@njit
def rotated_ellipse_dxdy(y: float, a: float, f: float) -> tuple[float, float]:
    """Calculate dx/dy for a rotated ellipse given the y coordinate and the flattening factor.

    Parameters
    ----------
    y
        The y-coordinate of the point on the ellipse.
    a
        The angle, in radians, of rotation of the ellipse.
    f
        The flattening parameter of the ellipse.

    Returns
    -------
    tuple[float, float]
        The ellipse derivatives (dx/dy) for the given y coordinate.
    """
    ca, sa = cos(a), sin(a)
    b = 1.0 - f
    d = b**2*ca**2 - y**2*sa**4 - 2*y**2*sa**2*ca**2 - y**2*ca**4 + sa**2
    if d >= 0.0:
        dxl = (-sa*ca - (-y*sa**4 - 2*y*sa**2*ca**2 - y*ca**4)/(b*sqrt(d)) + sa*ca/b**2)/(ca**2 + sa**2/b**2)
        dxr = (-sa*ca + (-y*sa**4 - 2*y*sa**2*ca**2 - y*ca**4)/(b*sqrt(d)) + sa*ca/b**2)/(ca**2 + sa**2/b**2)
        return dxl, dxr
    else:
        return nan, nan


@njit
def create_ellipse(ny: int, k: float, f: float, a: float) -> (ndarray, ndarray):
    """Create the coordinates for a rotated ellipse.

    Parameters
    ----------
    ny : int
        The number of sample points along the y-axis.
    k : float
        Scale factor for the ellipse.
    f : float
        Flattening factor for the ellipse. Determines the extent to which the ellipse is squashed along the y-axis.
    a : float
        The angle of rotation for the ellipse in radians.

    Returns
    -------
    (ndarray, ndarray)
        A tuple containing two ndarrays. The first ndarray is a 1D array of y-coordinates, and the second is a 2D array of x-coordinates
        corresponding to the left and right intersections of the ellipse with vertical lines through the y-coordinates.

    Notes
    -----
    The ellipse is initially defined with its major axis aligned with the x-axis. It is then rotated by an angle `a` and scaled by a factor `k`.
    The flattening factor `f` adjusts the size of the minor axis relative to the major axis, with a smaller value indicating a more flattened ellipse.
    """
    dy = 2 / (ny+1)
    ys = arange(1, 1+ny)*dy - 1
    xs = full((ny, 2), nan)
    ca, sa = cos(a), sin(a)
    b = 1.0 - f
    b2 = b * b
    ca2 = ca * ca
    sa2 = sa * sa

    for i in range(ny):
        y = ys[i]
        y2 = y * y
        d = b2 * ca2 + ca2 * ca2 * (-y2) - 2 * ca2 * sa2 * y2 - sa2 * sa2 * y2 + sa2
        if d >= 0:
            d = sqrt(d) / b
            u = (ca * sa * y) / b2 - ca * sa * y
            v = sa2 / b2 + ca2
            xs[i, 0] = (-d + u) / v
            xs[i, 1] = (d + u) / v
    return k*ys, k*xs


@njit
def ellipse_circle_intersection_area(cx: float, cy: float, z: float, k: float, f: float, xs: ndarray, ys: ndarray) -> float:
    """Calculate the intersection area between a rotated ellipse and a circle using a scanline fill approach.

   Parameters
    ----------
    cx : float
        The ellipse's center x coordinate.
    cy : float
        The ellipse's center y coordinate.
    z : float
        The center-center distance.
    k : float
        Radius ratio.
    f : float
        Flattening factor for the ellipse. Determines the extent to which the ellipse is squashed along the y-axis.
    xs : ndarray
        2D array of x-coordinates corresponding to the left and right intersections of the ellipse with vertical lines through the y-coordinates.
    ys : ndarray
        1D array of y-coordinates.

     Returns
     -------
     float
         The area of the intersection between the given ellipse and circle.

     Notes
     -----
     This function employs a numerical method that iterates over a set of horizontal lines (scanlines)
     to approximate the area of intersection between a circle and a rotated ellipse. The calculation considers
     the rotation of the ellipse, its eccentricity, and the relative position of the ellipse to the circle.
     The approach is based on the principles of scanline fill in computer graphics, adapted to the mathematical
     properties of ellipses and circles.

     The algorithm calculates the intersection points between each scanline and the ellipse, then integrates these
     intersections over the range of y-coordinates to find the total area. It handles different cases based on the
     relative position and size of the ellipse to optimize calculations.
     """
    if z <= 1.0 - k:
        return pi * k * (1.0 - f) * k
    elif z >= 1.0 + k:
        return 0.0
    else:
        ny = ys.size
        dy = (ys[1] - ys[0])
        l = 0.0
        for i in range(ny):
            if isfinite(xs[i, 0]):
                yy = ys[i] + cy
                if fabs(yy) <= 1.0:
                    w = sqrt(1.0 - yy * yy)
                    x0 = max(-w - cx, xs[i, 0])
                    x1 = min(w - cx, xs[i, 1])
                    if x1 > x0:
                        l += x1 - x0
        return l*dy


@njit
def create_ellipse_theta(ny: int, k: float, f: float, a: float) -> (ndarray, ndarray, ndarray):
    """Create the coordinates and quadrature weights for a rotated ellipse sampled uniformly in θ.

    Variant of `create_ellipse` that places the scanlines at y = hh sin(θ) with θ sampled
    uniformly in (-π/2, π/2), where hh is the vertical half-extent of the rotated ellipse.
    The substitution absorbs the square-root behavior of the chord length at the ellipse
    tips, so a weighted scanline sum over this grid converges as ~ny⁻² instead of the ~ny⁻¹·⁵
    reached with a uniform grid.

    Parameters
    ----------
    ny : int
        The number of sample points along the y-axis.
    k : float
        Scale factor for the ellipse.
    f : float
        Flattening factor for the ellipse. Determines the extent to which the ellipse is squashed along the y-axis.
    a : float
        The angle of rotation for the ellipse in radians.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        A tuple (ys, xs, ws) where ys is a 1D array of y-coordinates, xs is a 2D array of the left and
        right ellipse chord x-coordinates for each y, and ws is a 1D array of quadrature weights so that
        an integral over y is approximated by sum(L(ys) * ws).
    """
    ca, sa = cos(a), sin(a)
    b = 1.0 - f
    b2 = b * b
    ca2 = ca * ca
    sa2 = sa * sa
    hh = sqrt(sa2 + b2 * ca2)
    dth = pi / ny
    ys = empty(ny)
    ws = empty(ny)
    xs = full((ny, 2), nan)
    u0 = ca * sa / b2 - ca * sa
    v = sa2 / b2 + ca2
    for i in range(ny):
        th = -0.5 * pi + (i + 0.5) * dth
        y = hh * sin(th)
        ys[i] = y
        ws[i] = hh * cos(th) * dth
        d = b2 * ca2 + sa2 - y * y
        if d > 0.0:
            d = sqrt(d) / b
            xs[i, 0] = (u0 * y - d) / v
            xs[i, 1] = (u0 * y + d) / v
    return k * ys, k * xs, k * ws


@njit
def ellipse_circle_intersection_area_theta(cx: float, cy: float, z: float, k: float, f: float,
                                           xs: ndarray, ys: ndarray, ws: ndarray) -> float:
    """Calculate the intersection area between a rotated ellipse and the unit circle on a θ-sampled grid.

    Scanline variant of `ellipse_circle_intersection_area` that consumes the non-uniformly spaced
    grid and quadrature weights produced by `create_ellipse_theta`. In geometries where the
    integration error is dominated by the ellipse tips (ordinary ingress and egress), this is
    typically one to two orders of magnitude more accurate than the uniform grid at the same ny;
    in grazing geometries the error is dominated by the circle's top or bottom edge and the two
    variants perform alike. For exact areas use `ellipse_circle_intersection_area_exact`.

    Parameters
    ----------
    cx : float
        The ellipse's center x coordinate.
    cy : float
        The ellipse's center y coordinate.
    z : float
        The center-center distance.
    k : float
        Radius ratio.
    f : float
        Flattening factor for the ellipse. Determines the extent to which the ellipse is squashed along the y-axis.
    xs : ndarray
        2D array of the left and right ellipse chord x-coordinates from `create_ellipse_theta`.
    ys : ndarray
        1D array of y-coordinates from `create_ellipse_theta`.
    ws : ndarray
        1D array of quadrature weights from `create_ellipse_theta`.

    Returns
    -------
    float
        The area of the intersection between the given ellipse and the unit circle.
    """
    if z <= 1.0 - k:
        return pi * k * (1.0 - f) * k
    elif z >= 1.0 + k:
        return 0.0
    else:
        ny = ys.size
        l = 0.0
        for i in range(ny):
            if isfinite(xs[i, 0]):
                yy = ys[i] + cy
                if fabs(yy) <= 1.0:
                    w = sqrt(1.0 - yy * yy)
                    x0 = max(-w - cx, xs[i, 0])
                    x1 = min(w - cx, xs[i, 1])
                    if x1 > x0:
                        l += (x1 - x0) * ws[i]
        return l


@njit
def _refine_root(lo: float, hi: float, glo: float, ea: float, eb: float, ec: float, ed: float) -> float:
    """Refine a bracketed root of g(t) = ea cos²t + eb cos t + ec sin t + ed with safeguarded Newton iteration.

    The bracket [lo, hi] must satisfy g(lo) g(hi) < 0. Newton steps are taken when they stay
    inside the bracket; otherwise, and on every fourth iteration, the method falls back to
    bisection so convergence is guaranteed.
    """
    ghi = ea * cos(hi) * cos(hi) + eb * cos(hi) + ec * sin(hi) + ed
    t = lo - glo * (hi - lo) / (ghi - glo) if ghi != glo else 0.5 * (lo + hi)
    if not lo < t < hi:
        t = 0.5 * (lo + hi)
    for it in range(100):
        ct, st = cos(t), sin(t)
        gt = ea * ct * ct + eb * ct + ec * st + ed
        if gt == 0.0 or hi - lo < 1e-15:
            return t
        if (gt > 0.0) == (glo > 0.0):
            lo = t
            glo = gt
        else:
            hi = t
        dg = -(2.0 * ea * ct + eb) * st + ec * ct
        if dg != 0.0 and (it & 3) != 3:
            tn = t - gt / dg
            t = tn if lo < tn < hi else 0.5 * (lo + hi)
        else:
            t = 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


@njit
def ellipse_circle_intersection_area_exact(cx: float, cy: float, z: float, k: float, f: float, a: float) -> float:
    """Calculate the exact intersection area between a rotated ellipse and the unit circle.

    Parameters
    ----------
    cx : float
        The ellipse's center x coordinate.
    cy : float
        The ellipse's center y coordinate.
    z : float
        The center-center distance.
    k : float
        Radius ratio (the ellipse's semi-major axis).
    f : float
        Flattening factor for the ellipse. Determines the extent to which the ellipse is squashed along the y-axis.
    a : float
        The angle of rotation for the ellipse in radians.

    Returns
    -------
    float
        The area of the intersection between the given ellipse and the unit circle.

    Notes
    -----
    A point on the ellipse is E(t) = C + M (cos t, sin t) with C = (cx, cy) and M = R(a) diag(k, k(1-f)),
    so the signed distance function g(t) = |E(t)|² - 1 is a degree-two trigonometric polynomial

        g(t) = A cos²t + B cos t + C sin t + D

    with at most four roots per period; the roots are the ellipse-circle intersection points.
    The roots are isolated by scanning g on a fixed grid (using precomputed trigonometric tables)
    and adaptively subdividing any cell that cannot be certified root-free with the curvature
    bound |g''| <= 2|A| + sqrt(B² + C²), so near-tangency root pairs cannot hide between grid
    points. Bracketed roots are polished to machine precision with safeguarded Newton iteration.

    The intersection of two convex regions is convex, and its boundary alternates between ellipse
    and circle arcs joined at the intersection points, ordered consistently in both curve
    parameters. The area follows from Green's theorem, A = (1/2) ∮ (x dy - y dx), where both arc
    types integrate in closed form: a circle arc from angle θ₁ to θ₂ contributes (θ₂ - θ₁)/2, and
    an ellipse arc from t₁ to t₂ contributes (k²(1-f)(t₂ - t₁) + C × (E(t₂) - E(t₁)))/2.
    """
    b = 1.0 - f
    if z >= 1.0 + k:
        return 0.0
    full_area = pi * k * k * b
    if z <= 1.0 - k:
        return full_area

    ca, sa = cos(a), sin(a)
    px = cx * ca + cy * sa      # Ellipse center position in the ellipse-axis-aligned frame
    py = cy * ca - cx * sa
    ea = k * k * (1.0 - b * b)
    eb = 2.0 * k * px
    ec = 2.0 * k * b * py
    ed = cx * cx + cy * cy - 1.0 + k * k * b * b

    # Root-free certificate for a cell of width h: |g| >= max|g''| h²/8 at both endpoints
    # with equal signs implies g cannot cross zero inside the cell.
    m2 = 0.125 * (2.0 * fabs(ea) + sqrt(eb * eb + ec * ec))

    ws = empty(328)
    roots = ws[:8]
    sk_t0 = ws[8:88]
    sk_t1 = ws[88:168]
    sk_g0 = ws[168:248]
    sk_g1 = ws[248:328]
    nr = 0

    nneg = 0
    hscan = TWO_PI / NSCAN
    m2h2 = m2 * hscan * hscan
    g0 = ea * _C2SCAN[0] + eb * _CSCAN[0] + ec * _SSCAN[0] + ed
    for j in range(NSCAN):
        if g0 < 0.0:
            nneg += 1
        g1 = ea * _C2SCAN[j + 1] + eb * _CSCAN[j + 1] + ec * _SSCAN[j + 1] + ed
        # Fast path: most cells carry the same-signed, certifiably root-free case.
        if (g0 > 0.0) == (g1 > 0.0) and min(fabs(g0), fabs(g1)) >= m2h2:
            g0 = g1
            continue
        sk_t0[0] = _TSCAN[j]
        sk_t1[0] = _TSCAN[j + 1]
        sk_g0[0] = g0
        sk_g1[0] = g1
        top = 1
        while top > 0:
            top -= 1
            t0, t1 = sk_t0[top], sk_t1[top]
            gc0, gc1 = sk_g0[top], sk_g1[top]
            h = t1 - t0
            if (gc0 > 0.0) != (gc1 > 0.0):
                if h < 0.02:
                    if nr < 8:
                        roots[nr] = _refine_root(t0, t1, gc0, ea, eb, ec, ed)
                        nr += 1
                    continue
            elif min(fabs(gc0), fabs(gc1)) >= m2 * h * h or h < 1e-7:
                continue
            tm = 0.5 * (t0 + t1)
            ct, st = cos(tm), sin(tm)
            gm = ea * ct * ct + eb * ct + ec * st + ed
            sk_t0[top] = tm
            sk_t1[top] = t1
            sk_g0[top] = gm
            sk_g1[top] = gc1
            top += 1
            sk_t0[top] = t0
            sk_t1[top] = tm
            sk_g0[top] = gc0
            sk_g1[top] = gm
            top += 1
        g0 = g1

    # A smooth closed curve crosses the circle an even number of times: an odd count means
    # an unresolved tangential crossing, so drop the last root and fall through.
    if nr % 2 == 1:
        nr -= 1

    if nr == 0:
        if 2 * nneg >= NSCAN:
            return full_area                       # The ellipse is fully inside the circle
        if px * px / (k * k) + py * py / (k * k * b * b) < 1.0:
            return pi                              # The unit circle is fully inside the ellipse
        return 0.0

    # The roots are generated in increasing order, but sort defensively.
    for j in range(1, nr):
        rj = roots[j]
        i = j - 1
        while i >= 0 and roots[i] > rj:
            roots[i + 1] = roots[i]
            i -= 1
        roots[i + 1] = rj

    area = 0.0
    for j in range(nr):
        t1 = roots[j]
        t2 = roots[j + 1] if j + 1 < nr else roots[0] + TWO_PI
        tm = 0.5 * (t1 + t2)
        ct, st = cos(tm), sin(tm)
        gm = ea * ct * ct + eb * ct + ec * st + ed
        ct1, st1 = cos(t1), sin(t1)
        ct2, st2 = cos(t2), sin(t2)
        ex1 = k * (ca * ct1 - b * sa * st1)
        ey1 = k * (sa * ct1 + b * ca * st1)
        ex2 = k * (ca * ct2 - b * sa * st2)
        ey2 = k * (sa * ct2 + b * ca * st2)
        if gm <= 0.0:
            # The ellipse arc lies inside the circle: it is the boundary of the intersection here.
            area += 0.5 * (k * k * b * (t2 - t1) + cx * (ey2 - ey1) - cy * (ex2 - ex1))
        else:
            # The ellipse arc lies outside: the boundary follows the circle counterclockwise
            # between the same two intersection points.
            dth = atan2(cy + ey2, cx + ex2) - atan2(cy + ey1, cx + ex1)
            if dth < 0.0:
                dth += TWO_PI
            area += 0.5 * dth

    return min(max(area, 0.0), min(full_area, pi))
