from math import cos, sin, sqrt, isfinite, nan

from numpy import ndarray, fabs, pi, full
from numba import njit

@njit
def ellipse_circle_intersection_area(ny: int, k: float, b: float, f: float, a: float) -> float:
    """
     Calculate the intersection area between a rotated ellipse and a circle using a scanline fill approach.

     Parameters
     ----------
     ny : int
         Number of y-coordinates for the scanlines.
     k : float
         Equatorial planet-star radius ratio (semi-major ellipse radius).
     b : float
         The planet-star center distance.
     f : float
         The flattening factor of the ellipse, defining its eccentricity.
     a : float
         The rotation angle of the ellipse in radians.

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

     The function is decorated with `@njit` from Numba for performance optimization, which compiles the function
     to machine code at runtime. Thus, it expects NumPy arrays and floats as inputs and outputs a float representing
     the intersection area.

     The algorithm calculates the intersection points between each scanline and the ellipse, then integrates these
     intersections over the range of y-coordinates to find the total area. It handles different cases based on the
     relative position and size of the ellipse to optimize calculations.
     """
    b = fabs(b)
    if b <= 1.0 - k:
        return pi * k * (1.0 - f) * k
    elif b >= 1.0 + k:
        return 0.0
    else:
        dy = 2.0 / (ny - 1)
        sdy = k * dy
        ca, sa = cos(a), sin(a)
        t = 1.0 - f
        t2 = t * t
        ca2 = ca * ca
        sa2 = sa * sa

        area = 0.0
        y = -1.0
        for i in range(ny):
            y2 = y * y
            d = t2 * ca2 + ca2 * ca2 * (-y2) - 2 * ca2 * sa2 * y2 - sa2 * sa2 * y2 + sa2
            if d >= 0:
                d = sqrt(d) / t
                u = (ca * sa * y) / t2 - ca * sa * y
                v = sa2 / t2 + ca2
                xe0 = b + k * (-d + u) / v
                xe1 = b + k * (d + u) / v
                xs = sqrt(1.0 - (k * y) ** 2)
                if xs > xe0:
                    area += (min(xe1, xs) - xe0) * sdy
            y += dy
        return area


# Functions for plotting and debugging
# ------------------------------------
@njit
def ellipse(ys, f, a):
    ny = ys.size
    x = full((ny, 2), nan)
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
            x[i, 0] = (-d + u) / v
            x[i, 1] = (d + u) / v
    return x


@njit
def intersect(y, x, b, k):
    ny = y.size
    ys = k * y
    xs = k * x + b
    if b > 0.0:
        for i in range(ny):
            if isfinite(xs[i, 0]):
                xstar = sqrt(1.0 - ys[i] ** 2)
                if xstar < xs[i, 0]:
                    xs[i, :] = nan
                else:
                    xs[i, 1] = min(xs[i, 1], xstar)
    else:
        for i in range(ny):
            if isfinite(xs[i, 0]):
                xstar = -sqrt(1.0 - ys[i] ** 2)
                if xstar > xs[i, 1]:
                    xs[i, :] = nan
                else:
                    xs[i, 0] = max(xs[i, 0], xstar)
    return xs, ys


@njit
def area(xs, ys):
    dy = ys[1] - ys[0]
    a = 0.0
    for i in range(ys.size):
        if isfinite(xs[i, 0]):
            a += (xs[i, 1] - xs[i, 0]) * dy
    return a
