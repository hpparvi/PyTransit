from math import cos, sin, sqrt, isfinite, nan

from numpy import ndarray, fabs, pi, full, arange
from numba import njit

@njit
def create_ellipse(ny: int, k: float, f: float, a: float) -> (ndarray, ndarray):
    """
    Create the coordinates for a rotated ellipse.

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
def ellipse_circle_intersection_area(b: float, k: float, f: float, xs: ndarray, ys: ndarray) -> float:
    """
     Calculate the intersection area between a rotated ellipse and a circle using a scanline fill approach.

   Parameters
    ----------
    b : float
        The offset of the circle's center along the x-axis.
    k : float
        Scale factor for the ellipse.
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
    b = fabs(b)
    if b <= 1.0 - k:
        return pi * k * (1.0 - f) * k
    elif b >= 1.0 + k:
        return 0.0
    else:
        ny = ys.size
        dy = (ys[1] - ys[0])
        l = 0.0
        if b >= 1.0:
            for i in range(ny):
                if isfinite(xs[i,0]):
                    xstar = sqrt(1.0 - (ys[i])**2) - b
                    if xstar > xs[i,0]:
                        l += min(xstar, xs[i,1]) - xs[i,0]
            return l*dy
        else:
            for i in range(ny):
                if isfinite(xs[i,0]):
                    xstar = sqrt(1.0 - (ys[i])**2) - b
                    if xstar < xs[i,1]:
                        l += xs[i,1]-  max(xstar, xs[i,0])
            return pi*k*k*(1.0-f) - l*dy
