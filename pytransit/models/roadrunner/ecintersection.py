from math import cos, sin, sqrt, isfinite, nan

from numpy import ndarray, fabs, pi, full, arange
from numba import njit

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
        if cx < -k:
            for i in range(ny):
                if isfinite(xs[i, 0]):
                    xstar = -sqrt(1.0 - (ys[i] + cy) ** 2) - cx
                    if xstar <= xs[i, 0]:
                        l += xs[i, 1] - xs[i, 0]
                    elif xstar <= xs[i, 1]:
                        l += xs[i, 1] - xstar
        elif cx > k:
            for i in range(ny):
                if isfinite(xs[i, 0]):
                    xstar = sqrt(1.0 - (ys[i] + cy) ** 2) - cx
                    if xstar >= xs[i, 1]:
                        l += xs[i, 1] - xs[i, 0]
                    elif xstar >= xs[i, 0]:
                        l += xstar - xs[i, 0]
        else:
            for i in range(ny):
                if isfinite(xs[i, 0]):
                    if fabs(ys[i] + cy) <= 1.0:
                        xstar = sqrt(1.0 - (ys[i] + cy) ** 2)
                        xst1 = -xstar - cx
                        xst2 = xstar - cx
                        if xst1 <= xs[i, 0]:
                            l += min(xst2, xs[i, 1]) - xs[i, 0]
                        elif xst1 > xs[i, 0]:
                            l +=  min(xst2, xs[i, 1]) - xst1
        return l*dy

        #else:
        #    for i in range(ny):
        #        if isfinite(xs[i,0]):
        #            xstar = sqrt(1.0 - (ys[i])**2) - b
        #            if xstar < xs[i,1]:
        #                l += xs[i,1]-  max(xstar, xs[i,0])
        #    return pi*k*k*(1.0-f) - l*dy
