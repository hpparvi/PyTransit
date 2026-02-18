from numba import njit
from numpy import arccos, sqrt, arctan2, pi, nan, floor


@njit
def tsort(r1, r2, b):
    """
    Sort three values (radii and separation) in descending order.

    This utility ensures numerical stability for area calculations by
    assigning x, y, z such that x >= y >= z.

    Parameters
    ----------
    r1 : float
        Radius of the first circle.
    r2 : float
        Radius of the second circle.
    b : float
        Separation between the centers of the two circles.

    Returns
    -------
    x, y, z : float
        The input values sorted in descending order.
    """
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
    """
    Calculate the area of intersection between two circles and the angle k0.

    Adapted from Agol et al. (2020). Handles all overlap cases: no overlap,
    partial overlap, and complete occultation.

    Parameters
    ----------
    r1 : float
        Radius of the first circle (e.g., the occulted star).
    r2 : float
        Radius of the second circle (e.g., the occulting planet).
    b : float
        Distance between the centers of the two circles.

    Returns
    -------
    a_lens : float
        The area of the intersection (overlap region).
    k0 : float
        An auxiliary angle used in limb-darkening calculations, specifically
        the half-angle subtended by the intersection at the center of r2.
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
def dadz(z, dz, r1, r2):
    """
    Compute the derivative of the intersection area with respect to separation z.

    Parameters
    ----------
    z : float
        Separation between circle centers.
    dz : float
        The derivative of z with respect to a higher-level parameter (chain rule).
    r1 : float
        Radius of the first circle.
    r2 : float
        Radius of the second circle.

    Returns
    -------
    da_dz : float
        The partial derivative of the intersection area with respect to the
        orbital parameters, scaled by dz.
    """
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
