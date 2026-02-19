from numba import njit
from numpy import arccos, sqrt, arctan2, pi, nan, floor


@njit
def _tsort(r1, r2, b):
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
def ccia(r1, r2, b):
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
    """
    if r1 + r2 <= b:
        return 0.0
    elif abs(r1 - r2) < b and b <= r1 + r2:
        x, y, z = _tsort(r1, r2, b)
        a_kite = 0.5 * sqrt((x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z)))
        k0 = arctan2(2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b)
        k1 = arctan2(2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b)
        a_lens = r1 * r1 * k1 + r2 * r2 * k0 - a_kite
        return a_lens
    elif b <= r1 - r2:
        return pi * r2 ** 2
    elif b <= r2 - r1:
        return pi * r1 ** 2
    else:
        return nan


@njit
def ccia_and_grad(r1, r2, b):
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
    """
    if r1 + r2 <= b:
        return 0.0, (0.0, 0.0)
    elif abs(r1 - r2) < b and b <= r1 + r2:
        x, y, z = _tsort(r1, r2, b)
        a_kite = 0.5 * sqrt((x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z)))
        k0 = arctan2(2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b)
        k1 = arctan2(2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b)
        a_lens = r1 * r1 * k1 + r2 * r2 * k0 - a_kite
    elif b <= r1 - r2:
        a_lens = pi * r2 ** 2
        k0 = pi
    elif b <= r2 - r1:
        k0 = 0.0
        a_lens = pi * r1 ** 2
    else:
        return nan, (nan, nan)

    dadr2 = 2*k0*r2

    if abs(b) > 1e-8:
        dadb = -2*a_kite/b
    else:
        dadb = 0.0

    return a_lens, (dadr2, dadb)
