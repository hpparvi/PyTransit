from math import pi, sqrt

from numba import njit
from numpy import zeros

@njit(fastmath=True)
def ld_quadratic_tri(mu, pv):
    r"""Quadratic limb darkening profile with the triangular sampling parametrisation.

    The quadratic law reparametrised following Kipping (MNRAS 435, 2152, 2013) as

    .. math:: u = 2\sqrt{q_1} q_2, \qquad v = \sqrt{q_1}(1 - 2 q_2)

    A uniform prior on :math:`(q_1, q_2) \in [0,1]^2` maps to a uniform prior over the whole
    region of :math:`(u, v)` space that gives a physically valid (everywhere positive,
    monotonically decreasing) intensity profile. This makes it the right parametrization for
    sampling limb darkening coefficients with uninformative priors.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Triangular sampling coefficients ``[q1, q2]``, both in [0, 1].

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u * (1. - mu) - v * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic_tri(pv):
    """Disk-integrated intensity of the triangularly parametrised quadratic profile.
    """
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 2 * pi * 1 / 12 * (-2 * u - v + 6)
