from math import pi

from numba import njit
from numpy import zeros


@njit(fastmath=True)
def ld_quadratic(mu, pv):
    r"""Quadratic limb darkening profile (Kopal 1950).

    .. math:: I(\mu) = 1 - u(1 - \mu) - v(1 - \mu)^2

    The most widely used limb darkening law in transit modelling. It is flexible enough for
    most optical photometry but systematically misrepresents the profile of cool stars near the
    limb; see :func:`ld_power_2` for an alternative with the same number of coefficients.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Limb darkening coefficients ``[u, v]``.

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic(pv):
    r"""Disk-integrated intensity of the quadratic profile.

    .. math:: \frac{\pi}{6}(6 - 2u - v)
    """
    return 2 * pi * 1 / 12 * (-2 * pv[0] - pv[1] + 6)

@njit(fastmath=True)
def ldd_quadratic(mu, pv):
    """Quadratic limb darkening model derivatives.

    Quadratic limb darkening model derivatives as an array
    [di/dmu, di/da, di/db].

    Multiply di/dmu by -z/sqrt(1-z**2) to get di/dz"""
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0] + 2*pv[1] - 2*pv[1]*mu
    ldd[1] = mu - 1.0
    ldd[2] = -(1.0 - mu)**2
    return ldd