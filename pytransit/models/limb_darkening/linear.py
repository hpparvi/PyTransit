from math import pi

from numba import njit
from numpy import zeros


@njit(fastmath=True)
def ld_linear(mu, pv):
    r"""Linear limb darkening profile.

    .. math:: I(\mu) = 1 - u(1 - \mu)

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Limb darkening coefficients ``[u]``.

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    return 1. - pv[0] * (1. - mu)


@njit(fastmath=True)
def ldi_linear(pv):
    r"""Disk-integrated intensity of the linear profile.

    .. math:: 2\pi \int_0^1 I(\mu)\, z\, dz = \frac{\pi}{3}(3 - u)
    """
    return 2 * pi * 1 / 6 * (3 - pv[0])


@njit(fastmath=True)
def ldd_linear(mu, pv):
    ldd = zeros((2, mu.size))
    ldd[0] = pv[0]
    ldd[1] = mu - 1.0
    return ldd