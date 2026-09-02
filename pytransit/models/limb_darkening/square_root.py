from math import pi

from numba import njit
from numpy import sqrt


@njit(fastmath=True)
def ld_square_root(mu, pv):
    r"""Square-root limb darkening profile (Diaz-Cordoves & Gimenez, A&A 259, 227, 1992).

    .. math:: I(\mu) = 1 - u(1 - \mu) - v(1 - \sqrt{\mu})

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
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - sqrt(mu))


@njit(fastmath=True)
def ldi_square_root(pv):
    """Disk-integrated intensity of the square-root profile.
    """
    return 2 * pi * (0.5 - pv[0] / 6 - pv[1] / 10)
