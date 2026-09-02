from math import pi

from numba import njit
from numpy import log


@njit(fastmath=True)
def ld_logarithmic(mu, pv):
    r"""Logarithmic limb darkening profile (Klinglesmith & Sobieski, AJ 75, 175, 1970).

    .. math:: I(\mu) = 1 - u(1 - \mu) - v\,\mu \ln \mu

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight. The profile
        diverges as :math:`\mu \to 0`, so the exact stellar limb needs care.
    pv : ndarray
        Limb darkening coefficients ``[u, v]``.

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    return 1. - pv[0] * (1. - mu) - pv[1] * mu * log(mu)


@njit(fastmath=True)
def ldi_logarithmic(pv):
    """Disk-integrated intensity of the logarithmic profile.
    """
    return 2 * pi * (0.5 - pv[0] / 6 + pv[1] / 9)
