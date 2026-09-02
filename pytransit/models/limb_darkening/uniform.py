from math import pi

from numba import njit
from numpy import ones


@njit(fastmath=True)
def ld_uniform(mu, pv):
    r"""Uniform (no limb darkening) intensity profile.

    .. math:: I(\mu) = 1

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Unused, present for interface compatibility.

    Returns
    -------
    ndarray
        Stellar intensity, unity everywhere.
    """
    return ones(mu.size)


@njit(fastmath=True)
def ldi_uniform(pv):
    r"""Disk-integrated intensity of the uniform profile, :math:`\pi`.
    """
    return pi
