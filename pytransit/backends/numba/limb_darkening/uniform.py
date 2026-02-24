from math import pi

from numba import njit
from numpy import ones, zeros


@njit(fastmath=True)
def ld_uniform(mu, pv):
    """Uniform (constant) limb darkening: I(mu) = 1.

    Parameters
    ----------
    mu : ndarray
        Array of mu (= cos(theta)) values.
    pv : ndarray
        Limb darkening coefficients (unused, empty array).

    Returns
    -------
    intensity : ndarray
        Ones array of the same size as ``mu``.
    """
    return ones(mu.size)


@njit(fastmath=True)
def ldi_uniform(pv):
    """Disk-integrated intensity for the uniform limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients (unused).

    Returns
    -------
    intensity : float
        Integrated intensity, equal to pi.
    """
    return pi


@njit(fastmath=True)
def ldd_uniform(pv):
    """Derivative of the uniform limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients (unused).

    Returns
    -------
    derivative : float
        Always 0.0 (no dependence on any parameter).
    """
    return 0.0


@njit(fastmath=True)
def ldig_uniform(pv):
    """Gradient of integrated intensity for the uniform limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients (unused).

    Returns
    -------
    gradient : ndarray
        Empty array (no parameters to differentiate with respect to).
    """
    return zeros(0)