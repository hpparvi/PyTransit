from numba import njit
from numpy import zeros, pi


@njit(fastmath=True)
def ld_quadratic(mu, pv):
    """Quadratic limb darkening model: I(mu) = 1 - a*(1 - mu) - b*(1 - mu)^2.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    intensity : ndarray
        Limb darkening profile evaluated at each mu.
    """
    return 1. - pv[0] * (1. - mu) - pv[1] * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic(pv):
    """Disk-integrated intensity for the quadratic limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return 2 * pi * 1 / 12 * (-2 * pv[0] - pv[1] + 6)

@njit(fastmath=True)
def ldig_quadratic(pv):
    """Gradient of integrated intensity for the quadratic limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    gradient : ndarray
        Array of shape ``(2,)`` with ``[dI_int/da, dI_int/db]``.
    """
    g = zeros(2)
    g[0] = -pi / 3.0
    g[1] = -pi / 6.0
    return g


@njit(fastmath=True)
def ldd_quadratic(mu, pv):
    """Derivatives of the quadratic limb darkening model.

    Multiply dI/dmu by ``-z / sqrt(1 - z**2)`` to convert to dI/dz.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    ldd : ndarray
        Array of shape ``(3, n_mu)`` with rows ``[dI/dmu, dI/da, dI/db]``.
    """
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0] + 2*pv[1] - 2*pv[1]*mu
    ldd[1] = mu - 1.0
    ldd[2] = -(1.0 - mu)**2
    return ldd