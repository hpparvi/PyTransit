from numba import njit
from numpy import zeros, pi


@njit(fastmath=True)
def ld_linear(mu, pv):
    """Linear limb darkening model: I(mu) = 1 - u*(1 - mu).

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    intensity : ndarray
        Limb darkening profile evaluated at each mu.
    """
    return 1. - pv[0] * (1. - mu)


@njit(fastmath=True)
def ldi_linear(pv):
    """Disk-integrated intensity for the linear limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return 2 * pi * 1 / 6 * (3 - pv[0])


@njit(fastmath=True)
def ldig_linear(pv):
    """Gradient of integrated intensity for the linear limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    gradient : ndarray
        Array of shape ``(1,)`` with dI_int/du.
    """
    g = zeros(1)
    g[0] = -pi / 3.0
    return g


@njit(fastmath=True)
def ldd_linear(mu, pv):
    """Derivatives of the linear limb darkening model.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    ldd : ndarray
        Array of shape ``(2, n_mu)`` with rows ``[dI/dmu, dI/du]``.
    """
    ldd = zeros((2, mu.size))
    ldd[0] = pv[0]
    ldd[1] = mu - 1.0
    return ldd