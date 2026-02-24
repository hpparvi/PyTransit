from numba import njit
from numpy import zeros, log, pi


@njit(fastmath=True)
def ld_power_2(mu, pv):
    """Power-2 limb darkening model: I(mu) = 1 - c*(1 - mu^alpha).

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    intensity : ndarray
        Limb darkening profile evaluated at each mu.
    """
    return 1. - pv[0] * (1. - mu ** pv[1])


@njit(fastmath=True)
def ldi_power_2(pv):
    """Disk-integrated intensity for the power-2 limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return pi * (1.0 - pv[0] * pv[1] / (pv[1] + 2.0))


@njit(fastmath=True)
def ldig_power_2(pv):
    """Gradient of integrated intensity for the power-2 limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    gradient : ndarray
        Array of shape ``(2,)`` with ``[dI_int/dc, dI_int/dalpha]``.
    """
    c, a = pv[0], pv[1]
    g = zeros(2)
    g[0] = -pi * a / (a + 2.0)
    g[1] = -2.0 * pi * c / (a + 2.0) ** 2
    return g


@njit(fastmath=True)
def ldd_power_2(mu, pv):
    """Derivatives of the power-2 limb darkening model.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    ldd : ndarray
        Array of shape ``(3, n_mu)`` with rows
        ``[dI/dmu, dI/dc, dI/dalpha]``.
    """
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0]*pv[1]*mu**(pv[1]-1.0)
    ldd[1] = mu**pv[1] - 1.0
    ldd[2] = pv[0]*mu**pv[1] * log(mu)
    return ldd