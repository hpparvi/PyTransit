from numba import njit
from numpy import zeros, pi


@njit(fastmath=True)
def ld_general(mu, pv):
    """General limb darkening model: I(mu) = 1 - sum(c_i * (1 - mu^(i+1))).

    Supports an arbitrary number of coefficients.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    intensity : ndarray
        Limb darkening profile evaluated at each mu.
    """
    ldp = zeros(mu.size)
    for i in range(pv.size):
        ldp += pv[i] * (1.0 - mu ** (i + 1))
    return ldp


@njit(fastmath=True)
def ldi_general(pv):
    """Disk-integrated intensity for the general limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    s = 0.0
    for i in range(pv.size):
        s += pv[i] * (i + 1.0) / (2.0 * (i + 3.0))
    return 2 * pi * s


@njit(fastmath=True)
def ldig_general(pv):
    """Gradient of integrated intensity for the general limb darkening model.

    Parameters
    ----------
    pv : ndarray
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    gradient : ndarray
        Array of shape ``(n,)`` with ``[dI_int/dc_1, ..., dI_int/dc_n]``.
    """
    n = pv.size
    g = zeros(n)
    for i in range(n):
        g[i] = pi * (i + 1.0) / (i + 3.0)
    return g


@njit(fastmath=True)
def ldd_general(mu, pv):
    """Derivatives of the general limb darkening model.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    ldd : ndarray
        Array of shape ``(1 + n, n_mu)`` with rows
        ``[dI/dmu, dI/dc_1, ..., dI/dc_n]``.
    """
    n = pv.size
    ldd = zeros((1 + n, mu.size))
    for i in range(n):
        ldd[0] -= pv[i] * (i + 1.0) * mu ** i
        ldd[1 + i] = 1.0 - mu ** (i + 1)
    return ldd