from numba import njit
from numpy import zeros, pi, sqrt

@njit(fastmath=True)
def ld_quadratic_tri(mu, pv):
    """Quadratic limb darkening with Kipping (2013) triangular parameterization.

    Reparameterizes the standard quadratic coefficients (u, v) in terms of
    q1 and q2 via ``u = sqrt(q1) * 2*q2`` and ``v = sqrt(q1) * (1 - 2*q2)``.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    intensity : ndarray
        Limb darkening profile evaluated at each mu.
    """
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u * (1. - mu) - v * (1. - mu) ** 2


@njit(fastmath=True)
def ldi_quadratic_tri(pv):
    """Disk-integrated intensity for the triangular quadratic model.

    Parameters
    ----------
    pv : ndarray
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    a, b = sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1. - b)
    return 2 * pi * 1 / 12 * (-2 * u - v + 6)


@njit(fastmath=True)
def ldig_quadratic_tri(pv):
    """Gradient of integrated intensity for the triangular quadratic model.

    Parameters
    ----------
    pv : ndarray
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    gradient : ndarray
        Array of shape ``(2,)`` with ``[dI_int/dq1, dI_int/dq2]``.
    """
    a = sqrt(pv[0])
    g = zeros(2)
    g[0] = -pi * (1.0 + 2.0 * pv[1]) / (12.0 * a)
    g[1] = -pi * a / 3.0
    return g


@njit(fastmath=True)
def ldd_quadratic_tri(mu, pv):
    """Derivatives of the triangular quadratic limb darkening model.

    Parameters
    ----------
    mu : ndarray
        Array of mu values.
    pv : ndarray
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    ldd : ndarray
        Array of shape ``(3, n_mu)`` with rows ``[dI/dmu, dI/dq1, dI/dq2]``.
    """
    a = sqrt(pv[0])
    b = 2 * pv[1]
    u = a * b
    v = a * (1.0 - b)
    omu = 1.0 - mu

    ldd = zeros((3, mu.size))
    ldd[0] = u + 2.0 * v * omu
    ldd[1] = -(0.5 / a) * (b * omu + (1.0 - b) * omu ** 2)
    ldd[2] = -2.0 * a * mu * omu
    return ldd
