from math import pi

from numba import njit
from numpy import zeros, log, log2


@njit(fastmath=True)
def ld_power_2(mu, pv):
    r"""Power-2 limb darkening profile (Hestroffer, A&A 327, 199, 1997).

    .. math:: I(\mu) = 1 - c(1 - \mu^{\alpha})

    A two-coefficient law that reproduces the limb darkening of *cool stars* considerably
    better than the quadratic law, especially in the near-infrared, and is the recommended
    two-parameter law for M dwarfs.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    ndarray
        Stellar intensity.

    See Also
    --------
    ld_power_2_pm : the same law in the better-behaved (h1, h2) parametrisation.
    """
    return 1. - pv[0] * (1. - mu ** pv[1])


@njit
def ldi_power_2(pv):
    """Disk-integrated intensity of the power-2 profile.
    """
    return 2 * pi * (0.5 - 0.5 * pv[0] + pv[0] / (pv[1] + 2.0))


@njit(fastmath=True)
def ldd_power_2(mu, pv):
    ldd = zeros((3, mu.size))
    ldd[0] = pv[0]*pv[1]*mu**(pv[1]-1.0)
    ldd[1] = mu**pv[1] - 1.0
    ldd[2] = pv[0]*mu**pv[1] * log(mu)
    return ldd


@njit(fastmath=True)
def ld_power_2_pm(mu, pv):
    r"""Power-2 limb darkening profile in the (h1, h2) parametrisation.

    The power-2 law reparametrised following Maxted (A&A 622, A33, 2018) using the intensity at
    two fixed points on the disk,

    .. math:: h_1 = I(\mu = 1/2), \qquad h_2 = h_1 - I(\mu = 1/\sqrt{2}),

    which are far less correlated than :math:`(c, \alpha)` and therefore much better behaved as
    free parameters in a fit.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Limb darkening coefficients ``[h1, h2]``.

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    c = 1 - pv[0] + pv[1]
    a = log2(c/pv[1])
    return 1. - c * (1. - mu**a)


@njit
def ldi_power_2_pm(pv):
    """Disk-integrated intensity of the (h1, h2)-parametrised power-2 profile.
    """
    c = 1 - pv[0] + pv[1]
    a = log2(c/pv[1])
    return 2 * pi * (0.5 - 0.5 * c + c / (a + 2.0))
