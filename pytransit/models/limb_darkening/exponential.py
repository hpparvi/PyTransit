from math import pi

from numba import njit
from numpy import exp


@njit(fastmath=True)
def ld_exponential(mu, pv):
    r"""Exponential limb darkening profile (Claret & Hauschildt, A&A 412, 241, 2003).

    .. math:: I(\mu) = 1 - u(1 - \mu) - \frac{v}{1 - e^{\mu}}

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
    return 1. - pv[0] * (1. - mu) - pv[1] / (1. - exp(mu))


@njit(fastmath=True)
def ldi_exponential(pv):
    # -int_0^1 mu/(1-exp(mu)) dmu = pi^2/6 + ln(1-1/e) - Li2(1/e)
    return 2 * pi * (0.5 - pv[0] / 6 + 0.7775046341122480 * pv[1])
