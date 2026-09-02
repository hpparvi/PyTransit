from math import pi

from numba import njit


@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    r"""Four-coefficient non-linear limb darkening profile (Claret, A&A 363, 1081, 2000).

    .. math::
        I(\mu) = 1 - c_1(1 - \mu^{1/2}) - c_2(1 - \mu) - c_3(1 - \mu^{3/2}) - c_4(1 - \mu^2)

    The standard law for tabulated theoretical limb darkening coefficients. Its four
    coefficients are strongly correlated, so it is usually a poor choice for *fitting* but a
    good one for *fixing* limb darkening to model predictions.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle between the surface normal and the line of sight.
    pv : ndarray
        Limb darkening coefficients ``[c1, c2, c3, c4]``.

    Returns
    -------
    ndarray
        Stellar intensity.
    """
    return 1. - pv[0] * (1. - mu**0.5) - pv[1] * (1. - mu) - pv[2] * (1. - mu**1.5) - pv[3] * (1. - mu ** 2)


@njit(fastmath=True)
def ldi_nonlinear(pv):
    """Disk-integrated intensity of the four-coefficient non-linear profile.
    """
    return 2 * pi * (0.5 - pv[0] / 10 - pv[1] / 6 - 3 * pv[2] / 14 - pv[3] / 4)
