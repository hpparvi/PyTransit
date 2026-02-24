from numba import njit
from numpy import zeros


@njit
def evaluate_ld(ldm, mu, pvo):
    """Evaluate a limb darkening profile across parameter sets and passbands.

    Parameters
    ----------
    ldm : callable
        Limb darkening model function with signature ``ldm(mu, pv)``.
    mu : ndarray
        Array of mu (= cos(theta)) values.
    pvo : ndarray
        Limb darkening parameter array with 1, 2, or 3 dimensions. A 1D array
        is treated as a single parameter set for one passband, a 2D array as
        one parameter set with multiple passbands, and a 3D array as
        ``(n_sets, n_passbands, n_coeffs)``.

    Returns
    -------
    ldp : ndarray
        Limb darkening profiles with shape ``(n_sets, n_passbands, n_mu)``.
    """
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo

    npv = pv.shape[0]
    npb = pv.shape[1]
    ldp = zeros((npv, npb, mu.size))
    for ipv in range(npv):
        for ipb in range(npb):
            ldp[ipv, ipb, :] = ldm(mu, pv[ipv, ipb])
    return ldp


@njit
def evaluate_ldi(ldi, pvo):
    """Evaluate integrated limb darkening intensity across parameter sets and passbands.

    Parameters
    ----------
    ldi : callable
        Integrated limb darkening function with signature ``ldi(pv)``.
    pvo : ndarray
        Limb darkening parameter array (1D, 2D, or 3D).

    Returns
    -------
    istar : ndarray
        Integrated stellar intensities with shape ``(n_sets, n_passbands)``.
    """
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo

    npv = pv.shape[0]
    npb = pv.shape[1]
    istar = zeros((npv, npb))
    for ipv in range(npv):
        for ipb in range(npb):
            istar[ipv, ipb] = ldi(pv[ipv, ipb])
    return istar


@njit
def evaluate_ldig(ldig, pvo):
    """Evaluate the gradient of integrated limb darkening intensity.

    Parameters
    ----------
    ldig : callable
        Gradient function with signature ``ldig(pv)``.
    pvo : ndarray
        Limb darkening parameter array (1D, 2D, or 3D).

    Returns
    -------
    gradient : ndarray
        Gradient of the integrated intensity with respect to the limb
        darkening coefficients.
    """
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo
    return ldig(pv[0, 0])
