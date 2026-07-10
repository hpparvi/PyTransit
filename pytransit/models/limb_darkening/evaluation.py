from numba import njit
from numpy import zeros


@njit
def evaluate_ld(ldm, mu, pvo):
    """Evaluate a limb darkening model for 1D, 2D, or 3D parameter arrays.

    The parameters can be given as a 1D array with a shape (nldc), a 2D array
    with a shape (npb, nldc), or a 3D array with a shape (npv, npb, nldc),
    where nldc is the number of limb darkening coefficients, npb is the number
    of passbands, and npv is the number of parameter vectors.

    Returns the limb darkening profiles as a 3D array with a shape
    (npv, npb, nmu).
    """
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[0], -1))
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
    """Evaluate a disk-integrated stellar intensity for 1D, 2D, or 3D parameter arrays.

    The parameters follow the same conventions as in ``evaluate_ld``.

    Returns the disk-integrated stellar intensities as a 2D array with a
    shape (npv, npb).
    """
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[0], -1))
    else:
        pv = pvo

    npv = pv.shape[0]
    npb = pv.shape[1]
    istar = zeros((npv, npb))
    for ipv in range(npv):
        for ipb in range(npb):
            istar[ipv, ipb] = ldi(pv[ipv, ipb])
    return istar
