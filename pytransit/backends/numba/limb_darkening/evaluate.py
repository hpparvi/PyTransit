from numba import njit
from numpy import zeros


@njit
def evaluate_ld(ldm, mu, pvo):
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
    if pvo.ndim == 1:
        pv = pvo.reshape((1, 1, -1))
    elif pvo.ndim == 2:
        pv = pvo.reshape((1, pvo.shape[1], -1))
    else:
        pv = pvo
    return ldig(pv[0, 0])
