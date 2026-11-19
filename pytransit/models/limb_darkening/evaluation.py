from numba import njit
from numba.core.dispatcher import Dispatcher
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


_ld_evaluators = {}


def ld_evaluator(ld, ldi):
    """A compiled function evaluating a limb darkening law and its disk integral together.

    `evaluate_ld` and `evaluate_ldi` take the law as an argument, which costs several
    microseconds of dispatch overhead per call. Binding the law into a compiled closure removes
    that overhead, which matters when a transit model is evaluated for a short light curve. The
    closures are cached, so each law is compiled only once.

    Parameters
    ----------
    ld, ldi
        The Numba-compiled limb darkening profile and its disk integral.

    Returns
    -------
    callable or None
        A function ``f(mu, ldc) -> (ldp, istar)`` taking the ``(npv, npb, nldc)`` coefficient
        array and returning the ``(npv, npb, nmu)`` profiles and the ``(npv, npb)`` disk
        integrals, or None if either argument is not a Numba-compiled function.
    """
    if not (isinstance(ld, Dispatcher) and isinstance(ldi, Dispatcher)):
        return None
    key = (ld, ldi)
    if key not in _ld_evaluators:
        @njit
        def evaluate(mu, pv):
            npv = pv.shape[0]
            npb = pv.shape[1]
            ldp = zeros((npv, npb, mu.size))
            istar = zeros((npv, npb))
            for ipv in range(npv):
                for ipb in range(npb):
                    ldp[ipv, ipb, :] = ld(mu, pv[ipv, ipb])
                    istar[ipv, ipb] = ldi(pv[ipv, ipb])
            return ldp, istar
        _ld_evaluators[key] = evaluate
    return _ld_evaluators[key]
