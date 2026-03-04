from numba import njit
from numpy import floor


@njit(fastmath=True)
def _folded_time(t, t0, p):
    """Fold a time value to the interval around mid-transit.

    Parameters
    ----------
    t : float
        Time value.
    t0 : float
        Mid-transit time.
    p : float
        Orbital period.

    Returns
    -------
    tf : float
        Folded time centered on mid-transit.
    """
    epoch = floor((t - t0 + 0.5 * p) / p)
    return t - (t0 + epoch * p)
