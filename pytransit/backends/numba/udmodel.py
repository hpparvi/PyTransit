from meepmeep.backends.numba.ts2d import pd_t15c, solve_xy_p5
from meepmeep.backends.numba.ts2d.position import bounding_box
from meepmeep.backends.numba.utils import d_from_pkaiews

from numba import njit, prange
from numpy import floor, pi, zeros, nan, fabs, unique, isnan
from .ccintersection import ccia


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


@njit
def _udmodel(t, k, cf, flux):
    """Compute the flux deficit for a single time stamp (uniform disk).

    Evaluates the circle-circle intersection area between the stellar
    and planetary disks and writes the corresponding flux deficit into
    the output array.

    Parameters
    ----------
    t : float
        Folded time value relative to mid-transit.
    k : float
        Planet-to-star radius ratio.
    cf : ndarray
        Taylor-series polynomial coefficients for the projected distance.
    flux : ndarray
        Output array for the flux value (modified in-place).
    """
    z = pd_t15c(t, cf)
    if z <= 1.0 + k:
        is_area = ccia(1.0, k, z)
        flux[0] = -is_area / pi


def udmodel(times, k, t0, p, a, i, e, w):
    """Evaluate the uniform-disk transit model over an array of times.

    Computes the relative flux deficit caused by a planet transiting a
    star with no limb darkening. Returns NaN for unphysical parameter
    combinations (a <= 1 or e >= 0.99).

    Parameters
    ----------
    times : ndarray
        Array of observation times.
    k : float
        Planet-to-star radius ratio.
    t0 : float
        Mid-transit time.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a/R_star).
    i : float
        Orbital inclination [rad].
    e : float
        Orbital eccentricity.
    w : float
        Argument of periastron [rad].

    Returns
    -------
    flux : ndarray
        Relative flux deficit for each time stamp.
    """
    npt = times.size
    flux = zeros(npt)

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux

    cf = solve_xy_p5(0.0, p, a, i, e, w)

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1)
    for j in prange(npt):
        t = _folded_time(times[j], t0, p)
        if fabs(t) < half_window_width:
            _udmodel(t, k, cf, flux[j:j + 1])
    return flux


def udmodel_full(times, k, t0, p, a, i, e, w, lcids, pbids, epids, nsamples, exptimes):

    npt = times.size   # Number of points
    npv = k.shape[0]   # Number of parameter vectors
    npb = k.shape[1]   # Number of passbands
    nep = t0.shape[1]   # Number of epochs
    flux = zeros((npv, npt))

    for ipv in range(npv):

        xyc = zeros((nep, 2, 5))
        xyc[0, :, :] = solve_xy_p5(0.0, p[ipv, 0], a[ipv, 0], i[ipv, 0], e[ipv, 0], w[ipv, 0])
        if nep > 1 and isnan(p[ipv, 1]):
            xyc[:, :, :] = xyc[0:1, :, :]
        else:
            for iep in range(nep):
                xyc[iep, :, :] = solve_xy_p5(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003 + exptimes[0]
        bt4 += 0.003 + exptimes[0]

        for ipt in range(npt):
            ilc = lcids[ipt]
            ipb = pbids[ilc]
            iep = epids[ilc]

            t = _folded_time(times[ipt], t0[ipv, iep], p[ipv, iep])
            if (bt1 <= t <= bt4):
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    _udmodel(t + time_offset, k[ipv, ipb], xyc[iep], flux[ipv:ipv+1, ipt:ipt+1])
                flux[ipv, ipt] /= nsamples[ilc]
    return flux
