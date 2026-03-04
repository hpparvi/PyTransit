from meepmeep.backends.numba.ts2d import pd_t15c, solve_xy_p5
from meepmeep.backends.numba.ts2d.position import bounding_box

from numba import njit
from numpy import pi, zeros

from ._utils import _folded_time
from .ccintersection import ccia


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
        flux[0] -= is_area / pi


def udmodel(times, k, t0, p, a, i, e, w, lcids, pbids, epids, nsamples, exptimes, npv, npb, ntc, nor):
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

    npt = times.size   # Number of points
    flux = zeros((npv, npt))

    for ipv in range(npv):
        xyc = zeros((nor, 2, 5))
        for iep in range(nor):
            xyc[iep, :, :] = solve_xy_p5(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

        for ipt in range(npt):
            ilc = lcids[ipt]
            ipb = pbids[ilc]

            itc = epids[ilc]
            if nor > 1:
                iep = epids[ilc]
            else:
                iep = 0

            t = _folded_time(times[ipt], t0[ipv, itc], p[ipv, iep])
            if ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    _udmodel(t + time_offset, k[ipv, ipb], xyc[iep], flux[ipv:ipv+1, ipt:ipt+1])
                flux[ipv, ipt] /= nsamples[ilc]
    return flux
