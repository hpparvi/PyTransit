from meepmeep import eclipse_light_travel_time
from meepmeep.numba2d import sep_c, solve2d, bounding_box

# NOTE: ``eclipse_time_offset`` is not part of MeepMeep's public surface, but
# PyTransit and MeepMeep are developed together, so the deep import is fine.
from meepmeep.backends.numba.utils import eclipse_time_offset

from numba import njit, prange
from numpy import pi, zeros

from ._utils import _folded_time
from .ccintersection import ccia


@njit
def _semodel(t, k, cf, flux):
    """Accumulate the secondary-eclipse flux for a single time stamp.

    The planet is treated as a uniformly bright disk of radius ``k`` (in
    stellar radii). Out of eclipse its disk contributes a constant flux of
    ``pi * k**2``; during the eclipse the stellar disk occults part of it,
    removing the circle-circle intersection area between the two disks.

    Parameters
    ----------
    t : float
        Folded time value relative to mid-eclipse.
    k : float
        Planet-to-star radius ratio.
    cf : ndarray
        Taylor-series polynomial coefficients for the projected distance.
    flux : ndarray
        Output array for the flux value (modified in-place).
    """
    flux[0] += pi * k ** 2
    z = sep_c(t, cf)
    if z <= 1.0 + k:
        flux[0] -= ccia(1.0, k, z)


def semodel(times, k, t0, p, a, i, e, w, rstar, lcids, pbids, epids, nsamples, exptimes, npb, nep):
    """Evaluate the secondary-eclipse model over an array of times.

    Computes the occultation light curve produced when the planet passes
    behind the star. The planet is modelled as a uniformly bright disk, so
    the out-of-eclipse flux is ``pi * k**2`` and the eclipse removes the
    fraction of the planet's disk hidden by the star. Limb darkening of the
    host star is not modelled (only the geometric overlap matters).

    The eclipse centre is offset from the mid-transit time ``t0`` by the
    eccentricity-dependent ``eclipse_time_offset`` plus the transit-to-eclipse
    light-travel delay (which is why ``rstar`` is required).

    Parameters
    ----------
    times : ndarray
        Array of mid-observation times.
    k : ndarray
        Planet-to-star radius ratio, shape (npv, npb).
    t0 : ndarray
        Mid-transit time, shape (npv, nep).
    p : ndarray
        Orbital period, shape (npv, nep).
    a : ndarray
        Scaled semi-major axis (a/R_star), shape (npv, nep).
    i : ndarray
        Orbital inclination [rad], shape (npv, nep).
    e : ndarray
        Orbital eccentricity, shape (npv, nep).
    w : ndarray
        Argument of periastron [rad], shape (npv, nep).
    rstar : float
        Stellar radius [R_sun], used for the light-travel-time correction.
    lcids : ndarray
        Light curve index for each time stamp.
    pbids : ndarray
        Passband index for each light curve.
    epids : ndarray
        Epoch index for each light curve.
    nsamples : ndarray
        Number of supersamples per light curve.
    exptimes : ndarray
        Exposure time per light curve.
    npb : int
        Number of passbands.
    nep : int
        Number of epochs.

    Returns
    -------
    flux : ndarray
        Secondary-eclipse flux with shape (npv, npt). Equals ``pi * k**2``
        out of eclipse and drops towards zero at mid-eclipse.
    """

    npv = k.shape[0]
    npt = times.size   # Number of points
    flux = zeros((npv, npt))

    for ipv in range(npv):
        xyc = zeros((nep, 2, 5))
        eclipse_shifts = zeros(nep)
        ltts = zeros(nep)
        for iep in range(nep):
            eclipse_shifts[iep] = eclipse_time_offset(p[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])
            xyc[iep, :, :] = solve2d(eclipse_shifts[iep], p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])
            ltts[iep] = eclipse_light_travel_time(p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep], rstar)

        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

        for ipt in prange(npt):
            ilc = lcids[ipt]
            ipb = pbids[ilc]

            itc = epids[ilc]
            if nep > 1:
                iep = epids[ilc]
            else:
                iep = 0

            te = t0[ipv, itc] + eclipse_shifts[iep] + ltts[iep]
            t = _folded_time(times[ipt], te, p[ipv, iep])
            if ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    _semodel(t + time_offset, k[ipv, ipb], xyc[iep], flux[ipv:ipv+1, ipt:ipt+1])
                flux[ipv, ipt] /= nsamples[ilc]
            else:
                flux[ipv, ipt] = pi * k[ipv, ipb] ** 2
    return flux
