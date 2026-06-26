from meepmeep import eclipse_light_travel_time
from meepmeep.numba2d import solve2d_d, sep_cd, bounding_box

# NOTE: ``eclipse_time_offset`` is not part of MeepMeep's public surface, but
# PyTransit and MeepMeep are developed together, so the deep import is fine.
from meepmeep.backends.numba.utils import eclipse_time_offset

from numba import njit, prange
from numpy import pi, zeros

from ._utils import _folded_time
from .ccintersection import ccia_and_grad


@njit
def _ltt_grad(p, a, i, e, w, rstar):
    """Central-difference gradient of the eclipse light-travel time.

    Returns ``d(ltt)/d(p, a, i, e, w)`` as a 5-element array. MeepMeep does
    not expose an analytic gradient for ``eclipse_light_travel_time``, but the
    function is smooth and this is evaluated only once per epoch, so a central
    finite difference is both cheap and accurate.
    """
    g = zeros(5)
    hs = (1e-6, 1e-5, 1e-6, 1e-6, 1e-6)  # steps for (p, a, i, e, w)
    hp = hs[0]
    g[0] = (eclipse_light_travel_time(p + hp, a, i, e, w, rstar) - eclipse_light_travel_time(p - hp, a, i, e, w, rstar)) / (2.0 * hp)
    ha = hs[1]
    g[1] = (eclipse_light_travel_time(p, a + ha, i, e, w, rstar) - eclipse_light_travel_time(p, a - ha, i, e, w, rstar)) / (2.0 * ha)
    hi = hs[2]
    g[2] = (eclipse_light_travel_time(p, a, i + hi, e, w, rstar) - eclipse_light_travel_time(p, a, i - hi, e, w, rstar)) / (2.0 * hi)
    he = hs[3]
    g[3] = (eclipse_light_travel_time(p, a, i, e + he, w, rstar) - eclipse_light_travel_time(p, a, i, e - he, w, rstar)) / (2.0 * he)
    hw = hs[4]
    g[4] = (eclipse_light_travel_time(p, a, i, e, w + hw, rstar) - eclipse_light_travel_time(p, a, i, e, w - hw, rstar)) / (2.0 * hw)
    return g


@njit
def _semodel_grad(t, k, cf, dcf, flux, dflux):
    """Accumulate the secondary-eclipse flux and its gradient for one time stamp.

    The eclipse flux is ``pi * k**2 - A(k, z)``, where ``A`` is the
    circle-circle intersection area between the stellar and planetary disks
    and ``z`` is their projected separation. The constant ``pi * k**2`` term
    is the out-of-eclipse planet flux, so it contributes a ``2 * pi * k`` term
    to the radius-ratio derivative that has no counterpart in the transit
    (uniform-disk) model.

    Parameters
    ----------
    t : float
        Folded time value relative to mid-eclipse.
    k : float
        Planet-to-star radius ratio.
    cf : ndarray
        Taylor-series coefficients for the projected distance.
    dcf : ndarray
        Derivatives of the Taylor-series coefficients w.r.t. orbital parameters.
    flux : ndarray
        Output array for the flux value (modified in-place).
    dflux : ndarray
        Output array for flux gradients (modified in-place).
    """
    flux[0] += pi * k ** 2
    dflux[0] += 2.0 * pi * k
    z, dz = sep_cd(t, cf, dcf)
    if z <= 1.0 + k:
        is_area, (dadk, dadz) = ccia_and_grad(1.0, k, z)
        flux[0] -= is_area
        dflux[0] -= dadk
        # dz is the 7-element (tc, p, a, i, e, w, lan) gradient from sep_cd;
        # slot 0 is the eclipse-centre derivative and lan (slot 6) is unused.
        for i in range(6):
            dflux[i + 1] -= dadz * dz[i]


def semodel_grad(times, k, t0, p, a, i, e, w, rstar, lcids, pbids, epids, nsamples, exptimes, npb, nep):
    """Evaluate the secondary-eclipse model and its gradient over an array of times.

    Computes both the occultation flux and its analytical gradient with
    respect to the planet-to-star radius ratio and orbital parameters. See
    :func:`pytransit.backends.numba.semodel.semodel` for the physical model.

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
        Secondary-eclipse flux with shape (npv, npt).
    dflux : ndarray
        Gradient array with shape (npv, npt, 7), where the last axis
        corresponds to derivatives w.r.t. [k, t0, p, a, i, e, w].
    """

    npv = k.shape[0]
    npt = times.size
    flux = zeros((npv, npt))
    dflux = zeros((npv, npt, 7))

    for ipv in range(npv):
        xyc = zeros((nep, 2, 5))
        dxyc = zeros((nep, 7, 2, 5))
        eclipse_shifts = zeros(nep)
        ltts = zeros(nep)
        dltts = zeros((nep, 5))   # d(ltt)/d(p, a, i, e, w) per epoch
        for iep in range(nep):
            eclipse_shifts[iep] = eclipse_time_offset(p[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])
            xyc[iep], dxyc[iep] = solve2d_d(eclipse_shifts[iep], p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])
            ltts[iep] = eclipse_light_travel_time(p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep], rstar)
            dltts[iep] = _ltt_grad(p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep], rstar)

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
                    _semodel_grad(t + time_offset, k[ipv, ipb], xyc[iep], dxyc[iep],
                                  flux[ipv, ipt:ipt+1], dflux[ipv, ipt, :])
                flux[ipv, ipt] /= nsamples[ilc]
                dflux[ipv, ipt, :] /= nsamples[ilc]
                # Light-travel-time correction: ltt shifts the eclipse centre
                # exactly like t0 (te = t0 + offset + ltt), so its parameter
                # dependence contributes d(flux)/d(t0) * d(ltt)/dX to each
                # orbital parameter X = (p, a, i, e, w). Unlike the eclipse-time
                # offset, this term does not cancel against the expansion point.
                g_t0 = dflux[ipv, ipt, 1]
                for j in range(5):
                    dflux[ipv, ipt, 2 + j] += g_t0 * dltts[iep, j]
            else:
                # Out of eclipse the flux is the constant planet flux pi*k**2,
                # which only depends on the radius ratio.
                flux[ipv, ipt] = pi * k[ipv, ipb] ** 2
                dflux[ipv, ipt, 0] = 2.0 * pi * k[ipv, ipb]
    return flux, dflux
