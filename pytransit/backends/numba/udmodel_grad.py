from meepmeep.backends.numba.ts2d import solve_xy_p5_d, pd_t15c_d
from meepmeep.backends.numba.ts2d.position import bounding_box
from numba import prange, njit
from numpy import zeros, nan, fabs, pi

from .ccintersection import ccia_and_grad
from .udmodel import _folded_time


@njit
def _udmodel_grad(t, k, cf, dcf, flux, dflux):
    """Compute the flux deficit and its gradient for a single time stamp.

    Evaluates the circle-circle intersection area and its analytical
    derivatives with respect to the radius ratio and orbital parameters.

    Parameters
    ----------
    t : float
        Folded time value relative to mid-transit.
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
    z, dz = pd_t15c_d(t, cf, dcf)
    if z <= 1.0 + k:
        is_area, (dadk, dadz) = ccia_and_grad(1.0, k, z)
        flux[0] -= is_area / pi
        dflux[0] -= dadk / pi
        dflux[1] += dadz * dz[0] / pi
        for i in range(1, 6):
            dflux[i+1] -= dadz * dz[i] / pi


def udmodel_grad(times, k, t0, p, a, i, e, w, lcids, pbids, epids, nsamples, exptimes, npv, npb, ntc, nor):
    """Evaluate the uniform-disk transit model and gradient over an array of times.

    Computes both the relative flux deficit and its analytical gradient
    with respect to the planet-to-star radius ratio and orbital parameters.
    Supports heterogeneous light curves with multiple parameter vectors,
    passbands, epochs, and supersampling.

    Parameters
    ----------
    times : ndarray
        Array of mid-observation times.
    k : ndarray
        Planet-to-star radius ratio, shape (npv, npb).
    t0 : ndarray
        Mid-transit time, shape (npv, nor).
    p : ndarray
        Orbital period, shape (npv, nor).
    a : ndarray
        Scaled semi-major axis (a/R_star), shape (npv, nor).
    i : ndarray
        Orbital inclination [rad], shape (npv, nor).
    e : ndarray
        Orbital eccentricity, shape (npv, nor).
    w : ndarray
        Argument of periastron [rad], shape (npv, nor).
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
    npv : int
        Number of parameter vectors.
    npb : int
        Number of passbands.
    ntc : int
        Number of light curves.
    nor : int
        Number of orbits (epochs).

    Returns
    -------
    flux : ndarray
        Relative flux deficit with shape (npv, npt).
    dflux : ndarray
        Gradient array with shape (npv, npt, 7), where the last axis
        corresponds to derivatives w.r.t. [k, t0, p, a, i, e, w].
    """
    npt = times.size
    flux = zeros((npv, npt))
    dflux = zeros((npv, npt, 7))

    for ipv in range(npv):
        xyc = zeros((nor, 2, 5))
        dxyc = zeros((nor, 6, 2, 5))
        for iep in range(nor):
            xyc[iep], dxyc[iep] = solve_xy_p5_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep], e[ipv, iep], w[ipv, iep])

        bt1, bt4 = bounding_box(k[ipv, 0], xyc[0])
        bt1 -= 0.003
        bt4 += 0.003

        for ipt in range(npt):
            ilc = lcids[ipt]
            ipb = pbids[ilc]

            if nor > 1:
                iep = epids[ilc]
            else:
                iep = 0

            t = _folded_time(times[ipt], t0[ipv, iep], p[ipv, iep])
            if ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    _udmodel_grad(t + time_offset, k[ipv, ipb], xyc[iep], dxyc[iep],
                                  flux[ipv, ipt:ipt+1], dflux[ipv, ipt, :])
                flux[ipv, ipt] /= nsamples[ilc]
                dflux[ipv, ipt, :] /= nsamples[ilc]
    return flux, dflux
