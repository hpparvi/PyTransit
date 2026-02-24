from meepmeep.backends.numba.ts2d import solve_xy_p5_d, pd_t15c_d
from meepmeep.backends.numba.utils import d_from_pkaiews
from numba import prange, njit
from numpy import zeros, nan, fabs, pi

from .ccintersection import ccia_and_grad
from .udmodel import _folded_time


@njit
def _uniform_model_and_grad(t, k, cf, dcf, flux, dflux):
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
        flux[0] = - is_area / pi
        dflux[0] = - dadk / pi
        dflux[1] = 2 * dadz * dz[0] / pi
        for i in range(1, 6):
            dflux[i+1] = - dadz * dz[i] / pi


def uniform_model_and_grad(times, k, t0, p, a, i, e, w):
    """Evaluate the uniform-disk transit model and gradient over an array of times.

    Computes both the relative flux deficit and its analytical gradient
    with respect to the planet-to-star radius ratio and orbital parameters.
    Returns NaN for unphysical parameter combinations (a <= 1 or e >= 0.99).

    Parameters
    ----------
    times : ndarray
        Array of mid-observation times.
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
    dflux : ndarray
        Gradient array with shape (npt, 7), where columns correspond to
        derivatives w.r.t. [k, t0, p, a, i, e, w].
    """
    npt = times.size
    flux = zeros(npt)
    dflux = zeros((npt, 7))

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux, dflux

    cf, dcf = solve_xy_p5_d(0.0, p, a, i, e, w)

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1)
    for j in prange(npt):
        t = _folded_time(times[j], t0, p)
        if fabs(t) < half_window_width:
            _uniform_model_and_grad(t, k, cf, dcf, flux[j:j+1], dflux[j,:])
    return flux, dflux
