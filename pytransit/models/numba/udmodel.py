from numba import njit
from numpy import zeros, floor, nan, fabs, pi, sqrt

from meepmeep.xy.position import pd_t15sc, solve_xy_p5s, xyd_t15s
from meepmeep.xy.derivatives import pd_with_derivatives_s, xy_derivative_coeffs, pd_derivatives_s
from meepmeep.xy.par_fitting import coeffs as fit_coeffs
from meepmeep.xy.par_direct import coeffs as nat_coeffs, diffs
from meepmeep.utils import as_from_rhop, i_from_baew

from ...orbits import d_from_pkaiews
from .rrmodel import circle_circle_intersection_area_kite as ccia


@njit
def dfdz(z, r1, r2):
    """Circle-circle intersection derivative given a scalar z."""
    if r1 < z - r2:
        return 0.0
    elif r1 >= z + r2:
        return 0.0
    elif z - r2 <= -r1:
        return 0.0
    else:
        a = z**2 + r2**2 - r1**2
        b = z**2 + r1**2 - r2**2
        t1 = - r2**2 * (1/r2 - a / (2*r2*z**2)) / sqrt(1 - a**2 / (4*r2**2*z**2))
        t2 = - r1**2 * (1/r1 - b / (2*r1*z**2)) / sqrt(1 - b**2 / (4*r1**2*z**2))
        t3 = z*(r1**2 + r2**2 - z**2) / sqrt((-z + r2 + r1) * (z + r2 - r1) * (z - r2 + r1) * (z + r2 + r1))
        return (t1 + t2 - t3) / pi


@njit(fastmath=True)
def folded_time(t, t0, p):
    epoch = floor((t - t0 + 0.5 * p) / p)
    return t - (t0 + epoch * p)


@njit
def uniform_model_v(times, k, t0, p, a, i, e, w, with_derivatives):
    npt = times.size
    flux = zeros(npt)
    dflux = zeros((7, npt)) if with_derivatives else None
    ds = zeros(6)

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux, dflux

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1)

    # ---------------------------------------------------------------------
    # Solve the Taylor series coefficients for the planet's (x, y) location
    # and its derivatives if they're requested.
    # ---------------------------------------------------------------------
    cf = solve_xy_p5s(0.0, p, a, i, e, w)
    if with_derivatives:
        dcf = xy_derivative_coeffs(diffs(p, a, i, e, w, 1e-4), 1e-4, cf)
    else:
        dcf = None

    # -----------------------------------------------
    # Calculate the transit model and its derivatives
    # -----------------------------------------------
    for j in range(npt):
        t = folded_time(times[j], t0, p)
        if fabs(t) > half_window_width:
            flux[j] = 1.0
        else:
            x, y, d = xyd_t15s(t, cf)
            if d <= 1.0 + k:
                is_area, k0 = ccia(1.0, k, d)
                flux[j] += 1.0 - is_area / pi
                if with_derivatives:
                    dflux[0, j] = -1.0 * k * k0 / pi
                    ds = pd_derivatives_s(t, x, y, dcf, ds)
                    dflux[1:, j] += -dfdz(d, k, 1.0) * ds
            else:
                flux[j] += 1.
    return flux, dflux
