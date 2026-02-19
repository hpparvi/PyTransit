from meepmeep.backends.numba.ts2d import pd_t15c, solve_xy_p5, pd_t15_d, solve_xy_p5_d
from meepmeep.backends.numba.utils import d_from_pkaiews
from numba import njit
from numpy import floor, pi, zeros, nan, fabs
from ccintersection import ccia, ccia_and_grad

@njit(fastmath=True)
def folded_time(t, t0, p):
    epoch = floor((t - t0 + 0.5 * p) / p)
    return t - (t0 + epoch * p)


@njit
def _uniform_model(t, k, cf):
    z = pd_t15c(t, cf)
    if z <= 1.0 + k:
        is_area = ccia(1.0, k, z)
        return -is_area / pi
    else:
        return 0.0


@njit
def uniform_model(times, k, t0, p, a, i, e, w):
    npt = times.size
    flux = zeros(npt)

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux

    cf = solve_xy_p5(0.0, p, a, i, e, w)

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1)
    for j in range(npt):
        t = folded_time(times[j], t0, p)
        if fabs(t) < half_window_width:
            flux[j] = _uniform_model(t, k, cf)
    return flux


@njit
def _uniform_model_and_grad(t, k, t0, p, cf, dcf):
    z, dz = pd_t15_d(t, t0, p, cf, dcf)
    flux = 0.0
    dflux = zeros(7)
    if z <= 1.0 + k:
        is_area, (dadk, dadz) = ccia_and_grad(1.0, k, z)
        flux = - is_area / pi
        dflux[0] = - dadk / pi
        dflux[1] = 2 * dadz * dz[0] / pi
        for i in range(1, 6):
            dflux[i+1] = - dadz * dz[i] / pi
    return flux, dflux


@njit
def uniform_model_and_grad(times, k, t0, p, a, i, e, w):
    npt = times.size
    flux = zeros(npt)
    dflux = zeros((npt, 7))

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux, dflux

    cf, dcf = solve_xy_p5_d(0.0, p, a, i, e, w)

    half_window_width = 0.025 + 0.5 * d_from_pkaiews(p, k, a, i, e, w, 1)
    for j in range(npt):
        t = folded_time(times[j], t0, p)
        if fabs(t) < half_window_width:
            flux[j], dflux[j,:] = _uniform_model_and_grad(t, k, t0, p, cf, dcf)
    return flux, dflux