from math import isnan

from numba import njit
from numpy import zeros, floor, nan, pi, atleast_1d, asarray

from meepmeep.backends.numba.taylor.solve2d import solve2d
from meepmeep.backends.numba.taylor.position2d import pd2d
from meepmeep.backends.numba.taylor.util2d import bounding_box

from ..roadrunner.common import circle_circle_intersection_area_kite as ccia


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
def uniform_model_simple(times, k, t0, p, a, i, e, w, with_derivatives):
    npt = times.size
    flux = zeros(npt)
    dflux = zeros((7, npt))

    if a <= 1.0 or e >= 0.99:
        flux[:] = nan
        return flux, dflux

    if with_derivatives:
        raise NotImplementedError

    cf = solve2d(0.0, p, a, i, e, w)
    bt1, bt4 = bounding_box(k, cf)
    bt1 -= 0.025
    bt4 += 0.025

    for j in range(npt):
        t = folded_time(times[j], t0, p)
        if not (bt1 <= t <= bt4):
            flux[j] = 1.0
        else:
            x, y, d = pd2d(t, cf)
            if d <= 1.0 + k:
                is_area, _ = ccia(1.0, k, d)
                flux[j] += 1.0 - is_area / pi
            else:
                flux[j] += 1.
    return flux, dflux


@njit
def uniform_model_v(times, k, t0, p, dkdp, cfs, dcfs, with_derivatives,
                    lcids, pbids, epids, nsamples, exptimes):
    k = atleast_1d(asarray(k))
    dkdp = atleast_1d(asarray(dkdp))
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))

    npt = times.size
    nor = cfs.shape[0]

    flux = zeros(npt)
    dflux = zeros((7, npt))

    if with_derivatives:
        raise NotImplementedError

    for j in range(npt):
        ilc = lcids[j]
        ipb = pbids[ilc]
        iep = epids[ilc]
        ior = 0 if nor == 1 else iep
        if isnan(cfs[ior, 0, 0]):
            flux[j] = nan
            continue

        _k = k[ipb]
        t = folded_time(times[j], t0[iep], p[ior])
        exptime = exptimes[ilc]
        ns = nsamples[ilc]

        for isample in range(1, ns + 1):
            time_offset = exptime * ((isample - 0.5) / ns - 0.5)
            x, y, d = pd2d(t + time_offset, cfs[ior])
            if d <= 1.0 + _k:
                is_area, _ = ccia(1.0, _k, d)
                flux[j] += 1.0 - is_area / pi
            else:
                flux[j] += 1.
        flux[j] /= ns

    return flux, dflux
