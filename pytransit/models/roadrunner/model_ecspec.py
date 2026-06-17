from math import fabs, floor

from meepmeep.backends.numba.newton import eclipse_light_travel_time
from meepmeep.backends.numba.point2d import sep_c, solve2d, bounding_box
from meepmeep.backends.numba.utils import eclipse_time_offset
from numpy import zeros, ndarray, isnan, nan, pi

from .common import circle_circle_intersection_area_kite as ccia

__all__ = ['esmodel']


def esmodel(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
            rstar: ndarray, fratio: ndarray, nsamples: int, exptime: float) -> ndarray:
    if k.ndim != 1:
        raise ValueError(" The radius ratio must be given as an 1D array with shape (npv)")

    if fratio.ndim != 2:
        raise ValueError(" The flux ratio must be given as a 2D array with shape (npv, npb)")

    npt = times.size
    npv = fratio.shape[0]
    npb = fratio.shape[1]

    flux = zeros((npv, npb, npt))  # Model flux
    xyc = zeros((2, 5))  # Taylor series coefficients for the (x, y) position

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0):
            flux[ipv, :, :] = nan
            continue

        # -----------------------------------------------------#
        # Calculate the Taylor series expansions for the orbit #
        # -----------------------------------------------------#
        eclipse_shift = eclipse_time_offset(p[ipv], i[ipv], e[ipv], w[ipv])
        xyc[:, :] = solve2d(eclipse_shift, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        ltt = eclipse_light_travel_time(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], rstar[ipv])
        te = t0[ipv] + eclipse_shift + ltt

        # ---------------------------#
        # Calculate the bounding box #
        # ---------------------------#
        bt4, bt1 = bounding_box(k[ipv], xyc)
        bt1 -= 0.0015 + exptime
        bt4 += 0.0015 + exptime

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in range(npt):
            epoch = floor((times[ipt] - te + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (te + epoch * p[ipv])
            if not (bt1 <= tc <= bt4):
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples + 1):
                    time_offset = exptime * ((isample - 0.5) / nsamples - 0.5)
                    z = sep_c(tc + time_offset, xyc)
                    flux[ipv, :, ipt] += 1.0 - (fratio[ipv, :] * ccia(1.0, k[ipv], z)[0] / pi) / (
                                1.0 + fratio[ipv, :] * k[ipv] ** 2)
                flux[ipv, :, ipt] /= nsamples
    return flux
