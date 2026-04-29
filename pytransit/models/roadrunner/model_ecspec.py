from math import fabs, floor

from meepmeep.utils import d_from_pkaiews, eclipse_phase
from meepmeep.newton import eclipse_light_travel_time
from meepmeep.xy.position import solve_xy_p5s, pd_t15sc
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
        eclipse_shift = eclipse_phase(p[ipv], i[ipv], e[ipv], w[ipv])
        xyc[:, :] = solve_xy_p5s(eclipse_shift, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        ltt = eclipse_light_travel_time(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], rstar[ipv])
        te = t0[ipv] + eclipse_shift + ltt

        # --------------------------------#
        # Calculate the half-window width #
        # --------------------------------#
        hww = 0.5 * d_from_pkaiews(p[ipv], k[ipv], a[ipv], i[ipv], e[ipv], w[ipv], -1, 14)
        hww = 0.0015 + exptime + hww

        # --------------------------#
        # Calculate the light curve #
        # --------------------------#
        for ipt in range(npt):
            epoch = floor((times[ipt] - te + 0.5 * p[ipv]) / p[ipv])
            tc = times[ipt] - (te + epoch * p[ipv])
            if fabs(tc) > hww:
                flux[ipv, :, ipt] = 1.0
            else:
                for isample in range(1, nsamples + 1):
                    time_offset = exptime * ((isample - 0.5) / nsamples - 0.5)
                    z = pd_t15sc(tc + time_offset, xyc)
                    flux[ipv, :, ipt] += 1.0 - (fratio[ipv, :] * ccia(1.0, k[ipv], z)[0] / pi) / (
                                1.0 + fratio[ipv, :] * k[ipv] ** 2)
                flux[ipv, :, ipt] /= nsamples
    return flux
