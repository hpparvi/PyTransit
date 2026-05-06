from meepmeep import eclipse_light_travel_time
from meepmeep.tsorbit import bounding_box
from meepmeep.utils import eclipse_phase

from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, nan, full, floor, pi

from meepmeep.xy.position import solve_xy_p5s, pd_t15sc

from .common import circle_circle_intersection_area_kite as ccia

def eclipse_model(times: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray,
                   rstar: ndarray, nlc: int, npb: int, nep: int,
                   lcids: ndarray, pbids: ndarray, epids: ndarray, nsamples: ndarray, exptimes: ndarray):

    npv = k.shape[0]
    npt = times.size

    if k.ndim != 1:
        raise ValueError(" The radius ratio must be given as an 1D array with shape (npv)")

    _exptimes = zeros(nlc)
    _exptimes[:] = exptimes
    _nsamples = zeros(nlc)
    _nsamples[:] = nsamples

    pv_is_good = full(npv, True)
    xyc = zeros((npv, 2, 5))     # Taylor series coefficients for the (x, y) position
    bbs = zeros((npv, nlc, 2))
    eclipse_shifts = zeros(npv)
    ltts = zeros(npv)

    for ipv in range(npv):
        if isnan(a[ipv]) or (a[ipv] <= 1.0) or (e[ipv] < 0.0):
            pv_is_good[ipv] = False
            continue

        # ------------------------------------------------------#
        # Calculate the Taylor series expansions for the orbits #
        # ------------------------------------------------------#
        eclipse_shifts[ipv] = eclipse_phase(p[ipv], i[ipv], e[ipv], w[ipv])
        xyc[ipv, :, :] = solve_xy_p5s(eclipse_shifts[ipv], p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        ltts[ipv] = eclipse_light_travel_time(p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], rstar[ipv])

        # -----------------------------#
        # Calculate the bounding boxes #
        # -----------------------------#
        bt1, bt4 = bounding_box(k[ipv], xyc)
        bbs[ipv, :, 0] = bt1
        bbs[ipv, :, 1] = bt4
        for ilc in range(nlc):
            bbs[ipv, ilc, 0] -= 0.003 + _exptimes[ilc]
            bbs[ipv, ilc, 1] += 0.003 + _exptimes[ilc]

    # ---------------------------#
    # Calculate the light curves #
    # ---------------------------#
    flux = zeros((npv, npt))
    for j in prange(npv * npt):
        ipv = j // npt
        ipt = j % npt

        if not pv_is_good[ipv]:
            flux[ipv, ipt] = nan
            continue

        ilc = lcids[ipt]
        iep = epids[ilc]

        te =  t0[ipv, iep] + eclipse_shifts[ipv] + ltts[ipv]
        epoch = floor((times[ipt] - te + 0.5 * p[ipv]) / p[ipv])
        tc = times[ipt] - (te + epoch * p[ipv])
        if not (bbs[ipv, ilc, 0] <= tc <= bbs[ipv, ilc, 1]):
            flux[ipv, ipt] = 1.0
        else:
            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                z = pd_t15sc(tc + time_offset, xyc[ipv])
                flux[ipv, ipt] +=  1.0 - ccia(1.0, k[ipv], z)[0] / pi
            flux[ipv, ipt] /= nsamples[ilc]
    return flux